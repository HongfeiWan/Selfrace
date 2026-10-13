# teraflow_replication/simulator/offroad.py
import torch
from torch import Tensor
import sys
from road import RoadNetwork
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from spatial_hash import SpatialHash
from tensor_budget import is_cuda_oom, workspace_bytes
from execution import CompiledStage


def _points_inside_quads(points, ids, valid, poly_verts, poly_edges, poly_sign):
    safe_ids = ids.clamp_min(0)
    verts, edges = poly_verts[safe_ids], poly_edges[safe_ids]
    pv = points[:, None, None, :] - verts
    cross = edges[..., 0] * pv[..., 1] - edges[..., 1] * pv[..., 0]
    return (((poly_sign[safe_ids, None] * cross >= -1e-10).all(dim=-1) & valid).any(dim=1))

class OffroadChecker:
    """
    一个基于 GPU 加速的批量化离路检测器。
    它使用一个共享的、预初始化的 SpatialHash 对象来执行查询。
    """
    DEFAULT_POINT_CHUNK_SIZE = 524288

    def __init__(
        self,
        map_data: RoadNetwork,
        spatial_hash: SpatialHash,
        points_per_vehicle_edge: int = 3,
        offroad_point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
        workspace_mb: float = 64.0,
        compile_enabled: bool = False,
    ):
        """
        初始化离路检测器。
        Args:
            map_data (RoadNetwork): 包含路面几何信息的 RoadNetwork 对象。
            spatial_hash (SpatialHash): 预初始化的空间哈希对象。
            points_per_vehicle_edge (int): 沿着车辆边界框每条边采样的点数。
            offroad_point_chunk_size (int): padded point-in-polygon 每个 chunk 的点数。
        """
        self.device = map_data.device
        if points_per_vehicle_edge < 2:
            raise ValueError("points_per_vehicle_edge must be at least 2.")
        self.points_per_vehicle_edge = points_per_vehicle_edge
        self.offroad_point_chunk_size = max(1, int(offroad_point_chunk_size))
        self.workspace_bytes = workspace_bytes(self.device, workspace_mb)
        self.candidate_chunk_size = min(64, max(1, self.workspace_bytes // 256))
        self._inside_stage = CompiledStage(_points_inside_quads, compile_enabled)
        self.road_polygons = map_data.quads_vertices.to(self.device)

        self.spatial_hash = spatial_hash
        
        # 使用共享的 spatial_hash 构建静态路面索引
        poly_min_bounds = self.road_polygons.min(dim=1).values
        poly_max_bounds = self.road_polygons.max(dim=1).values
        poly_bounds = torch.stack([poly_min_bounds, poly_max_bounds], dim=1)
        self.spatial_hash.build_static_index(poly_bounds)
        self.local_bbox_points = self._create_local_bbox_points().to(self.device)
        
        # 预计算用于矢量叉乘半平面测试的凸四边形参数
        self._precompute_convex_quad_edges()

    def _create_local_bbox_points(self) -> Tensor:
        """
        为单位尺寸的边界框（范围从-0.5到0.5）创建点模板。
        """
        n = self.points_per_vehicle_edge
        edge1 = torch.stack([torch.linspace(-0.5, 0.5, n), torch.full((n,), -0.5)], dim=1)
        edge2 = torch.stack([torch.full((n,), 0.5), torch.linspace(-0.5, 0.5, n)], dim=1)
        edge3 = torch.stack([torch.linspace(0.5, -0.5, n), torch.full((n,), 0.5)], dim=1)
        edge4 = torch.stack([torch.full((n,), -0.5), torch.linspace(0.5, -0.5, n)], dim=1)
        points = torch.cat([edge1[:-1], edge2[:-1], edge3[:-1], edge4[:-1]], dim=0)
        center_point = torch.tensor([[0.0, 0.0]])
        points = torch.cat([points, center_point], dim=0)
        return points

    def _precompute_convex_quad_edges(self):
        """
        预计算用于矢量叉乘半平面测试的凸四边形参数：
        - poly_verts: 顶点坐标 [Q,4,2]
        - poly_edges: 顺序边向量 v_{i+1}-v_i [Q,4,2]
        - poly_sign: 顶点绕序符号（CCW=+1, CW=-1）[Q]
        """
        verts = self.road_polygons  # [Q,4,2]
        next_idx = torch.tensor([1, 2, 3, 0], device=self.device)
        self.poly_verts = verts
        self.poly_edges = verts[:, next_idx, :] - verts
        x = verts[..., 0]
        y = verts[..., 1]
        area2 = (x[:, 0] * y[:, 1] - y[:, 0] * x[:, 1] +
                 x[:, 1] * y[:, 2] - y[:, 1] * x[:, 2] +
                 x[:, 2] * y[:, 3] - y[:, 2] * x[:, 3] +
                 x[:, 3] * y[:, 0] - y[:, 3] * x[:, 0])
        self.poly_sign = torch.where(area2 >= 0,
                                     torch.tensor(1.0, device=self.device),
                                     torch.tensor(-1.0, device=self.device))

    def _get_discretized_bounding_boxes(self, states: Tensor) -> Tensor:
        """
        将本地边界框点集根据一批车辆的状态转换到世界坐标系。
        """
        N = states.shape[0]
        heading = states[:, 2]
        length, width = states[:, 3], states[:, 4]
        size_scaler = torch.stack([length, width], dim=1).view(N, 1, 2)
        scaled_points = self.local_bbox_points.unsqueeze(0) * size_scaler
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        rot_matrix = torch.stack([
            torch.stack([cos_h, -sin_h], dim=1),
            torch.stack([sin_h, cos_h], dim=1)
        ], dim=1)
        # Points are row vectors; transpose the column-vector rotation.
        rotated_points = torch.bmm(scaled_points, rot_matrix.transpose(1, 2))
        world_points = rotated_points + states[:, :2].unsqueeze(1)
        return world_points
    
    def _batch_point_in_polygon_test(self, points: Tensor) -> Tensor:
        """
        基于矢量叉乘（半平面）的方法：
        1) 用 padded 空间哈希取每个 point 的候选 quad；
        2) 对每个 chunk 的候选，计算四条边的 cross(e, p - v)；
        3) 若多边形为顺时针，则 cross <= 0，全为右侧；若为逆时针则 cross >= 0；
           统一写作 (sign * cross) >= -eps，sign=+1(CCW), -1(CW)。
        """
        M = points.shape[0]
        if M == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)

        return self._batch_point_in_polygon_test_padded(points)

    def _batch_point_in_polygon_test_padded(self, points: Tensor) -> Tensor:
        M = points.shape[0]
        flat_on_road_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
        if self.poly_verts.shape[0] == 0:
            return flat_on_road_mask

        columns = min(self.candidate_chunk_size, self.spatial_hash.static_max_candidates_per_cell)
        # Conservative accounting: IDs, masks, verts, edges, pv, cross and
        # arithmetic temporaries. Never size from the point count alone.
        chunk_size = min(self.offroad_point_chunk_size, max(1, self.workspace_bytes // (256 * max(1, columns))))
        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            try:
                flat_on_road_mask[start:end] = self._query_point_chunk(points[start:end], max(1, columns))
                start = end
            except RuntimeError as exc:
                if torch.device(self.device).type != 'cuda' or not is_cuda_oom(exc) or (chunk_size == 1 and columns <= 1):
                    raise
                if chunk_size > 1:
                    chunk_size = max(1, chunk_size // 2)
                    self.offroad_point_chunk_size = min(self.offroad_point_chunk_size, chunk_size)
                else:
                    columns = max(1, columns // 2)
                    self.candidate_chunk_size = columns

        return flat_on_road_mask

    def _query_point_chunk(self, points, columns):
        on_road = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for left in range(0, self.spatial_hash.static_max_candidates_per_cell, columns):
            ids, valid = self.spatial_hash.query_points_padded(points, left, columns)
            on_road |= self._inside_stage(points, ids, valid, self.poly_verts, self.poly_edges, self.poly_sign)
        return on_road

    def check_on_road(self, states: Tensor) -> Tensor:
        """
        批量检测车辆是否在道路上。
        """
        N = states.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        # Bound box-point materialization as well as polygon-query scratch.
        num_points_per_box = self.local_bbox_points.shape[0]
        rows = max(1, min(self.offroad_point_chunk_size, self.workspace_bytes // 256) // num_points_per_box)
        is_on_road = torch.empty(N, dtype=torch.bool, device=states.device)
        start = 0
        while start < N:
            end = min(start + rows, N)
            try:
                is_on_road[start:end] = self._check_vehicle_chunk(states[start:end])
                start = end
            except RuntimeError as exc:
                if states.device.type != 'cuda' or not is_cuda_oom(exc) or rows == 1:
                    raise
                rows = max(1, rows // 2)
                self.offroad_point_chunk_size = min(self.offroad_point_chunk_size, rows * num_points_per_box)
        return is_on_road

    def _check_vehicle_chunk(self, states):
        # Release failed box/point scratch before retrying a smaller car tile.
        points = self._get_discretized_bounding_boxes(states)
        on_road = self._batch_point_in_polygon_test(points.reshape(-1, 2))
        return on_road.view(states.shape[0], self.local_bbox_points.shape[0]).all(dim=1)

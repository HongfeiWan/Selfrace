# teraflow_replication/simulator/offroad.py
import torch
from torch import Tensor
from typing import Tuple
import sys
from road import RoadNetwork
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from spatial_hash import SpatialHash

class OffroadChecker:
    """
    一个基于 GPU 加速的批量化离路检测器。
    它使用一个共享的、预初始化的 SpatialHash 对象来执行查询。
    """
    def __init__(self, map_data: RoadNetwork, spatial_hash: SpatialHash, points_per_vehicle_edge: int = 3):
        """
        初始化离路检测器。
        Args:
            map_data (RoadNetwork): 包含路面几何信息的 RoadNetwork 对象。
            spatial_hash (SpatialHash): 预初始化的空间哈希对象。
            points_per_vehicle_edge (int): 沿着车辆边界框每条边采样的点数。
        """
        self.device = map_data.device
        if points_per_vehicle_edge < 2:
            raise ValueError("points_per_vehicle_edge must be at least 2.")
        self.points_per_vehicle_edge = points_per_vehicle_edge
        self.road_polygons = map_data.quads_vertices.to(self.device)
        self.spatial_hash = spatial_hash
        
        # 使用共享的 spatial_hash 构建静态路面索引
        poly_min_bounds = self.road_polygons.min(dim=1).values
        poly_max_bounds = self.road_polygons.max(dim=1).values
        poly_bounds = torch.stack([poly_min_bounds, poly_max_bounds], dim=1)
        self.spatial_hash.build_static_index(poly_bounds)
        self.local_bbox_points = self._create_local_bbox_points().to(self.device)

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

    def _get_discretized_bounding_boxes(self, states: Tensor) -> Tensor:
        """
        将本地边界框点集根据一批车辆的状态转换到世界坐标系。
        """
        N = states.shape[0]
        x, y, heading = states[:, 0], states[:, 1], states[:, 2]
        length, width = states[:, 3], states[:, 4]
        size_scaler = torch.stack([length, width], dim=1).view(N, 1, 2)
        scaled_points = self.local_bbox_points.unsqueeze(0) * size_scaler
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        rot_matrix = torch.stack([
            torch.stack([cos_h, -sin_h], dim=1),
            torch.stack([sin_h, cos_h], dim=1)
        ], dim=1)
        rotated_points = torch.bmm(scaled_points, rot_matrix)
        world_points = rotated_points + states[:, :2].unsqueeze(1)
        return world_points
    
    def _batch_point_in_polygon_test(self, points: Tensor) -> Tensor:
        """
        使用射线投射法（Ray Casting）执行并行的"点在多边形内"测试。
        """
        M = points.shape[0]
        if M == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        candidate_pairs = self.spatial_hash.query_points(points)
        if candidate_pairs.shape[0] == 0:
            return torch.zeros(M, dtype=torch.bool, device=self.device)
        point_indices, polygon_indices = candidate_pairs[:, 0], candidate_pairs[:, 1]
        test_points = points[point_indices]
        test_polygons = self.road_polygons[polygon_indices]
        px = test_points[:, None, 0]
        py = test_points[:, None, 1]
        v_start = test_polygons
        v_end = torch.roll(test_polygons, shifts=-1, dims=1)
        y1, x1 = v_start[..., 1], v_start[..., 0]
        y2, x2 = v_end[..., 1], v_end[..., 0]
        y_check = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
        denom = y2 - y1
        denom[denom.abs() < 1e-9] = 1e-9
        x_intersect = (py - y1) * (x2 - x1) / denom + x1
        x_check = px < x_intersect
        crossings = y_check & x_check
        intersection_counts = torch.sum(crossings, dim=1)
        is_inside_per_pair = (intersection_counts % 2 == 1)
        is_on_road = torch.zeros(M, dtype=torch.long, device=self.device)
        # 使用 scatter_add_ 以处理一个点在多个多边形内的情况（虽然不影响奇偶性，但更健壮）
        is_on_road.scatter_add_(0, point_indices, is_inside_per_pair.long())
        return is_on_road > 0

    def check_on_road(self, states: Tensor) -> Tensor:
        """
        批量检测车辆是否在道路上。
        """
        N = states.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        world_points = self._get_discretized_bounding_boxes(states)
        num_points_per_box = world_points.shape[1]
        flat_points = world_points.view(-1, 2)
        flat_on_road_mask = self._batch_point_in_polygon_test(flat_points)
        on_road_mask_per_point = flat_on_road_mask.view(N, num_points_per_box)
        is_on_road = torch.all(on_road_mask_per_point, dim=1)
        return is_on_road
    


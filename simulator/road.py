import torch
import json
import math
from typing import Dict, Tuple

class RoadNetwork:
    """
    负责加载和管理从预处理的 CARLA 地图数据中提取的道路网络。
    这个类将地图数据（主要是四边形路块 'quads'）加载到 PyTorch 张量中，
    以便于在 GPU 上进行高效的批量化计算。它提供了查询地图几何信息
    （如车道中心线、边界线）的核心功能。
    """
    def __init__(self, map_path: str, device: torch.device):
        """
        初始化道路网络。
        Args:
            map_path (str): 指向预处理后的地图 JSON 文件的路径。
            device (torch.device): 用于存储地图数据的计算设备 ('cpu' 或 'cuda')。
        """
        self.device = device
        # 加载和处理地图数据
        map_data = self._load_map_data(map_path)
        self._process_map_data(map_data)

    def _load_map_data(self, map_path: str) -> Dict:
        """从 JSON 文件加载地图数据。"""
        try:
            with open(map_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise

    def _process_map_data(self, map_data: Dict):
        """
        将从 JSON 加载的地图数据（字典列表）转换为 PyTorch 张量。
        'quads' 是地图的基本单元，每个 quad 代表一小块路面。
        """
        # 提取顶点坐标
        vertices = self._extract_vertices(map_data['quads'])
        # 计算道路几何信息
        self._compute_road_geometry(vertices)
        # 存储元数据
        self._store_metadata(map_data['quads'])
        
        # 加载全局航点
        self._load_global_waypoints(map_data)

        # 加载交通灯与停止线；这些数据用于构造 W_stop 观测和红灯越线惩罚。
        self._load_traffic_controls(map_data)
    
    def _extract_vertices(self, quads_data):
        """提取顶点坐标"""
        # 顶点顺序映射:
        # p0 (left_start)  -> vertices[2]
        # p1 (left_end)    -> vertices[1] 
        # p2 (right_end)   -> vertices[0]
        # p3 (right_start) -> vertices[3]
        p0 = torch.tensor([[q['vertices'][2]['x'], q['vertices'][2]['y']] for q in quads_data], dtype=torch.float32, device=self.device)
        p1 = torch.tensor([[q['vertices'][1]['x'], q['vertices'][1]['y']] for q in quads_data], dtype=torch.float32, device=self.device)
        p2 = torch.tensor([[q['vertices'][0]['x'], q['vertices'][0]['y']] for q in quads_data], dtype=torch.float32, device=self.device)
        p3 = torch.tensor([[q['vertices'][3]['x'], q['vertices'][3]['y']] for q in quads_data], dtype=torch.float32, device=self.device)
        return p0, p1, p2, p3
    
    def _compute_road_geometry(self, vertices):
        """计算道路几何信息"""
        p0, p1, p2, p3 = vertices
        # (num_quads, 2) -> (num_quads, 4, 2)
        self.quads_vertices = torch.stack([p0, p1, p2, p3], dim=1)
        self.num_quads = self.quads_vertices.shape[0]
        # 计算道路中心点和方向向量
        front_center = (p1 + p2) / 2.0  # 前中心点
        back_center = (p0 + p3) / 2.0   # 后中心点
        # 计算车道中心线 (从后中心点到前中心点)
        self.quad_centerlines = torch.stack([back_center, front_center], dim=1)
        self.quad_centers = self.quad_centerlines.mean(dim=1)
        # 计算车道边界
        self.left_boundaries = torch.stack([p0, p1], dim=1)
        self.right_boundaries = torch.stack([p3, p2], dim=1)
        # 计算道路方向向量 (从后到前)
        self.quad_directions = front_center - back_center
        self.quad_widths = 0.5 * (
            torch.norm(p0 - p3, dim=1) + torch.norm(p1 - p2, dim=1)
        )
        # 归一化方向向量
        direction_norms = torch.norm(self.quad_directions, dim=1, keepdim=True)
        zero_mask = (direction_norms == 0)
        self.quad_directions = torch.where(
            zero_mask, 
            torch.tensor([1.0, 0.0], device=self.device), 
            self.quad_directions / direction_norms
        )

    def _store_metadata(self, quads_data):
        """存储元数据"""
        self.quad_ids = torch.tensor([q['polyId'] for q in quads_data], dtype=torch.int64, device=self.device)
        self.road_ids = torch.tensor([q['road_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        self.lane_ids = torch.tensor([q['lane_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        self.q_values = torch.tensor([q.get('q', 0.0) for q in quads_data], dtype=torch.float32, device=self.device)
        self.quad_curvatures = self._compute_quad_curvatures()
        self.poly_id_to_index = {int(poly_id): i for i, poly_id in enumerate(self.quad_ids.detach().cpu().tolist())}

        # 加载并存储每个 quad 关联的航点 ID

    def _compute_quad_curvatures(self) -> torch.Tensor:
        """按 road/lane/q 顺序估算局部曲率，启动时预计算，运行时直接索引。"""
        curvatures = torch.zeros(self.num_quads, dtype=torch.float32, device=self.device)
        if self.num_quads < 3:
            return curvatures

        keys = {}
        road_ids_cpu = self.road_ids.detach().cpu().tolist()
        lane_ids_cpu = self.lane_ids.detach().cpu().tolist()
        q_cpu = self.q_values.detach().cpu().tolist()
        for idx, key in enumerate(zip(road_ids_cpu, lane_ids_cpu)):
            keys.setdefault(key, []).append(idx)

        for indices in keys.values():
            if len(indices) < 3:
                continue
            indices.sort(key=lambda i: q_cpu[i])
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            dirs = self.quad_directions[idx_t]
            q_vals = self.q_values[idx_t]
            prev_dirs = dirs[:-2]
            next_dirs = dirs[2:]
            dot = (prev_dirs * next_dirs).sum(dim=-1).clamp(-1.0, 1.0)
            cross = prev_dirs[:, 0] * next_dirs[:, 1] - prev_dirs[:, 1] * next_dirs[:, 0]
            delta_heading = torch.atan2(cross, dot)
            delta_s = (q_vals[2:] - q_vals[:-2]).abs().clamp_min(1e-3)
            curvatures[idx_t[1:-1]] = delta_heading / delta_s
        return curvatures
    
    def _load_global_waypoints(self, map_data):
        """加载全局航点"""
        # 加载车道航点
        w_lane_points = map_data.get('global_w_lane_waypoints', [])
        if w_lane_points:
            self.global_w_lane_waypoints = torch.tensor(
                [[p['x'], p['y']] for p in w_lane_points],
                dtype=torch.float32,
                device=self.device
            )
            self.global_w_lane_directions = torch.tensor(
                [[p.get('direction', [1.0, 0.0, 0.0])[0], p.get('direction', [1.0, 0.0, 0.0])[1]] for p in w_lane_points],
                dtype=torch.float32,
                device=self.device
            )
            dir_norm = torch.norm(self.global_w_lane_directions, dim=1, keepdim=True).clamp_min(1e-6)
            self.global_w_lane_directions = self.global_w_lane_directions / dir_norm
            poly_indices = [
                self.poly_id_to_index.get(int(p.get('poly_id', p.get('routing_poly_id', -1))), -1)
                for p in w_lane_points
            ]
            self.global_w_lane_poly_indices = torch.tensor(poly_indices, dtype=torch.long, device=self.device)
            safe_poly_indices = torch.clamp(self.global_w_lane_poly_indices, 0, max(self.num_quads - 1, 0))
            self.global_w_lane_widths = torch.where(
                self.global_w_lane_poly_indices >= 0,
                self.quad_widths[safe_poly_indices],
                torch.zeros_like(safe_poly_indices, dtype=torch.float32)
            )
        else:
            self.global_w_lane_waypoints = torch.empty((0, 2), device=self.device)
            self.global_w_lane_directions = torch.empty((0, 2), device=self.device)
            self.global_w_lane_poly_indices = torch.empty((0,), dtype=torch.long, device=self.device)
            self.global_w_lane_widths = torch.empty((0,), dtype=torch.float32, device=self.device)
        # 加载边界航点
        w_boundary_points = map_data.get('oob_points', [])
        self.global_w_boundary_points = torch.tensor([[p['x'], p['y']] for p in w_boundary_points], dtype=torch.float32, device=self.device) if w_boundary_points else torch.empty((0, 2), device=self.device)

    def _load_traffic_controls(self, map_data):
        """加载交通灯位置和停止线中心点/朝向。"""
        # Optional map contract: center=[x,y], heading_rad, length, width (metres).
        crosswalks = map_data.get('crosswalks', [])
        self.crosswalks = torch.tensor([
            [*entry['center'], entry['heading_rad'], entry['length'], entry['width']]
            for entry in crosswalks
        ], dtype=torch.float32, device=self.device).reshape(-1, 5)
        traffic_controls = map_data.get('traffic_controls', [])
        if not traffic_controls:
            self.traffic_light_locations = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self.stop_line_centers = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self.stop_line_yaws = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.stop_line_control_indices = torch.empty((0,), dtype=torch.long, device=self.device)
            return

        light_locations = []
        stop_line_centers = []
        stop_line_yaws = []
        stop_line_control_indices = []
        for control_idx, control in enumerate(traffic_controls):
            loc = control.get('traffic_light_location', {})
            light_locations.append([float(loc.get('x', 0.0)), float(loc.get('y', 0.0))])
            for waypoint in control.get('stop_line_waypoints', []):
                wp_loc = waypoint.get('location', {})
                wp_rot = waypoint.get('rotation', {})
                stop_line_centers.append([float(wp_loc.get('x', 0.0)), float(wp_loc.get('y', 0.0))])
                stop_line_yaws.append(math.radians(float(wp_rot.get('yaw', 0.0))))
                stop_line_control_indices.append(control_idx)

        self.traffic_light_locations = torch.tensor(light_locations, dtype=torch.float32, device=self.device)
        if stop_line_centers:
            self.stop_line_centers = torch.tensor(stop_line_centers, dtype=torch.float32, device=self.device)
            self.stop_line_yaws = torch.tensor(stop_line_yaws, dtype=torch.float32, device=self.device)
            self.stop_line_control_indices = torch.tensor(stop_line_control_indices, dtype=torch.long, device=self.device)
        else:
            self.stop_line_centers = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self.stop_line_yaws = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.stop_line_control_indices = torch.empty((0,), dtype=torch.long, device=self.device)

    def get_all_lanes_left_boundaries(self) -> torch.Tensor:
        """返回所有车道左边界线段。"""
        return self.left_boundaries

    def get_all_lanes_right_boundaries(self) -> torch.Tensor:
        """返回所有车道右边界线段。"""
        return self.right_boundaries

    def get_all_lanes_centerlines(self) -> torch.Tensor:
        """
        返回地图上所有 quad 的中心线段。
        Returns:
            torch.Tensor: 形状为 (num_quads, 2, 2) 的张量，代表所有中心线的起点和终点。
        """
        return self.quad_centerlines

    def find_nearest_lanes(self, points: torch.Tensor, k: int = 1, spatial_hash=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bound scratch independently of both vehicle and map size."""
        from tensor_budget import nearest_indices, is_cuda_oom
        points = points.reshape(-1, 2)
        budget = getattr(self, 'workspace_bytes', 64 * 1024**2)
        if spatial_hash is None:
            ids = nearest_indices(points, self.quad_centers, k, budget)
            valid = ids >= 0
            if self.num_quads == 0:
                return points.new_full(ids.shape, float('inf')), ids
            distances = (points[:, None] - self.quad_centers[ids.clamp_min(0)]).norm(dim=-1)
            return distances.masked_fill(~valid, float('inf')), ids
        distances = points.new_full((points.shape[0], k), float('inf'))
        indices = torch.full((points.shape[0], k), -1, dtype=torch.long, device=points.device)
        columns = max(1, min(64, budget // 128 - 2 * k))
        rows = max(1, budget // (64 * (columns + 2 * k)))
        start = 0
        while start < points.shape[0]:
            end = min(start + rows, points.shape[0])
            try:
                d, idx = self._nearest_lane_chunk(points[start:end], k, spatial_hash, columns)
                distances[start:end], indices[start:end] = d, idx
                start = end
            except RuntimeError as exc:
                if points.device.type != 'cuda' or not is_cuda_oom(exc) or rows == 1:
                    raise
                rows = max(1, rows // 2)
        return distances, indices

    def _nearest_lane_chunk(self, points, k, spatial_hash, columns):
        best_d = points.new_full((points.shape[0], k), float('inf'))
        best_i = torch.full(best_d.shape, -1, device=points.device, dtype=torch.long)
        if self.num_quads == 0:
            return best_d, best_i
        for left in range(0, spatial_hash.static_max_candidates_per_cell, columns):
            ids, valid = spatial_hash.query_points_padded(points, left, columns)
            diff = points[:, None, :] - self.quad_centers[ids.clamp_min(0)]
            d = diff.square().sum(-1).masked_fill(~valid, float('inf'))
            combined_d = torch.cat((best_d, d), dim=1)
            combined_i = torch.cat((best_i, ids), dim=1)
            best_d, positions = combined_d.topk(k, largest=False, dim=1)
            best_i = combined_i.gather(1, positions)
        return best_d.sqrt(), best_i.masked_fill(~best_d.isfinite(), -1)

    def calculate_frenet_coordinates(self, vehicle_positions: torch.Tensor, vehicle_headings: torch.Tensor, spatial_hash=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算车辆在Frenet坐标系中的横向距离d和角度误差theta_f。
        Args:
            vehicle_positions (torch.Tensor): 车辆位置，形状为 (B, M, 2) 或 (N, 2)
            vehicle_headings (torch.Tensor): 车辆朝向角度（弧度），形状为 (B, M) 或 (N,)
            spatial_hash (SpatialHash, optional): 空间哈希对象，用于加速查询。如果提供，将使用空间哈希；否则使用暴力搜索。
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - d: 横向距离，正值表示在道路右侧，负值表示在道路左侧
            - theta_f: 角度误差（弧度），正值表示车辆朝向偏右，负值表示偏左
        """
        # 确保输入是3D张量 (B, M, 2) 和 (B, M)
        if vehicle_positions.ndim == 2:
            vehicle_positions = vehicle_positions.unsqueeze(0)  # (N, 2) -> (1, N, 2)
            vehicle_headings = vehicle_headings.unsqueeze(0)    # (N,) -> (1, N)
        B, M, _ = vehicle_positions.shape
        
        # 为每个车辆找到最近的道路段
        vehicle_positions_flat = vehicle_positions.view(-1, 2)  # (B*M, 2)
        distances, nearest_indices = self.find_nearest_lanes(vehicle_positions_flat, k=1, spatial_hash=spatial_hash)
        
        return self.frenet_from_indices(vehicle_positions, vehicle_headings, nearest_indices.reshape(B, M))

    def frenet_from_indices(self, vehicle_positions, vehicle_headings, nearest_indices):
        """Right-positive reward geometry; observation explicitly flips its sign."""
        shape = vehicle_headings.shape
        vehicle_positions = vehicle_positions.reshape(-1, 2)
        vehicle_headings = vehicle_headings.reshape(-1)
        nearest_indices = nearest_indices.reshape(-1)
        valid = nearest_indices >= 0
        safe = nearest_indices.clamp_min(0)
        directions = self.quad_directions[safe]
        offset = vehicle_positions - self.quad_centerlines[safe, 0]
        vehicle_dir = torch.stack((vehicle_headings.cos(), vehicle_headings.sin()), dim=-1)
        d = offset[:, 0] * directions[:, 1] - offset[:, 1] * directions[:, 0]
        cross = vehicle_dir[:, 0] * directions[:, 1] - vehicle_dir[:, 1] * directions[:, 0]
        angle = torch.atan2(cross, (vehicle_dir * directions).sum(-1))
        return (d.masked_fill(~valid, 0).reshape(shape),
                angle.masked_fill(~valid, 0).reshape(shape))

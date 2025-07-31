import torch
import json
from typing import Dict, Tuple, List

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
        # 计算车道边界
        self.left_boundaries = torch.stack([p0, p1], dim=1)
        self.right_boundaries = torch.stack([p3, p2], dim=1)
        # 计算道路方向向量 (从后到前)
        self.quad_directions = front_center - back_center
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
        self.lane_ids = torch.tensor([q['lane_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        
        # 加载并存储每个 quad 关联的航点 ID
        self.quad_w_lane_ids_assoc = [q.get('w_lane_ids', []) for q in quads_data]
        self.quad_w_boundary_ids_assoc = [q.get('w_boundary_ids', []) for q in quads_data]
    
    def _load_global_waypoints(self, map_data):
        """加载全局航点"""
        # 加载车道航点
        w_lane_points = map_data.get('global_w_lane_waypoints', [])
        self.global_w_lane_waypoints = torch.tensor([[p['x'], p['y']] for p in w_lane_points], dtype=torch.float32, device=self.device) if w_lane_points else torch.empty((0, 2), device=self.device)
        # 加载边界航点
        w_boundary_points = map_data.get('oob_points', [])
        self.global_w_boundary_points = torch.tensor([[p['x'], p['y']] for p in w_boundary_points], dtype=torch.float32, device=self.device) if w_boundary_points else torch.empty((0, 2), device=self.device)

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

    def find_nearest_lanes(self, points: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为一批输入点找到最近的 k 个车道 (quads)。
        Args:
            points (torch.Tensor): 形状为 (N, 2) 的点坐标张量。
            k (int): 需要为每个点找到的最近车道的数量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - distances: 形状为 (N, k) 的距离张量。
            - indices: 形状为 (N, k) 的最近车道 (quads) 的索引张量。
        """
        if points.ndim == 1:
            points = points.unsqueeze(0)
        # 计算点到所有车道中心点 (quad 中心线的中点) 的距离
        quad_centers = self.quad_centerlines.mean(dim=1) # (num_quads, 2)
        # 使用广播计算欧氏距离的平方
        # (N, 1, 2) - (1, num_quads, 2) -> (N, num_quads, 2)
        diff = points.unsqueeze(1) - quad_centers.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1) # (N, num_quads)
        # 找到最近的 k 个
        distances, indices = torch.topk(dist_sq, k=k, dim=1, largest=False)
        
        return torch.sqrt(distances), indices

    def get_global_waypoints_by_ids(self, ids: torch.Tensor, point_type: str) -> torch.Tensor:
        """根据ID列表从全局航点库中获取航点坐标。"""
        # 映射点类型到对应的张量
        point_type_map = {
            'w_lane': self.global_w_lane_waypoints,
            'w_boundary': self.global_w_boundary_points
        }
        
        source_points = point_type_map.get(point_type)
        if source_points is None:
            return torch.empty((0, 2), device=self.device)
        
        # 过滤掉无效的ID (例如，填充的-1)
        valid_ids = ids[ids >= 0]
        if valid_ids.numel() == 0:
            return torch.empty((0, 2), device=self.device)
            
        return source_points[valid_ids]

    def calculate_frenet_coordinates(self, vehicle_positions: torch.Tensor, vehicle_headings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算车辆在Frenet坐标系中的横向距离d和角度误差theta_f。
        
        Args:
            vehicle_positions (torch.Tensor): 车辆位置，形状为 (B, M, 2) 或 (N, 2)
            vehicle_headings (torch.Tensor): 车辆朝向角度（弧度），形状为 (B, M) 或 (N,)
            
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
        distances, nearest_indices = self.find_nearest_lanes(vehicle_positions_flat, k=1)
        
        # 重塑回原始形状
        nearest_indices = nearest_indices.view(B, M)  # (B, M)
        
        # 获取最近道路段的方向向量
        road_directions = self.quad_directions[nearest_indices]  # (B, M, 2)
        
        # 计算车辆朝向向量
        vehicle_directions = torch.stack([
            torch.cos(vehicle_headings),
            torch.sin(vehicle_headings)
        ], dim=-1)  # (B, M, 2)
        
        # 计算角度误差 theta_f
        theta_f = self._calculate_heading_error(road_directions, vehicle_directions)
        # 计算横向距离 d
        d = self._calculate_lateral_distance(vehicle_positions, nearest_indices, road_directions)
        return d, theta_f
    
    def _calculate_heading_error(self, road_directions, vehicle_directions):
        """计算车辆朝向与道路方向的夹角误差"""
        # 使用叉积的符号来确定角度方向
        cross_product = (road_directions[:, :, 0] * vehicle_directions[:, :, 1] - 
                        road_directions[:, :, 1] * vehicle_directions[:, :, 0])
        dot_product = torch.sum(road_directions * vehicle_directions, dim=-1)
        return torch.atan2(cross_product, dot_product)  # (B, M)
    
    def _calculate_lateral_distance(self, vehicle_positions, nearest_indices, road_directions):
        """计算车辆到道路的横向距离"""
        # 获取最近道路段的起点和终点
        nearest_centerlines = self.quad_centerlines[nearest_indices]  # (B, M, 2, 2)
        road_starts = nearest_centerlines[:, :, 0, :]  # (B, M, 2) - 道路起点
        road_ends = nearest_centerlines[:, :, 1, :]    # (B, M, 2) - 道路终点
        
        # 计算道路向量和长度
        road_vectors = road_ends - road_starts  # (B, M, 2)
        road_lengths = torch.norm(road_vectors, dim=-1, keepdim=True)  # (B, M, 1)
        road_lengths = torch.clamp(road_lengths, min=1e-8)
        
        # 计算投影参数
        to_start = vehicle_positions - road_starts  # (B, M, 2)
        t = torch.sum(to_start * road_vectors, dim=-1, keepdim=True) / (road_lengths ** 2)  # (B, M, 1)
        t = torch.clamp(t, 0, 1)
        
        # 计算投影点
        projection_points = road_starts + t * road_vectors  # (B, M, 2)
        to_projection = vehicle_positions - projection_points  # (B, M, 2)
        
        # 计算横向距离
        road_perpendicular = torch.stack([
            -road_directions[:, :, 1],
            road_directions[:, :, 0]
        ], dim=-1)  # (B, M, 2)
        return torch.sum(to_projection * road_perpendicular, dim=-1)  # (B, M)

# 为了让这个文件可以独立测试，添加一个 main block
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print("RoadNetwork 可视化测试")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查地图文件是否存在
    map_files = [
        'maps/processed_map_Town01_stitched.json',
    ]
    map_path = None
    for file_path in map_files:
        if os.path.exists(file_path):
            map_path = file_path
            print(f"使用地图文件: {map_path}")
            break
    if map_path is None:
        print("未找到可用的地图文件，创建示例数据")
        # 创建示例数据用于演示
        device = torch.device('cpu')
        road_network = None
    else:
        device = torch.device('cpu')
        try:
            road_network = RoadNetwork(map_path, device)
            print(f"成功加载道路网络，包含 {road_network.num_quads} 个道路段")
        except Exception as e:
            print(f"加载地图文件失败: {e}")
            road_network = None
    
    if road_network is not None:
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RoadNetwork 可视化', fontsize=16)
        
        # 1. 道路四边形可视化
        ax1 = axes[0, 0]
        ax1.set_title('道路四边形 (Quads)')
        num_quads = road_network.num_quads
        indices = torch.arange(num_quads)
        for i in indices:
            quad = road_network.quads_vertices[i].cpu().numpy()
            # 绘制四边形
            ax1.plot([quad[0, 0], quad[1, 0], quad[2, 0], quad[3, 0], quad[0, 0]], 
                     [quad[0, 1], quad[1, 1], quad[2, 1], quad[3, 1], quad[0, 1]], 
                     'b-', alpha=0.3, linewidth=0.5)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        
        # 2. 道路中心线可视化
        ax2 = axes[0, 1]
        ax2.set_title('道路中心线')
        
        centerlines = road_network.get_all_lanes_centerlines()[indices].cpu().numpy()
        for centerline in centerlines:
            ax2.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=1, alpha=0.7)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        
        # 3. 道路边界线可视化
        ax3 = axes[1, 0]
        ax3.set_title('道路边界线')
        left_boundaries = road_network.get_all_lanes_left_boundaries()[indices].cpu().numpy()
        right_boundaries = road_network.get_all_lanes_right_boundaries()[indices].cpu().numpy()
        for left_boundary in left_boundaries:
            ax3.plot(left_boundary[:, 0], left_boundary[:, 1], 'g-', linewidth=1, alpha=0.7)
        for right_boundary in right_boundaries:
            ax3.plot(right_boundary[:, 0], right_boundary[:, 1], 'orange', linewidth=1, alpha=0.7)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('X 坐标')
        ax3.set_ylabel('Y 坐标')
        ax3.legend(['左边界', '右边界'])
        
        # 4. 全局航点可视化
        ax4 = axes[1, 1]
        ax4.set_title('全局航点')
        # 绘制车道航点
        if road_network.global_w_lane_waypoints.numel() > 0:
            lane_waypoints = road_network.global_w_lane_waypoints.cpu().numpy()
            ax4.scatter(lane_waypoints[:, 0], lane_waypoints[:, 1], 
                       c='blue', s=10, alpha=0.6, label='车道航点')
        
        # 绘制边界航点
        if road_network.global_w_boundary_points.numel() > 0:
            boundary_waypoints = road_network.global_w_boundary_points.cpu().numpy()
            ax4.scatter(boundary_waypoints[:, 0], boundary_waypoints[:, 1], 
                       c='red', s=10, alpha=0.6, label='边界航点')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('X 坐标')
        ax4.set_ylabel('Y 坐标')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 5. Frenet 坐标系测试
        print("\n测试 Frenet 坐标系计算:")
        # 创建一些测试车辆位置和朝向
        test_positions = torch.tensor([
            [0.0, 0.0],
            [10.0, 5.0],
            [-5.0, 3.0]
        ], device=device)
        
        test_headings = torch.tensor([0.0, np.pi/4, -np.pi/6], device=device)
        
        try:
            d, theta_f = road_network.calculate_frenet_coordinates(test_positions, test_headings)
            print(f"车辆位置: {test_positions.cpu().numpy()}")
            print(f"车辆朝向: {test_headings.cpu().numpy()}")
            print(f"横向距离 d: {d.cpu().numpy()}")
            print(f"角度误差 theta_f: {theta_f.cpu().numpy()}")
        except Exception as e:
            print(f"Frenet 坐标系计算失败: {e}")
        
        # 6. 最近车道查找测试
        print("\n测试最近车道查找:")
        try:
            test_points = torch.tensor([[0.0, 0.0], [10.0, 10.0]], device=device)
            distances, indices = road_network.find_nearest_lanes(test_points, k=3)
            print(f"测试点: {test_points.cpu().numpy()}")
            print(f"最近车道距离: {distances.cpu().numpy()}")
            print(f"最近车道索引: {indices.cpu().numpy()}")
        except Exception as e:
            print(f"最近车道查找失败: {e}")
    
    else:
        print("无法创建道路网络，跳过可视化")
    
    print("RoadNetwork 测试完成")




import torch
import json
from typing import Dict, Tuple, List

class RoadNetwork:
    """
    无需更改,已经通过测试
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
        
        # 获取最近道路段的起点
        nearest_centerlines = self.quad_centerlines[nearest_indices]  # (B, M, 2, 2)
        road_starts = nearest_centerlines[:, :, 0, :]  # (B, M, 2) - 道路起点
        # 计算从道路起点到车辆位置的向量 AP = P - A
        AP = vehicle_positions - road_starts  # (B, M, 2)
        # 计算二维叉积 cross = (Px - Ax) * dy - (Py - Ay) * dx
        # 这等价于 AP × road_directions 的z分量
        cross = (AP[:, :, 0] * road_directions[:, :, 1] - 
                AP[:, :, 1] * road_directions[:, :, 0])  # (B, M)
        # 计算角度误差 theta_f
        # 使用叉积的符号来确定角度方向
        # 注意：这里计算的是 vehicle_directions 相对于 road_directions 的角度
        cross_product = (vehicle_directions[:, :, 0] * road_directions[:, :, 1] - 
                        vehicle_directions[:, :, 1] * road_directions[:, :, 0])
        dot_product = torch.sum(road_directions * vehicle_directions, dim=-1)
        theta_f = torch.atan2(cross_product, dot_product)  # (B, M)
        # 横向距离就是叉积值（带符号）
        # cross > 0: 车辆在道路左侧
        # cross < 0: 车辆在道路右侧  
        # cross = 0: 车辆在道路中心线上
        d = cross  # (B, M)
        return d, theta_f
    

# 为了让这个文件可以独立测试，添加一个 main block
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    print("RoadNetwork 测试")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 加载地图数据
    map_path = "maps/processed_map_Town01_stitched.json"
    print(f"加载地图: {map_path}")
    try:
        # 创建RoadNetwork实例
        road_network = RoadNetwork(map_path, device)
        # 获取quads顶点数据
        quads_vertices_np = road_network.quads_vertices.cpu().numpy()
        # 随机选择一个quad并在其中生成车辆位置
        random_quad_idx = random.randint(0, road_network.num_quads - 1)
        print(f"随机选择quad索引: {random_quad_idx}")

        # 获取选中quad的顶点
        selected_quad = quads_vertices_np[random_quad_idx]

        # 在quad范围内随机生成车辆位置
        # 使用重心坐标法在quad内随机生成点
        def random_point_in_quad(quad_vertices):
            # 生成随机重心坐标
            r1, r2 = np.random.random(2)
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = r2 * sqrt_r1
            # 计算随机点
            point = (1-u-v) * quad_vertices[0] + u * quad_vertices[1] + v * quad_vertices[2]
            return point
        vehicle_pos = random_point_in_quad(selected_quad)
        vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向
        
        # 绘制地图
        print("绘制地图...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 只绘制车辆周围的quads
        vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
        vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
        
        # 找到距离车辆最近的200个quads
        distances, nearest_indices = road_network.find_nearest_lanes(vehicle_pos_tensor, k=200)
        nearest_indices = nearest_indices.cpu().numpy().flatten()
        nearest_quad_idx = nearest_indices[0]  # 最近的quad索引
        print(f"距离车辆最近的quad索引: {nearest_quad_idx}")
        
        # 使用找到的最近200个quads作为附近quads
        nearby_quads = nearest_indices.tolist()
        print(f"车辆周围最近200个quads")
        
        # 绘制车辆周围的quads
        for i in nearby_quads:
            quad = quads_vertices_np[i]
            # 绘制quad边界
            quad_x = [quad[0][0], quad[1][0], quad[2][0], quad[3][0], quad[0][0]]
            quad_y = [quad[0][1], quad[1][1], quad[2][1], quad[3][1], quad[0][1]]
            # 判断是否为最近的quad，决定颜色
            if i == nearest_quad_idx:
                # 最近的quad用红色
                ax.plot(quad_x, quad_y, 'r-', alpha=0.5, linewidth=2, label='nearest quad')
                centerline = road_network.quad_centerlines[i].cpu().numpy()
                ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=3, alpha=0.8)
                
                
                # 为最近quad的中线添加箭头
                start_point = centerline[0]
                end_point = centerline[1]
                # 计算箭头位置（在中心线的中点）
                arrow_pos = (start_point + end_point) / 2
                # 计算箭头方向
                arrow_direction = end_point - start_point
                arrow_length = np.linalg.norm(arrow_direction) * 0.3  # 箭头长度为线段长度的30%
                arrow_direction_normalized = arrow_direction / np.linalg.norm(arrow_direction)
                # 绘制箭头
                ax.arrow(arrow_pos[0], arrow_pos[1], 
                        arrow_direction_normalized[0] * arrow_length, 
                        arrow_direction_normalized[1] * arrow_length,
                        head_width=3, head_length=2, fc='red', ec='red', alpha=0.8)
                
            else:
                # 其他quad用蓝色
                ax.plot(quad_x, quad_y, 'b-', alpha=0.3, linewidth=0.5)
                centerline = road_network.quad_centerlines[i].cpu().numpy()
                ax.plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # 在图上标记车辆位置
        ax.plot(vehicle_pos[0], vehicle_pos[1], 'go', markersize=10, label='vehicle position')
        
        # 只显示一次图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # 绘制车辆朝向
        arrow_length = 5.0
        arrow_dx = arrow_length * np.cos(vehicle_yaw)
        arrow_dy = arrow_length * np.sin(vehicle_yaw)
        ax.arrow(vehicle_pos[0], vehicle_pos[1], arrow_dx, arrow_dy, 
                head_width=2, head_length=1, fc='green', ec='green', alpha=0.8)
        # 计算Frenet坐标
        vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
        vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
        vehicle_yaw_tensor = torch.tensor([vehicle_yaw], dtype=torch.float32, device=device)
        print("计算Frenet坐标...")
        d, theta_f = road_network.calculate_frenet_coordinates(vehicle_pos_tensor, vehicle_yaw_tensor)
        print(f"横向距离 d: {d.item():.2f} (正值表示在道路右侧，负值表示在道路左侧)")
        print(f"角度误差 theta_f: {theta_f.item():.2f} 弧度 ({np.degrees(theta_f.item()):.1f} 度)")
        print(f"角度误差解释: 正值表示车辆朝向偏右，负值表示偏左")

        # 设置图形属性
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('RoadNetwork Test - Map Visualization and Frenet Coordinate Calculation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # 保存图片
        plt.savefig('road_network_test.png', dpi=300, bbox_inches='tight')
        print("地图已保存为 road_network_test.png")
        # 显示图形
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到地图文件 {map_path}")
        print("请确保地图文件存在，或者修改map_path变量指向正确的地图文件")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
    




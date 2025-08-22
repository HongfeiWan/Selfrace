# teraflow_replication/simulator/offroad.py
import torch
from torch import Tensor
from typing import Tuple
import sys
from road import RoadNetwork
import os
import sys
import time
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
        基于矢量叉乘（半平面）的方法：
        1) 用空间哈希取候选 (point, quad)；
        2) 对每个候选，计算四条边的 cross(e, p - v)；
        3) 若多边形为顺时针，则 cross <= 0，全为右侧；若为逆时针则 cross >= 0；
           统一写作 (sign * cross) >= -eps，sign=+1(CCW), -1(CW)。
        4) 命中的点按原索引 scatter 回去。
        """

        M = points.shape[0]
        if M == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        candidate_pairs = self.spatial_hash.query_points(points)
        if candidate_pairs.shape[0] == 0:
            return torch.zeros(M, dtype=torch.bool, device=self.device)
        point_indices = candidate_pairs[:, 0]
        polygon_indices = candidate_pairs[:, 1]

        pts = points[point_indices]
        verts = self.poly_verts[polygon_indices]
        edges = self.poly_edges[polygon_indices]
        sign = self.poly_sign[polygon_indices]
        pv = pts.unsqueeze(1) - verts
        cross = edges[..., 0] * pv[..., 1] - edges[..., 1] * pv[..., 0]
        inside = (sign.unsqueeze(-1) * cross >= -1e-10).all(dim=-1)

        # 修复版本：使用scatter_add_来正确处理一个点被多个多边形包含的情况
        # 一个点只要被任何一个多边形包含，就应该被认为是"在道路上"
        flat_on_road_mask = torch.zeros(M, dtype=torch.int32, device=self.device)
        flat_on_road_mask.scatter_add_(0, point_indices, inside.to(torch.int32))
        flat_on_road_mask = flat_on_road_mask.gt_(0)  # 只要有一个多边形包含该点，就为True
        
        return flat_on_road_mask

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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # 添加utils目录到路径
    utils_dir = os.path.join(parent_dir, 'utils')
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from spatial_hash import SpatialHash
    from road import RoadNetwork
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import os
    import sys
    map_path = "maps/processed_map_Town01_stitched.json"
    device = torch.device('cuda')
    road_network = RoadNetwork(map_path, device)
    # 提取quads的顶点
    quads_vertices_np = road_network.quads_vertices.cpu().numpy()
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
        # 固定车辆位置，测试10个不同朝向
        vehicle_pos = random_point_in_quad(selected_quad)
        vehicle_yaws = [random.uniform(0, 2 * np.pi) for _ in range(10)]  # 10个随机朝向

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
        
        # 绘制车辆矩形
        def draw_vehicle(ax, x, y, yaw, length=4.5, width=2.0, color='green', alpha=0.8):
            """绘制车辆矩形"""
            # 车辆矩形的四个角点（相对于车辆中心）
            half_length = length / 2
            half_width = width / 2
            
            # 车辆矩形的四个角点（相对于车辆中心）
            corners = np.array([
                [-half_length, -half_width],  # 左下
                [half_length, -half_width],   # 右下
                [half_length, half_width],    # 右上
                [-half_length, half_width]    # 左上
            ])
            
            # 旋转矩阵
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            # 旋转角点
            rotated_corners = corners @ rotation_matrix.T
            
            # 平移到车辆位置
            vehicle_corners = rotated_corners + np.array([x, y])
            
            # 绘制车辆矩形
            vehicle_x = np.append(vehicle_corners[:, 0], vehicle_corners[0, 0])
            vehicle_y = np.append(vehicle_corners[:, 1], vehicle_corners[0, 1])
            ax.plot(vehicle_x, vehicle_y, color=color, linewidth=2, alpha=alpha)
            
            # 绘制车辆朝向箭头
            arrow_length = 3.0
            arrow_dx = arrow_length * cos_yaw
            arrow_dy = arrow_length * sin_yaw
            ax.arrow(x, y, arrow_dx, arrow_dy, 
                    head_width=1, head_length=0.5, fc=color, ec=color, alpha=alpha)
            # 标记车辆中心
            ax.plot(x, y, 'o', color=color, markersize=4, alpha=alpha)
            
        # 验证车辆是否在道路上
        print("🔍 验证车辆是否在道路上...")
        
        # 创建SpatialHash实例，使用正确的参数
        cell_size = 20.0  # 网格单元大小
        min_bounds = torch.tensor([-1000, -1000], device=device)  # 最小边界
        max_bounds = torch.tensor([1000, 1000], device=device)    # 最大边界
        spatial_hash = SpatialHash(cell_size=cell_size, min_bounds=min_bounds, max_bounds=max_bounds, device=device)
        offroad_checker = OffroadChecker(road_network, spatial_hash, points_per_vehicle_edge=3)
        
        # 测试10个不同朝向
        print(f"📍 车辆位置: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f})")
        print(f"📏 车辆尺寸: 4.5m × 2.0m")
        print("🧭 测试10个不同朝向的离路状态:")
        
        for i, yaw in enumerate(vehicle_yaws):
            # 准备车辆状态数据 [x, y, heading, length, width]
            vehicle_state = torch.tensor([
                [vehicle_pos[0], vehicle_pos[1], yaw, 4.5, 2.0]
            ], dtype=torch.float32, device=device)
            
            # 检查车辆是否在道路上
            is_on_road = offroad_checker.check_on_road(vehicle_state)
            
            # 确定车辆颜色
            if is_on_road[0].item():
                vehicle_color = 'green'
                status = "✅ 在道路上"
                alpha = 0.7
            else:
                vehicle_color = 'red'
                status = "❌ 离路"
                alpha = 0.3
            
            # 绘制车辆
            draw_vehicle(ax, vehicle_pos[0], vehicle_pos[1], yaw, 
                       length=4.5, width=2.0, color=vehicle_color, alpha=alpha)
        
            # 计算Frenet坐标
            vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
            vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
            vehicle_yaw_tensor = torch.tensor([vehicle_yaws[i]], dtype=torch.float32, device=device)
            d, theta_f = road_network.calculate_frenet_coordinates(vehicle_pos_tensor, vehicle_yaw_tensor)

            # 打印结果
            print(f"  朝向 {i+1}: {np.degrees(yaw):.1f}° - {status}")
            print(f"  横向距离 d: {d.item():.2f} (正值表示在道路右侧，负值表示在道路左侧)")
            print(f"  角度误差 theta_f: {theta_f.item():.2f} 弧度 ({np.degrees(theta_f.item()):.1f} 度)")
            print(f"  角度误差解释: 正值表示车辆朝向偏右，负值表示偏左")

        # 添加图例说明
        ax.plot([], [], color='green', linewidth=2, label='Vehicle (On Road)')
        ax.plot([], [], color='red', linewidth=2, label='Vehicle (Off Road)')
        
        # 只显示一次图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

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

    
   
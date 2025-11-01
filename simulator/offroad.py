# teraflow_replication/simulator/offroad.py
import torch
from torch import Tensor
from road import RoadNetwork
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.spatial_hash import SpatialHash


class OffroadChecker:
    """
    一个基于 GPU 加速的批量化离路检测器。
    它使用一个共享的、预初始化的 SpatialHash 对象来执行查询。
    """
    def __init__(self, RoadNetwork: RoadNetwork, spatial_hash: SpatialHash, points_per_vehicle_edge: int = 3):
        """
        初始化离路检测器。
        Args:
            RoadNetwork: 包含路面几何信息的 RoadNetwork 对象。
            spatial_hash (SpatialHash): 预初始化的空间哈希对象。
            points_per_vehicle_edge (int): 沿着车辆边界框每条边采样的点数。
        """
        # 设备/基本属性
        self.device = RoadNetwork.device
        self.points_per_vehicle_edge = int(points_per_vehicle_edge)
        # 路面多边形与空间哈希
        self.road_polygons = torch.empty((0, 4, 2), dtype=torch.float32, device=self.device)
        self.spatial_hash = spatial_hash
        # 局部边界框采样点模板
        self.local_bbox_points = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        # 预计算的凸四边形参数
        self.poly_verts = torch.empty((0, 4, 2), dtype=torch.float32, device=self.device)
        self.poly_edges = torch.empty((0, 4, 2), dtype=torch.float32, device=self.device)
        self.poly_sign = torch.empty((0,), dtype=torch.float32, device=self.device)
        if self.points_per_vehicle_edge < 2:
            raise ValueError("points_per_vehicle_edge must be at least 2.")
        # 赋值真实数据
        self.road_polygons = RoadNetwork.quads_vertices.to(self.device)
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


# if __name__ == "__main__":
#     from utils.geometry_utils import find_nearest_lanes, calculate_frenet_coordinates
#     import matplotlib.pyplot as plt
#     import random
#     import numpy as np
#     import os
#     import json
#     # 从配置读取地图目录，并指定 town2.json
#     config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
#     with open(config_path, 'r', encoding='utf-8') as f:
#         cfg = json.load(f)
#     map_dir_cfg = cfg.get('map_path', './maps')
#     repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
#     maps_dir = os.path.normpath(os.path.join(repo_root, map_dir_cfg))
#     map_path = os.path.join(maps_dir, 'town2.json')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     try:
#         # 创建RoadNetwork实例
#         road_network = RoadNetwork(map_path, device=device)
#         # 获取quads顶点数据
#         quads_vertices_np = road_network.quads_vertices.cpu().numpy()
#         # 随机选择一个quad并在其中生成车辆位置
#         num_quads = road_network.quads_vertices.shape[0]
#         random_quad_idx = random.randint(0, max(0, num_quads - 1))
#         print(f"随机选择quad索引: {random_quad_idx}")
#         # 获取选中quad的顶点
#         selected_quad = quads_vertices_np[random_quad_idx]
#         # 在quad范围内随机生成车辆位置
#         # 使用重心坐标法在quad内随机生成点
#         def random_point_in_quad(quad_vertices):
#             # 生成随机重心坐标
#             r1, r2 = np.random.random(2)
#             sqrt_r1 = np.sqrt(r1)
#             u = 1 - sqrt_r1
#             v = r2 * sqrt_r1
#             # 计算随机点
#             point = (1-u-v) * quad_vertices[0] + u * quad_vertices[1] + v * quad_vertices[2]
#             return point
#         # 固定车辆位置，测试10个不同朝向
#         vehicle_pos = random_point_in_quad(selected_quad)
#         current_yaw = random.uniform(0, 2 * np.pi)  # 初始随机朝向
#         current_yaw = [current_yaw]  # 可变容器，便于在回调中更新

#         # 绘制地图
#         print("绘制地图...")
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         # 只绘制车辆周围的quads
#         vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
#         vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)

#         # 找到距离车辆最近的200个quads（使用工具函数）
#         distances, nearest_indices = find_nearest_lanes(device, road_network.quad_centerlines, vehicle_pos_tensor, k=200)
#         nearest_indices = nearest_indices.cpu().numpy().flatten()
#         nearest_quad_idx = nearest_indices[0]  # 最近的quad索引
#         print(f"距离车辆最近的quad索引: {nearest_quad_idx}")
        
#         # 使用找到的最近200个quads作为附近quads
#         nearby_quads = nearest_indices.tolist()
#         # 车辆周围最近200个quads（仅用于绘制参考）
        
#         # 绘制车辆周围的quads
#         for i in nearby_quads:
#             quad = quads_vertices_np[i]
#             # 绘制quad边界
#             quad_x = [quad[0][0], quad[1][0], quad[2][0], quad[3][0], quad[0][0]]
#             quad_y = [quad[0][1], quad[1][1], quad[2][1], quad[3][1], quad[0][1]]
#             # 判断是否为最近的quad，决定颜色
#             if i == nearest_quad_idx:
#                 # 最近的quad用红色
#                 ax.plot(quad_x, quad_y, 'r-', alpha=0.5, linewidth=2, label='nearest quad')
#                 centerline = road_network.quad_centerlines[i].cpu().numpy()
#                 ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=3, alpha=0.8)
#                 # 为最近quad的方向添加箭头（使用地图中的quad方向）
#                 start_point = centerline[0]
#                 end_point = centerline[1]
#                 arrow_pos = (start_point + end_point) / 2  # 在中心线的中点绘制
#                 dir_vec = road_network.quad_directions[i].cpu().numpy()  # (dx, dy) 单位向量
#                 # 设定箭头长度相对中心线长度
#                 cl_len = np.linalg.norm(end_point - start_point)
#                 arrow_length = max(1e-6, cl_len * 0.3)
#                 arrow_dx = dir_vec[0] * arrow_length
#                 arrow_dy = dir_vec[1] * arrow_length
#                 ax.arrow(arrow_pos[0], arrow_pos[1],
#                          arrow_dx,
#                          arrow_dy,
#                          head_width=3, head_length=2, fc='red', ec='red', alpha=0.8)
                
#             else:
#                 # 其他quad用蓝色
#                 ax.plot(quad_x, quad_y, 'b-', alpha=0.3, linewidth=0.5)
#                 centerline = road_network.quad_centerlines[i].cpu().numpy()
#                 ax.plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=1, alpha=0.5)
        
#         # 绘制车辆矩形
#         def draw_vehicle(ax, x, y, yaw, length=4.5, width=2.0, color='green', alpha=0.8):
#             """绘制车辆矩形"""
#             # 车辆矩形的四个角点（相对于车辆中心）
#             half_length = length / 2
#             half_width = width / 2
            
#             # 车辆矩形的四个角点（相对于车辆中心）
#             corners = np.array([
#                 [-half_length, -half_width],  # 左下
#                 [half_length, -half_width],   # 右下
#                 [half_length, half_width],    # 右上
#                 [-half_length, half_width]    # 左上
#             ])
            
#             # 旋转矩阵
#             cos_yaw = np.cos(yaw)
#             sin_yaw = np.sin(yaw)
#             rotation_matrix = np.array([
#                 [cos_yaw, -sin_yaw],
#                 [sin_yaw, cos_yaw]
#             ])
            
#             # 旋转角点
#             rotated_corners = corners @ rotation_matrix.T
            
#             # 平移到车辆位置
#             vehicle_corners = rotated_corners + np.array([x, y])
            
#             # 绘制车辆矩形
#             vehicle_x = np.append(vehicle_corners[:, 0], vehicle_corners[0, 0])
#             vehicle_y = np.append(vehicle_corners[:, 1], vehicle_corners[0, 1])
#             line_list = ax.plot(vehicle_x, vehicle_y, color=color, linewidth=2, alpha=alpha)
#             artists_out = list(line_list)
            
#             # 绘制车辆朝向箭头
#             arrow_length = 3.0
#             arrow_dx = arrow_length * cos_yaw
#             arrow_dy = arrow_length * sin_yaw
#             arr = ax.arrow(x, y, arrow_dx, arrow_dy, 
#                            head_width=1, head_length=0.5, fc=color, ec=color, alpha=alpha)
#             artists_out.append(arr)
#             # 标记车辆中心
#             dot_list = ax.plot(x, y, 'o', color=color, markersize=4, alpha=alpha)
#             artists_out.extend(dot_list)
#             return artists_out
            
#         # 验证车辆是否在道路上
#         print("🔍 验证车辆是否在道路上...")
        
#         # 创建SpatialHash实例（从配置读取地图路径已完成，这里仅构建哈希）
#         cell_size = 20.0  # 网格单元大小
#         # 使用道路AABB范围更合理，这里示例保持原逻辑
#         min_bounds = torch.tensor([-1000, -1000], device=device)
#         max_bounds = torch.tensor([1000, 1000], device=device)
#         spatial_hash = SpatialHash(cell_size=cell_size, min_bounds=min_bounds, max_bounds=max_bounds, device=device)
#         offroad_checker = OffroadChecker(road_network, spatial_hash, points_per_vehicle_edge=3)

#         print(f"📍 车辆位置: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f})")
#         print(f"📏 车辆尺寸: 4.5m × 2.0m")
#         print("🧭 交互测试：使用键盘左右方向键调整车辆朝向。")
        
#         vehicle_artists = []
#         def redraw_vehicle():
#             # 清理旧的车辆元素
#             for a in vehicle_artists:
#                 try:
#                     a.remove()
#                 except Exception:
#                     pass
#             vehicle_artists.clear()
#             # 计算在/离路
#             vehicle_state = torch.tensor([[vehicle_pos[0], vehicle_pos[1], current_yaw[0], 4.5, 2.0]], dtype=torch.float32, device=device)
#             is_on_road = offroad_checker.check_on_road(vehicle_state)
#             if is_on_road[0].item():
#                 vehicle_color, alpha, status = 'green', 0.7, "✅ 在道路上"
#             else:
#                 vehicle_color, alpha, status = 'red', 0.3, "❌ 离路"
#             # 绘制车辆与朝向
#             vehicle_artists.extend(
#                 draw_vehicle(ax, vehicle_pos[0], vehicle_pos[1], current_yaw[0],
#                              length=4.5, width=2.0, color=vehicle_color, alpha=alpha)
#             )
#             # 计算Frenet
#             vehicle_pos_tensor = torch.tensor([vehicle_pos], dtype=torch.float32, device=device)
#             vehicle_yaw_tensor = torch.tensor([current_yaw[0]], dtype=torch.float32, device=device)
#             d, theta_f = calculate_frenet_coordinates(device,
#                                                       road_network.quad_directions,
#                                                       road_network.quad_centerlines,
#                                                       vehicle_pos_tensor,
#                                                       vehicle_yaw_tensor,
#                                                       k=1,
#                                                       spatial_hash=None)
#             print(f"  朝向: {np.degrees(current_yaw[0]):.1f}° - {status}")
#             print(f"  横向距离 d: {d.item():.2f}")
#             print(f"  角度误差 theta_f: {theta_f.item():.2f} rad ({np.degrees(theta_f.item()):.1f}°)")
#             ax.figure.canvas.draw_idle()

#         def on_key(event):
#             step = np.radians(5.0)
#             if event.key in ['left', 'kp_left']:
#                 current_yaw[0] = (current_yaw[0] + step) % (2 * np.pi)
#                 redraw_vehicle()
#             elif event.key in ['right', 'kp_right']:
#                 current_yaw[0] = (current_yaw[0] - step) % (2 * np.pi)
#                 redraw_vehicle()

#         # 初次绘制
#         redraw_vehicle()
#         fig.canvas.mpl_connect('key_press_event', on_key)

#         # 添加图例说明
#         ax.plot([], [], color='green', linewidth=2, label='Vehicle (On Road)')
#         ax.plot([], [], color='red', linewidth=2, label='Vehicle (Off Road)')
        
#         # 只显示一次图例
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys())

#         # 设置图形属性
#         ax.set_xlabel('X Coordinate')
#         ax.set_ylabel('Y Coordinate')
#         ax.set_title('RoadNetwork Test - Map Visualization and Frenet Coordinate Calculation')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal')
#         # 保存图片
#         plt.savefig('road_network_test.png', dpi=300, bbox_inches='tight')
#         print("地图已保存为 road_network_test.png")
#         # 显示图形
#         plt.show()

#     except FileNotFoundError:
#         print(f"错误: 找不到地图文件 {map_path}")
#         print("请确保地图文件存在，或者修改map_path变量指向正确的地图文件")
#     except Exception as e:
#         print(f"测试过程中发生错误: {e}")
#         import traceback
#         traceback.print_exc()

    
   
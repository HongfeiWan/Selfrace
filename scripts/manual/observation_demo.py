"""Manual diagnostics moved from simulator/observation.py."""

from _bootstrap import add_project_paths

add_project_paths()

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from observation import ObservationGenerator
from road import RoadNetwork
from spatial_hash import SpatialHash
print("RoadNetwork 测试")
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# 加载地图数据
map_path = "maps/processed_map_Town01_stitched.json"
print(f"加载地图: {map_path}")

# 绘制车辆矩形
def draw_vehicle(ax, x, y, yaw, speed , length=4.5, width=2.0, color='green', alpha=0.8):
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
    arrow_length = speed
    arrow_dx = arrow_length * cos_yaw
    arrow_dy = arrow_length * sin_yaw
    ax.arrow(x, y, arrow_dx, arrow_dy, 
            head_width=1, head_length=0.5, fc=color, ec=color, alpha=alpha)
    # 标记车辆中心
    ax.plot(x, y, 'o', color=color, markersize=4, alpha=alpha)    

try:
    # 创建RoadNetwork实例
    road_network = RoadNetwork(map_path, device)
    # 获取quads顶点数据
    quads_vertices_np = road_network.quads_vertices.cpu().numpy()
    # 随机选择一个quad_id
    random_quad_id = random.choice(road_network.quad_ids.cpu().numpy())
    print(f"随机选择quad_id: {random_quad_id}")
    # 根据quad_id找到对应的索引
    quad_id_positions = torch.where(road_network.quad_ids == random_quad_id)[0]
    random_quad_idx = quad_id_positions[0].item()

    # 获取选中quad的顶点
    selected_quad = quads_vertices_np[random_quad_idx]
    # 在quad范围内随机生成车辆位置
    # 改进的随机点生成方法，确保点在quad内
    def random_point_in_quad_improved(quad_vertices):
        """改进的quad内随机点生成，确保点在quad内部"""
        # 计算quad的边界框
        min_x, min_y = np.min(quad_vertices, axis=0)
        max_x, max_y = np.max(quad_vertices, axis=0)

        # 在边界框内随机生成点，直到找到在quad内的点
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y])

            # 检查点是否在quad内（使用射线法）
            if is_point_in_quad(point, quad_vertices):
                return point

        # 如果失败，返回quad的中心点
        center = np.mean(quad_vertices, axis=0)
        print(f"警告：无法在quad内生成随机点，使用中心点: {center}")
        return center

    def is_point_in_quad(point, quad_vertices):
        """使用射线法判断点是否在quad内"""
        x, y = point
        n = len(quad_vertices)
        inside = False

        p1x, p1y = quad_vertices[0]
        for i in range(n + 1):
            p2x, p2y = quad_vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # 使用改进的方法生成车辆位置
    vehicle_pos = random_point_in_quad_improved(selected_quad)
    vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向
    # 绘制地图
    print("绘制地图...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # 只绘制ego周围的quads
    vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
    vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
    # 找到距离ego最近的quads
    distances, nearest_indices = road_network.find_nearest_lanes(vehicle_pos_tensor, k=400)
    nearest_indices = nearest_indices.cpu().numpy().flatten()
    nearest_quad_idx = nearest_indices[0]  # 最近的quad索引
    nearby_quads = nearest_indices.tolist()

    # 在nearby_quads中随机选择一个quad生成第二辆车
    if len(nearby_quads) > 1:
        # 随机选择一个不同于最近quad的quad
        available_quads = [q for q in nearby_quads if q != nearest_quad_idx]
        if available_quads:
            second_quad_idx = random.choice(available_quads)
            second_quad = quads_vertices_np[second_quad_idx]
            # 在第二个quad中生成随机车辆位置
            second_vehicle_pos = random_point_in_quad_improved(second_quad)
            second_vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向

    # 创建agents_state (B=1, M=2, 7个特征)
    agents_state = torch.zeros(1, 2, 7, device=device)
    # 第一辆车的信息
    agents_state[0, 0, 0] = float(vehicle_pos[0])  # x
    agents_state[0, 0, 1] = float(vehicle_pos[1])  # y
    agents_state[0, 0, 2] = float(vehicle_yaw)     # yaw
    agents_state[0, 0, 3] = 10.0                   # speed (m/s)
    agents_state[0, 0, 4] = 4.5                    # vehicle_length (m)
    agents_state[0, 0, 5] = 2.0                    # vehicle_width (m)
    agents_state[0, 0, 6] = 1.0                    # active
    # 第二辆车的信息（如果存在）
    if len(nearby_quads) > 1 and 'second_vehicle_pos' in locals():
        agents_state[0, 1, 0] = float(second_vehicle_pos[0])  # x
        agents_state[0, 1, 1] = float(second_vehicle_pos[1])  # y
        agents_state[0, 1, 2] = float(second_vehicle_yaw)     # yaw
        agents_state[0, 1, 3] = 8.0                           # speed (m/s)
        agents_state[0, 1, 4] = 4.5                           # vehicle_length (m)
        agents_state[0, 1, 5] = 2.0                           # vehicle_width (m)
        agents_state[0, 1, 6] = 1.0                           # active




    # 测试ObservationGenerator
    print("\n=== 测试ObservationGenerator ===")
    # 创建配置字典
    config = {
        'num_neighbors': 1,  # 只有2个agents，所以最多1个邻居
        'num_w_lanes': 25,
        'num_w_boundaries': 26,
        'horizon': 100.0,
        'local_state_dim': 7,  # 修改为7个特征：x, y, yaw, speed, length, width, active
        'neighbor_feature_dim': 7,  # 修改为7个特征：dx, dy, vx, vy, length, width, active
        'waypoint_feature_dim': 2,
        'boundary_feature_dim': 2
    }

    # 创建空间哈希用于加速查询
    # 计算所有quad的边界框
    all_verts = road_network.quads_vertices.view(-1, 2)
    min_bounds, _ = torch.min(all_verts, dim=0)
    max_bounds, _ = torch.max(all_verts, dim=0)
    # 设置合适的网格大小
    cell_size = 5  # 5米的网格单元
    spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, device)
    # 构建静态索引
    quad_centers = road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
    quad_min_bounds = torch.min(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
    quad_max_bounds = torch.max(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
    quad_bounds = torch.stack([quad_min_bounds, quad_max_bounds], dim=1)  # (num_quads, 2, 2)
    spatial_hash.build_static_index(quad_bounds)
    print(f"空间哈希网格创建完成，网格大小: {cell_size:.2f}m")

    # 创建ObservationGenerator实例
    observation_generator = ObservationGenerator(road_network, config, device, spatial_hash)
    print(f"观测维度: {observation_generator.get_observation_dim()}")

    # 使用find_nearest_lanes找到最近的quad
    distances, quad_indices = road_network.find_nearest_lanes(agents_state[0, 0, :2], k=1, spatial_hash=spatial_hash)
    # 获取对应的quad_id (polyId)
    quad_id = road_network.quad_ids[quad_indices.squeeze(-1)]
    print(f"第一辆车所在quad_id: {quad_id.item()}")

    # 验证quad_id一致性
    print("\n=== 验证quad_id一致性 ===")
    print(f"随机选择的quad_id: {random_quad_id}")
    print(f"find_nearest_lanes得到的quad_id: {quad_id.item()}")
    print(f"是否一致: {random_quad_id == quad_id.item()}")

    if random_quad_id != quad_id.item():
        print("不一致的原因分析:")
        print(f"1. 车辆位置: {vehicle_pos}")

        # 计算车辆到随机选择quad的距离
        random_quad_center = road_network.quad_centerlines[random_quad_idx].mean(dim=0)
        dist_to_random = torch.norm(torch.tensor(vehicle_pos, device=device) - random_quad_center)
        print(f"2. 车辆到随机quad中心的距离: {dist_to_random.item():.2f}")

        # 计算车辆到最近quad的距离
        nearest_quad_center = road_network.quad_centerlines[quad_indices.item()].mean(dim=0)
        dist_to_nearest = torch.norm(torch.tensor(vehicle_pos, device=device) - nearest_quad_center)
        print(f"3. 车辆到最近quad中心的距离: {dist_to_nearest.item():.2f}")

        print(f"4. 距离差异: {abs(dist_to_random - dist_to_nearest).item():.2f}")

        # 检查车辆是否真的在随机选择的quad内
        def is_point_in_quad(point, quad_vertices):
            """使用射线法判断点是否在quad内"""
            x, y = point
            n = len(quad_vertices)
            inside = False

            p1x, p1y = quad_vertices[0]
            for i in range(n + 1):
                p2x, p2y = quad_vertices[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        is_in_random_quad = is_point_in_quad(vehicle_pos, selected_quad)
        print(f"5. 车辆是否在随机选择的quad内: {is_in_random_quad}")

        if not is_in_random_quad:
            print("6. 原因：重心坐标法生成的点不在quad内！")
            print("7. 建议：使用改进的随机点生成方法")

    # 生成观测
    observation = observation_generator.generate(agents_state)
    # 从observation中提取w_lanes_local和w_boundaries_local
    # 计算各部分在观测向量中的位置
    local_state_dim = config['local_state_dim']
    neighbor_feature_dim = config['neighbor_feature_dim']
    num_neighbors = config['num_neighbors']
    num_w_lanes = config['num_w_lanes']
    num_w_boundaries = config['num_w_boundaries']
    waypoint_feature_dim = config['waypoint_feature_dim']
    boundary_feature_dim = config['boundary_feature_dim']

    # 计算各部分在观测向量中的位置
    local_state_size = local_state_dim
    neighbors_size = num_neighbors * neighbor_feature_dim
    w_lanes_size = num_w_lanes * waypoint_feature_dim
    w_boundaries_size = num_w_boundaries * waypoint_feature_dim

    # 提取第一辆车的w_lanes_local和w_boundaries_local
    vehicle1_obs = observation[0, 0].cpu().numpy()
    w_lanes_start = local_state_size + neighbors_size
    w_boundaries_start = w_lanes_start + w_lanes_size
    w_lanes_local = vehicle1_obs[w_lanes_start:w_boundaries_start].reshape(num_w_lanes, waypoint_feature_dim)
    w_boundaries_local = vehicle1_obs[w_boundaries_start:].reshape(num_w_boundaries, waypoint_feature_dim)

    # 获取第一辆车的世界坐标和朝向（从agents_state中获取，确保一致性）
    vehicle_world_pos = np.array([float(agents_state[0, 0, 0]), float(agents_state[0, 0, 1])])
    vehicle_world_yaw = float(agents_state[0, 0, 3])  # 注意：agents_state[..., 3]是yaw

    # 绘制w_lanes_local (车道线)
    if w_lanes_local.shape[0] > 0:
        # 过滤掉无效的点（全零或NaN）
        valid_lanes = w_lanes_local[~np.all(w_lanes_local == 0, axis=1)]
        valid_lanes = valid_lanes[~np.any(np.isnan(valid_lanes), axis=1)]
        if valid_lanes.shape[0] > 0:
            # 逆变换：从local坐标转换回world坐标
            # 按照正确代码的实现方式
            ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
            cos_yaw = np.cos(ego_yaw)
            sin_yaw = np.sin(ego_yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            ego_pos_global = np.array([ego_x, ego_y])
            world_lanes = (valid_lanes @ rotation_matrix.T) + ego_pos_global
            # 绘制车道线点
            ax.scatter(world_lanes[:, 0], world_lanes[:, 1], c='orange', s=20, alpha=0.8, label='w_lanes_local')
        else:
            print("没有有效的车道线点")

    # 绘制w_boundaries_local (边界线)
    if w_boundaries_local.shape[0] > 0:
        # 过滤掉无效的点（全零或NaN）
        valid_boundaries = w_boundaries_local[~np.all(w_boundaries_local == 0, axis=1)]
        valid_boundaries = valid_boundaries[~np.any(np.isnan(valid_boundaries), axis=1)]
        if valid_boundaries.shape[0] > 0:
            # 逆变换：从local坐标转换回world坐标
            # 按照正确代码的实现方式
            ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
            cos_yaw = np.cos(ego_yaw)
            sin_yaw = np.sin(ego_yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            ego_pos_global = np.array([ego_x, ego_y])
            world_boundaries = (valid_boundaries @ rotation_matrix.T) + ego_pos_global
            # 绘制边界线点
            ax.scatter(world_boundaries[:, 0], world_boundaries[:, 1], c='purple', s=15, alpha=0.8, label='w_boundaries_local')
        else:
            print("没有有效的边界线点")

    # 计算邻居特征在observation中的位置
    neighbors_start = local_state_dim
    neighbors_end = neighbors_start + num_neighbors * neighbor_feature_dim
    # 提取第一辆车的邻居观测
    neighbors_obs = vehicle1_obs[neighbors_start:neighbors_end].reshape(num_neighbors, neighbor_feature_dim)
    # 过滤掉无效的邻居（全零）
    valid_neighbors = neighbors_obs[np.any(neighbors_obs != 0, axis=1)]
    if valid_neighbors.shape[0] > 0:
        print(f"观测到 {valid_neighbors.shape[0]} 个有效邻居")
        # 获取ego车辆信息用于逆变换
        ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        ego_pos_global = np.array([ego_x, ego_y])
        for i, neighbor_local in enumerate(valid_neighbors):
            # neighbor_local包含7个特征：[dx, dy, vx, vy, length, width, active]
            dx_local, dy_local, vx_local, vy_local, length, width, active = neighbor_local
            if active > 0.5:  # 只绘制活跃的邻居
                # 1. 逆变换邻居位置：从局部坐标转换回全局坐标
                neighbor_pos_local = np.array([dx_local, dy_local])
                neighbor_pos_global = (neighbor_pos_local @ rotation_matrix.T) + ego_pos_global

                # 2. 逆变换邻居相对速度：从局部坐标转换回全局坐标
                neighbor_vel_local = np.array([vx_local, vy_local])
                neighbor_vel_global = neighbor_vel_local @ rotation_matrix.T

                # 在邻居位置旁边显示相对速度文本
                ax.text(neighbor_pos_global[0] + 2, neighbor_pos_global[1] + 2, 
                       f'relative vel: ({vx_local:.1f}, {vy_local:.1f})', 
                       color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

                # 3. 计算邻居的绝对速度 = ego绝对速度 + 邻居相对速度
                ego_speed = agents_state[0, 0, 3].cpu().numpy()
                ego_yaw_rad = agents_state[0, 0, 2].cpu().numpy()
                ego_vel_global = np.array([
                    ego_speed * np.cos(ego_yaw_rad),
                    ego_speed * np.sin(ego_yaw_rad)
                ])
                neighbor_vel_absolute_global = ego_vel_global + neighbor_vel_global

                # 3. 绘制邻居位置
                ax.scatter(neighbor_pos_global[0], neighbor_pos_global[1], c='red', s=10, alpha=0.8, marker='o', label=f'Neighbor_{i}' if i == 0 else "")
                # 4. 绘制邻居相对速度箭头（正交分解）
                vel_x_arrow = neighbor_vel_absolute_global[0] 
                vel_y_arrow = neighbor_vel_absolute_global[1] 
                # X方向相对速度箭头（红色）
                if abs(vel_x_arrow) > 0.1:  # 只绘制有意义的箭头
                    ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                            vel_x_arrow, 0, head_width=2, head_length=1, 
                            fc='red', ec='red', alpha=0.8, zorder=10)
                # Y方向相对速度箭头（蓝色）
                if abs(vel_y_arrow) > 0.1:  # 只绘制有意义的箭头
                    ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                            0, vel_y_arrow, head_width=2, head_length=1, 
                            fc='green', ec='green', alpha=0.8, zorder=10)
                # 5. 绘制邻居绝对速度箭头（紫色）
                if np.linalg.norm(neighbor_vel_absolute_global) > 0.1:
                    ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                            neighbor_vel_absolute_global[0], neighbor_vel_absolute_global[1], 
                            head_width=3, head_length=2, fc='purple', ec='purple', 
                            alpha=0.9, zorder=11, linewidth=2, label=f'Absolute Vel_{i}' if i == 0 else "")
                # 6. 绘制邻居车辆的矩形（使用复原的长度和宽度）
                # 计算邻居的朝向（从绝对速度向量推断）
                if np.linalg.norm(neighbor_vel_absolute_global) > 0.1:
                    neighbor_yaw = np.arctan2(neighbor_vel_absolute_global[1], neighbor_vel_absolute_global[0])
                else:
                    neighbor_yaw = 0.0  # 如果速度很小，假设朝向为0

                # 绘制邻居车辆矩形
                draw_vehicle(ax, neighbor_pos_global[0], neighbor_pos_global[1], 
                           neighbor_yaw, np.linalg.norm(neighbor_vel_absolute_global), 
                           length, width, color='red', alpha=0.6)    

    else:
        print("没有观测到有效的邻居")

    # 绘制ego矩形
    draw_vehicle(ax, agents_state[0, 0, 0].cpu().numpy(), agents_state[0, 0, 1].cpu().numpy(), agents_state[0, 0, 2].cpu().numpy(), agents_state[0, 0, 3].cpu().numpy())
    # 绘制第二辆车的矩形(真值)
    draw_vehicle(ax, agents_state[0, 1, 0].cpu().numpy(), agents_state[0, 1, 1].cpu().numpy(), agents_state[0, 1, 2].cpu().numpy(), agents_state[0, 1, 3].cpu().numpy(), color='blue',alpha=0.5)

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

    # 只显示一次图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # 计算Frenet坐标
    vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
    vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
    vehicle_yaw_tensor = torch.tensor([vehicle_yaw], dtype=torch.float32, device=device)
    d, theta_f = road_network.calculate_frenet_coordinates(vehicle_pos_tensor, vehicle_yaw_tensor)
    print(f"横向距离 d: {d.item():.2f} (正值表示在道路右侧，负值表示在道路左侧)")
    print(f"角度误差 theta_f: {theta_f.item():.2f} 弧度 ({np.degrees(theta_f.item()):.1f} 度)")
    print("角度误差解释: 正值表示车辆朝向偏右，负值表示偏左")

    # 设置图形属性
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('RoadNetwork Test - Map Visualization and Frenet Coordinate Calculation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 保存图片
    plt.savefig('road_network_test.png', dpi=300, bbox_inches='tight')
    # 显示图形
    plt.show()

except FileNotFoundError:
    print(f"错误: 找不到地图文件 {map_path}")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()

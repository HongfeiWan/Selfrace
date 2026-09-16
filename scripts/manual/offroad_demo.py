"""Manual diagnostics moved from simulator/offroad.py."""

from _bootstrap import add_project_paths

add_project_paths()

import matplotlib.pyplot as plt
import random
import numpy as np
import torch

from offroad import OffroadChecker
from road import RoadNetwork
from spatial_hash import SpatialHash
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
    print("车辆周围最近200个quads")

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
    print("📏 车辆尺寸: 4.5m × 2.0m")
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
        print("  角度误差解释: 正值表示车辆朝向偏右，负值表示偏左")

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

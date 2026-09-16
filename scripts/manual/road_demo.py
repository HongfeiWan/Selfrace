"""Manual diagnostics moved from simulator/road.py."""

from _bootstrap import add_project_paths

add_project_paths()

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from road import RoadNetwork
from spatial_hash import SpatialHash
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
    # 创建空间哈希用于加速查询
    print("创建空间哈希网格...")


    # 计算所有quad的边界框
    quad_centers = road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
    quad_min_bounds = torch.min(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
    quad_max_bounds = torch.max(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
    # 计算整个地图的边界
    map_min_bounds = torch.min(quad_min_bounds, dim=0)[0]  # (2,)
    map_max_bounds = torch.max(quad_max_bounds, dim=0)[0]  # (2,)

    # 设置合适的网格大小（根据quad的平均大小）
    cell_size = torch.tensor(5.0)

    # 初始化空间哈希
    spatial_hash = SpatialHash(
        cell_size=cell_size,
        min_bounds=map_min_bounds,
        max_bounds=map_max_bounds,
        device=device
    )

    # 构建静态索引
    quad_bounds = torch.stack([quad_min_bounds, quad_max_bounds], dim=1)  # (num_quads, 2, 2)
    spatial_hash.build_static_index(quad_bounds)
    print(f"空间哈希网格创建完成，网格大小: {cell_size:.2f}m")

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

    # 使用空间哈希加速查询，找到距离车辆最近的200个quads
    print("使用空间哈希查询最近quads...")
    distances, nearest_indices = road_network.find_nearest_lanes(vehicle_pos_tensor, k=200, spatial_hash=spatial_hash)
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
    d, theta_f = road_network.calculate_frenet_coordinates(vehicle_pos_tensor, vehicle_yaw_tensor, spatial_hash=spatial_hash)
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

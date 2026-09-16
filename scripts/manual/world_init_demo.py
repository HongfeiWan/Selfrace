"""Manual diagnostics moved from simulator/world_init.py."""

from _bootstrap import PROJECT_ROOT, add_project_paths

add_project_paths()

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from collision import CollisionChecker
from offroad import OffroadChecker
from road import RoadNetwork
from spatial_hash import SpatialHash
from world_init import WorldInitializer

# 定义可视化函数
def plot_quads(ax, quads_data):
    """
    在地图上绘制道路四边形
    """
    if not quads_data:
        print("No quad data to plot.")
        return
    patches = []
    road_ids = []
    has_road_ids = 'road_id' in quads_data[0]

    for quad_info in quads_data:
        vertices = np.array([[point['x'], point['y']] for point in quad_info['vertices']])
        polygon = Polygon(vertices, closed=True)
        patches.append(polygon)
        if has_road_ids:
            road_ids.append(quad_info.get('road_id'))

    if has_road_ids and len(set(road_ids)) > 1:
        unique_road_ids = sorted(list(set(road_ids)))
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(unique_road_ids), vmax=max(unique_road_ids))
        colors = []
        for rid in road_ids:
            if rid != 99999999:
                colors.append(cmap(norm(rid)))
            else:
                colors.append('red')
        p = PatchCollection(patches, alpha=0.3, facecolors=colors, edgecolor='black', linewidth=0.1)
    else:
        p = PatchCollection(patches, alpha=0.1, facecolor='gray', edgecolor='black', linewidth=0.1)
    ax.add_collection(p)

def plot_traffic_controls(ax, traffic_data):
    """在地图上绘制交通信号灯和停止线"""
    if not traffic_data:
        print("No traffic control data to plot.")
        return

    light_locs_x = []
    light_locs_y = []
    STOP_LINE_WIDTH = 3.5 

    for i, control_info in enumerate(traffic_data):
        loc = control_info['traffic_light_location']
        light_locs_x.append(loc['x'])
        light_locs_y.append(loc['y'])

        for waypoint in control_info['stop_line_waypoints']:
            wp_loc = waypoint['location']
            wp_yaw_deg = waypoint['rotation']['yaw']

            rad_yaw = math.radians(wp_yaw_deg)

            perp_dx = math.sin(rad_yaw)
            perp_dy = math.cos(rad_yaw)

            half_width = STOP_LINE_WIDTH / 2.0
            p1_x = wp_loc['x'] - perp_dx * half_width
            p1_y_carla = wp_loc['y'] - perp_dy * half_width
            p2_x = wp_loc['x'] + perp_dx * half_width
            p2_y_carla = wp_loc['y'] + perp_dy * half_width

            label = 'Stop Line' if i == 0 else ""
            ax.plot([p1_x, p2_x], [p1_y_carla, p2_y_carla], color='red', linewidth=2.5, solid_capstyle='round', label=label, zorder=3)

    ax.scatter(light_locs_x, light_locs_y, c='red', s=50, marker='o', label='Traffic Light', zorder=3)

def plot_vehicles(ax, agents_state, ego_agents_idx, env_idx=0):
    """
    在地图上绘制车辆

    Args:
        ax: matplotlib轴对象
        agents_state: 车辆状态张量 (num_envs, max_agents, 7)
        ego_agents_idx: 主车索引张量 (num_envs,)
        env_idx: 要可视化的环境索引
    """
    if agents_state is None:
        print("No vehicle data to plot.")
        return

    # 获取指定环境的车辆状态
    env_states = agents_state[env_idx]  # (max_agents, 7)

    # 只绘制激活的车辆 (active = 1.0)
    active_mask = env_states[:, 6] == 1.0
    active_states = env_states[active_mask]

    if len(active_states) == 0:
        print("No active vehicles to plot.")
        return

    print(f"Plotting {len(active_states)} active vehicles for environment {env_idx}")

    # 定义不同agent的颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 获取激活的智能体索引（用于确定颜色）
    active_agents = torch.where(active_mask)[0]

    for i, state in enumerate(active_states):
        x, y, yaw, speed, length, width, active = state.cpu().numpy()

        # 创建车辆矩形
        # 车辆中心在(x, y)，需要根据yaw旋转
        cos_yaw_plot = math.cos(yaw)
        sin_yaw_plot = math.sin(yaw)

        # 车辆矩形的四个角点 (相对于中心)
        half_length = length / 2.0
        half_width = width / 2.0

        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_yaw_plot, -sin_yaw_plot],
            [sin_yaw_plot, cos_yaw_plot]
        ])

        # 旋转角点
        rotated_corners = corners @ rotation_matrix.T

        # 平移到车辆位置
        vehicle_corners = rotated_corners + np.array([x, y])

        # 创建矩形多边形
        vehicle_polygon = Polygon(vehicle_corners, closed=True)

        # 选择颜色
        agent_idx = active_agents[i].item()
        color = colors[i % len(colors)]
        alpha = 0.8
        label = f'Agent {agent_idx}' if i == 0 else ""

        # 添加车辆到图上
        ax.add_patch(vehicle_polygon)
        vehicle_polygon.set_facecolor(color)
        vehicle_polygon.set_alpha(alpha)
        vehicle_polygon.set_edgecolor('black')
        vehicle_polygon.set_linewidth(1)

        # 添加标签
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # 绘制速度向量
        speed_vector_length = 5.0  # 速度向量的显示长度
        speed_dx = speed_vector_length * cos_yaw_plot
        speed_dy = speed_vector_length * sin_yaw_plot

        ax.arrow(x, y, speed_dx, speed_dy, head_width=1.0, head_length=1.0, 
                fc=color, ec=color, alpha=0.8, zorder=5)

def visualize_vehicles_on_map(unified_data_path, agents_state, ego_agents_idx, env_idx=0):
    """
    在地图上可视化车辆状态

    Args:
        unified_data_path: 地图数据文件路径
        agents_state: 车辆状态张量
        ego_agents_idx: 主车索引张量
        env_idx: 要可视化的环境索引
    """
    if not os.path.exists(unified_data_path):
        print(f"Error: Unified map data file not found at '{unified_data_path}'")
        return

    with open(unified_data_path, 'r') as f:
        data = json.load(f)

    quads_data = data.get('quads', [])
    traffic_data = data.get('traffic_controls', [])
    map_name = data.get('map_name', 'Unknown')

    fig, ax = plt.subplots(figsize=(20, 20))

    # 绘制地图
    plot_quads(ax, quads_data)
    plot_traffic_controls(ax, traffic_data)

    # 绘制车辆
    plot_vehicles(ax, agents_state, ego_agents_idx, env_idx)

    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    title = f'Vehicle Visualization on {map_name} (Environment {env_idx})'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.grid(True, alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Agent 0'),
        Patch(facecolor='blue', alpha=0.6, label='NPC Vehicle'),
        Line2D([0], [0], color='red', linewidth=2.5, label='Stop Line'),
        Line2D([0], [0], marker='o', color='red', label='Traffic Light', markersize=8)
    ]

    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

# 添加utils目录到路径
utils_dir = os.path.join(PROJECT_ROOT, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
import yaml

# --- 测试设置 ---
# 基于文件位置解析项目根目录
config_path = PROJECT_ROOT / 'configs' / 'default_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
map_file_path = config['simulator']['map_path']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_config = config['simulator']

# 1. 实例化依赖项 RoadNetwork
try:
    road_network = RoadNetwork(map_path=map_file_path, device=device)
except FileNotFoundError:
    print("Error: Map file not found. Make sure the path is correct.")
    print("Please run this test from the root directory of the project.")
    exit()

# 2. 实例化 OffroadChecker, CollisionChecker, SpatialHash
all_verts = road_network.quads_vertices.view(-1, 2)
min_bounds, _ = torch.min(all_verts, dim=0)
max_bounds, _ = torch.max(all_verts, dim=0)
spatial_hash = SpatialHash(
    cell_size=test_config['hash']['hash_cell_size'],
    min_bounds=min_bounds,
    max_bounds=max_bounds,
    device=device
)
offroad_checker = OffroadChecker(road_network, spatial_hash)
collision_checker = CollisionChecker(test_config, spatial_hash)
print("Dependencies instantiated successfully.")

# 3. 实例化 WorldInitializer
initializer = WorldInitializer(road_network, offroad_checker, collision_checker, test_config)
print("WorldInitializer instantiated successfully.")

# 4. 调用 initialize_world
num_test_envs = 2400
agents_state, ego_idx, agents_start_quad_ids = initializer.initialize_world(num_envs=num_test_envs)

# 5. 打印结果进行验证
print("\n--- Initialization Results ---")
print(f"Agents state tensor shape: {agents_state.shape}")
print(f"Ego indices tensor shape: {ego_idx.shape}")

# 检查第一个环境 (env_idx = 0)
env_idx = 0
print(f"\n--- Details for Environment {env_idx} ---")
print(f"Ego agent index: {ego_idx[env_idx].item()}")

active_agents_mask = agents_state[env_idx, :, 6] == 1.0
num_active = active_agents_mask.sum().item()
print(f"Number of active agents: {int(num_active)}")

# 验证没有初始碰撞
initial_collisions = collision_checker.check(agents_state, agents_state)
assert not initial_collisions.any(), "Error: Initial collisions detected!"
print("PASSED: No initial collisions detected.")
ego_state = agents_state[env_idx, ego_idx[env_idx]]
print("Ego state (x, y, yaw, v, l, w, active):")
print(f"  {ego_state.cpu().numpy()}")

# 6. 在地图上可视化车辆
print("\n--- Visualizing vehicles on map ---")
try:
    # 使用新添加的可视化函数
    visualize_vehicles_on_map(map_file_path, agents_state, ego_idx, env_idx=0)

    # 如果有多个环境，也可以可视化其他环境
    if num_test_envs > 1:
        print("\nVisualizing environment 1...")
        visualize_vehicles_on_map(map_file_path, agents_state, ego_idx, env_idx=1)

except Exception as e:
    print(f"Error during visualization: {e}")
    print("Vehicle visualization failed.")

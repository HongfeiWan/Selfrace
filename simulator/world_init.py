import torch
from typing import Dict, Tuple
import logging
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import math
from matplotlib.lines import Line2D
import time

# 添加simulator目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
from road import RoadNetwork
from offroad import OffroadChecker
from collision import CollisionChecker

class WorldInitializer:
    """
    负责在模拟开始时初始化一批（batch）世界状态。
    遵循论文中的核心思想，通过迭代和拒绝采样，确保生成的初始交通流
    是无碰撞且在道路上的，从而为训练提供高质量的初始场景。
    """
    def __init__(self, road_network: RoadNetwork, offroad_checker: OffroadChecker, collision_checker: CollisionChecker, config: Dict):
        """
        初始化世界状态生成器。
        Args:
            road_network (RoadNetwork): 已实例化的道路网络对象。
            offroad_checker (OffroadChecker): 离路检测器实例。
            collision_checker (CollisionChecker): 碰撞检测器实例。
            config (Dict): 包含初始化相关参数的配置字典。
        """
        self.road_network = road_network
        self.offroad_checker = offroad_checker
        self.collision_checker = collision_checker
        self.device = road_network.device
        self.config = config
        # 获取simulator配置，支持嵌套配置结构
        simulator_config = config.get('simulator', config)
        self.max_agents = simulator_config.get('max_agents_num')
        self.num_agents_per_env = simulator_config.get('num_npc_vehicles')
        if self.num_agents_per_env > self.max_agents:
            raise ValueError("num_npc_vehicles exceeds max_agents_num")
        self.vehicle_length = simulator_config.get('vehicle_length', 4.5)
        self.vehicle_width = simulator_config.get('vehicle_width', 2.0)
        self.speed_range = simulator_config.get('vehicle_init_speed_range', (0.0, 5.0))
        self.local_state_dim = simulator_config.get('observation', simulator_config).get('local_state_dim')

    def _generate_states_on_quads(self, quad_indices: torch.Tensor) -> torch.Tensor:
        """
        在指定的道路四边形上生成车辆状态。
        Args:
            quad_indices: 形状为 (num_vehicles,) 的四边形索引张量
        Returns:
            形状为 (num_vehicles, 7) 的车辆状态张量
        """
        num_vehicles = len(quad_indices)
        if num_vehicles == 0:
            return torch.empty(0, 7, device=self.device)

        centerlines = self.road_network.quad_centerlines[quad_indices]
        # 在中心线上随机选择一个点
        t = torch.rand(num_vehicles, 1, 1, device=self.device)
        positions = centerlines[:, 0:1, :] + t * (centerlines[:, 1:2, :] - centerlines[:, 0:1, :])
        positions = positions.squeeze(1)
        
        centerline_vecs = centerlines[:, 1, :] - centerlines[:, 0, :]
        yaws = torch.atan2(centerline_vecs[:, 1], centerline_vecs[:, 0])
    
        speeds = (self.speed_range[0] + 
                  (self.speed_range[1] - self.speed_range[0]) * torch.rand(num_vehicles, device=self.device))
    
        new_states = torch.zeros(num_vehicles, 7, device=self.device)
        new_states[:, :2] = positions
        new_states[:, 2] = yaws
        new_states[:, 3] = speeds
        new_states[:, 4] = self.vehicle_length
        new_states[:, 5] = self.vehicle_width
        new_states[:, 6] = 1.0
        return new_states

    def initialize_world(self, num_envs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成一批新的、无碰撞的世界状态。
        采用并行生成和迭代优化的策略，确保初始化的交通流是有效的。
        """
        # 存储每个智能体的起始quad_id
        agents_state = torch.zeros(num_envs, self.max_agents, self.local_state_dim, device=self.device)
        agents_start_quad_ids = torch.full((num_envs, self.max_agents), -1, dtype=torch.long, device=self.device)
        start_time = time.time()
        max_retries = 1
        for retry in range(max_retries):
            # 1. 并行生成所有环境的所有agent slot的候选状态
            gen_start = time.time()
            total_agents = num_envs * self.num_agents_per_env
            # 为所有agent生成四边形索引
            spawn_quad_indices = torch.randint(0, self.road_network.num_quads, (total_agents,), device=self.device)
            # 生成所有候选状态
            all_candidate_states = self._generate_states_on_quads(spawn_quad_indices)
            # 重塑为 (num_envs, num_agents_per_env, 7)
            candidate_states = all_candidate_states.view(num_envs, self.num_agents_per_env, 7)
            gen_time = time.time() - gen_start
            
            # 2. 将候选状态放入agents_state张量
            agents_state[:, :self.num_agents_per_env] = candidate_states
            
            # 3. 并行检查所有agent的有效性
            check_start = time.time()
            # a) 离路检查 - 检查所有agent
            offroad_start = time.time()
            all_states_to_check = agents_state[:, :self.num_agents_per_env]  # (num_envs, num_agents_per_env, 7)
            offroad_states_for_checker = all_states_to_check[:, :, [0, 1, 2, 4, 5]]  # (num_envs, num_agents_per_env, 5)
            # 重塑为 (num_envs * num_agents_per_env, 5) 以便批量检查
            offroad_states_flat = offroad_states_for_checker.view(-1, 5)
            is_on_road_flat = self.offroad_checker.check_on_road(offroad_states_flat)
            offroad_time = time.time() - offroad_start
            is_on_road = is_on_road_flat.view(num_envs, self.num_agents_per_env)  # (num_envs, num_agents_per_env)
            
            # b) 碰撞检查 - 检查所有agent之间的碰撞
            collision_start = time.time()
            collisions = self.collision_checker.check(agents_state, agents_state)  # (num_envs, max_agents)
            collision_time = time.time() - collision_start
            collision_mask = collisions[:, :self.num_agents_per_env]  # (num_envs, num_agents_per_env)
            check_time = time.time() - check_start
            
            # 4. 确定哪些放置是无效的
            invalid_placement_mask = ~is_on_road | collision_mask  # (num_envs, num_agents_per_env)
            
            # 5. 如果所有放置都有效，则完成初始化
            if not invalid_placement_mask.any():
                # 记录有效的quad_id
                total_agents = num_envs * self.num_agents_per_env
                spawn_quad_indices = torch.randint(
                    0, self.road_network.num_quads, (total_agents,), device=self.device
                ).view(num_envs, self.num_agents_per_env)
                agents_start_quad_ids[:, :self.num_agents_per_env] = spawn_quad_indices

                if retry > 0:
                    logging.debug(f"All agents placed successfully after {retry+1} retries.")
                break
            
            # 6. 对于无效的放置，将对应的agent标记为不激活
            # 直接使用切片索引来更新agents_state
            agents_state[:, :self.num_agents_per_env, 6][invalid_placement_mask] = 0.0
            
            # 7. 记录有效的quad_id（对于有效的放置）
            valid_placement_mask = ~invalid_placement_mask
            if valid_placement_mask.any():
                # 为有效的放置记录quad_id
                total_agents = num_envs * self.num_agents_per_env
                spawn_quad_indices = torch.randint(
                    0, self.road_network.num_quads, (total_agents,), device=self.device
                ).view(num_envs, self.num_agents_per_env)
                # 使用布尔索引来更新有效的quad_id
                agents_start_quad_ids[valid_placement_mask] = spawn_quad_indices[valid_placement_mask]
            if retry == 0:  # 只在第一次迭代时打印性能信息
                print(f"Retry {retry}: gen_time={gen_time:.4f}s, check_time={check_time:.4f}s, offroad_time={offroad_time:.4f}s, collision_time={collision_time:.4f}s")
        end_time = time.time()
        print(f"World initialization time: {end_time - start_time} seconds")
        ego_agents_idx = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        logging.info("World initialization complete.")
        return agents_state, ego_agents_idx, agents_start_quad_ids

if __name__ == '__main__':
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
        
        # 获取主车索引
        ego_idx = ego_agents_idx[env_idx].item()
        
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
    utils_dir = os.path.join(parent_dir, 'utils')
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from spatial_hash import SpatialHash
    import yaml

    # --- 测试设置 ---
    config_path = 'configs/default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    map_file_path = config['simulator']['map_path']
    device = config['simulator']['device']
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
    print(f"Ego state (x, y, yaw, v, l, w, active):")
    print(f"  {ego_state.cpu().numpy()}")

    # 6. 在地图上可视化车辆
    print("\n--- Visualizing vehicles on map ---")
    try:
        # 使用新添加的可视化函数
        visualize_vehicles_on_map(map_file_path, agents_state, ego_idx, env_idx=0)
        
        # 如果有多个环境，也可以可视化其他环境
        if num_test_envs > 1:
            print(f"\nVisualizing environment 1...")
            visualize_vehicles_on_map(map_file_path, agents_state, ego_idx, env_idx=1)
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Vehicle visualization failed.")




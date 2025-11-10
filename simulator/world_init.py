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

from road import RoadNetwork
from offroad import OffroadChecker
from collision import CollisionChecker
from randomize import VehicleParameterSampler

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
        # INVALID 标记（从配置读取并作为 int，用于索引类张量）
        self.INVALID_MARKER = -1
        # 速度范围由 dynamics 的 min_velocity/max_velocity 推导
        dyn_cfg = simulator_config.get('dynamics', {}) if isinstance(simulator_config.get('dynamics', {}), dict) else {}
        min_velocity = float(dyn_cfg.get('min_velocity', -2.0))
        max_velocity = float(dyn_cfg.get('max_velocity', 20.0))
        self.speed_range = (min_velocity, max_velocity)
        obs_cfg = simulator_config.get('observation', {}) if isinstance(simulator_config.get('observation', {}), dict) else {}
        # 状态维度默认7: [x, y, yaw, v, length, width, active]
        self.local_state_dim = int(obs_cfg.get('local_state_dim'))
        self.neighbor_feature_dim = int(obs_cfg.get('neighbor_feature_dim'))
        # 读取批大小（B, M），供初始化世界使用
        self.B = int(simulator_config.get('B'))
        self.M = int(simulator_config.get('M'))
        # 车辆参数（episode 采样），在 initialize_world 中填充
        self.vehicle_params = {}

        # 提前初始化世界，并保存车辆参数供其他模块使用
        init_state, start_quad_ids, goal_quad_ids = self.initialize_world(self.B, self.M)
        self.initial_agents_state = init_state
        self.initial_agents_start_quad_ids = start_quad_ids
        self.initial_agents_goal_quad_ids = goal_quad_ids

    def _generate_states_on_quads(self, quad_indices: torch.Tensor, lengths: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """
        在指定的道路四边形上生成车辆状态。
        Args:
            quad_indices: 形状为 (num_vehicles,) 的四边形索引张量
            lengths: 形状为 (num_vehicles,) 的车辆长度张量
            widths: 形状为 (num_vehicles,) 的车辆宽度张量
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
        # 使用采样的长度与宽度
        new_states[:, 4] = lengths
        new_states[:, 5] = widths
        new_states[:, 6] = 1
        return new_states

    def initialize_world(self, B: int, M: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成一批新的、无碰撞的世界状态。
        采用并行生成和迭代优化的策略，确保初始化的交通流是有效的。
        """
        # 存储每个智能体的起始quad_id
        agents_state = torch.zeros(B, M, self.local_state_dim, device=self.device)
        agents_start_quad_ids = torch.full((B, M), self.INVALID_MARKER, dtype=torch.long, device=self.device)
        # 存储每个智能体的目标quad_id（新增）
        agents_goal_quad_ids = torch.full((B, M), self.INVALID_MARKER, dtype=torch.long, device=self.device)
        max_retries = 1
        for retry in range(max_retries):
            # 1. 并行生成所有环境的所有agent slot的候选状态
            total_agents = B * M
            # 为所有agent生成四边形索引
            num_quads = int(self.road_network.quads_vertices.shape[0])
            spawn_quad_indices = torch.randint(0, num_quads, (total_agents,), device=self.device)
            # 采样车辆参数（长度/宽度）
            sampler = VehicleParameterSampler(self.config, self.device)
            self.vehicle_params = sampler.sample_batch_vehicle_parameters(total_agents)
            lengths = self.vehicle_params['length']
            widths = self.vehicle_params['width']
            # 生成所有候选状态
            all_candidate_states = self._generate_states_on_quads(spawn_quad_indices, lengths, widths)

            # 重塑为 (num_envs, num_agents_per_env, 7)
            candidate_states = all_candidate_states.view(B, M, self.neighbor_feature_dim)
            
            # 2. 将候选状态放入agents_state张量
            agents_state[:, :M] = candidate_states
            # 为所有agent生成目标 quad 索引，尽量与起始不同
            num_quads = int(self.road_network.quads_vertices.shape[0])
            goal_quad_indices = torch.randint(0, num_quads, (B*M,), device=self.device)
            # 若与起始相同，做一次简单调整（+1 再取模）
            same_mask = goal_quad_indices == spawn_quad_indices
            if same_mask.any():
                goal_quad_indices[same_mask] = (goal_quad_indices[same_mask] + 1) % max(1, num_quads)
            # 注意：这里暂时赋值索引，后续会被转换为poly_id或设置为INVALID_MARKER
            # 在步骤5和7中会正确转换为poly_id
            agents_goal_quad_ids[:, :M] = goal_quad_indices.view(B, M)
            
            # 3. 并行检查所有agent的有效性
            # a) 离路检查 - 检查所有agent
            all_states_to_check = agents_state[:, :M]  # (B, M, 7)
            offroad_states_for_checker = all_states_to_check[:, :, [0, 1, 2, 4, 5]]  # (B, M, 5)
            # 重塑为 (num_envs * num_agents_per_env, 5) 以便批量检查
            offroad_states_flat = offroad_states_for_checker.view(-1, 5)
            is_on_road_flat = self.offroad_checker.check_on_road(offroad_states_flat)
            is_on_road = is_on_road_flat.view(B, M)  # (B, M)
            
            # b) 碰撞检查 - 检查所有agent之间的碰撞
            collisions = self.collision_checker.check(agents_state, agents_state)  # (B, max_agents)
            collision_mask = collisions[:, :M]  # (B, M)
            
            # 4. 确定哪些放置是无效的
            invalid_placement_mask = ~is_on_road | collision_mask  # (B, M)
            # 对无效的目标quad清空为 INVALID 标记（与 start 成对）
            agents_goal_quad_ids[:, :M][invalid_placement_mask] = self.INVALID_MARKER
            
            # 5. 如果所有放置都有效，则完成初始化
            if not invalid_placement_mask.any():
                # 记录有效的quad_id - 将索引转换为实际的poly_id
                # spawn_quad_indices 是数组索引（0-based），需要转换为 poly_id
                agents_start_quad_ids[:, :M] = self.road_network.quad_ids[spawn_quad_indices].view(B, M)
                # 目标quad也需要转换
                agents_goal_quad_ids[:, :M] = self.road_network.quad_ids[goal_quad_indices].view(B, M)
                if retry > 0:
                    logging.debug(f"All agents placed successfully after {retry+1} retries.")
                break

            # 6. 对于无效的放置，将对应的agent标记为不激活
            # 直接使用切片索引来更新agents_state
            agents_state[:, :M, 6][invalid_placement_mask] = 0
            # 7. 记录有效的quad_id（对于有效的放置）
            valid_placement_mask = ~invalid_placement_mask
            if valid_placement_mask.any():
                # 使用实际生成智能体位置时使用的quad_id
                # 将索引转换为实际的poly_id
                spawn_poly_ids = self.road_network.quad_ids[spawn_quad_indices].view(B, M)
                goal_poly_ids = self.road_network.quad_ids[goal_quad_indices].view(B, M)
                agents_start_quad_ids[:, :M][valid_placement_mask] = spawn_poly_ids[valid_placement_mask]
                agents_goal_quad_ids[:, :M][valid_placement_mask] = goal_poly_ids[valid_placement_mask]
        logging.info("World initialization complete.")
        return agents_state, agents_start_quad_ids, agents_goal_quad_ids

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
        
        def to_xy_array(verts):
            coords = []
            for pt in verts:
                if isinstance(pt, dict):
                    x = pt.get('x', pt.get('X'))
                    y = pt.get('y', pt.get('Y'))
                else:
                    # list/tuple like [x, y]
                    x, y = pt[0], pt[1]
                coords.append([float(x), float(y)])
            return np.array(coords, dtype=float)

        for quad_info in quads_data:
            vertices = to_xy_array(quad_info['vertices'])
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

    def plot_vehicles(ax, agents_state, env_idx=0):
        """
        在地图上绘制车辆
        
        Args:
            ax: matplotlib轴对象
            agents_state: 车辆状态张量 (num_envs, max_agents, 7)
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

    def visualize_vehicles_on_map(unified_data_path, agents_state, env_idx=0):
        """
        在地图上可视化车辆状态
        
        Args:
            unified_data_path: 地图数据文件路径
            agents_state: 车辆状态张量
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
        plot_vehicles(ax, agents_state, env_idx)
        
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
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    utils_dir = os.path.join(_proj_root, 'utils')
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from utils.spatial_hash import SpatialHash

    # --- 测试设置（读取 JSON 配置） ---
    config_path = os.path.join(_proj_root, 'configs', 'default_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)
    # 设备
    device = torch.device(full_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # 地图路径
    maps_dir = full_cfg.get('map_path', './maps')
    default_map = full_cfg.get('default_map', 'town2.json')
    map_file_path = os.path.join(_proj_root, maps_dir, default_map)
    # 模拟配置
    test_config = full_cfg.get('simulator', {})

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
    hash_cfg = test_config.get('hash', {})
    cell_size = hash_cfg.get('cell_size', 20.0)
    spatial_hash = SpatialHash(
        cell_size=cell_size,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        device=device
    )
    offroad_checker = OffroadChecker(road_network, spatial_hash)
    collision_checker = CollisionChecker(full_cfg, spatial_hash)
    print("Dependencies instantiated successfully.")

    # 3. 实例化 WorldInitializer
    initializer = WorldInitializer(road_network, offroad_checker, collision_checker, test_config)
    print("WorldInitializer instantiated successfully.")

    # 4. 调用 initialize_world
    B = test_config.get('B', 2400)
    M = test_config.get('M', 150)
    agents_state, agents_start_quad_ids, agents_goal_quad_ids = initializer.initialize_world(B, M)

    # 5. 打印结果进行验证
    print("\n--- Initialization Results ---")
    print(f"Agents state tensor shape: {agents_state.shape}")
    print(f"Agents start quad ids tensor shape: {agents_start_quad_ids.shape}")
    print(f"Agents goal quad ids tensor shape: {agents_goal_quad_ids.shape}")

    # 检查第一个环境 (env_idx = 0)
    env_idx = 0
    print(f"\n--- Details for Environment {env_idx} ---")
    
    active_agents_mask = agents_state[env_idx, :, 6] == 1.0
    num_active = active_agents_mask.sum().item()
    print(f"Number of active agents: {int(num_active)}")
    
    # 显示第一个激活车辆的状态
    if num_active > 0:
        first_active_idx = torch.where(active_agents_mask)[0][0].item()
        first_agent_state = agents_state[env_idx, first_active_idx]
        print(f"First active agent index: {first_active_idx}")
        print(f"First active agent state (x, y, yaw, v, l, w, active):")
        print(f"  {first_agent_state.cpu().numpy()}")

    # 验证没有初始碰撞
    initial_collisions = collision_checker.check(agents_state, agents_state)
    assert not initial_collisions.any(), "Error: Initial collisions detected!"
    print("PASSED: No initial collisions detected.")

    # 6. 在地图上可视化车辆
    print("\n--- Visualizing vehicles on map ---")
    try:
        # 使用新添加的可视化函数
        visualize_vehicles_on_map(map_file_path, agents_state, env_idx=0)
        
        # 如果有多个环境，也可以可视化其他环境
        if B > 1:
            print(f"\nVisualizing environment 1...")
            visualize_vehicles_on_map(map_file_path, agents_state, env_idx=1)
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Vehicle visualization failed.")




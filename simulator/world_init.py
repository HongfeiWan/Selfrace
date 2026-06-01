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
from randomize_components import VehicleParameterSampler

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
        self.verbose = simulator_config.get('verbose', False)
        self.max_agents = int(simulator_config.get('max_agents_num'))
        self.num_agents_per_env = int(simulator_config.get('num_npc_vehicles'))
        if self.num_agents_per_env > self.max_agents:
            raise ValueError("num_npc_vehicles exceeds max_agents_num")
        dynamics_config = simulator_config.get('dynamics', {})
        self.vehicle_length = dynamics_config.get('vehicle_length', simulator_config.get('vehicle_length', 4.5))
        self.vehicle_width = dynamics_config.get('vehicle_width', simulator_config.get('vehicle_width', 2.0))
        self.vehicle_parameter_sampler = VehicleParameterSampler(simulator_config, self.device)
        self.speed_range = simulator_config.get('vehicle_init_speed_range', (0.0, 5.0))
        # agents_state 是仿真世界状态，固定为 [x,y,yaw,speed,length,width,active]。
        # observation.local_state_dim 现在表示网络观测 S(t) 维度，二者不能混用。
        self.state_dim = int(simulator_config.get('state_dim', 7))
        self.last_agents_per_env = None
        self.init_candidates_per_slot = int(simulator_config.get('init_candidates_per_slot', 16))
        self.init_max_fill_attempts = int(simulator_config.get('init_max_fill_attempts', 4))
        self.init_collision_clearance = float(simulator_config.get('init_collision_clearance', 0.0))
        self.init_candidate_pool_size = int(simulator_config.get('init_candidate_pool_size', 262144))
        self.init_candidate_pool_refill_threshold = int(simulator_config.get('init_candidate_pool_refill_threshold', 8192))
        self.init_candidate_pool_refill_batch = int(simulator_config.get('init_candidate_pool_refill_batch', 65536))
        self.init_distribution_correction = bool(simulator_config.get('init_distribution_correction', True))
        self.init_distribution_calibration_samples = int(simulator_config.get('init_distribution_calibration_samples', 65536))
        self.init_distribution_smoothing = float(simulator_config.get('init_distribution_smoothing', 1.0))
        self.init_distribution_max_weight = float(simulator_config.get('init_distribution_max_weight', 20.0))
        self._spawn_quad_probs = None
        self._candidate_pool_states = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self._candidate_pool_quad_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        self._candidate_pool_cursor = 0
        self._build_spawn_distribution()

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
    
        vehicle_params = self.vehicle_parameter_sampler.sample_batch_vehicle_parameters(num_vehicles)

        new_states = torch.zeros(num_vehicles, 7, device=self.device)
        new_states[:, :2] = positions
        new_states[:, 2] = yaws
        new_states[:, 3] = speeds
        new_states[:, 4] = vehicle_params['length']
        new_states[:, 5] = vehicle_params['width']
        new_states[:, 6] = 1.0
        return new_states

    def _build_spawn_distribution(self):
        """
        Estimate the marginal acceptance rate of the raw initializer and bias the
        proposal by its inverse. With uniform 1m-ish quads as the target, this
        counteracts the tendency for rejection sampling to retain wider/easier
        road sections more often.
        """
        num_quads = int(self.road_network.num_quads)
        if num_quads <= 0:
            raise ValueError("Road network has no quads for world initialization.")

        uniform_probs = torch.full((num_quads,), 1.0 / num_quads, dtype=torch.float32, device=self.device)
        if not self.init_distribution_correction or self.init_distribution_calibration_samples <= 0:
            self._spawn_quad_probs = uniform_probs
            return

        sample_count = max(num_quads, self.init_distribution_calibration_samples)
        quad_indices = torch.randint(0, num_quads, (sample_count,), dtype=torch.long, device=self.device)
        candidate_states = self._generate_states_on_quads(quad_indices).view(1, sample_count, 7)
        accepted = self._candidate_onroad_mask(candidate_states).view(-1)

        attempts_per_quad = torch.bincount(quad_indices, minlength=num_quads).to(torch.float32)
        accepted_per_quad = torch.bincount(quad_indices[accepted], minlength=num_quads).to(torch.float32)
        smoothing = max(self.init_distribution_smoothing, 1e-6)
        acceptance_rate = (accepted_per_quad + smoothing) / (attempts_per_quad + smoothing)
        correction = torch.reciprocal(acceptance_rate.clamp_min(1e-6))
        correction = torch.clamp(correction, max=max(self.init_distribution_max_weight, 1.0))

        probs = correction / correction.sum().clamp_min(1e-12)
        self._spawn_quad_probs = torch.where(torch.isfinite(probs), probs, uniform_probs)

    def _sample_quad_indices(self, count: int) -> torch.Tensor:
        if self._spawn_quad_probs is None:
            self._build_spawn_distribution()
        return torch.multinomial(self._spawn_quad_probs, count, replacement=True)

    def _sample_candidate_batch(self, num_envs: int, candidates_per_env: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw candidates from the reusable on-road candidate pool."""
        total_candidates = num_envs * candidates_per_env
        candidate_states, candidate_quads = self._draw_from_candidate_pool(total_candidates)
        candidate_states = candidate_states.view(num_envs, candidates_per_env, 7)
        candidate_quads = candidate_quads.view(num_envs, candidates_per_env)
        return candidate_states, candidate_quads

    def _generate_valid_candidate_chunk(self, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            return (
                torch.empty((0, 7), dtype=torch.float32, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        quad_indices = self._sample_quad_indices(count)
        states = self._generate_states_on_quads(quad_indices)
        valid = self._candidate_onroad_mask(states.view(1, count, 7)).view(-1)
        return states[valid], quad_indices[valid]

    def _available_pool_count(self) -> int:
        return max(0, int(self._candidate_pool_states.shape[0]) - int(self._candidate_pool_cursor))

    def _ensure_candidate_pool(self, min_available: int):
        min_available = max(0, int(min_available))
        if self._available_pool_count() >= min_available:
            return

        remaining_states = self._candidate_pool_states[self._candidate_pool_cursor:]
        remaining_quads = self._candidate_pool_quad_ids[self._candidate_pool_cursor:]
        pieces_states = [remaining_states] if remaining_states.numel() > 0 else []
        pieces_quads = [remaining_quads] if remaining_quads.numel() > 0 else []
        total = int(remaining_states.shape[0])
        target = max(min_available, self.init_candidate_pool_size)
        refill_batch = max(1, self.init_candidate_pool_refill_batch)

        attempts = 0
        max_attempts = max(8, int(math.ceil(target / refill_batch)) * 8)
        while total < target and attempts < max_attempts:
            states, quads = self._generate_valid_candidate_chunk(refill_batch)
            if states.shape[0] > 0:
                pieces_states.append(states)
                pieces_quads.append(quads)
                total += int(states.shape[0])
            attempts += 1

        if total < min_available:
            raise RuntimeError(
                f"Unable to refill initialization candidate pool: need {min_available}, got {total} "
                f"after {attempts} refill attempts."
            )

        pool_states = torch.cat(pieces_states, dim=0) if pieces_states else torch.empty((0, 7), dtype=torch.float32, device=self.device)
        pool_quads = torch.cat(pieces_quads, dim=0) if pieces_quads else torch.empty((0,), dtype=torch.long, device=self.device)
        keep = min(int(pool_states.shape[0]), target)
        self._candidate_pool_states = pool_states[:keep].contiguous()
        self._candidate_pool_quad_ids = pool_quads[:keep].contiguous()
        self._candidate_pool_cursor = 0

    def _draw_from_candidate_pool(self, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        count = int(count)
        self._ensure_candidate_pool(count)
        start = self._candidate_pool_cursor
        end = start + count
        states = self._candidate_pool_states[start:end]
        quads = self._candidate_pool_quad_ids[start:end]
        self._candidate_pool_cursor = end
        if self._available_pool_count() < self.init_candidate_pool_refill_threshold:
            self._ensure_candidate_pool(max(self.init_candidate_pool_refill_threshold, 0))
        return states, quads

    def _candidate_onroad_mask(self, candidate_states: torch.Tensor) -> torch.Tensor:
        B, K, _ = candidate_states.shape
        states_for_checker = candidate_states[..., [0, 1, 2, 4, 5]].reshape(B * K, 5)
        return self.offroad_checker.check_on_road(states_for_checker).view(B, K)

    @staticmethod
    def _unit_axes(yaw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        axis_x = torch.stack([cos_yaw, sin_yaw], dim=-1)
        axis_y = torch.stack([-sin_yaw, cos_yaw], dim=-1)
        return axis_x, axis_y

    def _candidate_collision_mask(
        self,
        candidate_states: torch.Tensor,
        placed_states: torch.Tensor,
        placed_active: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return (B, K) mask for candidates overlapping any already placed vehicle.
        Uses vectorized oriented-box SAT against the current partial initialization.
        """
        B, K, _ = candidate_states.shape
        M = placed_states.shape[1]
        if M == 0 or not bool(placed_active.any().item()):
            return torch.zeros((B, K), dtype=torch.bool, device=self.device)

        cand_center = candidate_states[..., :2]              # (B,K,2)
        placed_center = placed_states[..., :2]               # (B,M,2)
        cand_x, cand_y = self._unit_axes(candidate_states[..., 2])
        placed_x, placed_y = self._unit_axes(placed_states[..., 2])

        clearance = self.init_collision_clearance
        cand_hl = 0.5 * (candidate_states[..., 4] + clearance)
        cand_hw = 0.5 * (candidate_states[..., 5] + clearance)
        placed_hl = 0.5 * (placed_states[..., 4] + clearance)
        placed_hw = 0.5 * (placed_states[..., 5] + clearance)

        delta = placed_center.unsqueeze(1) - cand_center.unsqueeze(2)  # (B,K,M,2)

        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a * b).sum(dim=-1)

        cand_x_b = cand_x.unsqueeze(2)
        cand_y_b = cand_y.unsqueeze(2)
        placed_x_b = placed_x.unsqueeze(1)
        placed_y_b = placed_y.unsqueeze(1)

        abs_px_cx = dot(placed_x_b, cand_x_b).abs()
        abs_py_cx = dot(placed_y_b, cand_x_b).abs()
        abs_px_cy = dot(placed_x_b, cand_y_b).abs()
        abs_py_cy = dot(placed_y_b, cand_y_b).abs()

        cand_hl_b = cand_hl.unsqueeze(2)
        cand_hw_b = cand_hw.unsqueeze(2)
        placed_hl_b = placed_hl.unsqueeze(1)
        placed_hw_b = placed_hw.unsqueeze(1)

        sep_cx = dot(delta, cand_x_b).abs() > (
            cand_hl_b + placed_hl_b * abs_px_cx + placed_hw_b * abs_py_cx
        )
        sep_cy = dot(delta, cand_y_b).abs() > (
            cand_hw_b + placed_hl_b * abs_px_cy + placed_hw_b * abs_py_cy
        )
        sep_px = dot(delta, placed_x_b).abs() > (
            placed_hl_b + cand_hl_b * abs_px_cx + cand_hw_b * abs_px_cy
        )
        sep_py = dot(delta, placed_y_b).abs() > (
            placed_hw_b + cand_hl_b * abs_py_cx + cand_hw_b * abs_py_cy
        )

        overlap = ~(sep_cx | sep_cy | sep_px | sep_py)
        overlap = overlap & placed_active.unsqueeze(1)
        return overlap.any(dim=2)

    def _sequential_collision_free_fill(
        self,
        agents_state: torch.Tensor,
        agents_start_quad_ids: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fill each world slot-by-slot from large candidate pools.
        For each target slot, all worlds draw K candidates in parallel and keep the
        first candidate that is on-road and collision-free against already placed cars.
        """
        num_envs = agents_state.shape[0]
        filled_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        candidates_per_slot = max(1, self.init_candidates_per_slot)
        max_attempts = max(1, self.init_max_fill_attempts)

        for slot_idx in range(self.num_agents_per_env):
            needs_slot = filled_counts < target_counts
            if not bool(needs_slot.any().item()):
                break

            placed_active = agents_state[..., 6] > 0.5
            slot_filled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            for _ in range(max_attempts):
                still_needs = needs_slot & (~slot_filled)
                if not bool(still_needs.any().item()):
                    break

                candidate_states, candidate_quads = self._sample_candidate_batch(num_envs, candidates_per_slot)
                collides = self._candidate_collision_mask(candidate_states, agents_state, placed_active)
                valid_candidates = (~collides) & still_needs.unsqueeze(1)

                has_choice = valid_candidates.any(dim=1)
                if not bool(has_choice.any().item()):
                    continue

                choice_idx = valid_candidates.to(torch.int32).argmax(dim=1)
                rows = torch.where(has_choice)[0]
                chosen_states = candidate_states[rows, choice_idx[rows]]
                chosen_quads = candidate_quads[rows, choice_idx[rows]]

                agents_state[rows, slot_idx, :7] = chosen_states
                agents_start_quad_ids[rows, slot_idx] = chosen_quads.to(agents_start_quad_ids.dtype)
                placed_active[rows, slot_idx] = True
                slot_filled[rows] = True

            filled_counts += slot_filled.long()

        return filled_counts

    def initialize_world(self, num_envs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成一批新的、无碰撞的世界状态。
        使用大候选池 + sequential rejection，尽量为每个世界填满目标车辆数。
        """
        agents_state = torch.zeros(num_envs, self.max_agents, self.state_dim, device=self.device)
        agents_start_quad_ids = torch.full((num_envs, self.max_agents), -1, dtype=torch.long, device=self.device)
        start_time = time.time()
        per_env_counts = torch.randint(
            1,
            self.num_agents_per_env + 1,
            (num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.last_agents_per_env = per_env_counts
        filled_counts = self._sequential_collision_free_fill(
            agents_state,
            agents_start_quad_ids,
            per_env_counts,
        )
        end_time = time.time()
        if self.verbose:
            requested = int(per_env_counts.sum().item())
            filled = int(filled_counts.sum().item())
            min_fill = int(filled_counts.min().item()) if filled_counts.numel() else 0
            print(
                f"World initialization time: {end_time - start_time:.4f}s, "
                f"filled={filled}/{requested}, min_env_fill={min_fill}, "
                f"candidates_per_slot={self.init_candidates_per_slot}, attempts={self.init_max_fill_attempts}"
            )
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
    # 基于文件位置解析项目根目录
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    config_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
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

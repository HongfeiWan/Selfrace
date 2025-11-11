import torch
import os
import sys
import time
import math
from typing import Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.spatial_hash import SpatialHash
from road import RoadNetwork
from offroad import OffroadChecker
from collision import CollisionChecker
from goal import PathPlanner
from reward import RewardCalculator
from world_init import WorldInitializer
from observation import ObservationGenerator
from dynamics import KinematicBicycleModel
from utils.geometry_utils import find_nearest_lanes

# 在 __init__ 方法中:
class TeraflowSimulator:
    """
    TeraFlow 模拟器核心类。
    负责管理和步进一个批次 (batch) 的交通模拟环境。
    核心思想：批量化、可微分（未来）以及与自博弈循环的兼容性。
    """
    def __init__(self, config:Dict, device: torch.device):
        """
        初始化模拟器。
        """
        self.config = config
        self.device = device
        simulator_config = config.get('simulator')
        # 批次数：默认沿用配置中的 B（环境批大小），若存在 num_envs 则优先使用
        self.num_envs = simulator_config.get('B')
        self.max_agents = simulator_config.get('M')
        self.dt = simulator_config['sim_dt']
        self.path_length = 10 #这个用于控制初始可以看到的路径长度，可以后续随着avg_reward增加而增加

        # 组合根配置中的地图目录与默认地图名
        maps_dir = config.get('map_path', './maps')
        default_map = config.get('default_map', 'town2.json')
        self.map_path = os.path.join(os.path.dirname(__file__), '..', maps_dir, default_map)

        # 1. 加载地图网络
        # road.py 中的 RoadNetwork 类负责解析地图文件并提供查询接口
        self.road_network = RoadNetwork(self.map_path, self.device)

        # 2. 初始化共享的空间哈希
        all_verts = self.road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 使用一个固定的 cell_size, 也可以从 config 读取
        hash_config = simulator_config['hash']
        cell_size = hash_config.get('cell_size')

        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)

        # 3. 初始化离路检测器 (传入共享的哈希对象)
        self.offroad_checker = OffroadChecker(self.road_network, self.spatial_hash)

        # 4. 初始化碰撞检测器 (传入共享的哈希对象)
        # collision.py 负责检测智能体之间的碰撞
        self.collision_checker = CollisionChecker(config, self.spatial_hash)

        # 5. 初始化世界状态管理器（为动力学提供车辆初始化参数）
        # world_init.py 负责在 reset 时初始化场景
        self.world_initializer = WorldInitializer(self.road_network, self.offroad_checker, self.collision_checker, config)

        # 6. 初始化车辆动力学模型（依赖 world_initializer 的车辆初始化参数）
        # dynamics.py 中的 KinematicBicycleModel 负责根据动作更新车辆状态
        self.dynamics_model = KinematicBicycleModel(config, self.device)

        # 7. 初始化观测生成器
        # observation.py 负责为每个自车生成局部观测
        self.observation_generator = ObservationGenerator(self.road_network, config, self.device, self.spatial_hash)

        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.device)

        # 9. 初始化路径规划器（直接复用 RoadNetwork 数据）
        self.path_planner = PathPlanner(self.device, self.road_network)

        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        self.agents_start_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的起始quad_id
        self.agents_goal_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的目标quad_id
        self.agents_path_plans: Optional[torch.Tensor] = None     # 存储所有智能体的路径规划（世界坐标）
        self.agents_path_plans_local: Optional[torch.Tensor] = None  # 存储所有智能体的路径规划（局部坐标）
        self.goal_radius_tensor: Optional[torch.Tensor] = None
        self.goal_positions: Optional[torch.Tensor] = None
        self.w_lanes_local_with_goal_distances: Optional[torch.Tensor] = None

    def reset(self) -> torch.Tensor:
        """
        重置所有环境，并返回所有智能体的初始观测。
        """
        # 必须先reset world。产生不同大小的车
        self.agents_state, self.agents_start_quad_ids, self.agents_goal_quad_ids = self.world_initializer.initialize_world(self.num_envs, self.max_agents)
        # 重置动力学模型：传入新一批车辆参数并重置内部控制状态与风格参数
        self.dynamics_model.reset(self.world_initializer.vehicle_params)
        # 重置reward风格参数
        self.reward_calculator.reset_episode()
        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        # 重置累积done状态
        self.cumulative_done_mask = None
        
        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation,d,theta_f = self.observation_generator.generate(self.agents_state)
        print("Initial observation generated")

        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        paths = self.path_planner.path_plan(self.agents_start_quad_ids, self.agents_goal_quad_ids)
        self.agents_path_plans = self.path_planner.collect_path_w_lane_ids(paths, self.agents_start_quad_ids, self.agents_goal_quad_ids)
        print(f"Reset complete. World state shape: {self.agents_state.shape}")

        # 将路径截断为前max_keep个有效 w_lane waypoint，并更新目标 quad_id
        max_keep = self.path_length
        invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
        if max_keep > 0:
            truncated_paths = self.agents_path_plans.clone()
            if self.agents_path_plans.shape[2] > max_keep:
                truncated_paths[:, :, max_keep:, :] = invalid_marker
            self.agents_path_plans = truncated_paths

            valid_mask = self.agents_path_plans[..., 0] != invalid_marker  # (B, M, L)
            valid_counts = valid_mask.sum(dim=-1)  # (B, M)
            has_valid = valid_counts > 0
            if has_valid.any():
                last_indices = valid_counts.clamp(min=1) - 1  # (B, M)
                max_index = self.agents_path_plans.shape[2] - 1
                last_indices = torch.clamp(last_indices, max=max_index)
                gather_idx = last_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
                last_points = torch.gather(
                    self.agents_path_plans[..., :2],  # (B, M, L, 2)
                    dim=2,
                    index=gather_idx
                ).squeeze(2)  # (B, M, 2)
                mask_flat = has_valid.view(-1)
                selected_points = last_points.view(-1, 2)[mask_flat]
                distances, nearest_indices = find_nearest_lanes(
                    self.device,
                    self.road_network.quad_centerlines,
                    selected_points,
                    k=1,
                    spatial_hash=self.offroad_checker.spatial_hash
                )
                nearest_indices = nearest_indices.squeeze(1)
                valid_goal_mask = nearest_indices >= 0
                if valid_goal_mask.any():
                    goal_poly_ids = self.road_network.quad_ids[nearest_indices[valid_goal_mask]]
                    goal_flat = self.agents_goal_quad_ids.view(-1)
                    flat_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)[valid_goal_mask]
                    goal_flat[flat_indices] = goal_poly_ids
                    self.agents_goal_quad_ids = goal_flat.view_as(self.agents_goal_quad_ids)

        # 预计算每个路径点到终点的累积距离
        self.precompute_path_plan_goal_distances()
        # self._update_observed_w_lane_goal_distances(
        #     self.agents_path_plans,
        #     self.agents_path_plan_goal_distances,
        #     initial_observation,)

        # 使用 preprocessor 预计算好的 quad centers（准确值）
        # agents_goal_quad_ids 存储的是 poly_id，需要转换为数组索引
        goal_center_indices = self.road_network.poly_id_to_center_idx[self.agents_goal_quad_ids]
        self.goal_positions = self.road_network.quad_centers[goal_center_indices]  # (B, M, 2)
        idx_delta_goal = self.reward_calculator._param_name_to_idx['delta_goal']
        self.goal_radius_tensor = self.reward_calculator.sampled_params[..., idx_delta_goal]
        self.frenet_d = d
        self.frenet_theta_f = theta_f
        # 仍然没有traffic内容
        self.stop_lines = torch.zeros((self.num_envs, self.max_agents,20), dtype=torch.int32, device=self.device)
        return initial_observation,d,theta_f
    
    def step(self, actions: torch.Tensor, debug_collision: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        让所有环境向前步进一个时间步。所有智能体都根据actions更新。
        Args:
            actions (torch.Tensor): 形状为 (B, M, 1) 的动作索引张量。
            debug_collision (bool): 是否为碰撞检测器开启调试模式。
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
                - observation (torch.Tensor): 新的观测 (B, M, obs_dim)。
                - reward (torch.Tensor): 奖励 (B, M)。
                - done (torch.Tensor): 是否结束的标志 (B, M)。
        """
        if self.agents_state is None:
            raise RuntimeError("Must call reset() before calling step().")
        actions = actions.to(self.device)     #action挪到当前显卡上

        states_t0 = self.agents_state.clone() #这一时刻的状态

        # 1. 基于收到的所有动作，更新：采用全批次恒定大小（B*M），并用mask混合回写
        active_mask = self.agents_state[..., 6] > 0.5
        if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
            alive_mask = ~self.cumulative_done_mask
        else:
            alive_mask = torch.ones_like(active_mask, dtype=torch.bool)
        effective_mask = active_mask & alive_mask  # 仅这些需要物理更新，出生了且没有死
        
        # 构造全批次输入 (B*M, 4) 和 (B*M,) 的动作索引
        Bsz, Msz, _S = self.agents_state.shape
        states_flat = self.agents_state[..., :4].contiguous().view(Bsz * Msz, 4)
        
        # 规整actions为 (B, M)，输入格式固定为 (B, M, 1)
        actions_idx = actions.squeeze(-1).long()  # (B, M)
        actions_flat = actions_idx.contiguous().view(Bsz * Msz)
        # 调用动力学（全批次大小恒定），再把无效位置用旧状态覆盖
        new_states_flat = self.dynamics_model.step(states_flat, actions_flat, self.dt)  # (B*M, 4)
        new_states = new_states_flat.view(Bsz, Msz, 4)

        # 只更新有效车辆的状态，无效车辆（未激活或已done）保持旧状态
        self.agents_state[..., :4] = torch.where(effective_mask.unsqueeze(-1), new_states, self.agents_state[..., :4])

        # 2. 离路检测
        is_on_road = torch.ones_like(active_mask) # 默认在路上
        if active_mask.any():
            active_states = self.agents_state[active_mask]
            # OffroadChecker 需要 [x, y, yaw, length, width]
            states_for_checker = active_states[:, [0, 1, 2, 4, 5]]
            active_is_on_road = self.offroad_checker.check_on_road(states_for_checker)
            is_on_road[active_mask] = active_is_on_road
        offroad_mask = ~is_on_road # (B, M)

        # 3. 动态碰撞检测（排除done的车辆）
        # 原地修改active标志，避免clone整个状态张量，减少内存开销
        if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
            # 暂存原始active标志（只clone第6列，内存占用减少7倍）
            original_active_t0 = states_t0[..., 6].clone()
            original_active_t1 = self.agents_state[..., 6].clone()
            
            # 原地修改active标志，将done车辆设为无效
            states_t0[..., 6] = torch.where(self.cumulative_done_mask, 0.0, states_t0[..., 6])
            self.agents_state[..., 6] = torch.where(self.cumulative_done_mask, 0.0, self.agents_state[..., 6])
            
            collision_check_result = self.collision_checker.check(
                states_t0, self.agents_state, debug=debug_collision, debug_env_idx=0
            )
            
            # 恢复原始active标志
            states_t0[..., 6] = original_active_t0
            self.agents_state[..., 6] = original_active_t1
        else:
            collision_check_result = self.collision_checker.check(
                states_t0, self.agents_state, debug=debug_collision, debug_env_idx=0
            )
        all_collisions = collision_check_result

        # 4. 生成新的观测（排除done的车辆），同时获取Frenet坐标信息
        # 原地修改active标志，避免clone整个状态张量
        if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
            # 暂存原始active标志
            original_active = self.agents_state[..., 6].clone()
            # 原地修改active标志，将done车辆设为无效
            self.agents_state[..., 6] = torch.where(self.cumulative_done_mask, 0.0, self.agents_state[..., 6])
            observation, d, theta_f = self.observation_generator.generate(self.agents_state)
            # 恢复原始active标志
            self.agents_state[..., 6] = original_active
        else:
            observation, d, theta_f = self.observation_generator.generate(self.agents_state)
        self.frenet_d = d
        self.frenet_theta_f = theta_f

        # TODO:这里产生goal_reached和waypoints_reached的mask
        B, M = self.agents_state.shape[:2]
        goal_positions = self.goal_positions
        agent_positions = self.agents_state[..., :2]
        distances = torch.norm(agent_positions - goal_positions, dim=-1)
        delta_goal_tensor = self.reward_calculator.sampled_params[..., self.reward_calculator._param_name_to_idx['delta_goal']]
        if delta_goal_tensor.shape != distances.shape:
            delta_goal_tensor = delta_goal_tensor.expand_as(distances)
        goal_reached = distances < delta_goal_tensor
        waypoint_reached = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        self.goal_radius_tensor = delta_goal_tensor

        # 根据距离阈值移除已经过的路径点(贪吃蛇的部分)
        if self.agents_path_plans is not None and self.agents_path_plans.numel() > 0:
            invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
            waypoints_xy = self.agents_path_plans[..., :2]
            valid_waypoints = waypoints_xy[..., 0] != invalid_marker
            if valid_waypoints.any():
                diffs = waypoints_xy - agent_positions.unsqueeze(-2)
                dists_wp = torch.norm(diffs, dim=-1)  # (B, M, L)
                goal_radii = self.goal_radius_tensor.unsqueeze(-1).expand_as(dists_wp)
                reached_mask = valid_waypoints & (dists_wp <= goal_radii)
                reached_any = reached_mask.any(dim=-1)
                if reached_any.any():
                    first_reached = torch.argmax(reached_mask.float(), dim=-1)
                    indices = torch.arange(self.agents_path_plans.shape[2], device=self.device).view(1, 1, -1)
                    prefix_mask = (indices <= first_reached.unsqueeze(-1)) & reached_any.unsqueeze(-1)
                    inv_tensor = torch.full_like(self.agents_path_plans[..., 0], invalid_marker)
                    self.agents_path_plans[..., 0] = torch.where(prefix_mask, inv_tensor, self.agents_path_plans[..., 0])
                    self.agents_path_plans[..., 1] = torch.where(prefix_mask, inv_tensor, self.agents_path_plans[..., 1])
                    self.agents_path_plans[..., 2] = torch.where(prefix_mask, inv_tensor, self.agents_path_plans[..., 2])
                    # 更新一下agents_path_plan_goal_distances，因为有的点消失了，它的距离要重新计算。（赋予无效值的点，距离就是无穷大    ）
                    if self.agents_path_plan_goal_distances is not None:
                        self.agents_path_plan_goal_distances = torch.where(
                            prefix_mask,
                            inv_tensor,
                            self.agents_path_plan_goal_distances)
                    #TODO: 更新一下w_lane

        # TODO: 这里判断目前观测中的w_lanes_local是否在agents_path_plans内？产生self.w_lanes_local_with_goal_distances(dx,dy,delta_goal)用于输入网络

        # 7. 计算奖励（传入Frenet坐标和动作）
        reward = self._calculate_reward(all_collisions, offroad_mask, d, theta_f, goal_reached, waypoint_reached, actions)

        # 8. 检查是否结束（包含目标到达判断）
        done = all_collisions|offroad_mask|goal_reached
        # 保存done状态供下次step使用（累积done状态，一旦done就保持done）
        if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
            self.cumulative_done_mask = self.cumulative_done_mask | done
        else:
            self.cumulative_done_mask = done.clone()
        return observation, reward, done
    
    def _calculate_reward(
        self,
        all_collisions: torch.Tensor,
        offroad_mask: torch.Tensor,
        d: torch.Tensor,
        theta_f: torch.Tensor,
        goal_reached: torch.Tensor,
        waypoint_reached: torch.Tensor,
        actions: torch.Tensor = None,) -> torch.Tensor:
        """
        为所有智能体计算奖励。
        Args:
            all_collisions (torch.Tensor): 碰撞状态 (B, M)
            offroad_mask (torch.Tensor): 离路状态 (B, M)
            d (torch.Tensor): Frenet横向距离 (B, M)
            theta_f (torch.Tensor): Frenet角度误差 (B, M)
            goal_reached (torch.Tensor): 目标达成掩码 (B, M)
            waypoint_reached (torch.Tensor): 路点达成掩码 (B, M)
            actions (torch.Tensor): 动作索引 (B, M, 1)，用于直接获取jerk值
        Returns:
            torch.Tensor: 奖励值 (B, M)
        """

        # 构建扩展的状态张量，包含加速度信息
        B, M, _ = self.agents_state.shape
        # 创建扩展的状态张量 (B, M, 10)
        # [0]: x, [1]: y, [2]: heading, [3]: speed, [4]: along, [5]: alat, 
        # [6]: along_jerk, [7]: alat_jerk, [8]: theta_f, [9]: d
        extended_state = torch.zeros((B, M, 10), device=self.device)
        # 复制原始状态信息
        extended_state[..., :4] = self.agents_state[..., :4]  # x, y, heading, speed
        
        # 从动力学模型与动作空间获取批量化的 (B,M) 张量
        if hasattr(self.dynamics_model, 'current_along') and hasattr(self.dynamics_model, 'current_alat'):
            # 激活掩码（与step阶段保持一致：仅激活且未done的为有效位置）
            active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
            if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
                alive_mask = ~self.cumulative_done_mask
            else:
                alive_mask = torch.ones_like(active_mask, dtype=torch.bool)
            effective_mask = active_mask & alive_mask
            
            # 1) 构造全局 along/alat 加速度 (B,M)，仅有效位置为有效值
            #    将连续的有效向量按掩码散射回批量形状
            along_active = self.dynamics_model.current_along  # (N_effective,) or None
            alat_active = self.dynamics_model.current_alat    # (N_effective,) or None
            flat_mask = effective_mask.view(-1)
            zeros_flat = torch.zeros(B * M, device=self.device)

            # 若动力学尚未初始化（None），以0填充
            if along_active is None:
                along_active = torch.zeros(int(flat_mask.sum().item()), device=self.device)
            if alat_active is None:
                alat_active = torch.zeros(int(flat_mask.sum().item()), device=self.device)
            full_along = zeros_flat.masked_scatter(flat_mask, along_active).view(B, M)
            full_alat  = zeros_flat.clone().masked_scatter(flat_mask, alat_active).view(B, M)

            # 2) 从动作空间一次性映射出所有智能体的 jerk (B,M,2)，未激活位置后续用掩码置零
            if actions is not None:
                # 规整为 (B, M) 的索引，输入格式固定为 (B, M, 1)
                actions_idx = actions.squeeze(-1).long()  # (B, M)
                jerk_all = self.dynamics_model.discrete_action_space.get_action(actions_idx.view(-1))  # (B*M,2)
                jerk_all = jerk_all.view(B, M, 2)
                full_along_jerk = jerk_all[..., 0]  # (B, M)
                full_alat_jerk  = jerk_all[..., 1]  # (B, M)
                # 仅对有效（激活且未done）体保留数值
                mask_f = effective_mask.float()
                full_along_jerk = full_along_jerk * mask_f
                full_alat_jerk  = full_alat_jerk  * mask_f
            else:
                full_along_jerk = torch.zeros((B, M), device=self.device)
                full_alat_jerk  = torch.zeros((B, M), device=self.device)

            # 3) 写入扩展状态（数值构造已完成，无需布尔掩码赋值）
            extended_state[..., 4] = full_along      # along
            extended_state[..., 5] = full_alat       # alat
            extended_state[..., 6] = full_along_jerk # along_jerk
            extended_state[..., 7] = full_alat_jerk  # alat_jerk

        # 直接使用传入的Frenet坐标信息，避免重复计算
        extended_state[..., 8] = theta_f  # theta_f - Frenet角度误差
        extended_state[..., 9] = d        # d - Frenet横向距离
        # 准备目标奖励计算的参数

        # 调用奖励计算器，这个速度很快
        reward = self.reward_calculator.calculate(
            extended_state,
            all_collisions,
            offroad_mask,
            dt=self.dt,
            goal_reached=goal_reached,
            waypoint_reached=waypoint_reached,
        )
        
        # 过滤掉非active或已done车辆的奖励（与动力学一致的有效掩码）
        active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
        if hasattr(self, 'cumulative_done_mask') and self.cumulative_done_mask is not None:
            alive_mask = ~self.cumulative_done_mask
        else:
            alive_mask = torch.ones_like(active_mask, dtype=torch.bool)
        effective_mask = active_mask & alive_mask
        reward = reward * effective_mask.float()  # 非有效车辆的奖励设为0，等价于"done后奖励不再更新"
        self.extend_state = extended_state # 用于传入网络
        return reward
    
    def precompute_path_plan_goal_distances(self) -> None:
        """Compute cumulative distance from each path waypoint to the final goal."""
        if self.agents_path_plans is None or self.agents_path_plans.numel() == 0:
            self.agents_path_plan_goal_distances = None
            return
        invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
        waypoints_xy = self.agents_path_plans[..., :2]
        valid_waypoints = waypoints_xy[..., 0] != invalid_marker

        distances = torch.full(
            valid_waypoints.shape,
            invalid_marker,
            device=self.device,
            dtype=waypoints_xy.dtype,
        )

        if not valid_waypoints.any():
            self.agents_path_plan_goal_distances = distances
            return

        coords = torch.where(
            valid_waypoints.unsqueeze(-1),
            waypoints_xy,
            torch.zeros_like(waypoints_xy),
        )
        segment_vecs = coords[..., 1:, :] - coords[..., :-1, :]
        segment_lengths = torch.norm(segment_vecs, dim=-1)
        segment_valid = valid_waypoints[..., :-1] & valid_waypoints[..., 1:]
        segment_lengths = torch.where(segment_valid, segment_lengths, torch.zeros_like(segment_lengths))

        zeros_tail = torch.zeros(
            (*segment_lengths.shape[:-1], 1),
            device=self.device,
            dtype=segment_lengths.dtype,
        )
        segment_with_tail = torch.cat([segment_lengths, zeros_tail], dim=-1)
        cumulative = torch.flip(
            torch.cumsum(torch.flip(segment_with_tail, dims=[-1]), dim=-1),
            dims=[-1],
        )

        distances = torch.where(valid_waypoints, cumulative, distances)
        self.agents_path_plan_goal_distances = distances

    def _update_observed_w_lane_goal_distances(
        self,
        agents_path_plans: Optional[torch.Tensor],
        path_plan_goal_distances: Optional[torch.Tensor],
        observation: Optional[torch.Tensor],) -> None:
        """Align observed w-lane points with path-plan goal distances and cache augmented tensors."""
        if (
            observation is None
            or agents_path_plans is None
            or path_plan_goal_distances is None
            or agents_path_plans.numel() == 0
            or path_plan_goal_distances.numel() == 0
        ):
            print("error")
            return

        B, M = observation.shape[0], observation.shape[1]
        obs_gen = self.observation_generator
        local_dim = obs_gen.local_state_dim
        neighbor_dim = obs_gen.num_neighbors * obs_gen.neighbor_feature_dim
        w_lane_feat_dim = obs_gen.w_lane_feature_dim
        num_w_lanes = obs_gen.num_w_lanes
        start = local_dim + neighbor_dim
        end = start + num_w_lanes * w_lane_feat_dim
        if end > observation.shape[-1]:
            self.w_lanes_local_with_goal_distances = None
            return
        w_lanes_flat = observation[..., start:end]
        w_lanes_local = w_lanes_flat.view(B, M, num_w_lanes, w_lane_feat_dim)
        lanes_valid_mask = (w_lanes_local[..., 0].abs() + w_lanes_local[..., 1].abs()) > 1e-6

        ego_pos = self.agents_state[..., :2]
        ego_yaw = self.agents_state[..., 2]
        cos_yaw = torch.cos(ego_yaw)
        sin_yaw = torch.sin(ego_yaw)
        rot_matrix = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw], dim=-1),
                torch.stack([sin_yaw, cos_yaw], dim=-1),
            ],
            dim=-2,
        )
        lanes_local_xy = w_lanes_local[..., :2]
        lanes_world = (
            torch.bmm(
                lanes_local_xy.view(B * M, num_w_lanes, 2),
                rot_matrix.view(B * M, 2, 2),
            ).view(B, M, num_w_lanes, 2)
            + ego_pos.unsqueeze(-2)
        )
        invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
        lanes_distance = torch.full(
            (B, M, num_w_lanes),
            invalid_marker,
            device=self.device,
            dtype=path_plan_goal_distances.dtype,
        )
        plan_world = agents_path_plans[..., :2]
        valid_plan_mask = plan_world[..., 0] != invalid_marker
        diff = plan_world.unsqueeze(3) - lanes_world.unsqueeze(2)  # (B, M, L, K, 2)
        dist_sq = diff.pow(2).sum(-1).permute(0, 1, 3, 2)  # (B, M, K, L)
        inf = torch.tensor(float("inf"), device=self.device, dtype=dist_sq.dtype)
        dist_sq = torch.where(valid_plan_mask.unsqueeze(2), dist_sq, inf)
        min_dist, closest_idx = dist_sq.min(dim=3)
        matched_plan_valid = torch.gather(valid_plan_mask, 2, closest_idx)
        tolerance_sq = 0.04
        match_mask = (min_dist <= tolerance_sq) & matched_plan_valid & lanes_valid_mask
        gathered_distances = torch.gather(path_plan_goal_distances, 2, closest_idx)
        lanes_distance = torch.where(match_mask, gathered_distances, lanes_distance)
        self.w_lanes_local_with_goal_distances = torch.cat(
            [w_lanes_local, lanes_distance.unsqueeze(-1)], dim=-1
        )

if __name__ == "__main__":
    # ==================== 1. 初始化 TeraflowSimulator ====================
    import json
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    sim = TeraflowSimulator(config=cfg, device=torch.device(device))
    initial_observation, d, theta_f = sim.reset()
    sim._last_action = torch.zeros((sim.num_envs, sim.max_agents), dtype=torch.long, device=sim.device)

    # ==================== 2. 使用 Pygame/OpenGL 可视化 active 车辆的路径规划 ====================
    from utils.pygame_utils import visualize_path_planning
    
    # 定义observation回调函数来获取观测数据
    def observation_callback(agents_state, b, m):
        neighbor_states_world = sim.observation_generator._get_nearest_neighbors(agents_state)
        w_lanes_world, w_boundaries_world = sim.observation_generator._get_precomputed_w_lanes(agents_state)
        local_state_tmp, neighbors_local_tmp, w_lanes_local_tmp, w_boundaries_local_tmp = \
            sim.observation_generator._world_to_ego_centric(
                agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
            )
        return neighbors_local_tmp[b, m], w_lanes_local_tmp[b, m], w_boundaries_local_tmp[b, m]

    # 定义step回调：按 W 键时执行一步仿真并返回最新状态
    def step_callback(b_idx: int, m_idx: int):
        with torch.no_grad():
            B, M = sim.agents_state.shape[:2]
            actions = torch.full((B, M, 1), 7, dtype=torch.long, device=sim.device)
            observation, reward, done = sim.step(actions)
            sim._last_action = actions.squeeze(-1).clone()
        print("已执行一步仿真。")
        return (
            sim.agents_state,
            sim.agents_path_plans,
            sim.goal_positions,
            sim.goal_radius_tensor,
            sim.cumulative_done_mask,
        )

    def info_callback(agents_state, goal_positions, goal_radii, done_mask, b, m):
        state = agents_state[b, m].detach().cpu().numpy()
        x, y, yaw, speed, length, width, active = state
        lines = [
            ("Agent", f"B={b}, M={m}"),
            ("Active", "Yes" if active > 0.5 else "No"),
            ("Position", f"{x:.2f}, {y:.2f}"),
            ("Yaw", f"{math.degrees(yaw):.2f}°"),
            ("Speed", f"{speed:.2f} m/s"),
            ("Size", f"L={length:.2f}, W={width:.2f}")
        ]
        # 绘制可视化中的终点红点（路径最后一个有效点）
        goal_point = None
        try:
            path = sim.agents_path_plans[b, m]
            invalid_marker_tensor = torch.tensor(
                sim.path_planner.INVALID_w_lane_id_MARKER,
                dtype=path.dtype,
                device=path.device,
            )
            valid_mask = path[:, 0] != invalid_marker_tensor
            if valid_mask.any():
                last_valid_idx = torch.nonzero(valid_mask, as_tuple=False)[-1].item()
                last_point = path[last_valid_idx, :2].detach().cpu().numpy()
                goal_point = (float(last_point[0]), float(last_point[1]))
        except Exception:
            goal_point = None

        if goal_point is not None:
            lines.append(("Goal", f"{goal_point[0]:.2f}, {goal_point[1]:.2f}"))
        else:
            lines.append(("Goal", "N/A"))

        radius_val = None
        if goal_radii is not None:
            try:
                radius_val = goal_radii[b, m]
                if torch.is_tensor(radius_val):
                    radius_val = radius_val.detach().cpu().item()
            except Exception:
                radius_val = None
        if radius_val is not None:
            lines.append(("Goal Radius", f"{float(radius_val):.2f} m"))
        else:
            lines.append(("Goal Radius", "N/A"))

        reward_calc = getattr(sim, "reward_calculator", None)
        if reward_calc is not None:
            def fetch_reward(component_name: str):
                value = getattr(reward_calc, component_name, None)
                if value is None:
                    return None
                try:
                    return float(value[b, m].detach().cpu().item())
                except Exception:
                    try:
                        return float(value[b, m])
                    except Exception:
                        return None

            reward_components = [
                ("Goal Reward", "last_goal_reward"),
                ("Collision Penalty", "last_collision_penalty"),
                ("Offroad Penalty", "last_offroad_penalty"),
                ("Comfort Penalty", "last_comfort_penalty"),
                ("Lane Align Reward", "last_lane_alignment_reward"),
                ("Lane Center Reward", "last_lane_center_reward"),
                ("Velocity Reward", "last_velocity_reward"),
                ("Reverse Penalty", "last_reverse_penalty"),
                ("Stop-Line Penalty", "last_stop_line_penalty"),
                ("Timestep Penalty", "last_timestep_penalty"),
            ]
            for label, attr in reward_components:
                comp_value = fetch_reward(attr)
                if comp_value is not None:
                    lines.append((label, f"{comp_value:+.6f}"))

        if hasattr(sim, "frenet_d") and sim.frenet_d is not None:
            try:
                d_val = sim.frenet_d[b, m].detach().cpu().item()
                lines.append(("d", f"{d_val:.2f} m"))
            except Exception:
                pass
        if hasattr(sim, "frenet_theta_f") and sim.frenet_theta_f is not None:
            try:
                theta_val = sim.frenet_theta_f[b, m].detach().cpu().item()
                lines.append(("theta_f", f"{math.degrees(theta_val):.2f}°"))
            except Exception:
                pass
        dyn = getattr(sim, "dynamics_model", None)
        if dyn is not None:
            def pick_value(tensor):
                if tensor is None:
                    return None
                try:
                    value = tensor.view(sim.num_envs, sim.max_agents)[b, m]
                except Exception:
                    idx = b * sim.max_agents + m
                    if idx < tensor.numel():
                        value = tensor[idx]
                    else:
                        return None
                return value.detach().cpu().item()
            long_acc = pick_value(getattr(dyn, "current_along", None))
            if long_acc is not None:
                lines.append(("Long Acc", f"{long_acc:.2f} m/s^2"))
            lat_acc = pick_value(getattr(dyn, "current_alat", None))
            if lat_acc is not None:
                lines.append(("Lat Acc", f"{lat_acc:.2f} m/s^2"))
            steering = pick_value(getattr(dyn, "current_steering_angle", None))
            if steering is not None:
                lines.append(("Steering", f"{math.degrees(steering):.2f}°"))
        last_action = getattr(sim, "_last_action", None)
        if last_action is not None and b < last_action.shape[0] and m < last_action.shape[1]:
            action_idx = int(last_action[b, m].detach().cpu().item())
            lines.append(("Action Index", action_idx))
            try:
                jerk = sim.dynamics_model.discrete_action_space.get_action(
                    torch.tensor([action_idx], device=sim.device)
                )[0].detach().cpu().numpy()
                lines.append(("Jerk (long, lat)", f"{jerk[0]:.2f}, {jerk[1]:.2f}"))
            except Exception:
                pass
        if done_mask is not None:
            try:
                done_val = bool(done_mask[b, m].item())
                lines.append(("Done", "Yes" if done_val else "No"))
            except Exception:
                pass
        return lines

    print("按 SPACE 切换车辆，按 W 运行一步仿真，按 ESC 退出。")
    visualize_path_planning(
        agents_state=sim.agents_state,
        agents_path_plans=sim.agents_path_plans,
        quads_vertices=sim.road_network.left_boundaries,
        batch_idx=0,
        invalid_marker_value=sim.path_planner.INVALID_w_lane_id_MARKER,
        horizon=sim.observation_generator.horizon,
        observation_callback=observation_callback,
        step_callback=step_callback,
        info_callback=info_callback,
        agents_start_quad_ids=sim.agents_start_quad_ids,
        agents_goal_quad_ids=sim.agents_goal_quad_ids,
        goal_positions=sim.goal_positions,
        goal_radii=sim.goal_radius_tensor,
        done_mask=sim.cumulative_done_mask,
    )
    print("退出可视化。")

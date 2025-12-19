import torch
import os
import sys
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
        self.agents_path_plans: Optional[torch.Tensor] = None     # 存储所有智能体的路径规划 w_lane_id 序列
        self.agents_path_plans_world: Optional[torch.Tensor] = None  # 存储所有智能体的路径规划（世界坐标 xy）(B, M, L, 2)
        self.agents_path_plans_local: Optional[torch.Tensor] = None  # 存储所有智能体的路径规划（局部坐标）
        self.goal_radius_tensor: Optional[torch.Tensor] = None
        self.goal_positions: Optional[torch.Tensor] = None
        self.w_lanes_local_with_goal_distances: Optional[torch.Tensor] = None
        self.w_lane_goal_distances_full: Optional[torch.Tensor] = None
        self.sampled_waypoint_ids: Optional[torch.Tensor] = None
        
        # 获取最大路径长度（从配置中读取）
        obs_config = simulator_config.get('observation', {})
        self.max_path_length = int(obs_config.get('num_navigation_chains', 128))

    def update_path_length(self, increment: int = 1) -> bool:
        """
        更新路径长度，每次增加 increment（默认为1）。
        如果已达到最大值，则不再增加。
        
        Args:
            increment: 要增加的路径长度（默认为1）
            
        Returns:
            bool: 如果成功更新返回 True，如果已达到最大值返回 False
        """
        if self.path_length >= self.max_path_length:
            return False
        new_length = min(self.path_length + increment, self.max_path_length)
        if new_length != self.path_length:
            self.path_length = new_length
            return True
        return False

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
        
        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        paths = self.path_planner.path_plan(self.agents_start_quad_ids, self.agents_goal_quad_ids)
        self.agents_path_plans = self.path_planner.collect_path_w_lane_ids(paths, self.agents_start_quad_ids, self.agents_goal_quad_ids)
        print(f"Reset complete. World state shape: {self.agents_state.shape}")

        # 将路径截断为前max_keep个有效 w_lane，并更新目标 quad_id
        max_keep = self.path_length
        invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER

        truncated_paths = self.agents_path_plans.clone()
        if self.agents_path_plans.shape[2] > max_keep:
            truncated_paths[:, :, max_keep:] = invalid_marker
        self.agents_path_plans = truncated_paths

        # 根据最后一个有效路径点的 w_lane_id 直接获取对应的 quad_id (poly_id)
        valid_mask = self.agents_path_plans != invalid_marker  # (B, M, L)
        valid_counts = valid_mask.sum(dim=-1)  # (B, M)
        has_valid = valid_counts > 0
        if has_valid.any():
            last_indices = valid_counts.clamp(min=1) - 1  # (B, M)
            max_index = self.agents_path_plans.shape[2] - 1
            last_indices = torch.clamp(last_indices, max=max_index)
            gather_idx = last_indices.unsqueeze(-1)
            last_w_lane_ids = torch.gather(self.agents_path_plans, 2, gather_idx).squeeze(2)  # (B, M)
            
            # 使用 w_lane_id_to_quad_id_tensor 批量查询对应的 quad_id
            # 只更新有有效路径点的 agent
            mask_flat = has_valid.view(-1)
            last_w_lane_ids_flat = last_w_lane_ids.view(-1)
            valid_w_lane_ids = last_w_lane_ids_flat[mask_flat]  # 只取有效的 w_lane_id
            
            if valid_w_lane_ids.numel() > 0:
                # 使用张量索引查询，无效的 w_lane_id 会返回 -1
                valid_w_lane_ids_clamped = torch.clamp(valid_w_lane_ids, min=0, max=self.road_network.w_lane_id_to_quad_id_tensor.shape[0] - 1)
                goal_poly_ids = self.road_network.w_lane_id_to_quad_id_tensor[valid_w_lane_ids_clamped]
                valid_goal_mask = goal_poly_ids >= 0  # 过滤掉无效的映射
                if valid_goal_mask.any():
                    goal_flat = self.agents_goal_quad_ids.view(-1)
                    flat_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)[valid_goal_mask]
                    goal_flat[flat_indices] = goal_poly_ids[valid_goal_mask]
                    self.agents_goal_quad_ids = goal_flat.view_as(self.agents_goal_quad_ids)

        # 使用 preprocessor 预计算好的 quad centers（准确值）
        # agents_goal_quad_ids 存储的是 poly_id，需要转换为数组索引
        goal_center_indices = self.road_network.poly_id_to_center_idx[self.agents_goal_quad_ids]
        self.goal_positions = self.road_network.quad_centers[goal_center_indices]  # (B, M, 2)
        idx_delta_goal = self.reward_calculator._param_name_to_idx['delta_goal']
        self.goal_radius_tensor = self.reward_calculator.sampled_params[..., idx_delta_goal]

        # 移除 reset 时已经处于 goal 半径内的前缀路径点，避免 Δs 错位
        if self.agents_path_plans is not None and self.agents_path_plans.numel() > 0:
            agent_positions = self.agents_state[..., :2]
            waypoints_xy = self.path_planner.get_w_lane_centers_by_id(self.agents_path_plans)
            diffs = waypoints_xy - agent_positions.unsqueeze(-2)
            dists_wp = torch.norm(diffs, dim=-1)
            goal_radii = self.goal_radius_tensor.unsqueeze(-1).expand_as(dists_wp)
            valid_waypoints = self.agents_path_plans != invalid_marker
            reached_mask = valid_waypoints & (dists_wp <= goal_radii)
            reached_any = reached_mask.any(dim=-1)
            if reached_any.any():
                indices = torch.arange(self.agents_path_plans.shape[2], device=self.device).view(1, 1, -1)
                last_reached = torch.where(
                    reached_mask,
                    indices,
                    torch.full_like(indices, -1),
                ).max(dim=-1).values
                prefix_mask = (
                    (indices <= last_reached.unsqueeze(-1))
                    & (last_reached.unsqueeze(-1) >= 0)
                    & reached_any.unsqueeze(-1)
                )
                inv_ids = torch.full_like(self.agents_path_plans, invalid_marker)
                self.agents_path_plans = torch.where(prefix_mask, inv_ids, self.agents_path_plans)

        # 预计算路径点的世界坐标（很多地方要用）
        self.agents_path_plans_world = self.path_planner.get_w_lane_centers_by_id(self.agents_path_plans)  # (B, M, L, 2)

        # 预计算每个路径点到终点的累积距离
        self.precompute_path_plan_goal_distances()

        # 构建全局 w_lane 的 Δs 特征
        self.w_lane_goal_distances_full = self._build_w_lane_features_with_goal()
        
        # TODO:为每一个agents_path_plans中随机抽取0-3个w_lane记录他们的的id，为后面waypoints_reached的mask做准备
        self._sample_waypoint_ids_for_mask()
        
        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation, d, theta_f = self.observation_generator.generate(self.agents_state)
        print("Initial observation generated")
        self.frenet_d = d
        self.frenet_theta_f = theta_f
        self._update_observed_w_lane_features(initial_observation)

        # 仍然没有traffic内容
        self.stop_lines = torch.ones((self.num_envs, self.max_agents,20), dtype=torch.int32, device=self.device)
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
        self._update_observed_w_lane_features(observation)

        # TODO:这里产生goal_reached的mask
        B, M = self.agents_state.shape[:2]
        goal_positions = self.goal_positions
        agent_positions = self.agents_state[..., :2]
        distances = torch.norm(agent_positions - goal_positions, dim=-1)
        delta_goal_tensor = self.goal_radius_tensor
        goal_reached = distances < delta_goal_tensor

        # TODO: 这里处理waypoint_reach的情况
        waypoint_reached = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        
        # 根据距离阈值移除已经过的路径点(贪吃蛇的部分)
        if self.agents_path_plans is not None and self.agents_path_plans.numel() > 0:
            invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
            valid_waypoints = self.agents_path_plans != invalid_marker
            if valid_waypoints.any():
                waypoints_xy = self.path_planner.get_w_lane_centers_by_id(self.agents_path_plans)
                diffs = waypoints_xy - agent_positions.unsqueeze(-2)
                dists_wp = torch.norm(diffs, dim=-1)  # (B, M, L)
                goal_radii = self.goal_radius_tensor.unsqueeze(-1).expand_as(dists_wp)
                reached_mask = valid_waypoints & (dists_wp <= goal_radii)

                reached_any = reached_mask.any(dim=-1)
                if reached_any.any():
                    indices = torch.arange(self.agents_path_plans.shape[2], device=self.device).view(1, 1, -1)
                    last_reached = torch.where(
                        reached_mask,
                        indices,
                        torch.full_like(indices, -1),
                    ).max(dim=-1).values
                    prefix_mask = (
                        (indices <= last_reached.unsqueeze(-1))
                        & (last_reached.unsqueeze(-1) >= 0)
                        & reached_any.unsqueeze(-1)
                    )
                    reached_ids_before = torch.where(
                        prefix_mask,
                        self.agents_path_plans,
                        torch.full_like(self.agents_path_plans, invalid_marker),
                    )
                    inv_ids = torch.full_like(self.agents_path_plans, invalid_marker)
                    self.agents_path_plans = torch.where(prefix_mask, inv_ids, self.agents_path_plans)
                    # 更新一下agents_path_plan_goal_distances，因为有的点消失了，它的距离要重新计算。（赋予无效值的点，距离就是无穷大）
                    if self.agents_path_plan_goal_distances is not None:
                        inv_dist = torch.full_like(
                            self.agents_path_plan_goal_distances,
                            float(self.path_planner.INVALID_MARKER),
                        )
                        self.agents_path_plan_goal_distances = torch.where(
                            prefix_mask,
                            inv_dist,
                            self.agents_path_plan_goal_distances)
                    
                    # 重新计算路径点到终点的距离（因为路径点被移除了）
                    self.precompute_path_plan_goal_distances()
                    
                    # 重新构建全局 w_lane 的 Δs 特征（因为路径点被移除了，需要更新距离）
                    self.w_lane_goal_distances_full = self._build_w_lane_features_with_goal()
                    
                    # 获取到达的路径点ID，用于更新 sampled_waypoint_ids
                    # 初始化变量，确保在条件块外也能使用
                    b_idx_reached = None
                    m_idx_reached = None
                    reached_ids_vals = None
                    
                    if self.w_lane_goal_distances_full is not None:
                        b_idx, m_idx, l_idx = torch.nonzero(prefix_mask, as_tuple=True)
                        if b_idx.numel() > 0:
                            reached_ids_vals_temp = reached_ids_before[b_idx, m_idx, l_idx]
                            valid_reached = reached_ids_vals_temp != invalid_marker
                            if valid_reached.any():
                                b_idx_reached = b_idx[valid_reached]
                                m_idx_reached = m_idx[valid_reached]
                                reached_ids_vals = reached_ids_vals_temp[valid_reached]
                    
                    # 更新 sampled_waypoint_ids：将到达的路径点标记为无效
                    if self.sampled_waypoint_ids is not None and b_idx_reached is not None:
                        sample_mask = (
                            self.sampled_waypoint_ids[b_idx_reached, m_idx_reached]
                            == reached_ids_vals.unsqueeze(-1)
                        )
                        if sample_mask.any():
                            self.sampled_waypoint_ids[b_idx_reached, m_idx_reached] = torch.where(
                                sample_mask,
                                torch.full_like(self.sampled_waypoint_ids[b_idx_reached, m_idx_reached], invalid_marker),
                                self.sampled_waypoint_ids[b_idx_reached, m_idx_reached],
                            )
                            waypoint_reached[b_idx_reached, m_idx_reached] = True
                    self._update_observed_w_lane_features(observation)
        
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
        invalid_distance_marker = float(self.path_planner.INVALID_MARKER)
        valid_waypoints = self.agents_path_plans != invalid_marker
        waypoints_xy = self.path_planner.get_w_lane_centers_by_id(self.agents_path_plans)
        distances = torch.full(
            valid_waypoints.shape,
            invalid_distance_marker,
            device=self.device,
            dtype=waypoints_xy.dtype,
        )
        if not valid_waypoints.any():
            self.agents_path_plan_goal_distances = distances
            return
        # 向量化计算累积距离，正确处理无效点，避免索引错位
        # 策略：只对连续有效点之间的段进行累积，跳过无效点
        
        # 1. 计算所有相邻点之间的段长度（直接使用原始坐标，不将无效点设为0）
        segment_vecs = waypoints_xy[..., 1:, :] - waypoints_xy[..., :-1, :]  # (B, M, L-1, 2)
        segment_lengths = torch.norm(segment_vecs, dim=-1)  # (B, M, L-1)
        # 2. 标记哪些段是有效的（两端点都有效）
        segment_valid = valid_waypoints[..., :-1] & valid_waypoints[..., 1:]  # (B, M, L-1)
        # 3. 将无效段的长度设为0（这些段会被跳过，不影响累积）
        segment_lengths = torch.where(segment_valid, segment_lengths, torch.zeros_like(segment_lengths))
        # 4. 在末尾添加0（最后一个点到终点的距离为0）
        zeros_tail = torch.zeros(
            (*segment_lengths.shape[:-1], 1),
            device=self.device,
            dtype=segment_lengths.dtype,
        )
        segment_with_tail = torch.cat([segment_lengths, zeros_tail], dim=-1)  # (B, M, L)
        # 5. 从后往前累积距离
        cumulative = torch.flip(
            torch.cumsum(torch.flip(segment_with_tail, dims=[-1]), dim=-1),
            dims=[-1],
        )  # (B, M, L)
        # 6. 关键修复：找到每个路径的最后一个有效点，将其距离设为0
        #    使用向量化操作找到最后一个有效点的位置
        B, M, L = valid_waypoints.shape
        # 6.1 创建位置索引：对于每个位置，如果是最后一个有效点，标记为True
        #     使用 argmax 在反向维度上找到最后一个有效点的索引
        valid_positions_reversed = torch.flip(valid_waypoints.long(), dims=[-1])  # (B, M, L)
        # 对于每个 (B, M)，找到第一个有效点的位置（在反向序列中，即原序列的最后一个有效点）
        last_valid_indices = (L - 1) - torch.argmax(valid_positions_reversed, dim=-1)  # (B, M)
        # 处理没有有效点的情况（argmax 会返回0，需要检查）
        has_any_valid = valid_waypoints.any(dim=-1)  # (B, M)
        last_valid_indices = torch.where(has_any_valid, last_valid_indices, torch.zeros_like(last_valid_indices))
        # 6.2 创建最后一个有效点的掩码
        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, M)
        agent_indices = torch.arange(M, device=self.device).view(1, M).expand(B, M)
        last_valid_mask = torch.zeros_like(valid_waypoints, dtype=torch.bool)
        last_valid_mask[batch_indices, agent_indices, last_valid_indices] = has_any_valid
        # 6.3 将最后一个有效点的距离设为0
        cumulative = torch.where(last_valid_mask, torch.zeros_like(cumulative), cumulative)
        # 7. 给路径上所有点都赋予到终点的距离（包括无效点）
        distances = cumulative
        self.agents_path_plan_goal_distances = distances

    def get_path_plan_features_with_goal_distances(self) -> Optional[torch.Tensor]:
        """返回包含 (x, y, angle, Δs) 的路径特征，保持与路径顺序一致，供可视化使用。"""
        if (
            self.agents_path_plans is None
            or self.agents_path_plans.numel() == 0
        ):
            return None

        coords = self.path_planner.get_w_lane_features_by_id(self.agents_path_plans)
        if coords is None:
            return None

        if self.agents_path_plan_goal_distances is None:
            delta = torch.full(
                self.agents_path_plans.shape,
                float(self.path_planner.INVALID_MARKER),
                dtype=coords.dtype,
                device=self.device,
            )
        else:
            delta = self.agents_path_plan_goal_distances.to(device=self.device, dtype=coords.dtype)

        return torch.cat([coords, delta.unsqueeze(-1)], dim=-1)

    def _build_w_lane_features_with_goal(self) -> Optional[torch.Tensor]:
        """为每个 agent 构建全局 w_lane 的 Δs 信息 (B, M, N_w_lane)。"""
        if (
            self.agents_path_plans is None
            or self.agents_path_plans.numel() == 0
            or self.agents_path_plan_goal_distances is None
        ):
            return None

        planner = self.path_planner
        total_w_lanes = planner.w_lane_features.shape[0]
        if total_w_lanes == 0:
            B, M, _ = self.agents_path_plans.shape
            return torch.empty((B, M, 0), dtype=self.agents_path_plan_goal_distances.dtype, device=self.device)

        plan_ids = self.agents_path_plans
        plan_dists = self.agents_path_plan_goal_distances
        idx = planner.map_w_lane_ids_to_indices(plan_ids)

        B, M, L = plan_ids.shape
        invalid_value = float(planner.INVALID_MARKER)

        idx_mask = (idx >= 0) & (idx < total_w_lanes)
        if not idx_mask.any():
            return torch.full(
                (B, M, total_w_lanes),
                invalid_value,
                dtype=plan_dists.dtype,
                device=self.device,
            )

        b_idx, m_idx, l_idx = idx_mask.nonzero(as_tuple=True)
        target_idx = idx[b_idx, m_idx, l_idx]
        src_vals = plan_dists[b_idx, m_idx, l_idx]

        flat_size = B * M
        inf_value = torch.tensor(float("inf"), dtype=plan_dists.dtype, device=self.device)
        flat_full = torch.full(
            (flat_size * total_w_lanes,),
            inf_value,
            dtype=plan_dists.dtype,
            device=self.device,
        )
        flat_row = b_idx * M + m_idx
        linear_idx = flat_row * total_w_lanes + target_idx

        flat_full.scatter_reduce_(
            dim=0,
            index=linear_idx,
            src=src_vals,
            reduce="amin",
            include_self=True,
        )

        full = flat_full.view(B, M, total_w_lanes)
        full = torch.where(torch.isinf(full), torch.full_like(full, invalid_value), full)
        return full

    def _update_observed_w_lane_features(self, observation: torch.Tensor) -> None:
        """结合观测到的 w_lane 与全局 Δs 构建网络输入特征。
        输出形状: (B, M, K, 4) = [dx, dy, angle_local, Δs]
        """
        obs_gen = self.observation_generator
        if (
            self.w_lane_goal_distances_full is None
            or not hasattr(obs_gen, "w_lanes_ids")
            or obs_gen.w_lanes_ids is None
        ):
            self.w_lanes_local_with_goal_distances = None
            return
        
        # 从 observation 中临时解包 w_lanes_local
        _, _, w_lanes_local, _ = ObservationGenerator.unpack_observation_components(
            observation,
            obs_gen.local_state_dim,
            obs_gen.num_neighbors,
            obs_gen.neighbor_feature_dim,
            obs_gen.num_w_lanes,
            obs_gen.w_lane_feature_dim,
            obs_gen.num_w_boundaries,
            obs_gen.boundary_feature_dim,
        )
        w_lanes_local = w_lanes_local.to(self.device)  # (B, M, K, 2) = [dx, dy]
        B, M, K, _ = w_lanes_local.shape
        # Reshape w_lanes_ids to match the actual K dimension from w_lanes_local
        # Calculate actual K from w_lanes_ids size to handle potential mismatches
        total_elements = obs_gen.w_lanes_ids.numel()
        actual_K = total_elements // (B * M)
        if total_elements != B * M * actual_K:
            raise ValueError(f"Cannot reshape w_lanes_ids: {obs_gen.w_lanes_ids.shape} to (B={B}, M={M}, K=?)")
        w_lane_ids = obs_gen.w_lanes_ids.view(B, M, actual_K).to(self.device)  # (B, M, actual_K)
        # Trim or pad to match w_lanes_local's K dimension
        if actual_K > K:
            w_lane_ids = w_lane_ids[:, :, :K]
        elif actual_K < K:
            # Pad with invalid marker to match K
            invalid_marker = int(self.config.get('simulator', {}).get('observation', {}).get('INVALID_MARKER', -1))
            pad = torch.full((B, M, K - actual_K), invalid_marker, dtype=w_lane_ids.dtype, device=w_lane_ids.device)
            w_lane_ids = torch.cat([w_lane_ids, pad], dim=2)
        
        # 获取 w_lane 的世界坐标特征 (x, y, angle)
        w_lane_features_world = self.path_planner.get_w_lane_features_by_id(w_lane_ids)  # (B, M, K, 3)
        angles_world = w_lane_features_world[..., 2]  # (B, M, K) - 提取 angle
        
        # 将 angle 转换到局部坐标系（相对于 ego 的 yaw）
        ego_yaw = self.agents_state[..., 2]  # (B, M) - ego 的 yaw
        angles_local = angles_world - ego_yaw.unsqueeze(-1)  # (B, M, K)
        # 归一化到 [-π, π]
        angles_local = torch.atan2(torch.sin(angles_local), torch.cos(angles_local))
        
        # 获取 Δs
        idx = self.path_planner.map_w_lane_ids_to_indices(w_lane_ids)
        invalid_value = float(self.path_planner.INVALID_MARKER)
        delta = torch.full((B, M, K), invalid_value, dtype=w_lanes_local.dtype, device=self.device)
        
        # 检查索引边界：确保索引在有效范围内，避免越界
        total_w_lanes = self.w_lane_goal_distances_full.shape[-1]
        valid = (idx >= 0) & (idx < total_w_lanes)
        
        if valid.any():
            batch_idx = torch.arange(B, device=self.device).view(B, 1, 1).expand_as(idx)
            agent_idx = torch.arange(M, device=self.device).view(1, M, 1).expand_as(idx)
            delta_vals = self.w_lane_goal_distances_full[batch_idx[valid], agent_idx[valid], idx[valid]]
            delta[valid] = delta_vals
        
        # 拼接: [dx, dy, angle_local, Δs] -> (B, M, K, 4)
        self.w_lanes_local_with_goal_distances = torch.cat([
            w_lanes_local,  # (B, M, K, 2)
            angles_local.unsqueeze(-1),  # (B, M, K, 1)
            delta.unsqueeze(-1)  # (B, M, K, 1)
        ], dim=-1)

    def _sample_waypoint_ids_for_mask(self) -> None:
        """为每个 agent 从路径中随机抽取 0-3 个有效 w_lane id."""
        if self.agents_path_plans is None or self.agents_path_plans.numel() == 0:
            self.sampled_waypoint_ids = None
            return
        path_ids = self.agents_path_plans
        invalid_marker = self.path_planner.INVALID_w_lane_id_MARKER
        valid_mask = path_ids != invalid_marker

        B, M, L = path_ids.shape
        valid_counts = valid_mask.sum(dim=-1)  # (B, M) 每个 agent 的有效路径点数量
        max_samples_per_agent = torch.clamp(valid_counts, max=3)  # (B, M) 每个 agent 最多采样 min(3, 有效数量)
        
        rand = torch.rand((B, M, L), device=self.device)
        rand = rand.masked_fill(~valid_mask, -1.0)
        perm = torch.argsort(rand, dim=-1, descending=True)
        
        # 确保输出形状始终是 (B, M, 3)，即使 L < 3
        # 先取前 min(3, L) 个，然后填充到 3 个位置
        num_samples = min(3, L)
        top_indices = perm[..., :num_samples]  # (B, M, num_samples)
        sampled = torch.gather(path_ids, 2, top_indices)  # (B, M, num_samples)
        selected_valid = torch.gather(valid_mask, 2, top_indices)  # (B, M, num_samples)
        
        # 如果 L < 3，需要填充到 (B, M, 3) 形状
        if L < 3:
            # 创建填充值（无效标记）
            padding_size = 3 - L
            padding_ids = torch.full(
                (B, M, padding_size),
                invalid_marker,
                dtype=sampled.dtype,
                device=self.device,
            )
            padding_valid = torch.zeros(
                (B, M, padding_size),
                dtype=torch.bool,
                device=self.device,
            )
            # 拼接原始数据和填充数据
            sampled = torch.cat([sampled, padding_ids], dim=-1)  # (B, M, 3)
            selected_valid = torch.cat([selected_valid, padding_valid], dim=-1)  # (B, M, 3)
        
        # 对于每个 agent，如果有效点数量少于 3，将超出的位置标记为无效
        position_idx = torch.arange(3, device=self.device).view(1, 1, -1).expand(B, M, -1)
        count_mask = position_idx < max_samples_per_agent.unsqueeze(-1)  # (B, M, 3)
        final_valid = selected_valid & count_mask
        self.sampled_waypoint_ids = torch.where(
            final_valid,
            sampled,
            torch.full_like(sampled, invalid_marker))

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
        
        # 只返回路径上的 w_lane，而不是所有观测到的 w_lane
        # 直接计算路径上的 w_lane 的局部坐标和特征
        w_lanes_path_only = None
        if sim.agents_path_plans is not None:
            try:
                # 获取路径上的 w_lane_id
                path_ids = sim.agents_path_plans[b, m]  # (L,)
                invalid_marker = sim.path_planner.INVALID_w_lane_id_MARKER
                valid_path_mask = path_ids != invalid_marker
                valid_path_ids = path_ids[valid_path_mask]  # 路径上的有效 w_lane_id
                
                if valid_path_ids.numel() > 0:
                    # 获取路径上的 w_lane 的世界坐标特征 (x, y, angle)
                    path_features_world = sim.path_planner.get_w_lane_features_by_id(
                        valid_path_ids.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)  # (L_valid, 3)
                    
                    # 获取 ego 状态
                    ego_state = agents_state[b, m]  # (7,)
                    ego_pos = ego_state[:2]  # (2,)
                    ego_yaw = ego_state[2]  # scalar
                    
                    # 转换到局部坐标系
                    cos_yaw = torch.cos(ego_yaw)
                    sin_yaw = torch.sin(ego_yaw)
                    rot_matrix = torch.stack([
                        torch.stack([cos_yaw, -sin_yaw], dim=0),
                        torch.stack([sin_yaw, cos_yaw], dim=0)
                    ], dim=0)  # (2, 2)
                    
                    # 计算局部坐标 (dx, dy)
                    world_pos = path_features_world[:, :2]  # (L_valid, 2)
                    rel_pos = world_pos - ego_pos.unsqueeze(0)  # (L_valid, 2)
                    local_pos = rel_pos @ rot_matrix.T  # (L_valid, 2)
                    
                    # 计算局部角度
                    angles_world = path_features_world[:, 2]  # (L_valid,)
                    angles_local = angles_world - ego_yaw
                    angles_local = torch.atan2(torch.sin(angles_local), torch.cos(angles_local))
                    
                    # 获取 Δs
                    if sim.agents_path_plan_goal_distances is not None:
                        path_deltas = sim.agents_path_plan_goal_distances[b, m][valid_path_mask]  # (L_valid,)
                    else:
                        invalid_value = float(sim.path_planner.INVALID_MARKER)
                        path_deltas = torch.full((valid_path_ids.shape[0],), invalid_value, 
                                                device=sim.device, dtype=path_features_world.dtype)
                    
                    # 拼接: [dx, dy, angle_local, Δs] -> (L_valid, 4)
                    w_lanes_path_only = torch.cat([
                        local_pos,  # (L_valid, 2)
                        angles_local.unsqueeze(-1),  # (L_valid, 1)
                        path_deltas.unsqueeze(-1)  # (L_valid, 1)
                    ], dim=-1)
            except Exception as e:
                print(f"获取路径上的 w_lane 失败: {e}")
                import traceback
                traceback.print_exc()
                w_lanes_path_only = None
        
        return neighbors_local_tmp[b, m], w_lanes_path_only, w_boundaries_local_tmp[b, m]

    # 定义step回调：按 W 键时执行一步仿真并返回最新状态
    def step_callback(b_idx: int, m_idx: int):
        with torch.no_grad():
            B, M = sim.agents_state.shape[:2]
            actions = torch.full((B, M, 1), 7, dtype=torch.long, device=sim.device)
            observation, reward, done = sim.step(actions)
            sim._last_action = actions.squeeze(-1).clone()
        print("已执行一步仿真。")
        sampled_features = None
        sampled_ids_cpu = None
        if sim.sampled_waypoint_ids is not None:
            try:
                sampled_features = sim.path_planner.get_w_lane_features_by_id(sim.sampled_waypoint_ids).detach().cpu()
            except Exception:
                sampled_features = None
            try:
                sampled_ids_cpu = sim.sampled_waypoint_ids.detach().cpu()
            except Exception:
                sampled_ids_cpu = None
        path_features = sim.get_path_plan_features_with_goal_distances()
        if path_features is not None:
            path_features = path_features.detach().cpu()
        return (
            sim.agents_state,
            path_features,
            sim.goal_positions,
            sim.goal_radius_tensor,
            sim.cumulative_done_mask,
            sampled_features,
            sampled_ids_cpu,
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
            path_ids = sim.agents_path_plans[b, m]
            invalid_marker_tensor = torch.tensor(
                sim.path_planner.INVALID_w_lane_id_MARKER,
                dtype=path_ids.dtype,
                device=path_ids.device,
            )
            valid_mask = path_ids != invalid_marker_tensor
            if valid_mask.any():
                path_features = sim.path_planner.get_w_lane_features_by_id(
                    path_ids.unsqueeze(0).unsqueeze(0)
                ).squeeze(0).squeeze(0)
                last_valid_idx = torch.nonzero(valid_mask, as_tuple=False)[-1].item()
                last_point = path_features[last_valid_idx, :2].detach().cpu().numpy()
                goal_point = (float(last_point[0]), float(last_point[1]))
        except Exception:
            goal_point = None

        if goal_point is not None:
            lines.append(("Goal", f"{goal_point[0]:.2f}, {goal_point[1]:.2f}"))
        else:
            lines.append(("Goal", "N/A"))
        if sim.sampled_waypoint_ids is not None:
            try:
                sampled_ids_tensor = sim.sampled_waypoint_ids[b, m]
                if torch.is_tensor(sampled_ids_tensor):
                    ids_list = sampled_ids_tensor.detach().cpu().tolist()
                else:
                    ids_list = list(sampled_ids_tensor)
                valid_ids = [
                    str(int(v))
                    for v in ids_list
                    if isinstance(v, (int, float))
                    and int(v) != int(sim.path_planner.INVALID_w_lane_id_MARKER)
                ]
                lines.append(("Sampled Waypoints", ", ".join(valid_ids) if valid_ids else "None"))
            except Exception:
                lines.append(("Sampled Waypoints", "N/A"))

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
    initial_sampled_features = None
    initial_sampled_ids = None
    if sim.sampled_waypoint_ids is not None:
        try:
            initial_sampled_features = sim.path_planner.get_w_lane_features_by_id(sim.sampled_waypoint_ids).detach().cpu()
        except Exception:
            initial_sampled_features = None
        try:
            initial_sampled_ids = sim.sampled_waypoint_ids.detach().cpu()
        except Exception:
            initial_sampled_ids = None
    path_features = sim.get_path_plan_features_with_goal_distances()
    if path_features is not None:
        path_features = path_features.detach().cpu()
    visualize_path_planning(
        agents_state=sim.agents_state,
        agents_path_plans=path_features,
        quads_vertices=sim.road_network.left_boundaries,
        batch_idx=0,
        invalid_marker_value=float(sim.path_planner.INVALID_MARKER),
        horizon=sim.observation_generator.horizon,
        observation_callback=observation_callback,
        step_callback=step_callback,
        info_callback=info_callback,
        agents_start_quad_ids=sim.agents_start_quad_ids,
        agents_goal_quad_ids=sim.agents_goal_quad_ids,
        goal_positions=sim.goal_positions,
        goal_radii=sim.goal_radius_tensor,
        done_mask=sim.cumulative_done_mask,
        sampled_waypoint_features=initial_sampled_features,
        sampled_waypoint_ids=initial_sampled_ids,
    )
    print("退出可视化。")

import torch
import os
import sys
import time
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
        self.dynamics_model = KinematicBicycleModel(config, self.device,self.world_initializer.vehicle_params)

        # 7. 初始化观测生成器
        # observation.py 负责为每个自车生成局部观测
        self.observation_generator = ObservationGenerator(self.road_network, config, self.device, self.spatial_hash)

        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.device)

        # 9. 初始化路径规划器（直接复用 RoadNetwork 数据）
        self.path_planner = PathPlanner(device=str(self.device).split(':')[0], road_network=self.road_network)

        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        self.agents_goal_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的目标quad_id
        self.agents_path_plans: Optional[torch.Tensor] = None     # 存储所有智能体的路径规划（世界坐标）
        self.agents_path_plans_local: Optional[torch.Tensor] = None  # 存储所有智能体的路径规划（局部坐标）

    def reset(self) -> torch.Tensor:
        """
        重置所有环境，并返回所有智能体的初始观测。
        Returns:
            torch.Tensor: 一批初始观测, 形状为 (B, M, obs_dim)。
        """
        print("Resetting simulator environments...")
        # 重置动力学模型的状态变量，避免不同episode之间的tensor大小不匹配
        self.dynamics_model.reset_control_state()
        print("Reset dynamics model state - cleared for fresh initialization")

        # 使用 WorldInitializer 来生成一批新的世界状态，包括起始quad_id
        self.agents_state, _, self.agents_start_quad_ids = self.world_initializer.initialize_world(self.num_envs)
        
        # 重置reward风格参数
        self.reward_calculator.reset_episode()

        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        
        # 重置done状态
        self.last_done = None

        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation = self.observation_generator.generate(self.agents_state)
        print(initial_observation.shape)
        print("Initial observation generated")

        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        self._initialize_path_planning()
        print("Path planning initialized")
        print(f"Reset complete. World state shape: {self.agents_state.shape}")
        self.stop_lines = torch.zeros((self.num_envs, self.world_initializer.max_agents,20), dtype=torch.int32, device=self.device)
        
        return initial_observation
    
    def step(self, actions: torch.Tensor, debug_collision: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        让所有环境向前步进一个时间步。所有智能体都根据actions更新。
        Args:
            actions (torch.Tensor): 形状为 (B, M, action) 的动作张量。
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
        if hasattr(self, 'last_done') and self.last_done is not None:
            not_done_mask = ~self.last_done
        else:
            not_done_mask = torch.ones_like(active_mask, dtype=torch.bool)
        effective_mask = active_mask & not_done_mask  # 仅这些需要物理更新
        
        # 构造全批次输入 (B*M, 4) 和 (B*M,) 的动作索引
        Bsz, Msz, _S = self.agents_state.shape
        states_flat = self.agents_state[..., :4].contiguous().view(Bsz * Msz, 4)
        # 规整actions为 (B, M)
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions_idx = actions.squeeze(-1).long()
        elif actions.ndim == 2:
            actions_idx = actions.long()
        else:
            actions_idx = actions.view(Bsz, Msz).long()

        actions_flat = actions_idx.contiguous().view(Bsz * Msz)
        # 调用动力学（全批次大小恒定），再把无效位置用旧状态覆盖
        new_states_flat = self.dynamics_model.step(states_flat, actions_flat, self.dt)  # (B*M, 4)
        new_states = new_states_flat.view(Bsz, Msz, 4)
        keep_old = ~effective_mask
        self.agents_state[..., :4] = torch.where(keep_old.unsqueeze(-1), self.agents_state[..., :4], new_states)

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
        # 临时将done车辆的状态设置为无效，避免它们参与碰撞检测
        original_states_t0 = states_t0.clone()
        original_states_t1 = self.agents_state.clone()
        
        if hasattr(self, 'last_done') and self.last_done is not None:
            # 将done车辆的状态设置为无效（active=0）
            states_t0_for_collision = states_t0.clone()
            states_t1_for_collision = self.agents_state.clone()
            states_t0_for_collision[..., 6] = torch.where(self.last_done, 0.0, states_t0_for_collision[..., 6])
            states_t1_for_collision[..., 6] = torch.where(self.last_done, 0.0, states_t1_for_collision[..., 6])
        else:
            states_t0_for_collision = states_t0
            states_t1_for_collision = self.agents_state
        
        collision_check_result = self.collision_checker.check(
            states_t0_for_collision, states_t1_for_collision, debug=debug_collision, debug_env_idx=0
        )
        all_collisions = collision_check_result

        # 4. 计算Frenet坐标信息
        vehicle_positions = self.agents_state[..., :2]  # (B, M, 2) - x, y
        vehicle_headings = self.agents_state[..., 2]    # (B, M) - heading
        d, theta_f = self.road_network.calculate_frenet_coordinates(vehicle_positions, vehicle_headings, self.spatial_hash)

        # 5. 更新path_plans的局部坐标
        self._update_path_plans_local()
        
        # 5.5. 检查并移除已到达的路径点
        self._check_and_remove_reached_waypoints()

        # 6. 生成新的观测（排除done的车辆）
        # 临时将done车辆的状态设置为无效，避免它们参与观测生成
        if hasattr(self, 'last_done') and self.last_done is not None:
            # 将done车辆的状态设置为无效（active=0）
            agents_state_for_obs = self.agents_state.clone()
            agents_state_for_obs[..., 6] = torch.where(self.last_done, 0.0, agents_state_for_obs[..., 6])
        else:
            agents_state_for_obs = self.agents_state
        
        observation = self.observation_generator.generate(agents_state_for_obs)

        # 7. 计算奖励（传入Frenet坐标和动作）
        reward, goal_reached = self._calculate_reward(all_collisions, offroad_mask, d, theta_f, actions)

        # 8. 检查是否结束（包含目标到达判断）
        done = all_collisions|offroad_mask|goal_reached
        
        # 保存done状态供下次step使用（累积done状态，一旦done就保持done）
        if hasattr(self, 'last_done') and self.last_done is not None:
            self.last_done = self.last_done | done
        else:
            self.last_done = done.clone()

        return observation, reward, done
    
    def _calculate_reward(self, all_collisions: torch.Tensor, offroad_mask: torch.Tensor, d: torch.Tensor, theta_f: torch.Tensor, actions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为所有智能体计算奖励。
        Args:
            all_collisions (torch.Tensor): 碰撞状态 (B, M)
            offroad_mask (torch.Tensor): 离路状态 (B, M)
            d (torch.Tensor): Frenet横向距离 (B, M)
            theta_f (torch.Tensor): Frenet角度误差 (B, M)
            actions (torch.Tensor): 动作索引 (B, M)，用于直接获取jerk值
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (奖励值 (B, M), 目标到达标志 (B, M))
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
            if hasattr(self, 'last_done') and self.last_done is not None:
                not_done_mask = ~self.last_done
            else:
                not_done_mask = torch.ones_like(active_mask, dtype=torch.bool)
            effective_mask = active_mask & not_done_mask
            
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
                # 规整为 (B, M) 的索引
                if actions.ndim == 3 and actions.shape[-1] == 1:
                    actions_idx = actions.squeeze(-1).long()
                elif actions.ndim == 2:
                    actions_idx = actions.long()
                else:
                    # 兜底：兼容 (N,1) after masking 的情况
                    actions_idx = actions.view(B, M).long()
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

        # 计算目标和初始化goal_reached，速度很快
        # print(self.agents_goal_quad_ids)

        # goal_positions = self.path_planner.get_quad_centers(self.agents_goal_quad_ids)
        goal_positions = self.goal_positions

        waypoint_reached = torch.ones((B, M), dtype=torch.bool, device=self.device)

        # 调用奖励计算器，这个速度很快
        reward, goal_reached = self.reward_calculator.calculate(
            extended_state,
            all_collisions,
            offroad_mask,
            dt=self.dt,
            goal_positions=goal_positions,
            waypoint_reached=waypoint_reached,
        )
        
        # 过滤掉非active或已done车辆的奖励（与动力学一致的有效掩码）
        active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
        if hasattr(self, 'last_done') and self.last_done is not None:
            not_done_mask = ~self.last_done
        else:
            not_done_mask = torch.ones_like(active_mask, dtype=torch.bool)
        effective_mask = active_mask & not_done_mask
        reward = reward * effective_mask.float()  # 非有效车辆的奖励设为0，等价于“done后奖励不再更新”

        self.extend_state = extended_state # 用于传入网络
        return reward, goal_reached
    
    def _initialize_path_planning(self):
        """
        为所有智能体初始化路径规划：
        1. 为每个激活的智能体随机分配一个目标quad_id
        2. 使用plan_path批量生成从起始位置到目标的路径规划
        """
        prepare_time = time.time()
        if self.agents_state is None:
            return
        # 基本尺寸与激活掩码（GPU）
        B, M, _ = self.agents_state.shape
        active_mask = self.agents_state[..., 6] > 0.5
        # 起点与目标（纯GPU、极简随机目标，仅在激活位赋值）
        if not hasattr(self, 'agents_start_quad_ids') or self.agents_start_quad_ids is None:
            print("Warning: No start quad IDs available for path planning")
            return
        start_i32 = self.agents_start_quad_ids.to(dtype=torch.int32, device=self.device)
        goal_i32 = torch.full_like(start_i32, -1, dtype=torch.int32, device=self.device)
        # 仅对"激活且起点有效"的样本，从最近K个quads中随机选一个作为终点
        valid_mask = (active_mask) & (start_i32 >= 0)
        if valid_mask.any():
            rand_vals = torch.randint(0, self.road_network.num_quads, (int(valid_mask.sum().item()),), device=self.device, dtype=torch.int32)
            goal_i32 = goal_i32.masked_scatter(valid_mask, rand_vals)
        # 一次性调用 planner（形状为 (B,M,1)）
        start_3d = start_i32.unsqueeze(-1)
        goal_3d = goal_i32.unsqueeze(-1)
        prepare_time_done = time.time()
        print(f"prepare_time_done: {prepare_time_done-prepare_time:.4f}s")
        
        # 只对有效的起点和终点进行路径规划
        valid_planning_mask = (start_i32 >= 0) & (goal_i32 >= 0) & active_mask
        if valid_planning_mask.any():
            # 创建临时的起点和终点，将无效的设置为-1
            start_3d_valid = torch.where(valid_planning_mask.unsqueeze(-1), start_3d, torch.tensor([[-1]], device=self.device))
            goal_3d_valid = torch.where(valid_planning_mask.unsqueeze(-1), goal_3d, torch.tensor([[-1]], device=self.device))
            path_plans = self.path_planner.plan_path(start_3d_valid, goal_3d_valid)  # 形状 (B,M,L,2)
        else:
            # 如果没有有效的路径规划，创建空的路径
            path_plans = torch.full((B, M, 128, 2), -1.0, dtype=torch.float32, device=self.device)
        # 存储结果
        self.agents_goal_quad_ids = goal_i32
        self.agents_path_plans = path_plans
        
        # 由于收敛性问题，需要将agents_path_plans的路径逐渐放长
        # 使用动态的path_observation_length，可以通过外部设置
        if not hasattr(self, 'path_observation_length'):
            self.path_observation_length = 2  # 初始值
        path_observation_length = self.path_observation_length
        # 创建全-1的中间tensor，保持原始长度128
        B, M, _, _ = self.agents_path_plans.shape
        filtered_paths = torch.full((B, M, 128, 2), -1.0, dtype=torch.float32, device=self.device)
        # 只将前path_observation_length个位置赋予有效值
        filtered_paths[:, :, :path_observation_length, :] = self.agents_path_plans[:, :, :path_observation_length, :]
        self.agents_path_plans = filtered_paths
        
        # 设置goal_positions为路径的第二个点（批量操作）
        B, M, L, _ = self.agents_path_plans.shape
        # 批量找到所有有效点
        valid_mask = (self.agents_path_plans[..., 0] != -1) & (self.agents_path_plans[..., 1] != -1)  # (B, M, L)
        # 对每个路径，找到前两个有效点的位置
        ar = torch.arange(L, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, M, -1)
        # 将无效位置排到后面
        big = L + 1000
        order_score = torch.where(valid_mask, ar, ar + big)
        order = torch.argsort(order_score, dim=2, stable=True)
        
        # 获取第二个有效点作为目标位置
        second_valid_indices = order[:, :, 1]  # (B, M)
        # 检查是否有第二个有效点
        has_second = valid_mask.gather(2, second_valid_indices.unsqueeze(-1)).squeeze(-1)  # (B, M)
        
        # 如果没有第二个有效点，使用第一个有效点
        first_valid_indices = order[:, :, 0]  # (B, M)
        has_first = valid_mask.gather(2, first_valid_indices.unsqueeze(-1)).squeeze(-1)  # (B, M)
        
        # 选择目标索引：优先第二个，其次第一个，最后使用当前位置
        target_indices = torch.where(
            has_second, 
            second_valid_indices,
            torch.where(has_first, first_valid_indices, torch.zeros_like(first_valid_indices))
        )
        
        # 批量获取目标位置
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, M)
        agent_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(B, -1)
        
        # 获取目标位置
        self.goal_positions = self.agents_path_plans[batch_indices, agent_indices, target_indices]
        
        # 对于没有有效点的智能体，使用当前位置
        no_valid_points = ~(has_second | has_first)
        if no_valid_points.any():
            # 正确索引：no_valid_points是(B,M)的布尔mask
            current_positions = self.agents_state[..., :2]  # (B, M, 2)
            self.goal_positions[no_valid_points] = current_positions[no_valid_points]

        # 初始化path_plans的局部坐标版本
        self._update_path_plans_local()
    
    def set_path_observation_length(self, length: int):
        """
        动态设置路径观察长度
        
        Args:
            length (int): 新的路径观察长度，范围[2, 128]
        """
        length = max(2, min(128, length))  # 限制在合理范围内
        self.path_observation_length = length
        print(f"路径观察长度已更新为: {self.path_observation_length}")
        
        # 如果当前有路径规划，需要重新应用新的长度
        if hasattr(self, 'agents_path_plans') and self.agents_path_plans is not None:
            self._apply_path_observation_length()
    
    def _apply_path_observation_length(self):
        """
        应用当前的path_observation_length到现有的路径规划
        """
        if not hasattr(self, 'agents_path_plans') or self.agents_path_plans is None:
            return
            
        B, M, _, _ = self.agents_path_plans.shape
        filtered_paths = torch.full((B, M, 128, 2), -1.0, dtype=torch.float32, device=self.device)
        # 只将前path_observation_length个位置赋予有效值
        filtered_paths[:, :, :self.path_observation_length, :] = self.agents_path_plans[:, :, :self.path_observation_length, :]
        self.agents_path_plans = filtered_paths
        
        # 重新更新局部坐标
        self._update_path_plans_local()

    def _update_path_plans_local(self):
        """
        将path_plans从世界坐标转换到每个智能体的局部坐标系。
        使用observation.py中_world_to_ego_centric的原理进行坐标转换。
        同时将-1,-1坐标转换为0，方便后续网络输入。
        """
        if self.agents_path_plans is None or self.agents_state is None:
            return
        
        B, M, L, _ = self.agents_path_plans.shape
        ego_states = self.agents_state  # (B, M, 7)
        
        # 获取ego车辆的位置和朝向
        ego_pos = ego_states[..., :2]  # (B, M, 2)
        ego_yaw = ego_states[..., 2]   # (B, M)
        
        # 计算旋转矩阵
        cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
        
        # 使用标准2D旋转矩阵（车左边为正）
        rot_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1), 
            torch.stack([sin_yaw, cos_yaw], dim=-1)
        ], dim=-2)  # (B, M, 2, 2)
        
        # 创建path_plans_local张量
        path_plans_local = torch.zeros_like(self.agents_path_plans)
        
        # 向量化bmm操作：将 (B, M) 批次展平为 (B*M)，执行bmm，然后重塑
        def batch_rotate_path_plans(path_plans_world, ego_pos, rot_matrix):
            # path_plans_world: (B, M, L, 2), ego_pos: (B, M, 2), rot_matrix: (B, M, 2, 2)
            B, M, L, D = path_plans_world.shape
            
            # 创建有效坐标掩码（排除-1,-1坐标）
            valid_mask = (path_plans_world[..., 0] != -1) & (path_plans_world[..., 1] != -1)  # (B, M, L)
            
            # 计算相对位置
            rel_pos = path_plans_world - ego_pos.unsqueeze(2)  # (B, M, L, 2)
            
            # 展平为 (B*M, L, 2) 和 (B*M, 2, 2)
            rel_pos_flat = rel_pos.view(B*M, L, D)
            rot_matrix_flat = rot_matrix.view(B*M, D, D)
            
            # 执行批量矩阵乘法
            rotated_flat = torch.bmm(rel_pos_flat, rot_matrix_flat)  # (B*M, L, 2)
            
            # 重塑回原始形状
            rotated = rotated_flat.view(B, M, L, D)
            
            # 将无效坐标（-1,-1）设置为0
            rotated[~valid_mask] = 0.0
            
            return rotated
        
        # 执行坐标转换
        path_plans_local = batch_rotate_path_plans(self.agents_path_plans, ego_pos, rot_matrix)
        
        # 存储转换后的局部坐标
        self.agents_path_plans_local = path_plans_local
        
    def _check_and_remove_reached_waypoints(self):
        """
        检查并移除已到达的路径点。
        当车辆与路径规划中的某个点的距离小于1米时，将该点从路径规划中移除（设置为-1, -1）。
        使用向量化操作提高效率。
        """
        if self.agents_path_plans_local is None or self.agents_state is None:
            return
        
        B, M, L, _ = self.agents_path_plans_local.shape
        
        # 获取激活掩码
        active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
        
        if not active_mask.any():
            return  # 没有激活的车辆
        
        # 计算所有车辆到所有路径点的距离（向量化）
        # agents_path_plans_local: (B, M, L, 2)
        # 计算每个路径点的模长（距离）
        distances = torch.norm(self.agents_path_plans_local, dim=-1)  # (B, M, L)
        
        # 找到有效的路径点（不是-1, -1）且距离小于1米的点
        valid_waypoints = (self.agents_path_plans_local[..., 0] != -1) & (self.agents_path_plans_local[..., 1] != -1)  # (B, M, L)
        reached_waypoints = valid_waypoints & (distances < 1.0)  # (B, M, L)
        
        # 只对激活的车辆处理
        reached_waypoints = reached_waypoints & active_mask.unsqueeze(-1)  # (B, M, L)
        
        if reached_waypoints.any():
            # 将到达的路径点设置为-1, -1
            invalid_fill = torch.full_like(self.agents_path_plans, -1.0)
            self.agents_path_plans = torch.where(reached_waypoints.unsqueeze(-1), invalid_fill, self.agents_path_plans)
            self.agents_path_plans_local = torch.where(reached_waypoints.unsqueeze(-1), invalid_fill, self.agents_path_plans_local)
            
            # 重新更新局部坐标（因为agents_path_plans可能已经改变）
            self._update_path_plans_local()

if __name__ == "__main__":
    import json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    sim = TeraflowSimulator(config=cfg, device=torch.device(device))
    print("Simulator initialized with dt=", sim.dt, "num_envs=", sim.num_envs, "map=", sim.map_path)
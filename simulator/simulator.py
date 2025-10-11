from multiprocessing.spawn import prepare
from sympy import N
import torch
import yaml
import os
import sys
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 依赖于 spatial_hash
# 添加utils目录到路径
# 添加simulator目录到路径
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
training_dir = os.path.join(parent_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
from spatial_hash import SpatialHash
from road import RoadNetwork
from offroad import OffroadChecker
from collision import CollisionChecker
from goals import PathPlanner
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
        Args:
            config (Dict): 包含所有模拟器配置的字典。
                - num_envs (int): 并行环境的数量 (batch_size)。
                - map_path (str): 预处理后的地图文件路径 (.json)。
                - device (str): 计算设备 ('cpu' 或 'cuda')。
                - sim_dt (float): 模拟时间步长 (秒)。
                - ... 其他配置，如车辆参数等
        """
        self.config = config
        self.device = device
        simulator_config = config.get('simulator')
        self.num_envs = simulator_config['num_envs']
        self.dt = simulator_config['sim_dt']
        self.map_path = simulator_config['map_path']

        # 1. 加载地图网络
        # road.py 中的 RoadNetwork 类负责解析地图文件并提供查询接口
        self.road_network = RoadNetwork(self.map_path, self.device)

        # 2. 初始化共享的空间哈希
        # 使用地图边界来定义哈希网格的范围
        all_verts = self.road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 使用一个固定的 cell_size, 也可以从 config 读取
        hash_config = simulator_config['hash']
        cell_size = hash_config['hash_cell_size']
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)

        # 3. 初始化车辆动力学模型
        # dynamics.py 中的 KinematicBicycleModel 负责根据动作更新车辆状态
        self.dynamics_model = KinematicBicycleModel(config, self.device)
        # TODO: 把动力学里面随机的参数加入conditioning传入网络

        # 4. 初始化离路检测器 (传入共享的哈希对象)
        self.offroad_checker = OffroadChecker(self.road_network, self.spatial_hash)

        # 5. 初始化碰撞检测器 (传入共享的哈希对象)
        # collision.py 负责检测智能体之间的碰撞
        self.collision_checker = CollisionChecker(config, self.spatial_hash)

        # 6. 初始化世界状态管理器
        # world_init.py 负责在 reset 时初始化场景
        self.world_initializer = WorldInitializer(self.road_network, self.offroad_checker, self.collision_checker, config)

        # 7. 初始化观测生成器
        # observation.py 负责为每个自车生成局部观测
        obs_config = simulator_config['observation']
        self.observation_generator = ObservationGenerator(self.road_network, obs_config, self.device, self.spatial_hash)

        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.device)

        # 9. 初始化路径规划器
        # PathPlanner现在会自动加载所需的数据
        self.path_planner = PathPlanner(map_path=self.map_path, device=self.device)

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
        path_observation_length = 2
        # 创建全-1的中间tensor，保持原始长度128
        B, M, _, _ = self.agents_path_plans.shape
        filtered_paths = torch.full((B, M, 128, 2), -1.0, dtype=torch.float32, device=self.device)
        # 只将前两个位置赋予有效值
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

if __name__ == '__main__':
    # 这是一个简单的使用示例，用于测试模拟器的基本功能
    # 从配置文件读取配置
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    from matplotlib.patches import Polygon, Circle
    from matplotlib.collections import PatchCollection
    
    config_path = 'configs/default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    simulator = TeraflowSimulator(config=config, device=device)

    initial_obs = simulator.reset()
    print(f"Initial observation batch shape: {initial_obs.shape}")

    # 可视化道路网络和智能体位置（与goals.py绘制风格保持一致）
    print("\n=== 可视化道路网络和智能体位置 ===")
    # 获取道路网络的四边形顶点 这里是测试road.py
    quads_vertices = simulator.road_network.quads_vertices  # (num_quads, 4, 2)
    quads_vertices_np = quads_vertices.cpu().numpy()
    # 获取智能体状态 这里已经测试过world_initializer.py
    agents_state_np = simulator.agents_state.cpu().numpy()  # (B, M, 7)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    # 方法1: 使用PatchCollection进行批量绘制（最快）
    patches = []
    # 批量创建Polygon对象
    for i in range(len(quads_vertices_np)):
        vertices = quads_vertices_np[i]  # (4, 2)
        polygon = Polygon(vertices, closed=True)
        patches.append(polygon)
    p = PatchCollection(patches, alpha=0.2, facecolor='lightblue', edgecolor='black', linewidth=0.1)
    # 一次性添加所有quads到图形
    ax.add_collection(p)

    # 构建可更新的智能体绘制（仅显示第一个环境）
    def build_agent_artists():
        ax_agents = []
        agents_state_np_local = simulator.agents_state.cpu().numpy()
        active_mask_local = agents_state_np_local[0, :, 6] > 0.5
        active_indices_local = np.where(active_mask_local)[0]
        if len(active_indices_local) == 0:
            return ax_agents, active_indices_local
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        import math
        for i, agent_idx in enumerate(active_indices_local):
            x, y, yaw, speed, length, width, active = agents_state_np_local[0, agent_idx]
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            # 智能体矩形的四个角点 (相对于中心)
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
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            agent_corners = corners @ rotation_matrix.T + np.array([x, y])
            poly = Polygon(agent_corners, closed=True)
            color = colors[i % len(colors)]
            ax.add_patch(poly)
            poly.set_facecolor(color)
            poly.set_alpha(0.8)
            poly.set_edgecolor('black')
            poly.set_linewidth(2)
            # 仅为第一个激活agent显示标签与速度文本
            if i == 0:
                label = f'Agent {agent_idx}'
                txt = ax.text(x, y, label, ha='center', va='center', fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                              weight='bold')
            else:
                txt = None
            speed_vec = 3.0
            arr = ax.arrow(x, y, speed_vec * cos_yaw, speed_vec * sin_yaw,
                           head_width=0.5, head_length=0.5, fc=color, ec=color,
                           alpha=0.8, zorder=5, linewidth=2)
            # 仅第一个激活agent显示速度文本
            if i == 0:
                info = ax.text(x, y + half_width + 1, f'v={speed:.1f}m/s', ha='center', va='bottom',
                               fontsize=8, color=color, weight='bold')
            else:
                info = None
            ax_agents.append((agent_idx, poly, txt, arr, info, color))
        return ax_agents, active_indices_local

    agent_artists, active_indices = build_agent_artists()

    # 构建策略网络与初始特征（延迟导入避免循环依赖）
    from ddppo import decompose_observation, build_network_features
    from network import create_network
    import json as _json

    config_ns = _json.loads(_json.dumps(config), object_hook=lambda d: SimpleNamespace(**d))
    model = create_network(config=config_ns, network_type="independent").to(device)
    model.eval()
    with torch.no_grad():
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_obs, config_ns)
        features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            simulator.agents_path_plans,
            simulator.stop_lines,
            simulator.reward_calculator.sampled_params,
            config_ns,
        )

    # 在主图上绘制局部要素（第一个环境第一个激活agent）
    overlay_artists = []
    def clear_overlays():
        global overlay_artists
        for art in overlay_artists:
            try:
                art.remove()
            except Exception:
                pass
        overlay_artists = []

    def draw_local_overlays(agents_state_dec_t, neighbors_local_t, w_lanes_local_t, w_boundaries_local_t):
        global overlay_artists, first_agent_idx
        try:
            clear_overlays()
            import numpy as np
            import math
            # 自车位姿
            ego = simulator.agents_state[0, first_agent_idx]
            ex = float(ego[0].item()); ey = float(ego[1].item()); eyaw = float(ego[2].item())
            cos_y = math.cos(eyaw); sin_y = math.sin(eyaw)
            R = np.array([[cos_y, -sin_y],[sin_y, cos_y]], dtype=float)

            # lanes
            lanes = w_lanes_local_t[0, first_agent_idx] if w_lanes_local_t is not None else None
            if lanes is not None:
                lanes_np = lanes.detach().to('cpu').numpy()
                if lanes_np.ndim >= 2 and lanes_np.shape[-1] >= 2:
                    valid = (lanes_np[...,0] != -1) & (lanes_np[...,1] != -1)
                    pts = lanes_np[valid][..., :2]
                    if pts.size > 0:
                        world = pts @ R.T + np.array([ex, ey])
                        h = ax.scatter(world[:,0], world[:,1], s=5, c='lime', alpha=0.8, label='w_lanes_local')
                        overlay_artists.append(h)

            # boundaries
            bounds = w_boundaries_local_t[0, first_agent_idx] if w_boundaries_local_t is not None else None
            if bounds is not None:
                bounds_np = bounds.detach().to('cpu').numpy()
                if bounds_np.ndim >= 2 and bounds_np.shape[-1] >= 2:
                    valid = (bounds_np[...,0] != -1) & (bounds_np[...,1] != -1)
                    pts = bounds_np[valid][..., :2]
                    if pts.size > 0:
                        world = pts @ R.T + np.array([ex, ey])
                        h = ax.scatter(world[:,0], world[:,1], s=4, c='k', alpha=0.5, label='w_boundaries_local')
                        overlay_artists.append(h)

            # neighbors_local: (B, M, K, 7) -> [dx, dy, dvx, dvy, length, width, active]
            neigh = neighbors_local_t[0, first_agent_idx] if neighbors_local_t is not None else None
            if neigh is not None:
                neigh_np = neigh.detach().to('cpu').numpy()
                if neigh_np.ndim >= 2 and neigh_np.shape[-1] >= 6:
                    # 有效点：active>0.5 或者 长宽>0
                    active_mask = neigh_np[..., 6] > 0.5 if neigh_np.shape[-1] >= 7 else np.ones(neigh_np.shape[0], dtype=bool)
                    valid = active_mask
                    dxdy = neigh_np[valid][..., :2]
                    if dxdy.size > 0:
                        world_pts = dxdy @ R.T + np.array([ex, ey])
                        # 只为被观察到的邻居绘制标签（一次性标注）
                        h = ax.scatter(world_pts[:,0], world_pts[:,1], s=20, facecolors='none', edgecolors='red', linewidths=2, label='neighbors_local')
                        overlay_artists.append(h)
                        # 标注被观察到的邻居（仅一次图例）
                        for j, (wx, wy) in enumerate(world_pts):
                            txtn = ax.text(wx, wy, 'N', fontsize=8, color='red', weight='bold')
                            overlay_artists.append(txtn)
                        try:
                            # 仅处理前N个，避免过多图元
                            max_draw = min(world_pts.shape[0], 20)
                            # 取对应的行索引
                            valid_indices = np.nonzero(valid)[0][:max_draw]
                            # 计算自车绝对速度（世界坐标）
                            ego_speed = float(simulator.agents_state[0, first_agent_idx, 3].item())
                            vx_ego = ego_speed * cos_y
                            vy_ego = ego_speed * sin_y
                            for ii in valid_indices:
                                row = neigh_np[ii]
                                nx, ny = float(row[0]), float(row[1])
                                dvx_local, dvy_local = float(row[2]), float(row[3])
                                nlen = float(row[4])
                                nwid = float(row[5])
                                # 局部中心 -> 世界中心
                                cx, cy = (R @ np.array([nx, ny])).tolist(); cx += ex; cy += ey
                                # 相对速度(局部) -> 世界相对速度
                                rvx_world, rvy_world = (R @ np.array([dvx_local, dvy_local])).tolist()
                                # 近似邻居绝对速度 = 自车绝对速度 + 相对世界速度
                                nvx_world = vx_ego + rvx_world
                                nvy_world = vy_ego + rvy_world
                                speed_mag = math.hypot(nvx_world, nvy_world)
                                if speed_mag > 1e-2:
                                    nyaw_world = math.atan2(nvy_world, nvx_world)
                                else:
                                    nyaw_world = eyaw
                                c = math.cos(nyaw_world); s = math.sin(nyaw_world)
                                Rn = np.array([[c, -s], [s, c]], dtype=float)
                                hl = max(0.1, nlen * 0.5); hw = max(0.1, nwid * 0.5)
                                rect_local = np.array([
                                    [-hl, -hw],
                                    [ hl, -hw],
                                    [ hl,  hw],
                                    [-hl,  hw]
                                ], dtype=float)
                                rect_world = rect_local @ Rn.T + np.array([cx, cy])
                                # 邻居整体涂黑 + 金色描边
                                poly = Polygon(rect_world, closed=True, facecolor='black', edgecolor='gold', linewidth=2.0, alpha=0.9)
                                ax.add_patch(poly)
                                overlay_artists.append(poly)
                                # 绘制邻居世界速度方向（金色箭头，长度按速度幅值裁剪）
                                if speed_mag > 1e-3:
                                    ux = nvx_world / speed_mag
                                    uy = nvy_world / speed_mag
                                    arrow_len = max(3.0, min(8.0, speed_mag))
                                    arr_v = ax.arrow(cx, cy, ux * arrow_len, uy * arrow_len,
                                                     head_width=0.8, head_length=0.8, fc='gold', ec='gold',
                                                     alpha=0.95, zorder=7, linewidth=2)
                                    overlay_artists.append(arr_v)
                        except Exception:
                            pass

            fig.canvas.draw_idle()
        except Exception as e:
            print(f"draw_local_overlays error: {e}")

    # 初始绘制一次
    try:
        draw_local_overlays(agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local)
    except Exception:
        pass
    
    # 绘制第一个激活agent的局部路径规划
    if simulator.agents_path_plans_local is not None and len(active_indices) > 0:
        first_agent_idx = int(active_indices[0])
        # 获取第一个激活agent的局部路径规划
        path_local = simulator.agents_path_plans_local[0, first_agent_idx].cpu().numpy()  # (L, 2)
        
        # 过滤有效点（非零坐标）
        valid_mask = (path_local[:, 0] != 0) | (path_local[:, 1] != 0)
        if valid_mask.any():
            valid_path = path_local[valid_mask]
            
            # 将局部坐标转换回世界坐标进行绘制
            ego = simulator.agents_state[0, first_agent_idx]
            ego_x, ego_y, ego_yaw = float(ego[0].item()), float(ego[1].item()), float(ego[2].item())
            cos_yaw = np.cos(ego_yaw)
            sin_yaw = np.sin(ego_yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            ego_pos = np.array([ego_x, ego_y])
            
            # 逆变换：从局部坐标转换回世界坐标
            world_path = (valid_path @ rotation_matrix.T) + ego_pos
            
            # 绘制局部路径规划（绿色实线）
            ax.plot(world_path[:, 0], world_path[:, 1], 'g-', linewidth=3, alpha=0.8, label='Local Path Plan')
            ax.scatter(world_path[0, 0], world_path[0, 1], c='green', marker='o', s=100, label='Path Start')
            if len(world_path) > 1:
                ax.scatter(world_path[-1, 0], world_path[-1, 1], c='green', marker='x', s=100, label='Path Goal')
    
    # 统一图形样式
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('road graph and agent positions, first agent local path plan')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # 绘制观测半径虚线圆（以第一个激活agent为圆心）
    horizon_circle = None
    try:
        horizon = float(config['simulator']['observation']['horizon'])
        if len(active_indices) > 0:
            first_idx = int(active_indices[0])
            cx = float(simulator.agents_state[0, first_idx, 0].item())
            cy = float(simulator.agents_state[0, first_idx, 1].item())
            horizon_circle = Circle((cx, cy), radius=horizon, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.add_patch(horizon_circle)
    except Exception:
        pass
    
    # 添加 Next Step 按钮
    btn_ax = fig.add_axes([0.82, 0.02, 0.15, 0.05])
    btn_next = Button(btn_ax, 'Next Step')

    # 第二个figure：动作概率分布（仅第一个环境的第一个agent）
    num_actions = simulator.dynamics_model.discrete_action_space.num_actions
    first_agent_idx = 0
    if len(active_indices) > 0:
        first_agent_idx = int(active_indices[0])
    fig_act, ax_act = plt.subplots(figsize=(6, 3))
    fig_act.canvas.manager.set_window_title('Action Probabilities (Agent 0)')
    bars = ax_act.bar(np.arange(num_actions), np.zeros(num_actions), color='tab:blue')
    ax_act.set_xlabel('Action Index')
    ax_act.set_ylabel('Probability')
    ax_act.set_title('First Agent Action Probabilities')
    ax_act.set_xlim(-0.5, num_actions - 0.5)
    ax_act.set_ylim(0.0, 1.0)
    fig_act.tight_layout()

    def refresh_agents():
        global agent_artists, active_indices, horizon_circle
        agents_state_np_local = simulator.agents_state.cpu().numpy()
        new_active_mask = agents_state_np_local[0, :, 6] > 0.5
        new_active_indices = np.where(new_active_mask)[0]
        if not np.array_equal(new_active_indices, active_indices):
            for _, poly, txt, arr, info, _ in agent_artists:
                try:
                    poly.remove(); txt.remove(); info.remove(); arr.remove()
                except Exception:
                    pass
            agent_artists, active_indices = build_agent_artists()
            fig.canvas.draw_idle()
            return
        import math
        for (agent_idx, poly, txt, arr, info, color) in agent_artists:
            x, y, yaw, speed, length, width, active = agents_state_np_local[0, agent_idx]
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            half_length = length / 2.0
            half_width = width / 2.0
            corners = np.array([
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width]
            ])
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            agent_corners = corners @ rotation_matrix.T + np.array([x, y])
            poly.set_xy(agent_corners)
            if txt is not None:
                txt.set_position((x, y))
            try:
                arr.remove()
            except Exception:
                pass
            speed_vec = 3.0
            new_arr = ax.arrow(x, y, speed_vec * cos_yaw, speed_vec * sin_yaw,
                               head_width=0.5, head_length=0.5, fc=color, ec=color,
                               alpha=0.8, zorder=5, linewidth=2)
            idx = [i for i, t in enumerate(agent_artists) if t[0] == agent_idx][0]
            agent_artists[idx] = (agent_idx, poly, txt, new_arr, info, color)
            if info is not None:
                info.set_position((x, y + half_width + 1))
                info.set_text(f'v={speed:.1f}m/s')
        
        # 更新虚线圆位置（跟随第一个激活agent）
        try:
            if horizon_circle is not None and len(active_indices) > 0:
                first_idx = int(active_indices[0])
                new_cx = float(simulator.agents_state[0, first_idx, 0].item())
                new_cy = float(simulator.agents_state[0, first_idx, 1].item())
                horizon_circle.center = (new_cx, new_cy)
        except Exception:
            pass
            
        fig.canvas.draw_idle()

    def on_next_clicked(event):
        # 使用网络输出的分布采样动作并推进一步
        global features_tensor, first_agent_idx
        with torch.no_grad():
            logits = model.forward(features_tensor, mode="policy")
            dist = torch.distributions.Categorical(logits=logits)
            # 先显示当前步的动作概率分布
            try:
                probs = dist.probs.detach().to('cpu').numpy()  # (B, M, A)
                probs_first = probs[0, first_agent_idx]
                for i, b in enumerate(bars):
                    b.set_height(float(probs_first[i]))
                ax_act.set_ylim(0.0, 1.0)
                fig_act.canvas.draw_idle()
            except Exception:
                pass
            actions = dist.sample()
        observation, reward, done = simulator.step(actions)
        # 显示当前观测agent的reward（B=0, M=first_agent_idx）
        try:
            cur_r = float(reward[0, first_agent_idx].item())
            print(f"当前观测agent(B=0, M={first_agent_idx}) reward: {cur_r:.4f}",'done:',done[0, first_agent_idx].item())
        except Exception:
            pass
        # 基于新观测重建特征，供下一步使用
        try:
            with torch.no_grad():
                agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config_ns)
                
                features_tensor = build_network_features(
                    agents_state_dec,
                    neighbors_local,
                    w_lanes_local,
                    w_boundaries_local,
                    simulator.agents_path_plans_local,
                    simulator.stop_lines if hasattr(simulator, 'stop_lines') else None,
                    simulator.reward_calculator.sampled_params,
                    config_ns, 
                )
                # 绘制局部要素，并打印本次 agents_state_dec
                try:
                    draw_local_overlays(agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local)
                    print('features_tensor:',features_tensor[0, first_agent_idx])
                    #print('neighbors_local:',neighbors_local[0, first_agent_idx])
                except Exception:
                    pass
        except Exception:
            pass
        refresh_agents()

    btn_next.on_clicked(on_next_clicked)

    # 绑定空格键为“下一步”
    def on_key_press(event):
        try:
            if event.key in (' ', 'space'):
                on_next_clicked(event)
        except Exception:
            pass

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig_act.canvas.mpl_connect('key_press_event', on_key_press)

    plt.tight_layout()
    plt.show()

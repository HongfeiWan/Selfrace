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
        # 重置done状态
        self.last_done = None
        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation,d,theta_f = self.observation_generator.generate(self.agents_state)
        print(initial_observation.shape)
        print("Initial observation generated")

        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        paths = self.path_planner.path_plan(self.agents_start_quad_ids, self.agents_goal_quad_ids)
        self.agents_path_plans = self.path_planner.collect_path_w_lane_ids(paths, self.agents_start_quad_ids, self.agents_goal_quad_ids)

        print(f"Reset complete. World state shape: {self.agents_state.shape}")
        self.stop_lines = torch.zeros((self.num_envs, self.max_agents,20), dtype=torch.int32, device=self.device)
        return initial_observation,d,theta_f
    
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
    # ==================== 1. 初始化 TeraflowSimulator ====================
    import json
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    sim = TeraflowSimulator(config=cfg, device=torch.device(device))
    initial_observation, d, theta_f = sim.reset()
    
    # ==================== 2. 获取路径规划数据（直接使用reset()中已生成的路径） ====================
    # reset()方法中已经调用了路径规划，直接使用其结果
    planner = sim.path_planner
    rn = sim.road_network  # 用于后续分析
    agents_state = sim.agents_state  # (B, M, 7) - 读取所有智能体的状态
    start_poly_tensor = sim.agents_start_quad_ids  # (B, M)
    end_poly_tensor = sim.agents_goal_quad_ids  # (B, M)
    agents_path_plans = sim.agents_path_plans  # (B, M, w_lane_ids_length, 3) - reset()中已生成
    
    # ==================== 检查无效 poly_id ====================
    print("=" * 80)
    print("检查无效 poly_id:")
    start_quad_ids = sim.agents_start_quad_ids  # (B, M)
    goal_quad_ids = sim.agents_goal_quad_ids    # (B, M)
    invalid_marker = sim.world_initializer.INVALID_MARKER
    
    # 检查 start_quad_ids 中的无效值
    invalid_start_mask = (start_quad_ids == invalid_marker)
    invalid_start_count = invalid_start_mask.sum().item()
    total_count = start_quad_ids.numel()
    print(f"Start quad_ids 中无效数量: {invalid_start_count} / {total_count}")
    
    # 检查 goal_quad_ids 中的无效值
    invalid_goal_mask = (goal_quad_ids == invalid_marker)
    invalid_goal_count = invalid_goal_mask.sum().item()
    print(f"Goal quad_ids 中无效数量: {invalid_goal_count} / {total_count}")
    
    # 检查 poly_id_lookup 中的无效值（即使不是 INVALID_MARKER，也可能在 lookup 中无效）
    if start_quad_ids.numel() > 0:
        # 获取所有有效的 start_quad_ids（排除 INVALID_MARKER）
        valid_start_mask = (start_quad_ids != invalid_marker)
        valid_start_ids = start_quad_ids[valid_start_mask]
        if valid_start_ids.numel() > 0:
            # 检查这些 poly_id 在 poly_id_lookup 中是否有效
            max_poly_id = valid_start_ids.max().item()
            if max_poly_id < planner.poly_id_lookup.shape[0]:
                lookup_results = planner.poly_id_lookup[valid_start_ids]
                invalid_in_lookup = (lookup_results < 0).sum().item()
                print(f"Start quad_ids 中在 poly_id_lookup 中无效的数量: {invalid_in_lookup} / {valid_start_ids.numel()}")
                if invalid_in_lookup > 0:
                    invalid_ids = valid_start_ids[lookup_results < 0]
                    print(f"  无效的 poly_id 示例（前5个）: {invalid_ids[:5].cpu().numpy()}")
            else:
                print(f"警告: 最大 poly_id ({max_poly_id}) 超出 poly_id_lookup 范围 ({planner.poly_id_lookup.shape[0]})")
    
    # 检查 poly_id_lookup 的统计信息
    print(f"poly_id_lookup 大小: {planner.poly_id_lookup.shape[0]}")
    valid_in_lookup = (planner.poly_id_lookup >= 0).sum().item()
    print(f"poly_id_lookup 中有效的 poly_id 数量: {valid_in_lookup} / {planner.poly_id_lookup.shape[0]}")
    print("=" * 80)
    
    # ==================== 打印第一条路径 ====================
    print("=" * 80)
    print("第一条路径 agents_path_plans[0, 0] 的内容:")
    print(f"路径形状: {agents_path_plans[0, 0].shape}")
    print(f"路径数据 (前10个点):")
    first_path = agents_path_plans[0, 0].cpu().numpy()  # 转换为numpy便于打印
    for i in range(min(10, len(first_path))):
        print(f"  点 {i}: x={first_path[i, 0]:.2f}, y={first_path[i, 1]:.2f}, angle={first_path[i, 2]:.4f}")
    print(f"无效标记值: {planner.INVALID_w_lane_id_MARKER}")
    print('state',agents_state[0,0].cpu().numpy())
    
    # 统计有效路径点数量
    invalid_marker_tensor = torch.tensor(planner.INVALID_w_lane_id_MARKER, device=agents_path_plans.device, dtype=agents_path_plans.dtype)
    valid_mask = agents_path_plans[0, 0, :, 0] != invalid_marker_tensor
    valid_count = valid_mask.sum().item()
    print(f"有效路径点数量: {valid_count} / {len(first_path)}")
    print("=" * 80)
    
    # ==================== 3. 绘制active车辆的路径规划 ====================
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import numpy as np
    
    # 获取active掩码：agents_state的第七位（索引6）表示是否active
    active_mask = agents_state[..., 6] > 0.5  # (B, M) - 布尔掩码
    
    # 获取无效标记值（转换为tensor以确保类型匹配）
    invalid_marker_value = planner.INVALID_w_lane_id_MARKER
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Active Vehicles Path Plans (Batch 0)')
    
    # ==================== 绘制road network中的四边形 ====================
    if rn.quads_vertices.numel() > 0:
        quads_np = rn.quads_vertices.detach().cpu().numpy()  # (N, 4, 2)
        for verts in quads_np:
            # 将顶点转换为matplotlib Polygon需要的格式
            vertices_2d = [(v[0], v[1]) for v in verts]
            poly = Polygon(vertices_2d, closed=True, 
                         facecolor='yellow', edgecolor='black', 
                         alpha=0.2, linewidth=0.2, zorder=0)
            ax.add_patch(poly)
        print(f"绘制了 {len(quads_np)} 个四边形")
    
    # 定义不同车辆的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 只绘制第一个批次（B=0）的车辆
    B, M = agents_state.shape[:2]
    b = 0  # 只绘制第一个批次
    path_count = 0
    
    # 仅选择 B=0 中第一个 active 的 agent 进行绘制
    active_agents = torch.nonzero(active_mask[b], as_tuple=False).squeeze(-1)
    if active_agents.numel() == 0:
        print("Batch 0 中没有active车辆")
    else:
        m = int(active_agents[0].item())
        # 获取该车辆的路径规划 (w_lane_ids_length, 3)
        path = agents_path_plans[b, m]  # (w_lane_ids_length, 3)

        # 过滤掉无效的路径点（第一个维度不等于invalid_marker）
        # 使用tensor比较以确保类型匹配
        invalid_marker_tensor = torch.tensor(invalid_marker_value, device=path.device, dtype=path.dtype)
        valid_mask = path[:, 0] != invalid_marker_tensor
        valid_path = path[valid_mask].cpu().numpy()  # 转换为numpy数组
        
        if len(valid_path) > 0:
            # 提取x, y坐标
            x_coords = valid_path[:, 0]
            y_coords = valid_path[:, 1]
            
            # 选择颜色
            color = colors[path_count % len(colors)]
            
            # 绘制路径点连线（zorder=2，确保在四边形之上）
            ax.plot(x_coords, y_coords, 'o-', color=color, markersize=4, 
                   linewidth=2, alpha=0.7, label=f'B{b}_M{m}', zorder=2)
            
            # 绘制起点（绿色，zorder=3，确保在最上层）
            ax.plot(x_coords[0], y_coords[0], 's', color='green', 
                   markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=3)
            
            # 绘制终点（红色，zorder=3）
            ax.plot(x_coords[-1], y_coords[-1], '^', color='red', 
                   markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=3)
            
            # 绘制所有有效点的方向箭头（zorder=2）
            angles = valid_path[:, 2]  # 角度信息
            arrow_length = 0.5
            for i in range(len(valid_path)):
                x, y, angle = x_coords[i], y_coords[i], angles[i]
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)
                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2, 
                       fc=color, ec=color, alpha=0.6, length_includes_head=True, zorder=2)
            path_count = 1
    
    # 添加图例（如果路径数量不多）
    if path_count <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    print(f"绘制了 {path_count} 个active车辆的路径规划")
    
    # 显示图形
    plt.tight_layout()
    plt.show()



    
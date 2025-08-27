from multiprocessing.spawn import prepare
from sympy import N
import torch
import yaml
import os
import sys
from typing import Dict, Tuple, Optional
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
    它的设计遵循 GigaFlow 的核心思想：批量化、可微分（未来）以及与自博弈循环的兼容性。
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
        reward_config = simulator_config['reward']
        self.reward_calculator = RewardCalculator(reward_config, self.device)

        # 9. 初始化路径规划器
        # PathPlanner现在会自动加载所需的数据
        self.path_planner = PathPlanner(map_path=self.map_path, device=self.device)

        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        self.agents_goal_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的目标quad_id
        self.agents_path_plans: Optional[torch.Tensor] = None     # 存储所有智能体的路径规划

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
        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation = self.observation_generator.generate(self.agents_state)
        print(initial_observation.shape)
        print("Initial observation generated")
        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        self._initialize_path_planning()
        print("Path planning initialized")
        print(f"Reset complete. World state shape: {self.agents_state.shape}")
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

        action_update_time=time.time()
        # 1. 基于收到的所有动作，更新所有激活智能体的状态
        active_mask = self.agents_state[..., 6] > 0.5
        if active_mask.any():
            active_states = self.agents_state[active_mask]
            active_actions = actions[active_mask]   #保留有效的actions，无效的actions会被填充为0
            # 确保active_actions的形状正确
            if active_actions.ndim == 3:
                active_actions = active_actions.squeeze(0)  # 移除多余的维度

            # 离散动作空间：actions是动作索引，需要转换为标量
            if active_actions.ndim > 1:
                # 如果actions是二维的，取第一个维度作为动作索引
                active_actions = active_actions[:, 0].long()
            else:
                active_actions = active_actions.long()
            active_dynamics_states = active_states[:, :4]
            new_active_dynamics_states = self.dynamics_model.step(active_dynamics_states, active_actions, self.dt)
            # 直接将更新后的动力学状态写回
            updated_states = self.agents_state[active_mask]
            updated_states[:, :4] = new_active_dynamics_states
            self.agents_state[active_mask] = updated_states
        action_update_time=time.time()-action_update_time
        print(f"action_update_time: {action_update_time:.4f}s")

        offroad_check_time=time.time()
        # 2. 离路检测
        is_on_road = torch.ones_like(active_mask) # 默认在路上
        if active_mask.any():
            active_states = self.agents_state[active_mask]
            # OffroadChecker 需要 [x, y, yaw, length, width]
            states_for_checker = active_states[:, [0, 1, 2, 4, 5]]
            active_is_on_road = self.offroad_checker.check_on_road(states_for_checker)
            is_on_road[active_mask] = active_is_on_road
        offroad_mask = ~is_on_road # (B, M)
        offroad_update_time=time.time()-offroad_check_time
        print(f"offroad_update_time: {offroad_update_time:.4f}s")


        collision_check_time=time.time()
        # 3. 动态碰撞检测
        collision_check_result = self.collision_checker.check(
            states_t0, self.agents_state, debug=debug_collision, debug_env_idx=0
        )
        all_collisions = collision_check_result
        collision_update_time=time.time()-collision_check_time
        print(f"collision_update_time: {collision_update_time:.4f}s")

        frenet_time=time.time()
        # 4. 计算Frenet坐标信息
        vehicle_positions = self.agents_state[..., :2]  # (B, M, 2) - x, y
        vehicle_headings = self.agents_state[..., 2]    # (B, M) - heading
        d, theta_f = self.road_network.calculate_frenet_coordinates(vehicle_positions, vehicle_headings, self.spatial_hash)
        frenet_update_time=time.time()-frenet_time
        print(f"frenet_update_time: {frenet_update_time:.4f}s")

        observation_time=time.time()
        # 5. 生成新的观测
        observation = self.observation_generator.generate(self.agents_state)
        observation_update_time=time.time()-observation_time
        print(f"observation_update_time: {observation_update_time:.4f}s")

        reward_time=time.time()
        # 6. 计算奖励（传入Frenet坐标和动作）
        reward, goal_reached = self._calculate_reward(all_collisions, offroad_mask, d, theta_f, actions)
        reward_update_time=time.time()-reward_time
        print(f"reward_update_time: {reward_update_time:.4f}s")

        done_time=time.time()
        # 7. 检查是否结束（包含目标到达判断）
        done = all_collisions|offroad_mask|goal_reached
        done_update_time=time.time()-done_time
        print(f"done_update_time: {done_update_time:.4f}s")
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
            # 激活掩码
            active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
            # 1) 构造全局 along/alat 加速度 (B,M)，仅激活位置为有效值
            #    将连续的 active 向量按掩码散射回批量形状
            along_active = self.dynamics_model.current_along  # (N_active,) or None
            alat_active = self.dynamics_model.current_alat    # (N_active,) or None
            flat_mask = active_mask.view(-1)
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
                # 仅对激活体保留数值
                mask_f = active_mask.float()
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
        goal_positions = self.path_planner.get_quad_centers(self.agents_goal_quad_ids)
        goal_reached = False

        # 调用奖励计算器，这个速度很快
        reward, goal_reached = self.reward_calculator.calculate(
            extended_state,
            all_collisions,
            offroad_mask,
            dt=self.dt,
            goal_positions=goal_positions,
            waypoint_reached=goal_reached,
        )

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
        if active_mask.any():
            rand_vals = torch.randint(0, self.road_network.num_quads, (int(active_mask.sum().item()),), device=self.device, dtype=torch.int32)
            goal_i32 = goal_i32.masked_scatter(active_mask, rand_vals)
        # 一次性调用 planner（形状为 (B,M,1)）
        start_3d = start_i32.unsqueeze(-1)
        goal_3d = goal_i32.unsqueeze(-1)
        prepare_time_done = time.time()
        print(f"prepare_time_done: {prepare_time_done-prepare_time:.4f}s")
        path_plans = self.path_planner.plan_path(start_3d, goal_3d)  # 形状 (B,M,L,2)
        # 存储结果
        self.agents_goal_quad_ids = goal_i32
        self.agents_path_plans = path_plans

if __name__ == '__main__':
    # 这是一个简单的使用示例，用于测试模拟器的基本功能
    # 从配置文件读取配置
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    config_path = 'configs/default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    simulator = TeraflowSimulator(config=config, device=torch.device('cuda:0'))
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

    # 绘制激活的智能体
    active_mask = agents_state_np[0, :, 6] > 0.5  # 第一个环境的激活智能体
    print(active_mask)
    active_agents = agents_state_np[0, active_mask]  # 激活智能体的状态
    if len(active_agents) > 0:
        print(f"绘制 {len(active_agents)} 个激活智能体...")
        # 定义不同智能体的颜色
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        # 获取激活的智能体索引
        active_indices = np.where(active_mask)[0]
        # 绘制智能体矩形
        for i, agent_state in enumerate(active_agents):
            x, y, yaw, speed, length, width, active = agent_state
            # 创建智能体矩形
            # 智能体中心在(x, y)，需要根据yaw旋转
            import math
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
            # 旋转角点
            rotated_corners = corners @ rotation_matrix.T
            # 平移到智能体位置
            agent_corners = rotated_corners + np.array([x, y])
            # 创建矩形多边形
            agent_polygon = Polygon(agent_corners, closed=True)
            # 选择颜色
            agent_idx = active_indices[i]
            color = colors[i % len(colors)]
            alpha = 0.8
            label = f'Agent {agent_idx}'
            # 添加智能体到图上
            ax.add_patch(agent_polygon)
            agent_polygon.set_facecolor(color)
            agent_polygon.set_alpha(alpha)
            agent_polygon.set_edgecolor('black')
            agent_polygon.set_linewidth(2)
            # 添加标签
            ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   weight='bold')
            # 绘制速度向量（方向指示）
            speed_vector_length = 3.0  # 速度向量的显示长度
            speed_dx = speed_vector_length * cos_yaw
            speed_dy = speed_vector_length * sin_yaw
            ax.arrow(x, y, speed_dx, speed_dy, head_width=0.5, head_length=0.5, 
                    fc=color, ec=color, alpha=0.8, zorder=5, linewidth=2)
            # 显示智能体信息
            info_text = f'v={speed:.1f}m/s'
            ax.text(x, y + half_width + 1, info_text, ha='center', va='bottom', 
                   fontsize=8, color=color, weight='bold')
    
    # 叠加绘制：对应mask位置的 agents_path_plans 路径（与goals.py一致）
    if simulator.agents_path_plans is not None:
        plans_np = simulator.agents_path_plans.cpu().numpy()  # (B, M, 512, 2)
        method2_paths = []
        for i in range(plans_np.shape[1]):
            path_i = plans_np[0, i]
            valid_mask = (path_i[:, 0] != -1) & (path_i[:, 1] != -1)
            if valid_mask.any():
                coords = path_i[valid_mask]
                method2_paths.append(coords)
                print(f"方法2路径 {i+1}: {len(coords)}个点")
                print(f"  起点: {coords[0]}")
                print(f"  终点: {coords[-1]}")
            else:
                method2_paths.append(None)
        # 绘制方法2的路径（红虚线 + 起点圆点/终点叉号）
        for i in range(len(method2_paths)):
            if method2_paths[i] is None:
                continue
            path2 = method2_paths[i]
            ax.plot(path2[:, 0], path2[:, 1], 'r--', linewidth=2, label='Method2: plan_path' if i == 0 else None)
            ax.scatter(path2[0, 0], path2[0, 1], c='red', marker='o', s=100, label='start' if i == 0 else None)
            ax.scatter(path2[-1, 0], path2[-1, 1], c='red', marker='x', s=100, label='goal' if i == 0 else None)
    # 统一图形样式
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('road graph and agent positions, path plans')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.tight_layout()
    plt.show()

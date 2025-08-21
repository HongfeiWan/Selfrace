import torch
import yaml
import os
import sys
from typing import Dict, Tuple, Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 依赖于 spatial_hash
# 添加utils目录到路径
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
# 添加simulator目录到路径
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
    
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
        self.observation_generator = ObservationGenerator(self.road_network, obs_config, self.device)

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
        # 使用 WorldInitializer 来生成一批新的世界状态，包括起始quad_id
        self.agents_state, _, self.agents_start_quad_ids = self.world_initializer.initialize_world(self.num_envs)
        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation = self.observation_generator.generate(self.agents_state)
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
            actions (torch.Tensor): 形状为 (num_envs, num_agents, action_dim) 的动作张量。
            debug_collision (bool): 是否为碰撞检测器开启调试模式。
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
                - observation (torch.Tensor): 新的观测 (B, M, obs_dim)。
                - reward (torch.Tensor): 奖励 (B, M)。
                - done (torch.Tensor): 是否结束的标志 (B, M)。
        """
        if self.agents_state is None:
            raise RuntimeError("Must call reset() before calling step().")
        actions = actions.to(self.device)
        states_t0 = self.agents_state.clone()
        # 1. 基于收到的所有动作，更新所有激活智能体的状态
        active_mask = self.agents_state[..., 6] > 0.5
        if active_mask.any():
            active_states = self.agents_state[active_mask]
            active_actions = actions[active_mask]
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
        
        # 2. 离路检测
        is_on_road = torch.ones_like(active_mask) # 默认在路上
        if active_mask.any():
            active_states = self.agents_state[active_mask]
            # OffroadChecker 需要 [x, y, yaw, length, width]
            states_for_checker = active_states[:, [0, 1, 2, 4, 5]]
            active_is_on_road = self.offroad_checker.check_on_road(states_for_checker)
            is_on_road[active_mask] = active_is_on_road
        offroad_mask = ~is_on_road # (B, M)

        # 3. 动态碰撞检测
        collision_check_result = self.collision_checker.check(
            states_t0, self.agents_state, debug=debug_collision, debug_env_idx=0
        )
        all_collisions = collision_check_result

        # 4. 计算Frenet坐标信息
        vehicle_positions = self.agents_state[..., :2]  # (B, M, 2) - x, y
        vehicle_headings = self.agents_state[..., 2]    # (B, M) - heading
        d, theta_f = self.road_network.calculate_frenet_coordinates(vehicle_positions, vehicle_headings)
        
        # 5. 生成新的观测
        observation = self.observation_generator.generate(self.agents_state)
        
        # 6. 计算奖励（传入Frenet坐标和动作）
        reward, goal_reached = self._calculate_reward(all_collisions, offroad_mask, d, theta_f, actions)

        # 7. 检查是否结束（包含目标到达判断）
        done = all_collisions|offroad_mask|goal_reached
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
        
        # 从动力学模型获取当前加速度
        if hasattr(self.dynamics_model, 'current_along') and hasattr(self.dynamics_model, 'current_alat'):
            # 获取当前加速度
            along = self.dynamics_model.current_along
            alat = self.dynamics_model.current_alat
            # 创建完整的加速度张量，为未激活的智能体填充零值
            full_along = torch.zeros((B, M), device=self.device)
            full_alat = torch.zeros((B, M), device=self.device)
            full_along_jerk = torch.zeros((B, M), device=self.device)
            full_alat_jerk = torch.zeros((B, M), device=self.device)
            # 只对激活的智能体应用加速度
            active_mask = self.agents_state[..., 6] > 0.5
            if active_mask.any():
                full_along[active_mask] = along
                full_alat[active_mask] = alat
                # 直接从动作空间获取jerk值，而不是从动力学模型计算
                if actions is not None:
                    # 获取激活智能体的动作索引
                    active_actions = actions[active_mask]
                    # 确保动作索引是正确的形状
                    if active_actions.ndim > 1:
                        active_actions = active_actions[:, 0].long()
                    else:
                        active_actions = active_actions.long()
                    # 从离散动作空间获取对应的jerk值
                    jerk_actions = self.dynamics_model.discrete_action_space.get_action(active_actions)  # (N, 2) [along_jerk, alat_jerk]
                    full_along_jerk[active_mask] = jerk_actions[:, 0]  # 纵向jerk
                    full_alat_jerk[active_mask] = jerk_actions[:, 1]   # 横向jerk
            # 填充加速度信息
            extended_state[..., 4] = full_along      # along
            extended_state[..., 5] = full_alat       # alat
            extended_state[..., 6] = full_along_jerk # along_jerk
            extended_state[..., 7] = full_alat_jerk  # alat_jerk
        # 直接使用传入的Frenet坐标信息，避免重复计算
        extended_state[..., 8] = theta_f  # theta_f - Frenet角度误差
        extended_state[..., 9] = d        # d - Frenet横向距离
        # 准备目标奖励计算的参数

        goal_positions = self.path_planner.get_quad_centers(self.agents_goal_quad_ids)
        goal_reached = False

        # 调用奖励计算器
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
        if self.agents_state is None:
            return
        B, M, _ = self.agents_state.shape  # batch_size, max_agents, state_dim
        
        # 1. 获取所有激活智能体的位置
        active_mask = self.agents_state[..., 6] > 0.5  # (B, M)

        # 2. 使用已知的起始quad_id（从世界初始化中获取）
        if not hasattr(self, 'agents_start_quad_ids') or self.agents_start_quad_ids is None:
            print("Warning: No start quad IDs available for path planning")
            return
        
        # 3. 为所有激活智能体随机分配目标quad_id
        # 获取所有可用的quad_id
        available_quad_ids = torch.arange(self.road_network.num_quads, device=self.device)
        
        # 创建完整的目标quad_id张量 (B, M)
        goal_quad_ids = torch.full((B, M), -1, dtype=torch.long, device=self.device)
        
        # 为每个激活的智能体分配目标
        active_indices = torch.where(active_mask)
        num_active_agents = len(active_indices[0])
        print(f"为 {num_active_agents} 个激活智能体分配目标...")
        
        for b, m in zip(active_indices[0], active_indices[1]):
            start_id = self.agents_start_quad_ids[b, m].item()
            # 排除起始quad_id，从其他quads中随机选择
            available_goals = available_quad_ids[available_quad_ids != start_id]
            if len(available_goals) > 0:
                # 修复：正确获取随机索引
                random_idx = torch.randint(0, len(available_goals), (1,)).item()
                goal_quad_ids[b, m] = available_goals[random_idx].item()
                print(f"智能体 ({b}, {m}): 起始 {start_id} -> 目标 {goal_quad_ids[b, m].item()}")
                
                # 验证目标quad_id的有效性
                goal_id = goal_quad_ids[b, m].item()
                if goal_id < 0 or goal_id >= self.road_network.num_quads:
                    print(f"错误: 目标quad_id {goal_id} 无效，范围应为 0-{self.road_network.num_quads-1}")
            else:
                # 如果没有其他选择，使用起始位置
                goal_quad_ids[b, m] = start_id
                print(f"智能体 ({b}, {m}): 起始 {start_id} -> 目标 {start_id} (无其他选择)")
        
        # 4. 使用plan_path批量生成路径规划
        path_plans = self.path_planner.plan_path(self.agents_start_quad_ids, goal_quad_ids)

        # 5. 存储结果
        self.agents_goal_quad_ids = goal_quad_ids
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
    # 可视化道路网络和智能体位置（优化版本）
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
            
    # 绘制每个智能体的路径（过滤掉 -1,-1）
    if simulator.agents_path_plans is not None:
        try:
            paths_np = simulator.agents_path_plans[0].detach().cpu().numpy()  # (M, L, 2)
        except Exception:
            paths_np = simulator.agents_path_plans.detach().cpu().numpy()      # (M, L, 2)
        M_paths = paths_np.shape[0]
        # 若颜色表不存在，则创建一个
        if 'colors' not in locals():
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for m in range(M_paths):
            coords = paths_np[m]  # (L,2)
            mask_valid = (coords[:, 0] != -1) & (coords[:, 1] != -1)
            valid = coords[mask_valid]
            if valid.shape[0] == 0:
                continue
            col = colors[m % len(colors)]
            ax.scatter(valid[:, 0], valid[:, 1], color=col , s=5)
            ax.scatter(valid[0, 0], valid[0, 1], c=col, s=16, marker='x', zorder=4)
            ax.scatter(valid[-1, 0], valid[-1, 1], c=col, s=16, marker='x', zorder=4)

    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('visualization of road network, agent positions and path plans')
    ax.grid(True, alpha=0.3)
    # 设置坐标轴范围
    all_vertices = quads_vertices_np.reshape(-1, 2)
    x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    # 添加一些边距
    margin = 1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    print(f"道路网络范围: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")
    print(f"激活智能体数量: {len(active_agents)}")
    plt.tight_layout()
    plt.show()
    print(simulator.agents_path_plans)

    
    
    


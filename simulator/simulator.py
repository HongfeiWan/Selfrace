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
from conditioning_and_goals import PathPlanner
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
    def __init__(self, config: Dict):
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
        # 获取simulator配置，支持嵌套配置结构
        simulator_config = config.get('simulator', config)
        self.device = torch.device(simulator_config['device'])
        self.num_envs = simulator_config['num_envs']
        self.dt = simulator_config['sim_dt']
        self.map = simulator_config['map_path']
        # 1. 加载地图网络
        # road.py 中的 RoadNetwork 类负责解析地图文件并提供查询接口
        self.road_network = RoadNetwork(self.map, self.device)
        # 2. 初始化共享的空间哈希
        # 使用地图边界来定义哈希网格的范围
        all_verts = self.road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 使用一个固定的 cell_size, 也可以从 config 读取
        cell_size = config.get('hash_cell_size', 20.0)
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)
        # 3. 初始化车辆动力学模型
        # dynamics.py 中的 KinematicBicycleModel 负责根据动作更新车辆状态
        self.dynamics_model = KinematicBicycleModel(config, self.device)
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
        obs_config = config.get('observation', {})
        self.observation_generator = ObservationGenerator(self.road_network, obs_config, self.device)
        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(config, self.device)
        # 9. 初始化路径规划器
        # PathPlanner现在会自动加载所需的数据
        self.path_planner = PathPlanner(map_path=self.map, device=self.device)
        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        # ego_agents_idx is no longer needed as a class attribute
        # 11. 初始化路径规划相关属性
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
        initial_observation = self._get_observation()
        print("Initial observation generated")
        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        self._initialize_path_planning()
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
        observation = self._get_observation()
        
        # 6. 计算奖励（传入Frenet坐标和动作）
        reward, goal_reached = self._calculate_reward(all_collisions, offroad_mask, d, theta_f, actions)

        # 7. 检查是否结束（包含目标到达判断）
        done = self._check_done(all_collisions, offroad_mask, goal_reached)
        
        return observation, reward, done
    
    def _get_observation(self) -> torch.Tensor:
        """
        调用观测生成器为所有智能体生成观测。
        """
        # 修正：移除循环，直接调用已完全向量化的观测生成器
        return self.observation_generator.generate(self.agents_state)

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
        goal_positions = None
        waypoint_reached = None
        
        # 如果有路径规划信息，计算目标奖励
        if self.agents_goal_quad_ids is not None and self.agents_path_plans is not None:
            # 获取智能体当前位置
            agent_positions = self.agents_state[..., :2]  # (B, M, 2)
            
            # 获取目标位置（从路径规划中获取最后一个有效点）
            B, M, path_length, _ = self.agents_path_plans.shape
            goal_positions = torch.zeros((B, M, 2), device=self.device)
            
            # 为每个智能体找到路径中的最后一个有效点作为目标
            for b in range(B):
                for m in range(M):
                    path = self.agents_path_plans[b, m]  # (path_length, 2)
                    # 找到最后一个非零点
                    valid_points = path[path.sum(dim=1) != 0]  # 过滤掉零值点
                    if len(valid_points) > 0:
                        goal_positions[b, m] = valid_points[-1]  # 最后一个有效点作为目标
                    else:
                        # 如果没有有效路径点，使用当前位置
                        goal_positions[b, m] = agent_positions[b, m]
            
            # 计算路点到达状态（简化：如果距离目标很近就认为到达了路点）
            distances_to_goal = torch.norm(agent_positions - goal_positions, dim=-1)
            waypoint_reached = distances_to_goal < 5.0  # 5米内认为到达路点
        
        # 调用奖励计算器
        reward, goal_reached = self.reward_calculator.calculate(
            extended_state,
            all_collisions,
            offroad_mask,
            dt=self.dt,
            goal_positions=goal_positions,
            waypoint_reached=waypoint_reached,
        )
        
        return reward, goal_reached
    
    def _check_done(self, all_collisions: torch.Tensor, offroad_mask: torch.Tensor, goal_reached: torch.Tensor = None) -> torch.Tensor:
        """
        为所有智能体检查是否应该结束。
        """
        # 碰撞或离路都会导致结束
        done = all_collisions | offroad_mask
        
        # 如果提供了目标到达标志，将其也作为结束条件
        if goal_reached is not None:
            done = done | goal_reached
        
        return done
    
    def _initialize_path_planning(self):
        """
        为所有智能体初始化路径规划：
        1. 为每个激活的智能体随机分配一个目标quad_id
        2. 使用plan_path_batch批量生成从起始位置到目标的路径规划
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
        # 获取激活智能体的起始quad_id
        active_indices = torch.where(active_mask)
        start_quad_ids = self.agents_start_quad_ids[active_indices]
        
        # 3. 为所有激活智能体随机分配目标quad_id
        # 获取所有可用的quad_id（排除起始quad_id）
        available_quad_ids = torch.arange(self.road_network.num_quads, device=self.device)
        
        # 为每个激活智能体随机选择一个目标（排除起始位置）
        num_active = len(start_quad_ids)
        goal_quad_ids = torch.zeros(num_active, dtype=torch.long, device=self.device)
        print(f"num_active: {num_active}")
        for i in range(num_active):
            start_id = start_quad_ids[i].item()
            # 排除起始quad_id，从其他quads中随机选择
            available_goals = available_quad_ids[available_quad_ids != start_id]
            if len(available_goals) > 0:
                goal_quad_ids[i] = available_goals[torch.randint(0, len(available_goals), (1,))]
            else:
                # 如果没有其他选择，使用起始位置
                goal_quad_ids[i] = start_id
        # 4. 使用plan_path_batch批量生成路径规划
        path_plans = self.path_planner.plan_path_batch(start_quad_ids, goal_quad_ids)
        # 5. 将结果存储到类属性中
        # 创建完整的目标和路径张量（包含未激活的智能体）
        self.agents_goal_quad_ids = torch.full((B, M), -1, dtype=torch.long, device=self.device)
        
        # 根据最大路径长度创建路径张量
        max_path_length = path_plans.shape[1] if path_plans.shape[1] > 0 else 100  # 默认长度
        self.agents_path_plans = torch.zeros((B, M, max_path_length, 2), device=self.device)
        self.agents_path_lengths = torch.zeros((B, M), dtype=torch.long, device=self.device)
        
        # 将激活智能体的目标quad_id和路径规划填入对应位置
        active_indices = torch.where(active_mask)
        for i, (b, m) in enumerate(zip(active_indices[0], active_indices[1])):
            self.agents_goal_quad_ids[b, m] = goal_quad_ids[i]
            self.agents_path_plans[b, m] = path_plans[i]
            # 计算路径长度（非零坐标点的数量）
            path_coords = path_plans[i]
            path_length = torch.sum(torch.any(path_coords != 0, dim=1)).item()
            self.agents_path_lengths[b, m] = path_length
        
        print(f"Path planning initialized: {num_active} active agents assigned goals and paths")
        print(f"Path plans shape: {self.agents_path_plans.shape}")
    
    def render(self):
        """
        (接口) 可视化当前所有环境的状态。
        """
        print("Rendering function is not implemented yet.")
        pass

if __name__ == '__main__':
    # 这是一个简单的使用示例，用于测试模拟器的基本功能
    # 从配置文件读取配置
    from matplotlib import pyplot as plt
    import numpy as np
    config_path = 'configs/default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    simulator = TeraflowSimulator(config=config)
    initial_obs = simulator.reset()
    print(f"Initial observation batch shape: {initial_obs.shape}")
    # 可视化道路网络和智能体位置（优化版本）
    print("\n=== 可视化道路网络和智能体位置 ===")
    
    # 获取道路网络的四边形顶点
    quads_vertices = simulator.road_network.quads_vertices  # (num_quads, 4, 2)
    quads_vertices_np = quads_vertices.cpu().numpy()
    # 获取智能体状态
    agents_state_np = simulator.agents_state.cpu().numpy()  # (B, M, 7)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    # 高效绘制道路网络 - 使用批量绘制
    print(f"绘制 {len(quads_vertices_np)} 个道路四边形...")
    # 高速绘制所有quads - 使用批量绘制技术
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np
    # 方法1: 使用PatchCollection进行批量绘制（最快）
    print("使用PatchCollection批量绘制quads...")
    patches = []
    road_ids = []
    # 检查是否有road_id信息用于着色
    has_road_ids = hasattr(simulator.road_network, 'lane_ids')
    # 批量创建Polygon对象
    for i in range(len(quads_vertices_np)):
        vertices = quads_vertices_np[i]  # (4, 2)
        polygon = Polygon(vertices, closed=True)
        patches.append(polygon)
        if has_road_ids:
            road_ids.append(simulator.road_network.lane_ids[i].item())
    # 根据road_id进行着色（如果可用）
    if has_road_ids and len(set(road_ids)) > 1:
        print(f"根据road_id着色，发现 {len(set(road_ids))} 条不同的道路")
        unique_road_ids = sorted(list(set(road_ids)))
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(unique_road_ids), vmax=max(unique_road_ids))
        colors = [cmap(norm(rid)) for rid in road_ids]
        p = PatchCollection(patches, alpha=0.3, facecolors=colors, edgecolor='black', linewidth=0.1)
    else:
        # 使用单一颜色
        p = PatchCollection(patches, alpha=0.2, facecolor='lightblue', edgecolor='black', linewidth=0.1)
    # 一次性添加所有quads到图形
    ax.add_collection(p)

    # 绘制激活的智能体
    active_mask = agents_state_np[0, :, 6] > 0.5  # 第一个环境的激活智能体
    active_agents = agents_state_np[0, active_mask]  # 激活智能体的状态
    if len(active_agents) > 0:
        print(f"绘制 {len(active_agents)} 个激活智能体...")
        # 批量绘制智能体位置
        ax.scatter(active_agents[:, 0], active_agents[:, 1], 
                  c='red', s=100, alpha=0.8, label='active agents', zorder=10)
        # 绘制智能体路径规划
        print("绘制智能体路径规划...")
        agents_path_plans_np = simulator.agents_path_plans.cpu().numpy()  # (B, M, max_path_length, 2)
        agents_path_lengths_np = simulator.agents_path_lengths.cpu().numpy()  # (B, M)
        # 获取第一个环境的路径数据
        env_path_plans = agents_path_plans_np[0]  # (M, max_path_length, 2)
        env_path_lengths = agents_path_lengths_np[0]  # (M)
        # 为每个激活的智能体绘制路径
        active_indices = np.where(active_mask)[0]
        colors = plt.cm.Set3(np.linspace(0.5, 1, len(active_indices)))  # 为每个智能体分配不同颜色
        
        for i, agent_idx in enumerate(active_indices):
            path_plan = env_path_plans[agent_idx]  # (max_path_length, 2)
            path_length = env_path_lengths[agent_idx]
            
            if path_length > 0:
                # 只绘制有效的路径点（非零点）
                valid_path = path_plan[:path_length]
                if len(valid_path) > 1:  # 至少需要2个点才能画线
                    ax.scatter(valid_path[:, 0], valid_path[:, 1], 
                           color=colors[i], s=10, alpha=0.7, 
                           label=f'Agent {agent_idx} path', zorder=15)
                    # 在路径起点和终点添加标记
                    ax.scatter(valid_path[0, 0], valid_path[0, 1], 
                             color=colors[i], s=50, marker='o', alpha=0.8, zorder=15)
                    ax.scatter(valid_path[-1, 0], valid_path[-1, 1], 
                             color=colors[i], s=100, marker='*', alpha=0.8, zorder=15)

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
    
    
    
    


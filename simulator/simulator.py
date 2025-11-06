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
    
    # ==================== 3. 使用 OpenGL + Pygame 可视化 active 车辆的路径规划 ====================
    import numpy as np
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (glClearColor, glClear, GL_COLOR_BUFFER_BIT, glMatrixMode, 
                              GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho, 
                              glBegin, glEnd, glVertex2f, glColor3f, glColor4f, GL_QUADS, GL_LINES, 
                              GL_POINTS, GL_POLYGON, GL_LINE_LOOP, glPointSize, glLineWidth, glEnable, glBlendFunc,
                              GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    except Exception as e:
        print(f'请安装 pygame 和 PyOpenGL: pip install pygame PyOpenGL')
        print(f'错误信息: {e}')
        exit(1)
    
    # 获取active掩码：agents_state的第七位（索引6）表示是否active
    active_mask = agents_state[..., 6] > 0.5  # (B, M) - 布尔掩码
    
    # 获取无效标记值
    invalid_marker_value = planner.INVALID_w_lane_id_MARKER
    
    # 准备数据
    quads_vertices_np = rn.quads_vertices.detach().cpu().numpy()  # (N, 4, 2)
    quads_by_id = rn.quads_by_id
    B, M = agents_state.shape[:2]
    b = 0  # 只绘制第一个批次
    
    # 获取所有 active 的 agents
    active_agents = torch.nonzero(active_mask[b], as_tuple=False).squeeze(-1)
    if active_agents.numel() == 0:
        print("Batch 0 中没有active车辆")
        exit(0)
    
    active_agents_list = active_agents.tolist()
    print(f"Batch 0 中有 {len(active_agents_list)} 个active车辆")
    
    # 计算坐标范围
    if quads_vertices_np.shape[0] > 0:
        xs = quads_vertices_np[:, :, 0].reshape(-1)
        ys = quads_vertices_np[:, :, 1].reshape(-1)
        margin = 20.0
        x_min, x_max = float(xs.min() - margin), float(xs.max() + margin)
        y_min, y_max = float(ys.min() - margin), float(ys.max() + margin)
    else:
        x_min, x_max, y_min, y_max = -100.0, 100.0, -100.0, 100.0
    
    # 初始化 Pygame 和 OpenGL
    pygame.init()
    screen = pygame.display.set_mode((1280, 960), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Active Vehicles Path Plans (SPACE: next, ESC: quit)')
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    # 启用透明度支持
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(x_min, x_max, y_min, y_max, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    
    # 当前显示的车辆索引
    cur_idx = 0
    
    # 缩放参数
    zoom_level = 1.0  # 1.0 = 原始视图
    zoom_min = 0.1    # 最小缩放（最大化视野）
    zoom_max = 10.0   # 最大缩放（最小化视野）
    
    # 获取 horizon 参数
    horizon = sim.observation_generator.horizon
    
    def draw_quads():
        """绘制所有 quads 作为背景"""
        glColor3f(0.95, 0.95, 0.7)  # 浅黄色
        for verts in quads_vertices_np:
            glBegin(GL_QUADS)
            glVertex2f(float(verts[0, 0]), float(verts[0, 1]))
            glVertex2f(float(verts[1, 0]), float(verts[1, 1]))
            glVertex2f(float(verts[2, 0]), float(verts[2, 1]))
            glVertex2f(float(verts[3, 0]), float(verts[3, 1]))
            glEnd()
        
        # 绘制 quad 边框
        glColor3f(0.6, 0.6, 0.6)
        glLineWidth(0.5)
        for verts in quads_vertices_np:
            glBegin(GL_LINES)
            for i in range(4):
                glVertex2f(float(verts[i, 0]), float(verts[i, 1]))
                glVertex2f(float(verts[(i+1)%4, 0]), float(verts[(i+1)%4, 1]))
            glEnd()
    
    def draw_vehicle_box(x, y, heading, length, width, r, g, b, line_width=2.0):
        """绘制车辆的矩形框"""
        import math
        # 计算车辆四个角的坐标
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        # 车辆中心在 (x, y)，长度方向沿着 heading
        half_length = length / 2.0
        half_width = width / 2.0
        
        # 四个角（相对于中心）
        corners = [
            (-half_length, -half_width),  # 后左
            (half_length, -half_width),   # 前左
            (half_length, half_width),    # 前右
            (-half_length, half_width)    # 后右
        ]
        
        # 旋转并平移
        world_corners = []
        for cx, cy in corners:
            wx = x + cx * cos_h - cy * sin_h
            wy = y + cx * sin_h + cy * cos_h
            world_corners.append((wx, wy))
        
        # 绘制矩形框
        glColor3f(r, g, b)
        glLineWidth(line_width)
        glBegin(GL_LINES)
        for i in range(4):
            glVertex2f(world_corners[i][0], world_corners[i][1])
            glVertex2f(world_corners[(i+1)%4][0], world_corners[(i+1)%4][1])
        glEnd()
        
        # 绘制前方指示线（车头方向）
        front_x = x + half_length * cos_h
        front_y = y + half_length * sin_h
        glBegin(GL_LINES)
        glVertex2f(x, y)
        glVertex2f(front_x, front_y)
        glEnd()
    
    def draw_horizon_box(ego_x, ego_y, horizon_size):
        """绘制观测范围的圆形"""
        import math
        radius = horizon_size / 2.0
        num_segments = 64  # 圆形分段数，越大越平滑
        
        # 绘制半透明的填充圆
        glColor4f(0.0, 0.8, 0.8, 0.08)  # 青色，8% 不透明度
        glBegin(GL_POLYGON)
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = ego_x + radius * math.cos(angle)
            y = ego_y + radius * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
        
        # 绘制边框圆线
        glColor4f(0.0, 0.8, 0.8, 0.8)  # 青色，80% 不透明度
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = ego_x + radius * math.cos(angle)
            y = ego_y + radius * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
    
    def draw_all_agents(agents_state, current_m):
        """绘制所有 active agents（除了当前选中的）"""
        for m_idx in range(M):
            if agents_state[b, m_idx, 6] < 0.5:  # 不是 active
                continue
            if m_idx == current_m:  # 跳过当前选中的车辆
                continue
            
            # 获取车辆状态：[x, y, heading, vx, vy, length, width]
            state = agents_state[b, m_idx].cpu().numpy()
            x, y, heading = state[0], state[1], state[2]
            length, width = state[4], state[5]
            
            # 绘制灰色车辆框
            draw_vehicle_box(x, y, heading, length, width, 0.4, 0.4, 0.4, line_width=1.5)
    
    def draw_observation_data(neighbors_local, w_lanes_local, w_boundaries_local, ego_x, ego_y, ego_heading, ego_speed):
        """绘制 observation 中的数据（透明显示）"""
        import math
        import numpy as np
        
        if neighbors_local is None:
            return
        
        # 旋转矩阵
        cos_yaw = math.cos(ego_heading)
        sin_yaw = math.sin(ego_heading)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # 1. 绘制观测到的其他车辆（红色框）
        try:
            neighbors_np = neighbors_local.cpu().numpy() if torch.is_tensor(neighbors_local) else neighbors_local
            # neighbors_local: (K, 7) - [dx, dy, dvx, dvy, length, width, active]
            
            for k in range(neighbors_np.shape[0]):
                neighbor = neighbors_np[k]
                dx, dy = neighbor[0], neighbor[1]
                
                # 检查是否是有效的 neighbor
                if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                    continue
                if neighbor[6] < 0.5:  # active flag
                    continue
                
                # 局部坐标转换为世界坐标
                dx_dy = np.array([dx, dy])
                world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                world_x, world_y = world_pos[0], world_pos[1]
                
                # 从相对速度恢复世界速度和朝向
                dvx, dvy = neighbor[2], neighbor[3]
                v_local = np.array([dvx, dvy])
                v_ego_world = np.array([ego_speed * cos_yaw, ego_speed * sin_yaw])
                v_neighbor_world = v_local @ rot.T + v_ego_world
                
                vx_k, vy_k = v_neighbor_world[0], v_neighbor_world[1]
                if (vx_k * vx_k + vy_k * vy_k) < 1e-6:
                    world_heading = ego_heading
                else:
                    world_heading = math.atan2(vy_k, vx_k)
                
                length = neighbor[4]
                width = neighbor[5]
                
                if length <= 0.0 or width <= 0.0:
                    continue
                
                # 绘制红色车辆框
                draw_vehicle_box(world_x, world_y, world_heading, length, width, 
                               1.0, 0.0, 0.0, line_width=2.5)
        except Exception as e:
            print(f"绘制 neighbors 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 绘制 observation 中的 w_lane（蓝色散点）
        try:
            if w_lanes_local is not None:
                wlane_np = w_lanes_local.cpu().numpy() if torch.is_tensor(w_lanes_local) else w_lanes_local
                # w_lanes_local: (num_w_lanes, 2) - [dx, dy] in local frame
                
                # 绘制 w_lane 散点
                glColor4f(0.2, 0.4, 0.9, 0.8)  # 蓝色，80% 不透明度
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(wlane_np.shape[0]):
                    dx, dy = wlane_np[i, 0], wlane_np[i, 1]
                    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                        continue
                    
                    # 局部坐标转世界坐标
                    dx_dy = np.array([dx, dy])
                    world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                    glVertex2f(world_pos[0], world_pos[1])
                glEnd()
        except Exception as e:
            print(f"绘制 w_lanes 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 绘制 boundary（红色散点）
        try:
            if w_boundaries_local is not None:
                boundary_np = w_boundaries_local.cpu().numpy() if torch.is_tensor(w_boundaries_local) else w_boundaries_local
                # w_boundaries_local: (num_w_boundaries, 2) - [dx, dy] in local frame
                
                # 绘制 boundary 散点
                glColor4f(0.9, 0.2, 0.2, 0.8)  # 红色，80% 不透明度
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(boundary_np.shape[0]):
                    dx, dy = boundary_np[i, 0], boundary_np[i, 1]
                    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                        continue
                    
                    dx_dy = np.array([dx, dy])
                    world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                    glVertex2f(world_pos[0], world_pos[1])
                glEnd()
        except Exception as e:
            print(f"绘制 boundaries 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_point_xy(x, y, r, g, b, size=8.0):
        """绘制一个点"""
        glColor3f(r, g, b)
        glPointSize(float(size))
        glBegin(GL_POINTS)
        glVertex2f(float(x), float(y))
        glEnd()
    
    def draw_path_with_arrows(valid_path):
        """绘制路径点、连线和方向箭头"""
        if len(valid_path) == 0:
            return
        
        # 1. 绘制路径连线（蓝色）
        glColor3f(0.2, 0.4, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i in range(len(valid_path) - 1):
            glVertex2f(float(valid_path[i, 0]), float(valid_path[i, 1]))
            glVertex2f(float(valid_path[i+1, 0]), float(valid_path[i+1, 1]))
        glEnd()
        
        # 2. 绘制所有路径点（蓝色小点）
        glColor3f(0.2, 0.4, 0.8)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for p in valid_path:
            glVertex2f(float(p[0]), float(p[1]))
        glEnd()
        
        # 3. 绘制方向箭头（紫色）
        glColor3f(0.5, 0.1, 0.7)
        glLineWidth(1.5)
        arrow_length = 5.0
        for p in valid_path:
            x, y, angle = float(p[0]), float(p[1]), float(p[2])
            dx = arrow_length * np.cos(angle)
            dy = arrow_length * np.sin(angle)
            glBegin(GL_LINES)
            glVertex2f(x, y)
            glVertex2f(x + dx, y + dy)
            glEnd()
        
        # 4. 绘制起点（绿色方块）
        draw_point_xy(valid_path[0, 0], valid_path[0, 1], 0.2, 0.8, 0.2, size=10.0)
        
        # 5. 绘制终点（红色三角 - 用大点代替）
        draw_point_xy(valid_path[-1, 0], valid_path[-1, 1], 0.9, 0.2, 0.2, size=12.0)
    
    # 主循环
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    cur_idx = (cur_idx + 1) % len(active_agents_list)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 处理滚轮缩放
                if event.button == 4:  # 滚轮向上 - 放大
                    zoom_level = min(zoom_level * 1.2, zoom_max)
                elif event.button == 5:  # 滚轮向下 - 缩小
                    zoom_level = max(zoom_level / 1.2, zoom_min)
        
        # 获取当前车辆的数据
        m = active_agents_list[cur_idx]
        path = agents_path_plans[b, m]  # (w_lane_ids_length, 3)
        
        # 过滤无效路径点
        invalid_marker_tensor = torch.tensor(invalid_marker_value, device=path.device, dtype=path.dtype)
        valid_mask = path[:, 0] != invalid_marker_tensor
        valid_path = path[valid_mask].cpu().numpy()
        
        # 获取起点和终点的 poly_id
        start_poly_id = int(start_poly_tensor[b, m].item())
        end_poly_id = int(end_poly_tensor[b, m].item())
        
        # 获取当前车辆的状态
        ego_state = agents_state[b, m].cpu().numpy()
        ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[2]
        ego_length, ego_width = ego_state[4], ego_state[5]
        
        # 获取当前车辆的 observation（使用内部方法获取未展平的数据）
        try:
            # 使用 ObservationGenerator 的内部方法获取结构化数据
            neighbor_states_world = sim.observation_generator._get_nearest_neighbors(agents_state)
            w_lanes_world, w_boundaries_world = sim.observation_generator._get_precomputed_waypoints(agents_state)
            local_state_tmp, neighbors_local_tmp, w_lanes_local_tmp, w_boundaries_local_tmp = \
                sim.observation_generator._world_to_ego_centric(
                    agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
                )
            
            # 提取当前车辆的数据
            neighbors_local = neighbors_local_tmp[b, m]  # (K, 7)
            w_lanes_local = w_lanes_local_tmp[b, m]      # (num_w_lanes, 2)
            w_boundaries_local = w_boundaries_local_tmp[b, m]  # (num_w_boundaries, 2)
            
            has_observation = True
        except Exception as e:
            print(f"获取 observation 失败: {e}")
            has_observation = False
            neighbors_local = None
            w_lanes_local = None
            w_boundaries_local = None
        
        # 更新窗口标题
        pygame.display.set_caption(
            f'Active Vehicles Path Plans - B={b}, M={m} ({cur_idx+1}/{len(active_agents_list)}) '
            f'(SPACE: next, ESC: quit, Scroll: zoom) - Path: {len(valid_path)} pts, Zoom: {zoom_level:.2f}x'
        )
        
        # 更新投影矩阵以应用缩放
        # 以当前车辆为中心进行缩放
        view_width = (x_max - x_min) / zoom_level
        view_height = (y_max - y_min) / zoom_level
        
        view_x_min = ego_x - view_width / 2
        view_x_max = ego_x + view_width / 2
        view_y_min = ego_y - view_height / 2
        view_y_max = ego_y + view_height / 2
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(view_x_min, view_x_max, view_y_min, view_y_max, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
        # 清空并绘制
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # 绘制背景（quads）
        draw_quads()
        
        # 绘制 horizon 观测范围框
        draw_horizon_box(ego_x, ego_y, horizon)
        
        # 绘制所有其他 agents（灰色）
        draw_all_agents(agents_state, m)
        
        # 绘制 observation 数据（透明显示）
        if has_observation:
            ego_speed = ego_state[3]
            draw_observation_data(neighbors_local, w_lanes_local, w_boundaries_local, 
                                ego_x, ego_y, ego_heading, ego_speed)
        
        # 绘制当前车辆（蓝绿色）
        draw_vehicle_box(ego_x, ego_y, ego_heading, ego_length, ego_width, 
                        0.0, 0.8, 0.8, line_width=3.0)
        
        # 绘制路径
        if len(valid_path) > 0:
            draw_path_with_arrows(valid_path)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print(f"可视化结束，共查看了 {len(active_agents_list)} 个active车辆")



    
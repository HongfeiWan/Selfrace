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
from randomize_components import DrivingStyleSampler

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
        profile_config = config.get('training', {}).get('profile', {})
        self.profile_enabled = bool(profile_config.get('enabled', False))
        self.profile_cuda_sync = bool(profile_config.get('cuda_sync', False))
        self.last_step_profile: Dict[str, float] = {}
        self.verbose = simulator_config.get('verbose', False)
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
        self.driving_style_sampler = DrivingStyleSampler(self.device)

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
        self.path_planner = PathPlanner(map_path=self.map_path, device=self.device, verbose=self.verbose)

        # 10. 初始化交通灯/停止线观测与奖励所需的静态几何
        self.traffic_config = simulator_config.get('traffic') or {}
        self.stop_line_width = float(self.traffic_config.get('stop_line_width', 3.5))
        self.stop_line_horizon = float(self.traffic_config.get('stop_line_horizon', obs_config.get('horizon', 100.0)))
        self.stop_line_observation_count = int(self.traffic_config.get('stop_line_observation_count', 5))
        self.red_light_probability = float(self.traffic_config.get('red_light_probability', 0.5))
        network_cfg = config.get('training', {}).get('network', {})
        permutation_dims = network_cfg.get('permutation_feature_dims', [160, 560, 20, 200])
        self.stop_line_feature_dim = int(permutation_dims[2])
        self._prepare_traffic_controls()

        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        self.agents_goal_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的目标quad_id
        self.agents_route_quad_ids: Optional[torch.Tensor] = None  # 中间waypoint + final goal的quad序列
        self.agents_route_target_count: Optional[torch.Tensor] = None
        self.agents_current_route_idx: Optional[torch.Tensor] = None
        self.max_route_targets: int = 4  # N_wp ~ U{0,3} plus one final goal
        self.goal_positions: Optional[torch.Tensor] = None
        self.final_goal_positions: Optional[torch.Tensor] = None
        self.route_candidate_samples: int = int(simulator_config.get('route_candidate_samples', 64))
        self.route_min_goal_distance: float = float(simulator_config.get('route_min_goal_distance', 20.0))
        self.route_max_goal_distance: float = float(simulator_config.get('route_max_goal_distance', 200.0))
        self.route_max_heading_delta: float = float(simulator_config.get('route_max_heading_delta_deg', 60.0)) * torch.pi / 180.0
        self.driving_style_params: Optional[torch.Tensor] = None
        self.traffic_light_states: Optional[torch.Tensor] = None
        self.stop_lines: Optional[torch.Tensor] = None
        self.stop_line_violation: Optional[torch.Tensor] = None

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _current_control_state(self) -> torch.Tensor:
        """返回当前动力学控制状态 [phi, a_long, a_lat]，用于原文式 S(t)。"""
        if self.agents_state is None:
            return torch.empty((0, 0, 3), dtype=torch.float32, device=self.device)
        B, M = self.agents_state.shape[:2]
        control = torch.zeros(B, M, 3, dtype=self.agents_state.dtype, device=self.device)
        fields = (
            ('current_steering_angle', 0),
            ('current_along', 1),
            ('current_alat', 2),
        )
        for name, idx in fields:
            value = getattr(self.dynamics_model, name, None)
            if value is not None and value.numel() == B * M:
                control[:, :, idx] = value.to(device=self.device, dtype=self.agents_state.dtype).view(B, M)
        if self.agents_state.shape[-1] > 6:
            active_mask = self.agents_state[..., 6] > 0.5
            if getattr(self, 'last_done', None) is not None:
                active_mask = active_mask & (~self.last_done)
            control = control * active_mask.unsqueeze(-1).to(dtype=control.dtype)
        return control

    def _profile_now(self) -> float:
        if self.profile_enabled and self.profile_cuda_sync and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.time()

    def _profile_record(self, profile: Optional[Dict[str, float]], name: str, start_time: float) -> float:
        if profile is None:
            return start_time
        now = self._profile_now()
        profile[name] = (now - start_time) * 1000.0
        return now

    def _prepare_traffic_controls(self):
        """从地图交通控制点构造停止线线段。"""
        centers = getattr(self.road_network, 'stop_line_centers', torch.empty((0, 2), device=self.device))
        yaws = getattr(self.road_network, 'stop_line_yaws', torch.empty((0,), device=self.device))
        control_indices = getattr(self.road_network, 'stop_line_control_indices', torch.empty((0,), dtype=torch.long, device=self.device))
        if centers.numel() == 0:
            self.stop_line_segments = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
            self.stop_line_control_indices = torch.empty((0,), dtype=torch.long, device=self.device)
            self.num_traffic_controls = int(getattr(self.road_network, 'traffic_light_locations', torch.empty((0, 2))).shape[0])
            return

        perp = torch.stack([torch.sin(yaws), torch.cos(yaws)], dim=-1)
        half_width = 0.5 * self.stop_line_width
        p0 = centers - perp * half_width
        p1 = centers + perp * half_width
        self.stop_line_segments = torch.stack([p0, p1], dim=1)
        self.stop_line_control_indices = control_indices.to(device=self.device, dtype=torch.long)
        self.num_traffic_controls = int(max(
            int(getattr(self.road_network, 'traffic_light_locations', torch.empty((0, 2))).shape[0]),
            int(self.stop_line_control_indices.max().item()) + 1 if self.stop_line_control_indices.numel() > 0 else 0,
        ))

    def _reset_traffic_lights(self):
        """每个 episode 为每个环境随机化红灯状态。"""
        if self.num_traffic_controls <= 0:
            self.traffic_light_states = torch.empty((self.num_envs, 0), dtype=torch.bool, device=self.device)
            return
        self.traffic_light_states = (
            torch.rand((self.num_envs, self.num_traffic_controls), device=self.device) < self.red_light_probability
        )

    def _red_stop_line_mask(self, batch_size: int, traffic_light_states: torch.Tensor = None) -> torch.Tensor:
        """返回每个环境中每条停止线当前是否对应红灯，形状 (B, S)。"""
        S = self.stop_line_segments.shape[0]
        if S == 0:
            return torch.empty((batch_size, 0), dtype=torch.bool, device=self.device)
        traffic_light_states = self.traffic_light_states if traffic_light_states is None else traffic_light_states
        if traffic_light_states is None or traffic_light_states.numel() == 0:
            return torch.zeros((batch_size, S), dtype=torch.bool, device=self.device)
        traffic_light_states = traffic_light_states.to(device=self.device, dtype=torch.bool)
        control_idx = torch.clamp(self.stop_line_control_indices, 0, traffic_light_states.shape[1] - 1)
        return traffic_light_states[:batch_size, control_idx]

    def _compute_stop_line_observation(
        self,
        agents_state: torch.Tensor,
        traffic_light_states: torch.Tensor = None,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        """构造局部坐标下的红灯停止线观测，形状 (B, M, stop_line_feature_dim)。"""
        B, M, _ = agents_state.shape
        expected_shape = (B, M, self.stop_line_feature_dim)
        if out is None or out.shape != expected_shape or out.device != agents_state.device or out.dtype != agents_state.dtype:
            out = torch.zeros(expected_shape, dtype=agents_state.dtype, device=agents_state.device)
        else:
            out.zero_()
        S = self.stop_line_segments.shape[0]
        max_lines = min(self.stop_line_observation_count, self.stop_line_feature_dim // 4)
        if S == 0 or max_lines <= 0:
            return out

        red_mask = self._red_stop_line_mask(B, traffic_light_states=traffic_light_states)
        centers = self.stop_line_segments.mean(dim=1)
        ego_pos = agents_state[..., :2]
        dist_sq = (ego_pos.unsqueeze(2) - centers.view(1, 1, S, 2)).pow(2).sum(dim=-1)
        active_mask = agents_state[..., 6] > 0.5
        dist_sq = dist_sq.masked_fill(~red_mask.unsqueeze(1), float('inf'))
        dist_sq = dist_sq.masked_fill(~active_mask.unsqueeze(-1), float('inf'))
        dist_sq = dist_sq.masked_fill(dist_sq > self.stop_line_horizon ** 2, float('inf'))

        k_eff = min(max_lines, S)
        nearest_dist_sq, nearest_idx = torch.topk(dist_sq, k=k_eff, dim=-1, largest=False)
        valid_lines = torch.isfinite(nearest_dist_sq)
        selected_segments = self.stop_line_segments[nearest_idx.clamp_min(0)]
        endpoints_world = selected_segments.reshape(B, M, k_eff * 2, 2)

        ego_yaw = agents_state[..., 2]
        cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
        rot_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1)
        ], dim=-2)
        rel = endpoints_world - ego_pos.unsqueeze(2)
        endpoints_local = torch.bmm(
            rel.reshape(B * M, k_eff * 2, 2),
            rot_matrix.reshape(B * M, 2, 2)
        ).reshape(B, M, k_eff * 2, 2)
        valid_points = valid_lines.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, M, k_eff * 2)
        endpoints_local = torch.where(valid_points.unsqueeze(-1), endpoints_local, torch.zeros_like(endpoints_local))

        flat = endpoints_local.reshape(B, M, k_eff * 4)
        out[:, :, :flat.shape[-1]] = flat
        return out

    def _update_stop_line_observation(self, agents_state: torch.Tensor):
        self.stop_lines = self._compute_stop_line_observation(agents_state)

    @staticmethod
    def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def _compute_stop_line_violation(self, states_t0: torch.Tensor, states_t1: torch.Tensor,
                                     effective_mask: torch.Tensor) -> torch.Tensor:
        """检测车辆一步内是否穿越红灯停止线。"""
        B, M, _ = states_t1.shape
        S = self.stop_line_segments.shape[0]
        if S == 0:
            return torch.zeros((B, M), dtype=torch.bool, device=self.device)

        red_mask = self._red_stop_line_mask(B)
        if red_mask.numel() == 0:
            return torch.zeros((B, M), dtype=torch.bool, device=self.device)

        p0 = states_t0[..., :2].unsqueeze(2)
        p1 = states_t1[..., :2].unsqueeze(2)
        q0 = self.stop_line_segments[:, 0, :].view(1, 1, S, 2)
        q1 = self.stop_line_segments[:, 1, :].view(1, 1, S, 2)

        ego_motion = p1 - p0
        line_vec = q1 - q0
        o1 = self._cross2d(ego_motion, q0 - p0)
        o2 = self._cross2d(ego_motion, q1 - p0)
        o3 = self._cross2d(line_vec, p0 - q0)
        o4 = self._cross2d(line_vec, p1 - q0)
        eps = 1e-6
        intersects = (o1 * o2 <= eps) & (o3 * o4 <= eps)

        moved = ego_motion.squeeze(2).norm(dim=-1) > 0.05
        heading = torch.stack([torch.cos(states_t1[..., 2]), torch.sin(states_t1[..., 2])], dim=-1)
        forward_motion = (ego_motion.squeeze(2) * heading).sum(dim=-1) > 0.0
        violation = (
            intersects
            & red_mask.unsqueeze(1)
            & effective_mask.unsqueeze(-1)
            & moved.unsqueeze(-1)
            & forward_motion.unsqueeze(-1)
        )
        return violation.any(dim=-1)

    def reset(self, return_observation: bool = True):
        """
        重置所有环境，并返回所有智能体的初始观测。
        Returns:
            torch.Tensor: 一批初始观测, 形状为 (B, M, obs_dim)。
        """
        self._log("Resetting simulator environments...")
        # 重置动力学模型的状态变量，避免不同episode之间的tensor大小不匹配
        self.dynamics_model.reset_control_state()
        self._log("Reset dynamics model state - cleared for fresh initialization")

        # 使用 WorldInitializer 来生成一批新的世界状态，包括起始quad_id
        self.agents_state, _, self.agents_start_quad_ids = self.world_initializer.initialize_world(self.num_envs)
        
        # 重置reward风格参数
        self.reward_calculator.reset_episode()

        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        active_mask = self.agents_state[..., 6] > 0.5
        self.driving_style_params = self.driving_style_sampler.sample_driving_style_params(
            self.num_envs, self.world_initializer.max_agents
        ).to(self.device)
        self.driving_style_params = torch.where(
            active_mask.unsqueeze(-1),
            self.driving_style_params,
            torch.ones_like(self.driving_style_params)
        )
        
        # 重置done状态
        self.last_done = None

        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        self._initialize_path_planning()
        self._reset_traffic_lights()
        self.stop_line_violation = torch.zeros_like(active_mask, dtype=torch.bool)
        self._update_stop_line_observation(self.agents_state)

        initial_observation = None
        if return_observation:
            # 生成初始观测。路径规划先完成，训练首帧才能拿到同步的局部路径特征。
            self._log("Generating initial observation...")
            initial_observation = self.observation_generator.generate(
                self.agents_state,
                control_state=self._current_control_state(),
                driving_style_params=self.driving_style_params,
            )
            self._log(initial_observation.shape)
            self._log("Initial observation generated")
        self._log("Path planning initialized")
        self._log(f"Reset complete. World state shape: {self.agents_state.shape}")
        
        return initial_observation
    
    def step(self, actions: torch.Tensor, debug_collision: bool = False,
             return_observation: bool = True):
        """
        让所有环境向前步进一个时间步。所有智能体都根据actions更新。
        Args:
            actions (torch.Tensor): 形状为 (B, M, action) 的动作张量。
            debug_collision (bool): 是否为碰撞检测器开启调试模式。
        Returns:
            return_observation=True:
                (observation, reward, done)
            return_observation=False:
                (reward, done)
        """
        if self.agents_state is None:
            raise RuntimeError("Must call reset() before calling step().")
        profile = {} if self.profile_enabled else None
        profile_start = self._profile_now() if profile is not None else 0.0
        profile_cursor = profile_start
        
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
        style_flat = None
        if self.driving_style_params is not None:
            style_flat = self.driving_style_params.contiguous().view(Bsz * Msz, 4)
        effective_flat = effective_mask.contiguous().view(Bsz * Msz)
        # 调用动力学（全批次大小恒定），再把无效位置用旧状态覆盖
        new_states_flat = self.dynamics_model.step(
            states_flat, actions_flat, self.dt,
            style_params=style_flat,
            active_mask=effective_flat,
            vehicle_length=self.agents_state[..., 4].contiguous().view(Bsz * Msz),
        )  # (B*M, 4)
        new_states = new_states_flat.view(Bsz, Msz, 4)
        keep_old = ~effective_mask
        self.agents_state[..., :4] = torch.where(keep_old.unsqueeze(-1), self.agents_state[..., :4], new_states)
        self.stop_line_violation = self._compute_stop_line_violation(states_t0, self.agents_state, effective_mask)
        profile_cursor = self._profile_record(profile, 'dynamics_ms', profile_cursor)

        # 2. 离路检测
        is_on_road = torch.ones_like(active_mask) # 默认在路上
        check_mask = effective_mask
        active_states = self.agents_state[check_mask]
        # OffroadChecker 需要 [x, y, yaw, length, width]
        states_for_checker = active_states[:, [0, 1, 2, 4, 5]]
        active_is_on_road = self.offroad_checker.check_on_road(states_for_checker)
        is_on_road[check_mask] = active_is_on_road
        offroad_mask = (~is_on_road) & check_mask # (B, M)
        profile_cursor = self._profile_record(profile, 'offroad_ms', profile_cursor)

        # 3. 动态碰撞检测：排除上一帧已done和本帧刚离路的车辆。
        # 离路本身已经是终止事件，不能再让它在同一帧把其他车撞成done。
        collision_exclude_mask = offroad_mask
        if hasattr(self, 'last_done') and self.last_done is not None:
            collision_exclude_mask = collision_exclude_mask | self.last_done
        states_t0_for_collision = states_t0.clone()
        states_t1_for_collision = self.agents_state.clone()
        states_t0_for_collision[..., 6] = torch.where(collision_exclude_mask, 0.0, states_t0_for_collision[..., 6])
        states_t1_for_collision[..., 6] = torch.where(collision_exclude_mask, 0.0, states_t1_for_collision[..., 6])
        
        collision_check_result = self.collision_checker.check(
            states_t0_for_collision, states_t1_for_collision, debug=debug_collision, debug_env_idx=0
        )
        all_collisions = collision_check_result
        profile_cursor = self._profile_record(profile, 'collision_ms', profile_cursor)

        # 4. 计算Frenet坐标信息
        vehicle_positions = self.agents_state[..., :2]  # (B, M, 2) - x, y
        vehicle_headings = self.agents_state[..., 2]    # (B, M) - heading
        d, theta_f = self.road_network.calculate_frenet_coordinates(vehicle_positions, vehicle_headings, self.spatial_hash)
        profile_cursor = self._profile_record(profile, 'frenet_ms', profile_cursor)

        # 5. 计算奖励。中间 waypoint 到达会给 R_goal，但只有最终目标到达才 done。
        reward, final_goal_reached, intermediate_goal_reached = self._calculate_reward(
            all_collisions, offroad_mask, d, theta_f, actions
        )
        profile_cursor = self._profile_record(profile, 'reward_ms', profile_cursor)

        # 6. 检查是否结束：中间 waypoint 不终止 episode。
        done = all_collisions | offroad_mask | final_goal_reached

        # 保存done状态供本次observation和后续step使用（累积done状态，一旦done就保持done）。
        if hasattr(self, 'last_done') and self.last_done is not None:
            self.last_done = self.last_done | done
        else:
            self.last_done = done.clone()

        # 7. 推进 route target。训练导航只依赖紧凑 route state，不再维护 dense path。
        self._advance_route_targets(intermediate_goal_reached)
        profile_cursor = self._profile_record(profile, 'route_update_ms', profile_cursor)

        observation = None
        if return_observation:
            # 8. 生成新的观测（排除已经done的车辆，包括当前step刚done的车辆）。
            if hasattr(self, 'last_done') and self.last_done is not None:
                agents_state_for_obs = self.agents_state.clone()
                agents_state_for_obs[..., 6] = torch.where(self.last_done, 0.0, agents_state_for_obs[..., 6])
            else:
                agents_state_for_obs = self.agents_state

            self._update_stop_line_observation(agents_state_for_obs)
            observation = self.observation_generator.generate(
                agents_state_for_obs,
                control_state=self._current_control_state(),
                driving_style_params=self.driving_style_params,
            )
            profile_cursor = self._profile_record(profile, 'observation_ms', profile_cursor)

        if profile is not None:
            profile['total_step_ms'] = (self._profile_now() - profile_start) * 1000.0
            self.last_step_profile = profile

        if return_observation:
            return observation, reward, done
        return reward, done
    
    def _calculate_reward(self, all_collisions: torch.Tensor, offroad_mask: torch.Tensor, d: torch.Tensor, theta_f: torch.Tensor, actions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为所有智能体计算奖励。
        Args:
            all_collisions (torch.Tensor): 碰撞状态 (B, M)
            offroad_mask (torch.Tensor): 离路状态 (B, M)
            d (torch.Tensor): Frenet横向距离 (B, M)
            theta_f (torch.Tensor): Frenet角度误差 (B, M)
            actions (torch.Tensor): 动作索引 (B, M)，用于直接获取jerk值
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                reward, final_goal_reached, intermediate_goal_reached
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
            
            # 1) 动力学控制状态始终保持 B*M 顺序，reshape 后按有效掩码置零。
            along_all = self.dynamics_model.current_along
            alat_all = self.dynamics_model.current_alat
            if along_all is None:
                full_along = torch.zeros((B, M), device=self.device)
            else:
                full_along = along_all.view(B, M) * effective_mask.float()
            if alat_all is None:
                full_alat = torch.zeros((B, M), device=self.device)
            else:
                full_alat = alat_all.view(B, M) * effective_mask.float()

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

        goal_positions = self.goal_positions
        if goal_positions is None:
            goal_positions = self.agents_state[..., :2]
        current_target_is_waypoint = self._current_target_is_intermediate()

        # 调用奖励计算器，这个速度很快
        reward, goal_reached = self.reward_calculator.calculate(
            extended_state,
            all_collisions,
            offroad_mask,
            dt=self.dt,
            goal_positions=goal_positions,
            waypoint_reached=current_target_is_waypoint,
            stop_line_violation=self.stop_line_violation,
        )
        
        # 过滤掉非active或已done车辆的奖励（与动力学一致的有效掩码）
        active_mask = self.agents_state[..., 6] > 0.5  # (B, M)
        if hasattr(self, 'last_done') and self.last_done is not None:
            not_done_mask = ~self.last_done
        else:
            not_done_mask = torch.ones_like(active_mask, dtype=torch.bool)
        effective_mask = active_mask & not_done_mask
        reward = reward * effective_mask.float()  # 非有效车辆的奖励设为0，等价于“done后奖励不再更新”
        goal_reached = goal_reached & effective_mask
        # 碰撞/离路终止的同一步即便碰巧进入中间 waypoint 半径，也不再推进 route。
        still_valid_route = ~(all_collisions | offroad_mask)
        intermediate_goal_reached = goal_reached & current_target_is_waypoint & still_valid_route
        final_goal_reached = goal_reached & (~current_target_is_waypoint)

        self.extend_state = extended_state # 用于传入网络
        return reward, final_goal_reached, intermediate_goal_reached
    
    def _sample_next_route_quads(
        self,
        prev_quad: torch.Tensor,
        valid_mask: torch.Tensor,
        min_distance: float,
        max_distance: float,
        max_heading_delta: float,
    ) -> torch.Tensor:
        """
        在上一目标附近采样下一目标：优先满足距离与车道朝向约束，失败时逐步放宽。
        这对应原文中 waypoint 序列的采样方式，避免目标序列在地图上随机跳跃。
        """
        B, M = prev_quad.shape
        if self.road_network.num_quads <= 0:
            return torch.full_like(prev_quad, -1, dtype=torch.int32)

        K = max(1, self.route_candidate_samples)
        sampled = torch.randint(
            0,
            self.road_network.num_quads,
            (B, M, K),
            dtype=torch.long,
            device=self.device,
        )
        safe_prev = torch.clamp(prev_quad.long(), 0, self.road_network.num_quads - 1)
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)
        quad_dirs = self.road_network.quad_directions

        prev_centers = quad_centers[safe_prev].unsqueeze(2)
        prev_dirs = quad_dirs[safe_prev].unsqueeze(2)
        cand_centers = quad_centers[sampled]
        cand_dirs = quad_dirs[sampled]

        distances = torch.norm(cand_centers - prev_centers, dim=-1)
        cos_delta = (cand_dirs * prev_dirs).sum(dim=-1).clamp(-1.0, 1.0)
        not_same = sampled != safe_prev.unsqueeze(-1)
        base_mask = valid_mask.unsqueeze(-1) & not_same

        strict_mask = (
            base_mask
            & (distances >= min_distance)
            & (distances <= max_distance)
            & (cos_delta >= torch.cos(torch.as_tensor(max_heading_delta, device=self.device)))
        )
        relaxed_mask = (
            base_mask
            & (distances >= 0.5 * min_distance)
            & (distances <= 1.5 * max_distance)
            & (cos_delta >= torch.cos(torch.as_tensor(max_heading_delta * 2.0, device=self.device)))
        )

        def choose_from(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            counts = mask.sum(dim=-1)
            max_counts = torch.clamp(counts, min=1)
            ranks = torch.floor(torch.rand((B, M), device=self.device) * max_counts.float()).long()
            cumsum = mask.long().cumsum(dim=-1)
            selected = (cumsum == (ranks.unsqueeze(-1) + 1)) & mask
            selected_idx = selected.float().argmax(dim=-1)
            chosen = sampled.gather(2, selected_idx.unsqueeze(-1)).squeeze(-1).to(torch.int32)
            return chosen, counts > 0

        strict_choice, has_strict = choose_from(strict_mask)
        relaxed_choice, has_relaxed = choose_from(relaxed_mask)
        fallback_choice, has_fallback = choose_from(base_mask)

        chosen = torch.where(has_strict, strict_choice, relaxed_choice)
        chosen = torch.where(has_strict | has_relaxed, chosen, fallback_choice)
        return torch.where(valid_mask & (has_strict | has_relaxed | has_fallback), chosen, torch.full_like(chosen, -1))

    def _sample_route_quad_ids(self, active_mask: torch.Tensor, start_i32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样 final goal 与 N_wp ~ U{0,3} 个中间 waypoint，并构造受距离/朝向约束的目标序列。"""
        B, M = active_mask.shape
        valid_start = active_mask & (start_i32 >= 0)
        num_intermediate = torch.randint(
            0,
            self.max_route_targets,
            (B, M),
            dtype=torch.long,
            device=self.device,
        )
        target_count = torch.where(valid_start, num_intermediate + 1, torch.zeros_like(num_intermediate))
        route_slots = torch.arange(self.max_route_targets, device=self.device).view(1, 1, -1)
        route_mask = route_slots < target_count.unsqueeze(-1)
        route_quads = torch.full((B, M, self.max_route_targets), -1, dtype=torch.int32, device=self.device)

        first_target = torch.randint(
            0,
            self.road_network.num_quads,
            (B, M),
            dtype=torch.int32,
            device=self.device,
        )
        if self.road_network.num_quads > 1:
            same_as_start = valid_start & (first_target == start_i32)
            first_target = torch.where(
                same_as_start,
                ((first_target.long() + 1) % self.road_network.num_quads).to(torch.int32),
                first_target,
            )
        route_quads[..., 0] = torch.where(valid_start & route_mask[..., 0], first_target, route_quads[..., 0])

        for route_idx in range(1, self.max_route_targets):
            valid_next = valid_start & route_mask[..., route_idx]
            prev_quad = route_quads[..., route_idx - 1]
            route_quads[..., route_idx] = self._sample_next_route_quads(
                prev_quad,
                valid_next,
                self.route_min_goal_distance,
                self.route_max_goal_distance,
                self.route_max_heading_delta,
            )
            target_count = torch.where(
                valid_next & (route_quads[..., route_idx] < 0),
                torch.full_like(target_count, route_idx),
                target_count,
            )
            route_mask = route_slots < target_count.unsqueeze(-1)

        return route_quads, target_count

    def get_route_state(self, clone: bool = False) -> Dict[str, torch.Tensor]:
        """Return compact routing state used to reconstruct navigation observations on demand."""
        if (
            self.agents_route_quad_ids is None
            or self.agents_route_target_count is None
            or self.agents_current_route_idx is None
        ):
            return {}
        route_state = {
            'route_quad_ids': self.agents_route_quad_ids.to(dtype=torch.int32),
            'target_count': self.agents_route_target_count.to(dtype=torch.int16),
            'current_idx': self.agents_current_route_idx.to(dtype=torch.int16),
        }
        if clone:
            route_state = {key: value.detach().clone() for key, value in route_state.items()}
        return route_state

    def _route_state_tensors(self, route_state: Optional[Dict[str, torch.Tensor]] = None):
        if route_state is None:
            return self.agents_route_quad_ids, self.agents_route_target_count, self.agents_current_route_idx
        route_quads = route_state.get('route_quad_ids')
        target_count = route_state.get('target_count')
        current_idx = route_state.get('current_idx')
        if route_quads is None or target_count is None or current_idx is None:
            return None, None, None
        return (
            route_quads.to(device=self.device, dtype=torch.int32),
            target_count.to(device=self.device, dtype=torch.long),
            current_idx.to(device=self.device, dtype=torch.long),
        )

    def _route_quad_at(self, indices: torch.Tensor, route_quad_ids: torch.Tensor = None) -> torch.Tensor:
        if route_quad_ids is None:
            route_quad_ids = self.agents_route_quad_ids
        if route_quad_ids is None:
            return torch.full(indices.shape, -1, dtype=torch.int32, device=self.device)
        route_quad_ids = route_quad_ids.to(device=self.device, dtype=torch.int32)
        safe_indices = torch.clamp(indices, 0, self.max_route_targets - 1).long()
        return torch.gather(route_quad_ids, 2, safe_indices.unsqueeze(-1)).squeeze(-1)

    def _refresh_goal_positions(self):
        """刷新当前目标点和最终目标点坐标。"""
        if self.agents_route_quad_ids is None or self.agents_current_route_idx is None:
            return
        current_quad = self._route_quad_at(self.agents_current_route_idx)
        final_idx = torch.clamp(self.agents_route_target_count - 1, min=0)
        final_quad = self._route_quad_at(final_idx)
        current_positions = self.agents_state[..., :2]
        current_goal_positions = self.path_planner.get_quad_centers(current_quad.to(torch.long))
        final_goal_positions = self.path_planner.get_quad_centers(final_quad.to(torch.long))
        current_valid = current_quad >= 0
        final_valid = final_quad >= 0
        self.goal_positions = torch.where(current_valid.unsqueeze(-1), current_goal_positions, current_positions)
        self.final_goal_positions = torch.where(final_valid.unsqueeze(-1), final_goal_positions, current_positions)
        self.agents_goal_quad_ids = final_quad

    def _current_target_is_intermediate(self, route_state: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        _, target_count, current_idx = self._route_state_tensors(route_state)
        if target_count is None or current_idx is None:
            B, M, _ = self.agents_state.shape
            return torch.zeros((B, M), dtype=torch.bool, device=self.device)
        return (
            (target_count > 0)
            & (current_idx < (target_count - 1))
        )

    def _remaining_route_goals_local(
        self,
        agents_state: torch.Tensor,
        route_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return remaining route targets as explicit G(t) in ego coordinates."""
        B, M, _ = agents_state.shape
        goal_slots = self.max_route_targets
        goals_local = torch.zeros((B, M, goal_slots, 2), dtype=agents_state.dtype, device=self.device)
        goals_valid = torch.zeros((B, M, goal_slots), dtype=torch.bool, device=self.device)
        route_quads, target_count, current_idx = self._route_state_tensors(route_state)
        if route_quads is None or target_count is None or current_idx is None:
            return goals_local, goals_valid

        slot_offsets = torch.arange(goal_slots, device=self.device, dtype=torch.long).view(1, 1, goal_slots)
        target_indices = current_idx.unsqueeze(-1) + slot_offsets
        goals_valid = target_indices < target_count.unsqueeze(-1)
        safe_indices = torch.clamp(target_indices, 0, goal_slots - 1)
        goal_quads = torch.gather(route_quads, 2, safe_indices)
        goals_valid = goals_valid & (goal_quads >= 0) & (agents_state[..., 6] > 0.5).unsqueeze(-1)

        goal_world = self.path_planner.get_quad_centers(goal_quads.reshape(-1).to(torch.long)).view(B, M, goal_slots, 2)
        ego_pos = agents_state[..., :2]
        ego_yaw = agents_state[..., 2]
        cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
        rot_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1)
        ], dim=-2)
        rel = goal_world - ego_pos.unsqueeze(2)
        goals_local = torch.bmm(
            rel.reshape(B * M, goal_slots, 2),
            rot_matrix.reshape(B * M, 2, 2),
        ).reshape(B, M, goal_slots, 2)
        goals_local = torch.where(goals_valid.unsqueeze(-1), goals_local, torch.zeros_like(goals_local))
        return goals_local, goals_valid

    def get_navigation_observation(
        self,
        agents_state: torch.Tensor = None,
        route_state: Optional[Dict[str, torch.Tensor]] = None,
        w_lane_keep_mask: torch.Tensor = None,
        w_lane_ids: torch.Tensor = None,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Build the paper-style navigation observation package.

        Rows [0:max_route_targets) encode explicit remaining G(t): dx, dy, valid.
        Rows after that align with W_lane observations and encode graph route
        distance to the current target: absolute distance, relative distance, valid.
        """
        if agents_state is None:
            agents_state = self.agents_state
        if agents_state is None:
            return torch.empty((0, 0, 0, 3), dtype=torch.float32, device=self.device)

        B, M, _ = agents_state.shape
        goal_slots = self.max_route_targets
        lane_slots = self.observation_generator.num_w_lanes
        expected_shape = (B, M, goal_slots + lane_slots, 3)
        if out is None or out.shape != expected_shape or out.device != agents_state.device or out.dtype != agents_state.dtype:
            navigation = torch.zeros(expected_shape, dtype=agents_state.dtype, device=agents_state.device)
        else:
            navigation = out
            navigation.zero_()

        goals_local, goals_valid = self._remaining_route_goals_local(agents_state, route_state=route_state)
        navigation[:, :, :goal_slots, :2] = goals_local
        navigation[:, :, :goal_slots, 2] = goals_valid.to(dtype=agents_state.dtype)

        route_quads, _, current_idx = self._route_state_tensors(route_state)
        if route_quads is None or current_idx is None:
            return navigation

        if w_lane_ids is None:
            w_lane_ids, _ = self.observation_generator.get_w_lane_ids_for_agents(agents_state)
        else:
            w_lane_ids = w_lane_ids.to(device=self.device, dtype=torch.long)
        if w_lane_keep_mask is not None:
            keep_mask = w_lane_keep_mask.to(device=self.device, dtype=torch.bool)
            w_lane_ids = torch.where(keep_mask, w_lane_ids, torch.full_like(w_lane_ids, -1))
        current_goal_quad = self._route_quad_at(current_idx, route_quad_ids=route_quads)
        route_abs = self.path_planner.route_distances_from_w_lanes_to_goal_quads(w_lane_ids, current_goal_quad)
        active = agents_state[..., 6] > 0.5
        route_valid = torch.isfinite(route_abs) & (w_lane_ids >= 0) & active.unsqueeze(-1)
        masked_route = route_abs.masked_fill(~route_valid, float('inf'))
        min_route = masked_route.amin(dim=-1, keepdim=True)
        min_route = torch.where(torch.isfinite(min_route), min_route, torch.zeros_like(min_route))
        route_rel = route_abs - min_route

        lane_slice = navigation[:, :, goal_slots:goal_slots + lane_slots]
        lane_slice[..., 0] = torch.where(route_valid, route_abs, torch.zeros_like(route_abs))
        lane_slice[..., 1] = torch.where(route_valid, route_rel, torch.zeros_like(route_rel))
        lane_slice[..., 2] = route_valid.to(dtype=agents_state.dtype)
        return navigation

    def _advance_route_targets(self, intermediate_goal_reached: torch.Tensor):
        """中间 waypoint 到达后推进到下一个 route target；final goal 不在这里推进。"""
        if self.agents_current_route_idx is None or self.agents_route_target_count is None:
            return
        advance_mask = intermediate_goal_reached & self._current_target_is_intermediate()
        if not bool(advance_mask.any().item()):
            self._refresh_goal_positions()
            return
        max_idx = torch.clamp(self.agents_route_target_count - 1, min=0)
        next_idx = torch.minimum(self.agents_current_route_idx + advance_mask.long(), max_idx)
        self.agents_current_route_idx = next_idx
        self._refresh_goal_positions()

    def _initialize_path_planning(self):
        """
        为所有智能体初始化路径规划：
        1. 为每个激活智能体采样 final goal 与 0~3 个中间 waypoint
        2. waypoint 序列按距离与车道朝向约束顺序生成
        3. 只保存紧凑 route state；导航观测在训练/推理时按需重建
        """
        if self.agents_state is None:
            return
        B, M, _ = self.agents_state.shape
        active_mask = self.agents_state[..., 6] > 0.5
        if not hasattr(self, 'agents_start_quad_ids') or self.agents_start_quad_ids is None:
            self._log("Warning: No start quad IDs available for path planning")
            return

        start_i32 = self.agents_start_quad_ids.to(dtype=torch.int32, device=self.device)
        route_quads, target_count = self._sample_route_quad_ids(active_mask, start_i32)
        self.agents_route_quad_ids = route_quads
        self.agents_route_target_count = target_count
        self.agents_current_route_idx = torch.zeros((B, M), dtype=torch.long, device=self.device)
        self._refresh_goal_positions()

if __name__ == '__main__':
    # 这是一个简单的使用示例，用于测试模拟器的基本功能
    # 从配置文件读取配置
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    from matplotlib.patches import Polygon, Circle
    from matplotlib.collections import PatchCollection
    
    # 基于文件位置解析项目根目录，避免依赖当前工作目录
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    config_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
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
    from ddppo import decompose_observation, build_network_features, current_navigation
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
            current_navigation(simulator),
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

            # neighbors_local: [dx, dy, heading_x, heading_y, dvx, dvy, length, width, z, active]
            neigh = neighbors_local_t[0, first_agent_idx] if neighbors_local_t is not None else None
            if neigh is not None:
                neigh_np = neigh.detach().to('cpu').numpy()
                if neigh_np.ndim >= 2 and neigh_np.shape[-1] >= 6:
                    # 有效点：active>0.5 或者 长宽>0
                    active_mask = neigh_np[..., -1] > 0.5 if neigh_np.shape[-1] >= 7 else np.ones(neigh_np.shape[0], dtype=bool)
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
                                if row.shape[0] >= 10:
                                    heading_x, heading_y = float(row[2]), float(row[3])
                                    dvx_local, dvy_local = float(row[4]), float(row[5])
                                    nlen = float(row[6])
                                    nwid = float(row[7])
                                else:
                                    heading_x = heading_y = None
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
                                if heading_x is not None and math.hypot(heading_x, heading_y) > 1e-3:
                                    nyaw_world = eyaw + math.atan2(heading_y, heading_x)
                                elif speed_mag > 1e-2:
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
    
    # 统一图形样式
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title('road graph and agent positions')
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
                    current_navigation(simulator),
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

import torch
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
training_dir = os.path.join(parent_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
from spatial_hash import SpatialHash
from tensor_budget import nearest_indices, workspace_bytes
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
                计算设备由构造函数的 ``device`` 参数提供，而不是从配置读取。
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
        self.geometry_workspace_mb = float(simulator_config.get('geometry_workspace_mb', 64.0))
        self.geometry_workspace_bytes = workspace_bytes(device, self.geometry_workspace_mb)

        # 1. 加载地图网络
        # road.py 中的 RoadNetwork 类负责解析地图文件并提供查询接口
        self.road_network = RoadNetwork(self.map_path, self.device)
        self.road_network.workspace_bytes = self.geometry_workspace_bytes

        # 2. 初始化共享的空间哈希
        # 使用地图边界来定义哈希网格的范围
        all_verts = self.road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 网格范围来自地图边界；分辨率由配置中的 cell_size 控制。
        hash_config = simulator_config['hash']
        cell_size = hash_config['hash_cell_size']
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)

        # 3. 初始化车辆动力学模型
        # dynamics.py 中的 KinematicBicycleModel 负责根据动作更新车辆状态
        self.dynamics_model = KinematicBicycleModel(config, self.device)
        self.driving_style_sampler = DrivingStyleSampler(self.device)

        # 4. 初始化离路检测器 (传入共享的哈希对象)
        self.offroad_checker = OffroadChecker(
            self.road_network, self.spatial_hash, workspace_mb=self.geometry_workspace_mb,
            compile_enabled=simulator_config.get('execution', {}).get('compile', False),
        )

        # 5. 初始化碰撞检测器 (传入共享的哈希对象)
        # collision.py 负责检测智能体之间的碰撞
        self.collision_checker = CollisionChecker(config, self.spatial_hash)

        # 6. 初始化世界状态管理器
        # world_init.py 负责在 reset 时初始化场景
        self.world_initializer = WorldInitializer(self.road_network, self.offroad_checker, self.collision_checker, config)

        # 7. 初始化观测生成器
        # observation.py 负责为每个自车生成局部观测
        obs_config = simulator_config['observation']
        self.observation_generator = ObservationGenerator(self.road_network, obs_config, self.device, self.spatial_hash, self.geometry_workspace_mb)

        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.device)

        # 9. 初始化路径规划器
        # PathPlanner现在会自动加载所需的数据
        self.path_planner = PathPlanner(map_path=self.map_path, device=self.device, verbose=self.verbose)
        self.path_planner.workspace_bytes = self.geometry_workspace_bytes

        # 10. 初始化交通灯/停止线观测与奖励所需的静态几何
        self.traffic_config = simulator_config.get('traffic') or {}
        self.stop_line_width = float(self.traffic_config.get('stop_line_width', 3.5))
        self.stop_line_horizon = float(self.traffic_config.get('stop_line_horizon', obs_config.get('horizon', 100.0)))
        self.stop_line_observation_count = int(self.traffic_config.get('stop_line_observation_count', 5))
        self.red_light_probability = float(self.traffic_config.get('red_light_probability', 0.5))
        # Each observed stop line contributes two (x, y) endpoints.
        self.stop_line_feature_dim = self.stop_line_observation_count * 4
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
        self.route_local_candidate_count: int = int(simulator_config.get('route_local_candidate_count', 128))
        self.route_local_candidate_chunk_quads: int = int(simulator_config.get('route_local_candidate_chunk_quads', 1024))
        self.route_min_goal_distance: float = float(simulator_config.get('route_min_goal_distance', 20.0))
        self.route_max_goal_distance: float = float(simulator_config.get('route_max_goal_distance', 200.0))
        self.route_max_heading_delta: float = float(simulator_config.get('route_max_heading_delta_deg', 60.0)) * torch.pi / 180.0
        self.reset_chunk_worlds: int = max(1, int(simulator_config.get('reset_chunk_worlds', 256)))
        self._route_local_candidate_ids: Optional[torch.Tensor] = None
        self.driving_style_params: Optional[torch.Tensor] = None
        self.traffic_light_states: Optional[torch.Tensor] = None
        self.stop_lines: Optional[torch.Tensor] = None
        self.stop_line_violation: Optional[torch.Tensor] = None
        self._active_selection = None
        self._geometry = None

    def active_selection(self, mask=None):
        """Compact once per frame and reuse slots across rollout and simulation."""
        if self._active_selection is None:
            if mask is None:
                mask = self.agents_state[..., 6] > 0.5
                if self.last_done is not None:
                    mask = mask & ~self.last_done
            slots = mask.reshape(-1).nonzero(as_tuple=False).flatten()
            width = self.agents_state.shape[1]
            self._active_selection = (slots, slots // width, slots % width, mask)
        return self._active_selection

    def selected_geometry(self, world_indices, agent_indices):
        if self._geometry is None:
            states = self.agents_state[world_indices, agent_indices]
            _, quads = self.road_network.find_nearest_lanes(states[:, :2], spatial_hash=self.spatial_hash)
            d, theta = self.road_network.frenet_from_indices(states[:, :2], states[:, 2], quads[:, 0])
            return quads, d[:, None], theta[:, None]
        return tuple(value[world_indices, agent_indices, None] for value in self._geometry)

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

        perp = torch.stack([-torch.sin(yaws), torch.cos(yaws)], dim=-1)
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
        traffic_light_states = traffic_light_states.to(device=self.device)
        control_idx = torch.clamp(self.stop_line_control_indices, 0, traffic_light_states.shape[1] - 1)
        states = traffic_light_states[:batch_size, control_idx]
        # bool is the current binary controller; integer phases reserve yellow
        # and unknown without silently treating either one as green.
        red = states if states.dtype == torch.bool else states == 1
        valid_control = (self.stop_line_control_indices >= 0) & (self.stop_line_control_indices < traffic_light_states.shape[1])
        return red & valid_control[None]

    def traffic_input_sets(self, agents_state, traffic_states, groups):
        """Bounded optional network inputs, in the ego frame, with explicit masks.

        Lights: stop-line endpoints / 200m, red/yellow/green, valid.
        Crosswalks: center / 200m, heading unit vector, half extents / 20m, valid.
        Integer light phases: 0 unknown, 1 red, 2 yellow, 3 green.
        """
        result = {}
        B, M = agents_state.shape[:2]
        ego = agents_state.reshape(-1, agents_state.shape[-1])
        def local(points):
            delta = points - ego[:, None, :2]
            c, s = ego[:, 2].cos()[:, None], ego[:, 2].sin()[:, None]
            return torch.stack((delta[..., 0] * c + delta[..., 1] * s,
                                -delta[..., 0] * s + delta[..., 1] * c), dim=-1)
        for group in groups:
            count = group.element_count
            out = agents_state.new_zeros((B * M, count, group.element_dim))
            if group.name == 'traffic_lights':
                centers = self.stop_line_segments.mean(1)
            else:
                centers = self.road_network.crosswalks[:, :2]
            ids = nearest_indices(ego[:, :2], centers, count, self.geometry_workspace_bytes)
            if centers.shape[0] > 0:
                safe = ids.clamp_min(0)
                valid = (ids >= 0) & ((ego[:, None, :2] - centers[safe]).square().sum(-1) <= self.stop_line_horizon ** 2)
                if group.name == 'traffic_lights':
                    points = self.stop_line_segments[safe]
                    out[..., :2] = (local(points[:, :, 0]) / 200.).clamp(-1, 1)
                    out[..., 2:4] = (local(points[:, :, 1]) / 200.).clamp(-1, 1)
                    controls = self.stop_line_control_indices[safe]
                    if traffic_states is None or traffic_states.shape[-1] == 0:
                        valid &= False
                    else:
                        worlds = torch.arange(B, device=ego.device).repeat_interleave(M)
                        phases = traffic_states[worlds[:, None], controls.clamp(0, traffic_states.shape[-1] - 1)]
                        valid &= (controls >= 0) & (controls < traffic_states.shape[-1])
                        if phases.dtype == torch.bool:
                            out[..., 4], out[..., 6] = phases, ~phases
                        else:
                            for channel, phase in enumerate((1, 2, 3), start=4):
                                out[..., channel] = phases == phase
                            valid &= (phases >= 1) & (phases <= 3)
                else:
                    rectangles = self.road_network.crosswalks[safe]
                    out[..., :2] = (local(rectangles[..., :2]) / 200.).clamp(-1, 1)
                    yaw = rectangles[..., 2] - ego[:, None, 2]
                    out[..., 2], out[..., 3] = yaw.cos(), yaw.sin()
                    out[..., 4:6] = (rectangles[..., 3:5] / 40.).clamp(0, 1)
                out[..., -1] = valid & (ego[:, None, 6] > .5)
                out *= out[..., -1:]
            result[group.name] = out.view(B, M, count, group.element_dim)
        return result

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
        k_eff = min(max_lines, S)
        # Keep the original nearest-RED-line contract without an ego-by-all-
        # signals expansion, including when future maps contain many signals.
        flat = agents_state.reshape(-1, agents_state.shape[-1])
        nearest_dist_sq = flat.new_full((B * M, k_eff), float('inf'))
        nearest_idx = torch.zeros((B * M, k_eff), dtype=torch.long, device=self.device)
        columns = min(S, 64)
        rows = max(1, self.geometry_workspace_bytes // (64 * (columns + 2 * k_eff)))
        for start in range(0, B * M, rows):
            end = min(start + rows, B * M)
            worlds = torch.arange(start, end, device=self.device) // M
            best_d, best_i = nearest_dist_sq[start:end], nearest_idx[start:end]
            for left in range(0, S, columns):
                right = min(left + columns, S)
                distance = (flat[start:end, None, :2] - centers[None, left:right]).square().sum(-1)
                valid = red_mask[worlds, left:right] & (flat[start:end, None, 6] > .5)
                valid &= distance <= self.stop_line_horizon ** 2
                distance = distance.masked_fill(~valid, float('inf'))
                ids = torch.arange(left, right, device=self.device).expand(end - start, -1)
                candidates = torch.cat((best_i, ids), dim=1)
                best_d, chosen = torch.cat((best_d, distance), dim=1).topk(k_eff, dim=1, largest=False)
                best_i = candidates.gather(1, chosen)
            nearest_dist_sq[start:end], nearest_idx[start:end] = best_d, best_i
        nearest_dist_sq = nearest_dist_sq.view(B, M, k_eff)
        nearest_idx = nearest_idx.view(B, M, k_eff)
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

    def _compute_stop_line_violation(self, states_t0, states_t1, effective_mask):
        """Bounded active-agent/line stream, including correct segment bounds."""
        B, M = effective_mask.shape
        result = torch.zeros_like(effective_mask)
        lines = self.stop_line_segments.shape[0]
        if lines == 0:
            return result
        _, worlds, agents, _ = self.active_selection(effective_mask)
        red = self._red_stop_line_mask(B)
        columns = min(64, lines)
        rows = max(1, self.geometry_workspace_bytes // (128 * columns))
        for left in range(0, worlds.numel(), rows):
            w, a = worlds[left:left + rows], agents[left:left + rows]
            p0, p1 = states_t0[w, a, :2][:, None], states_t1[w, a, :2][:, None]
            motion = p1 - p0
            yaw = states_t1[w, a, 2]
            forward = (motion[:, 0] * torch.stack((yaw.cos(), yaw.sin()), -1)).sum(-1) > 0
            moving = motion[:, 0].norm(dim=-1) > .05
            hit = torch.zeros_like(forward)
            for start in range(0, lines, columns):
                segments = self.stop_line_segments[start:start + columns]
                q0, q1 = segments[None, :, 0], segments[None, :, 1]
                line = q1 - q0
                o1, o2 = self._cross2d(motion, q0 - p0), self._cross2d(motion, q1 - p0)
                o3, o4 = self._cross2d(line, p0 - q0), self._cross2d(line, p1 - q0)
                # Orientation tests alone incorrectly intersect disjoint collinear segments.
                bounds = ((torch.minimum(p0, p1) <= torch.maximum(q0, q1) + 1e-6)
                          & (torch.minimum(q0, q1) <= torch.maximum(p0, p1) + 1e-6)).all(-1)
                hit |= ((o1 * o2 <= 1e-6) & (o3 * o4 <= 1e-6) & bounds
                        & red[w, start:start + columns]).any(-1)
            result[w, a] = hit & moving & forward
        return result

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
        self._active_selection = None
        self._geometry = None

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

    def _reset_world_indices(self, world_indices: torch.Tensor):
        reset_count = int(world_indices.numel())
        new_state, _, new_start_quad_ids = self.world_initializer.initialize_world(
            reset_count
        )
        new_state = new_state.to(self.device)
        self.agents_state[world_indices] = new_state
        self.agents_start_quad_ids[world_indices] = new_start_quad_ids.to(
            device=self.device, dtype=self.agents_start_quad_ids.dtype
        )

        self.reward_calculator.reset_worlds(world_indices)
        active_mask = new_state[..., 6] > 0.5
        new_style = self.driving_style_sampler.sample_driving_style_params(
            reset_count, self.world_initializer.max_agents
        ).to(self.device)
        new_style = torch.where(
            active_mask.unsqueeze(-1), new_style, torch.ones_like(new_style)
        )
        self.driving_style_params[world_indices] = new_style

        if self.last_done is None:
            self.last_done = torch.zeros(
                (self.num_envs, self.world_initializer.max_agents),
                dtype=torch.bool,
                device=self.device,
            )
        self.last_done[world_indices] = False
        self.dynamics_model.reset_worlds(
            world_indices, self.num_envs, self.world_initializer.max_agents
        )

        route_quads, target_count, current_idx = self._sample_path_planning_state(
            new_state, new_start_quad_ids
        )
        self.agents_route_quad_ids[world_indices] = route_quads
        self.agents_route_target_count[world_indices] = target_count
        self.agents_current_route_idx[world_indices] = current_idx

        if self.num_traffic_controls > 0:
            new_lights = (
                torch.rand((reset_count, self.num_traffic_controls), device=self.device)
                < self.red_light_probability
            )
            if self.traffic_light_states.dtype != torch.bool:
                new_lights = torch.where(new_lights, 1, 3).to(self.traffic_light_states.dtype)
            self.traffic_light_states[world_indices] = new_lights
        self.stop_line_violation[world_indices] = False

    def reset_worlds(self, world_mask: torch.Tensor, return_observation: bool = False):
        """Reset selected worlds in adaptive chunks and preserve every other world."""
        if self.agents_state is None:
            raise RuntimeError("Must call reset() before reset_worlds().")
        world_mask = world_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if world_mask.shape != (self.num_envs,):
            raise ValueError(
                f"world_mask must have shape {(self.num_envs,)}, got {tuple(world_mask.shape)}"
            )
        world_indices = torch.where(world_mask)[0]
        reset_count = int(world_indices.numel())
        if reset_count == 0:
            return None

        self._active_selection = None
        self._geometry = None

        chunk_size = min(reset_count, self.reset_chunk_worlds)
        start = 0
        while start < reset_count:
            end = min(start + chunk_size, reset_count)
            try:
                self._reset_world_indices(world_indices[start:end])
                start = end
            except RuntimeError as exc:
                is_oom = self.device.type == 'cuda' and (
                    isinstance(exc, getattr(torch.cuda, 'OutOfMemoryError', RuntimeError))
                    or 'out of memory' in str(exc).lower()
                )
                if not is_oom or chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                self.reset_chunk_worlds = min(self.reset_chunk_worlds, chunk_size)
                torch.cuda.empty_cache()

        self._refresh_goal_positions()
        self._update_stop_line_observation(self.agents_state)

        if not return_observation:
            return None
        return self.observation_generator.generate(
            self.agents_state,
            control_state=self._current_control_state(),
            driving_style_params=self.driving_style_params,
        )
    
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
        
        actions = actions.to(self.device).reshape(self.agents_state.shape[:2]).long()
        slots, world_idx, agent_idx, effective_mask = self.active_selection()
        states_t0 = self.agents_state
        # Stage the next world. No recurrent state is committed until every
        # geometric check and the reward calculation have succeeded.
        next_pose, next_controls = self.dynamics_model.propose_selected(
            states_t0, actions, self.dt, slots, self.driving_style_params
        )
        states_t1 = states_t0.clone()
        states_t1[world_idx, agent_idx, :4] = next_pose
        stop_violation = self._compute_stop_line_violation(states_t0, states_t1, effective_mask)
        profile_cursor = self._profile_record(profile, 'dynamics_ms', profile_cursor)

        selected_states = states_t1[world_idx, agent_idx]
        selected_on_road = self.offroad_checker.check_on_road(selected_states[:, [0, 1, 2, 4, 5]])
        offroad_mask = torch.zeros_like(effective_mask)
        offroad_mask[world_idx, agent_idx] = ~selected_on_road
        profile_cursor = self._profile_record(profile, 'offroad_ms', profile_cursor)

        collision_check_result = self.collision_checker.check(
            states_t0, states_t1, debug=debug_collision, debug_env_idx=0,
            active_mask_override=effective_mask & ~offroad_mask, profile=profile,
        )
        all_collisions = collision_check_result[0] if isinstance(collision_check_result, tuple) else collision_check_result
        profile_cursor = self._profile_record(profile, 'collision_ms', profile_cursor)

        _, quads = self.road_network.find_nearest_lanes(selected_states[:, :2], spatial_hash=self.spatial_hash)
        packed_d, packed_theta = self.road_network.frenet_from_indices(
            selected_states[:, :2], selected_states[:, 2], quads[:, 0]
        )
        geometry_quads = torch.full_like(actions, -1)
        d = torch.zeros_like(states_t0[..., 0])
        theta_f = torch.zeros_like(d)
        geometry_quads[world_idx, agent_idx] = quads[:, 0]
        d[world_idx, agent_idx] = packed_d
        theta_f[world_idx, agent_idx] = packed_theta
        profile_cursor = self._profile_record(profile, 'frenet_ms', profile_cursor)

        reward, final_goal_reached, intermediate_goal_reached = self._calculate_reward_selected(
            states_t1, slots, world_idx, agent_idx, next_controls, all_collisions,
            offroad_mask, packed_d, packed_theta, actions, stop_violation,
        )
        self.agents_state = states_t1
        self.dynamics_model.commit_selected(slots, next_controls)
        self.stop_line_violation = stop_violation
        self._geometry = (geometry_quads, d, theta_f)
        self._active_selection = None
        profile_cursor = self._profile_record(profile, 'reward_ms', profile_cursor)

        # 6. 检查是否结束：中间 waypoint 不终止 episode。
        done = all_collisions | offroad_mask | final_goal_reached
        reward_components = {}
        for name, value in getattr(self.reward_calculator, 'last_reward_components', {}).items():
            reward_components[name] = (value * effective_mask.float()).detach()
        self.last_step_train_info = {
            'effective_mask': effective_mask.detach(),
            'collision_mask': (all_collisions & effective_mask).detach(),
            'offroad_mask': (offroad_mask & effective_mask).detach(),
            'final_goal_mask': (final_goal_reached & effective_mask).detach(),
            'intermediate_goal_mask': (intermediate_goal_reached & effective_mask).detach(),
            'done_mask': (done & effective_mask).detach(),
            'reward_components': reward_components,
        }

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
    
    def _reward_chunk(self, states, slots, world_idx, agent_idx, controls,
                      collisions, offroad, d, theta, actions, stop_violation, intermediate):
        selected = states[world_idx, agent_idx]
        jerk = self.dynamics_model.discrete_action_space.get_action(actions[world_idx, agent_idx])
        extended = torch.cat((selected[:, :4], controls[:, 1:3], jerk, theta[:, None], d[:, None]), dim=-1)
        goals = selected[:, :2] if self.goal_positions is None else self.goal_positions[world_idx, agent_idx]
        return self.reward_calculator.calculate_selected(
            extended, slots, collisions[world_idx, agent_idx], offroad[world_idx, agent_idx],
            self.dt, goals, intermediate, stop_violation[world_idx, agent_idx],
        )

    def _calculate_reward_selected(self, states, slots, world_idx, agent_idx, controls,
                                   collisions, offroad, d, theta, actions, stop_violation):
        from tensor_budget import is_cuda_oom
        reward = states.new_zeros(states.shape[:2])
        final = torch.zeros_like(collisions)
        waypoint = torch.zeros_like(collisions)
        dense_components = {}
        diagnostics = self.config.get('training', {}).get('diagnostics', {}).get('enabled', False)
        intermediate = self._current_target_is_intermediate()[world_idx, agent_idx]
        ceiling = max(1, self.geometry_workspace_bytes // 1024)
        rows = min(ceiling, getattr(self, '_reward_chunk_limit', ceiling))
        start = 0
        while start < slots.numel():
            end = min(start + rows, slots.numel())
            w, a = world_idx[start:end], agent_idx[start:end]
            components = None
            try:
                packed_reward, reached, components = self._reward_chunk(
                    states, slots[start:end], w, a, controls[start:end],
                    collisions, offroad, d[start:end], theta[start:end],
                    actions, stop_violation, intermediate[start:end],
                )
                reward[w, a] = packed_reward
                final[w, a] = reached & ~intermediate[start:end]
                waypoint[w, a] = reached & intermediate[start:end] & ~(collisions | offroad)[w, a]
                if diagnostics:
                    for name, value in components.items():
                        if name not in dense_components:
                            dense_components[name] = torch.zeros_like(reward)
                        dense_components[name][w, a] = value
                start = end
            except RuntimeError as exc:
                if states.device.type != 'cuda' or not is_cuda_oom(exc) or rows == 1:
                    raise
                rows = max(1, rows // 2)
                self._reward_chunk_limit = rows
        self.reward_calculator.last_reward_components = dense_components
        return reward, final, waypoint
    
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
        routable_quad_ids = self._routable_quad_ids()
        if routable_quad_ids.numel() == 0:
            return torch.full_like(prev_quad, -1, dtype=torch.int32)

        K = max(1, self.route_candidate_samples)
        safe_prev = torch.clamp(prev_quad.long(), 0, self.road_network.num_quads - 1)
        local_candidates = self._route_local_candidates()
        if local_candidates is not None and local_candidates.numel() > 0:
            pool_size = int(local_candidates.shape[-1])
            sampled_pos = torch.randint(0, pool_size, (B * M, K), dtype=torch.long, device=self.device)
            sampled = local_candidates[safe_prev.reshape(-1).unsqueeze(1), sampled_pos].view(B, M, K)
        else:
            sampled_pos = torch.randint(
                0,
                int(routable_quad_ids.numel()),
                (B, M, K),
                dtype=torch.long,
                device=self.device,
            )
            sampled = routable_quad_ids[sampled_pos]
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)
        quad_dirs = self.road_network.quad_directions

        prev_centers = quad_centers[safe_prev].unsqueeze(2)
        prev_dirs = quad_dirs[safe_prev].unsqueeze(2)
        cand_centers = quad_centers[sampled]
        cand_dirs = quad_dirs[sampled]

        distances = torch.norm(cand_centers - prev_centers, dim=-1)
        cos_delta = (cand_dirs * prev_dirs).sum(dim=-1).clamp(-1.0, 1.0)
        not_same = sampled != safe_prev.unsqueeze(-1)
        reachable = torch.ones_like(not_same, dtype=torch.bool, device=self.device)
        reachable_fn = getattr(self.path_planner, 'are_goal_quads_reachable_from_quads', None)
        if callable(reachable_fn):
            reachable = reachable_fn(
                safe_prev.unsqueeze(-1).expand_as(sampled),
                sampled,
            )
        base_mask = valid_mask.unsqueeze(-1) & not_same & reachable

        def choose_from(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            counts = mask.sum(dim=-1)
            max_counts = torch.clamp(counts, min=1)
            ranks = torch.floor(torch.rand((B, M), device=self.device) * max_counts.float()).long()
            cumsum = mask.long().cumsum(dim=-1)
            selected = (cumsum == (ranks.unsqueeze(-1) + 1)) & mask
            selected_idx = selected.float().argmax(dim=-1)
            chosen = sampled.gather(2, selected_idx.unsqueeze(-1)).squeeze(-1).to(torch.int32)
            return chosen, counts > 0

        # Appendix A.1 says to gradually relax the 20--200 m / 60 degree
        # constraints when a waypoint cannot be sampled (for example at a
        # dead end).  Keep reachability mandatory at every stage so a relaxed
        # sample cannot silently create an impossible route.
        relaxation_stages = (
            (1.0, 1.0, 1.0),
            (0.75, 1.25, 1.5),
            (0.5, 1.5, 2.0),
        )
        chosen = torch.full_like(prev_quad, -1, dtype=torch.int32)
        has_choice = torch.zeros_like(valid_mask, dtype=torch.bool)
        for min_scale, max_scale, heading_scale in relaxation_stages:
            stage_mask = (
                base_mask
                & (distances >= min_scale * min_distance)
                & (distances <= max_scale * max_distance)
                & (
                    cos_delta
                    >= torch.cos(
                        torch.as_tensor(
                            min(float(torch.pi), heading_scale * max_heading_delta),
                            device=self.device,
                        )
                    )
                )
            )
            stage_choice, stage_has_choice = choose_from(stage_mask)
            take_stage = (~has_choice) & stage_has_choice
            chosen = torch.where(take_stage, stage_choice, chosen)
            has_choice |= stage_has_choice

        # Last-resort relaxation retains only the reachability and distinct
        # point constraints, matching the paper's stated intent at dead ends.
        fallback_choice, has_fallback = choose_from(base_mask)
        take_fallback = (~has_choice) & has_fallback
        chosen = torch.where(take_fallback, fallback_choice, chosen)
        has_choice |= has_fallback
        return torch.where(valid_mask & has_choice, chosen, torch.full_like(chosen, -1))

    def _routable_quad_ids(self) -> torch.Tensor:
        routable = getattr(self.path_planner, 'routable_quad_ids', None)
        if torch.is_tensor(routable) and routable.numel() > 0:
            routable = routable.to(device=self.device, dtype=torch.long)
            routable = routable[(routable >= 0) & (routable < self.road_network.num_quads)]
            if routable.numel() > 0:
                return routable
        return torch.arange(self.road_network.num_quads, device=self.device, dtype=torch.long)

    def _project_to_routable_quads(self, quad_ids: torch.Tensor) -> torch.Tensor:
        project_fn = getattr(self.path_planner, 'project_quads_to_routable', None)
        if callable(project_fn):
            return project_fn(quad_ids).to(device=self.device, dtype=torch.int32)
        return quad_ids.to(device=self.device, dtype=torch.int32)

    def _sample_uniform_route_quads(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Independently sample each active vehicle's first goal over the map."""
        routable_quad_ids = self._routable_quad_ids()
        sampled = torch.full(valid_mask.shape, -1, dtype=torch.int32, device=self.device)
        if routable_quad_ids.numel() == 0:
            return sampled
        positions = torch.randint(
            0,
            int(routable_quad_ids.numel()),
            valid_mask.shape,
            dtype=torch.long,
            device=self.device,
        )
        uniform_goals = routable_quad_ids[positions].to(torch.int32)
        return torch.where(valid_mask, uniform_goals, sampled)

    def _route_local_candidates(self) -> Optional[torch.Tensor]:
        """Nearest routable quad candidates for each quad, built once and reused during reset."""
        if self._route_local_candidate_ids is not None:
            return self._route_local_candidate_ids

        routable_quad_ids = self._routable_quad_ids()
        if routable_quad_ids.numel() == 0:
            self._route_local_candidate_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
            return self._route_local_candidate_ids

        candidate_count = min(
            max(1, int(self.route_local_candidate_count)),
            int(routable_quad_ids.numel()),
        )
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)
        routable_centers = quad_centers[routable_quad_ids]
        positions = nearest_indices(
            quad_centers, routable_centers, candidate_count,
            self.geometry_workspace_bytes, self.route_local_candidate_chunk_quads,
        )
        self._route_local_candidate_ids = routable_quad_ids[positions]
        return self._route_local_candidate_ids

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

        routable_quad_ids = self._routable_quad_ids()
        if routable_quad_ids.numel() == 0:
            return route_quads, torch.zeros_like(target_count)
        # The first route target is uniform over the map and independent for
        # every vehicle.  Only subsequent targets use the local constraints.
        first_target = self._sample_uniform_route_quads(valid_start & route_mask[..., 0])
        route_quads[..., 0] = torch.where(valid_start & route_mask[..., 0], first_target, route_quads[..., 0])
        target_count = torch.where(
            valid_start & route_mask[..., 0] & (route_quads[..., 0] < 0),
            torch.zeros_like(target_count),
            target_count,
        )
        route_mask = route_slots < target_count.unsqueeze(-1)

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

        route_quads = torch.where(
            (valid_start.unsqueeze(-1) & route_mask),
            self._project_to_routable_quads(route_quads),
            route_quads,
        )
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
        max_idx = torch.clamp(self.agents_route_target_count - 1, min=0)
        next_idx = torch.minimum(self.agents_current_route_idx + advance_mask.long(), max_idx)
        self.agents_current_route_idx = next_idx
        self._refresh_goal_positions()

    def _sample_path_planning_state(
        self,
        agents_state: torch.Tensor,
        start_quad_ids: torch.Tensor,
    ):
        B, M = agents_state.shape[:2]
        active_mask = agents_state[..., 6] > 0.5
        start_i32 = start_quad_ids.to(dtype=torch.int32, device=self.device)
        route_quads, target_count = self._sample_route_quad_ids(active_mask, start_i32)
        current_idx = torch.zeros((B, M), dtype=torch.long, device=self.device)
        return route_quads, target_count, current_idx

    def _initialize_path_planning(self):
        """
        为所有智能体初始化路径规划：
        1. 为每个激活智能体采样 final goal 与 0~3 个中间 waypoint
        2. waypoint 序列按距离与车道朝向约束顺序生成
        3. 只保存紧凑 route state；导航观测在训练/推理时按需重建
        """
        if self.agents_state is None:
            return
        if not hasattr(self, 'agents_start_quad_ids') or self.agents_start_quad_ids is None:
            self._log("Warning: No start quad IDs available for path planning")
            return

        (
            self.agents_route_quad_ids,
            self.agents_route_target_count,
            self.agents_current_route_idx,
        ) = self._sample_path_planning_state(self.agents_state, self.agents_start_quad_ids)
        self._refresh_goal_positions()

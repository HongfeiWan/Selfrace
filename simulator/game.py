import os
import sys
import json
import math
from typing import Any, Optional

import torch

# 将工程根目录加入 sys.path，方便从 simulator / training 导入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator import TeraflowSimulator
from training.ddppo import ddppo
from training.network import SimplePolicyNet, convert_path_world_to_ego
from utils.pygame_utils import visualize_path_planning


class FeatureBuilder:
    """
    复用训练用的特征构造逻辑，将 observation + simulator 内部信息
    转换为轻量策略网络 SimplePolicyNet 所需的扁平特征。
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            simulator_config = config.get("simulator", {})
            obs_config = simulator_config.get("observation", {})
            dynamics_config = simulator_config.get("dynamics", {})
            reward_config = simulator_config.get("reward", {})
        else:
            simulator_config = config.simulator
            obs_config = simulator_config.observation
            dynamics_config = simulator_config.dynamics
            reward_config = simulator_config.reward

        # 与 training/ddppo.py 中 __init__ 的定义保持一致
        self.local_state_feature_dim = 8  # [0,0,cos,sin,speed_norm,w_norm,l_norm,active]
        self.path_plan_length = int(obs_config.get("num_navigation_chains", 0))
        self.path_plan_feature_dim = max(self.path_plan_length, 0) * 2  # 仅使用 dx, dy
        self.reward_param_dim = 10

        num_w_boundaries = int(obs_config.get("num_w_boundaries", 0))
        boundary_feat_dim = int(obs_config.get("boundary_feature_dim", 2))
        self.w_boundary_feature_dim = num_w_boundaries * boundary_feat_dim

        num_w_lanes = int(obs_config.get("num_w_lanes", 0))
        self.w_lane_feature_dim = num_w_lanes * 2  # dx, dy

        num_neighbors = int(obs_config.get("num_neighbors", 0))
        neighbor_feat_dim = int(obs_config.get("neighbor_feature_dim", 0))
        self.neighbor_feature_dim = num_neighbors * neighbor_feat_dim

        self.stop_line_feature_dim = int(simulator_config.get("stop_line_feature_dim", 20))

        self.total_feature_dim = (
            self.local_state_feature_dim
            + self.path_plan_feature_dim
            + self.reward_param_dim
            + self.w_boundary_feature_dim
            + self.w_lane_feature_dim
            + self.stop_line_feature_dim
            + self.neighbor_feature_dim
        )

        self.speed_min = float(dynamics_config.get("min_velocity", -2.0))
        self.speed_mid = 0.0
        self.speed_max = float(dynamics_config.get("max_velocity", 20.0))
        self.w_min = float(dynamics_config.get("vehicle_width_min", 0.8))
        self.w_max = float(dynamics_config.get("vehicle_width_max", 3.0))
        self.l_min = float(dynamics_config.get("vehicle_length_min", 0.8))
        self.l_max = float(dynamics_config.get("vehicle_length_max", 7.0))
        self.dvx_min = -2.0
        self.dvx_mid = 0.0
        self.dvx_max = 20.0

        reward_param_mins = torch.tensor(
            [
                float(reward_config.get("delta_goal_min", 2.0)),
                float(reward_config.get("collision_alpha_min", 0.0)),
                float(reward_config.get("boundary_alpha_min", 0.0)),
                float(reward_config.get("comfort_alpha_min", 0.0)),
                float(reward_config.get("l_align_alpha_min", 2.5e-4)),
                float(reward_config.get("vel_align_alpha_min", 0.0)),
                float(reward_config.get("l_center_alpha_min", 2.5e-4)),
                float(reward_config.get("center_bias_alpha_min", -0.5)),
                float(reward_config.get("reverse_alpha_min", 2.5e-4)),
                float(reward_config.get("stop_line_alpha_min", 0.0)),
            ],
            dtype=torch.float32,
        )
        reward_param_maxs = torch.tensor(
            [
                float(reward_config.get("delta_goal_max", 12.0)),
                float(reward_config.get("collision_alpha_max", 3.0)),
                float(reward_config.get("boundary_alpha_max", 3.0)),
                float(reward_config.get("comfort_alpha_max", 0.1)),
                float(reward_config.get("l_align_alpha_max", 2.5e-2)),
                float(reward_config.get("vel_align_alpha_max", 1.0)),
                float(reward_config.get("l_center_alpha_max", 7.5e-3)),
                float(reward_config.get("center_bias_alpha_max", 0.5)),
                float(reward_config.get("reverse_alpha_max", 7.5e-3)),
                float(reward_config.get("stop_line_alpha_max", 1.0)),
            ],
            dtype=torch.float32,
        )
        self.reward_param_mins = reward_param_mins
        self.reward_param_maxs = reward_param_maxs

        self.observation_horizon = float(obs_config.get("horizon", 200.0))

    # === 与 ddppo 中相同的辅助函数 ===
    def _pad_or_trim(
        self,
        tensor: Optional[torch.Tensor],
        target_dim: int,
        B: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if target_dim <= 0:
            return torch.zeros(B, M, 0, device=device, dtype=dtype)
        if tensor is None:
            return torch.zeros(B, M, target_dim, device=device, dtype=dtype)
        if tensor.dim() == 2:
            tensor = tensor.view(B, M, -1)
        elif tensor.dim() == 3 and tensor.shape[0] == B and tensor.shape[1] == M:
            pass
        else:
            tensor = tensor.reshape(B, M, -1)
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            return tensor[..., :target_dim]
        pad = torch.zeros(B, M, target_dim - current_dim, device=device, dtype=dtype)
        return torch.cat([tensor, pad], dim=-1)

    def _normalize_reward_params(self, reward_params: torch.Tensor) -> torch.Tensor:
        B, M, _ = reward_params.shape
        mins = self.reward_param_mins.view(1, 1, -1).to(reward_params.device).expand(B, M, -1)
        maxs = self.reward_param_maxs.view(1, 1, -1).to(reward_params.device).expand(B, M, -1)
        denoms = torch.clamp(maxs - mins, min=1e-6)
        normalized = (reward_params - mins) / denoms
        center_bias_idx = 7
        center_bias = torch.clamp(
            normalized[..., center_bias_idx : center_bias_idx + 1] * 2.0 - 1.0, -1.0, 1.0
        )
        before = torch.clamp(normalized[..., :center_bias_idx], 0.0, 1.0)
        after = torch.clamp(normalized[..., center_bias_idx + 1 :], 0.0, 1.0)
        return torch.cat([before, center_bias, after], dim=-1)

    def _reshape_reward_params(
        self,
        reward_params: Optional[torch.Tensor],
        B: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if reward_params is None or (
            hasattr(reward_params, "numel") and reward_params.numel() == 0
        ):
            tensor = torch.zeros(B, M, 10, device=device, dtype=dtype)
            return self._normalize_reward_params(tensor)
        tensor = reward_params.to(device)
        if tensor.dim() == 2 and tensor.shape[0] == B * M:
            tensor = tensor.view(B, M, -1)
        elif tensor.dim() == 2 and tensor.shape[0] == B and tensor.shape[1] == 10:
            tensor = tensor.unsqueeze(1).expand(-1, M, -1)
        elif tensor.dim() == 3 and tensor.shape[0] == B and tensor.shape[1] == M:
            pass
        else:
            tensor = tensor.view(B, M, -1)
        if tensor.shape[-1] < 10:
            pad = torch.zeros(B, M, 10 - tensor.shape[-1], device=device, dtype=dtype)
            tensor = torch.cat([tensor, pad], dim=-1)
        return self._normalize_reward_params(tensor[..., :10])

    def _normalize_relative(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor / self.observation_horizon, min=-1.0, max=1.0)

    def _normalize_speed(self, speed: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(speed)
        mask_neg = (speed >= self.speed_min) & (speed < self.speed_mid)
        normalized = torch.where(
            mask_neg,
            (speed - self.speed_mid) / (self.speed_mid - self.speed_min + 1e-6),
            normalized,
        )
        mask_pos = (speed >= self.speed_mid) & (speed <= self.speed_max)
        normalized = torch.where(
            mask_pos,
            (speed - self.speed_mid) / (self.speed_max - self.speed_mid + 1e-6),
            normalized,
        )
        return torch.clamp(normalized, min=-1.0, max=1.0)

    def _normalize_dimension(self, value: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        denom = max(v_max - v_min, 1e-6)
        normalized = (value - v_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)

    def _build_local_state_features(self, agents_state: torch.Tensor) -> torch.Tensor:
        yaw = agents_state[..., 2]
        speed = agents_state[..., 3]
        length = agents_state[..., 4]
        width = agents_state[..., 5]
        active = agents_state[..., 6]
        zeros = torch.zeros_like(agents_state[..., :2])
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        speed_norm = self._normalize_speed(speed).unsqueeze(-1)
        length_norm = self._normalize_dimension(length, self.l_min, self.l_max).unsqueeze(-1)
        width_norm = self._normalize_dimension(width, self.w_min, self.w_max).unsqueeze(-1)
        active = active.unsqueeze(-1)
        return torch.cat([zeros, cos_yaw, sin_yaw, speed_norm, width_norm, length_norm, active], dim=-1)

    def _normalize_neighbor_features(self, neighbors: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(neighbors)
        normalized[..., :2] = self._normalize_relative(neighbors[..., :2])

        dvx = neighbors[..., 2]
        dvy = neighbors[..., 3]

        mask_neg = (dvx >= self.dvx_min) & (dvx < self.dvx_mid)
        normalized[..., 2] = torch.where(
            mask_neg,
            (dvx - self.dvx_mid) / (self.dvx_mid - self.dvx_min + 1e-6),
            torch.zeros_like(dvx),
        )
        mask_pos = (dvx >= self.dvx_mid) & (dvx <= self.dvx_max)
        normalized[..., 2] = torch.where(
            mask_pos,
            (dvx - self.dvx_mid) / (self.dvx_max - self.dvx_mid + 1e-6),
            normalized[..., 2],
        )
        normalized[..., 2] = torch.clamp(normalized[..., 2], min=-1.0, max=1.0)

        mask_neg = (dvy >= self.dvx_min) & (dvy < self.dvx_mid)
        normalized[..., 3] = torch.where(
            mask_neg,
            (dvy - self.dvx_mid) / (self.dvx_mid - self.dvx_min + 1e-6),
            torch.zeros_like(dvy),
        )
        mask_pos = (dvy >= self.dvx_mid) & (dvy <= self.dvx_max)
        normalized[..., 3] = torch.where(
            mask_pos,
            (dvy - self.dvx_mid) / (self.dvx_max - self.dvx_mid + 1e-6),
            normalized[..., 3],
        )
        normalized[..., 3] = torch.clamp(normalized[..., 3], min=-1.0, max=1.0)

        normalized[..., 4] = self._normalize_dimension(neighbors[..., 4], self.w_min, self.w_max)
        normalized[..., 5] = self._normalize_dimension(neighbors[..., 5], self.l_min, self.l_max)
        normalized[..., 6] = neighbors[..., 6]
        return normalized

    def compose_features(
        self,
        agents_state: torch.Tensor,
        neighbors_local: torch.Tensor,
        w_lanes_local: torch.Tensor,
        w_boundaries_local: torch.Tensor,
        path_plan_features: Optional[torch.Tensor],
        reward_params: Optional[torch.Tensor],
        stop_lines: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, M, _ = agents_state.shape
        device = agents_state.device
        dtype = agents_state.dtype

        if neighbors_local.device != device:
            neighbors_local = neighbors_local.to(device)
        if w_lanes_local.device != device:
            w_lanes_local = w_lanes_local.to(device)
        if w_boundaries_local.device != device:
            w_boundaries_local = w_boundaries_local.to(device)

        segments = []

        local_feats = self._build_local_state_features(agents_state)
        segments.append(self._pad_or_trim(local_feats, self.local_state_feature_dim, B, M, device, dtype))

        path_flat = None
        if self.path_plan_feature_dim > 0 and path_plan_features is not None and path_plan_features.numel() > 0:
            path_world = path_plan_features.to(device)
            path_local = convert_path_world_to_ego(agents_state, path_world, self.observation_horizon)
            path_local = self._normalize_relative(path_local)
            path_flat = path_local.view(B, M, -1)
        segments.append(self._pad_or_trim(path_flat, self.path_plan_feature_dim, B, M, device, dtype))

        reward_tensor = self._reshape_reward_params(reward_params, B, M, device, dtype)
        segments.append(self._pad_or_trim(reward_tensor, self.reward_param_dim, B, M, device, dtype))

        boundary_norm = self._normalize_relative(w_boundaries_local)
        boundary_tensor = boundary_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(boundary_tensor, self.w_boundary_feature_dim, B, M, device, dtype))

        w_lanes_norm = self._normalize_relative(w_lanes_local)
        w_lanes_tensor = w_lanes_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(w_lanes_tensor, self.w_lane_feature_dim, B, M, device, dtype))

        if stop_lines is not None:
            stop_lines_flat = stop_lines.to(device=device, dtype=dtype).reshape(B, M, -1)
        else:
            stop_lines_flat = None
        segments.append(self._pad_or_trim(stop_lines_flat, self.stop_line_feature_dim, B, M, device, dtype))

        neighbors_norm = self._normalize_neighbor_features(neighbors_local)
        neighbors_tensor = neighbors_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(neighbors_tensor, self.neighbor_feature_dim, B, M, device, dtype))

        features = torch.cat(segments, dim=-1)
        if features.shape[-1] != self.total_feature_dim:
            raise ValueError(
                f"组合后的特征维度 {features.shape[-1]} 与预期 {self.total_feature_dim} 不一致，请检查配置"
            )
        return features

    def build_from_observation(
        self,
        observation: torch.Tensor,
        simulator: TeraflowSimulator,
        config: Any,
    ) -> torch.Tensor:
        # 使用 ddppo 中的静态函数解包观测
        agents_state, neighbors_local, w_lanes_local, w_boundaries_local = ddppo.decompose_observation(
            observation, config
        )
        path_plan_ids = simulator.agents_path_plans
        if path_plan_ids is None:
            path_plan_features = None
        else:
            path_plan_features = simulator.path_planner.get_w_lane_features_by_id(path_plan_ids)
        reward_params = getattr(simulator.reward_calculator, "sampled_params", None)
        stop_lines = getattr(simulator, "stop_lines", None)
        return self.compose_features(
            agents_state,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            path_plan_features,
            reward_params,
            stop_lines,
        )


def _simple_observation_callback(sim: TeraflowSimulator):
    """
    为 pygame 可视化提供观测数据的回调：
    返回 (neighbors_local, w_lanes_path_only_local, w_boundaries_local)，
    其中 w_lanes 只包含当前智能体路径上的 w_lane 点，坐标在 ego 局部系下，
    与 simulator.py 中 __main__ 的 observation_callback 逻辑保持一致。
    """

    def callback(agents_state: torch.Tensor, b: int, m: int):
        # 先按照 observation_generator 的方式生成邻居和所有可见 w_lanes / w_boundaries 的局部坐标
        neighbor_states_world = sim.observation_generator._get_nearest_neighbors(agents_state)
        w_lanes_world, w_boundaries_world = sim.observation_generator._get_precomputed_w_lanes(agents_state)
        local_state_tmp, neighbors_local_tmp, w_lanes_local_tmp, w_boundaries_local_tmp = (
            sim.observation_generator._world_to_ego_centric(
                agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
            )
        )

        # 再单独构造“仅基于路径规划”的 w_lanes 局部坐标（与 simulator.py __main__ 完全一致）
        w_lanes_path_only = None
        if sim.agents_path_plans is not None:
            try:
                path_ids = sim.agents_path_plans[b, m]  # (L,)
                invalid_marker = sim.path_planner.INVALID_w_lane_id_MARKER
                valid_path_mask = path_ids != invalid_marker
                valid_path_ids = path_ids[valid_path_mask]
                if valid_path_ids.numel() > 0:
                    path_features_world = sim.path_planner.get_w_lane_features_by_id(
                        valid_path_ids.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)  # (L_valid, 3)

                    ego_state = agents_state[b, m]  # (7,)
                    ego_pos = ego_state[:2]
                    ego_yaw = ego_state[2]

                    cos_yaw = torch.cos(ego_yaw)
                    sin_yaw = torch.sin(ego_yaw)
                    rot_matrix = torch.stack(
                        [
                            torch.stack([cos_yaw, -sin_yaw], dim=0),
                            torch.stack([sin_yaw, cos_yaw], dim=0),
                        ],
                        dim=0,
                    )  # (2, 2)

                    world_pos = path_features_world[:, :2]
                    rel_pos = world_pos - ego_pos.unsqueeze(0)
                    local_pos = rel_pos @ rot_matrix.T

                    angles_world = path_features_world[:, 2]
                    angles_local = angles_world - ego_yaw
                    angles_local = torch.atan2(torch.sin(angles_local), torch.cos(angles_local))

                    if sim.agents_path_plan_goal_distances is not None:
                        path_deltas = sim.agents_path_plan_goal_distances[b, m][valid_path_mask]
                    else:
                        invalid_value = float(sim.path_planner.INVALID_MARKER)
                        path_deltas = torch.full(
                            (valid_path_ids.shape[0],),
                            invalid_value,
                            device=sim.device,
                            dtype=path_features_world.dtype,
                        )

                    w_lanes_path_only = torch.cat(
                        [
                            local_pos,
                            angles_local.unsqueeze(-1),
                            path_deltas.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
            except Exception:
                w_lanes_path_only = None

        return (
            neighbors_local_tmp[b, m],
            w_lanes_path_only,
            w_boundaries_local_tmp[b, m],
        )

    return callback


def _simple_info_callback(sim: TeraflowSimulator):
    """
    信息显示回调：在左上角展示当前车辆的状态信息，
    风格和内容参考 simulator/dynamics.py 中 main 里的信息面板。
    """

    def callback(
        agents_state: torch.Tensor,
        goal_positions: Optional[torch.Tensor],
        goal_radii: Optional[torch.Tensor],
        done_mask: Optional[torch.Tensor],
        b: int,
        m: int,
    ):
        state = agents_state[b, m].detach().cpu().numpy()
        x, y, yaw, speed, length, width, active = state

        dyn = getattr(sim, "dynamics_model", None)
        reward_calc = getattr(sim, "reward_calculator", None)
        last_action = getattr(sim, "_last_action", None)

        # 当前纵向/横向加速度
        long_acc = None
        lat_acc = None
        steering_rad = None
        long_jerk = None
        lat_jerk = None

        if dyn is not None:
            try:
                if dyn.current_along is not None:
                    long_acc = float(dyn.current_along.view(sim.num_envs, sim.max_agents)[b, m].detach().cpu().item())
            except Exception:
                long_acc = None
            try:
                if dyn.current_alat is not None:
                    lat_acc = float(dyn.current_alat.view(sim.num_envs, sim.max_agents)[b, m].detach().cpu().item())
            except Exception:
                lat_acc = None
            try:
                if dyn.current_steering_angle is not None:
                    steering_rad = float(
                        dyn.current_steering_angle.view(sim.num_envs, sim.max_agents)[b, m].detach().cpu().item()
                    )
            except Exception:
                steering_rad = None

        # 当前 jerk（通过上一次动作索引反推）
        if dyn is not None and last_action is not None:
            try:
                if b < last_action.shape[0] and m < last_action.shape[1]:
                    a_idx = int(last_action[b, m].detach().cpu().item())
                    jerk = dyn.discrete_action_space.get_action(
                        torch.tensor([a_idx], device=sim.device)
                    )[0].detach().cpu().numpy()
                    long_jerk = float(jerk[0])
                    lat_jerk = float(jerk[1])
            except Exception:
                long_jerk = None
                lat_jerk = None

        lines = [
            ("Agent", f"B={b}, M={m}"),
            ("Active", "Yes" if active > 0.5 else "No"),
            ("Position", f"{x:.2f}, {y:.2f}"),
            ("Yaw", f"{yaw:.3f} rad"),
            ("Speed", f"{speed:.2f} m/s"),
            ("Size", f"L={length:.2f}, W={width:.2f}"),
        ]
        # 当前动力学量
        lines.append(("--- Current ---", ""))
        if long_acc is not None:
            lines.append(("Long Acc", f"{long_acc:.2f} m/s²"))
        if lat_acc is not None:
            lines.append(("Lat Acc", f"{lat_acc:.2f} m/s²"))
        if steering_rad is not None:
            lines.append(("Steering", f"{math.degrees(steering_rad):.2f}°"))
        if long_jerk is not None:
            lines.append(("Long Jerk", f"{long_jerk:.2f} m/s³"))
        if lat_jerk is not None:
            lines.append(("Lat Jerk", f"{lat_jerk:.2f} m/s³"))

        # 奖励分量（如果可用，简单展示上一步的总奖励与关键项）
        if reward_calc is not None:
            try:
                total_r = getattr(reward_calc, "last_total_reward", None)
                if total_r is not None:
                    val = float(total_r[b, m].detach().cpu().item())
                    lines.append(("-- Reward (last step) --", ""))
                    lines.append(("Total R", f"{val:+.6f}"))
            except Exception:
                pass

        if goal_positions is not None:
            try:
                gx, gy = goal_positions[b, m].detach().cpu().numpy().tolist()
                lines.append(("Goal", f"{gx:.2f}, {gy:.2f}"))
            except Exception:
                pass
        if goal_radii is not None:
            try:
                gr = float(goal_radii[b, m].detach().cpu().item())
                lines.append(("Goal Radius", f"{gr:.2f} m"))
            except Exception:
                pass
        if done_mask is not None:
            try:
                done_val = bool(done_mask[b, m].item())
            except Exception:
                done_val = False
            lines.append(("Done", "Yes" if done_val else "No"))
        return lines

    return callback


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用 TeraflowSimulator 初始化世界
    sim = TeraflowSimulator(config=config, device=device)
    initial_observation, d, theta_f = sim.reset()

    # 用轻量网络和 FeatureBuilder 构建策略
    feature_builder = FeatureBuilder(config)
    policy_net = SimplePolicyNet(input_dim=feature_builder.total_feature_dim, config=config).to(device)
    policy_net.eval()

    # 尝试加载最近一次训练的策略权重（如果存在 ckpt）
    ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt")
    if os.path.isdir(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if ckpt_files:
            ckpt_files.sort()
            latest = ckpt_files[-1]
            ckpt_path = os.path.join(ckpt_dir, latest)
            print(f"加载策略 checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            policy_state = checkpoint.get("policy_state")
            if policy_state is not None:
                policy_net.load_state_dict(policy_state, strict=False)
            else:
                print("警告: checkpoint 中未找到 'policy_state'，使用随机初始化策略网络。")
    else:
        print("未找到 ckpt 目录，将使用随机初始化的策略网络进行可视化。")

    current_observation = initial_observation

    def step_callback(b_idx: int, m_idx: int):
        """由可视化工具调用：让策略网络和模拟器向前执行一步。"""
        nonlocal current_observation
        with torch.no_grad():
            features_tensor = feature_builder.build_from_observation(current_observation, sim, config)
            logits = policy_net(features_tensor.to(device))
            action_probs = torch.softmax(logits, dim=-1)
            actions = torch.argmax(action_probs, dim=-1)  # (B, M)
            sim._last_action = actions.clone()
            observation, reward, done = sim.step(actions.unsqueeze(-1))
            current_observation = observation

        sampled_features = None
        sampled_ids_cpu = None
        if sim.sampled_waypoint_ids is not None:
            try:
                sampled_features = sim.path_planner.get_w_lane_features_by_id(sim.sampled_waypoint_ids)
            except Exception:
                sampled_features = None
            try:
                sampled_ids_cpu = sim.sampled_waypoint_ids
            except Exception:
                sampled_ids_cpu = None

        path_features = sim.get_path_plan_features_with_goal_distances()
        return (
            sim.agents_state,
            path_features,
            sim.goal_positions,
            sim.goal_radius_tensor,
            sim.cumulative_done_mask,
            sampled_features,
            sampled_ids_cpu,
        )

    print("启动可视化：SPACE 切换车辆，W 让策略网络执行一步，ESC 退出。")
    path_features = sim.get_path_plan_features_with_goal_distances()
    visualize_path_planning(
        agents_state=sim.agents_state,
        agents_path_plans=path_features,
        quads_vertices=sim.road_network.left_boundaries,
        batch_idx=0,
        invalid_marker_value=float(sim.path_planner.INVALID_MARKER),
        horizon=sim.observation_generator.horizon,
        observation_callback=_simple_observation_callback(sim),
        step_callback=step_callback,
        info_callback=_simple_info_callback(sim),
        agents_start_quad_ids=sim.agents_start_quad_ids,
        agents_goal_quad_ids=sim.agents_goal_quad_ids,
        goal_positions=sim.goal_positions,
        goal_radii=sim.goal_radius_tensor,
        done_mask=sim.cumulative_done_mask,
        sampled_waypoint_features=None,
        sampled_waypoint_ids=None,
    )


if __name__ == "__main__":
    main()



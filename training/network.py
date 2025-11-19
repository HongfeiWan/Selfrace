import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _load_default_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE

def _normalize_relative_distances(tensor: torch.Tensor, horizon: float) -> torch.Tensor:
    if horizon <= 0:
        print("horizon is less than 0, ERROR!")
    scale = torch.as_tensor(horizon, dtype=tensor.dtype, device=tensor.device)
    normalized = torch.clamp(tensor / scale, min=-1.0, max=1.0)
    return normalized

def convert_path_world_to_ego(
    agents_state: torch.Tensor,
    path_world: torch.Tensor,
    horizon: float) -> torch.Tensor:
    """将世界坐标系下的路径点 (x, y) 转换到车辆坐标系 (dx, dy)，并正规化到 [-1, 1] 区间。"""
    if agents_state.dim() != 3 or path_world.dim() != 4:
        raise ValueError("agents_state 必须是 (B, M, 7)，path_world 必须是 (B, M, L, 3)")
    if agents_state.device != path_world.device:
        path_world = path_world.to(agents_state.device)
    B, M, L, _ = path_world.shape
    ego_pos = agents_state[..., :2]
    ego_yaw = agents_state[..., 2]
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    rot_matrix = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw], dim=-1),
            torch.stack([-sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )
    rel_pos_world = path_world[..., :2] - ego_pos.unsqueeze(-2)
    rel_pos_local = torch.bmm(rel_pos_world.view(B * M, L, 2), rot_matrix.view(B * M, 2, 2)).view(B, M, L, 2)
    normalized_local = torch.clamp(rel_pos_local / horizon, min=-1.0, max=1.0)
    return normalized_local

class WBoundaryNet(nn.Module):
    """Encode boundary point sets and return pooled embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get('simulator', {}) if isinstance(config, dict) else {}
        obs_cfg = sim_cfg.get('observation', {}) if isinstance(sim_cfg.get('observation', {}), dict) else {}
        net_cfg = sim_cfg.get('network', {}) if isinstance(sim_cfg.get('network', {}), dict) else {}

        self.point_dim = int(obs_cfg.get('boundary_feature_dim'))
        self.embed_dim = int(net_cfg.get('WBoundaryNet_embed_dim'))
        self.encoded_dim = int(net_cfg.get('WBoundaryNet_encoded_dim'))
        self.horizon = float(obs_cfg.get('horizon'))
        
        self.point_mlp = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )

        self.pool_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Linear(self.encoded_dim, self.encoded_dim),
            nn.ReLU(),
        )

    def forward(self, w_boundaries: torch.Tensor) -> torch.Tensor:
        """Args: w_boundaries shape (B, M, K, point_dim)."""
        B, M, K, D = w_boundaries.shape
        if D != self.point_dim:
            raise ValueError(f"Expected point_dim={self.point_dim}, but got {D}")
        processed = _normalize_relative_distances(w_boundaries, self.horizon)
        points = processed.reshape(B * M * K, D)
        point_features = self.point_mlp(points).view(B, M, K, self.embed_dim)
        max_pool = point_features.max(dim=2).values
        return self.pool_proj(max_pool)

class GoalsNet(nn.Module):
    """Encode ego-centric goal/path plans (dx, dy) into pooled embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        obs_cfg = sim_cfg.get("observation", {}) if isinstance(sim_cfg.get("observation", {}), dict) else {}
        self.horizon = float(obs_cfg.get("horizon"))
        self.point_dim = int(obs_cfg.get("navigation_feature_dim"))
        self.embed_dim = int(net_cfg.get("GoalsNet_embed_dim"))
        self.encoded_dim = int(net_cfg.get("GoalsNet_encoded_dim"))
        self.point_mlp = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Linear(self.encoded_dim, self.encoded_dim),
            nn.ReLU(),
        )
    def forward(self, agents_state: torch.Tensor, agents_path_plans: torch.Tensor) -> torch.Tensor:
        if agents_state.dim() != 3 or agents_state.shape[-1] != 7:
            raise ValueError("agents_state 必须是 (B, M, 7)")
        if agents_path_plans.dim() != 4 or agents_path_plans.shape[-1] != 3:
            raise ValueError("agents_path_plans 必须是 (B, M, L, 3)")
        processed = convert_path_world_to_ego(
            agents_state,
            agents_path_plans,
            self.horizon,
        )  # (B, M, L, 2)
        B, M, L, _ = processed.shape
        point_features = self.point_mlp(processed.view(B * M * L, self.point_dim)).view(B, M, L, self.embed_dim)
        max_pool = point_features.max(dim=2).values  # (B, M, embed_dim)
        return self.pool_proj(max_pool)

class WlaneNet(nn.Module):
    """Encode observed w_lane points (dx, dy, Δs) into pooled embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        obs_cfg = sim_cfg.get("observation", {}) if isinstance(sim_cfg.get("observation", {}), dict) else {}

        self.horizon = float(obs_cfg.get("horizon"))
        base_dim = int(obs_cfg.get("w_lane_feature_dim", 2))
        # 输入包含 dx, dy 以及额外的 Δs
        self.point_dim = base_dim + 1
        self.embed_dim = int(net_cfg.get("GoalsNet_embed_dim", 64))
        self.encoded_dim = int(net_cfg.get("GoalsNet_encoded_dim", 64))
        self.path_invalid_marker = float(obs_cfg.get("INVALID_MARKER", -1e10))

        self.point_mlp = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Linear(self.encoded_dim, self.encoded_dim),
            nn.ReLU(),
        )

    def forward(self, w_lanes_local_with_goal_distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_lanes_local_with_goal_distances: (B, M, K, 3) with [dx, dy, Δs]
        Returns:
            Encoded features of shape (B, M, encoded_dim)
        """
        if w_lanes_local_with_goal_distances.dim() != 4 or w_lanes_local_with_goal_distances.shape[-1] != self.point_dim:
            raise ValueError(
                f"Expected input shape (B, M, K, {self.point_dim}), got {tuple(w_lanes_local_with_goal_distances.shape)}"
            )
        processed = _normalize_relative_distances(w_lanes_local_with_goal_distances, self.horizon)
        B, M, K, _ = processed.shape
        point_features = self.point_mlp(processed.view(B * M * K, self.point_dim)).view(B, M, K, self.embed_dim)
        max_pool = point_features.max(dim=2).values
        return self.pool_proj(max_pool)

class OtherAgentsNet(nn.Module):
    """Encode neighbor vehicles (dx, dy, dvx, dvy, w, l, active) into pooled embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        obs_cfg = sim_cfg.get("observation", {}) if isinstance(sim_cfg.get("observation", {}), dict) else {}
        
        self.horizon = float(obs_cfg.get("horizon"))
        self.point_dim = int(obs_cfg.get("neighbor_feature_dim"))  # dx, dy, dvx, dvy, w, l, active
        self.embed_dim = int(net_cfg.get("OtherAgentsNet_embed_dim"))
        self.encoded_dim = int(net_cfg.get("OtherAgentsNet_encoded_dim"))
        
        # 归一化参数
        self.dvx_dvy_min = -2.0
        self.dvx_dvy_mid = 0.0
        self.dvx_dvy_max = 20.0
        self.w_min = 0.8
        self.w_max = 3.0
        self.l_min = 0.8
        self.l_max = 7.0
        
        self.point_mlp = nn.Sequential(
            nn.Linear(self.point_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )
        
        self.pool_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Linear(self.encoded_dim, self.encoded_dim),
            nn.ReLU(),
        )
    
    def _normalize_neighbor_features(self, neighbors: torch.Tensor) -> torch.Tensor:
        """
        归一化邻居车辆特征。
        Args:
            neighbors: (B, M, K, 7) with [dx, dy, dvx, dvy, w, l, active]
        Returns:
            normalized: (B, M, K, 7) 归一化后的特征
        """
        B, M, K, D = neighbors.shape
        if D != self.point_dim:
            raise ValueError(f"Expected point_dim={self.point_dim}, but got {D}")
        
        normalized = torch.zeros_like(neighbors)
        
        # 1. dx, dy 使用 _normalize_relative_distances 归一化
        normalized[..., :2] = _normalize_relative_distances(neighbors[..., :2], self.horizon)
        
        # 2. dvx, dvy 分段归一化
        dvx = neighbors[..., 2]
        dvy = neighbors[..., 3]
        
        # -2到0映射到-1到0
        mask_neg = (dvx >= self.dvx_dvy_min) & (dvx < self.dvx_dvy_mid)
        normalized[..., 2] = torch.where(
            mask_neg,
            (dvx - self.dvx_dvy_mid) / (self.dvx_dvy_mid - self.dvx_dvy_min),  # 映射到 [-1, 0]
            torch.zeros_like(dvx)
        )
        # 0到20归一化到0到1
        mask_pos = (dvx >= self.dvx_dvy_mid) & (dvx <= self.dvx_dvy_max)
        normalized[..., 2] = torch.where(
            mask_pos,
            (dvx - self.dvx_dvy_mid) / (self.dvx_dvy_max - self.dvx_dvy_mid),  # 映射到 [0, 1]
            normalized[..., 2]
        )
        # 范围外的值映射到-1到1（<-1设置-1，>1设置1）
        normalized[..., 2] = torch.clamp(normalized[..., 2], min=-1.0, max=1.0)
        
        # 对 dvy 做同样的处理
        mask_neg = (dvy >= self.dvx_dvy_min) & (dvy < self.dvx_dvy_mid)
        normalized[..., 3] = torch.where(
            mask_neg,
            (dvy - self.dvx_dvy_mid) / (self.dvx_dvy_mid - self.dvx_dvy_min),
            torch.zeros_like(dvy)
        )
        mask_pos = (dvy >= self.dvx_dvy_mid) & (dvy <= self.dvx_dvy_max)
        normalized[..., 3] = torch.where(
            mask_pos,
            (dvy - self.dvx_dvy_mid) / (self.dvx_dvy_max - self.dvx_dvy_mid),
            normalized[..., 3]
        )
        normalized[..., 3] = torch.clamp(normalized[..., 3], min=-1.0, max=1.0)
        
        # 3. w 使用范围（0.8, 3）到（0, 1）映射
        w = neighbors[..., 4]
        normalized[..., 4] = torch.clamp(
            (w - self.w_min) / (self.w_max - self.w_min),
            min=0.0,
            max=1.0
        )
        
        # 4. l 使用范围（0.8, 7）到（0, 1）映射
        l = neighbors[..., 5]
        normalized[..., 5] = torch.clamp(
            (l - self.l_min) / (self.l_max - self.l_min),
            min=0.0,
            max=1.0
        )
        
        # 5. active 输入是0或1，直接使用即可
        normalized[..., 6] = neighbors[..., 6]
        
        return normalized
    
    def forward(self, neighbors_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbors_local: (B, M, K, 7) with [dx, dy, dvx, dvy, w, l, active]
        Returns:
            Encoded features of shape (B, M, encoded_dim)
        """
        if neighbors_local.dim() != 4 or neighbors_local.shape[-1] != self.point_dim:
            raise ValueError(
                f"Expected input shape (B, M, K, {self.point_dim}), got {tuple(neighbors_local.shape)}"
            )
        
        # 归一化所有特征
        processed = self._normalize_neighbor_features(neighbors_local)
        
        B, M, K, _ = processed.shape
        # 将所有邻居的特征通过 MLP 处理
        point_features = self.point_mlp(processed.view(B * M * K, self.point_dim)).view(B, M, K, self.embed_dim)
        # Max pooling 在 K 维度上
        max_pool = point_features.max(dim=2).values  # (B, M, embed_dim)
        
        return self.pool_proj(max_pool)

class ConditionNet(nn.Module):
    """Fuse simulator-level conditioning signals into agent embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        dynamics_cfg = sim_cfg.get("dynamics", {}) if isinstance(sim_cfg.get("dynamics", {}), dict) else {}
        reward_cfg = sim_cfg.get("reward", {}) if isinstance(sim_cfg.get("reward", {}), dict) else {}

        self.embed_dim = int(net_cfg.get("ConditionNet_embed_dim", 64))
        self.encoded_dim = int(net_cfg.get("ConditionNet_encoded_dim", 64))

        # 归一化所需的范围（从 dynamics 配置读取）
        self.curvature_scale = float(dynamics_cfg.get("ConditionNet_curvature_scale", 0.077))
        
        # wheelbase 范围从 dynamics 配置计算
        vehicle_length_min = float(dynamics_cfg.get("vehicle_length_min", 0.8))
        vehicle_length_max = float(dynamics_cfg.get("vehicle_length_max", 7.0))
        wheelbase_ratio = float(dynamics_cfg.get("wheelbase_ratio", 0.6))
        self.wheelbase_min = vehicle_length_min * wheelbase_ratio
        self.wheelbase_max = vehicle_length_max * wheelbase_ratio
        
        # 驾驶风格参数归一化范围（从 dynamics 配置读取）
        self.c_throttle_min = float(dynamics_cfg.get("ConditionNet_c_throttle_min", 0.900))
        self.c_throttle_max = float(dynamics_cfg.get("ConditionNet_c_throttle_max", 1.125))
        self.c_steer_min = float(dynamics_cfg.get("ConditionNet_c_steer_min", 0.900))
        self.c_steer_max = float(dynamics_cfg.get("ConditionNet_c_steer_max", 1.125))
        self.c_acc_min = float(dynamics_cfg.get("ConditionNet_c_acc_min", 0.833))
        self.c_acc_max = float(dynamics_cfg.get("ConditionNet_c_acc_max", 1.25))
        self.c_vel_min = float(dynamics_cfg.get("ConditionNet_c_vel_min", 0.833))
        self.c_vel_max = float(dynamics_cfg.get("ConditionNet_c_vel_max", 1.25))

        # Reward 参数归一化范围（从 reward 配置读取）
        # 顺序对应 reward.py 中的 _param_name_to_idx: delta_goal, collision_alpha, boundary_alpha, 
        # comfort_alpha, l_align_alpha, vel_align_alpha, l_center_alpha, center_bias_alpha, 
        # reverse_alpha, stop_line_alpha
        reward_param_mins = torch.tensor([
            float(reward_cfg.get("delta_goal_min", 2.0)),
            float(reward_cfg.get("collision_alpha_min", 0.0)),
            float(reward_cfg.get("boundary_alpha_min", 0.0)),
            float(reward_cfg.get("comfort_alpha_min", 0.0)),
            float(reward_cfg.get("l_align_alpha_min", 2.5e-4)),
            float(reward_cfg.get("vel_align_alpha_min", 0.0)),
            float(reward_cfg.get("l_center_alpha_min", 2.5e-4)),
            float(reward_cfg.get("center_bias_alpha_min", -0.5)),
            float(reward_cfg.get("reverse_alpha_min", 2.5e-4)),
            float(reward_cfg.get("stop_line_alpha_min", 0.0)),
        ])
        reward_param_maxs = torch.tensor([
            float(reward_cfg.get("delta_goal_max", 12.0)),
            float(reward_cfg.get("collision_alpha_max", 3.0)),
            float(reward_cfg.get("boundary_alpha_max", 3.0)),
            float(reward_cfg.get("comfort_alpha_max", 0.1)),
            float(reward_cfg.get("l_align_alpha_max", 2.5e-2)),
            float(reward_cfg.get("vel_align_alpha_max", 1.0)),
            float(reward_cfg.get("l_center_alpha_max", 7.5e-3)),
            float(reward_cfg.get("center_bias_alpha_max", 0.5)),
            float(reward_cfg.get("reverse_alpha_max", 7.5e-3)),
            float(reward_cfg.get("stop_line_alpha_max", 1.0)),
        ])
        # 注册为 buffer，确保能正确移动到设备上
        self.register_buffer("reward_param_mins", reward_param_mins)
        self.register_buffer("reward_param_maxs", reward_param_maxs)

        # Condition features: [curvature, Cthrottle, Csteer, Cacc, Cvel] + reward_params(10) + wheelbase
        self.reward_param_dim = 10 #10个reward参数
        self.input_dim = 1 + 4 + self.reward_param_dim + 1  # curvature + 4 coeffs + reward params + wheelbase

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.LayerNorm(self.encoded_dim),
            nn.ReLU(),
        )

    def _reshape_feature(self, tensor: torch.Tensor, B: int, M: int) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.view(B, M)
        if tensor.dim() == 2 and tensor.shape[0] == B and tensor.shape[1] == M:
            return tensor
        if tensor.dim() == 2 and tensor.shape[0] == B * M:
            return tensor.view(B, M)
        raise ValueError(f"Unexpected tensor shape {tuple(tensor.shape)} for ConditionNet feature.")

    def _normalize_curvature(self, curvature: torch.Tensor) -> torch.Tensor:
        return torch.clamp(curvature / self.curvature_scale, min=-1.0, max=1.0)

    def _normalize_wheelbase(self, wheelbase: torch.Tensor) -> torch.Tensor:
        denom = max(self.wheelbase_max - self.wheelbase_min, 1e-6)
        normalized = (wheelbase - self.wheelbase_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_c_throttle(self, c_throttle: torch.Tensor) -> torch.Tensor:
        denom = max(self.c_throttle_max - self.c_throttle_min, 1e-6)
        normalized = (c_throttle - self.c_throttle_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_c_steer(self, c_steer: torch.Tensor) -> torch.Tensor:
        denom = max(self.c_steer_max - self.c_steer_min, 1e-6)
        normalized = (c_steer - self.c_steer_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_c_acc(self, c_acc: torch.Tensor) -> torch.Tensor:
        denom = max(self.c_acc_max - self.c_acc_min, 1e-6)
        normalized = (c_acc - self.c_acc_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_c_vel(self, c_vel: torch.Tensor) -> torch.Tensor:
        denom = max(self.c_vel_max - self.c_vel_min, 1e-6)
        normalized = (c_vel - self.c_vel_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_reward_params(self, reward_params: torch.Tensor) -> torch.Tensor:
        """
        对 reward_params 的每一维进行归一化。
        Args:
            reward_params: (B, M, 10)
        Returns:
            normalized: (B, M, 10) 归一化后的参数，大部分映射到 [0, 1]，center_bias_alpha 映射到 [-1, 1]
        """
        B, M, R = reward_params.shape
        if R != self.reward_param_dim:
            raise ValueError(f"Expected reward_params last dim {self.reward_param_dim}, got {R}")
        
        # 将 mins 和 maxs 扩展到 (B, M, R) 形状
        mins = self.reward_param_mins.view(1, 1, -1).expand(B, M, -1).to(reward_params.device)
        maxs = self.reward_param_maxs.view(1, 1, -1).expand(B, M, -1).to(reward_params.device)
        
        # 计算分母，避免除零
        denoms = torch.clamp(maxs - mins, min=1e-6)
        
        # 归一化到 [0, 1] 或 [-1, 1]
        normalized = (reward_params - mins) / denoms
        
        # center_bias_alpha (索引 7) 需要映射到 [-1, 1]，其他映射到 [0, 1]
        # 对于 center_bias_alpha: 从 [-0.5, 0.5] 映射到 [-1, 1]
        # 公式: normalized = (value - min) / (max - min) * 2 - 1
        center_bias_idx = 7
        center_bias_normalized = normalized[..., center_bias_idx:center_bias_idx+1] * 2.0 - 1.0
        center_bias_normalized = torch.clamp(center_bias_normalized, min=-1.0, max=1.0)
        
        # 其他参数映射到 [0, 1]
        other_normalized = torch.clamp(normalized[..., :center_bias_idx], min=0.0, max=1.0)
        if center_bias_idx + 1 < R:
            other_normalized_after = torch.clamp(normalized[..., center_bias_idx+1:], min=0.0, max=1.0)
            normalized = torch.cat([other_normalized, center_bias_normalized, other_normalized_after], dim=-1)
        else:
            normalized = torch.cat([other_normalized, center_bias_normalized], dim=-1)
        
        return normalized

    def forward(
        self,
        curvature: torch.Tensor,
        c_throttle: torch.Tensor,
        c_steer: torch.Tensor,
        c_acc: torch.Tensor,
        c_vel: torch.Tensor,
        reward_params: torch.Tensor,
        wheelbase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            curvature: (B, M)
            c_throttle/c_steer/c_acc/c_vel: (B*M,) 或 (B, M)
            reward_params: (B, M, R)
            wheelbase: (B*M,) 或 (B, M)
        Returns:
            Encoded conditioning features of shape (B, M, encoded_dim)
        """
        if curvature.dim() != 2:
            raise ValueError("curvature must be of shape (B, M)")
        B, M = curvature.shape

        curv_norm = self._normalize_curvature(curvature)
        cth = self._normalize_c_throttle(self._reshape_feature(c_throttle, B, M))
        cst = self._normalize_c_steer(self._reshape_feature(c_steer, B, M))
        cac = self._normalize_c_acc(self._reshape_feature(c_acc, B, M))
        cvl = self._normalize_c_vel(self._reshape_feature(c_vel, B, M))
        wheelbase_reshaped = self._normalize_wheelbase(self._reshape_feature(wheelbase, B, M))
        # 归一化 reward_params
        reward_params_norm = self._normalize_reward_params(reward_params)
        features = torch.cat(
            [
                curv_norm.unsqueeze(-1),
                cth.unsqueeze(-1),
                cst.unsqueeze(-1),
                cac.unsqueeze(-1),
                cvl.unsqueeze(-1),
                reward_params_norm,
                wheelbase_reshaped.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, M, input_dim)
        output = self.mlp(features.view(B * M, self.input_dim))
        return output.view(B, M, self.encoded_dim)

class VehicleStateNet(nn.Module):
    """Encode vehicle local state (dx=0, dy=0, yaw, speed, w, l, active) into embeddings."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        dynamics_cfg = sim_cfg.get("dynamics", {}) if isinstance(sim_cfg.get("dynamics", {}), dict) else {}
        
        self.embed_dim = int(net_cfg.get("VehicleStateNet_embed_dim"))
        self.encoded_dim = int(net_cfg.get("VehicleStateNet_encoded_dim"))
        
        # 归一化参数（从 dynamics 配置读取）
        self.speed_min = float(dynamics_cfg.get("min_velocity"))
        self.speed_mid = 0.0
        self.speed_max = float(dynamics_cfg.get("max_velocity"))
        self.w_min = float(dynamics_cfg.get("vehicle_width_min"))
        self.w_max = float(dynamics_cfg.get("vehicle_width_max"))
        self.l_min = float(dynamics_cfg.get("vehicle_length_min"))
        self.l_max = float(dynamics_cfg.get("vehicle_length_max"))
        # 输入特征维度：8 (0, 0, cos(yaw), sin(yaw), speed_norm, w_norm, l_norm, active)
        self.input_dim = 8
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.encoded_dim),
            nn.LayerNorm(self.encoded_dim),
            nn.ReLU(),
        )
    
    def _normalize_speed(self, speed: torch.Tensor) -> torch.Tensor:
        """
        对 speed 进行分段归一化：-2到0映射到-1到0，0到20映射到0到1。
        Args:
            speed: (B, M) 速度值
        Returns:
            normalized: (B, M) 归一化后的速度，范围 [-1, 1]
        """
        normalized = torch.zeros_like(speed)
        
        # -2到0映射到-1到0
        mask_neg = (speed >= self.speed_min) & (speed < self.speed_mid)
        normalized = torch.where(
            mask_neg,
            (speed - self.speed_mid) / (self.speed_mid - self.speed_min),  # 映射到 [-1, 0]
            normalized
        )
        
        # 0到20映射到0到1
        mask_pos = (speed >= self.speed_mid) & (speed <= self.speed_max)
        normalized = torch.where(
            mask_pos,
            (speed - self.speed_mid) / (self.speed_max - self.speed_mid),  # 映射到 [0, 1]
            normalized
        )
        
        # 范围外的值映射到-1到1
        normalized = torch.clamp(normalized, min=-1.0, max=1.0)
        return normalized
    
    def _normalize_w(self, w: torch.Tensor) -> torch.Tensor:
        """将宽度从 (0.8, 3) 映射到 (0, 1)"""
        denom = max(self.w_max - self.w_min, 1e-6)
        normalized = (w - self.w_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def _normalize_l(self, l: torch.Tensor) -> torch.Tensor:
        """将长度从 (0.8, 7) 映射到 (0, 1)"""
        denom = max(self.l_max - self.l_min, 1e-6)
        normalized = (l - self.l_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)
    
    def forward(self, local_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_state: (B, M, 7) with [dx=0, dy=0, yaw, speed, w, l, active]
        Returns:
            Encoded features of shape (B, M, encoded_dim)
        """
        if local_state.dim() != 3 or local_state.shape[-1] != 7:
            raise ValueError(f"Expected input shape (B, M, 7), got {tuple(local_state.shape)}")
        
        B, M, _ = local_state.shape
        
        # 提取各个特征
        # local_state: [dx=0, dy=0, yaw, speed, w, l, active]
        yaw = local_state[..., 2]
        speed = local_state[..., 3]
        w = local_state[..., 4]
        l = local_state[..., 5]
        active = local_state[..., 6]
        
        # 构建8维特征：[0, 0, cos(yaw), sin(yaw), speed_norm, w_norm, l_norm, active]
        zeros = torch.zeros(B, M, 2, device=local_state.device, dtype=local_state.dtype)
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        speed_norm = self._normalize_speed(speed).unsqueeze(-1)
        w_norm = self._normalize_w(w).unsqueeze(-1)
        l_norm = self._normalize_l(l).unsqueeze(-1)
        active_reshaped = active.unsqueeze(-1)
        
        features = torch.cat(
            [zeros, cos_yaw, sin_yaw, speed_norm, w_norm, l_norm, active_reshaped],
            dim=-1
        )  # (B, M, 8)
        
        # 通过 MLP 处理
        output = self.mlp(features.view(B * M, self.input_dim))
        return output.view(B, M, self.encoded_dim)

class MLP_policy(nn.Module):
    """汇总所有网络的输出，通过三层MLP输出动作空间的概率分布。"""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        dynamics_cfg = sim_cfg.get("dynamics", {}) if isinstance(sim_cfg.get("dynamics", {}), dict) else {}
        
        # 获取各个网络的 encoded_dim
        w_boundary_dim = int(net_cfg.get("WBoundaryNet_encoded_dim", 64))
        goals_dim = int(net_cfg.get("GoalsNet_encoded_dim", 64))
        w_lane_dim = int(net_cfg.get("GoalsNet_encoded_dim", 64))  # WlaneNet 使用 GoalsNet_encoded_dim
        other_agents_dim = int(net_cfg.get("OtherAgentsNet_encoded_dim", 64))
        condition_dim = int(net_cfg.get("ConditionNet_encoded_dim", 64))
        vehicle_state_dim = int(net_cfg.get("VehicleStateNet_encoded_dim", 64))
        
        # 计算总输入维度
        self.input_dim = w_boundary_dim + goals_dim + w_lane_dim + other_agents_dim + condition_dim + vehicle_state_dim
        # 输出维度（动作空间）
        self.output_dim = int(dynamics_cfg.get("dynamics_jerk_dim", 12))
        # 三层 MLP: 1024x1024x1024
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
        )
    
    def forward(
        self,
        w_boundary_features: torch.Tensor,
        goals_features: torch.Tensor,
        w_lane_features: torch.Tensor,
        other_agents_features: torch.Tensor,
        condition_features: torch.Tensor,
        vehicle_state_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        汇总所有网络的输出并输出动作空间的概率分布。
        
        Args:
            w_boundary_features: (B, M, WBoundaryNet_encoded_dim)
            goals_features: (B, M, GoalsNet_encoded_dim)
            w_lane_features: (B, M, GoalsNet_encoded_dim)
            other_agents_features: (B, M, OtherAgentsNet_encoded_dim)
            condition_features: (B, M, ConditionNet_encoded_dim)
            vehicle_state_features: (B, M, VehicleStateNet_encoded_dim)
        
        Returns:
            action_probs: (B, M, dynamics_jerk_dim) 动作空间的概率分布
        """
        # 拼接所有特征
        combined = torch.cat(
            [
                w_boundary_features,
                goals_features,
                w_lane_features,
                other_agents_features,
                condition_features,
                vehicle_state_features,
            ],
            dim=-1,
        )  # (B, M, input_dim)
        
        # 通过 MLP
        B, M, _ = combined.shape
        logits = self.mlp(combined.view(B * M, self.input_dim))  # (B*M, output_dim)
        # 输出概率分布（使用 softmax）
        action_probs = torch.softmax(logits, dim=-1)  # (B*M, output_dim)
        return action_probs.view(B, M, self.output_dim)

class MLP_value(nn.Module):
    """汇总所有网络的输出，通过三层MLP输出价值估计（单头输出）。"""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = _load_default_config()
        sim_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        net_cfg = sim_cfg.get("network", {}) if isinstance(sim_cfg.get("network", {}), dict) else {}
        
        # 获取各个网络的 encoded_dim
        w_boundary_dim = int(net_cfg.get("WBoundaryNet_encoded_dim", 64))
        goals_dim = int(net_cfg.get("GoalsNet_encoded_dim", 64))
        w_lane_dim = int(net_cfg.get("GoalsNet_encoded_dim", 64))  # WlaneNet 使用 GoalsNet_encoded_dim
        other_agents_dim = int(net_cfg.get("OtherAgentsNet_encoded_dim", 64))
        condition_dim = int(net_cfg.get("ConditionNet_encoded_dim", 64))
        vehicle_state_dim = int(net_cfg.get("VehicleStateNet_encoded_dim", 64))
        
        # 计算总输入维度
        self.input_dim = w_boundary_dim + goals_dim + w_lane_dim + other_agents_dim + condition_dim + vehicle_state_dim
        
        # 三层 MLP: 1024x1024x1024，最后输出单个价值
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),  # 输出单个价值
        )
    
    def forward(
        self,
        w_boundary_features: torch.Tensor,
        goals_features: torch.Tensor,
        w_lane_features: torch.Tensor,
        other_agents_features: torch.Tensor,
        condition_features: torch.Tensor,
        vehicle_state_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        汇总所有网络的输出并输出价值估计。
        
        Args:
            w_boundary_features: (B, M, WBoundaryNet_encoded_dim)
            goals_features: (B, M, GoalsNet_encoded_dim)
            w_lane_features: (B, M, GoalsNet_encoded_dim)
            other_agents_features: (B, M, OtherAgentsNet_encoded_dim)
            condition_features: (B, M, ConditionNet_encoded_dim)
            vehicle_state_features: (B, M, VehicleStateNet_encoded_dim)
        
        Returns:
            values: (B, M, 1) 价值估计
        """
        # 拼接所有特征
        combined = torch.cat(
            [
                w_boundary_features,
                goals_features,
                w_lane_features,
                other_agents_features,
                condition_features,
                vehicle_state_features,
            ],
            dim=-1,
        )  # (B, M, input_dim)
        
        # 通过 MLP
        B, M, _ = combined.shape
        values = self.mlp(combined.view(B * M, self.input_dim))  # (B*M, 1)
        return values.view(B, M, 1)


if __name__ == "__main__":
    # 测试一下，WBoundaryNet 是否能正常工作
    # sample_points = torch.tensor([
    #     [0.1164, 1.85],
    #     [1.1283, 1.8475],
    #     [-0.8955, 1.8475],
    #     [2.1401, 1.84],
    #     [-1.9073, 1.84],
    #     [3.1519, 1.8275],
    #     [-2.9192, 1.8275],
    #     [4.1636, 1.81],
    #     [-3.9309, 1.81],
    #     [0.5437, -4.8504],
    #     [-0.4475, -4.8508],
    #     [1.535, -4.8551],
    #     [-1.4388, -4.8561],
    #     [5.1753, 1.7875],
    #     [-4.9425, 1.7875],
    #     [2.5262, -4.8646],
    #     [-2.43, -4.8664],
    #     [3.5174, -4.8791],
    #     [-3.4211, -4.8816],
    #     [6.1867, 1.7601],
    #     [-5.954, 1.76],
    #     [4.5084, -4.8986],
    #     [-4.4121, -4.9018],
    #     [7.1981, 1.7276],
    #     [5.4994, -4.9231],
    #     [-6.9653, 1.7275],
    #     [-5.403, -4.9269],
    #     [6.4902, -4.9525],
    #     [-6.3938, -4.957],
    #     [8.2094, 1.6901],
    #     [-7.9765, 1.69],
    #     [7.4808, -4.9868],
    #     [-7.3845, -4.992],
    #     [9.2203, 1.6476],
    #     [-8.9875, 1.6475],
    #     [8.4713, -5.0262],
    #     [-8.3749, -5.0321],
    #     [10.2311, 1.6002],
    #     [-9.9982, 1.6001],
    #     [9.4615, -5.0704],
    #     [-9.3651, -5.077],
    #     [11.2416, 1.5477],
    #     [-11.0088, 1.5476],
    #     [10.4516, -5.1196],
    #     [-10.3551, -5.1269],
    #     [12.2518, 1.4903],
    #     [-12.0191, 1.4901],
    #     [11.4413, -5.1738],
    #     [-11.3449, -5.1818],
    #     [13.2617, 1.4278],
    #     [-13.029, 1.4277],
    #     [12.4309, -5.2329],
    #     [-12.3343, -5.2416],
    #     [14.2714, 1.3604],
    #     [-14.0386, 1.3603],
    #     [13.4201, -5.297],
    #     [-13.3235, -5.3064],
    #     [15.2806, 1.288],
    #     [-15.0479, 1.2878],
    #     [14.4089, -5.366],
    #     [-14.3123, -5.3761],
    #     [16.2896, 1.2107],
    #     [15.3974, -5.44],
    #     [-16.0568, 1.2104],
    #     [-15.3007, -5.4508],
    #     [16.3855, -5.5189],
    #     [17.2982, 1.1283],
    #     [-17.0653, 1.1281],
    #     [-16.2888, -5.5304],
    #     [17.3732, -5.6028],
    #     [18.3062, 1.0409],
    #     [-18.0734, 1.0407],
    #     [-17.2764, -5.6149],
    #     [18.3605, -5.6915],
    #     [19.3139, 0.9486],
    #     [-19.0811, 0.9483],
    #     [-18.2636, -5.7044],
    #     [19.3472, -5.7853],
    #     [20.3211, 0.8513],
    #     [-19.2503, -5.7988],
    # ], dtype=torch.float32)
    # sample_tensor = sample_points.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 2),相当于B=1,M=1
    # model = WBoundaryNet()
    # output = model(sample_tensor)
    # print("WBoundaryNet output shape:", tuple(output.shape))
    # print("WBoundaryNet output:")
    # print(output.detach().cpu())

    # 测试一下，convert_path_world_to_ego 是否能正常工作
    # agents_state_example = torch.tensor(
    #     [[[10.0, -5.0, 0.5 * torch.pi, 5.0, 4.5, 2.0, 1.0]]], dtype=torch.float32
    # )
    # agents_path_plan_world = torch.tensor(
    #     [[[[12.0, -3.0, 0.0], [8.0, -3.0, 0.0]]]], dtype=torch.float32
    # )
    # local_path = convert_path_world_to_ego(
    #     agents_state_example,
    #     agents_path_plan_world,
    #     horizon=200.0,
    # )
    # print("convert_path_world_to_ego 输出示例 (B=0, M=0):")
    # print(local_path[0, 0])

    # 测试 GoalsNet with sample data
    # goals_net = GoalsNet()
    # agents_state_example = torch.tensor(
    #     [[[0.0, 0.0, 0.0, 5.0, 4.5, 2.0, 1.0], [1.0, 1.0, 0.5, 4.0, 4.5, 2.0, 1.0], [2.0, -1.0, 1.0, 3.0, 4.5, 2.0, 1.0]]],
    #     dtype=torch.float32,
    # )
    # agents_path_plan_example = torch.full((1, 3, 128, 3), -1.0e10, dtype=torch.float32)
    # agents_path_plan_example[0, 2, :2] = torch.tensor(
    #     [[-2.2690e02, -2.0805e02, 1.9404e00], [-2.2913e02, -1.9613e02, 1.5708e00]],
    #     dtype=torch.float32,
    # )
    # goals_output = goals_net(agents_state_example, agents_path_plan_example)
    # print("GoalsNet output:")
    # print(goals_output).
    
    pass
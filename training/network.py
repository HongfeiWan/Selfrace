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
    goals_net = GoalsNet()
    agents_state_example = torch.tensor(
        [[[0.0, 0.0, 0.0, 5.0, 4.5, 2.0, 1.0], [1.0, 1.0, 0.5, 4.0, 4.5, 2.0, 1.0], [2.0, -1.0, 1.0, 3.0, 4.5, 2.0, 1.0]]],
        dtype=torch.float32,
    )
    agents_path_plan_example = torch.full((1, 3, 128, 3), -1.0e10, dtype=torch.float32)
    agents_path_plan_example[0, 2, :2] = torch.tensor(
        [[-2.2690e02, -2.0805e02, 1.9404e00], [-2.2913e02, -1.9613e02, 1.5708e00]],
        dtype=torch.float32,
    )
    goals_output = goals_net(agents_state_example, agents_path_plan_example)
    print("GoalsNet output:")
    print(goals_output)

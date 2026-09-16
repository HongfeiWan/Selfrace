import torch
from typing import Dict
import sys
import os

# 添加simulator目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from road import RoadNetwork
from spatial_hash import SpatialHash
    
class ObservationGenerator:
    """
    已经通过测试
    负责为批次中的所有 Agent 生成局部观测。
    通过完全向量化的操作，该模块可以一次性为所有环境中的所有智能体高效地计算观测，
    避免了在 Python 中进行循环，从而最大限度地利用 GPU 并行能力。
    """
    def __init__(self, road_network: RoadNetwork, config: Dict, device: torch.device, spatial_hash: SpatialHash = None):
        """
        初始化观测生成器。
        """
        self.road_network = road_network
        self.config = config
        self.device = device
        self.num_neighbors = config.get('num_neighbors', 1) # 邻居数量
        self.num_w_lanes = config.get('num_w_lanes', 80) # 原文 W_lane 使用 80 个 coarse map features
        self.num_w_boundaries = config.get('num_w_boundaries', 80) # 原文 W_boundary 使用 80 个 coarse map features
        self.horizon = config.get('horizon', 200.0) # 原文 coarse map horizon 200m
        self.speed_limit = float(config.get('speed_limit', 20.0))
        # 定义观测空间维度
        self.local_state_dim = config.get('local_state_dim', 13)           # 原文式 S(t)
        self.neighbor_feature_dim = config.get('neighbor_feature_dim', 10)  # dx,dy,heading_x,heading_y,dvx,dvy,length,width,z,active
        self.waypoint_feature_dim = config.get('waypoint_feature_dim', 5)  # W_lane: dx,dy,dir_x,dir_y,width
        self.boundary_feature_dim = config.get('boundary_feature_dim', 2)  # 修改为2个特征：x,y
        # 使用来自 SelfraceSimulator 的共享哈希，仅作网格坐标与单元ID计算，不在此处重建静态索引
        self.spatial_hash = spatial_hash
        
        # 预计算每个quad_id对应的最近w_lanes和w_boundaries的ID
        self._precompute_quad_waypoint_associations()


    def _precompute_quad_waypoint_associations(self):
        """
        预计算每个quad_id对应的最近w_lanes和w_boundaries的ID。
        这样在generate时可以直接通过quad_id查找，避免重复计算。
        """
        num_quads = self.road_network.num_quads
        
        # 获取所有quad的中心点作为查询点
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
        
        # 预计算w_lanes关联
        if self.road_network.global_w_lane_waypoints.numel() > 0:
            self.quad_to_w_lanes_ids = self._compute_nearest_waypoint_ids(
                quad_centers, 
                self.road_network.global_w_lane_waypoints, 
                self.num_w_lanes
            )  # (num_quads, num_w_lanes)
        else:
            self.quad_to_w_lanes_ids = torch.full((num_quads, self.num_w_lanes), -1, dtype=torch.long, device=self.device)
        
        # 预计算w_boundaries关联
        if self.road_network.global_w_boundary_points.numel() > 0:
            self.quad_to_w_boundaries_ids = self._compute_nearest_waypoint_ids(
                quad_centers, 
                self.road_network.global_w_boundary_points, 
                self.num_w_boundaries
            )  # (num_quads, num_w_boundaries)
        else:
            self.quad_to_w_boundaries_ids = torch.full((num_quads, self.num_w_boundaries), -1, dtype=torch.long, device=self.device)

        self._cache_quad_waypoint_static_tensors()

    def _cache_quad_waypoint_static_tensors(self):
        """Cache per-quad static W_lane/W_boundary tensors; per-step code only transforms them."""
        self.quad_to_w_lanes_world = self._get_waypoints_by_ids(
            self.quad_to_w_lanes_ids,
            self.road_network.global_w_lane_waypoints,
        )
        self.quad_to_w_lane_dirs_world = self._get_waypoints_by_ids(
            self.quad_to_w_lanes_ids,
            self.road_network.global_w_lane_directions,
        )
        self.quad_to_w_lane_widths = self._get_scalar_by_ids(
            self.quad_to_w_lanes_ids,
            self.road_network.global_w_lane_widths,
        )
        self.quad_to_w_boundaries_world = self._get_waypoints_by_ids(
            self.quad_to_w_boundaries_ids,
            self.road_network.global_w_boundary_points,
        )

    def _compute_nearest_waypoint_ids(self, query_points: torch.Tensor, waypoints: torch.Tensor, num_nearest: int) -> torch.Tensor:
        """
        计算每个查询点到waypoints的最近num_nearest个点的ID。
        Args:
            query_points: 查询点坐标 (N, 2)
            waypoints: waypoints坐标 (M, 2)
            num_nearest: 需要找到的最近点数量
        Returns:
            最近点的ID (N, num_nearest)
        """
        if waypoints.numel() == 0 or num_nearest == 0:
            return torch.full((query_points.shape[0], num_nearest), -1, dtype=torch.long, device=self.device)
        
        # 计算所有查询点到所有waypoints的距离
        distances = torch.cdist(query_points, waypoints, p=2)  # (N, M)
        
        # 找到最近的num_nearest个点
        _, nearest_indices = torch.topk(distances, k=min(num_nearest, waypoints.shape[0]), dim=1, largest=False)
        
        # 如果waypoints数量不足，用-1填充，避免把真实第0个点误当padding。
        if waypoints.shape[0] < num_nearest:
            padding = torch.full((query_points.shape[0], num_nearest - waypoints.shape[0]), -1, dtype=torch.long, device=self.device)
            nearest_indices = torch.cat([nearest_indices, padding], dim=1)
        
        return nearest_indices

    def _get_precomputed_waypoints(self, agents_state: torch.Tensor, return_ids: bool = False) -> tuple:
        """
        使用预计算的数据获取w_lanes和w_boundaries。
        Args:
            agents_state: 形状为 (B, M, 7) 的agent状态张量
        Returns:
            tuple: (w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices)
        """
        batch_size, max_agents, _ = agents_state.shape
        
        # 获取每个agent所在的quad_id
        agent_positions = agents_state[..., :2]  # (B, M, 2)
        agent_positions_flat = agent_positions.view(-1, 2)  # (B*M, 2)
        
        # 找到每个agent最近的quad索引
        distances, quad_indices = self.road_network.find_nearest_lanes(agent_positions_flat, k=1, spatial_hash=self.spatial_hash)
        quad_indices = quad_indices.squeeze(-1)  # (B*M,)
        valid_quad = quad_indices >= 0
        safe_quad_indices = torch.where(valid_quad, quad_indices, torch.zeros_like(quad_indices))
        
        # 使用预计算的关联获取waypoint IDs
        w_lanes_ids = self.quad_to_w_lanes_ids[safe_quad_indices]  # (B*M, num_w_lanes)
        w_boundaries_ids = self.quad_to_w_boundaries_ids[safe_quad_indices]  # (B*M, num_w_boundaries)
        w_lanes_ids = torch.where(valid_quad.unsqueeze(-1), w_lanes_ids, torch.full_like(w_lanes_ids, -1))
        w_boundaries_ids = torch.where(valid_quad.unsqueeze(-1), w_boundaries_ids, torch.full_like(w_boundaries_ids, -1))
        
        # 静态坐标/方向/宽度已按 quad 缓存；这里仅按当前 quad gather。
        w_lanes_world = self.quad_to_w_lanes_world[safe_quad_indices]
        w_lane_dirs_world = self.quad_to_w_lane_dirs_world[safe_quad_indices]
        w_lane_widths = self.quad_to_w_lane_widths[safe_quad_indices]
        w_boundaries_world = self.quad_to_w_boundaries_world[safe_quad_indices]
        w_lanes_world = torch.where(valid_quad.view(-1, 1, 1), w_lanes_world, torch.zeros_like(w_lanes_world))
        w_lane_dirs_world = torch.where(valid_quad.view(-1, 1, 1), w_lane_dirs_world, torch.zeros_like(w_lane_dirs_world))
        w_lane_widths = torch.where(valid_quad.view(-1, 1), w_lane_widths, torch.zeros_like(w_lane_widths))
        w_boundaries_world = torch.where(valid_quad.view(-1, 1, 1), w_boundaries_world, torch.zeros_like(w_boundaries_world))
        
        # 恢复原始形状
        w_lanes_world = w_lanes_world.view(batch_size, max_agents, self.num_w_lanes, 2)
        w_lane_dirs_world = w_lane_dirs_world.view(batch_size, max_agents, self.num_w_lanes, 2)
        w_lane_widths = w_lane_widths.view(batch_size, max_agents, self.num_w_lanes)
        w_boundaries_world = w_boundaries_world.view(batch_size, max_agents, self.num_w_boundaries, 2)
        quad_indices = quad_indices.view(batch_size, max_agents)
        
        if not return_ids:
            return w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices

        w_lanes_ids = w_lanes_ids.view(batch_size, max_agents, self.num_w_lanes)
        w_boundaries_ids = w_boundaries_ids.view(batch_size, max_agents, self.num_w_boundaries)
        return (
            w_lanes_world,
            w_lane_dirs_world,
            w_lane_widths,
            w_boundaries_world,
            quad_indices,
            w_lanes_ids,
            w_boundaries_ids,
        )

    def get_w_lane_ids_for_agents(self, agents_state: torch.Tensor) -> tuple:
        """
        Return the precomputed W_lane ids in the same order used by generate().

        This lets downstream navigation features attach route-level quantities to
        each lane observation without duplicating nearest-neighbor map lookup.
        """
        batch_size, max_agents, _ = agents_state.shape
        agent_positions_flat = agents_state[..., :2].view(-1, 2)
        _, quad_indices = self.road_network.find_nearest_lanes(
            agent_positions_flat, k=1, spatial_hash=self.spatial_hash
        )
        quad_indices = quad_indices.squeeze(-1)
        valid_quad = quad_indices >= 0
        safe_quad_indices = torch.where(valid_quad, quad_indices, torch.zeros_like(quad_indices))
        w_lanes_ids = self.quad_to_w_lanes_ids[safe_quad_indices]
        w_lanes_ids = torch.where(valid_quad.unsqueeze(-1), w_lanes_ids, torch.full_like(w_lanes_ids, -1))
        return w_lanes_ids.view(batch_size, max_agents, self.num_w_lanes), quad_indices.view(batch_size, max_agents)

    def get_w_boundary_observation_for_agents(self, agents_state: torch.Tensor) -> tuple:
        """Return the exact W_boundary point set selected for each ego vehicle.

        Coordinates stay in the world frame for visualization; point ordering and
        padding IDs are identical to :meth:`generate_components`, whose network
        observation converts the same points to the ego-local frame.

        Returns:
            tuple: ``(points_world, point_ids, quad_indices)`` with shapes
            ``(B, M, num_w_boundaries, 2)``, ``(B, M, num_w_boundaries)`` and
            ``(B, M)`` respectively.  A point ID of ``-1`` denotes padding.
        """
        waypoint_data = self._get_precomputed_waypoints(agents_state, return_ids=True)
        return waypoint_data[3], waypoint_data[6], waypoint_data[4]

    def _get_waypoints_by_ids(self, waypoint_ids: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """
        根据ID列表获取waypoint坐标。
        Args:
            waypoint_ids: waypoint ID张量 (N, K)
            waypoints: 所有waypoints坐标 (M, 2)
        Returns:
            waypoint坐标张量 (N, K, 2)
        """
        if waypoints.numel() == 0:
            return torch.zeros(waypoint_ids.shape[0], waypoint_ids.shape[1], 2, device=self.device)
        
        # 创建有效ID掩码（-1表示无效）
        valid_mask = waypoint_ids >= 0
        
        safe_ids = torch.clamp(waypoint_ids, 0, waypoints.shape[0] - 1)
        result = waypoints[safe_ids]
        return torch.where(valid_mask.unsqueeze(-1), result, torch.zeros_like(result))

    def _get_scalar_by_ids(self, waypoint_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros(waypoint_ids.shape[0], waypoint_ids.shape[1], device=self.device)
        valid_mask = waypoint_ids >= 0
        safe_ids = torch.clamp(waypoint_ids, 0, values.shape[0] - 1)
        result = values[safe_ids]
        return torch.where(valid_mask, result, torch.zeros_like(result))

    def get_observation_dim(self) -> int:
        """
        计算观测向量的总维度
        Returns:
            int: 观测向量的总维度
        """
        # 计算各部分维度
        local_state_size = self.local_state_dim  # 局部状态维度
        neighbors_size = self.num_neighbors * self.neighbor_feature_dim  # 邻居特征维度
        w_lanes_size = self.num_w_lanes * self.waypoint_feature_dim  # 车道航点维度
        w_boundaries_size = self.num_w_boundaries * self.boundary_feature_dim  # 边界点维度
        # 总维度
        total_dim = local_state_size + neighbors_size + w_lanes_size + w_boundaries_size
        return total_dim
        
    def generate(self, agents_state: torch.Tensor,
                 control_state: torch.Tensor = None,
                 driving_style_params: torch.Tensor = None) -> torch.Tensor:
        """
        为所有环境中的所有 agent 生成一批观测。
        Args:
            agents_state (torch.Tensor): 全局状态张量 (B, M, 7)。
            agents_state[..., 0] = x
            agents_state[..., 1] = y
            agents_state[..., 2] = yaw
            agents_state[..., 3] = speed
            agents_state[..., 4] = vehicle_length
            agents_state[..., 5] = vehicle_width
            agents_state[..., 6] = active
        Returns:
            torch.Tensor: 展平后的观测向量张量 (B, M, feature_dim)。
            local_state: (B, M, 13)
            neighbors_local: (B, M, K, neighbor_feature_dim)
            w_lanes_local: (B, M, N_lanes, 5)
            w_boundaries_local: (B, M, N_boundaries, 2)
        """
        local_state, neighbors_local, w_lanes_local, w_boundaries_local = self.generate_components(
            agents_state,
            control_state=control_state,
            driving_style_params=driving_style_params,
        )

        # 3. 展平并拼接成最终的观测向量
        # 返回：自身绝对状态，邻居相对状态，车道线相对状态，边界线相对状态
        observation = torch.cat([
            local_state,
            neighbors_local.flatten(start_dim=2),
            w_lanes_local.flatten(start_dim=2),
            w_boundaries_local.flatten(start_dim=2)
        ], dim=2)
        
        return observation

    def generate_components(self, agents_state: torch.Tensor,
                            control_state: torch.Tensor = None,
                            driving_style_params: torch.Tensor = None,
                            return_map_ids: bool = False):
        """Return unpacked observation components without concatenating a full observation tensor."""
        neighbor_states_world = self._get_nearest_neighbors(agents_state)
        waypoint_data = self._get_precomputed_waypoints(agents_state, return_ids=return_map_ids)
        w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices = waypoint_data[:5]
        components = self._world_to_ego_centric(
            agents_state,
            neighbor_states_world,
            w_lanes_world,
            w_lane_dirs_world,
            w_lane_widths,
            w_boundaries_world,
            quad_indices,
            control_state,
            driving_style_params,
        )
        if not return_map_ids:
            return components
        return (*components, {
            'quad_indices': quad_indices,
            'w_lane_ids': waypoint_data[5],
            'w_boundary_ids': waypoint_data[6],
        })

    def _get_selected_neighbors(self, agents_state: torch.Tensor, agent_indices: torch.Tensor,
                                ego_states: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbor states for one selected ego per batch row."""
        batch_size, max_agents, _ = agents_state.shape
        if self.num_neighbors == 0:
            return torch.zeros(batch_size, 1, 0, 7, device=self.device, dtype=agents_state.dtype)

        query_pos = agents_state[..., :2]
        ego_pos = ego_states[:, 0, :2]
        dist_sq = (query_pos - ego_pos.unsqueeze(1)).pow(2).sum(dim=-1)

        agent_range = torch.arange(max_agents, device=self.device).view(1, max_agents)
        self_mask = agent_range == agent_indices.view(batch_size, 1)
        inactive_mask = agents_state[..., 6] < 0.5
        dist_sq = dist_sq.masked_fill(self_mask | inactive_mask, float('inf'))
        dist_sq = dist_sq.masked_fill(dist_sq > self.horizon ** 2, float('inf'))

        k_eff = min(self.num_neighbors, max_agents)
        _, topk_indices = torch.topk(dist_sq, k=k_eff, dim=-1, largest=False)
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1)
        neighbor_states = agents_state[batch_idx, topk_indices].unsqueeze(1)
        valid_neighbor_dists = dist_sq[batch_idx, topk_indices].unsqueeze(1)
        is_valid_neighbor = torch.isfinite(valid_neighbor_dists)

        replacement = ego_states.unsqueeze(2).expand(-1, -1, k_eff, -1).clone()
        replacement[..., 4] = 0.0
        replacement[..., 5] = 0.0
        replacement[..., 6] = 0.0
        neighbor_states = torch.where(is_valid_neighbor.unsqueeze(-1), neighbor_states, replacement)
        if k_eff < self.num_neighbors:
            pad = replacement.new_zeros(batch_size, 1, self.num_neighbors - k_eff, neighbor_states.shape[-1])
            neighbor_states = torch.cat([neighbor_states, pad], dim=2)
        return neighbor_states

    def generate_selected_components(self, agents_state: torch.Tensor, agent_indices: torch.Tensor,
                                     control_state: torch.Tensor = None,
                                     driving_style_params: torch.Tensor = None,
                                     return_map_ids: bool = False):
        """Return observation components for one selected agent per batch row."""
        batch_size, max_agents, _ = agents_state.shape
        agent_indices = agent_indices.to(device=self.device, dtype=torch.long).view(batch_size)
        safe_indices = torch.clamp(agent_indices, 0, max_agents - 1)
        batch_idx = torch.arange(batch_size, device=self.device)

        ego_states = agents_state[batch_idx, safe_indices].unsqueeze(1)
        neighbor_states_world = self._get_selected_neighbors(agents_state, safe_indices, ego_states)
        waypoint_data = self._get_precomputed_waypoints(ego_states, return_ids=return_map_ids)
        w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices = waypoint_data[:5]

        control_selected = None
        if control_state is not None:
            if control_state.shape[1] == 1:
                control_selected = control_state
            else:
                control_selected = control_state[batch_idx, safe_indices].unsqueeze(1)
        style_selected = None
        if driving_style_params is not None:
            if driving_style_params.shape[1] == 1:
                style_selected = driving_style_params
            else:
                style_selected = driving_style_params[batch_idx, safe_indices].unsqueeze(1)

        components = self._world_to_ego_centric(
            ego_states,
            neighbor_states_world,
            w_lanes_world,
            w_lane_dirs_world,
            w_lane_widths,
            w_boundaries_world,
            quad_indices,
            control_selected,
            style_selected,
        )
        if not return_map_ids:
            return components
        return (*components, {
            'quad_indices': quad_indices,
            'w_lane_ids': waypoint_data[5],
            'w_boundary_ids': waypoint_data[6],
        })
    
    def _get_nearest_neighbors(self, agents_state: torch.Tensor) -> torch.Tensor:
        """为每个 agent 找到最近的 K 个邻居。完全向量化版本。"""
        batch_size, max_agents, _ = agents_state.shape
        # 如果不需要邻居，直接返回空的张量
        if self.num_neighbors == 0:
            return torch.zeros(batch_size, max_agents, 0, 7, device=self.device)
        # 获取所有agent的坐标
        query_pos = agents_state[..., :2] # (B, M, 2)
        # 使用 torch.cdist 计算每个环境中所有 agent 之间的配对距离
        dist_sq = torch.cdist(query_pos, query_pos, p=2).pow(2) # (B, M, M)

        # 创建一个掩码来过滤掉不应被视为邻居的 agent
        # 1. Agent 不能是其自身的邻居 (对角线)
        self_mask = torch.eye(max_agents, device=self.device, dtype=torch.bool).expand(batch_size, -1, -1)
        # 2. 不活跃的 agent 不能作为邻居
        inactive_mask = (agents_state[..., 6] < 0.5).unsqueeze(1).expand(-1, max_agents, -1)
        # 3. 距离超过视野范围的邻居不考虑
        dist_sq[self_mask | inactive_mask] = float('inf')
        dist_sq[dist_sq > self.horizon**2] = float('inf') 
        # 4. 找到最近的 K 个。小规模 smoke test 中 M 可能小于配置的 num_neighbors，
        #    因此先取可用数量，再在后面补齐到固定观测维度。
        k_eff = min(self.num_neighbors, max_agents)
        _, topk_indices = torch.topk(dist_sq, k=k_eff, dim=-1, largest=False) # (B, M, k_eff)
        # 5. 使用高级索引高效地收集邻居状态
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1)
        agent_idx = torch.arange(max_agents, device=self.device).view(1, max_agents, 1)
        neighbor_states = agents_state[batch_idx, topk_indices] # (B, M, K, 7)
        # 6. 如果邻居是无效的 (距离为inf)，则其状态需要被掩码；
        #    为避免后续局部坐标计算出现 dx,dy = -ego_pos 的伪值，
        #    将无效邻居的状态设置为等同于对应 ego 的状态（使相对量为0）。
        valid_neighbor_dists = dist_sq[batch_idx, agent_idx, topk_indices]
        is_valid_neighbor = torch.isfinite(valid_neighbor_dists) # (B, M, K)
        K_neighbors = topk_indices.shape[-1]
        ego_states_expanded = agents_state.unsqueeze(2).expand(-1, -1, K_neighbors, -1)  # (B, M, K, 7)
        # 使无效邻居的相对位置/速度为0：复制ego的 [x,y,yaw,speed]；
        # 同时将尺寸与active置零，避免下游看到伪造的车辆尺寸与激活标志。
        replacement = ego_states_expanded.clone()
        replacement[..., 4] = 0.0  # length
        replacement[..., 5] = 0.0  # width
        replacement[..., 6] = 0.0  # active
        neighbor_states = torch.where(is_valid_neighbor.unsqueeze(-1), neighbor_states, replacement)
        if k_eff < self.num_neighbors:
            pad = replacement.new_zeros(batch_size, max_agents, self.num_neighbors - k_eff, neighbor_states.shape[-1])
            neighbor_states = torch.cat([neighbor_states, pad], dim=2)
        return neighbor_states
    
    def _world_to_ego_centric(self, ego_states, neighbor_states, w_lanes_world, w_lane_dirs_world,
                              w_lane_widths, w_boundaries_world, quad_indices,
                              control_state=None, driving_style_params=None):
        """将世界坐标系下的状态转换为以每个 agent 为中心的坐标系。"""
        B, M, _ = ego_states.shape
        K_neighbors = neighbor_states.shape[2]
        ego_pos = ego_states[..., :2] # (B, M, 2)
        ego_yaw = ego_states[..., 2]  # (B, M)
        cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)

        # 使用标准2D旋转矩阵（车左边为正）
        rot_matrix = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=-1), 
            torch.stack([sin_yaw, cos_yaw], dim=-1)
        ], dim=-2)

        # 向量化bmm操作：将 (B, M) 批次展平为 (B*M)，执行bmm，然后重塑
        def batch_rotate(points_world, ego_pos, rot_matrix):
            # points_world: (B, M, N, 2), ego_pos: (B, M, 2), rot_matrix: (B, M, 2, 2)
            rel_pos = points_world - ego_pos.unsqueeze(2)
            B, M, N, D = rel_pos.shape
            return torch.bmm(rel_pos.view(B*M, N, D), rot_matrix.view(B*M, D, D)).view(B, M, N, D)
        # 将世界坐标系下的车道线和边界线转换到局部坐标系
        w_lanes_xy_local = batch_rotate(w_lanes_world, ego_pos, rot_matrix)
        w_boundaries_local = batch_rotate(w_boundaries_world, ego_pos, rot_matrix)
        B_l, M_l, N_lanes, _ = w_lane_dirs_world.shape
        w_lane_dirs_local = torch.bmm(
            w_lane_dirs_world.view(B_l * M_l, N_lanes, 2),
            rot_matrix.view(B_l * M_l, 2, 2)
        ).view(B_l, M_l, N_lanes, 2)
        w_lanes_local = torch.cat([w_lanes_xy_local, w_lane_dirs_local, w_lane_widths.unsqueeze(-1)], dim=-1)
        # --- 转换邻居 ---
        if K_neighbors > 0:
            rel_pos_neighbors = neighbor_states[..., :2] - ego_pos.unsqueeze(2) # (B, M, K, 2)
            local_pos_neighbors = torch.bmm(
                rel_pos_neighbors.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)
            ).view(B, M, K_neighbors, 2)
            # 速度转换：计算邻居相对于ego的相对速度
            # 1. 获取邻居的绝对速度
            neighbor_speed = neighbor_states[..., 3]
            neighbor_yaw = neighbor_states[..., 2]
            vx_neighbor_world = neighbor_speed * torch.cos(neighbor_yaw)
            vy_neighbor_world = neighbor_speed * torch.sin(neighbor_yaw)
            v_neighbor_world = torch.stack([vx_neighbor_world, vy_neighbor_world], dim=-1)  # (B, M, K, 2)
            
            # 2. 获取ego的绝对速度
            ego_speed = ego_states[..., 3]  # (B, M)
            ego_yaw = ego_states[..., 2]    # (B, M)
            vx_ego_world = ego_speed * torch.cos(ego_yaw)
            vy_ego_world = ego_speed * torch.sin(ego_yaw)
            v_ego_world = torch.stack([vx_ego_world, vy_ego_world], dim=-1)  # (B, M, 2)
            
            # 3. 计算相对速度：v_relative = v_neighbor - v_ego
            v_relative_world = v_neighbor_world - v_ego_world.unsqueeze(2)  # (B, M, K, 2)
            
            # 4. 将相对速度转换到ego的局部坐标系
            v_local = torch.bmm(v_relative_world.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)).view(B, M, K_neighbors, 2)
            length = neighbor_states[..., 4].unsqueeze(-1)
            width = neighbor_states[..., 5].unsqueeze(-1)
            heading_world = torch.stack([torch.cos(neighbor_yaw), torch.sin(neighbor_yaw)], dim=-1)
            heading_local = torch.bmm(
                heading_world.view(B * M, K_neighbors, 2),
                rot_matrix.view(B * M, 2, 2)
            ).view(B, M, K_neighbors, 2)
            z = torch.zeros(B, M, K_neighbors, 1, device=self.device, dtype=ego_states.dtype)
            active_flag = neighbor_states[..., 6].unsqueeze(-1)
            if self.neighbor_feature_dim <= 7:
                neighbors_local = torch.cat([local_pos_neighbors, v_local, length, width, active_flag], dim=-1)
            else:
                neighbors_local = torch.cat([
                    local_pos_neighbors,
                    heading_local,
                    v_local,
                    length,
                    width,
                    z,
                    active_flag,
                ], dim=-1)
            if neighbors_local.shape[-1] != self.neighbor_feature_dim:
                fitted = torch.zeros(B, M, K_neighbors, self.neighbor_feature_dim, device=self.device, dtype=ego_states.dtype)
                copy_dim = min(neighbors_local.shape[-1], self.neighbor_feature_dim)
                fitted[..., :copy_dim] = neighbors_local[..., :copy_dim]
                if self.neighbor_feature_dim > neighbors_local.shape[-1]:
                    fitted[..., -1] = active_flag.squeeze(-1)
                neighbors_local = fitted
        else:
            # 如果没有邻居，创建空的邻居特征张量
            neighbors_local = torch.zeros(B, M, 0, self.neighbor_feature_dim, device=self.device)
        # --- 创建每个 Agent 自身的原文式 S(t) ---
        local_state = torch.zeros(B, M, self.local_state_dim, device=self.device, dtype=ego_states.dtype)
        safe_quad_indices = torch.where(quad_indices >= 0, quad_indices, torch.zeros_like(quad_indices))
        road_directions = self.road_network.quad_directions[safe_quad_indices]
        nearest_centerlines = self.road_network.quad_centerlines[safe_quad_indices]
        road_starts = nearest_centerlines[:, :, 0, :]
        ap = ego_pos - road_starts
        # 与局部坐标系保持一致：x 为车道前向，y 为车道左侧，c/theta 左正右负。
        lane_center_dist = road_directions[..., 0] * ap[..., 1] - road_directions[..., 1] * ap[..., 0]
        vehicle_directions = torch.stack([torch.cos(ego_yaw), torch.sin(ego_yaw)], dim=-1)
        cross_heading = road_directions[..., 0] * vehicle_directions[..., 1] - road_directions[..., 1] * vehicle_directions[..., 0]
        dot_heading = (road_directions * vehicle_directions).sum(dim=-1)
        theta = torch.atan2(cross_heading, dot_heading)
        valid_quad = quad_indices >= 0
        lane_center_dist = torch.where(valid_quad, lane_center_dist, torch.zeros_like(lane_center_dist))
        theta = torch.where(valid_quad, theta, torch.zeros_like(theta))

        control = torch.zeros(B, M, 3, device=self.device, dtype=ego_states.dtype) if control_state is None else control_state.to(self.device, dtype=ego_states.dtype)
        style = torch.ones(B, M, 4, device=self.device, dtype=ego_states.dtype) if driving_style_params is None else driving_style_params.to(self.device, dtype=ego_states.dtype)
        if self.local_state_dim >= 13:
            local_state[..., 0] = lane_center_dist
            local_state[..., 1] = theta
            local_state[..., 2] = self.road_network.quad_curvatures[safe_quad_indices]
            local_state[..., 3] = ego_states[..., 3]
            local_state[..., 4] = self.speed_limit * style[..., 3]
            local_state[..., 5] = control[..., 0]  # steering angle phi
            local_state[..., 6] = control[..., 1]  # a_long
            local_state[..., 7] = control[..., 2]  # a_lat
            local_state[..., 8] = style[..., 2]   # C_acc
            local_state[..., 9] = style[..., 0]   # C_throttle
            local_state[..., 10] = style[..., 1]  # C_steer
            local_state[..., 11] = ego_states[..., 4]
            local_state[..., 12] = ego_states[..., 5]
        else:
            local_state[..., 0] = 0.0
            local_state[..., 1] = 0.0
            local_state[..., 2] = 0.0
            local_state[..., 3] = ego_states[..., 3]
            local_state[..., 4] = ego_states[..., 4]
            local_state[..., 5] = ego_states[..., 5]
            local_state[..., 6] = ego_states[..., 6]
        return local_state, neighbors_local, w_lanes_local, w_boundaries_local

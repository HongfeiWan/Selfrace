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

    def _get_precomputed_waypoints(self, agents_state: torch.Tensor) -> tuple:
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
        
        # 通过ID获取waypoint坐标
        w_lanes_world = self._get_waypoints_by_ids(w_lanes_ids, self.road_network.global_w_lane_waypoints)
        w_lane_dirs_world = self._get_waypoints_by_ids(w_lanes_ids, self.road_network.global_w_lane_directions)
        w_lane_widths = self._get_scalar_by_ids(w_lanes_ids, self.road_network.global_w_lane_widths)
        w_boundaries_world = self._get_waypoints_by_ids(w_boundaries_ids, self.road_network.global_w_boundary_points)
        
        # 恢复原始形状
        w_lanes_world = w_lanes_world.view(batch_size, max_agents, self.num_w_lanes, 2)
        w_lane_dirs_world = w_lane_dirs_world.view(batch_size, max_agents, self.num_w_lanes, 2)
        w_lane_widths = w_lane_widths.view(batch_size, max_agents, self.num_w_lanes)
        w_boundaries_world = w_boundaries_world.view(batch_size, max_agents, self.num_w_boundaries, 2)
        quad_indices = quad_indices.view(batch_size, max_agents)
        
        return w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices

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
        # 1. 获取世界坐标系下的特征
        # (B, M, K, 7)
        neighbor_states_world = self._get_nearest_neighbors(agents_state)
        
        # 使用预计算的数据获取w_lanes和w_boundaries
        w_lanes_world, w_lane_dirs_world, w_lane_widths, w_boundaries_world, quad_indices = self._get_precomputed_waypoints(agents_state)

        # 2. 将所有信息转换到每个 Agent 的局部坐标系
        local_state, neighbors_local, w_lanes_local, w_boundaries_local = self._world_to_ego_centric(
            agents_state, neighbor_states_world, w_lanes_world, w_lane_dirs_world,
            w_lane_widths, w_boundaries_world, quad_indices, control_state, driving_style_params
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
        ego_vel = ego_states[..., 3] # (B, M)
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    print("RoadNetwork 测试")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 加载地图数据
    map_path = "maps/processed_map_Town01_stitched.json"
    print(f"加载地图: {map_path}")

    # 绘制车辆矩形
    def draw_vehicle(ax, x, y, yaw, speed , length=4.5, width=2.0, color='green', alpha=0.8):
        """绘制车辆矩形"""
        # 车辆矩形的四个角点（相对于车辆中心）
        half_length = length / 2
        half_width = width / 2
        
        # 车辆矩形的四个角点（相对于车辆中心）
        corners = np.array([
            [-half_length, -half_width],  # 左下
            [half_length, -half_width],   # 右下
            [half_length, half_width],    # 右上
            [-half_length, half_width]    # 左上
        ])
        # 旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # 旋转角点
        rotated_corners = corners @ rotation_matrix.T
        
        # 平移到车辆位置
        vehicle_corners = rotated_corners + np.array([x, y])
        
        # 绘制车辆矩形
        vehicle_x = np.append(vehicle_corners[:, 0], vehicle_corners[0, 0])
        vehicle_y = np.append(vehicle_corners[:, 1], vehicle_corners[0, 1])
        ax.plot(vehicle_x, vehicle_y, color=color, linewidth=2, alpha=alpha)
        
        # 绘制车辆朝向箭头
        arrow_length = speed
        arrow_dx = arrow_length * cos_yaw
        arrow_dy = arrow_length * sin_yaw
        ax.arrow(x, y, arrow_dx, arrow_dy, 
                head_width=1, head_length=0.5, fc=color, ec=color, alpha=alpha)
        # 标记车辆中心
        ax.plot(x, y, 'o', color=color, markersize=4, alpha=alpha)    
    
    try:
        # 创建RoadNetwork实例
        road_network = RoadNetwork(map_path, device)
        # 获取quads顶点数据
        quads_vertices_np = road_network.quads_vertices.cpu().numpy()
        # 随机选择一个quad_id
        random_quad_id = random.choice(road_network.quad_ids.cpu().numpy())
        print(f"随机选择quad_id: {random_quad_id}")
        # 根据quad_id找到对应的索引
        quad_id_positions = torch.where(road_network.quad_ids == random_quad_id)[0]
        random_quad_idx = quad_id_positions[0].item()
        
        # 获取选中quad的顶点
        selected_quad = quads_vertices_np[random_quad_idx]
        # 在quad范围内随机生成车辆位置
        # 改进的随机点生成方法，确保点在quad内
        def random_point_in_quad_improved(quad_vertices):
            """改进的quad内随机点生成，确保点在quad内部"""
            # 计算quad的边界框
            min_x, min_y = np.min(quad_vertices, axis=0)
            max_x, max_y = np.max(quad_vertices, axis=0)
            
            # 在边界框内随机生成点，直到找到在quad内的点
            max_attempts = 100
            for _ in range(max_attempts):
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                point = np.array([x, y])
                
                # 检查点是否在quad内（使用射线法）
                if is_point_in_quad(point, quad_vertices):
                    return point
            
            # 如果失败，返回quad的中心点
            center = np.mean(quad_vertices, axis=0)
            print(f"警告：无法在quad内生成随机点，使用中心点: {center}")
            return center
        
        def is_point_in_quad(point, quad_vertices):
            """使用射线法判断点是否在quad内"""
            x, y = point
            n = len(quad_vertices)
            inside = False
            
            p1x, p1y = quad_vertices[0]
            for i in range(n + 1):
                p2x, p2y = quad_vertices[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
        
        # 使用改进的方法生成车辆位置
        vehicle_pos = random_point_in_quad_improved(selected_quad)
        vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向
        # 绘制地图
        print("绘制地图...")
        fig, ax = plt.subplots(figsize=(12, 8))

        # 只绘制ego周围的quads
        vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
        vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
        # 找到距离ego最近的quads
        distances, nearest_indices = road_network.find_nearest_lanes(vehicle_pos_tensor, k=400)
        nearest_indices = nearest_indices.cpu().numpy().flatten()
        nearest_quad_idx = nearest_indices[0]  # 最近的quad索引
        nearby_quads = nearest_indices.tolist()

        # 在nearby_quads中随机选择一个quad生成第二辆车
        if len(nearby_quads) > 1:
            # 随机选择一个不同于最近quad的quad
            available_quads = [q for q in nearby_quads if q != nearest_quad_idx]
            if available_quads:
                second_quad_idx = random.choice(available_quads)
                second_quad = quads_vertices_np[second_quad_idx]
                # 在第二个quad中生成随机车辆位置
                second_vehicle_pos = random_point_in_quad_improved(second_quad)
                second_vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向

        # 创建agents_state (B=1, M=2, 7个特征)
        agents_state = torch.zeros(1, 2, 7, device=device)
        # 第一辆车的信息
        agents_state[0, 0, 0] = float(vehicle_pos[0])  # x
        agents_state[0, 0, 1] = float(vehicle_pos[1])  # y
        agents_state[0, 0, 2] = float(vehicle_yaw)     # yaw
        agents_state[0, 0, 3] = 10.0                   # speed (m/s)
        agents_state[0, 0, 4] = 4.5                    # vehicle_length (m)
        agents_state[0, 0, 5] = 2.0                    # vehicle_width (m)
        agents_state[0, 0, 6] = 1.0                    # active
        # 第二辆车的信息（如果存在）
        if len(nearby_quads) > 1 and 'second_vehicle_pos' in locals():
            agents_state[0, 1, 0] = float(second_vehicle_pos[0])  # x
            agents_state[0, 1, 1] = float(second_vehicle_pos[1])  # y
            agents_state[0, 1, 2] = float(second_vehicle_yaw)     # yaw
            agents_state[0, 1, 3] = 8.0                           # speed (m/s)
            agents_state[0, 1, 4] = 4.5                           # vehicle_length (m)
            agents_state[0, 1, 5] = 2.0                           # vehicle_width (m)
            agents_state[0, 1, 6] = 1.0                           # active


        

        # 测试ObservationGenerator
        print("\n=== 测试ObservationGenerator ===")
        # 创建配置字典
        config = {
            'num_neighbors': 1,  # 只有2个agents，所以最多1个邻居
            'num_w_lanes': 25,
            'num_w_boundaries': 26,
            'horizon': 100.0,
            'local_state_dim': 7,  # 修改为7个特征：x, y, yaw, speed, length, width, active
            'neighbor_feature_dim': 7,  # 修改为7个特征：dx, dy, vx, vy, length, width, active
            'waypoint_feature_dim': 2,
            'boundary_feature_dim': 2
        }

        # 创建空间哈希用于加速查询
        # 计算所有quad的边界框
        all_verts = road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 设置合适的网格大小
        cell_size = 5  # 5米的网格单元
        spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, device)
        # 构建静态索引
        quad_centers = road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
        quad_min_bounds = torch.min(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
        quad_max_bounds = torch.max(road_network.quad_centerlines, dim=1)[0]  # (num_quads, 2)
        quad_bounds = torch.stack([quad_min_bounds, quad_max_bounds], dim=1)  # (num_quads, 2, 2)
        spatial_hash.build_static_index(quad_bounds)
        print(f"空间哈希网格创建完成，网格大小: {cell_size:.2f}m")
        
        # 创建ObservationGenerator实例
        observation_generator = ObservationGenerator(road_network, config, device, spatial_hash)
        print(f"观测维度: {observation_generator.get_observation_dim()}")
        
        # 使用find_nearest_lanes找到最近的quad
        distances, quad_indices = road_network.find_nearest_lanes(agents_state[0, 0, :2], k=1, spatial_hash=spatial_hash)
        # 获取对应的quad_id (polyId)
        quad_id = road_network.quad_ids[quad_indices.squeeze(-1)]
        print(f"第一辆车所在quad_id: {quad_id.item()}")
        
        # 验证quad_id一致性
        print(f"\n=== 验证quad_id一致性 ===")
        print(f"随机选择的quad_id: {random_quad_id}")
        print(f"find_nearest_lanes得到的quad_id: {quad_id.item()}")
        print(f"是否一致: {random_quad_id == quad_id.item()}")
        
        if random_quad_id != quad_id.item():
            print(f"不一致的原因分析:")
            print(f"1. 车辆位置: {vehicle_pos}")
            
            # 计算车辆到随机选择quad的距离
            random_quad_center = road_network.quad_centerlines[random_quad_idx].mean(dim=0)
            dist_to_random = torch.norm(torch.tensor(vehicle_pos, device=device) - random_quad_center)
            print(f"2. 车辆到随机quad中心的距离: {dist_to_random.item():.2f}")
            
            # 计算车辆到最近quad的距离
            nearest_quad_center = road_network.quad_centerlines[quad_indices.item()].mean(dim=0)
            dist_to_nearest = torch.norm(torch.tensor(vehicle_pos, device=device) - nearest_quad_center)
            print(f"3. 车辆到最近quad中心的距离: {dist_to_nearest.item():.2f}")
            
            print(f"4. 距离差异: {abs(dist_to_random - dist_to_nearest).item():.2f}")
            
            # 检查车辆是否真的在随机选择的quad内
            def is_point_in_quad(point, quad_vertices):
                """使用射线法判断点是否在quad内"""
                x, y = point
                n = len(quad_vertices)
                inside = False
                
                p1x, p1y = quad_vertices[0]
                for i in range(n + 1):
                    p2x, p2y = quad_vertices[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                
                return inside
            
            is_in_random_quad = is_point_in_quad(vehicle_pos, selected_quad)
            print(f"5. 车辆是否在随机选择的quad内: {is_in_random_quad}")
            
            if not is_in_random_quad:
                print("6. 原因：重心坐标法生成的点不在quad内！")
                print("7. 建议：使用改进的随机点生成方法")
        
        # 生成观测
        observation = observation_generator.generate(agents_state)
        # 从observation中提取w_lanes_local和w_boundaries_local
        # 计算各部分在观测向量中的位置
        local_state_dim = config['local_state_dim']
        neighbor_feature_dim = config['neighbor_feature_dim']
        num_neighbors = config['num_neighbors']
        num_w_lanes = config['num_w_lanes']
        num_w_boundaries = config['num_w_boundaries']
        waypoint_feature_dim = config['waypoint_feature_dim']
        boundary_feature_dim = config['boundary_feature_dim']
        
        # 计算各部分在观测向量中的位置
        local_state_size = local_state_dim
        neighbors_size = num_neighbors * neighbor_feature_dim
        w_lanes_size = num_w_lanes * waypoint_feature_dim
        w_boundaries_size = num_w_boundaries * waypoint_feature_dim
        
        # 提取第一辆车的w_lanes_local和w_boundaries_local
        vehicle1_obs = observation[0, 0].cpu().numpy()
        w_lanes_start = local_state_size + neighbors_size
        w_boundaries_start = w_lanes_start + w_lanes_size
        w_lanes_local = vehicle1_obs[w_lanes_start:w_boundaries_start].reshape(num_w_lanes, waypoint_feature_dim)
        w_boundaries_local = vehicle1_obs[w_boundaries_start:].reshape(num_w_boundaries, waypoint_feature_dim)
        
        # 获取第一辆车的世界坐标和朝向（从agents_state中获取，确保一致性）
        vehicle_world_pos = np.array([float(agents_state[0, 0, 0]), float(agents_state[0, 0, 1])])
        vehicle_world_yaw = float(agents_state[0, 0, 3])  # 注意：agents_state[..., 3]是yaw
        
        # 绘制w_lanes_local (车道线)
        if w_lanes_local.shape[0] > 0:
            # 过滤掉无效的点（全零或NaN）
            valid_lanes = w_lanes_local[~np.all(w_lanes_local == 0, axis=1)]
            valid_lanes = valid_lanes[~np.any(np.isnan(valid_lanes), axis=1)]
            if valid_lanes.shape[0] > 0:
                # 逆变换：从local坐标转换回world坐标
                # 按照正确代码的实现方式
                ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
                cos_yaw = np.cos(ego_yaw)
                sin_yaw = np.sin(ego_yaw)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                ego_pos_global = np.array([ego_x, ego_y])
                world_lanes = (valid_lanes @ rotation_matrix.T) + ego_pos_global
                # 绘制车道线点
                ax.scatter(world_lanes[:, 0], world_lanes[:, 1], c='orange', s=20, alpha=0.8, label='w_lanes_local')
            else:
                print("没有有效的车道线点")
        
        # 绘制w_boundaries_local (边界线)
        if w_boundaries_local.shape[0] > 0:
            # 过滤掉无效的点（全零或NaN）
            valid_boundaries = w_boundaries_local[~np.all(w_boundaries_local == 0, axis=1)]
            valid_boundaries = valid_boundaries[~np.any(np.isnan(valid_boundaries), axis=1)]
            if valid_boundaries.shape[0] > 0:
                # 逆变换：从local坐标转换回world坐标
                # 按照正确代码的实现方式
                ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
                cos_yaw = np.cos(ego_yaw)
                sin_yaw = np.sin(ego_yaw)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                ego_pos_global = np.array([ego_x, ego_y])
                world_boundaries = (valid_boundaries @ rotation_matrix.T) + ego_pos_global
                # 绘制边界线点
                ax.scatter(world_boundaries[:, 0], world_boundaries[:, 1], c='purple', s=15, alpha=0.8, label='w_boundaries_local')
            else:
                print("没有有效的边界线点")
        
        # 计算邻居特征在observation中的位置
        neighbors_start = local_state_dim
        neighbors_end = neighbors_start + num_neighbors * neighbor_feature_dim
        # 提取第一辆车的邻居观测
        neighbors_obs = vehicle1_obs[neighbors_start:neighbors_end].reshape(num_neighbors, neighbor_feature_dim)
        # 过滤掉无效的邻居（全零）
        valid_neighbors = neighbors_obs[np.any(neighbors_obs != 0, axis=1)]
        if valid_neighbors.shape[0] > 0:
            print(f"观测到 {valid_neighbors.shape[0]} 个有效邻居")
            # 获取ego车辆信息用于逆变换
            ego_x, ego_y, ego_yaw, *_ = agents_state[0, 0].cpu().numpy()
            cos_yaw = np.cos(ego_yaw)
            sin_yaw = np.sin(ego_yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            ego_pos_global = np.array([ego_x, ego_y])
            for i, neighbor_local in enumerate(valid_neighbors):
                # neighbor_local包含7个特征：[dx, dy, vx, vy, length, width, active]
                dx_local, dy_local, vx_local, vy_local, length, width, active = neighbor_local
                if active > 0.5:  # 只绘制活跃的邻居
                    # 1. 逆变换邻居位置：从局部坐标转换回全局坐标
                    neighbor_pos_local = np.array([dx_local, dy_local])
                    neighbor_pos_global = (neighbor_pos_local @ rotation_matrix.T) + ego_pos_global
                    
                    # 2. 逆变换邻居相对速度：从局部坐标转换回全局坐标
                    neighbor_vel_local = np.array([vx_local, vy_local])
                    neighbor_vel_global = neighbor_vel_local @ rotation_matrix.T

                    # 在邻居位置旁边显示相对速度文本
                    ax.text(neighbor_pos_global[0] + 2, neighbor_pos_global[1] + 2, 
                           f'relative vel: ({vx_local:.1f}, {vy_local:.1f})', 
                           color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                    
                    # 3. 计算邻居的绝对速度 = ego绝对速度 + 邻居相对速度
                    ego_speed = agents_state[0, 0, 3].cpu().numpy()
                    ego_yaw_rad = agents_state[0, 0, 2].cpu().numpy()
                    ego_vel_global = np.array([
                        ego_speed * np.cos(ego_yaw_rad),
                        ego_speed * np.sin(ego_yaw_rad)
                    ])
                    neighbor_vel_absolute_global = ego_vel_global + neighbor_vel_global

                    # 3. 绘制邻居位置
                    ax.scatter(neighbor_pos_global[0], neighbor_pos_global[1], c='red', s=10, alpha=0.8, marker='o', label=f'Neighbor_{i}' if i == 0 else "")
                    # 4. 绘制邻居相对速度箭头（正交分解）
                    vel_x_arrow = neighbor_vel_absolute_global[0] 
                    vel_y_arrow = neighbor_vel_absolute_global[1] 
                    # X方向相对速度箭头（红色）
                    if abs(vel_x_arrow) > 0.1:  # 只绘制有意义的箭头
                        ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                                vel_x_arrow, 0, head_width=2, head_length=1, 
                                fc='red', ec='red', alpha=0.8, zorder=10)
                    # Y方向相对速度箭头（蓝色）
                    if abs(vel_y_arrow) > 0.1:  # 只绘制有意义的箭头
                        ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                                0, vel_y_arrow, head_width=2, head_length=1, 
                                fc='green', ec='green', alpha=0.8, zorder=10)
                    # 5. 绘制邻居绝对速度箭头（紫色）
                    if np.linalg.norm(neighbor_vel_absolute_global) > 0.1:
                        ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                                neighbor_vel_absolute_global[0], neighbor_vel_absolute_global[1], 
                                head_width=3, head_length=2, fc='purple', ec='purple', 
                                alpha=0.9, zorder=11, linewidth=2, label=f'Absolute Vel_{i}' if i == 0 else "")
                    # 6. 绘制邻居车辆的矩形（使用复原的长度和宽度）
                    # 计算邻居的朝向（从绝对速度向量推断）
                    if np.linalg.norm(neighbor_vel_absolute_global) > 0.1:
                        neighbor_yaw = np.arctan2(neighbor_vel_absolute_global[1], neighbor_vel_absolute_global[0])
                    else:
                        neighbor_yaw = 0.0  # 如果速度很小，假设朝向为0
                    
                    # 绘制邻居车辆矩形
                    draw_vehicle(ax, neighbor_pos_global[0], neighbor_pos_global[1], 
                               neighbor_yaw, np.linalg.norm(neighbor_vel_absolute_global), 
                               length, width, color='red', alpha=0.6)    
                    
        else:
            print("没有观测到有效的邻居")

        # 绘制ego矩形
        draw_vehicle(ax, agents_state[0, 0, 0].cpu().numpy(), agents_state[0, 0, 1].cpu().numpy(), agents_state[0, 0, 2].cpu().numpy(), agents_state[0, 0, 3].cpu().numpy())
        # 绘制第二辆车的矩形(真值)
        draw_vehicle(ax, agents_state[0, 1, 0].cpu().numpy(), agents_state[0, 1, 1].cpu().numpy(), agents_state[0, 1, 2].cpu().numpy(), agents_state[0, 1, 3].cpu().numpy(), color='blue',alpha=0.5)

        # 绘制车辆周围的quads
        for i in nearby_quads:
            quad = quads_vertices_np[i]
            # 绘制quad边界
            quad_x = [quad[0][0], quad[1][0], quad[2][0], quad[3][0], quad[0][0]]
            quad_y = [quad[0][1], quad[1][1], quad[2][1], quad[3][1], quad[0][1]]
            # 判断是否为最近的quad，决定颜色
            if i == nearest_quad_idx:
                # 最近的quad用红色
                ax.plot(quad_x, quad_y, 'r-', alpha=0.5, linewidth=2, label='nearest quad')
                centerline = road_network.quad_centerlines[i].cpu().numpy()
                ax.plot(centerline[:, 0], centerline[:, 1], 'r-', linewidth=3, alpha=0.8)
                                
                # 为最近quad的中线添加箭头
                start_point = centerline[0]
                end_point = centerline[1]
                # 计算箭头位置（在中心线的中点）
                arrow_pos = (start_point + end_point) / 2
                # 计算箭头方向
                arrow_direction = end_point - start_point
                arrow_length = np.linalg.norm(arrow_direction) * 0.3  # 箭头长度为线段长度的30%
                arrow_direction_normalized = arrow_direction / np.linalg.norm(arrow_direction)
                # 绘制箭头
                ax.arrow(arrow_pos[0], arrow_pos[1], 
                        arrow_direction_normalized[0] * arrow_length, 
                        arrow_direction_normalized[1] * arrow_length,
                        head_width=3, head_length=2, fc='red', ec='red', alpha=0.8)
            else:
                # 其他quad用蓝色
                ax.plot(quad_x, quad_y, 'b-', alpha=0.3, linewidth=0.5)
                centerline = road_network.quad_centerlines[i].cpu().numpy()
                ax.plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=1, alpha=0.5)   
        
        # 只显示一次图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # 计算Frenet坐标
        vehicle_pos_array = np.array([vehicle_pos], dtype=np.float32)
        vehicle_pos_tensor = torch.tensor(vehicle_pos_array, dtype=torch.float32, device=device)
        vehicle_yaw_tensor = torch.tensor([vehicle_yaw], dtype=torch.float32, device=device)
        d, theta_f = road_network.calculate_frenet_coordinates(vehicle_pos_tensor, vehicle_yaw_tensor)
        print(f"横向距离 d: {d.item():.2f} (正值表示在道路右侧，负值表示在道路左侧)")
        print(f"角度误差 theta_f: {theta_f.item():.2f} 弧度 ({np.degrees(theta_f.item()):.1f} 度)")
        print(f"角度误差解释: 正值表示车辆朝向偏右，负值表示偏左")

        # 设置图形属性
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('RoadNetwork Test - Map Visualization and Frenet Coordinate Calculation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
         
        # 保存图片
        plt.savefig('road_network_test.png', dpi=300, bbox_inches='tight')
        # 显示图形
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到地图文件 {map_path}")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

import torch
from typing import Dict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from road import RoadNetwork
from utils.spatial_hash import SpatialHash
from utils.geometry_utils import *
    
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
        # 这里的 config 是完整的 default_config.json 根配置
        # 提取 simulator.observation 段作为本模块的观测配置
        sim_cfg = config.get('simulator', {}) if isinstance(config, dict) else {}
        obs_cfg = sim_cfg.get('observation', {}) if isinstance(sim_cfg.get('observation', {}), dict) else {}
        self.config = obs_cfg
        self.device = device
        # 标量/数量配置（提供合理的缺省）
        self.num_neighbors = int(obs_cfg.get('num_neighbors'))           # 邻居数量
        self.num_w_lanes = int(obs_cfg.get('num_w_lanes'))               # 车道数量
        self.num_w_boundaries = int(obs_cfg.get('num_w_boundaries'))     # 边界数量
        self.horizon = float(obs_cfg.get('horizon'))                  # 视野范围
        # 观测维度配置（与 default_config.json 对齐）
        # local_state: x, y, yaw, speed, length, width, active
        self.local_state_dim = int(obs_cfg.get('local_state_dim'))
        self.neighbor_feature_dim = int(obs_cfg.get('neighbor_feature_dim'))
        self.w_lane_feature_dim = int(obs_cfg.get('w_lane_feature_dim'))
        self.boundary_feature_dim = int(obs_cfg.get('boundary_feature_dim'))
        # 额外：导航与链长度（供上游模块使用时读取）
        self.navigation_feature_dim = int(obs_cfg.get('navigation_feature_dim'))
        self.num_start_end_chains = int(obs_cfg.get('num_start&end_chains'))
        self.num_navigation_chains = int(obs_cfg.get('num_navigation_chains'))
        # 使用来自 SelfraceSimulator 的共享哈希，仅作网格坐标与单元ID计算，不在此处重建静态索引
        self.spatial_hash = spatial_hash
        # 预计算映射改由 RoadNetwork 在加载时完成，直接引用即可
        self.quad_to_w_lanes_ids = self.road_network.quad_to_w_lanes_ids
        self.quad_to_w_boundaries_ids = self.road_network.quad_to_w_boundaries_ids

    def get_observation_dim(self) -> int:
        """
        计算观测向量的总维度
        Returns:
            int: 观测向量的总维度
        """
        # 计算各部分维度
        local_state_size = self.local_state_dim  # 局部状态维度
        neighbors_size = self.num_neighbors * self.neighbor_feature_dim  # 邻居特征维度
        w_lanes_size = self.num_w_lanes * self.w_lane_feature_dim  # 车道航点维度
        w_boundaries_size = self.num_w_boundaries * self.boundary_feature_dim  # 边界点维度
        # 总维度
        total_dim = local_state_size + neighbors_size + w_lanes_size + w_boundaries_size
        return total_dim

    def _get_precomputed_waypoints(self, agents_state: torch.Tensor) -> tuple:
        """
        使用预计算的数据获取w_lanes和w_boundaries。
        Args:
            agents_state: 形状为 (B, M, 7) 的agent状态张量
        Returns:
            tuple: (w_lanes_world, w_boundaries_world)
        """
        batch_size, max_agents, _ = agents_state.shape
        # 获取每个agent所在的quad_id
        agent_positions = agents_state[..., :2]  # (B, M, 2)
        agent_positions_flat = agent_positions.view(-1, 2)  # (B*M, 2)
        # 找到每个agent最近的quad索引
        distances, quad_indices = find_nearest_lanes(self.device, self.road_network.quad_centerlines, agent_positions_flat, k=1, spatial_hash=self.spatial_hash)
        quad_indices = quad_indices.squeeze(-1)  # (B*M,)
        # 使用预计算的关联获取 waypoint 索引并直接索引坐标
        w_lanes_ids = self.quad_to_w_lanes_ids[quad_indices]      # (B*M, K_lanes)
        w_bounds_ids = self.quad_to_w_boundaries_ids[quad_indices]# (B*M, K_bounds)

        wl_world = self.road_network.global_w_lane_waypoints[w_lanes_ids]   # (B*M, wl_K, 2)
        wb_world = self.road_network.global_w_boundary_points[w_bounds_ids] # (B*M, wb_K, 2)

        # 若 K 与配置的目标数量不同，进行右侧零填充/裁剪
        def _pad_or_trim(t: torch.Tensor, target_k: int) -> torch.Tensor:
            N, K, D = t.shape if t.ndimension() == 3 else (t.shape[0], 0, 2)
            if K == target_k:
                return t
            if K == 0:
                return torch.zeros((N, target_k, D), device=self.device)
            if K > target_k:
                return t[:, :, :target_k]
            pad = torch.zeros((N, target_k - K, D), device=self.device)
            return torch.cat([t, pad], dim=1)
        wl_world = _pad_or_trim(wl_world, self.num_w_lanes)
        wb_world = _pad_or_trim(wb_world, self.num_w_boundaries)
        # 恢复原始形状
        w_lanes_world = wl_world.view(batch_size, max_agents, self.num_w_lanes, 2)
        w_boundaries_world = wb_world.view(batch_size, max_agents, self.num_w_boundaries, 2)
        return w_lanes_world, w_boundaries_world
    
    def _get_nearest_neighbors(self, agents_state: torch.Tensor) -> torch.Tensor:
        """为每个 agent 找到最近的 K 个邻居。完全向量化版本。"""
        batch_size, max_agents, _ = agents_state.shape
        # 如果不需要邻居，直接返回空的张量
        if self.num_neighbors == 0:
            return torch.zeros(batch_size, max_agents, 0, self.neighbor_feature_dim, device=self.device)
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
        # 4. 找到最近的 K 个
        _, topk_indices = torch.topk(dist_sq, k=self.num_neighbors, dim=-1, largest=False) # (B, M, K)
        # 5. 使用高级索引高效地收集邻居状态
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1)
        agent_idx = torch.arange(max_agents, device=self.device).view(1, max_agents, 1)
        neighbor_states = agents_state[batch_idx, topk_indices] # (B, M, K, 7)
        # 6. 如果邻居是无效的 (距离为inf)，则其状态需要被掩码；
        #    为避免后续局部坐标计算出现 dx,dy = -ego_pos 的伪值，
        #    将无效邻居的状态设置为等同于对应 ego 的状态（使相对量为0）。
        valid_neighbor_dists = dist_sq[batch_idx, agent_idx, topk_indices]
        is_valid_neighbor = torch.isfinite(valid_neighbor_dists) # (B, M, K)
        if (~is_valid_neighbor).any():
            K_neighbors = topk_indices.shape[-1]
            ego_states_expanded = agents_state.unsqueeze(2).expand(-1, -1, K_neighbors, -1)  # (B, M, K, 7)
            # 使无效邻居的相对位置/速度为0：复制ego的 [x,y,yaw,speed]
            # 同时将尺寸与active置零，避免下游看到伪造的车辆尺寸与激活标志
            replacement = ego_states_expanded.clone()
            replacement[..., 4] = 0.0  # length
            replacement[..., 5] = 0.0  # width
            replacement[..., 6] = 0.0  # active
            neighbor_states[~is_valid_neighbor] = replacement[~is_valid_neighbor]
        return neighbor_states
    
    def _world_to_ego_centric(self, ego_states, neighbor_states, w_lanes_world, w_boundaries_world):
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
        w_lanes_local = batch_rotate(w_lanes_world, ego_pos, rot_matrix)
        w_boundaries_local = batch_rotate(w_boundaries_world, ego_pos, rot_matrix)
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
            active_flag = neighbor_states[..., 6].unsqueeze(-1)
            neighbors_local = torch.cat([local_pos_neighbors, v_local, length, width, active_flag], dim=-1)
            #包含七个特征：dx, dy, dvx, dvy, length, width, active
        else:
            # 如果没有邻居，创建空的邻居特征张量
            neighbors_local = torch.zeros(B, M, 0, self.neighbor_feature_dim, device=self.device)
        # --- 创建每个 Agent 自身在局部坐标系下的状态 ---
        local_state = torch.zeros(B, M, self.local_state_dim, device=self.device) # 7个特征：x, y, yaw, speed, length, width, active
        local_state[..., 4] = ego_states[..., 4] # 长度
        local_state[..., 5] = ego_states[..., 5] # 宽度
        local_state[..., 6] = ego_states[..., 6] # 活跃状态
        return local_state, neighbors_local, w_lanes_local, w_boundaries_local

    def generate(self, agents_state: torch.Tensor) -> torch.Tensor:
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
            额外返回：d (B, M), theta_f (B, M)
            说明：local_state 仍为几何/占位等内部拼接所用，不再塞入 d/theta_f。
        """
        # 1. 获取世界坐标系下的特征
        # (B, M, K, 7)
        neighbor_states_world = self._get_nearest_neighbors(agents_state)
        # 使用预计算的数据获取w_lanes和w_boundaries
        w_lanes_world, w_boundaries_world = self._get_precomputed_waypoints(agents_state)
        # 2. 将所有信息转换到每个 Agent 的局部坐标系
        local_state, neighbors_local, w_lanes_local, w_boundaries_local = self._world_to_ego_centric(
            agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
        )
        
        # 2.1 计算每个 agent 的 Frenet 坐标 d 与 theta_f（单独返回，不写入 local_state）
        try:
            d, theta_f = calculate_frenet_coordinates(
                self.device,
                self.road_network.quad_directions,
                self.road_network.quad_centerlines,
                agents_state[..., :2],
                agents_state[..., 2],
                k=1,
                spatial_hash=self.spatial_hash
            )  # 均为 (B, M)
        except Exception:
            B, M, _ = agents_state.shape
            d = torch.zeros(B, M, device=self.device)
            theta_f = torch.zeros(B, M, device=self.device)
        
        # 3. 展平并拼接成最终的观测向量
        # 返回：自身绝对状态，邻居相对状态，车道线相对状态，边界线相对状态
        observation = torch.cat([
            local_state,
            neighbors_local.flatten(start_dim=2),
            w_lanes_local.flatten(start_dim=2),
            w_boundaries_local.flatten(start_dim=2)
        ], dim=2)
        return observation, d, theta_f

if __name__ == '__main__':
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from world_init import WorldInitializer
    from offroad import OffroadChecker
    from collision import CollisionChecker

    # 1) 读取完整配置
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(proj_root, 'configs', 'default_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        full_cfg = json.load(f)

    device = torch.device(full_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    sim_cfg = full_cfg.get('simulator', {})

    # 2) 地图与路网
    maps_dir = full_cfg.get('map_path', './maps')
    default_map = full_cfg.get('default_map', 'town2.json')
    map_file_path = os.path.join(proj_root, maps_dir, default_map)
    rn = RoadNetwork(map_path=map_file_path, device=device)

    # 3) 空间哈希（用于最近邻/离路检测）
    from utils.spatial_hash import SpatialHash
    all_verts = rn.quads_vertices.view(-1, 2)
    if all_verts.numel() > 0:
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
    else:
        min_bounds = torch.tensor([-1000.0, -1000.0], device=device)
        max_bounds = torch.tensor([1000.0, 1000.0], device=device)
    hash_cfg = sim_cfg.get('hash', {}) if isinstance(sim_cfg.get('hash', {}), dict) else {}
    cell_size = float(hash_cfg.get('cell_size', 20.0))
    spatial_hash = SpatialHash(cell_size=cell_size, min_bounds=min_bounds, max_bounds=max_bounds, device=device)

    # 4) 检查器与世界初始化器
    offroad_checker = OffroadChecker(rn, spatial_hash)
    collision_checker = CollisionChecker(full_cfg, spatial_hash)
    world_initializer = WorldInitializer(rn, offroad_checker, collision_checker, full_cfg)

    B = int(sim_cfg.get('B'))
    M = int(sim_cfg.get('M'))
    agents_state, ego_idx, agents_start_quad_ids = world_initializer.initialize_world(B, M)

    # 5) 构建 ObservationGenerator 并生成观测
    obs_gen = ObservationGenerator(rn, full_cfg, device, spatial_hash)
    observation, d, theta_f = obs_gen.generate(agents_state)

    # 6) 选择第一个环境中第一辆 active 车辆
    active_mask = agents_state[0, :, 6] > 0.5
    if active_mask.any():
        m_idx = int(torch.nonzero(active_mask, as_tuple=False)[0].item())
    else:
        m_idx = 0

    # 7) 使用预计算的世界坐标航点获取该车的观测内容（世界坐标）
    w_lanes_world, w_boundaries_world = obs_gen._get_precomputed_waypoints(agents_state)
    wl = w_lanes_world[0, m_idx].detach().cpu().numpy()  # (K,2)
    wb = w_boundaries_world[0, m_idx].detach().cpu().numpy()  # (K,2)

    # 打印 d 与 theta_f
    print(f"First active agent index in env 0: {m_idx}")
    print(f"d[0,{m_idx}] = {float(d[0, m_idx].item()):.6f}")
    print(f"theta_f[0,{m_idx}] = {float(theta_f[0, m_idx].item()):.6f}")

    # 8) 仅绘制 quads 与该车的观测点（wl/wb）
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Observation preview: quads + first active agent waypoints')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if rn.quads_vertices.numel() > 0:
        quads_np = rn.quads_vertices.detach().cpu().numpy()
        for verts in quads_np:
            poly = Polygon(verts, closed=True, facecolor='none', edgecolor='black', linewidth=0.2)
            ax.add_patch(poly)

        xs = quads_np[:, :, 0].reshape(-1)
        ys = quads_np[:, :, 1].reshape(-1)
        margin = 10.0
        ax.set_xlim(xs.min() - margin, xs.max() + margin)
        ax.set_ylim(ys.min() - margin, ys.max() + margin)

    # 绘制环境0的所有 active 车辆为矩形，并高亮第一辆 active 车
    def _draw_vehicle_rect(ax, x, y, yaw, length, width, facecolor, edgecolor='black', alpha=0.8, lw=1.0):
        cos_yaw = float(torch.cos(yaw).item()) if isinstance(yaw, torch.Tensor) else float(np.cos(yaw))
        sin_yaw = float(torch.sin(yaw).item()) if isinstance(yaw, torch.Tensor) else float(np.sin(yaw))
        half_l = float(length) / 2.0
        half_w = float(width) / 2.0
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])
        rot = np.array([[cos_yaw, -sin_yaw],[sin_yaw, cos_yaw]])
        rect = corners @ rot.T + np.array([float(x), float(y)])
        poly = Polygon(rect, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
        ax.add_patch(poly)

    active_mask0 = (agents_state[0, :, 6] > 0.5)
    active_indices0 = torch.nonzero(active_mask0, as_tuple=False).view(-1)
    for idx in active_indices0.tolist():
        x = agents_state[0, idx, 0].item()
        y = agents_state[0, idx, 1].item()
        yaw = agents_state[0, idx, 2]
        length = agents_state[0, idx, 4].item()
        width = agents_state[0, idx, 5].item()
        color = '#7fa6c9' if idx != m_idx else '#e74c3c'  # 普通车蓝灰，目标车红
        _draw_vehicle_rect(ax, x, y, yaw, length, width, facecolor=color, edgecolor='black', alpha=0.85, lw=1.0)

    # 观测到的最近 w_lane / w_boundary 世界点
    if wl.size > 0:
        ax.scatter(wl[:, 0], wl[:, 1], c='orange', s=10, alpha=0.8, label='w_lanes (agent0)')
    if wb.size > 0:
        ax.scatter(wb[:, 0], wb[:, 1], c='purple', s=8, alpha=0.6, label='w_boundaries (agent0)')

    # 9) 使用 observation 复原 neighbors_local（env0, agent=m_idx）的世界中心位置，并画红色虚线框
    neighbor_states_world = obs_gen._get_nearest_neighbors(agents_state)
    local_state_tmp, neighbors_local_tmp, _, _ = obs_gen._world_to_ego_centric(
        agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
    )
    ego_x = float(agents_state[0, m_idx, 0].item())
    ego_y = float(agents_state[0, m_idx, 1].item())
    ego_yaw = float(agents_state[0, m_idx, 2].item())
    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    nbr = neighbors_local_tmp[0, m_idx]  # (K,7): dx,dy,dvx,dvy,l,w,active
    if nbr.numel() > 0:
        dx_dy = nbr[:, 0:2].detach().cpu().numpy()  # (K,2)
        sizes = nbr[:, 4:6].detach().cpu().numpy()  # (K,2)
        act = (nbr[:, 6] > 0.5).detach().cpu().numpy()  # (K,)
        # 将局部坐标中心还原到世界坐标
        world_centers = (dx_dy @ rot.T) + np.array([ego_x, ego_y])
        # 仅基于 observation 重建邻居朝向：
        # v_neighbor_world = v_ego_world + R(ego_yaw) @ v_local
        v_local = nbr[:, 2:4].detach().cpu().numpy()  # (K,2) dvx,dvy in ego-local
        v_ego = float(agents_state[0, m_idx, 3].item())
        v_ego_world = np.array([v_ego * np.cos(ego_yaw), v_ego * np.sin(ego_yaw)])  # (2,)
        v_neighbor_world = (v_local @ rot.T) + v_ego_world  # (K,2)
        for k in range(world_centers.shape[0]):
            if not act[k]:
                continue
            cx, cy = world_centers[k, 0], world_centers[k, 1]
            length, width = float(sizes[k, 0]), float(sizes[k, 1])
            if length <= 0.0 or width <= 0.0:
                continue
            half_l = length / 2.0
            half_w = width / 2.0
            # 由速度方向近似邻居朝向；若速度过小则退化为使用自车朝向
            vx_k, vy_k = v_neighbor_world[k, 0], v_neighbor_world[k, 1]
            if (vx_k * vx_k + vy_k * vy_k) < 1e-6:
                yaw_k = ego_yaw
            else:
                yaw_k = float(np.arctan2(vy_k, vx_k))
            c, s = np.cos(yaw_k), np.sin(yaw_k)
            Rn = np.array([[c, -s], [s, c]])
            corners_local = np.array([
                [-half_l, -half_w],
                [ half_l, -half_w],
                [ half_l,  half_w],
                [-half_l,  half_w]
            ])
            rect = corners_local @ Rn.T + np.array([cx, cy])
            poly = Polygon(rect, closed=True, facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--')
            ax.add_patch(poly)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
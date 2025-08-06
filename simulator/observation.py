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
from road import RoadNetwork

class ObservationGenerator:
    """
    已经通过测试
    负责为批次中的所有 Agent 生成局部观测。
    通过完全向量化的操作，该模块可以一次性为所有环境中的所有智能体高效地计算观测，
    避免了在 Python 中进行循环，从而最大限度地利用 GPU 并行能力。
    """
    def __init__(self, road_network: RoadNetwork, config: Dict, device: torch.device):
        """
        初始化观测生成器。
        """
        self.road_network = road_network
        self.config = config
        self.device = device
        self.num_neighbors = config.get('num_neighbors', 1) # 邻居数量
        self.num_w_lanes = config.get('num_w_lanes', 25) # 车道数量
        self.num_w_boundaries = config.get('num_w_boundaries', 26) # 边界数量
        self.horizon = config.get('horizon', 100.0) # 视野范围
        # 定义观测空间维度
        self.local_state_dim = config.get('local_state_dim', 7)            # 修改为7个特征: x,y,yaw,speed,length,width,active       
        self.neighbor_feature_dim = config.get('neighbor_feature_dim', 7)  # 修改为7个特征：dx, dy, vx, vy, length, width, active 
        self.waypoint_feature_dim = config.get('waypoint_feature_dim', 2) 
        self.boundary_feature_dim = config.get('boundary_feature_dim', 2)  
        

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
            local_state: (B, M, 4)
            neighbors_local: (B, M, K, 7)  # dx, dy, vx, vy, length, width, active
            w_lanes_local: (B, M, N_lanes, 2)
            w_boundaries_local: (B, M, N_boundaries, 2)
        """
        batch_size, max_agents, _ = agents_state.shape
        # B,M

        # 1. 获取世界坐标系下的特征
        # (B, M, K, 7)
        neighbor_states_world = self._get_nearest_neighbors(agents_state)
        # (B, M, N_lanes, 2)
        w_lanes_world = self._get_nearby_global_points(agents_state, self.road_network.global_w_lane_waypoints, self.num_w_lanes)
        # (B, M, N_bounds, 2)
        w_boundaries_world = self._get_nearby_global_points(agents_state, self.road_network.global_w_boundary_points, self.num_w_boundaries)
        # 2. 将所有信息转换到每个 Agent 的局部坐标系
        local_state, neighbors_local, w_lanes_local, w_boundaries_local = self._world_to_ego_centric(
            agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
        )
        # 3. 展平并拼接成最终的观测向量
        observation = torch.cat([
            agents_state,
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
        # 4. 找到最近的 K 个
        _, topk_indices = torch.topk(dist_sq, k=self.num_neighbors, dim=-1, largest=False) # (B, M, K)
        # 5. 使用高级索引高效地收集邻居状态
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1)
        agent_idx = torch.arange(max_agents, device=self.device).view(1, max_agents, 1)
        neighbor_states = agents_state[batch_idx, topk_indices] # (B, M, K, 7)
        # 6. 如果邻居是无效的 (距离为inf)，则其状态需要被掩码/清零
        valid_neighbor_dists = dist_sq[batch_idx, agent_idx, topk_indices]
        is_valid_neighbor = torch.isfinite(valid_neighbor_dists) # (B, M, K)
        neighbor_states[~is_valid_neighbor] = 0.0
        return neighbor_states
    
    def _get_nearby_global_points(self, agents_state: torch.Tensor, source_points: torch.Tensor, num_points: int) -> torch.Tensor:
        """为所有 agent 从全局点集中找到 k 个最近的点。"""
        batch_size, max_agents, _ = agents_state.shape
        if source_points.numel() == 0 or num_points == 0:
            return torch.zeros(batch_size, max_agents, num_points, 2, device=self.device)

        query_pos = agents_state[..., :2].view(-1, 2) # (B*M, 2)
        dist_sq = torch.cdist(query_pos, source_points, p=2).pow(2)
        
        # 如果 num_points 为 0，直接返回空张量
        if num_points == 0:
            return torch.zeros(batch_size, max_agents, 0, 2, device=self.device)
            
        _, topk_indices = torch.topk(dist_sq, k=num_points, dim=1, largest=False) # (B*M, k)
        selected_points = source_points[topk_indices] # (B*M, k, 2)
        return selected_points.view(batch_size, max_agents, num_points, 2)
    
    def _world_to_ego_centric(self, ego_states, neighbor_states, w_lanes_world, w_boundaries_world):
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
        w_lanes_local = batch_rotate(w_lanes_world, ego_pos, rot_matrix)
        w_boundaries_local = batch_rotate(w_boundaries_world, ego_pos, rot_matrix)
        # --- 转换邻居 ---
        if K_neighbors > 0:
            rel_pos_neighbors = neighbor_states[..., :2] - ego_pos.unsqueeze(2) # (B, M, K, 2)
            local_pos_neighbors = torch.bmm(
                rel_pos_neighbors.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)
            ).view(B, M, K_neighbors, 2)
            neighbor_speed = neighbor_states[..., 3]
            neighbor_yaw = neighbor_states[..., 2]
            vx_world = neighbor_speed * torch.cos(neighbor_yaw)
            vy_world = neighbor_speed * torch.sin(neighbor_yaw)
            v_world = torch.stack([vx_world, vy_world], dim=-1) # (B, M, K, 2)
            v_local = torch.bmm(
                v_world.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)
            ).view(B, M, K_neighbors, 2)
            length = neighbor_states[..., 4].unsqueeze(-1)
            width = neighbor_states[..., 5].unsqueeze(-1)
            active_flag = neighbor_states[..., 6].unsqueeze(-1)
            neighbors_local = torch.cat([local_pos_neighbors, v_local, length, width, active_flag], dim=-1)
            #包含七个特征：dx, dy, vx, vy, length, width, active
        else:
            # 如果没有邻居，创建空的邻居特征张量
            neighbors_local = torch.zeros(B, M, 0, self.neighbor_feature_dim, device=self.device)
        # --- 创建每个 Agent 自身在局部坐标系下的状态 ---
        local_state = torch.zeros(B, M, self.local_state_dim, device=self.device) # 7个特征：x, y, yaw, speed, length, width, active
        local_state[..., 4] = ego_states[..., 4] # 长度
        local_state[..., 5] = ego_states[..., 5] # 宽度
        local_state[..., 6] = ego_states[..., 6] # 活跃状态
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
        # 随机选择一个quad并在其中生成车辆位置
        random_quad_idx = random.randint(0, road_network.num_quads - 1)
        print(f"随机选择quad索引: {random_quad_idx}")
        # 获取选中quad的顶点
        selected_quad = quads_vertices_np[random_quad_idx]
        # 在quad范围内随机生成车辆位置
        # 使用重心坐标法在quad内随机生成点
        def random_point_in_quad(quad_vertices):
            # 生成随机重心坐标
            r1, r2 = np.random.random(2)
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = r2 * sqrt_r1
            # 计算随机点
            point = (1-u-v) * quad_vertices[0] + u * quad_vertices[1] + v * quad_vertices[2]
            return point
        vehicle_pos = random_point_in_quad(selected_quad)
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
                second_vehicle_pos = random_point_in_quad(second_quad)
                second_vehicle_yaw = random.uniform(0, 2 * np.pi)  # 随机朝向
        # 创建agents_state (B=1, M=2, 7个特征)
        # agents_state[..., 0] = x, agents_state[..., 1] = y, agents_state[..., 2] = yaw
        # agents_state[..., 3] = speed, agents_state[..., 4] = vehicle_length, agents_state[..., 5] = vehicle_width, agents_state[..., 6] = active
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
            'waypoint_feature_dim': 2
        }
        # 创建ObservationGenerator实例
        observation_generator = ObservationGenerator(road_network, config, device)
        print(f"观测维度: {observation_generator.get_observation_dim()}")
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
                    # 2. 逆变换邻居速度：从局部坐标转换回全局坐标
                    neighbor_vel_local = np.array([vx_local, vy_local])
                    neighbor_vel_global = neighbor_vel_local @ rotation_matrix.T
                    # 3. 绘制邻居位置
                    ax.scatter(neighbor_pos_global[0], neighbor_pos_global[1], 
                             c='red', s=10, alpha=0.8, marker='o', label=f'Neighbor_{i}' if i == 0 else "")
                    # 4. 绘制邻居速度箭头（正交分解）
                    vel_x_arrow = neighbor_vel_global[0] 
                    vel_y_arrow = neighbor_vel_global[1] 
                    # X方向速度箭头（红色）
                    if abs(vel_x_arrow) > 0.1:  # 只绘制有意义的箭头
                        ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                                vel_x_arrow, 0, head_width=2, head_length=1, 
                                fc='red', ec='red', alpha=0.8, zorder=10)
                    # Y方向速度箭头（蓝色）
                    if abs(vel_y_arrow) > 0.1:  # 只绘制有意义的箭头
                        ax.arrow(neighbor_pos_global[0], neighbor_pos_global[1], 
                                0, vel_y_arrow, head_width=2, head_length=1, 
                                fc='green', ec='green', alpha=0.8, zorder=10)
                    
                    # 5. 绘制邻居车辆的矩形（使用复原的长度和宽度）
                    # 计算邻居的朝向（从速度向量推断）
                    if np.linalg.norm(neighbor_vel_global) > 0.1:
                        neighbor_yaw = np.arctan2(neighbor_vel_global[1], neighbor_vel_global[0])
                    else:
                        neighbor_yaw = 0.0  # 如果速度很小，假设朝向为0
                    
                    # 绘制邻居车辆矩形
                    draw_vehicle(ax, neighbor_pos_global[0], neighbor_pos_global[1], 
                               neighbor_yaw, np.linalg.norm(neighbor_vel_global), 
                               length, width, color='red', alpha=1)
                    
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
        print("计算Frenet坐标...")
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
        print("地图已保存为 road_network_test.png")
        # 显示图形
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到地图文件 {map_path}")
        print("请确保地图文件存在，或者修改map_path变量指向正确的地图文件")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


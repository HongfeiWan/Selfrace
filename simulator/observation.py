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
        self.local_state_dim = config.get('local_state_dim', 4)           # (vel_x, vel_y, acc_x, acc_y)
        self.neighbor_feature_dim = config.get('neighbor_feature_dim', 5) # (dx, dy, vx, vy, active)
        self.waypoint_feature_dim = config.get('waypoint_feature_dim', 2) # (dx, dy) for each point
        
    def generate(self, agents_state: torch.Tensor) -> torch.Tensor:
        """
        为所有环境中的所有 agent 生成一批观测。
        Args:
            agents_state (torch.Tensor): 全局状态张量 (B, M, 7)。
        Returns:
            torch.Tensor: 展平后的观测向量张量 (B, M, feature_dim)。
        """
        batch_size, max_agents, _ = agents_state.shape
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
        
        query_pos = agents_state[..., :2] # (B, M, 2)

        # 使用 torch.cdist 计算每个环境中所有 agent 之间的配对距离
        dist_sq = torch.cdist(query_pos, query_pos, p=2).pow(2) # (B, M, M)

        # 创建一个掩码来过滤掉不应被视为邻居的 agent
        # 1. Agent 不能是其自身的邻居 (对角线)
        self_mask = torch.eye(max_agents, device=self.device, dtype=torch.bool).expand(batch_size, -1, -1)
        # 2. 不活跃的 agent 不能作为邻居
        inactive_mask = (agents_state[..., 6] < 0.5).unsqueeze(1).expand(-1, max_agents, -1)
        
        dist_sq[self_mask | inactive_mask] = float('inf')
        dist_sq[dist_sq > self.horizon**2] = float('inf') # 距离超过视野范围的邻居不考虑

        # 找到最近的 K 个
        _, topk_indices = torch.topk(dist_sq, k=self.num_neighbors, dim=-1, largest=False) # (B, M, K)
        
        # 使用高级索引高效地收集邻居状态
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1)
        agent_idx = torch.arange(max_agents, device=self.device).view(1, max_agents, 1)
        neighbor_states = agents_state[batch_idx, topk_indices] # (B, M, K, 7)

        # 如果邻居是无效的 (距离为inf)，则其状态需要被掩码/清零
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
        N_lanes = w_lanes_world.shape[2]
        N_boundaries = w_boundaries_world.shape[2]

        ego_pos = ego_states[..., :2] # (B, M, 2)
        ego_yaw = ego_states[..., 2]  # (B, M)

        # 修正: 为行向量构建正确的旋转矩阵
        # 要将世界坐标点按 -ego_yaw 旋转，对于行向量 v' = v @ R,
        # 旋转矩阵 R 应为 [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]
        cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
        # (B, M, 2, 2)
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
            
            active_flag = neighbor_states[..., 6].unsqueeze(-1)
            neighbors_local = torch.cat([local_pos_neighbors, v_local, active_flag], dim=-1)
        else:
            # 如果没有邻居，创建空的邻居特征张量
            neighbors_local = torch.zeros(B, M, 0, self.neighbor_feature_dim, device=self.device)

        # --- 创建每个 Agent 自身在局部坐标系下的状态 ---
        local_state = torch.zeros(B, M, self.local_state_dim, device=self.device) # 151维
        local_state[..., 0] = ego_states[..., 3] # vx_local = speed 
        return local_state, neighbors_local, w_lanes_local, w_boundaries_local

if __name__ == '__main__':
    # --- 测试设置 ---
    import yaml
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_file_path = 'teraflow_replication/maps/carla_maps/processed_map_Town01_stitched.json'

    # 从配置文件读取配置
    config_path = 'teraflow_replication/configs/default_config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        #print(f"Loaded configuration from {config_path}")
        # 获取simulator配置中的observation部分
        if 'simulator' in config and 'observation' in config['simulator']:
            test_config = config['simulator']['observation']
        else:
            # 如果没有嵌套结构，直接使用根级别的observation配置
            test_config = config.get('observation', {})
        print(f"Using observation config:\n {test_config}")
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using default values")

    # 1. 实例化依赖项 RoadNetwork
    try:
        road_network = RoadNetwork(map_path=map_file_path, device=device)
    except FileNotFoundError:
        print("Error: Map file not found. Make sure the path is correct.")
        exit()

    # 2. 实例化 ObservationGenerator
    obs_generator = ObservationGenerator(road_network, test_config, device)
    #print("ObservationGenerator instantiated successfully.")

    # 3. 创建模拟的全局状态
    B, M = 1, int(obs_generator.num_neighbors)+1 # 1 envs, num_neighbors+1 agents
    agents_state = torch.zeros(B, M, 7, device=device)
    # 随机填充一些状态
    agents_state[:, :, :2] = torch.randn(B, M, 2, device=device) * 20 # positions
    agents_state[:, :, 2] = (torch.rand(B, M, device=device) * 2 - 1) * torch.pi # yaws
    agents_state[:, :, 3] = torch.rand(B, M, device=device) * 10 # speeds
    agents_state[:, :, 4:6] = torch.tensor([4.5, 2.0], device=device) # size
    agents_state[:, :, 6] = (torch.rand(B, M, device=device) > 0).float() # 100% active
    # 假设 ego 都是第一个 agent
    agents_state[:, 0, 6] = 1.0 #确保ego是active的

    # 4. 生成观测
    observation = obs_generator.generate(agents_state)

    # 5. 打印结果
    print(f"\n--- Observation Generation Results ---")
    print(f"Generated observation shape: {observation.shape}")

    # 计算期望的维度
    expected_dim = (test_config['local_state_dim'] + 
                    test_config['num_neighbors'] * test_config['neighbor_feature_dim'] +
                    test_config['num_w_lanes'] * test_config['waypoint_feature_dim'] +
                    test_config['num_w_boundaries'] * test_config['waypoint_feature_dim'])
    print(f"Expected feature dimension: {expected_dim}")
    assert observation.shape[2] == expected_dim, f"Observation dim mismatch! Got {observation.shape[2]}, expected {expected_dim}"
    print("Dimension check PASSED.")

    # 6. 可视化部分
    print("\n--- Starting Visualization ---")
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    # 绘制静态地图背景
    def draw_static_map():
        """绘制道路网络的静态背景。"""
        print("Drawing static map background...")
        # 获取并绘制车道边界
        left_bounds = road_network.get_all_lanes_left_boundaries().cpu().numpy()
        right_bounds = road_network.get_all_lanes_right_boundaries().cpu().numpy()
        # 绘制所有车道的边界线
        for line in np.concatenate([left_bounds, right_bounds]):
            ax.plot(line[:, 0], line[:, 1], color='white', linewidth=1.0, zorder=1)
        print("Map background drawn.")
    draw_static_map()
    
    # 获取车辆OBB顶点
    def get_world_vertices_from_state(states: torch.Tensor) -> torch.Tensor:
        """根据车辆状态计算OBB顶点。"""
        x, y, yaw = states[..., 0], states[..., 1], states[..., 2]
        length, width = states[..., 4], states[..., 5]

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        
        half_l, half_w = length / 2, width / 2
        
        corners_local = states.new_tensor([
            [1, 1], [1, -1], [-1, -1], [-1, 1]
        ], device=states.device) * torch.stack([half_l, half_w], dim=-1).unsqueeze(-2)
        # 旋转矩阵
        rot_matrix = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=-1).view(*states.shape[:-1], 2, 2)
        
        # 旋转并平移到世界坐标
        verts_world = corners_local @ rot_matrix
        verts_world += states[..., :2].unsqueeze(-2)
        return verts_world
    
    # 绘制车辆
    agents_state_env = agents_state[0]  # 取第一个环境
    ego_idx = 0  # ego是第一个agent
    
    # 获取ego状态（提前提取，避免在循环中重复计算）
    ego_state = agents_state_env[ego_idx]
    ego_x, ego_y, ego_yaw, *_ = ego_state.cpu().numpy()
    
    # 获取所有车辆的OBB顶点
    all_agent_vertices = get_world_vertices_from_state(agents_state_env)
    
    # 绘制每个活跃的车辆
    for i, agent_state in enumerate(agents_state_env):
        if agent_state[6] > 0.5:  # 如果agent是活跃的
            is_ego = (i == ego_idx)
            color = 'royalblue' if is_ego else 'crimson'
            zorder = 10 if is_ego else 5
            
            # 绘制车辆OBB（使用Rectangle，与test_and_visualize_simulator.py保持一致）
            _, _, _, _, length, width, _ = agent_state.cpu().numpy()
            x, y, yaw = agent_state[:3].cpu().numpy()
            
            # 创建Rectangle patch
            rect = patches.Rectangle(
                (-length / 2, -width / 2), 
                length, 
                width, 
                facecolor=color, 
                edgecolor='black',
                linewidth=1.0,
                zorder=zorder
            )
            
            # 应用旋转和平移变换
            rotation = plt.matplotlib.transforms.Affine2D().rotate_deg(np.degrees(yaw))
            translation = plt.matplotlib.transforms.Affine2D().translate(x, y)
            final_transform = rotation + translation + ax.transData
            rect.set_transform(final_transform)
            ax.add_patch(rect)
            
            # 添加车辆标签
            agent_yaw = agent_state[2].item()
            
            if is_ego:
                # ego车辆显示基本信息
                label_text = f'ID:{i}\nSpeed:{agent_state[3]:.1f}m/s'
            else:
                # 非ego车辆显示转换到ego坐标系的信息
                # 计算相对位置
                rel_x = x - ego_x
                rel_y = y - ego_y
                # 转换到ego坐标系（旋转）
                cos_yaw = np.cos(-ego_yaw)  # 注意是负角度，因为要转换到ego坐标系
                sin_yaw = np.sin(-ego_yaw)
                ego_rel_x = rel_x * cos_yaw - rel_y * sin_yaw
                ego_rel_y = rel_x * sin_yaw + rel_y * cos_yaw
                # 计算相对速度
                agent_speed = agent_state[3].item()
                agent_yaw = agent_state[2].item()
                
                # 世界坐标系下的速度分量
                agent_vx_world = agent_speed * np.cos(agent_yaw)
                agent_vy_world = agent_speed * np.sin(agent_yaw)
                
                # 转换到ego坐标系的速度
                ego_rel_vx = agent_vx_world * cos_yaw - agent_vy_world * sin_yaw
                ego_rel_vy = agent_vx_world * sin_yaw + agent_vy_world * cos_yaw
                
                # 翻转速度的Y分量以匹配matplotlib坐标系
                ego_rel_vy = -ego_rel_vy
                
                # 计算距离
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                label_text = f'ID:{i}\nPos:({ego_rel_x:.1f},{ego_rel_y:.1f})\nVel:({ego_rel_vx:.1f},{ego_rel_vy:.1f})\nDist:{distance:.1f}m'
            
            ax.text(x, y, label_text, fontsize=7, ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 可视化观测内容
    ego_obs = observation[0, ego_idx]  # ego的观测，B=0，M=ego_idx,151维
    # 解析观测向量的各个部分
    local_state_dim = test_config['local_state_dim']
    neighbor_feature_dim = test_config['neighbor_feature_dim']
    num_neighbors = test_config['num_neighbors']
    waypoint_feature_dim = test_config.get('waypoint_feature_dim', 2)
    num_w_lanes = test_config['num_w_lanes']
    num_w_boundaries = test_config['num_w_boundaries']
    
    # 计算切片索引
    start_w_lane = local_state_dim + num_neighbors * neighbor_feature_dim
    end_w_lane = start_w_lane + num_w_lanes * waypoint_feature_dim
    start_w_boundary = end_w_lane
    end_w_boundary = start_w_boundary + num_w_boundaries * waypoint_feature_dim
    
    # 提取局部坐标特征
    w_lane_local_feats = ego_obs[start_w_lane:end_w_lane].view(-1, waypoint_feature_dim).cpu().numpy()
    w_boundary_local_feats = ego_obs[start_w_boundary:end_w_boundary].view(-1, waypoint_feature_dim).cpu().numpy()

    # 过滤掉填充值 (通常是全0)
    w_lane_local_feats = w_lane_local_feats[np.any(w_lane_local_feats != 0, axis=1)]
    w_boundary_local_feats = w_boundary_local_feats[np.any(w_boundary_local_feats != 0, axis=1)]

    # 转换到全局坐标
    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    
    ego_pos_global = np.array([ego_x, ego_y])
    w_lane_obs_global = (w_lane_local_feats @ rotation_matrix.T) + ego_pos_global
    w_boundary_obs_global = (w_boundary_local_feats @ rotation_matrix.T) + ego_pos_global
    
    # 绘制观测点
    if w_lane_obs_global.shape[0] > 0:
        ax.scatter(w_lane_obs_global[:, 0], w_lane_obs_global[:, 1], 
                  c='lime', marker='+', s=60, zorder=14, label='Obs W_Lanes')
    
    if w_boundary_obs_global.shape[0] > 0:
        ax.scatter(w_boundary_obs_global[:, 0], w_boundary_obs_global[:, 1], 
                  s=40, facecolors='none', edgecolors='cyan', zorder=13, label='Obs W_Boundaries')
    
    # 绘制ego位置标记
    ax.scatter([ego_x], [ego_y], c='yellow', marker='X', s=100, zorder=15, label='Ego Vehicle')
    
    # 设置坐标轴范围 - 显示整个地图
    # 获取地图的边界范围
    all_verts = road_network.quads_vertices.view(-1, 2)
    min_bounds, _ = torch.min(all_verts, dim=0)
    max_bounds, _ = torch.max(all_verts, dim=0)
    
    # 添加一些边距
    margin = 20.0
    ax.set_xlim(min_bounds[0].item() - margin, max_bounds[0].item() + margin)
    ax.set_ylim(min_bounds[1].item() - margin, max_bounds[1].item() + margin)
    
    # 添加图例和标题
    ax.legend(loc='upper right')
    #ax.set_title('Observation Visualization - Full Map View\nBlue: Ego Vehicle, Red: Other Vehicles\nGreen +: Lane Waypoints, Cyan O: Boundary Waypoints')
    
    # 添加信息文本
    info_text = (
        f"Ego Speed: {ego_state[3]:.2f} m/s\n"
        f"Ego Yaw: {np.degrees(ego_yaw):.1f}°\n"
        f"Ego Position: ({ego_x:.1f}, {ego_y:.1f})\n"
        f"Active Vehicles: {(agents_state_env[:, 6] > 0.5).sum().item()}\n"
        f"Lane Waypoints: {w_lane_obs_global.shape[0]}\n"
        f"Boundary Waypoints: {w_boundary_obs_global.shape[0]}\n"
        f"Map Bounds: ({min_bounds[0]:.0f}, {min_bounds[1]:.0f}) to ({max_bounds[0]:.0f}, {max_bounds[1]:.0f})"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            ha='left', va='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 创建以ego为中心的坐标系可视化
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.set_aspect('equal')
    ax2.set_facecolor('lightgray')
    ax2.grid(True, alpha=0.3)
    # 绘制ego坐标系原点
    ax2.scatter([0], [0], c='yellow', marker='X', s=150, zorder=20, label='Ego (Origin)')
    
    # 绘制ego坐标轴
    axis_length = 5.0
    # X轴（ego前方）
    ax2.arrow(0, 0, axis_length, 0, head_width=2, head_length=3, fc='red', ec='red', zorder=15, label='Ego X-axis')
    ax2.text(axis_length + 2, 0, 'X (Ego Forward)', fontsize=10, ha='left', va='center', color='red', weight='bold')
    # Y轴（ego右侧，向下为正以匹配matplotlib）
    ax2.arrow(0, 0, 0, axis_length, head_width=2, head_length=3, fc='green', ec='green', zorder=15, label='Ego Y-axis')
    ax2.text(0, axis_length, 'Y (Ego Right)', fontsize=10, ha='center', va='top', color='green', weight='bold')
    
    # 添加坐标系说明
    ax2.text(0.02, 0.02, 'ego coordinate system:\n• X-axis(red): ego forward direction\n• Y-axis(green): ego right direction (down=positive)\n• Origin(yellow X): ego current position', 
             transform=ax2.transAxes, ha='left', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 转换地图信息到ego坐标系
    def draw_map_in_ego_coords():
        """在ego坐标系中绘制地图信息。"""
        print("Drawing map in ego coordinates...")
        
        # 获取车道边界
        left_bounds = road_network.get_all_lanes_left_boundaries().cpu().numpy()
        right_bounds = road_network.get_all_lanes_right_boundaries().cpu().numpy()
        
        # 合并所有边界线
        all_boundaries = np.concatenate([left_bounds, right_bounds])
        
        # 转换每条边界线到ego坐标系
        for boundary_line in all_boundaries:
            if len(boundary_line) > 0:
                # 计算相对位置
                rel_positions = boundary_line - np.array([ego_x, ego_y])
                
                # 旋转到ego坐标系
                cos_yaw = np.cos(-ego_yaw)
                sin_yaw = np.sin(-ego_yaw)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                
                ego_coord_positions = rel_positions @ rotation_matrix.T
                
                # 翻转Y轴以匹配matplotlib坐标系（向下为正）
                ego_coord_positions[:, 1] = -ego_coord_positions[:, 1]
                
                # 只绘制在显示范围内的线段
                mask = (ego_coord_positions[:, 0] >= -100) & (ego_coord_positions[:, 0] <= 100) & \
                       (ego_coord_positions[:, 1] >= -100) & (ego_coord_positions[:, 1] <= 100)
                
                if np.any(mask):
                    # 找到连续的线段段
                    segments = []
                    start_idx = None
                    for i, is_visible in enumerate(mask):
                        if is_visible and start_idx is None:
                            start_idx = i
                        elif not is_visible and start_idx is not None:
                            segments.append((start_idx, i))
                            start_idx = None
                    
                    # 处理最后一个段
                    if start_idx is not None:
                        segments.append((start_idx, len(mask)))
                    
                    # 绘制每个可见段
                    for start, end in segments:
                        if end - start > 1:  # 至少需要2个点才能画线
                            segment_coords = ego_coord_positions[start:end]
                            ax2.plot(segment_coords[:, 0], segment_coords[:, 1], 
                                   color='white', linewidth=1.0, zorder=1, alpha=0.7)
        
        print("Map drawn in ego coordinates.")
    
    # 绘制ego坐标系中的地图
    draw_map_in_ego_coords()
    
    # 转换所有车辆到ego坐标系
    for i, agent_state in enumerate(agents_state_env):
        if agent_state[6] > 0.5:  # 如果agent是活跃的
            is_ego = (i == ego_idx)
            
            if is_ego:
                # ego在原点
                ego_coord_x, ego_coord_y = 0, 0
                color = 'royalblue'
                zorder = 10
                # 绘制ego车辆（在原点）
                ego_length, ego_width = agent_state[4:6].cpu().numpy()
                ego_rect = patches.Rectangle((-ego_length/2, -ego_width/2), ego_length, ego_width, 
                                           facecolor=color, edgecolor='black', linewidth=1.0, alpha=0.8, zorder=zorder)
                ax2.add_patch(ego_rect)
                
                # 绘制ego的速度向量（朝向ego的前进方向）
                ego_speed = agent_state[3].item()
                # ego在ego坐标系中，其前进方向就是X轴正方向
                ego_vx = ego_speed  # ego的速度在ego坐标系中就是沿X轴正方向
                ego_vy = 0.0
                
                # 绘制速度向量（从车辆中心开始，朝向ego的前进方向）
                velocity_scale = 1.0  # 速度向量的缩放因子
                ax2.arrow(0, 0, ego_vx * velocity_scale, ego_vy * velocity_scale, 
                         head_width=1.5, head_length=2, fc='blue', ec='blue', 
                         linewidth=2, zorder=zorder+1, alpha=0.8)
                # 添加速度向量标签
                ax2.text(ego_vx * velocity_scale + 2, ego_vy * velocity_scale, 
                        f'Forward Speed: {ego_speed:.1f}m/s', fontsize=8, ha='left', va='center', 
                        color='blue', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                
                ax2.text(0, 0, f'ID:{i}\nSpeed:{agent_state[3]:.1f}m/s', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                # 转换其他车辆到ego坐标系
                x, y = agent_state[:2].cpu().numpy()
                rel_x = x - ego_x
                rel_y = y - ego_y
                
                # 旋转到ego坐标系
                cos_yaw = np.cos(-ego_yaw)
                sin_yaw = np.sin(-ego_yaw)
                ego_coord_x = rel_x * cos_yaw - rel_y * sin_yaw
                ego_coord_y = rel_x * sin_yaw + rel_y * cos_yaw
                
                # 翻转Y轴以匹配matplotlib坐标系（向下为正）
                # 这样ego右侧的物体在figure2中会显示在Y轴正方向（朝下）
                ego_coord_y = -ego_coord_y
                
                # 计算车辆在ego坐标系中的朝向
                agent_yaw = agent_state[2].item()
                ego_rel_yaw = agent_yaw - ego_yaw
                # 由于figure2中Y轴向下为正，需要翻转yaw角度
                ego_rel_yaw = -ego_rel_yaw
                
                # 绘制车辆（在ego坐标系中）
                color = 'crimson'
                zorder = 5
                length, width = agent_state[4:6].cpu().numpy()
                
                # 创建旋转的矩形
                rect = patches.Rectangle((-length/2, -width/2), length, width, 
                                       facecolor=color, edgecolor='black', linewidth=1.0, alpha=0.8, zorder=zorder)
                
                # 应用旋转和平移变换（注意ego_coord_y已经翻转过了）
                rotation = plt.matplotlib.transforms.Affine2D().rotate(ego_rel_yaw)
                translation = plt.matplotlib.transforms.Affine2D().translate(ego_coord_x, ego_coord_y)
                final_transform = rotation + translation + ax2.transData
                rect.set_transform(final_transform)
                ax2.add_patch(rect)
                
                # 计算相对速度
                agent_speed = agent_state[3].item()
                agent_vx_world = agent_speed * np.cos(agent_yaw)
                agent_vy_world = agent_speed * np.sin(agent_yaw)
                ego_rel_vx = agent_vx_world * cos_yaw - agent_vy_world * sin_yaw
                ego_rel_vy = agent_vx_world * sin_yaw + agent_vy_world * cos_yaw
                
                # 翻转速度的Y分量以匹配matplotlib坐标系
                ego_rel_vy = -ego_rel_vy
                
                # 绘制邻居车辆的速度向量（朝向邻居自己的前进方向）
                agent_speed = agent_state[3].item()
                # 使用已经计算好的ego_rel_yaw（已经考虑了Y轴翻转）
                
                # 邻居车辆在ego坐标系中的速度向量（朝向邻居自己的前进方向）
                # 模仿ego的方式：速度向量沿着车辆自己的前进方向
                # 在ego坐标系中，邻居车辆的前进方向就是ego_rel_yaw
                # 注意：ego_rel_yaw已经考虑了Y轴翻转，所以这里不需要再次翻转Y分量
                neighbor_vx = agent_speed * np.cos(ego_rel_yaw)
                neighbor_vy = agent_speed * np.sin(ego_rel_yaw)
                
                velocity_scale = 1.0  # 速度向量的缩放因子
                ax2.arrow(ego_coord_x, ego_coord_y, neighbor_vx * velocity_scale, neighbor_vy * velocity_scale, 
                         head_width=1.5, head_length=2, fc='red', ec='red', 
                         linewidth=2, zorder=zorder+1, alpha=0.8)
                # 添加速度向量标签
                ax2.text(ego_coord_x + neighbor_vx * velocity_scale + 2, ego_coord_y + neighbor_vy * velocity_scale, 
                        f'Forward Speed: {agent_speed:.1f}m/s', fontsize=8, ha='left', va='center', 
                        color='red', weight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
                
                # 计算距离
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # 添加标签
                label_text = f'ID:{i}\nPos:({ego_coord_x:.1f},{ego_coord_y:.1f})\nVel:({ego_rel_vx:.1f},{ego_rel_vy:.1f})\nDist:{distance:.1f}m'
                ax2.text(ego_coord_x, ego_coord_y, label_text, fontsize=7, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 转换观测点到ego坐标系
    if w_lane_obs_global.shape[0] > 0:
        # 观测点已经在ego坐标系中，翻转Y轴以匹配matplotlib坐标系
        w_lane_ego_coords = w_lane_local_feats.copy()
        w_lane_ego_coords[:, 1] = -w_lane_ego_coords[:, 1]
        ax2.scatter(w_lane_ego_coords[:, 0], w_lane_ego_coords[:, 1], 
                   c='lime', marker='+', s=60, zorder=14, label='Obs W_Lanes (Ego Coords)')
    
    if w_boundary_obs_global.shape[0] > 0:
        # 观测点已经在ego坐标系中，翻转Y轴以匹配matplotlib坐标系
        w_boundary_ego_coords = w_boundary_local_feats.copy()
        w_boundary_ego_coords[:, 1] = -w_boundary_ego_coords[:, 1]
        ax2.scatter(w_boundary_ego_coords[:, 0], w_boundary_ego_coords[:, 1], 
                   s=40, facecolors='none', edgecolors='cyan', zorder=13, label='Obs W_Boundaries (Ego Coords)')
    
    # 设置ego坐标系的显示范围
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(100, -100)  # 反转Y轴，让向下为正数
    ax2.set_xlabel('Ego X-axis (Forward)')
    ax2.set_ylabel('Ego Y-axis (Right)')
    #ax2.set_title('Observation Visualization - Ego Centered Coordinates\nBlue: Ego Vehicle, Red: Other Vehicles\nBlue/Red Arrows: Velocity Vectors\nGreen +: Lane Waypoints, Cyan O: Boundary Waypoints\nWhite: Road Boundaries')
    
    # 添加ego坐标系信息
    ego_info_text = (
        f"Ego Speed: {ego_state[3]:.2f} m/s\n"
        f"Ego Yaw: {np.degrees(ego_yaw):.1f}°\n"
        f"Active Vehicles: {(agents_state_env[:, 6] > 0.5).sum().item()}\n"
        f"Lane Waypoints: {w_lane_local_feats.shape[0]}\n"
        f"Boundary Waypoints: {w_boundary_local_feats.shape[0]}\n"
        f"Coordinate System: Ego-centered"
    )
    ax2.text(0.02, 0.98, ego_info_text, transform=ax2.transAxes, 
             ha='left', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization completed!")


import torch
import os
import sys
import math
import numpy as np
import vispy
from vispy import scene, app
from vispy.scene import visuals
from vispy.visuals import LineVisual
from typing import Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.spatial_hash import SpatialHash
from road import RoadNetwork
from offroad import OffroadChecker
from collision import CollisionChecker
from goal import PathPlanner
from reward import RewardCalculator
from world_init import WorldInitializer
from observation import ObservationGenerator
from dynamics import KinematicBicycleModel

class selfrace:
    """
    selfrace 类，用于模拟自博弈环境
    核心模拟器,可以通过不同显卡导入不同的config以做不同地图的模拟
    """
    def __init__(self, config:Dict, device: torch.device):
        self.config = config
        self.device = device

        simulator_config = config.get('simulator')
        self.num_envs = simulator_config.get('B')
        self.max_agents = simulator_config.get('M')
        self.dt = simulator_config['sim_dt']

        maps_dir = config.get('map_path', './maps')
        default_map = config.get('default_map')
        self.map_path = os.path.join(os.path.dirname(__file__), '..', maps_dir, default_map)

        # 1. 加载地图网络
        # road.py 中的 RoadNetwork 类负责解析地图文件并提供查询接口
        self.road_network = RoadNetwork(self.map_path, self.device)

        # 2. 初始化共享的空间哈希
        all_verts = self.road_network.quads_vertices.view(-1, 2)
        min_bounds, _ = torch.min(all_verts, dim=0)
        max_bounds, _ = torch.max(all_verts, dim=0)
        # 使用一个固定的 cell_size, 也可以从 config 读取
        hash_config = simulator_config['hash']
        cell_size = hash_config.get('cell_size')
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)

        # 3. 初始化离路检测器 (传入共享的哈希对象)
        self.offroad_checker = OffroadChecker(self.road_network, self.spatial_hash)

        # 4. 初始化碰撞检测器 (传入共享的哈希对象)
        # collision.py 负责检测智能体之间的碰撞
        self.collision_checker = CollisionChecker(config, self.spatial_hash)

        # 5. 初始化世界状态管理器（为动力学提供车辆初始化参数）
        # world_init.py 负责在 reset 时初始化场景
        self.world_initializer = WorldInitializer(self.road_network, self.offroad_checker, self.collision_checker, config)
        
        # 6. 初始化车辆动力学模型（依赖 world_initializer 的车辆初始化参数）
        # dynamics.py 中的 KinematicBicycleModel 负责根据动作更新车辆状态
        self.dynamics_model = KinematicBicycleModel(config, self.device)

        # 7. 初始化观测生成器
        # observation.py 负责为每个自车生成局部观测
        self.observation_generator = ObservationGenerator(self.road_network, config, self.device, self.spatial_hash)

        # 8. 初始化奖励计算器
        self.reward_calculator = RewardCalculator(self.config, self.device)

        # 9. 初始化路径规划器（直接复用 RoadNetwork 数据）
        self.path_planner = PathPlanner(self.device, self.road_network)

        # 10. 初始化模拟世界的状态张量
        # 这些张量将在 reset() 中被具体填充
        self.agents_state: Optional[torch.Tensor] = None
        self.agents_start_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的起始quad_id
        self.agents_goal_quad_ids: Optional[torch.Tensor] = None  # 存储所有智能体的目标quad_id

        self.agents_path_plans: Optional[torch.Tensor] = None     # 存储所有智能体的路径规划 w_lane_id 序列
        self.agents_path_plans_world: Optional[torch.Tensor] = None  # 存储所有智能体的路径规划（世界坐标 xy）(B, M, L, 2)

        self.goal_radius_tensor: Optional[torch.Tensor] = None
        self.goal_positions: Optional[torch.Tensor] = None

        self.w_lanes_local_with_goal_distances: Optional[torch.Tensor] = None
        self.w_lane_goal_distances_full: Optional[torch.Tensor] = None
        self.sampled_waypoint_ids: Optional[torch.Tensor] = None

        # 可视化相关
        self.canvas = None
        self.view = None
        self.line_visual = None  # 保存 line visual 引用以便更新
        self.agent_visuals = []  # 保存 agent visual 引用列表
        self.path_visuals = []  # 保存路径 visual 引用列表
        self.goal_quad_visual = None  # 保存目标quad visual 引用
        self.observation_w_lanes_visual = None  # 保存观测w_lanes visual 引用
        self.observation_w_boundaries_visual = None  # 保存观测w_boundaries visual 引用
        
        # 信息显示窗口相关
        self.info_canvas = None  # 信息显示窗口
        self.info_view = None
        self.info_text_visuals = []  # 保存文本 visual 引用列表

        self.reset()
    
    def reset(self):
        """重置模拟世界"""
         # 必须先reset world。产生不同大小的车
        self.agents_state, self.agents_start_quad_ids, self.agents_goal_quad_ids = self.world_initializer.initialize_world(self.num_envs, self.max_agents)
        # 重置动力学模型：传入新一批车辆参数并重置内部控制状态与风格参数
        self.dynamics_model.reset(self.world_initializer.vehicle_params)
        # 重置reward风格参数
        self.reward_calculator.reset_episode()
        # 将状态数据移动到正确的设备
        self.agents_state = self.agents_state.to(self.device)
        # 重置累积done状态
        self.cumulative_done_mask = None
        self.agents_start_quad_ids = self.agents_start_quad_ids.to(self.device)
        self.agents_goal_quad_ids = self.agents_goal_quad_ids.to(self.device)
        paths = self.path_planner.path_plan(self.agents_start_quad_ids, self.agents_goal_quad_ids)
        self.agents_path_plans = self.path_planner.collect_path_w_lane_ids(paths, self.agents_start_quad_ids, self.agents_goal_quad_ids)

        # 生成初始观测
        print("Generating initial observation...") 
        initial_observation, d, theta_f = self.observation_generator.generate(self.agents_state)
        print("Initial observation generated")
        self.frenet_d = d
        self.frenet_theta_f = theta_f
        self.initial_observation = initial_observation  # 保存初始观测用于可视化

    def render(self):
        """使用 vispy OpenGL 加速可视化 road_network 中的 quads_vertices"""
        # 获取 quads_vertices: (N, 4, 2) - 保持在 GPU 上
        def _render_quads():
            quads_vertices = self.road_network.quads_vertices
            if quads_vertices.numel() == 0:
                return
            num_quads = quads_vertices.shape[0]
            # 在 GPU 上构建闭合的顶点数据（每个四边形需要5个点：4个顶点+1个闭合点）
            # 使用 tensor 操作在 GPU 上完成，只在最后需要时转换
            with torch.no_grad():
                # 添加 z=0 维度以符合 OpenGL 3D 坐标要求
                # quads_vertices: (N, 4, 2) -> (N, 4, 3) with z=0
                quads_3d = torch.zeros((num_quads, 4, 3), dtype=torch.float32, device=quads_vertices.device)
                quads_3d[:, :, :2] = quads_vertices
                
                # 为每个四边形添加闭合点（第一个顶点复制到最后）
                closed_quads = torch.cat([quads_3d, quads_3d[:, 0:1, :]], dim=1)  # (N, 5, 3)
                
                # 批量构建所有顶点：将所有四边形连接成一个长序列（用于 line_strip）
                # 但需要在四边形之间插入断开标记（使用 NaN 或使用 segments）
                # 更好的方法：构建 segments，每个四边形是独立的线段段
                # 构建所有线段段的起始和结束点
                all_segments = []
                for i in range(num_quads):
                    quad = closed_quads[i]  # (5, 3)
                    # 为每条边创建线段：0->1, 1->2, 2->3, 3->4(即0)
                    for j in range(4):
                        all_segments.append(quad[j:j+2, :2])  # 只取 x,y，vispy Line 用 2D
                
                # 连接所有线段段
                if all_segments:
                    all_lines_tensor = torch.cat(all_segments, dim=0)  # (N*4*2, 2)
                    # 只在渲染时转换为 numpy（延迟转换）
                    all_lines_np = all_lines_tensor.detach().cpu().numpy()
                else:
                    return
            
            # 计算边界（在 GPU 上计算后转换）
            with torch.no_grad():
                all_vertices_2d = quads_vertices.reshape(-1, 2)  # (N*4, 2)
                x_min, y_min = all_vertices_2d.min(dim=0)[0].cpu().numpy()
                x_max, y_max = all_vertices_2d.max(dim=0)[0].cpu().numpy()
            
            # 创建 canvas 和 view（使用 OpenGL 后端）
            if self.canvas is None:
                # 确保使用 OpenGL 后端（让 vispy 自动选择可用的后端）
                try:
                    vispy.use(app='glfw')  # 优先尝试 glfw 后端
                except Exception:
                    try:
                        vispy.use(app='pyqt5')  # 如果 glfw 不可用，尝试 pyqt5
                    except Exception:
                        pass  # 使用默认后端

                self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(800, 800))
                self.view = self.canvas.central_widget.add_view()
                self.view.camera = 'panzoom'
            
            # 设置相机范围
            margin_x = float((x_max - x_min) * 0.1)
            margin_y = float((y_max - y_min) * 0.1)
            self.view.camera.set_range(x=(float(x_min) - margin_x, float(x_max) + margin_x),
                                    y=(float(y_min) - margin_y, float(y_max) + margin_y))
            
            # 更新或创建 Line visual
            # 如果已存在，更新数据；否则创建新的
            if self.line_visual is None:
                # 创建新的 Line visual
                self.line_visual = visuals.Line(
                    pos=all_lines_np,
                    color=(0.3, 0.3, 0.3, 0.8),  # 深灰色
                    width=1.0,
                    connect='segments',  # 每两个点构成一条线段
                    method='gl'  # 使用 OpenGL 渲染
                )
                self.view.add(self.line_visual)
            else:
                # 更新现有 visual 的数据
                self.line_visual.set_data(pos=all_lines_np)

        def _find_first_active_agent():
            """找到第一个 active 的 agent，返回 (env_idx, agent_idx)"""
            if self.agents_state is None or self.agents_state.numel() == 0:
                return None, None
            B, M = self.agents_state.shape[:2]
            # 遍历所有环境，找到第一个有 active agent 的环境
            for env_idx in range(B):
                env_states = self.agents_state[env_idx]  # (M, 7)
                active_mask = env_states[:, 6] == 1.0
                if active_mask.any():
                    # 找到第一个 active 的 agent
                    agent_idx = torch.nonzero(active_mask, as_tuple=False)[0].item()
                    return env_idx, agent_idx
            return None, None
        
        def _render_agents(env_idx=0, agent_idx=0):
            """绘制指定环境的指定 agent 状态到地图上"""
            if self.agents_state is None or self.agents_state.numel() == 0:
                return
            if env_idx >= self.agents_state.shape[0]:
                return
            env_states = self.agents_state[env_idx]  # (M, 7)
            if agent_idx >= env_states.shape[0] or env_states[agent_idx, 6] != 1.0:
                return
            
            # 只取指定的 agent
            active_states = env_states[agent_idx:agent_idx+1]  # (1, 7)
            
            # 在 GPU 上计算车辆矩形顶点
            with torch.no_grad():
                # 提取状态信息：[x, y, yaw, v, length, width, active]
                positions = active_states[:, :2]  # (N_active, 2)
                yaws = active_states[:, 2]  # (N_active,)
                lengths = active_states[:, 4]  # (N_active,)
                widths = active_states[:, 5]  # (N_active,)
                
                n_active = positions.shape[0]
                
                # 计算车辆矩形的四个角点（相对于中心）
                cos_yaw = torch.cos(yaws)  # (N_active,)
                sin_yaw = torch.sin(yaws)  # (N_active,)
                
                half_length = lengths / 2.0  # (N_active,)
                half_width = widths / 2.0   # (N_active,)
                
                # 车辆矩形的四个角点（局部坐标系）
                # 顺序：左下, 右下, 右上, 左上
                # 使用 torch.stack 来构建 (N_active, 4, 2) 形状
                corners_local = torch.stack([
                    torch.stack([-half_length, -half_width], dim=1),  # 左下 (N_active, 2)
                    torch.stack([half_length, -half_width], dim=1),   # 右下 (N_active, 2)
                    torch.stack([half_length, half_width], dim=1),    # 右上 (N_active, 2)
                    torch.stack([-half_length, half_width], dim=1)    # 左上 (N_active, 2)
                ], dim=1)  # (N_active, 4, 2)
                
                # 构建旋转矩阵 (N_active, 2, 2)
                # R = [[cos(yaw), -sin(yaw)],
                #      [sin(yaw),  cos(yaw)]]
                # 用于将局部坐标 (x_local, y_local) 旋转到世界坐标
                # 构建旋转矩阵的行
                row1 = torch.stack([cos_yaw, -sin_yaw], dim=1)  # (N_active, 2) - 第一行 [cos, -sin]
                row2 = torch.stack([sin_yaw, cos_yaw], dim=1)   # (N_active, 2) - 第二行 [sin, cos]
                
                # 堆叠成 (N_active, 2, 2) 矩阵
                rotation_matrices = torch.stack([row1, row2], dim=1)  # (N_active, 2, 2)
                
                # 旋转角点：corners_local @ rotation_matrices^T
                # corners_local: (N_active, 4, 2), rotation_matrices: (N_active, 2, 2)
                # 需要转置旋转矩阵的最后一个维度：rotation_matrices.transpose(-2, -1)
                corners_rotated = torch.bmm(corners_local, rotation_matrices.transpose(-2, -1))  # (N_active, 4, 2)
                
                # 平移到世界坐标
                positions_expanded = positions.unsqueeze(1).expand(-1, 4, -1)  # (N_active, 4, 2)
                corners_world = corners_rotated + positions_expanded  # (N_active, 4, 2)
                
                # 转换为 numpy（延迟转换）
                corners_np = corners_world.detach().cpu().numpy()  # (N_active, 4, 2)
                positions_np = positions.detach().cpu().numpy()  # (N_active, 2)
                
            # 移除旧的 agent visuals
            for visual in self.agent_visuals:
                if visual.parent is not None:
                    visual.parent = None  # 从场景中移除
            self.agent_visuals.clear()
            
            # 为每个活跃的 agent 绘制车辆矩形
            # 定义颜色（循环使用）
            colors = [(1.0, 1.0, 1.0, 1.0),]
            
            for i in range(n_active):
                quad_verts = corners_np[i]  # (4, 2)
                # 闭合矩形：添加第一个点到最后
                closed_verts = np.vstack([quad_verts, quad_verts[0:1]])  # (5, 2)
                
                color = colors[i % len(colors)]
                
                # 绘制车辆轮廓
                vehicle_line = visuals.Line(
                    pos=closed_verts,
                    color=color,
                    width=2.0,
                    connect='strip',
                    method='gl'
                )
                self.view.add(vehicle_line)
                self.agent_visuals.append(vehicle_line)
                
                # 绘制车辆中心点
                # center_marker = visuals.Markers(
                #     pos=positions_np[i:i+1],
                #     size=8,
                #     face_color=color,
                #     edge_color=(1.0, 1.0, 1.0, 1.0),
                #     edge_width=1
                # )
                # self.view.add(center_marker)
                # self.agent_visuals.append(center_marker)
        
        def _render_paths(env_idx=0, agent_idx=0):
            """绘制指定环境的指定 agent 路径规划：用白色点显示所有路径点，并带朝向箭头"""
            if self.agents_path_plans is None or self.agents_path_plans.numel() == 0:
                return
            if env_idx >= self.agents_path_plans.shape[0]:
                return
            env_path_ids = self.agents_path_plans[env_idx]  # (M, L)
            if agent_idx >= env_path_ids.shape[0]:
                return
            if self.agents_state is None or env_idx >= self.agents_state.shape[0]:
                return
            env_states = self.agents_state[env_idx]  # (M, 7)
            if agent_idx >= env_states.shape[0] or env_states[agent_idx, 6] != 1.0:
                return
            
            # 定义颜色（与 _render_agents 保持一致）
            color = (0.0, 1.0, 0.0, 0.8)  # 红色，只绘制一个 agent 所以只需要一个颜色
            
            with torch.no_grad():
                agent_path_ids = env_path_ids[agent_idx]
                agent_path_features = self.path_planner.get_w_lane_features_by_id(agent_path_ids.unsqueeze(0))[0]
                
                invalid_value = float(self.path_planner.INVALID_MARKER)
                valid_mask = agent_path_features[:, 0] != invalid_value
                if not valid_mask.any():
                    return
                
                valid_path = agent_path_features[valid_mask]
                valid_points = valid_path[:, :2]
                valid_angles = valid_path[:, 2]
                
                valid_points_np = valid_points.detach().cpu().numpy()
                valid_angles_np = valid_angles.detach().cpu().numpy()
                
                if valid_points_np.shape[0] == 0:
                    return
            
            for visual in self.path_visuals:
                if visual.parent is not None:
                    visual.parent = None
            self.path_visuals.clear()
            
            arrow_length = 4
            points_marker = visuals.Markers(
                pos=valid_points_np,
                size=8,
                face_color=color,
                edge_color=(1.0, 1.0, 1.0, 1.0),  # 白色边框
                edge_width=1
            )
            self.view.add(points_marker)
            self.path_visuals.append(points_marker)
            
            # 为每个点绘制朝向箭头
            arrow_segments = []
            for i in range(valid_points_np.shape[0]):
                point = valid_points_np[i]
                angle = valid_angles_np[i]
                arrow_end = point + arrow_length * np.array([np.cos(angle), np.sin(angle)])
                arrow_segments.append(point)
                arrow_segments.append(arrow_end)
            
            if arrow_segments:
                arrow_segments_np = np.array(arrow_segments)
                arrow_color = (color[0], color[1], color[2], 0.8)
                arrow_lines = visuals.Line(
                    pos=arrow_segments_np,
                    color=arrow_color,
                    width=1.5,
                    connect='segments',  # 每两个点构成一条线段
                    method='gl'
                )
                self.view.add(arrow_lines)
                self.path_visuals.append(arrow_lines)
        
        def _render_goal_quad(env_idx=0, agent_idx=0):
            """绘制指定环境的指定 agent 的目标quad的白色边框"""
            if self.agents_goal_quad_ids is None:
                return
            if env_idx >= self.agents_goal_quad_ids.shape[0]:
                return
            if agent_idx >= self.agents_goal_quad_ids.shape[1]:
                return
            
            goal_quad_id = self.agents_goal_quad_ids[env_idx, agent_idx].item()
            invalid_marker = -1
            if goal_quad_id == invalid_marker:
                return
            
            # 通过 quad_id 找到在 quad_ids 中的索引
            with torch.no_grad():
                quad_ids = self.road_network.quad_ids  # (N,)
                matching_indices = (quad_ids == goal_quad_id).nonzero(as_tuple=False)
                if matching_indices.numel() == 0:
                    return
                quad_idx = matching_indices[0].item()
                
                # 从 quads_vertices 获取顶点
                quads_vertices = self.road_network.quads_vertices  # (N, 4, 2)
                if quad_idx >= quads_vertices.shape[0]:
                    return
                vertices = quads_vertices[quad_idx]  # (4, 2)
                vertices_np = vertices.detach().cpu().numpy()
                
                # 闭合quad：添加第一个点到最后
                closed_vertices = np.vstack([vertices_np, vertices_np[0:1]])  # (5, 2)
            
            # 移除旧的goal quad visual
            if self.goal_quad_visual is not None and self.goal_quad_visual.parent is not None:
                self.goal_quad_visual.parent = None
            
            # 绘制白色边框
            self.goal_quad_visual = visuals.Line(
                pos=closed_vertices,
                color=(0.5, 0.5, 0.5, 0.8),  # 白色
                width=5.0,  # 较粗的边框
                connect='strip',
                method='gl'
            )
            self.view.add(self.goal_quad_visual)
        
        def _render_observation(env_idx=0, agent_idx=0):
            """绘制指定环境的指定 agent 的观测内容：w_lanes 和 w_boundaries"""
            if not hasattr(self, 'initial_observation') or self.initial_observation is None:
                return
            if env_idx >= self.initial_observation.shape[0]:
                return
            if agent_idx >= self.initial_observation.shape[1]:
                return
            
            # 解包观测以获取 w_lanes_local 和 w_boundaries_local
            obs_gen = self.observation_generator
            local_state, neighbors_local, w_lanes_local, w_boundaries_local = \
                obs_gen.unpack_observation_components(
                    self.initial_observation,
                    obs_gen.local_state_dim,
                    obs_gen.num_neighbors,
                    obs_gen.neighbor_feature_dim,
                    obs_gen.num_w_lanes,
                    obs_gen.w_lane_feature_dim,
                    obs_gen.num_w_boundaries,
                    obs_gen.boundary_feature_dim
                )
            
            # 获取指定agent的局部观测
            w_lanes_local_agent = w_lanes_local[env_idx, agent_idx]  # (num_w_lanes, 2)
            w_boundaries_local_agent = w_boundaries_local[env_idx, agent_idx]  # (num_w_boundaries, 2)
            
            # 获取ego状态以转换回世界坐标
            ego_state = self.agents_state[env_idx, agent_idx]  # (7,)
            ego_pos = ego_state[:2]  # (2,)
            ego_yaw = ego_state[2]  # scalar
            
            # 构建逆旋转矩阵（从局部坐标到世界坐标）
            cos_yaw = torch.cos(ego_yaw)
            sin_yaw = torch.sin(ego_yaw)
            rot_matrix = torch.stack([
                torch.stack([cos_yaw, -sin_yaw], dim=0),
                torch.stack([sin_yaw, cos_yaw], dim=0)
            ], dim=0)  # (2, 2)
            
            # 将局部坐标转换回世界坐标
            with torch.no_grad():
                # w_lanes: (num_w_lanes, 2) -> (num_w_lanes, 2)
                w_lanes_world_agent = (w_lanes_local_agent @ rot_matrix.T) + ego_pos.unsqueeze(0)
                # w_boundaries: (num_w_boundaries, 2) -> (num_w_boundaries, 2)
                w_boundaries_world_agent = (w_boundaries_local_agent @ rot_matrix.T) + ego_pos.unsqueeze(0)
                
                # 过滤掉无效点（距离为0的点通常表示超出视野范围）
                w_lanes_dist = torch.norm(w_lanes_local_agent, dim=-1)
                w_boundaries_dist = torch.norm(w_boundaries_local_agent, dim=-1)
                w_lanes_valid = w_lanes_dist > 1e-6  # 有效点
                w_boundaries_valid = w_boundaries_dist > 1e-6  # 有效点
                
                if w_lanes_valid.any():
                    w_lanes_valid_points = w_lanes_world_agent[w_lanes_valid].detach().cpu().numpy()
                else:
                    w_lanes_valid_points = np.empty((0, 2))
                
                if w_boundaries_valid.any():
                    w_boundaries_valid_points = w_boundaries_world_agent[w_boundaries_valid].detach().cpu().numpy()
                else:
                    w_boundaries_valid_points = np.empty((0, 2))
            
            # 移除旧的观测 visuals
            if self.observation_w_lanes_visual is not None and self.observation_w_lanes_visual.parent is not None:
                self.observation_w_lanes_visual.parent = None
            if self.observation_w_boundaries_visual is not None and self.observation_w_boundaries_visual.parent is not None:
                self.observation_w_boundaries_visual.parent = None
            
            # 绘制 w_lanes (橙色点)
            if w_lanes_valid_points.shape[0] > 0:
                self.observation_w_lanes_visual = visuals.Markers(
                    pos=w_lanes_valid_points,
                    size=6,
                    face_color=(1.0, 0.5, 0.0, 0.8),  # 橙色
                    edge_color=(1.0, 0.5, 0.0, 1.0),
                    edge_width=1
                )
                self.view.add(self.observation_w_lanes_visual)
            
            # 绘制 w_boundaries (紫色点)
            if w_boundaries_valid_points.shape[0] > 0:
                self.observation_w_boundaries_visual = visuals.Markers(
                    pos=w_boundaries_valid_points,
                    size=5,
                    face_color=(0.5, 0.0, 0.5, 0.8),  # 紫色
                    edge_color=(0.5, 0.0, 0.5, 1.0),
                    edge_width=1
                )
                self.view.add(self.observation_w_boundaries_visual)
        
        def _create_info_window():
            """创建信息显示窗口，显示当前观测agent的状态信息"""
            if self.info_canvas is None:
                # 创建新的canvas用于显示信息（不使用交互），增大宽度以容纳更多文字
                self.info_canvas = scene.SceneCanvas(keys=None, show=True, size=(1200, 400), 
                                                     title='Agent State Information', bgcolor='black')
                self.info_view = self.info_canvas.central_widget.add_view()
                # 使用2D相机，禁用交互，设置固定范围
                self.info_view.camera = scene.PanZoomCamera(aspect=1.0)
                # 禁用相机的交互功能
                self.info_view.camera.interactive = False
            
            # 获取窗口实际大小
            canvas_size = self.info_canvas.size
            width, height = canvas_size[0], canvas_size[1]
            
            # 根据窗口大小动态设置相机范围
            self.info_view.camera.set_range(x=(0, width), y=(0, height))
            
            # 清除旧的文本visuals
            for text_visual in self.info_text_visuals:
                if text_visual.parent is not None:
                    text_visual.parent = None
            self.info_text_visuals.clear()
            
            # 获取当前观测的agent状态
            env_idx, agent_idx = _find_first_active_agent()
            if env_idx is not None and agent_idx is not None:
                # 解包观测以获取local_state和neighbors_local
                if hasattr(self, 'initial_observation') and self.initial_observation is not None:
                    obs_gen = self.observation_generator
                    local_state, neighbors_local, w_lanes_local, w_boundaries_local = \
                        obs_gen.unpack_observation_components(
                            self.initial_observation,
                            obs_gen.local_state_dim,
                            obs_gen.num_neighbors,
                            obs_gen.neighbor_feature_dim,
                            obs_gen.num_w_lanes,
                            obs_gen.w_lane_feature_dim,
                            obs_gen.num_w_boundaries,
                            obs_gen.boundary_feature_dim
                        )
                    
                    # 获取指定agent的local_state
                    local_state_agent = local_state[env_idx, agent_idx]  # (local_state_dim,) 通常是 (7,)
                    # local_state: 在局部坐标系下，前3个值(x, y, yaw)通常为0（因为是以自己为原点），
                    #             只包含 [length, width, active] 等不变的特征
                    # 将tensor转换为列表（不需要转到CPU，tolist()会自动处理）
                    local_state_values = local_state_agent.detach().tolist()
                    
                    # 根据local_state_dim格式化显示（通常是7维）
                    local_state_dim = local_state_agent.shape[0]
                    if local_state_dim >= 7:
                        x, y, yaw, v, length, width, active = local_state_values[:7]
                        state_text = f"Local_State (env={env_idx}, agent={agent_idx}): x={x:.2f}, y={y:.2f}, yaw={yaw:.3f}, v={v:.2f}, len={length:.2f}, wid={width:.2f}, active={int(active)}"
                    else:
                        # 如果维度不足7，显示所有值
                        values_str = ", ".join([f"{val:.3f}" for val in local_state_values])
                        state_text = f"Local_State (env={env_idx}, agent={agent_idx}): [{values_str}]"
                    
                    # 在第一行显示local_state，位置动态匹配窗口大小（距离顶部和左边一定距离）
                    margin_x = 10  # 距离左边的边距
                    line_height = 22  # 行高（用于控制多行文本间距）
                    margin_y = height - 20  # 距离顶部的边距（确保文字可见）
                    text_visual = scene.visuals.Text(
                        text=state_text,
                        pos=(margin_x, margin_y),  # 使用左上角为锚点
                        color='white',
                        font_size=16,
                        parent=self.info_view.scene,
                        anchor_x='left',
                        anchor_y='top'
                    )
                    self.info_text_visuals.append(text_visual)
                    
                    # 获取指定agent的neighbors_local，形状应该是 (num_neighbors, neighbor_feature_dim)
                    neighbors_local_agent = neighbors_local[env_idx, agent_idx]  # (num_neighbors, neighbor_feature_dim)
                    
                    # 将neighbors_local转换为numpy并格式化为矩阵字符串
                    neighbors_np = neighbors_local_agent.detach().cpu().numpy()
                    # neighbors_np 形状: (num_neighbors, neighbor_feature_dim)，例如 (20, 7)
                    
                    # 格式化矩阵：以7x20矩阵形式呈现（7个特征，20个neighbors）
                    # 矩阵转置：显示为7行（特征）x 20列（neighbors）
                    matrix_lines = ["Neighbors (7x20 matrix - rows: features, cols: neighbors):"]
                    
                    # 转置矩阵以便显示：从 (20, 7) 转为 (7, 20)
                    neighbors_t = neighbors_np.T  # (7, 20)
                    
                    # 显示7行，每行对应一个特征，包含20个neighbors的值
                    feature_names = ["x", "y", "yaw", "v", "len", "wid", "active"]
                    for feat_idx in range(neighbors_t.shape[0]):  # 遍历7个特征
                        feature_values = neighbors_t[feat_idx]  # (20,)
                        # 格式化：每个值保留2位小数，紧凑格式
                        values_str = " ".join([f"{val:7.2f}" for val in feature_values])
                        matrix_lines.append(f"  {feature_names[feat_idx]:6s}: [{values_str}]")
                    
                    # 第二行开始：显示neighbors矩阵（每行单独创建Text visual以避免重叠）
                    # neighbors矩阵起始位置：在state下方，留出足够空间
                    neighbors_start_y = margin_y - line_height   # state下方留出更多空间
                    current_y = neighbors_start_y
                    
                    # 显示标题行
                    title_visual = scene.visuals.Text(
                        text=matrix_lines[0],
                        pos=(margin_x, current_y),
                        color='cyan',
                        font_size=12,
                        parent=self.info_view.scene,
                        anchor_x='left',
                        anchor_y='top'
                    )
                    self.info_text_visuals.append(title_visual)
                    current_y -= line_height  # 移动到下一行
                    
                    # 显示7行特征数据（每行单独创建Text visual）
                    for i in range(1, len(matrix_lines)):  # 跳过标题行
                        feature_visual = scene.visuals.Text(
                            text=matrix_lines[i],
                            pos=(margin_x, current_y),
                            color='cyan',
                            font_size=12,
                            parent=self.info_view.scene,
                            anchor_x='left',
                            anchor_y='top'
                        )
                        self.info_text_visuals.append(feature_visual)
                        current_y -= line_height  # 移动到下一行
            else:
                # 没有active agent时显示提示信息
                margin_x = 10
                margin_y = height - 30
                text_visual = scene.visuals.Text(
                    text="No active agent found",
                    pos=(margin_x, margin_y),
                    color='yellow',
                    font_size=16,
                    parent=self.info_view.scene
                )
                self.info_text_visuals.append(text_visual)
        
        env_idx, agent_idx = _find_first_active_agent()
        if env_idx is not None and agent_idx is not None:
            _render_quads()
            _render_observation(env_idx, agent_idx)  # 先绘制观测（在背景上）
            _render_paths(env_idx, agent_idx)
            _render_agents(env_idx, agent_idx)
            _render_goal_quad(env_idx, agent_idx)
        else:
            _render_quads()
        
        # 创建信息显示窗口
        _create_info_window()
        
        self.canvas.title = f'Selfrace(OpenGL)'
        vispy.app.run()

if __name__ == '__main__':
    import json
    config = json.load(open('configs/default_config.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = selfrace(config, device)
    simulator.render()
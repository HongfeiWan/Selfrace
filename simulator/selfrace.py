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
        # 初始化路径规划器 - 为所有智能体分配目标和生成路径规划
        # 确保输入tensor在正确的设备上
        self.agents_start_quad_ids = self.agents_start_quad_ids.to(self.device)
        self.agents_goal_quad_ids = self.agents_goal_quad_ids.to(self.device)
        # path_plan 期望输入: (B, M) 的 start_poly_ids 和 end_poly_ids
        # 返回: (B, M, max_path_len) 的路径（lane索引）
        paths = self.path_planner.path_plan(self.agents_start_quad_ids, self.agents_goal_quad_ids)
        # collect_path_w_lane_ids 期望输入: paths (B, M, max_path_len), start_poly_ids (B, M), end_poly_ids (B, M)
        # 返回: (B, M, w_lane_ids_length) 的 w_lane_id 序列
        self.agents_path_plans = self.path_planner.collect_path_w_lane_ids(paths, self.agents_start_quad_ids, self.agents_goal_quad_ids)

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
            colors = [(1.0, 0.0, 0.0, 0.8), (0.0, 0.0, 1.0, 0.8), (0.0, 1.0, 0.0, 0.8), 
                    (1.0, 0.5, 0.0, 0.8), (0.5, 0.0, 1.0, 0.8), (0.5, 0.25, 0.0, 0.8),
                    (1.0, 0.0, 1.0, 0.8), (0.5, 0.5, 0.5, 0.8), (0.5, 0.5, 0.0, 0.8), (0.0, 1.0, 1.0, 0.8)]
            
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
            """绘制指定环境的指定 agent 路径规划：用白色点显示所有路径点，并带朝向箭头
            使用 self.agents_state 来检查 active 状态，使用 self.agents_path_plans 来获取路径
            """
            if self.agents_path_plans is None or self.agents_path_plans.numel() == 0:
                return
            # 检查索引范围
            if env_idx >= self.agents_path_plans.shape[0]:
                return
            env_path_ids = self.agents_path_plans[env_idx]  # (M, L)
            if agent_idx >= env_path_ids.shape[0]:
                return
            
            # 检查指定的 agent 是否 active（使用 self.agents_state，与路径规划对应）
            if self.agents_state is not None and self.agents_state.numel() > 0:
                if env_idx < self.agents_state.shape[0]:
                    env_states = self.agents_state[env_idx]  # (M, 7)
                    if agent_idx >= env_states.shape[0] or env_states[agent_idx, 6] != 1.0:
                        return
            
            # 定义颜色（与 _render_agents 保持一致）
            color = (1.0, 0.0, 0.0, 0.8)  # 红色，只绘制一个 agent 所以只需要一个颜色
            
            # 获取路径特征（包含坐标和角度）
            with torch.no_grad():
                # 获取指定 agent 的路径 ID
                # agents_path_plans 的形状是 (B, M, L)，其中 L = w_lane_ids_length
                agent_path_ids = env_path_ids[agent_idx]  # (L,)
                
                # 获取路径点的特征 (L, 3) - (x, y, angle)
                agent_path_features = self.path_planner.get_w_lane_features_by_id(agent_path_ids.unsqueeze(0))  # (1, L, 3)
                agent_path_features = agent_path_features[0]  # (L, 3)
                
                # 过滤无效点
                invalid_value = float(self.path_planner.INVALID_MARKER)
                valid_mask = agent_path_features[:, 0] != invalid_value
                if not valid_mask.any():
                    return
                
                valid_path = agent_path_features[valid_mask]  # (L_valid, 3)
                valid_points = valid_path[:, :2]  # (L_valid, 2) - (x, y)
                valid_angles = valid_path[:, 2]  # (L_valid,) - angle
                
                # 转换为 numpy
                valid_points_np = valid_points.detach().cpu().numpy()
                valid_angles_np = valid_angles.detach().cpu().numpy()
                
                if valid_points_np.shape[0] == 0:
                    return
            
            # 移除旧的路径 visuals
            for visual in self.path_visuals:
                if visual.parent is not None:
                    visual.parent = None
            self.path_visuals.clear()
            
            # 绘制路径点和箭头
            arrow_length = 0.5  # 箭头长度（可以根据地图尺度调整）
            
            # 绘制路径点
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
                point = valid_points_np[i]  # (2,)
                angle = valid_angles_np[i]
                
                # 计算箭头终点
                arrow_end = point + arrow_length * np.array([np.cos(angle), np.sin(angle)])
                
                # 创建箭头线段：从点到箭头终点
                arrow_segments.append(point)
                arrow_segments.append(arrow_end)
            
            if arrow_segments:
                arrow_segments_np = np.array(arrow_segments)  # (N_points*2, 2)
                
                # 绘制箭头线段（稍微透明）
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
        
        # 找到第一个 active 的 agent
        env_idx, agent_idx = _find_first_active_agent()
        if env_idx is None or agent_idx is None:
            print("Warning: No active agents found")
            _render_quads()
        else:
            # 绘制顺序：先绘制背景（quads），再绘制路径，最后绘制 agents（确保 agents 在最上层）
            _render_quads()
            _render_paths(env_idx, agent_idx)
            _render_agents(env_idx, agent_idx)
        # 设置标题
        self.canvas.title = f'Selfrace(OpenGL)'
        # 运行应用（如果还没有运行）
        vispy.app.run()
    
if __name__ == '__main__':
    import json
    config = json.load(open('configs/default_config.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = selfrace(config, device)
    simulator.render()
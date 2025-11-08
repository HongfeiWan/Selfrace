"""
Pygame/OpenGL 可视化工具模块
用于可视化车辆路径规划和观测数据
"""
import torch
import numpy as np
import math

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    from OpenGL.GL import (glClearColor, glClear, GL_COLOR_BUFFER_BIT, glMatrixMode, 
                          GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho, 
                          glBegin, glEnd, glVertex2f, glColor3f, glColor4f, GL_QUADS, GL_LINES, 
                          GL_POINTS, GL_POLYGON, GL_LINE_LOOP, glPointSize, glLineWidth, glEnable, glBlendFunc,
                          GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame or PyOpenGL not available. Install with: pip install pygame PyOpenGL")


class PathPlanningVisualizer:
    """路径规划可视化器"""
    
    def __init__(self, simulator, batch_idx=0):
        """
        初始化可视化器
        Args:
            simulator: TeraflowSimulator实例
            batch_idx: 要可视化的批次索引（默认0）
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame and PyOpenGL are required for visualization")
        
        self.sim = simulator
        self.batch_idx = batch_idx
        
        # 获取数据
        self.agents_state = simulator.agents_state
        self.agents_path_plans = simulator.agents_path_plans
        self.start_poly_tensor = simulator.agents_start_quad_ids
        self.end_poly_tensor = simulator.agents_goal_quad_ids
        self.invalid_marker_value = simulator.path_planner.INVALID_w_lane_id_MARKER
        self.horizon = simulator.observation_generator.horizon
        
        # 准备地图数据
        self.quads_vertices_np = simulator.road_network.quads_vertices.detach().cpu().numpy()
        
        # 获取active agents
        active_mask = self.agents_state[..., 6] > 0.5
        active_agents = torch.nonzero(active_mask[batch_idx], as_tuple=False).squeeze(-1)
        if active_agents.numel() == 0:
            raise ValueError(f"Batch {batch_idx} 中没有active车辆")
        
        self.active_agents_list = active_agents.tolist()
        self.B, self.M = self.agents_state.shape[:2]
        
        print(f"Batch {batch_idx} 中有 {len(self.active_agents_list)} 个active车辆")
        
        # 计算坐标范围
        if self.quads_vertices_np.shape[0] > 0:
            xs = self.quads_vertices_np[:, :, 0].reshape(-1)
            ys = self.quads_vertices_np[:, :, 1].reshape(-1)
            margin = 20.0
            self.x_min, self.x_max = float(xs.min() - margin), float(xs.max() + margin)
            self.y_min, self.y_max = float(ys.min() - margin), float(ys.max() + margin)
        else:
            self.x_min, self.x_max, self.y_min, self.y_max = -100.0, 100.0, -100.0, 100.0
        
        # 初始化Pygame和OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 960), DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Active Vehicles Path Plans (SPACE: next, ESC: quit)')
        
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.x_min, self.x_max, self.y_min, self.y_max, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
        # 状态变量
        self.cur_idx = 0
        self.zoom_level = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 10.0
        self.clock = pygame.time.Clock()
    
    def draw_quads(self):
        """绘制所有 quads 作为背景"""
        glColor3f(0.95, 0.95, 0.7)  # 浅黄色
        for verts in self.quads_vertices_np:
            glBegin(GL_QUADS)
            glVertex2f(float(verts[0, 0]), float(verts[0, 1]))
            glVertex2f(float(verts[1, 0]), float(verts[1, 1]))
            glVertex2f(float(verts[2, 0]), float(verts[2, 1]))
            glVertex2f(float(verts[3, 0]), float(verts[3, 1]))
            glEnd()
        
        # 绘制 quad 边框
        glColor3f(0.6, 0.6, 0.6)
        glLineWidth(0.5)
        for verts in self.quads_vertices_np:
            glBegin(GL_LINES)
            for i in range(4):
                glVertex2f(float(verts[i, 0]), float(verts[i, 1]))
                glVertex2f(float(verts[(i+1)%4, 0]), float(verts[(i+1)%4, 1]))
            glEnd()
    
    def draw_vehicle_box(self, x, y, heading, length, width, r, g, b, line_width=2.0):
        """绘制车辆的矩形框"""
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        half_length = length / 2.0
        half_width = width / 2.0
        
        corners = [
            (-half_length, -half_width),  # 后左
            (half_length, -half_width),   # 前左
            (half_length, half_width),    # 前右
            (-half_length, half_width)    # 后右
        ]
        
        world_corners = []
        for cx, cy in corners:
            wx = x + cx * cos_h - cy * sin_h
            wy = y + cx * sin_h + cy * cos_h
            world_corners.append((wx, wy))
        
        glColor3f(r, g, b)
        glLineWidth(line_width)
        glBegin(GL_LINES)
        for i in range(4):
            glVertex2f(world_corners[i][0], world_corners[i][1])
            glVertex2f(world_corners[(i+1)%4][0], world_corners[(i+1)%4][1])
        glEnd()
        
        # 绘制前方指示线
        front_x = x + half_length * cos_h
        front_y = y + half_length * sin_h
        glBegin(GL_LINES)
        glVertex2f(x, y)
        glVertex2f(front_x, front_y)
        glEnd()
    
    def draw_horizon_box(self, ego_x, ego_y, horizon_size):
        """绘制观测范围的圆形"""
        radius = horizon_size
        num_segments = 64
        
        # 绘制半透明填充圆
        glColor4f(0.0, 0.8, 0.8, 0.08)
        glBegin(GL_POLYGON)
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = ego_x + radius * math.cos(angle)
            y = ego_y + radius * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
        
        # 绘制边框圆线
        glColor4f(0.0, 0.8, 0.8, 0.8)
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = ego_x + radius * math.cos(angle)
            y = ego_y + radius * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
    
    def draw_all_agents(self, current_m):
        """绘制所有 active agents（除了当前选中的）"""
        b = self.batch_idx
        for m_idx in range(self.M):
            if self.agents_state[b, m_idx, 6] < 0.5:  # 不是 active
                continue
            if m_idx == current_m:  # 跳过当前选中的车辆
                continue
            
            state = self.agents_state[b, m_idx].cpu().numpy()
            x, y, heading = state[0], state[1], state[2]
            length, width = state[4], state[5]
            
            self.draw_vehicle_box(x, y, heading, length, width, 0.4, 0.4, 0.4, line_width=1.5)
    
    def draw_observation_data(self, neighbors_local, w_lanes_local, w_boundaries_local, 
                            ego_x, ego_y, ego_heading, ego_speed):
        """绘制 observation 中的数据（透明显示）"""
        if neighbors_local is None:
            return
        
        cos_yaw = math.cos(ego_heading)
        sin_yaw = math.sin(ego_heading)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # 1. 绘制观测到的其他车辆（红色框）
        try:
            neighbors_np = neighbors_local.cpu().numpy() if torch.is_tensor(neighbors_local) else neighbors_local
            
            for k in range(neighbors_np.shape[0]):
                neighbor = neighbors_np[k]
                dx, dy = neighbor[0], neighbor[1]
                
                # 跳过无效邻居
                if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                    continue
                if neighbor[6] < 0.5:  # active flag
                    continue
                
                # 局部坐标转世界坐标
                dx_dy = np.array([dx, dy])
                world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                world_x, world_y = world_pos[0], world_pos[1]
                
                # 从相对速度恢复世界速度和朝向
                dvx, dvy = neighbor[2], neighbor[3]
                v_local = np.array([dvx, dvy])
                v_ego_world = np.array([ego_speed * cos_yaw, ego_speed * sin_yaw])
                v_neighbor_world = v_local @ rot.T + v_ego_world
                
                vx_k, vy_k = v_neighbor_world[0], v_neighbor_world[1]
                if (vx_k * vx_k + vy_k * vy_k) < 1e-6:
                    world_heading = ego_heading
                else:
                    world_heading = math.atan2(vy_k, vx_k)
                
                length = neighbor[4]
                width = neighbor[5]
                
                if length <= 0.0 or width <= 0.0:
                    continue
                
                # 绘制红色车辆框
                self.draw_vehicle_box(world_x, world_y, world_heading, length, width, 
                                   1.0, 0.0, 0.0, line_width=2.5)
        except Exception as e:
            print(f"绘制 neighbors 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 绘制 w_lane（蓝色散点）
        try:
            if w_lanes_local is not None:
                wlane_np = w_lanes_local.cpu().numpy() if torch.is_tensor(w_lanes_local) else w_lanes_local
                
                glColor4f(0.2, 0.4, 0.9, 0.8)
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(wlane_np.shape[0]):
                    dx, dy = wlane_np[i, 0], wlane_np[i, 1]
                    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                        continue
                    
                    dx_dy = np.array([dx, dy])
                    world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                    glVertex2f(world_pos[0], world_pos[1])
                glEnd()
        except Exception as e:
            print(f"绘制 w_lanes 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 绘制 boundary（红色散点）
        try:
            if w_boundaries_local is not None:
                boundary_np = w_boundaries_local.cpu().numpy() if torch.is_tensor(w_boundaries_local) else w_boundaries_local
                
                glColor4f(0.9, 0.2, 0.2, 0.8)
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(boundary_np.shape[0]):
                    dx, dy = boundary_np[i, 0], boundary_np[i, 1]
                    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
                        continue
                    
                    dx_dy = np.array([dx, dy])
                    world_pos = dx_dy @ rot.T + np.array([ego_x, ego_y])
                    glVertex2f(world_pos[0], world_pos[1])
                glEnd()
        except Exception as e:
            print(f"绘制 boundaries 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_point_xy(self, x, y, r, g, b, size=8.0):
        """绘制一个点"""
        glColor3f(r, g, b)
        glPointSize(float(size))
        glBegin(GL_POINTS)
        glVertex2f(float(x), float(y))
        glEnd()
    
    def draw_path_with_arrows(self, valid_path):
        """绘制路径点、连线和方向箭头"""
        if len(valid_path) == 0:
            return
        
        # 1. 绘制路径连线
        glColor3f(0.2, 0.4, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for i in range(len(valid_path) - 1):
            glVertex2f(float(valid_path[i, 0]), float(valid_path[i, 1]))
            glVertex2f(float(valid_path[i+1, 0]), float(valid_path[i+1, 1]))
        glEnd()
        
        # 2. 绘制所有路径点
        glColor3f(0.2, 0.4, 0.8)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for p in valid_path:
            glVertex2f(float(p[0]), float(p[1]))
        glEnd()
        
        # 3. 绘制方向箭头
        glColor3f(0.5, 0.1, 0.7)
        glLineWidth(1.5)
        arrow_length = 5.0
        for p in valid_path:
            x, y, angle = float(p[0]), float(p[1]), float(p[2])
            dx = arrow_length * np.cos(angle)
            dy = arrow_length * np.sin(angle)
            glBegin(GL_LINES)
            glVertex2f(x, y)
            glVertex2f(x + dx, y + dy)
            glEnd()
        
        # 4. 绘制起点
        self.draw_point_xy(valid_path[0, 0], valid_path[0, 1], 0.2, 0.8, 0.2, size=10.0)
        
        # 5. 绘制终点
        self.draw_point_xy(valid_path[-1, 0], valid_path[-1, 1], 0.9, 0.2, 0.2, size=12.0)
    
    def run(self):
        """运行可视化主循环"""
        running = True
        b = self.batch_idx
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.cur_idx = (self.cur_idx + 1) % len(self.active_agents_list)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # 滚轮向上
                        self.zoom_level = min(self.zoom_level * 1.2, self.zoom_max)
                    elif event.button == 5:  # 滚轮向下
                        self.zoom_level = max(self.zoom_level / 1.2, self.zoom_min)
            
            # 获取当前车辆数据
            m = self.active_agents_list[self.cur_idx]
            path = self.agents_path_plans[b, m]
            
            # 过滤无效路径点
            invalid_marker_tensor = torch.tensor(self.invalid_marker_value, 
                                                device=path.device, dtype=path.dtype)
            valid_mask = path[:, 0] != invalid_marker_tensor
            valid_path = path[valid_mask].cpu().numpy()
            
            # 获取当前车辆状态
            ego_state = self.agents_state[b, m].cpu().numpy()
            ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[2]
            ego_length, ego_width = ego_state[4], ego_state[5]
            
            # 获取观测数据
            try:
                neighbor_states_world = self.sim.observation_generator._get_nearest_neighbors(self.agents_state)
                w_lanes_world, w_boundaries_world = self.sim.observation_generator._get_precomputed_w_lanes(self.agents_state)
                local_state_tmp, neighbors_local_tmp, w_lanes_local_tmp, w_boundaries_local_tmp = \
                    self.sim.observation_generator._world_to_ego_centric(
                        self.agents_state, neighbor_states_world, w_lanes_world, w_boundaries_world
                    )
                
                neighbors_local = neighbors_local_tmp[b, m]
                w_lanes_local = w_lanes_local_tmp[b, m]
                w_boundaries_local = w_boundaries_local_tmp[b, m]
                has_observation = True
            except Exception as e:
                print(f"获取 observation 失败: {e}")
                has_observation = False
                neighbors_local = None
                w_lanes_local = None
                w_boundaries_local = None
            
            # 更新窗口标题
            pygame.display.set_caption(
                f'Active Vehicles Path Plans - B={b}, M={m} ({self.cur_idx+1}/{len(self.active_agents_list)}) '
                f'(SPACE: next, ESC: quit, Scroll: zoom) - Path: {len(valid_path)} pts, Zoom: {self.zoom_level:.2f}x'
            )
            
            # 更新投影矩阵（缩放）
            view_width = (self.x_max - self.x_min) / self.zoom_level
            view_height = (self.y_max - self.y_min) / self.zoom_level
            
            view_x_min = ego_x - view_width / 2
            view_x_max = ego_x + view_width / 2
            view_y_min = ego_y - view_height / 2
            view_y_max = ego_y + view_height / 2
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(view_x_min, view_x_max, view_y_min, view_y_max, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            
            # 清空并绘制
            glClear(GL_COLOR_BUFFER_BIT)
            glLoadIdentity()
            
            # 绘制背景
            self.draw_quads()
            
            # 绘制 horizon 观测范围
            self.draw_horizon_box(ego_x, ego_y, self.horizon)
            
            # 绘制所有其他 agents
            self.draw_all_agents(m)
            
            # 绘制 observation 数据
            if has_observation:
                ego_speed = ego_state[3]
                self.draw_observation_data(neighbors_local, w_lanes_local, w_boundaries_local, 
                                        ego_x, ego_y, ego_heading, ego_speed)
            
            # 绘制当前车辆
            self.draw_vehicle_box(ego_x, ego_y, ego_heading, ego_length, ego_width, 
                                0.0, 0.8, 0.8, line_width=3.0)
            
            # 绘制路径
            if len(valid_path) > 0:
                self.draw_path_with_arrows(valid_path)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print(f"可视化结束，共查看了 {len(self.active_agents_list)} 个active车辆")


def visualize_path_planning(simulator, batch_idx=0):
    """
    便捷函数：可视化路径规划
    
    Args:
        simulator: TeraflowSimulator实例
        batch_idx: 要可视化的批次索引（默认0）
    """
    if not PYGAME_AVAILABLE:
        print("错误：请安装 pygame 和 PyOpenGL: pip install pygame PyOpenGL")
        return
    
    visualizer = PathPlanningVisualizer(simulator, batch_idx)
    visualizer.run()


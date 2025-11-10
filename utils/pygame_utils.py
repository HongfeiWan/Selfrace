"""
Pygame/OpenGL 可视化工具模块
用于可视化车辆路径规划和观测数据
"""
import torch
import numpy as np
import math
from typing import Optional, Callable, Tuple

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    from OpenGL.GL import (glClearColor, glClear, GL_COLOR_BUFFER_BIT, glMatrixMode, 
                          GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho, 
                          glBegin, glEnd, glVertex2f, glColor3f, glColor4f, GL_QUADS, GL_LINES, 
                          GL_POINTS, GL_POLYGON, GL_LINE_LOOP, glPointSize, glLineWidth, glEnable, glBlendFunc,
                          GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glPushMatrix, glPopMatrix,
                          glTexParameteri, glTexImage2D, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                          GL_TEXTURE_MAG_FILTER, GL_LINEAR, glBindTexture, glTexCoord2f, glDeleteTextures,
                          glGenTextures,
                          glDisable)
    from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame or PyOpenGL not available. Install with: pip install pygame PyOpenGL")


class PathPlanningVisualizer:
    """路径规划可视化器（通用版本）"""
    
    def __init__(
        self, 
        agents_state: torch.Tensor,
        agents_path_plans: torch.Tensor,
        quads_vertices: torch.Tensor,
        batch_idx: int = 0,
        invalid_marker_value: float = -999999.0,
        horizon: float = 80.0,
        observation_callback: Optional[Callable] = None,
        step_callback: Optional[Callable[[], Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]]] = None,
        info_callback: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int, int], Optional[object]]] = None,
        agents_start_quad_ids: Optional[torch.Tensor] = None,
        agents_goal_quad_ids: Optional[torch.Tensor] = None,
        goal_positions: Optional[torch.Tensor] = None,
        goal_radii: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None):
        """
        初始化可视化器
        
        Args:
            agents_state: 车辆状态张量 [B, M, 7] - (x, y, heading, speed, length, width, active)
            agents_path_plans: 路径规划张量 [B, M, N, 3] - (x, y, angle)
            quads_vertices: 道路网格顶点 [num_quads, 4, 2] - 每个quad的4个顶点坐标
            batch_idx: 要可视化的批次索引（默认0）
            invalid_marker_value: 无效路径点的标记值（默认-999999.0）
            horizon: 观测范围半径（默认80.0）
            observation_callback: 可选的观测数据获取回调函数
                                 签名: (agents_state, batch_idx, agent_idx) -> (neighbors_local, w_lanes_local, w_boundaries_local)
            agents_start_quad_ids: 可选的起始quad ID
            agents_goal_quad_ids: 可选的目标quad ID
            goal_positions: 可选的目标位置 (B, M, 2)
            goal_radii: 可选的目标半径 (B, M)
            done_mask: 可选的完成掩码 (B, M)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame and PyOpenGL are required for visualization")
        
        # 保存数据
        self.agents_state = agents_state
        self.agents_path_plans = agents_path_plans
        self.batch_idx = batch_idx
        self.invalid_marker_value = invalid_marker_value
        self.horizon = horizon
        self.observation_callback = observation_callback
        self.step_callback = step_callback
        self.info_callback = info_callback
        self.agents_start_quad_ids = agents_start_quad_ids
        self.agents_goal_quad_ids = agents_goal_quad_ids
        self.goal_positions = goal_positions
        self.goal_radii = goal_radii
        self.done_mask = done_mask
        
        # 准备道路几何数据
        self.road_geometry_np = quads_vertices.detach().cpu().numpy()
        
        # 获取active agents
        active_mask = self.agents_state[..., 6] > 0.5
        active_agents = torch.nonzero(active_mask[batch_idx], as_tuple=False).squeeze(-1)
        if active_agents.numel() == 0:
            raise ValueError(f"Batch {batch_idx} 中没有active车辆")
        
        self.active_agents_list = active_agents.tolist()
        self.B, self.M = self.agents_state.shape[:2]
        
        print(f"Batch {batch_idx} 中有 {len(self.active_agents_list)} 个active车辆")
        
        # 计算坐标范围
        if self.road_geometry_np.size > 0:
            coords = self.road_geometry_np.reshape(-1, self.road_geometry_np.shape[-1])
            xs = coords[:, 0]
            ys = coords[:, 1]
            margin = 20.0
            self.x_min, self.x_max = float(xs.min() - margin), float(xs.max() + margin)
            self.y_min, self.y_max = float(ys.min() - margin), float(ys.max() + margin)
        else:
            self.x_min, self.x_max, self.y_min, self.y_max = -100.0, 100.0, -100.0, 100.0
        
        # 初始化Pygame和OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 960), DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Active Vehicles Path Plans (SPACE: next, W: step, ESC: quit)')
        self.screen_width, self.screen_height = self.screen.get_size()
        
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
        
        pygame.font.init()
        try:
            self.info_font = pygame.font.SysFont('simhei', 18)
            self.info_title_font = pygame.font.SysFont('simhei', 20)
        except Exception:
            self.info_font = pygame.font.Font(None, 18)
            self.info_title_font = pygame.font.Font(None, 20)
        self.info_panel_width = 320
        self.info_panel_margin = 10
    
    def draw_quads(self):
        """绘制道路几何（支持四边形或线段）"""
        if self.road_geometry_np.size == 0:
            return
        geom = self.road_geometry_np
        if geom.ndim >= 3 and geom.shape[1] == 4:
            glColor3f(0.95, 0.95, 0.7)
            for verts in geom:
                glBegin(GL_QUADS)
                for i in range(4):
                    glVertex2f(float(verts[i, 0]), float(verts[i, 1]))
                glEnd()
            glColor3f(0.6, 0.6, 0.6)
            glLineWidth(0.5)
            for verts in geom:
                glBegin(GL_LINES)
                for i in range(4):
                    glVertex2f(float(verts[i, 0]), float(verts[i, 1]))
                    glVertex2f(float(verts[(i + 1) % 4, 0]), float(verts[(i + 1) % 4, 1]))
                glEnd()
        elif geom.ndim >= 3 and geom.shape[1] == 2:
            glColor3f(0.6, 0.6, 0.6)
            glLineWidth(1.5)
            for segment in geom:
                glBegin(GL_LINES)
                glVertex2f(float(segment[0, 0]), float(segment[0, 1]))
                glVertex2f(float(segment[1, 0]), float(segment[1, 1]))
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
    
    def draw_horizon_box(self, ego_x, ego_y, horizon_size, goal_radius=None):
        """绘制观测范围与目标半径"""
        radius = horizon_size
        num_segments = 64
        
        # 仅绘制边框圆线（避免区域被不透明填充）
        glColor4f(0.0, 0.8, 0.8, 0.8)
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = ego_x + radius * math.cos(angle)
            y = ego_y + radius * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
        
        if goal_radius is not None and goal_radius > 0:
            glColor4f(0.9, 0.4, 0.1, 0.9)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for i in range(num_segments):
                angle = 2.0 * math.pi * i / num_segments
                x = ego_x + goal_radius * math.cos(angle)
                y = ego_y + goal_radius * math.sin(angle)
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
            
            is_done = False
            if self.done_mask is not None:
                try:
                    is_done = bool(self.done_mask[b, m_idx].item())
                except Exception:
                    is_done = False

            if is_done:
                glColor4f(0.0, 0.6, 0.0, 0.1)
                glBegin(GL_QUADS)
                cos_h = math.cos(heading)
                sin_h = math.sin(heading)
                half_length = length / 2.0
                half_width = width / 2.0
                corners = [
                    (-half_length, -half_width),
                    (half_length, -half_width),
                    (half_length, half_width),
                    (-half_length, half_width),
                ]
                for cx, cy in corners:
                    wx = x + cx * cos_h - cy * sin_h
                    wy = y + cx * sin_h + cy * cos_h
                    glVertex2f(wx, wy)
                glEnd()
            else:
                self.draw_vehicle_box(x, y, heading, length, width, 0.4, 0.4, 0.4, line_width=1.5)
    
    def draw_observation_data(self, neighbors_local, w_lanes_local, w_boundaries_local, 
                            ego_x, ego_y, ego_heading, ego_speed):
        """绘制 observation 中的数据（透明显示）"""
        if neighbors_local is None:
            return
        
        cos_yaw = math.cos(ego_heading)
        sin_yaw = math.sin(ego_heading)
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])  # world -> ego
        rot_T = rot.T  # ego -> world
        v_ego_world = np.array([ego_speed * cos_yaw, ego_speed * sin_yaw])
        
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
                local_pos = np.array([dx, dy])
                world_pos = local_pos @ rot_T + np.array([ego_x, ego_y])
                world_x, world_y = world_pos
                
                # 从相对速度恢复世界速度和朝向
                dvx, dvy = neighbor[2], neighbor[3]
                v_local = np.array([dvx, dvy])
                v_neighbor_world = v_local @ rot_T + v_ego_world
                
                vx_k, vy_k = v_neighbor_world
                if (vx_k * vx_k + vy_k * vy_k) < 1e-6:
                    world_heading = math.atan2(world_y - ego_y, world_x - ego_x)
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
    
    def _format_info_lines(self, info_obj: Optional[object]) -> Optional[list]:
        if info_obj is None:
            return None
        lines = []
        if isinstance(info_obj, dict):
            for key, value in info_obj.items():
                lines.append(f"{key}: {value}")
        elif isinstance(info_obj, (list, tuple)):
            for item in info_obj:
                if isinstance(item, tuple) and len(item) == 2:
                    lines.append(f"{item[0]}: {item[1]}")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(info_obj))
        return lines if lines else None

    def draw_info_panel(self, lines: list):
        if not lines:
            return
        line_height = self.info_font.get_linesize()
        max_text_width = self.info_title_font.size("车辆状态")[0]
        rendered_lines = []
        for text in lines:
            try:
                surface = self.info_font.render(text, True, (220, 220, 220))
            except Exception:
                surface = self.info_font.render(text.encode('utf-8', 'ignore').decode('utf-8', 'ignore'), True, (220, 220, 220))
            rendered_lines.append(surface)
            max_text_width = max(max_text_width, surface.get_width())
        panel_width = max(self.info_panel_width, max_text_width + 20)
        panel_height = line_height * (len(rendered_lines) + 1) + 30
        surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        surface.fill((30, 30, 30, 220))
        y = 10
        try:
            title_surface = self.info_title_font.render("车辆状态", True, (255, 255, 255))
            surface.blit(title_surface, (10, y))
            y += line_height + 4
        except Exception:
            pass
        for text_surface in rendered_lines:
            surface.blit(text_surface, (10, y))
            y += line_height
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        x = self.info_panel_margin
        y = self.info_panel_margin
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(x, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + width, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + width, y + height)
        glTexCoord2f(0, 0)
        glVertex2f(x, y + height)
        glEnd()
        glDeleteTextures([texture])
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def run(self):
        """运行可视化主循环
        
        Returns:
            str: 退出原因，'quit' 表示退出可视化，'step' 表示请求外部执行一步仿真
        """
        running = True
        b = self.batch_idx
        exit_reason = 'quit'
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    exit_reason = 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        exit_reason = 'quit'
                    elif event.key == pygame.K_SPACE:
                        self.cur_idx = (self.cur_idx + 1) % len(self.active_agents_list)
                    elif event.key == pygame.K_w:
                        if self.step_callback is not None:
                            try:
                                step_result = self.step_callback()
                                if step_result is not None:
                                    if isinstance(step_result, tuple):
                                        new_agents_state = step_result[0] if len(step_result) > 0 else None
                                        new_agents_path_plans = step_result[1] if len(step_result) > 1 else None
                                        new_goal_positions = step_result[2] if len(step_result) > 2 else None
                                        new_goal_radii = step_result[3] if len(step_result) > 3 else None
                                        new_done_mask = step_result[4] if len(step_result) > 4 else None
                                    else:
                                        new_agents_state = step_result
                                        new_agents_path_plans = None
                                        new_goal_positions = None
                                        new_goal_radii = None
                                        new_done_mask = None
                                    if new_agents_state is not None:
                                        self.agents_state = new_agents_state
                                    if new_agents_path_plans is not None:
                                        self.agents_path_plans = new_agents_path_plans
                                    if new_goal_positions is not None:
                                        self.goal_positions = new_goal_positions
                                    if new_goal_radii is not None:
                                        self.goal_radii = new_goal_radii
                                    if new_done_mask is not None:
                                        self.done_mask = new_done_mask
                                    active_mask = self.agents_state[..., 6] > 0.5
                                    active_agents = torch.nonzero(active_mask[self.batch_idx], as_tuple=False).squeeze(-1)
                                    if active_agents.numel() > 0:
                                        self.active_agents_list = active_agents.tolist()
                                        self.cur_idx %= len(self.active_agents_list)
                                    else:
                                        print("警告：Step 后没有 active 车辆。")
                            except Exception as e:
                                print(f"执行 step_callback 失败: {e}")
                                import traceback
                                traceback.print_exc()
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
            goal_radius_value = None
            if self.goal_radii is not None:
                try:
                    goal_radius_value = float(self.goal_radii[self.batch_idx, m].detach().cpu().item())
                except Exception:
                    try:
                        goal_radius_value = float(self.goal_radii[self.batch_idx, m])
                    except Exception:
                        goal_radius_value = None
            
            # 获取当前车辆状态
            ego_state = self.agents_state[b, m].cpu().numpy()
            ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[2]
            ego_length, ego_width = ego_state[4], ego_state[5]
            
            # 获取观测数据（如果提供了回调函数）
            has_observation = False
            neighbors_local = None
            w_lanes_local = None
            w_boundaries_local = None
            
            if self.observation_callback is not None:
                try:
                    neighbors_local, w_lanes_local, w_boundaries_local = \
                        self.observation_callback(self.agents_state, b, m)
                    has_observation = True
                except Exception as e:
                    print(f"获取 observation 失败: {e}")
            info_lines = None
            if self.info_callback is not None:
                try:
                    info_lines = self._format_info_lines(
                        self.info_callback(
                            self.agents_state,
                            self.goal_positions,
                            self.goal_radii,
                            self.done_mask,
                            b,
                            m,
                        )
                    )
                except Exception as e:
                    print(f"获取 info 失败: {e}")
            
            # 更新窗口标题
            pygame.display.set_caption(
                f'Active Vehicles Path Plans - B={b}, M={m} ({self.cur_idx+1}/{len(self.active_agents_list)}) '
                 f'(SPACE: next, W: step, ESC: quit, Scroll: zoom) - Path: {len(valid_path)} pts, Zoom: {self.zoom_level:.2f}x'
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
            
            # 绘制 horizon 观测范围及目标半径
            self.draw_horizon_box(ego_x, ego_y, self.horizon, goal_radius_value)
            
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
            
            if info_lines:
                self.draw_info_panel(info_lines)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print(f"可视化结束，共查看了 {len(self.active_agents_list)} 个active车辆")
        return exit_reason


def visualize_path_planning(
    agents_state: torch.Tensor,
    agents_path_plans: torch.Tensor,
    quads_vertices: torch.Tensor,
    batch_idx: int = 0,
    invalid_marker_value: float = -999999.0,
    horizon: float = 80.0,
    observation_callback: Optional[Callable] = None,
    step_callback: Optional[Callable[[], Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]]] = None,
    info_callback: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int, int], Optional[object]]] = None,
    agents_start_quad_ids: Optional[torch.Tensor] = None,
    agents_goal_quad_ids: Optional[torch.Tensor] = None,
    goal_positions: Optional[torch.Tensor] = None,
    goal_radii: Optional[torch.Tensor] = None,
    done_mask: Optional[torch.Tensor] = None):
    """
    便捷函数：可视化路径规划（通用版本）
    
    Args:
        agents_state: 车辆状态张量 [B, M, 7] - (x, y, heading, speed, length, width, active)
        agents_path_plans: 路径规划张量 [B, M, N, 3] - (x, y, angle)
        quads_vertices: 道路网格顶点 [num_quads, 4, 2] - 每个quad的4个顶点坐标
        batch_idx: 要可视化的批次索引（默认0）
        invalid_marker_value: 无效路径点的标记值（默认-999999.0）
        horizon: 观测范围半径（默认80.0）
        observation_callback: 可选的观测数据获取回调函数
        info_callback: 可选的信息展示回调函数
        agents_start_quad_ids: 可选的起始quad ID
        agents_goal_quad_ids: 可选的目标quad ID
    """
    if not PYGAME_AVAILABLE:
        print("错误：请安装 pygame 和 PyOpenGL: pip install pygame PyOpenGL")
        return 'quit'
    
    visualizer = PathPlanningVisualizer(
        agents_state=agents_state,
        agents_path_plans=agents_path_plans,
        quads_vertices=quads_vertices,
        batch_idx=batch_idx,
        invalid_marker_value=invalid_marker_value,
        horizon=horizon,
        observation_callback=observation_callback,
        step_callback=step_callback,
        info_callback=info_callback,
        agents_start_quad_ids=agents_start_quad_ids,
        agents_goal_quad_ids=agents_goal_quad_ids,
        goal_positions=goal_positions,
        goal_radii=goal_radii,
        done_mask=done_mask
    )
    return visualizer.run()

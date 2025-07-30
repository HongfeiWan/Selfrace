import pygame
import math
import numpy as np
import torch
import yaml
import os
import sys
from typing import Dict, List, Tuple
import random

# 添加simulator目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
# 添加utils目录到路径
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from reward import RewardCalculator
from randomize_components import RewardParameterSampler
from road import RoadNetwork
from offroad import OffroadChecker
from spatial_hash import SpatialHash
from dynamics import KinematicBicycleModel, DiscreteActionSpace

# 每个step按0.3s计算运动学以及reward，显示按一秒60step(帧数)显示
class Car:
    """汽车类 - 使用KinematicBicycleModel的step函数"""
    def __init__(self, x: float, y: float, heading: float = 0.0, device: torch.device = None, dynamics_model: KinematicBicycleModel = None):
        self.x = x
        self.y = y
        self.heading = heading  # 弧度
        self.speed = 0.0
        self.acceleration = 0.0
        self.steering = 0.0
        self.length = 4.5
        self.width = 2.0
        self.max_speed = 20.0
        self.max_acceleration = 2.5
        self.max_steering = math.radians(35.0)
        # 历史数据用于计算jerk
        self.prev_acceleration = 0.0
        self.prev_steering = 0.0
        # 初始化离散动作空间
        self.device = device if device is not None else torch.device('cpu')
        self.discrete_action_space = DiscreteActionSpace(self.device, config={})
        # 当前动作索引
        self.current_action = 7  # 默认动作（停止）
        # 动力学模型
        self.dynamics_model = dynamics_model
        # 状态张量 [x, y, yaw, speed]
        self.state_tensor = torch.tensor([[x, y, heading, self.speed]], device=self.device)

    def update(self, dt: float, action_idx: int):
        """使用KinematicBicycleModel的step函数更新汽车状态"""
        if self.dynamics_model is not None:
            # 使用动力学模型的step函数
            action_tensor = torch.tensor([action_idx], device=self.device)
            new_state = self.dynamics_model.step(self.state_tensor, action_tensor, dt)
            # 更新状态
            self.x = new_state[0, 0].item()
            self.y = new_state[0, 1].item()
            self.heading = new_state[0, 2].item()
            self.speed = new_state[0, 3].item()
            # 更新状态张量
            self.state_tensor = new_state
            # 保存当前动作
            self.current_action = action_idx
    
    def get_state(self) -> np.ndarray:
        """获取汽车状态向量"""
        # 计算jerk
        along_jerk = (self.acceleration - self.prev_acceleration) / 0.3  # 假设dt=0.3
        alat_jerk = (self.steering - self.prev_steering) / 0.3
        # 计算横向加速度
        alat = self.speed * self.speed * math.tan(self.steering) / self.length
        # 计算Frenet坐标系信息（简化版本）
        theta_f = 0.0  # 车道角度误差，这里简化处理
        d = 0.0  # 横向距离，这里简化处理
        return np.array([
            self.x, self.y,           # 位置
            self.heading, self.speed,  # 运动学信息
            self.acceleration, alat,   # 动力学信息
            along_jerk, alat_jerk,     # jerk信息
            theta_f, d                 # Frenet坐标系信息
        ])
    
    def get_offroad_state(self) -> torch.Tensor:
        """获取用于离路检测的状态张量"""
        # OffroadChecker需要的状态格式: [x, y, heading, length, width]
        return torch.tensor([[self.x, self.y, self.heading, self.length, self.width]], 
                           dtype=torch.float32)

class Goal:
    """目标点类"""
    def __init__(self, x: float, y: float, radius: float = 10.0):
        self.x = x
        self.y = y
        self.radius = radius
    def is_reached(self, x: float, y: float) -> bool:
        """检查是否到达目标"""
        distance = math.sqrt((x - self.x)**2 + (y - self.y)**2)
        return distance < self.radius

class CarGame:
    """汽车游戏主类"""
    def __init__(self, config_path: str = None):
        # 初始化pygame
        pygame.init()
        
        # 游戏设置
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("汽车驾驶游戏 - GIGAFLOW奖励系统")
        
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 初始化奖励计算器
        self.device = torch.device('cpu')
        self.reward_calculator = RewardCalculator(self.config, self.device)
        
        # 初始化动力学模型
        self.dynamics_model = KinematicBicycleModel(self.config, self.device)
        
        # 初始化RoadNetwork
        self.road_network = self._initialize_road_network()
        
        # 初始化游戏对象
        # 移除self.road的初始化，因为现在完全依赖RoadNetwork
        
        # 初始化汽车位置
        self.player_car = self._initialize_car_position()
        self.goal = self._initialize_goal_position()
        
        # 初始化OffroadChecker
        # 创建SpatialHash实例
        # 获取道路网络的边界
        if hasattr(self.road_network, 'quads_vertices') and self.road_network.quads_vertices.shape[0] > 0:
            # 先展平所有维度，然后计算最小值和最大值
            flattened_vertices = self.road_network.quads_vertices.view(-1, 2)
            min_bounds = flattened_vertices.min(dim=0).values
            max_bounds = flattened_vertices.max(dim=0).values
        else:
            # 如果没有道路数据，使用默认边界
            min_bounds = torch.tensor([-1000, -1000], device=self.device)
            max_bounds = torch.tensor([1000, 1000], device=self.device)
        
        # 创建SpatialHash
        cell_size = 10.0  # 10米的网格单元
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)
        
        # 创建OffroadChecker
        self.offroad_checker = OffroadChecker(self.road_network, self.spatial_hash)
        
        # 游戏状态
        self.clock = pygame.time.Clock()
        self.dt = 0.3  # 时间步长（每个step对应0.3秒）
        self.running = True
        self.score = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.max_steps = 1000
        
        # 碰撞检测
        self.collision_occurred = False
        self.off_road = False
        
        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # 颜色定义
        self.colors = {
            'road': (100, 100, 100),
            'grass': (34, 139, 34),
            'car': (255, 0, 0),
            'goal': (0, 255, 0),
            'text': (255, 255, 255),
            'lane_markings': (255, 255, 255)
        }

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"成功加载配置文件: {config_path}")
            return config
        except FileNotFoundError:
            print(f"警告: 配置文件 {config_path} 未找到，使用默认配置")
            return {'reward': {}}
        except yaml.YAMLError as e:
            print(f"错误: 解析YAML文件时出错: {e}")
            return {'reward': {}}
    
    def _initialize_road_network(self) -> RoadNetwork:
        """初始化RoadNetwork"""
        # 从配置中获取地图路径
        map_path = self.config.get('simulator', {}).get('map_path')
        if not map_path:
            raise ValueError("配置文件中未找到simulator.map_path字段")
        
        # 构建完整路径
        if os.path.isabs(map_path):
            map_file_path_full = map_path
        else:
            # 相对于项目根目录的路径
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            map_file_path_full = os.path.normpath(os.path.join(project_root, map_path.lstrip('./')))
        
        if not os.path.exists(map_file_path_full):
            raise FileNotFoundError(f"地图文件不存在: {map_file_path_full}")
        
        # 初始化RoadNetwork
        road_network = RoadNetwork(map_path=map_file_path_full, device=self.device)
        print(f"成功加载RoadNetwork，地图文件: {map_file_path_full}")
        print(f"道路边界点数量: {road_network.global_w_boundary_points.shape[0]}")
        return road_network
    
    def handle_events(self):
        """处理游戏事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def get_random_action(self) -> int:
        """随机从12个离散动作空间中抽样动作"""
        # 12个离散动作：
        # 0-2: 最大减速 + 左转/直行/右转
        # 3-5: 减速 + 左转/直行/右转  
        # 6-8: 直行 + 左转/直行/右转
        # 9-11: 加速 + 左转/直行/右转
        return random.randint(0, 11)
    
    def update_game_state(self, action_idx: int):
        """更新游戏状态"""
        # 更新汽车状态
        self.player_car.update(self.dt, action_idx)
        
        # 检查碰撞和离路
        self.check_collisions()
        self.check_off_road()
        
        # 检查是否到达目标
        goal_reached = self.goal.is_reached(self.player_car.x, self.player_car.y)
        
        # 计算奖励
        reward = self.calculate_reward(goal_reached)
        self.episode_reward += reward
        self.score += reward
        
        # 更新步数
        self.step_count += 1
        
        # 检查episode是否结束
        if goal_reached or self.step_count >= self.max_steps:
            self.reset_episode()
    
    def check_collisions(self):
        """检查碰撞（简化版本）"""
        # 这里简化处理，实际游戏中需要更复杂的碰撞检测
        self.collision_occurred = False
    
    def check_off_road(self):
        """检查是否离路"""
        # 使用OffroadChecker进行精确的离路检测
        car_state = self.player_car.get_offroad_state()
        # 检查是否离路（返回True表示离路）
        is_offroad = self.offroad_checker.check_offroad(car_state)
        self.off_road = is_offroad[0].item()  # 取第一个（也是唯一）结果
    
    def _initialize_car_position(self) -> Car:
        """根据道路网络初始化汽车位置在quad上"""
        # 获取所有quad的中心点
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
        # 随机选择一个quad作为起始位置
        quad_idx = random.randint(0, quad_centers.shape[0] - 1)
        start_point = quad_centers[quad_idx]
        # 获取该quad的方向向量
        quad_direction = self.road_network.quad_directions[quad_idx]
        heading = math.atan2(quad_direction[1].item(), quad_direction[0].item())
        return Car(start_point[0].item(), start_point[1].item(), heading, self.device, self.dynamics_model)
    
    def _initialize_goal_position(self) -> Goal:
        """根据道路网络初始化目标位置在quad上"""
        # 获取所有quad的中心点
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
        # 随机选择一个quad作为目标位置（与汽车位置不同）
        available_quads = list(range(quad_centers.shape[0]))
        if hasattr(self, 'player_car') and self.player_car is not None:
            # 找到汽车所在的quad
            car_pos = torch.tensor([[self.player_car.x, self.player_car.y]], 
                                 dtype=torch.float32, device=self.device)
            distances = torch.norm(quad_centers.unsqueeze(0) - car_pos.unsqueeze(1), dim=2)
            car_quad_idx = torch.argmin(distances).item()
            # 排除汽车所在的quad
            if car_quad_idx in available_quads:
                available_quads.remove(car_quad_idx)
        if available_quads:
            goal_quad_idx = random.choice(available_quads)
            goal_point = quad_centers[goal_quad_idx]
            return Goal(goal_point[0].item(), goal_point[1].item())
        else:
            # 如果没有可用的quad，选择最远的quad
            goal_quad_idx = random.randint(0, quad_centers.shape[0] - 1)
            goal_point = quad_centers[goal_quad_idx]
            return Goal(goal_point[0].item(), goal_point[1].item())

    def get_boundary_points_info(self) -> str:
        """获取边界点信息"""
        boundary_points = self.road_network.global_w_boundary_points
        if boundary_points.shape[0] == 0:
            return "无边界点数据"
        
        # 计算边界点的坐标范围
        min_x = torch.min(boundary_points[:, 0]).item()
        max_x = torch.max(boundary_points[:, 0]).item()
        min_y = torch.min(boundary_points[:, 1]).item()
        max_y = torch.max(boundary_points[:, 1]).item()
        
        return f"边界点数量: {boundary_points.shape[0]}, 范围: X[{min_x:.1f},{max_x:.1f}] Y[{min_y:.1f},{max_y:.1f}]"
    
    def calculate_reward(self, goal_reached: bool) -> float:
        """计算奖励"""
        # 准备状态张量
        car_state = self.player_car.get_state()
        
        # 转换为torch张量
        agents_state = torch.tensor([car_state], dtype=torch.float32).unsqueeze(0)  # (1, 1, 10)
        
        # 准备其他输入
        all_collisions = torch.tensor([[self.collision_occurred]], dtype=torch.bool)
        offroad_mask = torch.tensor([[self.off_road]], dtype=torch.bool)
        goal_positions = torch.tensor([[[self.goal.x, self.goal.y]]], dtype=torch.float32)
        waypoint_reached = torch.tensor([[goal_reached]], dtype=torch.bool)
        stop_line_violation = torch.tensor([[False]], dtype=torch.bool)  # 简化处理
        
        # 计算奖励
        reward, goal_reached_tensor = self.reward_calculator.calculate(
            agents_state=agents_state,
            all_collisions=all_collisions,
            offroad_mask=offroad_mask,
            dt=self.dt,
            goal_positions=goal_positions,
            waypoint_reached=waypoint_reached,
            stop_line_violation=stop_line_violation
        )
        
        return reward[0, 0].item()
    
    def reset_episode(self):
        """重置episode"""
        # 重置汽车位置
        self.player_car = self._initialize_car_position()
        self.goal = self._initialize_goal_position()
        
        # 重置游戏状态
        self.collision_occurred = False
        self.off_road = False
        self.step_count = 0
        
        # 重新采样奖励参数
        self.reward_calculator.reset_episode()
        
        print(f"Episode重置，总奖励: {self.episode_reward:.4f}")
        self.episode_reward = 0.0
    
    def draw(self):
        """绘制游戏画面"""
        # 清屏
        self.screen.fill(self.colors['grass'])
        
        # 绘制道路
        self.draw_road_from_network()
        
        # 绘制目标
        # 将目标位置转换为屏幕坐标
        goal_screen_pos = self.convert_world_to_screen(torch.tensor([[self.goal.x, self.goal.y]]))
        if goal_screen_pos:
            pygame.draw.circle(self.screen, self.colors['goal'], 
                             goal_screen_pos[0], int(self.goal.radius))
        else:
            # 如果目标不在视野范围内，在屏幕边缘显示一个指示器
            self.draw_goal_indicator()
        
        # 绘制汽车
        self.draw_car(self.player_car)
        
        # 绘制UI
        self.draw_ui()
        
        # 更新显示
        pygame.display.flip()
    
    def draw_road_from_network(self):
        """使用RoadNetwork数据绘制道路（带视野限制）"""
        # 获取汽车视野范围内的边界点
        visible_boundary_points = self.get_visible_boundary_points()
        if visible_boundary_points.shape[0] == 0:
            # 如果没有边界点，不绘制任何内容
            return
        # 将边界点转换为屏幕坐标
        screen_points = self.convert_world_to_screen(visible_boundary_points)

        # 绘制边界点作为散点
        self.draw_boundary_points(screen_points)
        
        # 绘制道路中心线
        self.draw_road_centerlines()
    
    def draw_road_centerlines(self):
        """绘制道路中心线（绿色小箭头）"""
        try:
            # 获取所有quad的中心线
            quad_centerlines = self.road_network.quad_centerlines  # (num_quads, 2, 2)
            quad_directions = self.road_network.quad_directions    # (num_quads, 2)
            
            if quad_centerlines.shape[0] == 0:
                return
            
            # 获取汽车位置用于视野筛选
            car_pos = torch.tensor([[self.player_car.x, self.player_car.y]], 
                                 dtype=torch.float32, device=self.device)
            
            # 视野范围（米）
            vision_radius = 100.0
            
            # 计算每个quad中心点到汽车的距离
            quad_centers = quad_centerlines.mean(dim=1)  # (num_quads, 2)
            diff = quad_centers - car_pos.squeeze(0)
            distances_squared = torch.sum(diff * diff, dim=1)
            visible_mask = distances_squared <= (vision_radius * vision_radius)
            
            # 获取可见的quad
            visible_centerlines = quad_centerlines[visible_mask]
            visible_directions = quad_directions[visible_mask]
            
            # 绘制中心线箭头
            arrow_color = (0, 255, 0)  # 绿色
            arrow_length = 5  # 箭头长度（像素）
            arrow_width = 1   # 箭头宽度（像素）
            
            # 稀疏绘制箭头：每隔几个quad绘制一个箭头
            arrow_spacing =30  # 每隔3个quad绘制一个箭头
            
            for i in range(0, visible_centerlines.shape[0], arrow_spacing):
                # 获取当前quad的中心线（两个端点）
                centerline = visible_centerlines[i]  # (2, 2)
                direction = visible_directions[i]    # (2,)
                
                # 将中心线端点转换为屏幕坐标
                screen_points = self.convert_world_to_screen(centerline)
                
                if len(screen_points) >= 2:
                    # 绘制中心线线段
                    start_point = screen_points[0]
                    end_point = screen_points[1]
                    pygame.draw.line(self.screen, arrow_color, start_point, end_point, 2)
                    
                    # 在中心点绘制方向箭头
                    center_x = (start_point[0] + end_point[0]) // 2
                    center_y = (start_point[1] + end_point[1]) // 2
                    
                    # 计算箭头方向
                    arrow_angle = math.atan2(direction[1].item(), direction[0].item())
                    
                    # 绘制箭头
                    arrow_end_x = center_x + arrow_length * math.cos(arrow_angle)
                    arrow_end_y = center_y + arrow_length * math.sin(arrow_angle)
                    
                    # 绘制箭头头部
                    head_angle1 = arrow_angle + math.pi * 0.75
                    head_angle2 = arrow_angle - math.pi * 0.75
                    head_length = 10
                    
                    head1_x = arrow_end_x - head_length * math.cos(head_angle1)
                    head1_y = arrow_end_y - head_length * math.sin(head_angle1)
                    head2_x = arrow_end_x - head_length * math.cos(head_angle2)
                    head2_y = arrow_end_y - head_length * math.sin(head_angle2)
                    
                    pygame.draw.line(self.screen, arrow_color,
                                   (int(arrow_end_x), int(arrow_end_y)),
                                   (int(head1_x), int(head1_y)), arrow_width)
                    pygame.draw.line(self.screen, arrow_color,
                                   (int(arrow_end_x), int(arrow_end_y)),
                                   (int(head2_x), int(head2_y)), arrow_width)
            
        except Exception as e:
            print(f"警告: 绘制道路中心线失败: {e}")
    
    def get_visible_boundary_points(self) -> torch.Tensor:
        """获取汽车视野范围内的边界点（优化版本）"""
        try:
            # 缓存视野点，避免每帧重复计算
            if not hasattr(self, '_cached_visible_points') or not hasattr(self, '_last_car_pos'):
                self._cached_visible_points = None
                self._last_car_pos = None
            # 检查汽车位置是否发生显著变化
            current_car_pos = (self.player_car.x, self.player_car.y)
            if (self._last_car_pos is not None and 
                abs(current_car_pos[0] - self._last_car_pos[0]) < 5.0 and 
                abs(current_car_pos[1] - self._last_car_pos[1]) < 5.0 and
                self._cached_visible_points is not None):
                return self._cached_visible_points
            # 获取所有边界点
            all_boundary_points = self.road_network.global_w_boundary_points
            if all_boundary_points.shape[0] == 0:
                return all_boundary_points

            # 使用更高效的距离计算
            car_pos = torch.tensor([[self.player_car.x, self.player_car.y]], 
                                 dtype=torch.float32, device=self.device)
            
            # 视野范围（米）
            vision_radius = 100.0
            
            # 使用更高效的距离计算和筛选
            diff = all_boundary_points - car_pos.squeeze(0)
            distances_squared = torch.sum(diff * diff, dim=1)
            visible_mask = distances_squared <= (vision_radius * vision_radius)
            visible_points = all_boundary_points[visible_mask]
            
            # 缓存结果
            self._cached_visible_points = visible_points
            self._last_car_pos = current_car_pos
            return visible_points
        
        except Exception as e:
            print(f"警告: 获取视野边界点失败: {e}")
            return torch.empty((0, 2), device=self.device)
    
    def draw_goal_indicator(self):
        """在屏幕边缘绘制目标指示器"""
        try:
            # 计算目标相对于汽车的方向
            dx = self.goal.x - self.player_car.x
            dy = self.goal.y - self.player_car.y
            
            # 计算方向角度
            angle = math.atan2(dy, dx)
            
            # 在屏幕边缘绘制指示器
            screen_center_x = self.width // 2
            screen_center_y = self.height // 2
            indicator_radius = min(self.width, self.height) // 2 - 30
            
            indicator_x = screen_center_x + int(indicator_radius * math.cos(angle))
            indicator_y = screen_center_y + int(indicator_radius * math.sin(angle))
            
            # 绘制指示器
            pygame.draw.circle(self.screen, (255, 255, 0), (indicator_x, indicator_y), 5)
            
        except Exception as e:
            print(f"警告: 绘制目标指示器失败: {e}")
    
    def draw_boundary_points(self, screen_points: List[Tuple[int, int]]):
        """绘制边界点作为散点（优化版本）"""
        if len(screen_points) == 0:
            return
        # 限制绘制的点数量以提高性能
        max_points = 500
        if len(screen_points) > max_points:
            # 均匀采样点
            step = len(screen_points) // max_points
            screen_points = screen_points[::step]
        
        # 绘制每个边界点
        point_radius = 1  # 点的大小
        point_color = (255, 255, 255)  # 白色点
    
        for point in screen_points:
            pygame.draw.circle(self.screen, point_color, point, point_radius)
        
    
    def convert_world_to_screen(self, world_points: torch.Tensor) -> List[Tuple[int, int]]:
        """将世界坐标转换为屏幕坐标（基于汽车视野，优化版本）"""
        if world_points.shape[0] == 0:
            return []
        
        # 缓存转换参数，避免重复计算
        if not hasattr(self, '_cached_scale_params') or not hasattr(self, '_last_scale_car_pos'):
            self._cached_scale_params = None
            self._last_scale_car_pos = None
        
        # 检查汽车位置是否发生显著变化
        current_car_pos = (self.player_car.x, self.player_car.y)
        if (self._last_scale_car_pos is not None and 
            abs(current_car_pos[0] - self._last_scale_car_pos[0]) < 5.0 and 
            abs(current_car_pos[1] - self._last_scale_car_pos[1]) < 5.0 and
            self._cached_scale_params is not None):
            scale_x, scale_y, car_x, car_y = self._cached_scale_params
        else:
            # 获取汽车位置作为视野中心
            car_x = self.player_car.x
            car_y = self.player_car.y
            
            # 视野范围（米）
            vision_radius = 100.0
            
            # 计算缩放比例，留出边距
            margin = 50
            scale_x = (self.width - 2 * margin) / (2 * vision_radius)
            scale_y = (self.height - 2 * margin) / (2 * vision_radius)
            # 使用较小的缩放比例以保持宽高比
            scale = min(scale_x, scale_y)
            scale_x = scale_y = scale
            # 缓存参数
            self._cached_scale_params = (scale_x, scale_y, car_x, car_y)
            self._last_scale_car_pos = current_car_pos
        
        # 转换坐标（使用向量化操作）
        screen_points = []
        for point in world_points:
            # 将世界坐标相对于汽车位置进行偏移
            relative_x = point[0].item() - car_x
            relative_y = point[1].item() - car_y
            # 转换为屏幕坐标
            screen_x = int(relative_x * scale_x + self.width // 2)
            screen_y = int(relative_y * scale_y + self.height // 2)
            # 检查是否在屏幕范围内
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                screen_points.append((screen_x, screen_y))
        return screen_points
    
    def draw_car(self, car: Car):
        """绘制汽车"""
        # 汽车在视野中心显示
        screen_x = self.width // 2
        screen_y = self.height // 2
        
        # 绘制汽车（方形版本）
        car_size = 16  # 方形边长
        car_rect = pygame.Rect(screen_x - car_size // 2, screen_y - car_size // 2, car_size, car_size)
        pygame.draw.rect(self.screen, self.colors['car'], car_rect)
        
        # 绘制汽车朝向指示器（从中心到前方的线）
        front_x = screen_x + car_size // 2 * math.cos(car.heading)
        front_y = screen_y + car_size // 2 * math.sin(car.heading)
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (screen_x, screen_y), 
                        (int(front_x), int(front_y)), 3)
    
    def draw_ui(self):
        """绘制用户界面"""
        # 显示奖励和步数
        reward_text = self.font.render(f"episode reward: {self.episode_reward:.4f}", True, self.colors['text'])
        step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
        
        self.screen.blit(reward_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        
        # 显示汽车状态
        speed_text = self.small_font.render(f"speed: {self.player_car.speed:.2f} m/s", True, self.colors['text'])
        accel_text = self.small_font.render(f"acceleration: {self.player_car.acceleration:.2f} m/s²", True, self.colors['text'])
        heading_text = self.small_font.render(f"heading: {math.degrees(self.player_car.heading):.1f}°", True, self.colors['text'])
        
        # 显示离散动作信息
        action_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        current_action = action_names[self.player_car.current_action]
        action_text = self.small_font.render(f"current action: {current_action} (idx: {self.player_car.current_action})", True, self.colors['text'])
        
        self.screen.blit(speed_text, (10, 90))
        self.screen.blit(accel_text, (10, 110))
        self.screen.blit(heading_text, (10, 130))
        self.screen.blit(action_text, (10, 150))
        
        # 显示汽车世界坐标
        car_world_pos = f"car world pos: ({self.player_car.x:.1f}, {self.player_car.y:.1f})"
        car_pos_text = self.small_font.render(car_world_pos, True, self.colors['text'])
        self.screen.blit(car_pos_text, (10, 190))
        
        # 显示目标世界坐标
        goal_world_pos = f"goal world pos: ({self.goal.x:.1f}, {self.goal.y:.1f})"
        goal_pos_text = self.small_font.render(goal_world_pos, True, self.colors['text'])
        self.screen.blit(goal_pos_text, (10, 210))
        
        # 显示视野信息
        vision_info = f"vision radius: 100m, visible points: {len(self.get_visible_boundary_points())}"
        vision_text = self.small_font.render(vision_info, True, self.colors['text'])
        self.screen.blit(vision_text, (10, 230))
        
        # 显示游戏状态
        status_text = ""
        if self.collision_occurred:
            status_text = "collision!"
        elif self.off_road:
            status_text = "off road!"
        elif self.goal.is_reached(self.player_car.x, self.player_car.y):
            status_text = "goal reached!"
        if status_text:
            status_surface = self.font.render(status_text, True, (255, 0, 0))
            self.screen.blit(status_surface, (10, 260))
        

    def run(self):
        """运行游戏主循环"""
        print("游戏开始！")
        print("使用随机动作控制汽车：")
        print("每个step随机从12个离散动作中抽样")
        print("按R重置episode，按ESC退出")
        while self.running:
            # 处理事件
            self.handle_events()
            # 获取随机动作
            action_idx = self.get_random_action()
            # 更新游戏状态
            self.update_game_state(action_idx)
            # 绘制画面
            self.draw()
            # 控制帧率
            self.clock.tick(60)
        pygame.quit()
        print("游戏结束！")

def main():
    """主函数"""
    # 创建游戏实例
    game = CarGame()
    # 运行游戏
    game.run()

if __name__ == "__main__":
    main()

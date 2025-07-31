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
# 添加training目录到路径
training_dir = os.path.join(parent_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from reward import RewardCalculator
from road import RoadNetwork
from offroad import OffroadChecker
from spatial_hash import SpatialHash
from dynamics import KinematicBicycleModel, DiscreteActionSpace
from network import FeatureEncoder,SharedNetwork
from collision import CollisionChecker
from world_init import WorldInitializer

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
    
    def get_state(self, action_idx: int = None) -> np.ndarray:
        """获取汽车状态向量"""
        # 如果提供了action_idx，通过离散动作空间查询jerk值
        # 获取当前动作对应的jerk值
        action_tensor = torch.tensor([action_idx], device=self.device)
        jerk_actions = self.dynamics_model.discrete_action_space.get_action(action_tensor)
        along_jerk = jerk_actions[0, 0].item()  # 纵向jerk
        alat_jerk = jerk_actions[0, 1].item()   # 横向jerk
        current_along = self.dynamics_model.current_along[0].item()
        current_alat = self.dynamics_model.current_alat[0].item()

        # 计算Frenet坐标系信息（简化版本）
        theta_f = 0.0  # 车道角度误差，这里简化处理
        d = 0.0  # 横向距离，这里简化处理
        return np.array([
            self.x, self.y,           # 位置
            self.heading, self.speed,  # 运动学信息
            current_along, current_alat,   # 动力学信息（使用动力学模型的值）
            along_jerk, alat_jerk,     # jerk信息
            theta_f, d                 # Frenet坐标系信息
        ])
    
    def get_offroad_state(self) -> torch.Tensor:
        """获取用于离路检测的状态张量"""
        # OffroadChecker需要的状态格式: [x, y, heading, length, width]
        return torch.tensor([[self.x, self.y, self.heading, self.length, self.width]], 
                           dtype=torch.float32)

class CarGame:
    def __init__(self, config_path: str = None, training_mode: bool = False):
        # 训练模式标志
        self.training_mode = training_mode
        # 加载配置
        self.config = self.load_config(config_path)
        # 从配置中获取车辆数量
        self.num_cars = self.config.get('simulator', {}).get('num_npc_vehicles', 10)
        if not training_mode:
            # 初始化pygame
            pygame.init()
            # 游戏设置
            self.width = 1200
            self.height = 800
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("汽车驾驶游戏 - GIGAFLOW奖励系统")
        else:
            # 训练模式下不初始化pygame
            self.width = 1200
            self.height = 800
            self.screen = None
        
        # 初始化奖励计算器
        self.device = torch.device('cpu')
        self.reward_calculator = RewardCalculator(self.config, self.device)
        
        # 初始化动力学模型
        self.dynamics_model = KinematicBicycleModel(self.config, self.device)
        
        # 初始化RoadNetwork
        self.road_network = self._initialize_road_network()
        
        # 初始化OffroadChecker和CollisionChecker
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
        cell_size = 20.0  # 20米的网格单元
        self.spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, self.device)
        
        # 创建OffroadChecker和CollisionChecker
        self.offroad_checker = OffroadChecker(self.road_network, self.spatial_hash)
        self.collision_checker = CollisionChecker(self.config, self.spatial_hash)
        
        # 初始化WorldInitializer
        # 动态设置num_npc_vehicles以匹配请求的车辆数量
        if 'simulator' not in self.config:
            self.config['simulator'] = {}
        self.config['simulator']['num_npc_vehicles'] = self.num_cars
        
        self.world_initializer = WorldInitializer(self.road_network, self.offroad_checker, self.collision_checker, self.config)
        
        # 使用WorldInitializer初始化车辆
        self.initialize_vehicles()
        
        # 为每个agent分配独立的目标和状态跟踪
        self.agent_goals = []
        self.agent_collisions = []
        self.agent_off_road = []
        self.agent_rewards = []
        self.agent_goal_reached = []  # 新增：跟踪每个agent的目标达成状态
        self.current_step_rewards = []  # 新增：跟踪当前步骤的奖励
        
        # 初始化每个agent的目标和状态
        actual_num_cars = len(self.cars)
        for i in range(actual_num_cars):
            self.agent_goals.append(self._initialize_agent_goal())
            self.agent_collisions.append(False)
            self.agent_off_road.append(False)
            self.agent_rewards.append(0.0)
            self.agent_goal_reached.append(False)  # 新增：初始化目标达成状态
            self.current_step_rewards.append(0.0)  # 新增：初始化当前步骤奖励
        
        # 保持玩家车辆的目标作为主目标（用于UI显示）
        if self.agent_goals:
            self.goal = self.agent_goals[0]
        
        # 游戏状态
        if not training_mode:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None
        self.dt = 0.3  # 时间步长（每个step对应0.3秒）
        self.running = True
        self.score = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.max_steps = 1000
        
        # 碰撞检测
        self.collision_occurred = False
        self.off_road = False
        
        # 字体（仅在非训练模式下初始化）
        if not training_mode:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        else:
            self.font = None
            self.small_font = None

        # 颜色定义
        self.colors = {
            'road': (100, 100, 100),
            'grass': (34, 139, 34),
            'car': (255, 0, 0),
            'other_car': (0, 0, 255),  # 蓝色表示其他车辆
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
    
    def initialize_vehicles(self):
        """使用WorldInitializer初始化车辆"""
        # 使用WorldInitializer生成无碰撞的车辆状态
        # 这里我们只生成一个环境，但WorldInitializer支持批量生成
        agents_state, ego_agents_idx, agents_start_quad_ids = self.world_initializer.initialize_world(num_envs=1)
        
        # 从生成的状态中创建Car对象
        self.cars = []
        env_states = agents_state[0]  # 获取第一个环境的状态 (max_agents, 7)
        
        # 只处理激活的车辆
        active_mask = env_states[:, 6] == 1.0
        active_states = env_states[active_mask]
        
        # 确保至少有num_cars辆车
        num_active = active_states.shape[0]
        if num_active < self.num_cars:
            print(f"警告: WorldInitializer只生成了{num_active}辆车，少于请求的{self.num_cars}辆")
            # 如果车辆不够，使用现有的车辆
            self.num_cars = num_active
        
        # 创建Car对象
        for i in range(min(self.num_cars, num_active)):
            state = active_states[i]
            x, y, yaw, speed, length, width, active = state.cpu().numpy()
            
            # 创建Car对象
            car = Car(x, y, yaw, self.device, self.dynamics_model)
            car.speed = speed
            car.length = length
            car.width = width
            car.current_action = 7  # 默认停止动作
            self.cars.append(car)
        # 设置玩家车辆
        self.player_car = self.cars[0] if self.cars else None
        print(f"成功初始化{len(self.cars)}辆车")
    
    def _initialize_agent_goal(self) -> Dict:
        """统一的agent目标初始化方法"""
        # 获取所有quad的中心点
        quad_centers = self.road_network.quad_centerlines.mean(dim=1)  # (num_quads, 2)
        
        # 随机选择一个quad作为目标位置
        goal_quad_idx = random.randint(0, quad_centers.shape[0] - 1)
        goal_point = quad_centers[goal_quad_idx]
        
        return {'x': goal_point[0].item(), 'y': goal_point[1].item()}
    
    def get_agent_random_action(self) -> int:
        """统一的agent随机动作方法"""
        return random.randint(0, 11)  # 12个离散动作
    
    def get_random_action(self) -> int:
        """随机从12个离散动作空间中抽样动作"""
        return self.get_agent_random_action()
    
    def update_game_state(self, action_idx: int):
        """更新游戏状态"""
        # 更新所有车辆状态
        for i, car in enumerate(self.cars):
            if i == 0:
                # 玩家车辆使用传入的动作
                car.update(self.dt, action_idx)
            else:
                # 其他车辆使用统一的随机动作方法
                random_action = self.get_agent_random_action()
                car.update(self.dt, random_action)
        
        # 检查所有车辆的碰撞和离路状态
        self.check_all_agent_collisions()
        self.check_all_agent_off_road()
        
        # 为每个agent计算reward
        for i, car in enumerate(self.cars):
            reward, goal_reached = self.calculate_agent_reward(
                car, 
                car.current_action, 
                self.agent_goals[i], 
                self.agent_collisions[i], 
                self.agent_off_road[i]
            )
            self.agent_rewards[i] += reward
            self.current_step_rewards[i] = reward  # 保存当前步骤的奖励
            
            # 更新agent的目标达成状态
            if i < len(self.agent_goal_reached):
                self.agent_goal_reached[i] = goal_reached
            
            # 如果任何agent到达目标，重置episode
            if goal_reached:
                self.reset_episode()
                return
        
        # 更新玩家车辆的奖励（用于UI显示）
        if self.cars:
            self.episode_reward = self.agent_rewards[0]
            self.score = self.agent_rewards[0]
        
        # 更新步数
        self.step_count += 1
        
        # 检查episode是否结束（任何agent离路或达到最大步数）
        if any(self.agent_off_road) or self.step_count >= self.max_steps:
            self.reset_episode()
    
    def check_all_agent_collisions(self):
        """检查所有agent的碰撞状态"""
        if len(self.cars) < 2:
            # 重置所有agent的碰撞状态
            for i in range(len(self.agent_collisions)):
                self.agent_collisions[i] = False
            self.collision_occurred = False
            return
        
        # 准备车辆状态张量
        # 状态格式: [x, y, heading, speed, length, width, active]
        states_t0 = torch.zeros(1, len(self.cars), 7, device=self.device)
        states_t1 = torch.zeros(1, len(self.cars), 7, device=self.device)
        
        for i, car in enumerate(self.cars):
            # 当前状态
            states_t1[0, i, 0] = float(car.x)
            states_t1[0, i, 1] = float(car.y)
            states_t1[0, i, 2] = float(car.heading)
            states_t1[0, i, 3] = float(car.speed)
            states_t1[0, i, 4] = float(car.length)
            states_t1[0, i, 5] = float(car.width)
            states_t1[0, i, 6] = 1.0  # 激活状态
            
            # 前一状态（简化处理，使用当前状态）
            states_t0[0, i, :] = states_t1[0, i, :]
        
        # 使用CollisionChecker检测碰撞
        collision_results = self.collision_checker.check(states_t0, states_t1)
        
        # 更新每个agent的碰撞状态
        for i in range(len(self.cars)):
            if i < len(self.agent_collisions):
                self.agent_collisions[i] = collision_results[0, i].item()
        
        # 更新玩家车辆的碰撞状态（用于兼容性）
        if self.cars:
            self.collision_occurred = self.agent_collisions[0] if len(self.agent_collisions) > 0 else False
        
        # 打印碰撞信息（调试用）
        if any(self.agent_collisions):
            print(f"🚗 碰撞检测到！车辆数量: {len(self.cars)}")
            for i, car in enumerate(self.cars):
                if i < len(self.agent_collisions) and self.agent_collisions[i]:
                    print(f"  车辆 {i} 发生碰撞，位置: ({car.x:.1f}, {car.y:.1f})")
    
    def check_all_agent_off_road(self):
        """检查所有agent的离路状态"""
        for i, car in enumerate(self.cars):
            # 使用OffroadChecker进行精确的离路检测
            car_state = car.get_offroad_state()
            # 检查是否在道路上（返回True表示在道路上）
            is_on_road = self.offroad_checker.check_on_road(car_state)
            # 如果不在道路上，则为离路
            is_offroad = ~is_on_road
            if i < len(self.agent_off_road):
                self.agent_off_road[i] = is_offroad[0].item()
        
        # 更新玩家车辆的离路状态（用于兼容性）
        if self.cars and len(self.agent_off_road) > 0:
            self.off_road = self.agent_off_road[0]
        else:
            self.off_road = False
    
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
    
    def calculate_agent_reward(self, car: Car, action_idx: int, goal: Dict, collision_occurred: bool, off_road: bool) -> Tuple[float, bool]:
        """统一的agent reward计算方法"""
        # 准备状态张量，传递当前动作索引以获取正确的jerk值
        car_state = car.get_state(action_idx)
        
        # 转换为torch张量
        agents_state = torch.tensor(car_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 10)
        # 准备其他输入
        all_collisions = torch.tensor([[collision_occurred]], dtype=torch.bool)
        offroad_mask = torch.tensor([[off_road]], dtype=torch.bool)
        goal_positions = torch.tensor([[[goal['x'], goal['y']]]], dtype=torch.float32)
        waypoint_reached = torch.tensor([[False]], dtype=torch.bool)  # 初始化为False，由reward_calculator判断
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
        
        return reward[0, 0].item(), goal_reached_tensor[0, 0].item()
    
    def calculate_reward(self, action_idx: int = None) -> Tuple[float, bool]:
        """计算玩家车辆的奖励（保持兼容性）"""
        return self.calculate_agent_reward(self.player_car, action_idx, self.goal, self.collision_occurred, self.off_road)
    
    def reset_episode(self):
        """重置episode"""
        # 使用WorldInitializer重新初始化车辆
        self.initialize_vehicles()
        
        # 重置每个agent的目标和状态跟踪
        self.agent_goals = []
        self.agent_collisions = []
        self.agent_off_road = []
        self.agent_rewards = []
        self.agent_goal_reached = []  # 重置目标达成状态跟踪
        self.current_step_rewards = []  # 重置当前步骤奖励跟踪
        
        # 根据实际车辆数量初始化状态跟踪
        actual_num_cars = len(self.cars)
        for i in range(actual_num_cars):
            self.agent_goals.append(self._initialize_agent_goal())
            self.agent_collisions.append(False)
            self.agent_off_road.append(False)
            self.agent_rewards.append(0.0)
            self.agent_goal_reached.append(False)  # 重置目标达成状态
            self.current_step_rewards.append(0.0)  # 重置当前步骤奖励
        
        # 保持玩家车辆的目标作为主目标（用于UI显示）
        if self.agent_goals:
            self.goal = self.agent_goals[0]
        
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
        if self.training_mode:
            return  # 训练模式下跳过绘制

        # 清屏
        self.screen.fill(self.colors['grass'])
        
        # 绘制道路
        self.draw_road_from_network()
        
        # 绘制目标
        # 将目标位置转换为屏幕坐标
        goal_screen_pos = self.convert_world_to_screen(torch.tensor([[self.goal['x'], self.goal['y']]]))
        if goal_screen_pos:
            pygame.draw.circle(self.screen, self.colors['goal'], 
                             goal_screen_pos[0], int(10.0)) # 使用固定的半径
        else:
            # 如果目标不在视野范围内，在屏幕边缘显示一个指示器
            self.draw_goal_indicator()
        
        # 绘制所有汽车
        for i, car in enumerate(self.cars):
            if i == 0:
                # 玩家车辆用红色
                self.draw_car(car, is_player=True)
            else:
                # 其他车辆用蓝色
                self.draw_car(car, is_player=False)
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
            dx = self.goal['x'] - self.player_car.x
            dy = self.goal['y'] - self.player_car.y
            
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
        
        # 转换坐标（使用向量化操作）地图翻转
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
    
    def draw_car(self, car: Car, is_player: bool = True):
        """绘制汽车"""
        # 将汽车世界坐标转换为屏幕坐标
        car_screen_pos = self.convert_world_to_screen(torch.tensor([[car.x, car.y]]))
        if not car_screen_pos:
            return  # 如果汽车不在视野范围内，不绘制
        
        screen_x, screen_y = car_screen_pos[0]
        
        # 绘制汽车（方形版本）
        car_size = 16  # 方形边长
        car_rect = pygame.Rect(screen_x - car_size // 2, screen_y - car_size // 2, car_size, car_size)
        
        # 根据是否为玩家车辆选择颜色
        if is_player:
            car_color = self.colors['car']  # 红色
        else:
            car_color = self.colors['other_car']  # 蓝色
        
        pygame.draw.rect(self.screen, car_color, car_rect)
        
        # 绘制汽车朝向指示器（从中心到前方的线）
        front_x = screen_x + car_size // 2 * math.cos(car.heading)
        front_y = screen_y + car_size // 2 * math.sin(car.heading)
        pygame.draw.line(self.screen, (255, 255, 0), 
                        (screen_x, screen_y), 
                        (int(front_x), int(front_y)), 3)
    
    def draw_ui(self):
        """绘制用户界面"""
        # 安全检查
        if not self.cars or not self.player_car or not self.agent_rewards:
            return
        
        # 显示玩家车辆的奖励和步数
        current_reward = self.current_step_rewards[0] if len(self.current_step_rewards) > 0 else 0.0
        total_reward = self.agent_rewards[0] if len(self.agent_rewards) > 0 else 0.0
        reward_text = self.font.render(f"current reward: {current_reward:.4f}, total: {total_reward:.4f}", True, self.colors['text'])
        step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
        
        self.screen.blit(reward_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        
        # 显示玩家车辆状态
        speed_text = self.small_font.render(f"speed: {self.player_car.speed:.2f} m/s", True, self.colors['text'])
        accel_text = self.small_font.render(f"acceleration: {self.player_car.acceleration:.2f} m/s²", True, self.colors['text'])
        heading_text = self.small_font.render(f"heading: {math.degrees(self.player_car.heading):.1f}°", True, self.colors['text'])
        
        # 显示离散动作信息
        action_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        current_action = action_names[self.player_car.current_action]
        action_text = self.small_font.render(f"current action: {current_action} (idx: {self.player_car.current_action})", True, self.colors['text'])
        
        self.screen.blit(speed_text, (10, 200))
        self.screen.blit(accel_text, (10, 220))
        self.screen.blit(heading_text, (10, 240))
        self.screen.blit(action_text, (10, 260))
        
        # 显示汽车世界坐标
        car_world_pos = f"car world pos: ({self.player_car.x:.1f}, {self.player_car.y:.1f})"
        car_pos_text = self.small_font.render(car_world_pos, True, self.colors['text'])
        self.screen.blit(car_pos_text, (10, 300))
        
        # 显示目标世界坐标
        if hasattr(self, 'goal') and self.goal:
            goal_world_pos = f"goal world pos: ({self.goal['x']:.1f}, {self.goal['y']:.1f})"
            goal_pos_text = self.small_font.render(goal_world_pos, True, self.colors['text'])
            self.screen.blit(goal_pos_text, (10, 320))
        
        # 显示视野信息
        vision_info = f"vision radius: 100m, visible points: {len(self.get_visible_boundary_points())}"
        vision_text = self.small_font.render(vision_info, True, self.colors['text'])
        self.screen.blit(vision_text, (10, 340))
        
        # 显示车辆信息
        car_info = f"total cars: {len(self.cars)}, player car: red, others: blue"
        car_text = self.small_font.render(car_info, True, self.colors['text'])
        self.screen.blit(car_text, (10, 360))
        
        # 显示游戏状态
        status_text = ""
        if any(self.agent_collisions):
            status_text = "collision!"
        elif any(self.agent_off_road):
            status_text = "off road!"
        # 检查是否到达目标（使用reward返回的goal_reached标志）
        elif len(self.agent_goal_reached) > 0 and self.agent_goal_reached[0]:  # 检查玩家车辆（索引0）的目标达成状态
            status_text = "goal reached!"
        if status_text:
            status_surface = self.font.render(status_text, True, (255, 0, 0))
            self.screen.blit(status_surface, (10, 380))
    
    def run(self):
        """运行游戏主循环"""
        if self.training_mode:
            print("训练模式开始！")
            print("使用随机动作控制汽车（无图形界面）：")
            print("每个step随机从12个离散动作中抽样")
        else:
            print("可视化模式")
            print("每个step随机从12个离散动作中抽样")

        while self.running:
            # 获取随机动作
            action_idx = self.get_random_action()
            # 更新游戏状态
            self.update_game_state(action_idx)
            # 绘制画面（训练模式下跳过）
            self.draw()
    
def main():
    """主函数"""
    import sys
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--training":
        # 训练模式
        print("启动训练模式...")
        game = CarGame(training_mode=True)
        game.run()
    else:
        # 正常可视化模式
        print("启动可视化模式...")
        game = CarGame(training_mode=False)
        game.run()

if __name__ == "__main__":
    main()

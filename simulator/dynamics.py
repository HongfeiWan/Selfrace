import torch
from typing import Dict, Tuple
import math
import numpy as np
from randomize import DrivingStyleSampler

class DiscreteActionSpace:
    # 离散动作空间定义，支持jerk控制
    """
    离散动作空间定义，支持jerk控制
    """
    def __init__(self, device: torch.device, config: Dict = None):
        # 定义12个离散动作
        # along: [-15, -4, 0, 4] m/s³ (纵向jerk)
        # alat: [-4, 0, 4] m/s³ (横向jerk)
        min_long_jerk = config.get('min_longitudinal_jerk', -15.0)
        max_long_jerk = config.get('max_longitudinal_jerk', 4.0)
        min_lat_jerk = config.get('min_lateral_jerk', -4.0)
        max_lat_jerk = config.get('max_lateral_jerk', 4.0)
        self.along_values = [min_long_jerk, -max_long_jerk, 0, max_long_jerk]
        self.alat_values = [min_lat_jerk, 0, max_lat_jerk]
        # 创建所有可能的动作组合
        self.actions = []
        for along in self.along_values:
            for alat in self.alat_values:
                self.actions.append([along, alat])
        self.num_actions = len(self.actions)  # 12个动作
        self.actions_tensor = torch.tensor(self.actions, dtype=torch.float32, device=device)

    def get_action(self, action_idx: torch.Tensor) -> torch.Tensor:
        """
        根据动作索引获取实际动作值
        Args:
            action_idx: 动作索引张量，形状为 (N,) 或标量
            
        Returns:
            torch.Tensor: 实际动作值，形状为 (N, 2) 或 (2,)
        """
        if action_idx.ndim == 0:  # 标量
            return self.actions_tensor[action_idx]
        else:  # 批量
            return self.actions_tensor[action_idx]
        
    def get_all_actions(self) -> torch.Tensor:
        """获取所有动作"""
        return self.actions_tensor

class KinematicBicycleModel:
    """
    实现了一个精确且批量化的运动学自行车模型。
    该模型使用运动学方程的解析解，以确保在离散时间步长内的物理准确性，
    特别是在转弯场景中。所有操作都已完全向量化，以实现最高性能。
    支持离散动作空间。
    """
    def __init__(self, config: Dict, device: torch.device, vehicle_params: Dict[str, torch.Tensor]):
        """
        初始化动力学模型。
        Args:
            config (Dict): 包含车辆物理参数的配置字典。
                支持从simulator.dynamics子配置中读取参数，也支持直接从根级别读取（向后兼容）。
            device (torch.device): 计算设备。
            vehicle_params (Dict[str, torch.Tensor]): 批量车辆参数，必需参数，包含：
                - 'length': (N,) 车辆长度
                - 'width': (N,) 车辆宽度
                - 'wheelbase': (N,) 车辆轴距
                批量参数应与world_init采样顺序一致。
        """
        self.device = device
        # 获取dynamics配置，支持嵌套配置结构
        # 首先尝试从 simulator.dynamics 获取，如果没有则从根级别的 dynamics 获取，最后回退到整个config
        if 'simulator' in config and isinstance(config.get('simulator'), dict):
            simulator_config = config['simulator']
            dynamics_config = simulator_config.get('dynamics', {})
            if not dynamics_config:  # 如果simulator.dynamics不存在，尝试直接使用simulator配置
                dynamics_config = simulator_config
        else:
            dynamics_config = config.get('dynamics', config)
        
        max_steer_deg = dynamics_config.get('vehicle_max_steer_angle')
        if max_steer_deg is None:
            max_steer_deg = 35.0  # 默认值
        self.max_steer_rad = math.radians(max_steer_deg) # 将角度转换为弧度
        
        # 存储批量车辆参数
        self.vehicle_params = {
            'length': vehicle_params['length'].to(device),
            'width': vehicle_params['width'].to(device),
            'wheelbase': vehicle_params['wheelbase'].to(device)
        }
        self.num_vehicles = self.vehicle_params['wheelbase'].shape[0]
        
        # 通过采样获取车辆驾驶风格参数（批量）
        driving_style_sampler = DrivingStyleSampler(device=device)
        Cthrottle, Csteer, Cacc, Cvel = driving_style_sampler.sample_driving_style(size=self.num_vehicles)
        self.Cthrottle = Cthrottle  # 形状 (num_vehicles,)
        self.Csteer = Csteer        # 形状 (num_vehicles,)
        self.Cacc = Cacc            # 形状 (num_vehicles,)
        self.Cvel = Cvel            # 形状 (num_vehicles,)
        
        # 转向角限制参数
        self.max_steering_rate = dynamics_config.get('max_steering_rate', 0.6)  # 最大转向角变化率 (rad/s)
        
        # 加速度约束参数
        self.max_longitudinal_accel = dynamics_config.get('max_longitudinal_accel', 2.5)  # 最大纵向加速度 (m/s²)
        self.min_longitudinal_accel = dynamics_config.get('min_longitudinal_accel', -5.0)  # 最小纵向加速度 (m/s²)
        self.max_lateral_accel = dynamics_config.get('max_lateral_accel', 4.0)       # 最大横向加速度 (m/s²)
        self.min_lateral_accel = dynamics_config.get('min_lateral_accel', -4.0)      # 最小横向加速度 (m/s²)
        
        # 速度约束参数
        self.max_velocity = dynamics_config.get('max_velocity', 20.0)  # 最大速度 (m/s)
        self.min_velocity = dynamics_config.get('min_velocity', -2.0)  # 最小速度 (m/s)
        
        # 数值稳定性参数
        self.curvature_epsilon = float(dynamics_config.get('curvature_epsilon', 1e-5))      # 曲率计算的数值稳定性参数
        #self.steering_epsilon = float(dynamics_config.get('steering_epsilon', 1e-5))       # 转向角计算的数值稳定性参数
        #self.straight_motion_threshold = float(dynamics_config.get('straight_motion_threshold', 1e-5))  # 直线运动判断阈值 (rad)

        # 使用离散动作空间
        self.discrete_action_space = DiscreteActionSpace(device, dynamics_config)
        
        # 当前加速度状态（用于jerk控制）
        # 这些将在step方法中根据批量大小动态调整
        self.current_along = None
        self.current_alat = None
        self.current_steering_angle = None  # 当前有效转向角

    def step(self, states: torch.Tensor, actions: torch.Tensor, dt: float) -> torch.Tensor:
        """
        对一批车辆状态进行一步精确更新。
        Args:
            states (torch.Tensor): 形状为 (N, 4) 的当前状态张量 [x, y, yaw, speed]。
            actions (torch.Tensor): 动作张量，形状为 (N,) 的动作索引。
            dt (float): 模拟时间步长 (s)。
        Returns:
            torch.Tensor: 形状为 (N, 4) 的下一时刻状态张量。
        """
        # 检查输入维度
        assert states.ndim == 2 and states.shape[1] == 4, f"States shape must be (N, 4), but got {states.shape}"
        # 获取批次大小
        batch_size = states.shape[0]
        # 离散动作空间：actions是动作索引
        assert actions.ndim == 1, f"Discrete actions shape must be (N,), but got {actions.shape}"
        # 使用批量wheelbase参数，要求batch_size与车辆参数数量匹配
        assert batch_size == self.num_vehicles, f"batch_size ({batch_size}) must match num_vehicles ({self.num_vehicles})"
        wheelbases = self.vehicle_params['wheelbase']  # (batch_size,)
        
        # 检查并初始化控制状态（确保batch_size正确）
        if (self.current_along is None or self.current_along.shape[0] != batch_size or
            self.current_alat is None or self.current_alat.shape[0] != batch_size):
            print(f"Initializing dynamics state for batch_size: {batch_size}")
            self.current_along = torch.zeros(batch_size, device=self.device)
            self.current_alat = torch.zeros(batch_size, device=self.device)
            # 同时重置prev_along以确保一致性
            if hasattr(self, 'prev_along'):
                self.prev_along = torch.zeros(batch_size, device=self.device)
        
        # 获取实际的jerk动作
        jerk_actions = self.discrete_action_space.get_action(actions)  # (N, 2) [along_jerk, alat_jerk]
        # 更新当前加速度和转向角（jerk控制）
        along_jerk = jerk_actions[:, 0]  # 纵向jerk
        alat_jerk = jerk_actions[:, 1]   # 横向jerk
        
        # 更新加速度和转向角（应用控制系数）
        new_along = self.current_along + along_jerk * dt * self.Cthrottle
        new_alat = self.current_alat + alat_jerk * dt * self.Csteer
        
        # 检测纵向加速度符号变化：a(t-1)_long * a(t)_long < 0
        accel_sign_change = (self.current_along * new_along) < 0
        
        # 如果纵向加速度改变符号，将横向加速度设置为0,纵向加速度设置为0
        if torch.any(accel_sign_change):
            new_alat = torch.where(accel_sign_change, torch.zeros_like(new_alat), new_alat)
            new_along = torch.where(accel_sign_change, torch.zeros_like(new_along), new_along)
            # 如果纵向加速度改变符号，将纵向加速度设置为0，并保持当前的横向加速度
            # 会使得agent更容易停在原地，或者驾驶时速度变化更平缓
        
        # 应用约束：a(t)_long ← clip(a(t)_long, min_long_accel, max_long_accel*Cacc), a(t)_lat ← clip(a(t)_lat, min_lat_accel, max_lat_accel)
        max_along = torch.tensor(self.max_longitudinal_accel, device=self.device) * self.Cacc
        min_along = torch.tensor(self.min_longitudinal_accel, device=self.device)
        along = torch.clamp(new_along, min_along, max_along)
        min_alat = torch.tensor(self.min_lateral_accel, device=self.device)
        max_alat = torch.tensor(self.max_lateral_accel, device=self.device)
        alat = torch.clamp(new_alat, min_alat, max_alat) # 横向加速度约束（从jerk计算的，用于转向角计算）
        
        # 更新当前纵向加速度状态（横向加速度稍后会根据转向角重新计算）
        self.current_along = along
            
        # 从状态张量中解包
        x, y, yaw, speed = states.T
        # 使用梯形法则更新速度：v(t) = v(t-1) + 0.5 * (a(t)_long + a(t-1)_long) * dt
        # 需要保存前一步的纵向加速度用于梯形积分
        if not hasattr(self, 'prev_along'):
            self.prev_along = torch.zeros_like(self.current_along)
        
        # 梯形法则：v(t) = v(t-1) + 0.5 * (a(t)_long + a(t-1)_long) * dt
        new_speed = speed + 0.5 * (along + self.prev_along) * dt
        
        # 检测速度符号变化：v(t-1) * v(t) < 0
        speed_sign_change = (speed * new_speed) < 0
        
        # 如果速度改变符号，将速度设置为0
        if torch.any(speed_sign_change):
            new_speed = torch.where(speed_sign_change, torch.zeros_like(new_speed), new_speed)
        
        # 应用速度约束：v(t) ← clip(v(t), min_velocity, max_velocity*Cvel)
        max_vel = torch.tensor(self.max_velocity, device=self.device) * self.Cvel
        min_vel = torch.tensor(self.min_velocity, device=self.device)
        new_speed = torch.clamp(new_speed, min_vel, max_vel)
        
        # 更新前一步的纵向加速度
        self.prev_along = along.clone()

        # 根据横向加速度计算目标转向角（使用当前速度 v^(t)，即 new_speed）
        # 原文：从横向加速度反推转向角，使用当前时刻的速度
        target_steering_angle = self.calculate_steering_angle(alat, new_speed, wheelbases)
        
        # 初始化当前转向角（如果还没有初始化）
        if self.current_steering_angle is None or self.current_steering_angle.shape[0] != batch_size:
            self.current_steering_angle = torch.zeros(batch_size, device=self.device)
        
        # 计算转向角变化：δφ = φ_target - φ(t-1)
        steering_change = target_steering_angle - self.current_steering_angle
        
        # 限制转向角变化率：δφ = clip(δφ, -δmax*dt, δmax*dt)
        max_change = self.max_steering_rate * dt
        limited_steering_change = torch.clamp(steering_change, -max_change, max_change)
        
        # 更新有效转向角：φ(t) = clip(φ(t-1) + δφ, -φmax, φmax)
        new_steering_angle = self.current_steering_angle + limited_steering_change
        steering_angle = torch.clamp(new_steering_angle, -self.max_steer_rad, self.max_steer_rad)
        
        # 更新当前转向角状态
        self.current_steering_angle = steering_angle
        
        # 根据有效转向角更新曲率和横向加速度（物理一致性修正）
        # 原文：ρ^(-1) ← tan(φ^(t)) / l_wb
        effective_curvature = torch.tan(steering_angle) / wheelbases
        # 原文：a_lat(t) ← (v^(t))^2 * ρ^(-1)，使用当前时刻的速度 v^(t)，即 new_speed
        # 注意：这里根据实际转向角重新计算横向加速度，确保物理一致性
        # （之前从jerk计算的alat只是用于反推目标转向角）
        effective_alat = new_speed ** 2 * effective_curvature
        
        # 在计算 effective_alat 后，再次应用约束
        min_alat = torch.tensor(self.min_lateral_accel, device=self.device)
        max_alat = torch.tensor(self.max_lateral_accel, device=self.device)
        effective_alat = torch.clamp(effective_alat, min_alat, max_alat)
        # 更新横向加速度状态（用于下一次step的jerk计算）
        self.current_alat = effective_alat
        
        # 使用时间步内的平均速度进行位移计算，提高精度
        avg_speed = (speed + new_speed) / 2.0

        # 使用自行车动力学模型更新车辆位置
        # 计算位移：d = 0.5(v(t) + v(t-1)) * Δt
        displacement = avg_speed * dt
        
        # 计算角位移：θ = d * ρ^(-1)
        angular_displacement = displacement * effective_curvature
        
        # 原文：Δx = ρ sin(θ), Δy = ρ cos(θ)
        # 其中 ρ = 1 / ρ^(-1)，θ = d * ρ^(-1)
        # 需要处理曲率为0的情况（直线运动）
        # curvature_threshold = self.curvature_epsilon
        # is_straight = torch.abs(effective_curvature) < curvature_threshold
        
        # 计算半径：ρ = 1 / ρ^(-1)
        # radius = torch.where(is_straight,
        #                     torch.ones_like(effective_curvature),  # 直线时占位，实际不会使用
        #                     1.0 / effective_curvature)
        
        # 位置变化：Δx = ρ sin(θ), Δy = ρ cos(θ)
        # dx_curved = radius * torch.sin(angular_displacement)
        # dy_curved = radius * torch.cos(angular_displacement)
             
        # 根据是否直线选择相应的计算方式
        dx = displacement * torch.cos(yaw)
        dy = displacement * torch.sin(yaw)
        
        # 偏航角变化
        d_yaw = angular_displacement 
        
        # --- 计算新状态 ---
        new_x = x + dx
        new_y = y + dy
        new_yaw = yaw + d_yaw

        # 归一化偏航角到 [-pi, pi]
        new_yaw = torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw))
        
        # 将新状态组合成一个张量返回
        new_states = torch.stack([new_x, new_y, new_yaw, new_speed], dim=1)
        return new_states
    
    def reset_control_state(self):
        """重置控制状态（加速度和转向角）"""
        # 清除所有控制状态变量，让step方法在需要时重新初始化正确的batch_size
        self.current_along = None
        self.current_alat = None
        self.current_steering_angle = None
        # 清除前一步的纵向加速度，让step方法重新创建
        if hasattr(self, 'prev_along'):
            delattr(self, 'prev_along')
        print("Dynamics control state reset - variables cleared for fresh initialization")

    def calculate_steering_angle(self, alat: torch.Tensor, speed: torch.Tensor, wheelbases: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """
        根据横向加速度和速度计算转向角
        Args:
            alat (torch.Tensor): 横向加速度
            speed (torch.Tensor): 速度
            wheelbases (torch.Tensor): 批量轴距参数，形状为 (N,)，必需参数
            epsilon (float): 数值稳定性参数，如果为None则使用配置中的默认值
        Returns:
            torch.Tensor: 转向角 (弧度)
        """
        # 使用配置中的默认值或传入的参数
        if epsilon is None:
            epsilon = self.curvature_epsilon
        
        # 使用传入的wheelbase值
        L_used = wheelbases
            
        # 计算曲率：ρ^(-1) = alat / max(v^2, ε)
        speed_squared = torch.clamp(speed ** 2, min=epsilon)
        curvature = alat / speed_squared
        
        # 应用数值稳定性：ρ^(-1) ← sign(ρ^(-1)) * max(|ρ^(-1)|, ε)
        curvature_sign = torch.sign(curvature)
        curvature_magnitude = torch.clamp(torch.abs(curvature), min=epsilon)
        curvature = curvature_sign * curvature_magnitude

        # 计算转向角：φ = arctan(ρ^(-1) * lwb)
        steering_angle = torch.atan(curvature * L_used)
        return steering_angle
    
    def get_discrete_action_space(self) -> DiscreteActionSpace:
        """获取离散动作空间"""
        return self.discrete_action_space

# 为了让这个文件可以独立测试，添加一个 main block
# if __name__ == '__main__':
#     import json
#     import os
#     import sys
#     import pygame
#     from pygame.locals import *
#     from OpenGL.GL import *
#     from OpenGL.GLU import *
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.animation as animation
#     from matplotlib.backends.backend_agg import FigureCanvasAgg
#     from road import RoadNetwork
#     from randomize import VehicleParameterSampler
    
#     # 尝试导入Windows API来管理窗口焦点（仅在Windows系统上）
#     try:
#         import win32gui
#         import win32con
#         WINDOWS_FOCUS_AVAILABLE = True
#     except ImportError:
#         WINDOWS_FOCUS_AVAILABLE = False
#         print("注意: win32gui不可用，无法自动管理窗口焦点（不影响功能）")
    
#     # 加载配置文件
#     config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = json.load(f)
    
#     # 获取配置参数
#     map_path = os.path.join(config['map_path'], config['default_map'])
#     map_path = os.path.join(os.path.dirname(__file__), '..', map_path)
#     device_str = config.get('device', 'cuda')
#     device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
#     simulator_config = config.get('simulator', {})
#     sim_dt = simulator_config.get('sim_dt', 0.3)
    
#     print(f"使用设备: {device}")
#     print(f"地图路径: {map_path}")
#     print(f"模拟时间步长: {sim_dt}秒")
    
#     # 初始化路网
#     print("加载路网...")
#     road_network = RoadNetwork(map_path, device)
#     print(f"加载了 {len(road_network.quad_ids)} 个道路段")
    
#     # 初始化车辆参数
#     print("初始化车辆参数...")
#     vehicle_sampler = VehicleParameterSampler(config, device)
#     vehicle_params = vehicle_sampler.sample_batch_vehicle_parameters(1)  # 单辆车
    
#     # 初始化动力学模型
#     print("初始化动力学模型...")
#     dynamics_model = KinematicBicycleModel(config, device, vehicle_params)
    
#     # 初始化车辆状态：在第一个道路段的中心线上
#     initial_state = torch.zeros(1, 4, device=device)
#     initial_speed = 0  # 初始速度 5 m/s
#     if len(road_network.quad_centerlines) > 0:
#         first_centerline = road_network.quad_centerlines[0]
#         center_point = (first_centerline[0] + first_centerline[1]) / 2.0
#         direction_vec = first_centerline[1] - first_centerline[0]
#         yaw = torch.atan2(direction_vec[1], direction_vec[0])
#         initial_state[0, 0] = center_point[0]
#         initial_state[0, 1] = center_point[1]
#         initial_state[0, 2] = yaw
#         initial_state[0, 3] = initial_speed
#     else:
#         initial_state[0, 0] = 0.0
#         initial_state[0, 1] = 0.0
#         initial_state[0, 2] = 0.0
#         initial_state[0, 3] = initial_speed
    
#     current_state = initial_state.clone()
#     # 确保初始速度被正确设置（避免被约束修改）
#     print(f"初始车辆状态: 位置=({initial_state[0, 0].item():.2f}, {initial_state[0, 1].item():.2f}), "
#           f"偏航角={np.degrees(initial_state[0, 2].item()):.2f}°, 速度={initial_state[0, 3].item():.2f} m/s")
#     print(f"注意：时间步长={sim_dt}秒，以初始速度{initial_speed}m/s计算，每步车辆可移动约 {initial_speed * sim_dt:.2f}米")
    
#     # 按键映射：qwertyuiop[] -> 0-11
#     key_to_action = {
#         K_q: 0, K_w: 1, K_e: 2, K_r: 3, K_t: 4, K_y: 5,
#         K_u: 6, K_i: 7, K_o: 8, K_p: 9,
#         K_LEFTBRACKET: 10, K_RIGHTBRACKET: 11
#     }
#     # 按键到动作的字符映射（用于调试）
#     key_char_to_action = {
#         'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4, 'y': 5,
#         'u': 6, 'i': 7, 'o': 8, 'p': 9,
#         '[': 10, ']': 11
#     }
    
#     # 初始化Pygame
#     pygame.init()
    
#     # 创建主渲染窗口（OpenGL）
#     screen_width, screen_height = 1200, 800
#     gl_screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
#     pygame.display.set_caption('车辆动力学模拟 - 主视图 (请确保窗口有焦点)')
    
#     # 禁用键盘重复按键（按一次执行一次）
#     pygame.key.set_repeat()
    
#     # 设置OpenGL
#     glEnable(GL_BLEND)
#     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#     glEnable(GL_LINE_SMOOTH)
#     glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
#     # 存储历史状态用于显示
#     prev_along = 0.0
#     prev_alat = 0.0
#     prev_speed = current_state[0, 3].item()
#     prev_steering = 0.0
#     prev_along_jerk = 0.0
#     prev_alat_jerk = 0.0
    
#     current_along = 0.0
#     current_alat = 0.0
#     current_speed = current_state[0, 3].item()
#     current_steering = 0.0
#     current_along_jerk = 0.0
#     current_alat_jerk = 0.0
    
#     # 数据记录用于曲线可视化
#     history_time = []  # 时间点（步数）
#     history_speed = []  # 速度历史
#     history_long_accel = []  # 纵向加速度历史
#     history_lat_accel = []  # 横向加速度历史
#     history_steering = []  # 转向角历史（度）
#     history_steering_rate = []  # 转向角速率历史（度/秒）
    
#     # 初始化matplotlib图表窗口
#     plt.ion()  # 开启交互模式
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle('车辆状态曲线', fontsize=14)
    
#     # 设置子图
#     ax_speed = axes[0, 0]
#     ax_speed.set_title('速度 (m/s)')
#     ax_speed.set_xlabel('步数')
#     ax_speed.set_ylabel('速度 (m/s)')
#     ax_speed.grid(True, alpha=0.3)
#     line_speed, = ax_speed.plot([], [], 'b-', linewidth=2, label='Speed')
#     ax_speed.legend()
    
#     ax_accel = axes[0, 1]
#     ax_accel.set_title('加速度 (m/s²)')
#     ax_accel.set_xlabel('步数')
#     ax_accel.set_ylabel('加速度 (m/s²)')
#     ax_accel.grid(True, alpha=0.3)
#     line_long_accel, = ax_accel.plot([], [], 'r-', linewidth=2, label='Longitudinal')
#     line_lat_accel, = ax_accel.plot([], [], 'g-', linewidth=2, label='Lateral')
#     ax_accel.legend()
    
#     ax_steering = axes[1, 0]
#     ax_steering.set_title('转向角 (deg)')
#     ax_steering.set_xlabel('步数')
#     ax_steering.set_ylabel('转向角 (deg)')
#     ax_steering.grid(True, alpha=0.3)
#     line_steering, = ax_steering.plot([], [], 'm-', linewidth=2, label='Steering')
#     ax_steering.legend()
    
#     ax_steering_rate = axes[1, 1]
#     ax_steering_rate.set_title('转向角速率 (rad/s)')
#     ax_steering_rate.set_xlabel('步数')
#     ax_steering_rate.set_ylabel('转向角速率 (rad/s)')
#     ax_steering_rate.grid(True, alpha=0.3)
#     line_steering_rate, = ax_steering_rate.plot([], [], 'c-', linewidth=2, label='Steering Rate')
#     ax_steering_rate.legend()
    
#     plt.tight_layout()
#     plt.show(block=False)
    
#     # 设置matplotlib窗口为不抢夺焦点（仅在Windows上）
#     if WINDOWS_FOCUS_AVAILABLE:
#         try:
#             # 获取matplotlib窗口句柄
#             fig_manager = fig.canvas.manager
#             if hasattr(fig_manager, 'window'):
#                 mpl_hwnd = fig_manager.window.winfo_id()
#                 # 尝试使用pygame获取窗口句柄（在某些后端可能需要）
#                 # 这里先尝试设置matplotlib窗口为不激活
#                 pass
#         except:
#             pass
    
#     # 获取pygame窗口句柄（用于Windows焦点管理）
#     pygame_hwnd = None
#     if WINDOWS_FOCUS_AVAILABLE:
#         try:
#             # 通过窗口标题查找pygame窗口
#             window_title = '车辆动力学模拟 - 主视图 (请确保窗口有焦点)'
#             def enum_windows_callback(hwnd, windows):
#                 if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == window_title:
#                     windows.append(hwnd)
#                 return True
            
#             windows = []
#             try:
#                 win32gui.EnumWindows(enum_windows_callback, windows)
#                 if windows:
#                     pygame_hwnd = windows[0]
#                     print(f"已找到pygame窗口句柄: {pygame_hwnd}")
#             except Exception as e:
#                 print(f"查找pygame窗口时出错: {e}")
#         except Exception as e:
#             print(f"初始化窗口焦点管理时出错: {e}")
    
#     # 保持pygame窗口焦点
#     def ensure_pygame_focus():
#         """确保pygame窗口保持焦点"""
#         if WINDOWS_FOCUS_AVAILABLE and pygame_hwnd:
#             try:
#                 # 使用ShowWindow而不是SetForegroundWindow，避免抢夺焦点时的限制
#                 # SW_RESTORE = 9, SW_SHOW = 5
#                 win32gui.ShowWindow(pygame_hwnd, win32con.SW_RESTORE)
#                 # 尝试设置焦点（可能需要特定条件）
#                 try:
#                     win32gui.SetForegroundWindow(pygame_hwnd)
#                 except:
#                     # 如果SetForegroundWindow失败（Windows安全限制），忽略
#                     pass
#             except Exception as e:
#                 # 如果失败，静默忽略（不影响主程序）
#                 pass
    
#     # 记录初始状态到历史数据
#     history_time.append(0)
#     history_speed.append(current_speed)
#     history_long_accel.append(0.0)
#     history_lat_accel.append(0.0)
#     # 初始转向角（从yaw角度获取，因为动力学模型还没初始化转向角）
#     initial_yaw_deg = np.degrees(current_state[0, 2].item())
#     history_steering.append(initial_yaw_deg)
#     history_steering_rate.append(0.0)
    
#     # 相机参数
#     camera_x = current_state[0, 0].item()
#     camera_y = current_state[0, 1].item()
#     camera_zoom = 50.0  # 缩放级别
    
#     # 渲染函数
#     def setup_projection():
#         """设置OpenGL投影矩阵"""
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         aspect = screen_width / screen_height
#         # 注意：OpenGL的Y轴是向上的，所以bottom < top
#         gluOrtho2D(
#             camera_x - camera_zoom * aspect,
#             camera_x + camera_zoom * aspect,
#             camera_y - camera_zoom,  # bottom
#             camera_y + camera_zoom   # top
#         )
#         glMatrixMode(GL_MODELVIEW)
#         glLoadIdentity()
    
#     def draw_line(start, end, color=(1.0, 1.0, 1.0), width=1.0):
#         """绘制一条线"""
#         glColor3f(*color)
#         glLineWidth(width)
#         glBegin(GL_LINES)
#         glVertex2f(start[0], start[1])
#         glVertex2f(end[0], end[1])
#         glEnd()
    
#     def draw_point(pos, color=(1.0, 0.0, 0.0), size=3.0):
#         """绘制一个点"""
#         glColor3f(*color)
#         glPointSize(size)
#         glBegin(GL_POINTS)
#         glVertex2f(pos[0], pos[1])
#         glEnd()
    
#     def render_road_network():
#         """渲染路网（只绘制边界）"""
#         # 绘制左边界
#         glColor3f(0.0, 0.5, 1.0)  # 蓝色
#         glLineWidth(2.0)
#         glBegin(GL_LINES)
#         left_boundaries = road_network.left_boundaries.cpu().numpy()
#         for boundary in left_boundaries:
#             glVertex2f(boundary[0, 0], boundary[0, 1])
#             glVertex2f(boundary[1, 0], boundary[1, 1])
#         glEnd()
        
#         # 绘制右边界
#         glColor3f(0.0, 1.0, 0.5)  # 绿色
#         glLineWidth(2.0)
#         glBegin(GL_LINES)
#         right_boundaries = road_network.right_boundaries.cpu().numpy()
#         for boundary in right_boundaries:
#             glVertex2f(boundary[0, 0], boundary[0, 1])
#             glVertex2f(boundary[1, 0], boundary[1, 1])
#         glEnd()
    
#     def render_vehicle(state):
#         """渲染车辆"""
#         x, y, yaw, speed = state[0].cpu().numpy()
        
#         # 获取车辆尺寸
#         length = vehicle_params['length'][0].item()
#         width = vehicle_params['width'][0].item()
        
#         # 计算车辆四个角点
#         cos_yaw = np.cos(yaw)
#         sin_yaw = np.sin(yaw)
#         half_length = length / 2.0
#         half_width = width / 2.0
        
#         corners = np.array([
#             [-half_length, -half_width],
#             [half_length, -half_width],
#             [half_length, half_width],
#             [-half_length, half_width]
#         ])
        
#         # 旋转和平移
#         rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
#         rotated_corners = corners @ rotation_matrix.T
#         world_corners = rotated_corners + np.array([x, y])
        
#         # 绘制车辆矩形
#         glColor3f(1.0, 0.0, 0.0)  # 红色
#         glLineWidth(3.0)
#         glBegin(GL_LINE_LOOP)
#         for corner in world_corners:
#             glVertex2f(corner[0], corner[1])
#         glEnd()
        
#         # 绘制方向箭头
#         arrow_length = length * 0.6
#         arrow_end_x = x + arrow_length * cos_yaw
#         arrow_end_y = y + arrow_length * sin_yaw
#         glColor3f(1.0, 1.0, 0.0)  # 黄色
#         glLineWidth(2.0)
#         glBegin(GL_LINES)
#         glVertex2f(x, y)
#         glVertex2f(arrow_end_x, arrow_end_y)
#         glEnd()
        
#         # 绘制车辆中心点
#         draw_point([x, y], color=(1.0, 1.0, 1.0), size=5.0)
    
#     # 文本渲染辅助函数（使用pygame生成文本纹理）
#     def render_text_to_texture(text, font_size=20):
#         """将文本渲染为OpenGL纹理"""
#         font = pygame.font.Font(None, font_size)
#         text_surface = font.render(str(text), True, (255, 255, 255))
#         text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
#         width, height = text_surface.get_size()
#         texture = glGenTextures(1)
#         glBindTexture(GL_TEXTURE_2D, texture)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
#         return texture, width, height
    
#     def render_info_display(state, speed, along, alat, steering, along_jerk, alat_jerk,
#                             prev_speed_val, prev_along_val, prev_alat_val, prev_steering_val, 
#                             prev_along_jerk_val, prev_alat_jerk_val):
#         """在OpenGL窗口中渲染信息显示"""
#         # 切换到2D正交投影模式绘制UI
#         glMatrixMode(GL_PROJECTION)
#         glPushMatrix()
#         glLoadIdentity()
#         glOrtho(0, screen_width, screen_height, 0, -1, 1)  # 注意：Y轴反转
#         glMatrixMode(GL_MODELVIEW)
#         glPushMatrix()
#         glLoadIdentity()
        
#         # 绘制半透明背景框
#         glEnable(GL_BLEND)
#         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#         glDisable(GL_TEXTURE_2D)
        
#         # 信息框位置和大小
#         box_x = 10
#         box_y = 10
#         box_width = 480
#         box_height = 600
        
#         # 绘制背景
#         glColor4f(0.1, 0.1, 0.1, 0.85)
#         glBegin(GL_QUADS)
#         glVertex2f(box_x, box_y)
#         glVertex2f(box_x + box_width, box_y)
#         glVertex2f(box_x + box_width, box_y + box_height)
#         glVertex2f(box_x, box_y + box_height)
#         glEnd()
        
#         # 绘制边框
#         glColor4f(0.5, 0.5, 0.5, 1.0)
#         glLineWidth(2.0)
#         glBegin(GL_LINE_LOOP)
#         glVertex2f(box_x, box_y)
#         glVertex2f(box_x + box_width, box_y)
#         glVertex2f(box_x + box_width, box_y + box_height)
#         glVertex2f(box_x, box_y + box_height)
#         glEnd()
        
#         # 使用pygame在临时surface上渲染所有文本，然后转换为纹理
#         # 为了简化，我们创建一个大的文本surface
#         info_surface = pygame.Surface((box_width - 20, box_height - 20), pygame.SRCALPHA)
#         info_surface.fill((30, 30, 30, 255))
        
#         # 尝试加载系统字体以支持中文，如果失败则使用默认字体
#         try:
#             # 尝试使用系统默认字体（pygame已经在外部导入）
#             font = pygame.font.SysFont('simhei', 20)  # 黑体，如果不存在会回退
#             title_font = pygame.font.SysFont('simhei', 24)
#         except:
#             # 如果无法加载系统字体，使用默认字体（可能不支持中文）
#             font = pygame.font.Font(None, 20)
#             title_font = pygame.font.Font(None, 24)
        
#         y_pos = 10
#         line_height = 22
        
#         # 标题（如果字体不支持中文，使用英文）
#         try:
#             title = title_font.render("车辆状态信息", True, (255, 255, 255))
#         except:
#             title = title_font.render("Vehicle State Info", True, (255, 255, 255))
#         info_surface.blit(title, (10, y_pos))
#         y_pos += line_height + 5
        
#         # 当前值（使用英文避免中文乱码问题）
#         texts = [
#             ("Current Values:", (100, 200, 100)),
#             (f"Speed: {speed:.2f} m/s", (255, 255, 255)),
#             (f"Long Accel: {along:.2f} m/s2", (255, 255, 255)),
#             (f"Lat Accel: {alat:.2f} m/s2", (255, 255, 255)),
#             (f"Steering: {np.degrees(steering):.2f} deg", (255, 255, 255)),
#             (f"Long Jerk: {along_jerk:.2f} m/s3", (255, 255, 255)),
#             (f"Lat Jerk: {alat_jerk:.2f} m/s3", (255, 255, 255)),
#             ("", (255, 255, 255)),
#             ("Previous Values:", (200, 100, 100)),
#             (f"Speed: {prev_speed_val:.2f} m/s", (255, 255, 255)),
#             (f"Long Accel: {prev_along_val:.2f} m/s2", (255, 255, 255)),
#             (f"Lat Accel: {prev_alat_val:.2f} m/s2", (255, 255, 255)),
#             (f"Steering: {np.degrees(prev_steering_val):.2f} deg", (255, 255, 255)),
#             (f"Long Jerk: {prev_along_jerk_val:.2f} m/s3", (255, 255, 255)),
#             (f"Lat Jerk: {prev_alat_jerk_val:.2f} m/s3", (255, 255, 255)),
#             ("", (255, 255, 255)),
#             (f"Position: ({state[0, 0].item():.2f}, {state[0, 1].item():.2f})", (200, 200, 255)),
#             (f"Yaw: {np.degrees(state[0, 2].item()):.2f} deg", (200, 200, 255)),
#         ]
        
#         for text, color in texts:
#             if text:
#                 text_surf = font.render(text, True, color)
#                 info_surface.blit(text_surf, (10, y_pos))
#             y_pos += line_height
        
#         # 将surface转换为OpenGL纹理并绘制
#         # 确保surface格式正确
#         if info_surface.get_flags() & pygame.SRCALPHA:
#             text_data = pygame.image.tostring(info_surface, "RGBA", True)
#             format = GL_RGBA
#         else:
#             text_data = pygame.image.tostring(info_surface, "RGB", True)
#             format = GL_RGB
#         width, height = info_surface.get_size()
        
#         glEnable(GL_TEXTURE_2D)
#         texture = glGenTextures(1)
#         glBindTexture(GL_TEXTURE_2D, texture)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#         glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, text_data)
        
#         # 绘制纹理（注意Y轴反转，因为OpenGL的Y轴是向上的）
#         glColor4f(1.0, 1.0, 1.0, 1.0)
#         glBegin(GL_QUADS)
#         # OpenGL纹理坐标：左下角为(0,0)，右上角为(1,1)
#         # 但由于我们在反转的Y轴坐标系中，需要调整顶点顺序
#         glTexCoord2f(0, 1)  # 纹理左下
#         glVertex2f(box_x + 10, box_y + 10)  # 屏幕左上（在反转坐标系中）
#         glTexCoord2f(1, 1)  # 纹理右下
#         glVertex2f(box_x + 10 + width, box_y + 10)  # 屏幕右上
#         glTexCoord2f(1, 0)  # 纹理右上
#         glVertex2f(box_x + 10 + width, box_y + 10 + height)  # 屏幕右下
#         glTexCoord2f(0, 0)  # 纹理左上
#         glVertex2f(box_x + 10, box_y + 10 + height)  # 屏幕左下
#         glEnd()
        
#         # 清理纹理
#         glDeleteTextures([texture])
#         glDisable(GL_TEXTURE_2D)
        
#         # 恢复之前的投影矩阵
#         glPopMatrix()
#         glMatrixMode(GL_PROJECTION)
#         glPopMatrix()
    
#     # 主循环
#     clock = pygame.time.Clock()
#     running = True
#     # 默认动作：查找对应的0 jerk动作（应该是action=4，对应[-4, 0]）
#     # 或者action=7，对应[0, 0]，这是真正的0 jerk
#     # 检查动作空间，找到0 jerk对应的索引
#     action_space = dynamics_model.get_discrete_action_space()
#     zero_jerk_action = None
#     for i in range(action_space.num_actions):
#         action_val = action_space.get_action(torch.tensor([i], device=device))
#         if abs(action_val[0, 0].item()) < 0.01 and abs(action_val[0, 1].item()) < 0.01:
#             zero_jerk_action = i
#             break
#     if zero_jerk_action is None:
#         zero_jerk_action = 4  # 如果没有找到，使用action=4作为默认值
#     current_action = torch.tensor([zero_jerk_action], device=device)  # 默认动作：无jerk
#     print(f"默认动作索引: {zero_jerk_action} (对应0 jerk)")
    
#     # 显示matplotlib窗口后，立即将焦点返回到pygame窗口
#     ensure_pygame_focus()
    
#     print("\n=== 控制说明 ===")
#     print("按键: q-w-e-r-t-y-u-i-o-p-[ ] 对应动作索引 0-11")
#     print("ESC: 退出")
#     print("WASD: 移动相机")
#     print("鼠标滚轮: 缩放\n")
    
#     while running:
#         # 等待按键输入（阻塞等待）
#         action_updated = False
#         for event in pygame.event.get():
#             if event.type == QUIT:
#                 running = False
#                 break
#             elif event.type == KEYDOWN:
#                 # 调试输出
#                 print(f"检测到按键: key={event.key}, unicode={event.unicode if event.unicode else 'None'}")
                
#                 if event.key == K_ESCAPE:
#                     running = False
#                     break
#                 elif event.key in key_to_action:
#                     action_idx = key_to_action[event.key]
#                     current_action[0] = action_idx
#                     # 获取当前jerk值用于显示
#                     jerk_actions = dynamics_model.discrete_action_space.get_action(current_action)
#                     current_along_jerk = jerk_actions[0, 0].item()
#                     current_alat_jerk = jerk_actions[0, 1].item()
#                     print(f"✓ 执行动作 {action_idx}: "
#                           f"纵向jerk={current_along_jerk:.2f}, 横向jerk={current_alat_jerk:.2f}")
#                     action_updated = True
#                 elif event.unicode and event.unicode in key_char_to_action:
#                     # 备用方案：通过字符识别
#                     action_idx = key_char_to_action[event.unicode]
#                     current_action[0] = action_idx
#                     jerk_actions = dynamics_model.discrete_action_space.get_action(current_action)
#                     current_along_jerk = jerk_actions[0, 0].item()
#                     current_alat_jerk = jerk_actions[0, 1].item()
#                     print(f"✓ 执行动作 {action_idx} (通过字符 '{event.unicode}'): "
#                           f"纵向jerk={current_along_jerk:.2f}, 横向jerk={current_alat_jerk:.2f}")
#                     action_updated = True
        
#         # 只有在有按键输入时才更新状态
#         if action_updated:
#             # 保存上一时刻的值和位置
#             prev_state_pos = current_state[0, :2].clone()
#             prev_along = current_along
#             prev_alat = current_alat
#             prev_speed = current_speed
#             prev_steering = current_steering
#             prev_along_jerk = current_along_jerk
#             prev_alat_jerk = current_alat_jerk
            
#             # 执行一步动力学更新
#             current_state = dynamics_model.step(current_state, current_action, sim_dt)
            
#             # 计算位置变化（用于调试）
#             new_pos = current_state[0, :2]
#             pos_change = new_pos - prev_state_pos
#             pos_change_norm = torch.norm(pos_change).item()
            
#             # 获取当前状态值
#             current_speed = current_state[0, 3].item()
#             if dynamics_model.current_along is not None:
#                 current_along = dynamics_model.current_along[0].item()
#             else:
#                 current_along = 0.0
#             if dynamics_model.current_alat is not None:
#                 current_alat = dynamics_model.current_alat[0].item()
#             else:
#                 current_alat = 0.0
#             if dynamics_model.current_steering_angle is not None:
#                 current_steering = dynamics_model.current_steering_angle[0].item()
#             else:
#                 current_steering = 0.0
            
#             # 计算转向角速率（弧度/秒）
#             # 注意：steering是弧度，所以steering_rate直接是rad/s
#             if len(history_time) > 0:  # 至少有一个历史数据点
#                 steering_rate_rad = (current_steering - prev_steering) / sim_dt  # rad/s
#             else:
#                 steering_rate_rad = 0.0
            
#             # 记录历史数据
#             step_num = len(history_time)
#             history_time.append(step_num)
#             history_speed.append(current_speed)
#             history_long_accel.append(current_along)
#             history_lat_accel.append(current_alat)
#             history_steering.append(np.degrees(current_steering))
#             history_steering_rate.append(steering_rate_rad)  # 直接使用rad/s
            
#             # 限制历史数据长度（保持最近1000个点）
#             max_history_len = 1000
#             if len(history_time) > max_history_len:
#                 history_time = history_time[-max_history_len:]
#                 history_speed = history_speed[-max_history_len:]
#                 history_long_accel = history_long_accel[-max_history_len:]
#                 history_lat_accel = history_lat_accel[-max_history_len:]
#                 history_steering = history_steering[-max_history_len:]
#                 history_steering_rate = history_steering_rate[-max_history_len:]
            
#             # 更新曲线图表
#             if len(history_time) > 0:
#                 time_data = np.array(history_time)
                
#                 # 更新速度曲线
#                 line_speed.set_data(time_data, history_speed)
#                 ax_speed.relim()
#                 ax_speed.autoscale_view()
                
#                 # 更新加速度曲线
#                 line_long_accel.set_data(time_data, history_long_accel)
#                 line_lat_accel.set_data(time_data, history_lat_accel)
#                 ax_accel.relim()
#                 ax_accel.autoscale_view()
                
#                 # 更新转向角曲线
#                 line_steering.set_data(time_data, history_steering)
#                 ax_steering.relim()
#                 ax_steering.autoscale_view()
                
#                 # 更新转向角速率曲线
#                 line_steering_rate.set_data(time_data, history_steering_rate)
#                 ax_steering_rate.relim()
#                 ax_steering_rate.autoscale_view()
                
#                 # 刷新图表（使用canvas.draw_idle()避免抢夺焦点）
#                 fig.canvas.draw_idle()
#                 fig.canvas.flush_events()
                
#                 # 确保pygame窗口保持焦点
#                 ensure_pygame_focus()
            
#             # 调试输出：位置变化信息
#             print(f"[动力学更新] dt={sim_dt:.3f}s, "
#                   f"位置: ({prev_state_pos[0].item():.2f}, {prev_state_pos[1].item():.2f}) -> "
#                   f"({new_pos[0].item():.2f}, {new_pos[1].item():.2f}), "
#                   f"变化量: dx={pos_change[0].item():.4f}, dy={pos_change[1].item():.4f}, "
#                   f"距离={pos_change_norm:.4f}m, 速度={current_speed:.2f}m/s")
            
#             # 检查位置变化是否异常（超过速度*dt的合理范围）
#             expected_max_distance = abs(current_speed) * sim_dt * 1.5  # 允许1.5倍误差
#             if pos_change_norm > expected_max_distance:
#                 print(f"⚠️  警告：位置变化过大！预期最大变化: {expected_max_distance:.4f}m, "
#                       f"实际变化: {pos_change_norm:.4f}m")
            
#             # 更新相机（跟随车辆）
#             camera_x = current_state[0, 0].item()
#             camera_y = current_state[0, 1].item()
            
#             # 调试输出：相机和可视化信息
#             print(f"[可视化] 相机位置: ({camera_x:.2f}, {camera_y:.2f}), "
#                   f"缩放: {camera_zoom:.1f}, 视场范围: "
#                   f"x=[{camera_x - camera_zoom * (screen_width / screen_height):.2f}, "
#                   f"{camera_x + camera_zoom * (screen_width / screen_height):.2f}], "
#                   f"y=[{camera_y - camera_zoom:.2f}, {camera_y + camera_zoom:.2f}]")
        
#         # 总是渲染当前状态（即使没有更新，也显示当前状态）
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glClearColor(0.1, 0.1, 0.1, 1.0)  # 深灰色背景
        
#         setup_projection()
        
#         # 绘制路网
#         render_road_network()
        
#         # 绘制车辆
#         render_vehicle(current_state)
        
#         # 绘制信息显示（UI覆盖层）- 传入当前状态值
#         render_info_display(current_state, current_speed, current_along, current_alat, 
#                            current_steering, current_along_jerk, current_alat_jerk,
#                            prev_speed, prev_along, prev_alat, prev_steering, 
#                            prev_along_jerk, prev_alat_jerk)
        
#         pygame.display.flip()
        
#         # 如果没有按键输入，等待一下避免CPU占用过高，但不更新状态
#         if not action_updated:
#             clock.tick(30)  # 降低刷新率，等待输入
#         else:
#             clock.tick(60)  # 有更新时正常刷新
    
#     pygame.quit()
#     plt.close('all')  # 关闭所有matplotlib窗口
#     print("程序退出")
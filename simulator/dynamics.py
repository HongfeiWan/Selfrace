import torch
from typing import Dict, Tuple
import math
import numpy as np
from randomize_components import DrivingStyleSampler

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
        self.along_values = [min_long_jerk, -4, 0, max_long_jerk]
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
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化动力学模型。
        Args:
            config (Dict): 包含车辆物理参数的配置字典。
                支持从simulator.dynamics子配置中读取参数，也支持直接从根级别读取（向后兼容）。
            device (torch.device): 计算设备。
        """
        self.device = device
        
        # 获取dynamics配置，支持嵌套配置结构
        dynamics_config = config.get('dynamics', config)
        
        # 从配置中读取车辆参数
        self.L = dynamics_config.get('vehicle_wheelbase', 2.9)  # 轴距, m
        max_steer_deg = dynamics_config.get('vehicle_max_steer_angle', 35.0)
        self.max_steer_rad = math.radians(max_steer_deg) # 将角度转换为弧度
        
        # 添加车辆几何尺寸参数（用于状态表示和碰撞检测）
        self.vehicle_length = dynamics_config.get('vehicle_length', 4.5)  # 车辆长度, m
        self.vehicle_width = dynamics_config.get('vehicle_width', 2.0)    # 车辆宽度, m
        
        # 添加油门、转向、加速度和速度控制系数
        self.Cthrottle = dynamics_config.get('Cthrottle', 1.0)  # 油门控制系数
        self.Csteer = dynamics_config.get('Csteer', 1.0)        # 转向控制系数
        self.Cacc = dynamics_config.get('Cacc', 1.0)            # 加速度控制系数
        self.Cvel = dynamics_config.get('Cvel', 1.0)            # 速度控制系数
        
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
        
        # Jerk约束参数（用于离散动作空间）
        self.max_longitudinal_jerk = dynamics_config.get('max_longitudinal_jerk', 15.0)  # 最大纵向jerk (m/s³)
        self.min_longitudinal_jerk = dynamics_config.get('min_longitudinal_jerk', -15.0) # 最小纵向jerk (m/s³)
        self.max_lateral_jerk = dynamics_config.get('max_lateral_jerk', 4.0)        # 最大横向jerk (m/s³)
        self.min_lateral_jerk = dynamics_config.get('min_lateral_jerk', -4.0)       # 最小横向jerk (m/s³)
        
        # 数值稳定性参数
        self.curvature_epsilon = float(dynamics_config.get('curvature_epsilon', 1e-8))      # 曲率计算的数值稳定性参数
        self.steering_epsilon = float(dynamics_config.get('steering_epsilon', 1e-5))       # 转向角计算的数值稳定性参数
        self.straight_motion_threshold = float(dynamics_config.get('straight_motion_threshold', 1e-5))  # 直线运动判断阈值 (rad)

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
        
        # 初始化控制状态（如果还没有初始化）
        if self.current_along is None or self.current_along.shape[0] != batch_size:
            self.current_along = torch.zeros(batch_size, device=self.device)
            self.current_alat = torch.zeros(batch_size, device=self.device)
        
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
        along = torch.clamp(new_along, self.min_longitudinal_accel, self.max_longitudinal_accel * self.Cacc)
        alat = torch.clamp(new_alat, self.min_lateral_accel, self.max_lateral_accel) # 横向加速度约束
        
        # 更新当前状态
        self.current_along = along
        self.current_alat = alat
            
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
        new_speed = torch.clamp(new_speed, self.min_velocity, self.max_velocity * self.Cvel)
        
        # 更新前一步的纵向加速度
        self.prev_along = along.clone()

        # 使用时间步内的平均速度进行位移计算，提高精度
        avg_speed = (speed + new_speed) / 2.0

        # 根据横向加速度计算目标转向角
        target_steering_angle = self.calculate_steering_angle(alat, avg_speed)
        
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
        
        # 根据有效转向角更新曲率和横向加速度
        # ρ^(-1) ← tan(φ^(t)) / l_wb
        effective_curvature = torch.tan(steering_angle) / self.L
        # a_lat(t) ← (v(t))^2 * ρ^(-1)
        effective_alat = avg_speed ** 2 * effective_curvature
        
        # 在计算 effective_alat 后，再次应用约束
        effective_alat = torch.clamp(effective_alat, self.min_lateral_accel, self.max_lateral_accel)
        self.current_alat = effective_alat
        
        # 使用自行车动力学模型更新车辆位置
        # 计算位移：d = 0.5(v(t) + v(t-1)) * Δt
        displacement = 0.5 * (new_speed + speed) * dt
        
        # 计算角位移：θ = d * ρ^(-1)
        angular_displacement = displacement * effective_curvature
        
        # 统一的状态更新公式（适用于直线和转弯）
        # 使用平均朝向：yaw + angular_displacement/2
        avg_yaw = yaw + angular_displacement / 2.0
        
        # 位置变化
        dx = displacement * torch.cos(avg_yaw)
        dy = displacement * torch.sin(avg_yaw)
        
        # 偏航角变化
        d_yaw = angular_displacement 
        
        # --- 计算新状态 ---
        new_x = x + dx
        new_y = y + dy
        new_yaw = yaw + d_yaw
        # 将新状态组合成一个张量返回
        new_states = torch.stack([new_x, new_y, new_yaw, new_speed], dim=1)
        return new_states
    
    def reset_control_state(self, batch_size: int = 1):
        """重置控制状态（加速度和转向角）"""
        self.current_along = torch.zeros(batch_size, device=self.device)
        self.current_alat = torch.zeros(batch_size, device=self.device)
        self.current_steering_angle = torch.zeros(batch_size, device=self.device)
        # 重置前一步的纵向加速度
        if hasattr(self, 'prev_along'):
            self.prev_along = torch.zeros(batch_size, device=self.device)

    def calculate_steering_angle(self, alat: torch.Tensor, speed: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """
        根据横向加速度和速度计算转向角
        Args:
            alat (torch.Tensor): 横向加速度
            speed (torch.Tensor): 速度
            epsilon (float): 数值稳定性参数，如果为None则使用配置中的默认值  
        Returns:
            torch.Tensor: 转向角 (弧度)
        """
        # 使用配置中的默认值或传入的参数
        if epsilon is None:
            epsilon = self.steering_epsilon
        
        # 确保 epsilon 是数值类型
        if isinstance(epsilon, str):
            epsilon = float(epsilon)
        elif not isinstance(epsilon, (int, float)):
            epsilon = float(epsilon)
            
        # 计算曲率：ρ^(-1) = alat / max(v^2, ε)
        speed_squared = torch.clamp(speed ** 2, min=epsilon)
        curvature = alat / speed_squared
        
        # 应用数值稳定性：ρ^(-1) ← sign(ρ^(-1)) * max(|ρ^(-1)|, ε)
        curvature_sign = torch.sign(curvature)
        curvature_magnitude = torch.clamp(torch.abs(curvature), min=epsilon)
        curvature = curvature_sign * curvature_magnitude

        # 计算转向角：φ = arctan(ρ^(-1) * lwb)
        steering_angle = torch.atan(curvature * self.L)
        return steering_angle
    
    def get_discrete_action_space(self) -> DiscreteActionSpace:
        """获取离散动作空间"""
        return self.discrete_action_space

# 为了让这个文件可以独立测试，添加一个 main block
if __name__ == '__main__':
    test=DiscreteActionSpace(torch.device('cuda'), config={})
    print(test.get_all_actions())
    print(test.get_action(torch.tensor([0])))
    
    

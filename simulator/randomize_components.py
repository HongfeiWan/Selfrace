import torch
from typing import Dict, Tuple

class DrivingStyleSampler:
    """
    车辆行驶风格抽样器
    从混合均匀分布 X(a) = 0.5U(a^{-1},1) + 0.5U(1,a) 中采样控制系数
    其中 a > 1，用于生成不同的车辆行驶风格
    其中 Cthrottle 和 Csteer 从 X(1.25) 采样，Cacc 从 X(1.5) 采样
    """
    def __init__(self, device: torch.device = None):
        """
        初始化行驶风格抽样器
        Args:
            device (torch.device): 计算设备
        """
        self.device = device if device is not None else torch.device('cuda')

    def sample_mixed_uniform(self, a: float, size: int = 1) -> torch.Tensor:
        """
        从混合均匀分布 X(a) = 0.5U(a^{-1},1) + 0.5U(1,a) 中采样
        Args:
            a (float): 混合均匀分布参数，必须大于1
            size (int): 采样数量
        Returns:
            torch.Tensor: 采样的值，形状为 (size,)
        """
        if a <= 1:
            raise ValueError("Parameter 'a' must be greater than 1")
        lower_bound_1 = 1.0 / a  # 第一个均匀分布的下界
        upper_bound_1 = 1.0    # 第一个均匀分布的上界
        lower_bound_2 = 1.0    # 第二个均匀分布的下界
        upper_bound_2 = a      # 第二个均匀分布的上界
        shape = (size,) if isinstance(size, int) else tuple(size)
        choose_upper = torch.rand(shape, device=self.device) >= 0.5
        lower_samples = torch.empty(shape, device=self.device).uniform_(lower_bound_1, upper_bound_1)
        upper_samples = torch.empty(shape, device=self.device).uniform_(lower_bound_2, upper_bound_2)
        return torch.where(choose_upper, upper_samples, lower_samples)
    
    def sample_driving_style(self, size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样车辆行驶风格参数 Cthrottle 和 Csteer，从 X(1.25) 分布采样
        
        Args:
            size (int): 采样数量
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Cthrottle, Csteer) 参数对
        """
        Cthrottle = self.sample_mixed_uniform(a=1.25, size=size)
        Csteer = self.sample_mixed_uniform(a=1.25, size=size)
        return Cthrottle, Csteer    
    
    def sample_driving_Cacc(self, size: int = 1) -> torch.Tensor:
        """
        采样车辆行驶风格参数 Cacc，从 X(1.5) 分布采样
        Args:
            size (int): 采样数量
        Returns:
            torch.Tensor: Cacc 参数
        """
        Cacc = self.sample_mixed_uniform(a=1.5, size=size)
        return Cacc    
    
    def sample_driving_Cvel(self, size: int = 1) -> torch.Tensor:
        """
        采样车辆行驶风格参数 Cvel，从 X(1.5) 分布采样
        
        Args:
            size (int): 采样数量
        Returns:
            torch.Tensor: Cvel 参数
        """
        Cvel = self.sample_mixed_uniform(a=1.5, size=size)
        return Cvel    

    def sample_driving_style_params(self, *shape: int) -> torch.Tensor:
        """
        批量采样 [Cthrottle, Csteer, Cacc, Cvel]，返回形状为 (*shape, 4) 的张量。
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        Cthrottle = self.sample_mixed_uniform(a=1.25, size=shape)
        Csteer = self.sample_mixed_uniform(a=1.25, size=shape)
        Cacc = self.sample_mixed_uniform(a=1.5, size=shape)
        Cvel = self.sample_mixed_uniform(a=1.5, size=shape)
        return torch.stack([Cthrottle, Csteer, Cacc, Cvel], dim=-1)
    
    def get_distribution_info(self, a: float) -> Dict:
        """
        获取分布信息
        Args:
            a (float): 混合均匀分布参数
        Returns:
            Dict: 包含分布参数的字典
        """
        if a <= 1:
            raise ValueError("Parameter 'a' must be greater than 1")
        
        lower_bound_1 = 1.0 / a
        upper_bound_1 = 1.0
        lower_bound_2 = 1.0
        upper_bound_2 = a
        
        return {
            'a': a,
            'distribution': f"X({a}) = 0.5U({lower_bound_1:.2f}, {upper_bound_1:.2f}) + 0.5U({lower_bound_2:.2f}, {upper_bound_2:.2f})",
            'support': f"[{lower_bound_1:.2f}, {upper_bound_2:.2f}]",
            'expected_value': (lower_bound_1 + upper_bound_1 + lower_bound_2 + upper_bound_2) / 4.0
        }

class RewardParameterSampler:
    """
    参数采样器类，用于从各种分布中采样奖励计算所需的参数。
    该类负责管理所有与奖励计算相关的随机参数采样。
    """
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化参数采样器。
        Args:
            config (Dict): 包含奖励参数的配置字典。
            device (torch.device): 计算设备。
        """
        self.device = device
        self.reward_config = config.get('reward', config)
        # 从配置中加载参数范围
        self._load_parameter_ranges()
        
    def _load_parameter_ranges(self):
        """加载所有参数的范围配置。"""
        # Rgoal相关参数
        self.delta_goal_min = self.reward_config.get('delta_goal_min', 2.0)
        self.delta_goal_max = self.reward_config.get('delta_goal_max', 12.0)
        # 碰撞相关参数
        self.collision_alpha_min = self.reward_config.get('collision_alpha_min', 0.0)
        self.collision_alpha_max = self.reward_config.get('collision_alpha_max', 3.0)
        # 边界相关参数
        self.boundary_alpha_min = self.reward_config.get('boundary_alpha_min', 0.0)
        self.boundary_alpha_max = self.reward_config.get('boundary_alpha_max', 3.0)
        # 舒适度相关参数
        self.comfort_alpha_min = self.reward_config.get('comfort_alpha_min', 0.0)
        self.comfort_alpha_max = self.reward_config.get('comfort_alpha_max', 0.1)
        # 车道对齐相关参数
        self.l_align_alpha_min = self.reward_config.get('l_align_alpha_min', 2.5e-4)
        self.l_align_alpha_max = self.reward_config.get('l_align_alpha_max', 2.5e-2)
        self.vel_align_alpha_min = self.reward_config.get('vel_align_alpha_min', 0.0)
        self.vel_align_alpha_max = self.reward_config.get('vel_align_alpha_max', 1.0)
        # 车道中心对齐相关参数
        self.l_center_alpha_min = self.reward_config.get('l_center_alpha_min', 2.5e-4)
        self.l_center_alpha_max = self.reward_config.get('l_center_alpha_max', 7.5e-3)
        self.center_bias_alpha_min = self.reward_config.get('center_bias_alpha_min', -0.5)
        self.center_bias_alpha_max = self.reward_config.get('center_bias_alpha_max', 0.5)
        # 倒车相关参数
        self.reverse_alpha_min = self.reward_config.get('reverse_alpha_min', 2.5e-4)
        self.reverse_alpha_max = self.reward_config.get('reverse_alpha_max', 7.5e-3)
        # 停止线相关参数
        self.stop_line_alpha_min = self.reward_config.get('stop_line_alpha_min', 0.0)
        self.stop_line_alpha_max = self.reward_config.get('stop_line_alpha_max', 1.0)
        # 原文 C_reward 也包含固定的 velocity/timestep reward 系数；默认 min=max。
        velocity_alpha = self.reward_config.get('velocity_alpha', 2.5e-3)
        timestep_alpha = self.reward_config.get('timestep_alpha', 2.5e-5)
        self.velocity_alpha_min = self.reward_config.get('velocity_alpha_min', velocity_alpha)
        self.velocity_alpha_max = self.reward_config.get('velocity_alpha_max', velocity_alpha)
        self.timestep_alpha_min = self.reward_config.get('timestep_alpha_min', timestep_alpha)
        self.timestep_alpha_max = self.reward_config.get('timestep_alpha_max', timestep_alpha)
    
    def sample_delta_goal(self) -> torch.Tensor:
        """
        从均匀分布采样delta_goal值。
        
        Returns:
            torch.Tensor: 采样的delta_goal值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.delta_goal_min, self.delta_goal_max
        )
    
    def sample_collision_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样碰撞alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.collision_alpha_min, self.collision_alpha_max
        )
    
    def sample_boundary_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样边界alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.boundary_alpha_min, self.boundary_alpha_max
        )
    
    def sample_comfort_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样舒适度alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.comfort_alpha_min, self.comfort_alpha_max
        )
    
    def sample_l_align_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样车道对齐alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.l_align_alpha_min, self.l_align_alpha_max
        )
    
    def sample_vel_align_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样速度对齐alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.vel_align_alpha_min, self.vel_align_alpha_max
        )
    
    def sample_l_center_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样车道中心对齐alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.l_center_alpha_min, self.l_center_alpha_max
        )
    
    def sample_center_bias_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样中心偏置alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.center_bias_alpha_min, self.center_bias_alpha_max
        )
    
    def sample_reverse_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样倒车alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.reverse_alpha_min, self.reverse_alpha_max
        )
    
    def sample_stop_line_alpha(self) -> torch.Tensor:
        """
        从均匀分布采样停止线alpha值。
        
        Returns:
            torch.Tensor: 采样的alpha值
        """
        return torch.empty(1, device=self.device).uniform_(
            self.stop_line_alpha_min, self.stop_line_alpha_max
        )
    
    def sample_all_parameters(self, B: int = 1, M: int = 1) -> torch.Tensor:
        """
        批量采样所有参数并返回张量。
        Args:
            B: 批量大小 B (默认1)
            M: 智能体数 M (默认1)
        Returns:
            torch.Tensor: 形状为 (B, M, 12) 的reward系数张量
        """
        device = self.device
        def uniform(min_v, max_v):
            return torch.empty(B, M, device=device).uniform_(min_v, max_v)
        
        # 采样所有参数
        params = {
            'delta_goal': uniform(self.delta_goal_min, self.delta_goal_max),
            'collision_alpha': uniform(self.collision_alpha_min, self.collision_alpha_max),
            'boundary_alpha': uniform(self.boundary_alpha_min, self.boundary_alpha_max),
            'comfort_alpha': uniform(self.comfort_alpha_min, self.comfort_alpha_max),
            'l_align_alpha': uniform(self.l_align_alpha_min, self.l_align_alpha_max),
            'vel_align_alpha': uniform(self.vel_align_alpha_min, self.vel_align_alpha_max),
            'l_center_alpha': uniform(self.l_center_alpha_min, self.l_center_alpha_max),
            'center_bias_alpha': uniform(self.center_bias_alpha_min, self.center_bias_alpha_max),
            'velocity_alpha': uniform(self.velocity_alpha_min, self.velocity_alpha_max),
            'reverse_alpha': uniform(self.reverse_alpha_min, self.reverse_alpha_max),
            'stop_line_alpha': uniform(self.stop_line_alpha_min, self.stop_line_alpha_max),
            'timestep_alpha': uniform(self.timestep_alpha_min, self.timestep_alpha_max),
        }

        # 将参数堆叠成 (B, M, 12) 的张量；顺序对应原文 reward table。
        reward_coef_list = [
            params['delta_goal'],
            params['collision_alpha'],
            params['boundary_alpha'],
            params['comfort_alpha'],
            params['l_align_alpha'],
            params['vel_align_alpha'],
            params['l_center_alpha'],
            params['center_bias_alpha'],
            params['velocity_alpha'],
            params['reverse_alpha'],
            params['stop_line_alpha'],
            params['timestep_alpha'],
        ]

        return torch.stack(reward_coef_list, dim=-1)  # (B, M, 12)

class VehicleParameterSampler:
    """
    批量车辆参数采样器类，用于world_init中多辆车的批量采样。
    支持批量采样车辆长度、宽度和轴距，并应用约束条件。
    """
    def __init__(self, config: Dict, device: torch.device):
        self.device = device
        sim_config = config.get('simulator', config)
        dynamics_config = sim_config.get('dynamics', config.get('dynamics', sim_config))
        self.vehicle_length_min = dynamics_config.get('vehicle_length_min', 0.8)
        self.vehicle_length_max = dynamics_config.get('vehicle_length_max', 7.0)
        self.vehicle_width_min = dynamics_config.get('vehicle_width_min', 0.8)
        self.vehicle_width_max = dynamics_config.get('vehicle_width_max', 3.0)
        self.wheelbase_ratio = 0.6  # 轴距为长度的0.6倍

    def sample_batch_vehicle_parameters(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        批量采样车辆参数
        Args:
            batch_size: 批量大小，即要采样的车辆数量
        Returns:
            Dict[str, torch.Tensor]: 包含车辆参数的字典
                - 'length': 车辆长度 [batch_size]
                - 'width': 车辆宽度 [batch_size] (已应用约束)
                - 'wheelbase': 轴距 [batch_size]
        """
        # 采样车辆长度
        lengths = torch.empty(batch_size, device=self.device).uniform_(
            self.vehicle_length_min, self.vehicle_length_max
        )
        # 采样车辆宽度
        widths = torch.empty(batch_size, device=self.device).uniform_(
            self.vehicle_width_min, self.vehicle_width_max
        )
        # 应用约束：宽度不能超过长度
        widths = torch.min(widths, lengths)
        # 计算轴距：长度为长度的0.6倍
        wheelbases = lengths * self.wheelbase_ratio
        return {
            'length': lengths,
            'width': widths,
            'wheelbase': wheelbases
        }

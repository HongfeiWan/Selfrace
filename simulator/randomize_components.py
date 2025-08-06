import torch
from typing import Dict, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback

class DrivingStyleSampler:
    """
    车辆行驶风格抽样器
    从混合均匀分布 X(a) = 0.5U(a-1,1) + 0.5U(1,a) 中采样 Cthrottle、Csteer 和 Cacc
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
        从混合均匀分布 X(a) = 0.5U(a-1,1) + 0.5U(1,a) 中采样
        Args:
            a (float): 混合均匀分布参数，必须大于1
            size (int): 采样数量
        Returns:
            torch.Tensor: 采样的值，形状为 (size,)
        """
        if a <= 1:
            raise ValueError("Parameter 'a' must be greater than 1")
        # 计算混合均匀分布的参数
        lower_bound_1 = a - 1  # 第一个均匀分布的下界
        upper_bound_1 = 1.0    # 第一个均匀分布的上界
        lower_bound_2 = 1.0    # 第二个均匀分布的下界
        upper_bound_2 = a      # 第二个均匀分布的上界
        # 生成随机数决定使用哪个均匀分布
        uniform_choice = np.random.random(size)
        # 初始化结果数组
        samples = np.zeros(size)
        # 50% 的概率从第一个均匀分布采样
        mask_1 = uniform_choice < 0.5
        if np.any(mask_1):
            samples[mask_1] = np.random.uniform(
                lower_bound_1, 
                upper_bound_1, 
                size=np.sum(mask_1)
            )
        # 50% 的概率从第二个均匀分布采样
        mask_2 = uniform_choice >= 0.5
        if np.any(mask_2):
            samples[mask_2] = np.random.uniform(
                lower_bound_2, 
                upper_bound_2, 
                size=np.sum(mask_2)
            )
        return torch.tensor(samples, dtype=torch.float32, device=self.device)
    
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
        
        lower_bound_1 = a - 1
        upper_bound_1 = 1.0
        lower_bound_2 = 1.0
        upper_bound_2 = a
        
        return {
            'a': a,
            'distribution': f"X({a}) = 0.5U({lower_bound_1:.2f}, {upper_bound_1:.2f}) + 0.5U({lower_bound_2:.2f}, {upper_bound_2:.2f})",
            'support': f"[{lower_bound_1:.2f}, {upper_bound_2:.2f}]",
            'expected_value': 1.0  # 混合均匀分布的期望值
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
        self.reward_config = config.get('reward', {})
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
    
    def sample_all_parameters(self) -> Dict[str, torch.Tensor]:
        """
        采样所有参数并返回字典。
        
        Returns:
            Dict[str, torch.Tensor]: 包含所有采样参数的字典
        """
        return {
            'delta_goal': self.sample_delta_goal(),
            'collision_alpha': self.sample_collision_alpha(),
            'boundary_alpha': self.sample_boundary_alpha(),
            'comfort_alpha': self.sample_comfort_alpha(),
            'l_align_alpha': self.sample_l_align_alpha(),
            'vel_align_alpha': self.sample_vel_align_alpha(),
            'l_center_alpha': self.sample_l_center_alpha(),
            'center_bias_alpha': self.sample_center_bias_alpha(),
            'reverse_alpha': self.sample_reverse_alpha(),
            'stop_line_alpha': self.sample_stop_line_alpha()
        }

class VehicleParameterSampler:
    """
    批量车辆参数采样器类，用于world_init中多辆车的批量采样。
    支持批量采样车辆长度、宽度和轴距，并应用约束条件。
    """
    def __init__(self, config: Dict, device: torch.device):
        self.device = device
        dynamics_config = config.get('dynamics', {})
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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import yaml
    import os
    
    def load_config_from_yaml(config_path: str) -> dict:
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"成功从 {config_path} 加载配置")
            return config
        except FileNotFoundError:
            print(f"警告: 配置文件 {config_path} 未找到，使用默认配置")
            return {}
        except yaml.YAMLError as e:
            print(f"错误: 解析YAML文件时出错: {e}")
            return {}
    
    def test_reward_parameter_sampler():
        """测试 RewardParameterSampler 类的参数采样功能"""
        print("="*60)
        print("测试 RewardParameterSampler 类")
        print("="*60)
        
        # 从配置文件加载配置
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml')
        test_config = load_config_from_yaml(config_path)
        
        if not test_config:
            print("错误: 无法加载配置，测试终止")
            return
        
        # 初始化采样器
        device = torch.device('cpu')
        sampler = RewardParameterSampler(test_config, device)
        
        # 采样次数
        n_samples = 10000
        
        # 存储所有采样结果
        all_samples = defaultdict(list)
        
        # 进行多次采样
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"已完成 {i}/{n_samples} 次采样")
            
            # 采样所有参数
            sampled_params = sampler.sample_all_parameters()
            
            # 存储每个参数的值
            for param_name, param_value in sampled_params.items():
                all_samples[param_name].append(param_value.item())
        
        # 计算每个参数的统计信息
        for param_name, values in all_samples.items():
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"\n{param_name}:")
            print(f"  理论范围: [{sampler.reward_config.get(f'{param_name}_min', 'N/A')}, {sampler.reward_config.get(f'{param_name}_max', 'N/A')}]")
            print(f"  实际范围: [{min_val:.6f}, {max_val:.6f}]")
            print(f"  均值: {mean_val:.6f}")
            print(f"  标准差: {std_val:.6f}")
            print(f"  中位数: {np.median(values):.6f}")
        
        # 可视化分布
        try:
            # 创建子图
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()
            
            param_names = list(all_samples.keys())
            
            for i, param_name in enumerate(param_names):
                if i < len(axes):
                    values = np.array(all_samples[param_name])
                    
                    # 绘制直方图
                    axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{param_name}')
                    axes[i].set_xlabel('value')
                    axes[i].set_ylabel('frequency')
                    
                    # 添加理论范围线
                    min_range = sampler.reward_config.get(f'{param_name}_min', None)
                    max_range = sampler.reward_config.get(f'{param_name}_max', None)
                    if min_range is not None:
                        axes[i].axvline(min_range, color='red', linestyle='--', alpha=0.7, label=f'min: {min_range}')
                    if max_range is not None:
                        axes[i].axvline(max_range, color='red', linestyle='--', alpha=0.7, label=f'max: {max_range}')
                    
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(param_names), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('reward_parameter_distributions.png', dpi=300, bbox_inches='tight')
            print(f"\n分布图已保存为 'reward_parameter_distributions.png'")
            
            # 显示图形
            plt.show()
            
        except ImportError:
            print("\n注意: matplotlib 未安装，跳过可视化部分")
            print("可以通过以下命令安装: pip install matplotlib")
        
        # 验证均匀分布
        print("\n" + "="*60)
        print("均匀分布验证")
        print("="*60)
        for param_name, values in all_samples.items():
            values = np.array(values)
            min_range = sampler.reward_config.get(f'{param_name}_min', None)
            max_range = sampler.reward_config.get(f'{param_name}_max', None)
            
            if min_range is not None and max_range is not None:
                # 计算理论均值和标准差
                theoretical_mean = (min_range + max_range) / 2
                theoretical_std = (max_range - min_range) / np.sqrt(12)
                
                actual_mean = np.mean(values)
                actual_std = np.std(values)
                
                print(f"\n{param_name}:")
                print(f"  理论均值: {theoretical_mean:.6f}, 实际均值: {actual_mean:.6f}")
                print(f"  理论标准差: {theoretical_std:.6f}, 实际标准差: {actual_std:.6f}")
                print(f"  均值误差: {abs(theoretical_mean - actual_mean):.6f}")
                print(f"  标准差误差: {abs(theoretical_std - actual_std):.6f}")
                print("\nRewardParameterSampler 测试完成！")
    
    def test_driving_style_sampler():
        """测试 DrivingStyleSampler 类的参数采样功能"""
        print("="*60)
        print("测试 DrivingStyleSampler 类")
        print("="*60)
        
        # 初始化采样器
        device = torch.device('cpu')
        sampler = DrivingStyleSampler(device=device)
        
        # 采样次数
        n_samples = 1000
        
        print("测试 sample_mixed_uniform...")
        try:
            samples_1_25 = sampler.sample_mixed_uniform(a=1.25, size=n_samples)
            samples_1_5 = sampler.sample_mixed_uniform(a=1.5, size=n_samples)
            print(f"  X(1.25) 采样结果范围: [{samples_1_25.min():.3f}, {samples_1_25.max():.3f}]")
            print(f"  X(1.5) 采样结果范围: [{samples_1_5.min():.3f}, {samples_1_5.max():.3f}]")
            print("  ✓ sample_mixed_uniform 测试通过")
        except Exception as e:
            print(f"  ✗ sample_mixed_uniform 测试失败: {e}")
        
        print("测试 sample_driving_style...")
        try:
            Cthrottle, Csteer = sampler.sample_driving_style(size=n_samples)
            print(f"  Cthrottle 范围: [{Cthrottle.min():.3f}, {Cthrottle.max():.3f}]")
            print(f"  Csteer 范围: [{Csteer.min():.3f}, {Csteer.max():.3f}]")
            print("  ✓ sample_driving_style 测试通过")
        except Exception as e:
            print(f"  ✗ sample_driving_style 测试失败: {e}")
        
        print("测试 sample_driving_Cacc...")
        try:
            Cacc = sampler.sample_driving_Cacc(size=n_samples)
            print(f"  Cacc 范围: [{Cacc.min():.3f}, {Cacc.max():.3f}]")
            print("  ✓ sample_driving_Cacc 测试通过")
        except Exception as e:
            print(f"  ✗ sample_driving_Cacc 测试失败: {e}")
        
        print("测试 sample_driving_Cvel...")
        try:
            Cvel = sampler.sample_driving_Cvel(size=n_samples)
            print(f"  Cvel 范围: [{Cvel.min():.3f}, {Cvel.max():.3f}]")
            print("  ✓ sample_driving_Cvel 测试通过")
        except Exception as e:
            print(f"  ✗ sample_driving_Cvel 测试失败: {e}")
        
        print("测试 get_distribution_info...")
        try:
            info_1_25 = sampler.get_distribution_info(a=1.25)
            info_1_5 = sampler.get_distribution_info(a=1.5)
            print(f"  X(1.25) 分布信息: {info_1_25}")
            print(f"  X(1.5) 分布信息: {info_1_5}")
            print("  ✓ get_distribution_info 测试通过")
        except Exception as e:
            print(f"  ✗ get_distribution_info 测试失败: {e}")
        
        print("\nDrivingStyleSampler 测试完成！")
    
    def test_vehicle_parameter_sampler():
        """测试 VehicleParameterSampler 类的参数采样功能"""
        print("="*60)
        print("测试 VehicleParameterSampler 类")
        print("="*60)
        
        try:
            # 从配置文件加载配置
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml')
            test_config = load_config_from_yaml(config_path)
            if not test_config:
                print("错误: 无法加载配置，测试终止")
                return
            
            # 使用CUDA设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {device}")
            sampler = VehicleParameterSampler(test_config, device)
            
            # 测试批量采样
            print("测试批量车辆参数采样...")
            batch_size = 1000000
            vehicle_params = sampler.sample_batch_vehicle_parameters(batch_size)
            
            # 打印参数统计信息
            print(f"\n批量采样结果 (batch_size={batch_size}):")
            for param_name, param_tensor in vehicle_params.items():
                print(f"  {param_name}:")
                print(f"    形状: {param_tensor.shape}")
                print(f"    设备: {param_tensor.device}")
                print(f"    最小值: {param_tensor.min():.3f}")
                print(f"    最大值: {param_tensor.max():.3f}")
                print(f"    均值: {param_tensor.mean():.3f}")
                print(f"    标准差: {param_tensor.std():.3f}")
            
            # 验证约束条件
            print("\n验证约束条件:")
            lengths = vehicle_params['length']
            widths = vehicle_params['width']
            wheelbases = vehicle_params['wheelbase']
            
            # 检查宽度约束: width <= length
            width_constraint = torch.all(widths <= lengths)
            print(f"  宽度约束 (width <= length): {'✓' if width_constraint else '✗'}")
            
            # 检查轴距约束: wheelbase = 0.6 * length
            expected_wheelbases = lengths * 0.6
            wheelbase_diff = torch.abs(wheelbases - expected_wheelbases)
            wheelbase_constraint = torch.all(wheelbase_diff < 1e-6)
            print(f"  轴距约束 (wheelbase = 0.6 * length): {'✓' if wheelbase_constraint else '✗'}")
            
            # 绘制三个分布图
            print("\n绘制车辆参数分布图...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 将张量转移到CPU并转换为numpy数组用于绘图
            lengths_cpu = lengths.cpu().numpy()
            widths_cpu = widths.cpu().numpy()
            wheelbases_cpu = wheelbases.cpu().numpy()
            
            # 绘制长度分布
            axes[0].hist(lengths_cpu, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0].set_title('length')
            axes[0].set_xlabel('length (m)')
            axes[0].set_ylabel('frequency')
            axes[0].grid(True, alpha=0.3)
            
            # 标记长度分布的最大最小值
            length_min = lengths_cpu.min()
            length_max = lengths_cpu.max()
            axes[0].axvline(length_min, color='red', linestyle='--', alpha=0.8, label=f'min: {length_min:.3f}')
            axes[0].axvline(length_max, color='red', linestyle='--', alpha=0.8, label=f'max: {length_max:.3f}')
            axes[0].legend()
            
            # 绘制宽度分布
            axes[1].hist(widths_cpu, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_title('width')
            axes[1].set_xlabel('width (m)')
            axes[1].set_ylabel('frequency')
            axes[1].grid(True, alpha=0.3)
            
            # 标记宽度分布的最大最小值
            width_min = widths_cpu.min()
            width_max = widths_cpu.max()
            axes[1].axvline(width_min, color='red', linestyle='--', alpha=0.8, label=f'min: {width_min:.3f}')
            axes[1].axvline(width_max, color='red', linestyle='--', alpha=0.8, label=f'max: {width_max:.3f}')
            axes[1].legend()
            
            # 绘制轴距分布
            axes[2].hist(wheelbases_cpu, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[2].set_title('wheelbase')
            axes[2].set_xlabel('wheelbase (m)')
            axes[2].set_ylabel('frequency')
            axes[2].grid(True, alpha=0.3)
            
            # 标记轴距分布的最大最小值
            wheelbase_min = wheelbases_cpu.min()
            wheelbase_max = wheelbases_cpu.max()
            axes[2].axvline(wheelbase_min, color='red', linestyle='--', alpha=0.8, label=f'min: {wheelbase_min:.3f}')
            axes[2].axvline(wheelbase_max, color='red', linestyle='--', alpha=0.8, label=f'max: {wheelbase_max:.3f}')
            axes[2].legend()
            
            plt.tight_layout()
            plt.show()


            print("\n✓ VehicleParameterSampler 测试完成！")

        except Exception as e:
            print(f"✗ VehicleParameterSampler 测试失败: {e}")
            traceback.print_exc()

    def main():
        # 测试 DrivingStyleSampler
        # test_driving_style_sampler()
        
        # 测试 RewardParameterSampler
        # test_reward_parameter_sampler()

        # 测试 VehicleParameterSampler
        test_vehicle_parameter_sampler()

        print("\n所有测试完成！")
    main()



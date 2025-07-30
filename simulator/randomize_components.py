import torch
from typing import Dict, Tuple
import math
import numpy as np

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
        self.device = device if device is not None else torch.device('cpu')

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

class CarParameterSampler:
    """
    汽车参数采样器类，用于从各种分布中采样汽车参数。
    """
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化参数采样器。
        """
        self.device = device

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
    
    def main():
        # 测试 DrivingStyleSampler
        test_driving_style_sampler()
        
        # 测试 RewardParameterSampler
        test_reward_parameter_sampler()
        
        # 测试其他类（将来添加）
        # test_other_classes()
        print("\n所有测试完成！")
    
    # 运行测试
    if __name__ == "__main__":
        main()



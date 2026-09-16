"""Manual diagnostics moved from simulator/randomize_components.py."""

from _bootstrap import PROJECT_ROOT, add_project_paths

add_project_paths()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import yaml
import os
import traceback
import torch

from randomize_components import DrivingStyleSampler, RewardParameterSampler, VehicleParameterSampler

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
    config_path = PROJECT_ROOT / 'configs' / 'default_config.yaml'
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

        # 存储每个参数的值 (sampled_params 是形状为 (1, 1, 12) 的张量)
        param_names = ['delta_goal', 'collision_alpha', 'boundary_alpha', 'comfort_alpha',
                      'l_align_alpha', 'vel_align_alpha', 'l_center_alpha', 'center_bias_alpha',
                      'velocity_alpha', 'reverse_alpha', 'stop_line_alpha', 'timestep_alpha']
        for j, param_name in enumerate(param_names):
            all_samples[param_name].append(sampled_params[0, 0, j].item())

    # 计算每个参数的统计信息
    for param_name, values in all_samples.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"\n{param_name}:")
        # 获取理论范围
        min_range = getattr(sampler, f'{param_name}_min', 'N/A')
        max_range = getattr(sampler, f'{param_name}_max', 'N/A')
        print(f"  理论范围: [{min_range}, {max_range}]")
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
                min_range = getattr(sampler, f'{param_name}_min', None)
                max_range = getattr(sampler, f'{param_name}_max', None)
                if min_range is not None:
                    axes[i].axvline(min_range, color='red', linestyle='--', alpha=0.7, label=f'min: {min_range}')
                if max_range is not None:
                    axes[i].axvline(max_range, color='red', linestyle='--', alpha=0.7, label=f'max: {max_range}')

                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(len(param_names), len(axes)):
            axes[i].set_visible(False)

        # 确保 images 目录存在
        os.makedirs('./images', exist_ok=True)
        plt.tight_layout()
        plt.savefig('./images/reward_parameter_distributions.png', dpi=300, bbox_inches='tight')
        print("\n分布图已保存为 './images/reward_parameter_distributions.png'")

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
        min_range = getattr(sampler, f'{param_name}_min', None)
        max_range = getattr(sampler, f'{param_name}_max', None)

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
        config_path = PROJECT_ROOT / 'configs' / 'default_config.yaml'
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
        batch_size = 100000  # 减少批量大小以避免内存问题
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

        # 确保 images 目录存在
        os.makedirs('./images', exist_ok=True)

        plt.tight_layout()
        plt.savefig('./images/vehicle_parameter_distributions.png', dpi=300, bbox_inches='tight')
        print("\n分布图已保存为 './images/vehicle_parameter_distributions.png'")
        plt.show()

        print("\n✓ VehicleParameterSampler 测试完成！")

    except Exception as e:
        print(f"✗ VehicleParameterSampler 测试失败: {e}")
        traceback.print_exc()

def main():
    # 测试 DrivingStyleSampler
    test_driving_style_sampler()

    # 测试 RewardParameterSampler
    # test_reward_parameter_sampler()

    # 测试 VehicleParameterSampler
    #  test_vehicle_parameter_sampler()

    print("\n所有测试完成！")
main()

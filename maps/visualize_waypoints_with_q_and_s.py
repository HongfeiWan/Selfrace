#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图Quads可视化脚本

这个脚本调用plot_quads函数来可视化地图的quads。
默认显示quads、waypoints和随机选择的quads的q值（绿色粗体文本）。
支持从配置文件读取地图路径，也可以手动指定地图文件路径。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import yaml
import sys

# 添加maps目录到Python路径，以便导入plot_quads函数
sys.path.append(os.path.join(os.path.dirname(__file__), 'maps'))
from visualize_quads_map import plot_quads


def load_map_data(map_path):
    """
    加载地图数据
    
    Args:
        map_path (str): 地图文件路径
        
    Returns:
        dict: 包含地图数据的字典
    """
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"地图文件不存在: {map_path}")
    
    print(f"正在加载地图文件: {map_path}")
    with open(map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"地图文件加载成功")
    print(f"  - 地图名称: {data.get('map_name', 'Unknown')}")
    print(f"  - Quads数量: {len(data.get('quads', []))}")
    print(f"  - 交通控制点数量: {len(data.get('traffic_controls', []))}")
    print(f"  - Global W-lane Waypoints数量: {len(data.get('global_w_lane_waypoints', []))}")
    
    return data


def plot_waypoints(ax, waypoints_data, color='red', marker='x', markersize=3, alpha=0.7, label='Global W-lane Waypoints', sample_s_interval=20):
    """
    绘制waypoints
    
    Args:
        ax: matplotlib轴对象
        waypoints_data: waypoints数据列表
        color: 颜色
        marker: 标记样式
        markersize: 标记大小
        alpha: 透明度
        label: 图例标签
        sample_s_interval: 每隔多少个waypoint显示一个s值
    """
    if not waypoints_data:
        print("没有waypoints数据可绘制")
        return
    
    # 转换坐标，保持原始坐标系
    coords = []
    s_values = []
    for wp in waypoints_data:
        x = wp.get('x', 0)
        y = wp.get('y', 0)  
        coords.append([x, y])
        
        # 提取s值
        carla_info = wp.get('carla_waypoint_info', {})
        s_value = carla_info.get('s', 0)
        s_values.append(s_value)
    
    coords = np.array(coords)
    
    # 绘制waypoints
    ax.scatter(coords[:, 0], coords[:, 1], 
              c=color, marker=marker, s=markersize, alpha=alpha, 
              label=label, zorder=5)
    
    # 抽样显示s值
    if sample_s_interval > 0 and len(coords) > 0:
        sampled_indices = range(0, len(coords), sample_s_interval)
        for idx in sampled_indices:
            if idx < len(coords) and idx < len(s_values):
                x, y = coords[idx]
                s_val = s_values[idx]
                # 使用透明背景的文本显示s值，保留2位小数
                ax.text(x + 2, y + 2, f's={s_val:.2f}', 
                       fontsize=8, color='black', 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7),
                       ha='left', va='bottom', zorder=6)
    
    print(f"绘制了 {len(coords)} 个waypoints")
    if sample_s_interval > 0:
        print(f"抽样显示了 {len(sampled_indices)} 个s值")


def plot_quads_with_q_values(ax, quads_data, num_samples=5):
    """
    绘制quads并随机抽取指定数量的quads显示它们的q值
    
    Args:
        ax: matplotlib轴对象
        quads_data: quads数据列表
        num_samples: 要显示的q值数量
    """
    if not quads_data:
        print("没有quads数据可绘制")
        return
    
    # 调用原始的plot_quads函数绘制quads
    from visualize_quads_map import plot_quads
    plot_quads(ax, quads_data)
    
    # 随机抽取quads显示q值
    if len(quads_data) > 0:
        import random
        # 随机选择quads
        selected_quads = random.sample(quads_data, min(num_samples, len(quads_data)))
        
        print(f"随机选择了 {len(selected_quads)} 个quads显示q值:")
        
        for i, quad in enumerate(selected_quads):
            # 计算quad的中心点
            vertices = quad['vertices']
            center_x = sum(v['x'] for v in vertices) / len(vertices)
            center_y = sum(v['y'] for v in vertices) / len(vertices) 
            
            # 获取q值
            q_value = quad.get('q', 'N/A')
            q_value = round(q_value, 2)
            # 使用绿色文本标注q值
            ax.text(center_x, center_y, f'q={q_value}', 
                   fontsize=10, color='green', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='green'),
                   ha='center', va='center', zorder=10)
            
            print(f"  Quad {i+1}: q={q_value} at ({center_x:.1f}, {center_y:.1f})")


def visualize_quads_only(map_path, output_path=None, figsize=(20, 20)):
    """
    只可视化地图的quads，不包含交通控制点
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
    """
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取quads数据
    quads_data = data.get('quads', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 调用plot_quads函数绘制quads
    print("正在绘制quads...")
    plot_quads(ax, quads_data)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Quads: {map_name}', fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def visualize_quads_with_waypoints(map_path, output_path=None, figsize=(20, 20), s_sample_interval=20):
    """
    可视化地图的quads和waypoints
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
        s_sample_interval: 每隔多少个waypoint显示一个s值
    """
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取数据
    quads_data = data.get('quads', [])
    waypoints_data = data.get('global_w_lane_waypoints', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制quads
    print("正在绘制quads...")
    plot_quads(ax, quads_data)
    
    # 绘制waypoints
    if waypoints_data:
        print("正在绘制waypoints...")
        plot_waypoints(ax, waypoints_data, color='red', marker='x', markersize=2, alpha=0.6, sample_s_interval=s_sample_interval)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    title = f'Quads + Waypoints: {map_name}'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend()
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def visualize_quads_with_traffic(map_path, output_path=None, figsize=(20, 20)):
    """
    可视化地图的quads和交通控制点
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
    """
    # 导入交通控制点绘制函数
    from visualize_quads_map import plot_traffic_controls
    
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取数据
    quads_data = data.get('quads', [])
    traffic_data = data.get('traffic_controls', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制quads
    print("正在绘制quads...")
    plot_quads(ax, quads_data)
    
    # 绘制交通控制点
    if traffic_data:
        print("正在绘制交通控制点...")
        plot_traffic_controls(ax, traffic_data)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    title = f'地图可视化: {map_name}'
    if traffic_data:
        title += ' (包含交通控制点)'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加图例（如果有交通控制点）
    if traffic_data:
        ax.legend()
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def visualize_all(map_path, output_path=None, figsize=(20, 20), s_sample_interval=20):
    """
    可视化地图的所有元素：quads、waypoints和交通控制点
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
        s_sample_interval: 每隔多少个waypoint显示一个s值
    """
    # 导入交通控制点绘制函数
    from visualize_quads_map import plot_traffic_controls
    
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取数据
    quads_data = data.get('quads', [])
    waypoints_data = data.get('global_w_lane_waypoints', [])
    traffic_data = data.get('traffic_controls', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制quads
    print("正在绘制quads...")
    plot_quads(ax, quads_data)
    
    # 绘制waypoints
    if waypoints_data:
        print("正在绘制waypoints...")
        plot_waypoints(ax, waypoints_data, color='red', marker='x', markersize=2, alpha=0.6, sample_s_interval=s_sample_interval)
    
    # 绘制交通控制点
    if traffic_data:
        print("正在绘制交通控制点...")
        plot_traffic_controls(ax, traffic_data)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    title = f'完整地图可视化: {map_name}'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend()
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def visualize_quads_with_q_values(map_path, output_path=None, figsize=(20, 20), num_q_samples=5):
    """
    可视化地图的quads并显示随机选择的quads的q值
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
        num_q_samples (int): 要显示的q值数量
    """
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取quads数据
    quads_data = data.get('quads', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制quads并显示q值
    print("正在绘制quads并显示q值...")
    plot_quads_with_q_values(ax, quads_data, num_q_samples)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Quads with Q-values: {map_name}', fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def visualize_quads_with_waypoints_and_q_values(map_path, output_path=None, figsize=(20, 20), s_sample_interval=20, num_q_samples=5):
    """
    可视化地图的quads、waypoints和随机选择的quads的q值
    
    Args:
        map_path (str): 地图文件路径
        output_path (str, optional): 输出图片路径，如果为None则显示图片
        figsize (tuple): 图片大小
        s_sample_interval: 每隔多少个waypoint显示一个s值
        num_q_samples (int): 要显示的q值数量
    """
    # 加载地图数据
    data = load_map_data(map_path)
    
    # 提取数据
    quads_data = data.get('quads', [])
    waypoints_data = data.get('global_w_lane_waypoints', [])
    map_name = data.get('map_name', 'Unknown')
    
    if not quads_data:
        print("错误: 地图文件中没有找到quads数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制quads并显示q值
    print("正在绘制quads并显示q值...")
    plot_quads_with_q_values(ax, quads_data, num_q_samples)
    
    # 绘制waypoints
    if waypoints_data:
        print("正在绘制waypoints...")
        plot_waypoints(ax, waypoints_data, color='red', marker='x', markersize=2, alpha=0.6, sample_s_interval=s_sample_interval)
    
    # 设置图形属性
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    title = f'Quads + Waypoints + Q-values: {map_name}'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X  (m)', fontsize=12)
    ax.set_ylabel('Y  (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend()
    
    # 保存或显示图片
    if output_path:
        print(f"正在保存图片到: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print("图片保存完成")
    else:
        print("正在显示图片...")
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="地图Quads可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # F5运行：默认显示quads、waypoints和随机选择的quads的q值
  python waypoints_visualization.py
  
  # 指定地图文件路径，显示quads、waypoints和q值
  python waypoints_visualization.py --map-path ./maps/carla_maps/processed_map_Town01_stitched.json
  
  # 只显示quads和waypoints（不显示q值）
  python waypoints_visualization.py --include-waypoints
  
  # 只显示quads和q值（不显示waypoints）
  python waypoints_visualization.py --show-q-values
  
  # 显示quads和交通控制点
  python waypoints_visualization.py --include-traffic
  
  # 显示所有元素（quads、waypoints、交通控制点）
  python waypoints_visualization.py --include-all
  
  # 自定义q值显示数量
  python waypoints_visualization.py --num-q-samples 10
  
  # 保存图片到文件
  python waypoints_visualization.py --output map_visualization.png
  
  # 自定义图片大小
  python waypoints_visualization.py --figsize 15 15
  
  # 控制s值显示间隔
  python waypoints_visualization.py --s-sample-interval 20
        """
    )
    
    parser.add_argument(
        '--map-path', 
        type=str,
        help='地图文件路径 (默认从配置文件读取)'
    )
    
    parser.add_argument(
        '--include-waypoints',
        action='store_true',
        help='包含waypoints的可视化'
    )
    
    parser.add_argument(
        '--include-traffic',
        action='store_true',
        help='包含交通控制点的可视化'
    )
    
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='包含所有元素的可视化（quads、waypoints、交通控制点）'
    )
    
    parser.add_argument(
        '--show-q-values',
        action='store_true',
        help='显示随机选择的quads的q值'
    )
    
    parser.add_argument(
        '--num-q-samples',
        type=int,
        default=5,
        help='要显示的q值数量 (默认: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出图片文件路径 (如果不指定则显示图片)'
    )
    
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[20, 20],
        metavar=('WIDTH', 'HEIGHT'),
        help='图片大小 (默认: 20 20)'
    )
    
    parser.add_argument(
        '--s-sample-interval',
        type=int,
        default=10,
        help='每隔多少个waypoint显示一个s值 (默认: 20, 设为0禁用)'
    )
    
    args = parser.parse_args()
    
    # 确定地图文件路径
    if args.map_path:
        map_path = os.path.abspath(args.map_path)
    else:
        # 从配置文件读取默认路径
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 获取配置文件中的地图路径
            config_map_path = config['simulator']['map_path']
            
            # 简化处理：直接使用相对于当前工作目录的路径
            if config_map_path.startswith('./'):
                # 移除开头的 ./，直接使用相对路径
                relative_path = config_map_path[2:]  # 移除 './'
                map_path = os.path.abspath(relative_path)
            else:
                map_path = os.path.abspath(config_map_path)
            
            print(f"使用配置文件中的默认地图路径: {map_path}")
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            print("请使用 --map-path 参数指定地图文件路径")
            return
    
    # 检查地图文件是否存在
    if not os.path.exists(map_path):
        print(f"错误: 地图文件不存在: {map_path}")
        return
    
    # 执行可视化
    try:
        if args.include_all:
            visualize_all(map_path, args.output, tuple(args.figsize), args.s_sample_interval)
        elif args.include_waypoints and args.show_q_values:
            visualize_quads_with_waypoints_and_q_values(map_path, args.output, tuple(args.figsize), args.s_sample_interval, args.num_q_samples)
        elif args.include_waypoints:
            visualize_quads_with_waypoints(map_path, args.output, tuple(args.figsize), args.s_sample_interval)
        elif args.include_traffic:
            visualize_quads_with_traffic(map_path, args.output, tuple(args.figsize))
        elif args.show_q_values:
            visualize_quads_with_q_values(map_path, args.output, tuple(args.figsize), args.num_q_samples)
        else:
            # 默认显示quads、waypoints和随机选择的quads的q值在同一张图中
            visualize_quads_with_waypoints_and_q_values(map_path, args.output, tuple(args.figsize), args.s_sample_interval, args.num_q_samples)
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
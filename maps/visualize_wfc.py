"""
可视化WFC生成的地图
支持直接可视化几何数据（lines, circles, arcs），即使没有quads也可以显示
"""
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualize_utils import visualize_map_from_json


def visualize_wfc_map(json_path: str, show_geometry_only: bool = True):
    """
    可视化WFC生成的地图
    
    参数:
    json_path: JSON文件路径
    show_geometry_only: 如果为True，只显示几何数据（lines, circles, arcs），不依赖quads
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取几何数据
    geometry = data.get('geometry', {})
    lines = geometry.get('lines', [])
    circles = geometry.get('circles', [])
    arcs = geometry.get('arcs', [])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'WFC生成的地图: {data.get("map_name", "Unknown")}', fontsize=14, fontweight='bold')
    
    # 统计信息
    stats = {
        'lines': len(lines),
        'circles': len(circles),
        'arcs': len(arcs),
        'quads': len(data.get('quads', [])),
        'w_lanes': len(data.get('w_lanes', [])),
        'oob_points': len(data.get('oob_points', []))
    }
    
    print(f"=== 地图统计信息 ===")
    print(f"直线段: {stats['lines']}")
    print(f"圆形: {stats['circles']}")
    print(f"圆弧: {stats['arcs']}")
    print(f"四边形: {stats['quads']}")
    print(f"W_lanes: {stats['w_lanes']}")
    print(f"OOB点: {stats['oob_points']}")
    
    # 绘制直线
    for line in lines:
        start = line['start']
        end = line['end']
        road_id = line.get('road_id', 0)
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                'b-', linewidth=1.5, alpha=0.7, label='直线' if road_id == lines[0].get('road_id') else '')
        
        # 在起点和终点添加小标记
        ax.plot(start[0], start[1], 'go', markersize=4, alpha=0.6)
        ax.plot(end[0], end[1], 'ro', markersize=4, alpha=0.6)
    
    # 绘制圆形
    for circle in circles:
        center = circle['center']
        radius = circle['radius']
        road_id = circle.get('road_id', 0)
        
        circle_patch = Circle(
            (center[0], center[1]),
            radius,
            fill=False,
            edgecolor='purple',
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            # 只给第一条圆形添加图例标签，避免重复
            label='圆形' if (circles and road_id == circles[0].get('road_id')) else ''
        )
        ax.add_patch(circle_patch)
        
        # 在圆心添加标记
        ax.plot(center[0], center[1], 'mo', markersize=6, alpha=0.8)
    
    # 绘制圆弧
    for arc_data in arcs:
        center = arc_data['center']
        radius = arc_data['radius']
        start_angle = arc_data['start_angle']
        end_angle = arc_data['end_angle']
        road_id = arc_data.get('road_id', 0)
        
        # 计算角度差
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 360
        
        # 绘制圆弧
        arc_patch = Arc(
            (center[0], center[1]),
            2 * radius,
            2 * radius,
            angle=0,
            theta1=start_angle,
            theta2=end_angle,
            color='orange',
            linewidth=2,
            alpha=0.7,
            # 只给第一条圆弧添加图例标签，避免重复
            label='圆弧' if (arcs and road_id == arcs[0].get('road_id')) else ''
        )
        ax.add_patch(arc_patch)
        
        # 绘制起点和终点
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        start_x = center[0] + radius * math.cos(start_rad)
        start_y = center[1] + radius * math.sin(start_rad)
        end_x = center[0] + radius * math.cos(end_rad)
        end_y = center[1] + radius * math.sin(end_rad)
        
        ax.plot(start_x, start_y, 'go', markersize=4, alpha=0.6)
        ax.plot(end_x, end_y, 'ro', markersize=4, alpha=0.6)
        ax.plot(center[0], center[1], 'yo', markersize=4, alpha=0.5)
    
    # 如果有quads，也绘制它们
    quads = data.get('quads', [])
    if quads and not show_geometry_only:
        for quad in quads:
            vertices = quad['vertices']
            vertices_2d = [(v[0], v[1]) for v in vertices]
            from matplotlib.patches import Polygon
            poly = Polygon(vertices_2d, closed=True,
                          facecolor='yellow', edgecolor='black',
                          alpha=0.2, linewidth=0.3)
            ax.add_patch(poly)
    
    # 如果有w_lanes，绘制它们
    w_lanes = data.get('w_lanes', [])
    if w_lanes:
        for w_lane in w_lanes:
            center = w_lane['center']
            ax.plot(center[0], center[1], 'c*', markersize=8, alpha=0.6)
    
    # 如果有oob_points，绘制它们
    oob_points = data.get('oob_points', [])
    if oob_points:
        oob_x = [p['x'] for p in oob_points]
        oob_y = [p['y'] for p in oob_points]
        ax.scatter(oob_x, oob_y, c='red', s=1, alpha=0.3, marker='x')
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # 去重
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    # 添加统计信息文本框
    stats_text = f"直线: {stats['lines']} | 圆形: {stats['circles']} | 圆弧: {stats['arcs']}"
    if stats['quads'] > 0:
        stats_text += f"\n四边形: {stats['quads']} | W_lanes: {stats['w_lanes']} | OOB点: {stats['oob_points']}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 自动调整坐标轴范围
    ax.autoscale()
    
    # 添加说明
    legend_text = ("图例说明:\n"
                   "绿色圆点: 起点\n"
                   "红色圆点: 终点\n"
                   "蓝色线: 直线段\n"
                   "紫色虚线圆: 圆形道路\n"
                   "橙色弧线: 圆弧段\n"
                   "青色星号: W_lane点\n"
                   "红色x: OOB点")
    
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # 默认可视化wfc_generated.json
    json_path = os.path.join(os.path.dirname(__file__), "wfc_generated.json")
    
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        print("请先运行 maps/wave.py 生成地图数据")
        sys.exit(1)
    
    print(f"正在可视化: {json_path}")
    fig, ax = visualize_wfc_map(json_path, show_geometry_only=True)
    
    print("\n图形已显示，关闭窗口后程序结束")
    plt.show()


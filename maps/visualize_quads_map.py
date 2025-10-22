import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import argparse
import os
import math
import yaml
from matplotlib.lines import Line2D


def plot_quads(ax, quads_data):
    """
    Plots road quads on the given matplotlib axes.
    If 'road_id' is present, it colors the quads based on their road.
    """
    if not quads_data:
        print("No quad data to plot.")
        return
    #print(f"Plotting {len(quads_data)} quads...")
    patches = []
    road_ids = []
    # Check if the first element has road_id to decide on coloring strategy
    # 检查quads_data中的每个quad_info是否存在road_id
    has_road_ids = 'road_id' in quads_data[0]

    for quad_info in quads_data: # 遍历quads_data中的每个quad_info
        # 将quad_info中的每个顶点转换为Matplotlib的坐标系
        vertices = np.array([[point['x'], point['y']] for point in quad_info['vertices']])
        polygon = Polygon(vertices, closed=True)
        patches.append(polygon)
        if has_road_ids:
            road_ids.append(quad_info.get('road_id'))

    if has_road_ids and len(set(road_ids)) > 1:
        # Color by road_id if the data is available and there's more than one road.
        # 如果quads_data中存在多个road_id，则根据road_id对quads进行着色
        # print(set(road_ids))
        #print(set(road_ids))
        print(f"Found {len(set(road_ids))} unique roads. Coloring by road_id.")
        unique_road_ids = sorted(list(set(road_ids)))
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(unique_road_ids), vmax=max(unique_road_ids))
        colors = []
        for rid in road_ids:
            if rid != 99999999: #用于找到不同的road_id，并给它们着色
                colors.append(cmap(norm(rid)))
            else:
                colors.append('red')
        p = PatchCollection(patches, alpha=0.5, facecolors=colors, edgecolor='black', linewidth=0.1)
        
        # 调试代码：看某一条路的quad
        # # 从set(road_ids)里面挑一个
        # if set(road_ids):  # 检查集合是否不为空
        #     road_id = list(set(road_ids))[80]
        #     print(f"road_id: {road_id}")
        #     # 挑选出所有road_id为road_id的quad
        #     patches_0 = []
        #     for quad_info in quads_data:
        #         if quad_info.get('road_id') == road_id:
        #             vertices = np.array([[point['x'], -point['y']] for point in quad_info['vertices']])
        #             polygon = Polygon(vertices, closed=True)
        #             patches_0.append(polygon)
        #     p = PatchCollection(patches_0, alpha=1, facecolor='red', edgecolor='black', linewidth=0.1)
        #     ax.add_collection(p)
        # else:
        #     print("Warning: No road_ids found in the data")

    else:
        # Default to a single color if no road_id or only one road.
        # 如果quads_data中不存在road_id或者只有一条road，则使用默认颜色
        p = PatchCollection(patches, alpha=0.1, facecolor='gray', edgecolor='black', linewidth=0.1)
    ax.add_collection(p)


def plot_traffic_controls(ax, traffic_data):
    """Plots traffic lights and their associated stop lines on the given axes."""
    if not traffic_data:
        print("No traffic control data to plot.")
        return
    #print(f"Plotting {len(traffic_data)} traffic lights and stop lines...")
    light_locs_x = []
    light_locs_y = []
    STOP_LINE_WIDTH = 3.5 

    for i, control_info in enumerate(traffic_data):
        loc = control_info['traffic_light_location']
        light_locs_x.append(loc['x'])
        light_locs_y.append(loc['y']) 
        
        for waypoint in control_info['stop_line_waypoints']:
            wp_loc = waypoint['location']
            wp_yaw_deg = waypoint['rotation']['yaw']
            
            rad_yaw = math.radians(wp_yaw_deg)
            
            perp_dx = math.sin(rad_yaw)
            perp_dy = -math.cos(rad_yaw)
            
            half_width = STOP_LINE_WIDTH / 2.0
            p1_x = wp_loc['x'] - perp_dx * half_width
            p1_y_carla = wp_loc['y'] - perp_dy * half_width
            p2_x = wp_loc['x'] + perp_dx * half_width
            p2_y_carla = wp_loc['y'] + perp_dy * half_width
            
            label = 'Stop Line' if i == 0 else ""
            ax.plot([p1_x, p2_x], [p1_y_carla, p2_y_carla], color='red', linewidth=2.5, solid_capstyle='round', label=label, zorder=3)

    ax.scatter(light_locs_x, light_locs_y, c='red', s=50, marker='o', label='Traffic Light', zorder=3)


def visualize_map(unified_data_path):
    """
    (已修复) 从单一的、结构化的JSON文件加载地图数据并创建可视化。
    """
    if not os.path.exists(unified_data_path):
        print(f"Error: Unified map data file not found at '{unified_data_path}'")
        return

    #print(f"Loading unified map data from {unified_data_path}...")
    with open(unified_data_path, 'r') as f:
        data = json.load(f)

    # --- 核心修复: 从加载的字典中提取 'quads' 和 'traffic_controls' 列表 ---
    quads_data = data.get('quads', [])
    traffic_data = data.get('traffic_controls', [])
    map_name = data.get('map_name', 'Unknown')
    # --- 修复结束 ---
    fig, ax = plt.subplots(figsize=(25, 25))
    
    # 将正确的列表传递给绘图函数
    plot_quads(ax, quads_data)
    plot_traffic_controls(ax, traffic_data)
    
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Map Visualization: {map_name}', fontsize=20)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.grid(True)

    if traffic_data:
        ax.legend()

    #print("Displaying map. Close the plot window to exit.")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize unified map data (quads and traffic lights) extracted from CARLA."
    )
    # --- 核心修复: 命令行参数现在指向单一的统一文件 ---
    parser.add_argument(
        'unified_json_file', 
        type=str, 
        nargs='?',  # 允许参数可选
        help='Path to the unified JSON file containing all map data. Example: carla_map_data_Town01.json'
    )
    args = parser.parse_args()

    if args.unified_json_file is not None:
        unified_json_path = os.path.abspath(args.unified_json_file)
    else:
        # 没有输入参数时，读取项目根目录下的 configs/default_config.yaml
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _proj_root = os.path.dirname(_this_dir)
        _cfg_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
        with open(_cfg_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        unified_json_path = os.path.abspath(config['simulator']['map_path'])
        #print(f"未指定输入文件，自动读取默认配置: {unified_json_path}")

    visualize_map(unified_json_path)
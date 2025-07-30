import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys
import os
import yaml
import math

def visualize_processed_map(file_path, sample_size=5):
    """
    Visualize preprocessed map data, allowing interactive inspection of individual quads by clicking.

    Args:
    - file_path (str): Path to the processed JSON file.
    - sample_size (int): Number of samples to randomly select and pre-draw for detailed view.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Loading and parsing file: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("File loaded. Initializing visualization...")

    # --- Extract Data ---
    quads = data.get('quads', [])
    oob_points = data.get('oob_points', [])
    global_w_lane_waypoints = data.get('global_w_lane_waypoints', [])
    traffic_controls = data.get('traffic_controls', [])

    if not quads:
        print("Error: 'quads' data not found in the JSON file.")
        return

    # --- Prepare data for fast lookup and plotting ---
    quads_by_id = {q['polyId']: q for q in quads}
    
    # Calculate centers and directions
    quad_centers_3d = {}
    quad_directions = {}
    for q in quads:
        poly_id = q['polyId']
        verts_3d = np.array([[v['x'], v['y'], v['z']] for v in q['vertices']])
        quad_centers_3d[poly_id] = np.mean(verts_3d, axis=0)
        
        # Vertices order is: front-right, front-left, back-left, back-right
        verts_2d = verts_3d[:, :2]
        front_center = (verts_2d[0] + verts_2d[1]) / 2.0
        back_center = (verts_2d[2] + verts_2d[3]) / 2.0
        quad_directions[poly_id] = front_center - back_center

    # Convert points to numpy arrays for efficient plotting, ensuring consistent order
    poly_ids_sorted = sorted(quad_centers_3d.keys())
    all_quad_centers = np.array([quad_centers_3d[pid] for pid in poly_ids_sorted])
    all_quad_dirs = np.array([quad_directions[pid] for pid in poly_ids_sorted])
    
    oob_coords = np.array([[p['x'], p['y']] for p in oob_points]) if oob_points else np.empty((0, 2))
    wlane_coords = np.array([[p['x'], p['y']] for p in global_w_lane_waypoints]) if global_w_lane_waypoints else np.empty((0, 2))

    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Preprocessed Map Data Visualization (Click a quad to inspect)')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    
    # 生成颜色映射
    colors = ['lightgray' for _ in poly_ids_sorted]
    arrow_colors = ['green' for _ in poly_ids_sorted]

    # 1. Plot base map skeleton (all quad centers)
    ax.scatter(all_quad_centers[:, 0], all_quad_centers[:, 1], 
               c=colors, s=1, label='Quad Centers (Map Skeleton)')
    
    # 2. Plot quad directions as arrows
    ax.quiver(all_quad_centers[:, 0], all_quad_centers[:, 1], 
              all_quad_dirs[:, 0], all_quad_dirs[:, 1], 
              color=arrow_colors, alpha=0.4, width=0.002,
              headwidth=3, headlength=4, label='Quad Directions')

    # 3. Plot global point sets
    if oob_coords.any():
        ax.plot(oob_coords[:, 0], oob_coords[:, 1], '.', color='skyblue', markersize=2, label='Global OOB Points (w_boundary source)')
    if wlane_coords.any():
        ax.plot(wlane_coords[:, 0], wlane_coords[:, 1], 'x', color='salmon', markersize=4, label='Global W_lane Waypoints')
    
    # 4. Plot traffic controls
    if traffic_controls:
        print(f"Plotting {len(traffic_controls)} traffic lights and stop lines...")
        light_locs_x = []
        light_locs_y = []
        STOP_LINE_WIDTH = 3.5 

        for i, control_info in enumerate(traffic_controls):
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
                p1_y = wp_loc['y'] - perp_dy * half_width
                p2_x = wp_loc['x'] + perp_dx * half_width
                p2_y = wp_loc['y'] + perp_dy * half_width
                
                label = 'Stop Line' if i == 0 else ""
                ax.plot([p1_x, p2_x], [p1_y, p2_y], color='red', linewidth=2.5, 
                       solid_capstyle='round', label=label, zorder=3)

        ax.scatter(light_locs_x, light_locs_y, c='red', s=50, marker='o', 
                  label='Traffic Light', zorder=3)

    # --- Interactivity ---
    # Store last plotted elements to clear them on new click
    last_selected_elements = []

    def on_click(event):
        nonlocal last_selected_elements
        # Clear previous highlights
        for element in last_selected_elements:
            element.remove()
        last_selected_elements.clear()

        if event.inaxes != ax:
            return

        click_x, click_y = event.xdata, event.ydata
        
        # Find the closest quad to the click event
        # NOTE: all_quad_centers has sorted poly_ids, so we use it to find the index, then get the poly_id
        distances = np.linalg.norm(all_quad_centers[:, :2] - np.array([click_x, click_y]), axis=1)
        closest_idx = np.argmin(distances)
        closest_poly_id = poly_ids_sorted[closest_idx]
        
        # Get the detailed data for the selected quad
        selected_quad = quads_by_id.get(closest_poly_id)
        if not selected_quad:
            return
            
        print("-" * 50)
        print(f"Clicked on quad: polyId = {selected_quad['polyId']}")
        print(f"  - road_id: {selected_quad.get('road_id', 'N/A')}")
        print(f"  - lane_id: {selected_quad.get('lane_id', 'N/A')}")

        # a. Highlight the selected quad
        quad_verts = np.array([[v['x'], v['y']] for v in selected_quad['vertices']])
        patch = patches.Polygon(quad_verts, closed=True, facecolor='gold', alpha=0.8, zorder=10)
        ax.add_patch(patch)
        last_selected_elements.append(patch)
        
        # b. Visualize its associated w_boundary points
        boundary_ids = selected_quad.get('w_boundary_ids', [])
        if boundary_ids:
            boundary_points = np.array([oob_coords[i] for i in boundary_ids])
            scatter_b = ax.scatter(boundary_points[:, 0], boundary_points[:, 1], c='blue', s=30, zorder=11, label='Associated W_boundary Points')
            last_selected_elements.append(scatter_b)
            print(f"  - Associated {len(boundary_ids)} w_boundary points (blue)")

        # c. Visualize its associated w_lane points
        lane_ids = selected_quad.get('w_lane_ids', [])
        if lane_ids:
            # Ensure IDs are integer indices
            lane_ids = [int(i) for i in lane_ids]
            lane_points = np.array([wlane_coords[i] for i in lane_ids if i < len(wlane_coords)])
            if lane_points.any():
                scatter_l = ax.scatter(lane_points[:, 0], lane_points[:, 1], c='magenta', marker='x', s=50, zorder=12, label='Associated W_lane Points')
                last_selected_elements.append(scatter_l)
                print(f"  - Associated {len(lane_ids)} w_lane points (magenta)")

        fig.canvas.draw_idle()

    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # --- Pre-draw a few samples ---
    print("\n--- Random Samples ---")
    if len(quads) > sample_size:
        sample_indices = random.sample(range(len(quads)), sample_size)
        for i in sample_indices:
            quad = quads[i]
            center = quad_centers_3d[quad['polyId']]
            # Use a simple arrow to mark the sample location
            ax.plot(center[0], center[1], '>', color='green', markersize=4, alpha=0.7)
            
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    """主函数"""
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml')
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    map_path = config.get('simulator', {}).get('map_path')
    if not map_path:
        print("错误: 配置文件中未找到simulator.map_path字段")
        return
    
    # 构建完整路径
    if os.path.isabs(map_path):
        map_path_full = map_path
    else:
        # 相对于项目根目录的路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        map_path_full = os.path.normpath(os.path.join(project_root, map_path.lstrip('./')))
    
    if not os.path.exists(map_path_full):
        print(f"错误: 地图文件不存在: {map_path_full}")
        return
    
    # 执行可视化
    visualize_processed_map(map_path_full)

if __name__ == '__main__':
    main()

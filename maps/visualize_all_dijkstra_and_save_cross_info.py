import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import yaml
import torch
import networkx as nx
from dijkstra import build_graph, find_shortest_path
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
# 导入空间哈希模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.spatial_hash import SpatialHash
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 全局变量：存储所有车道的起点和终点航点
all_start_waypoints = []
all_end_waypoints = []

def plot_quads(ax, quads_data, filtered_quad_indices=None):
    """
    Plots road quads on the given matplotlib axes.
    If 'road_id' is present, it colors the quads based on their road.
    If filtered_quad_indices is provided, quads not in filtered indices will be colored white.
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
        for i, rid in enumerate(road_ids):
            if filtered_quad_indices is not None and i not in filtered_quad_indices:
                # 不在筛选车道中的quad设置为白色
                colors.append('white')
            else:
                #colors.append(cmap(norm(rid))) #彩色
                colors.append('gray')
        p = PatchCollection(patches, alpha=0.5, facecolors=colors, edgecolor='black', linewidth=0.1)
    else:
        # Default to a single color if no road_id or only one road.
        # 如果quads_data中不存在road_id或者只有一条road，则使用默认颜色
        colors = []
        for i in range(len(patches)):
            if filtered_quad_indices is not None and i not in filtered_quad_indices:
                colors.append('red')
            else:
                colors.append('blue')
        p = PatchCollection(patches, alpha=0.1, facecolors=colors, edgecolor='black', linewidth=0.1)
    ax.add_collection(p)

def _group_waypoints_by_lane(waypoints):
    """
    按车道分组并排序航点
    Args:
        waypoints: 航点列表
    Returns:
        dict: 按车道分组的航点字典
    """
    lanes = defaultdict(list)
    for wp in waypoints:
        lanes[(wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'])].append(wp)
    # 对每个车道内的航点进行排序
    for lane_id_tuple, wps_in_lane in lanes.items():
        if wps_in_lane:
            is_reverse_lane = wps_in_lane[0]['carla_waypoint_info']['lane_id'] < 0
            wps_in_lane.sort(key=lambda w: w['carla_waypoint_info']['s'], reverse=is_reverse_lane)
    return lanes

def find_nearest_quad_for_waypoint(waypoint, quad_centers_3d, spatial_hash=None, quad_bounds=None):
    """
    为航点找到最近的quad（通过寻找最近的中心点）
    使用空间哈希加速查询
    Args:
        waypoint: 航点数据
        quad_centers_3d: quad中心点字典
        spatial_hash: 空间哈希对象（可选）
        quad_bounds: quad边界信息（可选）
    Returns:
        tuple: (nearest_poly_id, distance)
    """
    wp_pos = np.array([waypoint['x'], -waypoint['y']])  # 翻转Y轴匹配quads坐标系
    
    # 如果提供了空间哈希，使用哈希查询加速
    if spatial_hash is not None and quad_bounds is not None:
        # 将航点转换为tensor
        wp_tensor = torch.tensor([wp_pos], dtype=torch.float32, device=spatial_hash.device)
        
        # 查询候选quad
        candidate_pairs = spatial_hash.query_points(wp_tensor)
        
        if candidate_pairs.numel() > 0:
            # 获取候选quad的ID
            candidate_quad_ids = candidate_pairs[:, 1].cpu().numpy()
            
            # 在候选quad中寻找最近的
            min_distance = float('inf')
            nearest_poly_id = None
            
            for quad_id in candidate_quad_ids:
                if quad_id in quad_centers_3d:
                    center = quad_centers_3d[quad_id][:2]
                    distance = np.linalg.norm(wp_pos - center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_poly_id = quad_id
            
            if nearest_poly_id is not None:
                return nearest_poly_id, min_distance
    
    # 如果没有空间哈希或查询失败，回退到原始方法
    min_distance = float('inf')
    nearest_poly_id = None
    
    # 遍历所有quad的中心点，找到最近的
    for poly_id in quad_centers_3d.keys():
        center = quad_centers_3d[poly_id][:2]
        distance = np.linalg.norm(wp_pos - center)
        if distance < min_distance:
            min_distance = distance
            nearest_poly_id = poly_id
    return nearest_poly_id, min_distance

def visualize_lane_paths(map_data_path):
    """
    使用Dijkstra算法可视化每条车道的路径，结合航点真实位置和quad网络
    并使用DBSCAN对未筛选的quads进行空间聚类，并绘制聚类结果
    并绘制所有车道的起点和终点
    并绘制所有车道的起点和终点的连线
    并保存cross信息到JSON文件
    Args:
        map_data_path: 地图数据文件路径
    """
    # 清空全局变量
    global all_start_waypoints, all_end_waypoints
    all_start_waypoints.clear()
    all_end_waypoints.clear()
    
    # 加载地图数据
    with open(map_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取航点数据和quads数据
    waypoints = data.get('global_w_lane_waypoints', [])
    quads_data = data.get('quads', [])
    map_name = data.get('map_name', 'Unknown')

    # 构建一个用于查找waypoint_id的索引
    waypoint_id_lookup = {}
    for wp in waypoints:
        key = (wp['x'], wp['y'], wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'], wp['carla_waypoint_info']['s'])
        waypoint_id_lookup[key] = wp.get('waypoint_id', None)
    
    if not waypoints:
        print("错误: 未找到航点数据")
        return
    if not quads_data:
        print("错误: 未找到quads数据")
        
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 翻转quads的Y轴以匹配dijkstra.py中的处理
    for q in quads_data:
        for v in q['vertices']:
            v['y'] = -v['y']
    
    # 使用dijkstra.py的build_graph构建图
    (adjacency_list, quads_by_id, quad_centers_3d, quad_directions, 
     poly_ids_sorted) = build_graph(
         quads_data, neighbor_radius=3, direction_threshold=0.9)# cosθ>0.5，abs(θ)<60°

    # 构建空间哈希以加速quad查询
    print("正在构建空间哈希索引...")
    # 计算所有quad的边界
    all_quad_vertices = []
    for quad in quads_data:
        vertices = np.array([[v['x'], v['y']] for v in quad['vertices']])
        all_quad_vertices.append(vertices)
    # 计算所有顶点的边界
    all_verts = np.vstack(all_quad_vertices)
    min_bounds = torch.tensor(np.min(all_verts, axis=0), dtype=torch.float32)
    max_bounds = torch.tensor(np.max(all_verts, axis=0), dtype=torch.float32)
    # 创建空间哈希
    cell_size = 20.0  # 可以根据需要调整
    device = torch.device('cpu')  # 可视化时使用CPU
    spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, device)
    # 计算每个quad的AABB边界
    quad_bounds = []
    for vertices in all_quad_vertices:
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        quad_bounds.append([min_coords, max_coords])
    quad_bounds_tensor = torch.tensor(quad_bounds, dtype=torch.float32, device=device)
    spatial_hash.build_static_index(quad_bounds_tensor)
    print(f"空间哈希构建完成，网格大小: {spatial_hash.grid_size.cpu().numpy()}")

    # 按车道分组航点
    lanes = _group_waypoints_by_lane(waypoints)
    # 过滤掉首尾waypoint距离太短的车道
    print("正在过滤太短的车道...")
    min_lane_length = 10 # 最小车道长度阈值（米）
    lanes_to_remove = []
    for lane_id, wps_in_lane in lanes.items():
        if len(wps_in_lane) >=2:
            # 计算首尾waypoint之间的距离
            start_wp = wps_in_lane[0]
            end_wp = wps_in_lane[-1]
            distance = np.sqrt((end_wp['x'] - start_wp['x'])**2 + (end_wp['y'] - start_wp['y'])**2)
            
            if distance < min_lane_length:
                lanes_to_remove.append(lane_id)
                print(f"删除车道 {lane_id[0]}_{lane_id[1]}，首尾距离: {distance:.2f}m < {min_lane_length}m")
    # 从lanes字典中删除太短的车道
    for lane_id in lanes_to_remove:
        del lanes[lane_id]
    print(f"过滤完成，剩余 {len(lanes)} 条车道")

    # 收集属于筛选后车道的quad索引
    # 知道剩余车道有哪些road_id
    filtered_quad_indices = set()
    road_id_to_indices = {}
    for idx, quad in enumerate(quads_data):
        rid = quad.get('road_id', None)
        if rid is not None:
            if rid not in road_id_to_indices:
                road_id_to_indices[rid] = []
            road_id_to_indices[rid].append(idx)
    for wps_in_lane in lanes.values():
        for wp in wps_in_lane:
            wp_road_id = wp['carla_waypoint_info']['road_id']
            if wp_road_id in road_id_to_indices:
                filtered_quad_indices.update(road_id_to_indices[wp_road_id])

    # 1. 画quads，未被筛选的为白色
    plot_quads(ax, quads_data, filtered_quad_indices)
    
    # 2. 标记所有waypoints，灰色
    print("正在标记所有waypoints...")
    all_waypoint_x = []
    all_waypoint_y = []
    for wps_in_lane in lanes.values():
        for wp in wps_in_lane:
            all_waypoint_x.append(wp['x'])
            all_waypoint_y.append(-wp['y'])  # 翻转Y轴匹配quads坐标系
    ax.scatter(all_waypoint_x, all_waypoint_y, c='gray', s=10, marker='x', 
              alpha=0.5, zorder=5, label='Waypoints')
    
    # 3. 画车道路径，彩色
    print("正在绘制车道路径（使用Dijkstra算法）...")
    # 为不同车道使用不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(lanes)))
    for i, (lane_id, wps_in_lane) in enumerate(lanes.items()):
        if len(wps_in_lane) < 2:
            continue
        print(f"处理车道 {lane_id[0]}_{lane_id[1]}，包含 {len(wps_in_lane)} 个航点")
        # 为每个航点找到最近的quad
        waypoint_quads = []
        for wp in wps_in_lane:
            nearest_quad, distance = find_nearest_quad_for_waypoint(
                wp, quad_centers_3d, spatial_hash, quad_bounds)
            waypoint_quads.append(nearest_quad)
        # 使用Dijkstra算法连接相邻的航点对应的quads
        color = colors[i % len(colors)]
        color = 'black'
        for j in range(len(waypoint_quads) - 1):
            start_quad = waypoint_quads[j]
            end_quad = waypoint_quads[j + 1]
            
            # 使用Dijkstra算法找到最短路径
            path_nodes, distance = find_shortest_path(adjacency_list, start_quad, end_quad)
            
            if path_nodes:
                # 使用quad中心点绘制路径
                path_x = [quad_centers_3d[node_id][0] for node_id in path_nodes]
                path_y = [quad_centers_3d[node_id][1] for node_id in path_nodes]
                ax.plot(path_x, path_y, color=color, linewidth=2, alpha=0.8)
                
                # 添加方向箭头（在路径中间位置）
                if len(path_nodes) >= 2:
                    mid_idx = len(path_nodes) // 2
                    if mid_idx < len(path_nodes) - 1:
                        # 计算箭头位置和方向
                        arrow_start = quad_centers_3d[path_nodes[mid_idx]][:2]
                        arrow_end = quad_centers_3d[path_nodes[mid_idx + 1]][:2]
                        # 绘制箭头
                        ax.annotate('', xy=(arrow_end[0], arrow_end[1]), 
                                   xytext=(arrow_start[0], arrow_start[1]),
                                   arrowprops=dict(arrowstyle='->', color=color, 
                                                 lw=2, alpha=0.8, mutation_scale=15))
            else:
                # 对于无法生成Dijkstra路径的相邻waypoints，直接连接
                wp1 = wps_in_lane[j]
                wp2 = wps_in_lane[j + 1]
                
                # 转换坐标到quads坐标系（翻转Y轴）
                point1 = (wp1['x'], -wp1['y'])
                point2 = (wp2['x'], -wp2['y'])
                
                # 绘制直接连线
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                       color=color, linewidth=2, alpha=0.8, linestyle='--')
                
                # 在连线中间添加方向箭头
                mid_x = (point1[0] + point2[0]) / 2
                mid_y = (point1[1] + point2[1]) / 2
                
                # 计算箭头方向
                dx = point2[0] - point1[0]
                dy = point2[1] - point1[1]
                
                # 绘制箭头
                ax.annotate('', xy=(mid_x + dx * 0.3, mid_y + dy * 0.3), 
                           xytext=(mid_x - dx * 0.3, mid_y - dy * 0.3),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                         lw=2, alpha=0.8, mutation_scale=15))
                
                print(f"直接连接: 从 {start_quad} 到 {end_quad} 的waypoints")
        
        # 标记车道起点和终点（使用真正的航点坐标），绿色和红色
        if wps_in_lane:
            # 标记起点（第一个航点）
            start_waypoint = wps_in_lane[0]
            start_x, start_y = start_waypoint['x'], -start_waypoint['y']  # 翻转Y轴匹配quads显示
            ax.scatter(start_x, start_y, c='green', s=1, marker='o', 
                      edgecolors='green', linewidth=2, zorder=10, alpha=0.8)
            # 将起点航点添加到全局变量
            all_start_waypoints.append(start_waypoint)
            
            # 标记终点（最后一个航点）
            end_waypoint = wps_in_lane[-1]
            end_x, end_y = end_waypoint['x'], -end_waypoint['y']  # 翻转Y轴匹配quads显示
            ax.scatter(end_x, end_y, c='red', s=1, marker='s', 
                      edgecolors='red', linewidth=2, zorder=10, alpha=0.8)
            # 将终点航点添加到全局变量
            all_end_waypoints.append(end_waypoint)
    
    # 对未筛选的quads进行空间聚类
    print("正在对未筛选的quads进行空间聚类...")
    unfiltered_indices = []
    unfiltered_centers = []
    for i, quad in enumerate(quads_data):
        if i not in filtered_quad_indices:
            unfiltered_indices.append(i)
            if i in quad_centers_3d:
                unfiltered_centers.append(quad_centers_3d[i][:2])  # 只取x,y坐标
    
    # 准备保存cross信息的字典
    cross_data = {
        'filtered_quad_indices': list(filtered_quad_indices)  # 添加直路部分的quad索引
    }
    
    if len(unfiltered_centers) > 0:        # 转换为numpy数组
        unfiltered_centers = np.array(unfiltered_centers)
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_centers = scaler.fit_transform(unfiltered_centers)
        
        # 使用DBSCAN进行聚类
        eps = 0.1  # 可以根据需要调整
        min_samples = 1  # 可以根据需要调整
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_centers)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # 噪声点
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(unfiltered_indices[i])
        
        print(f"空间聚类完成，共找到 {len(clusters)} 个聚类")
        for cluster_id, quad_indices in clusters.items():
            print(f"聚类 {cluster_id}: {len(quad_indices)} 个quads")
        
        # 计算每个聚类的包络矩形
        print("正在计算包络矩形...")
        margin = 3.0  # 统一margin
        
        for cluster_id, quad_indices in clusters.items():
            # 收集该聚类所有quads的顶点
            all_vertices = []
            for quad_idx in quad_indices:
                if quad_idx < len(quads_data):
                    quad = quads_data[quad_idx]
                    vertices = np.array([[point['x'], point['y']] for point in quad['vertices']])
                    all_vertices.extend(vertices)
            if all_vertices:
                all_vertices = np.array(all_vertices)
                # 计算边界框并加margin
                min_x, min_y = np.min(all_vertices, axis=0)
                max_x, max_y = np.max(all_vertices, axis=0)
                min_x -= margin
                min_y -= margin
                max_x += margin
                max_y += margin

                # 绘制包络矩形（已加margin）
                rect_width = max_x - min_x
                rect_height = max_y - min_y
                rect = plt.Rectangle((min_x, min_y), rect_width, rect_height, 
                                   fill=False, edgecolor='red', linewidth=2, linestyle='--',alpha=0.5)
                ax.add_patch(rect)
                
                # 统计在加margin后的cluster包络内的航点
                start_count = 0
                end_count = 0
                for wp in all_start_waypoints:
                    wp_x, wp_y = wp['x'], -wp['y']  # 翻转Y轴匹配quads坐标系
                    if (min_x <= wp_x <= max_x and min_y <= wp_y <= max_y):
                        start_count += 1
                for wp in all_end_waypoints:
                    wp_x, wp_y = wp['x'], -wp['y']
                    if (min_x <= wp_x <= max_x and min_y <= wp_y <= max_y):
                        end_count += 1

                # 在矩形中心添加聚类ID标签和统计信息
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                stats_text = f'Cross {cluster_id}\nStart: {start_count}\nEnd: {end_count}'
                ax.text(center_x, center_y, stats_text, 
                       fontsize=6, ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5))
                print(f"Cluster {cluster_id} Envelope Rectangle: ({min_x:.2f}, {min_y:.2f}) -> ({max_x:.2f}, {max_y:.2f})")
                

                # 使用全局图，判断路径是否在聚类内
                print(f"正在为聚类 {cluster_id} 绘制路径...")
                
                # 找到该聚类内的起点和终点航点
                cluster_start_waypoints = []
                cluster_end_waypoints = []
                
                for wp in all_start_waypoints:
                    wp_x, wp_y = wp['x'], -wp['y']
                    if (min_x <= wp_x <= max_x and min_y <= wp_y <= max_y):
                        cluster_start_waypoints.append(wp)
                
                for wp in all_end_waypoints:
                    wp_x, wp_y = wp['x'], -wp['y']
                    if (min_x <= wp_x <= max_x and min_y <= wp_y <= max_y):
                        cluster_end_waypoints.append(wp)
                
                print(f"聚类 {cluster_id} 内找到 {len(cluster_start_waypoints)} 个起点和 {len(cluster_end_waypoints)} 个终点")
                
                # 初始化该cross的数据结构
                cross_data[f'cross_{cluster_id}'] = {
                    'cross_id': cluster_id,
                    'start_waypoints': [],
                    'end_waypoints': [],
                    'paths': []
                }
                
                # 添加起点航点信息
                for wp in cluster_start_waypoints:
                    key = (wp['x'], wp['y'], wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'], wp['carla_waypoint_info']['s'])
                    waypoint_id = waypoint_id_lookup.get(key, None)
                    if waypoint_id is None:
                        print(f"警告: 未找到起点waypoint的waypoint_id: {key}")
                    cross_data[f'cross_{cluster_id}']['start_waypoints'].append({
                        'waypoint_id': waypoint_id,
                        'x': wp['x'],
                        'y': wp['y'],
                        'road_id': wp['carla_waypoint_info']['road_id'],
                        'lane_id': wp['carla_waypoint_info']['lane_id'],
                        's': wp['carla_waypoint_info']['s']
                    })
                
                # 添加终点航点信息
                for wp in cluster_end_waypoints:
                    key = (wp['x'], wp['y'], wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'], wp['carla_waypoint_info']['s'])
                    waypoint_id = waypoint_id_lookup.get(key, None)
                    if waypoint_id is None:
                        print(f"警告: 未找到终点waypoint的waypoint_id: {key}")
                    cross_data[f'cross_{cluster_id}']['end_waypoints'].append({
                        'waypoint_id': waypoint_id,
                        'x': wp['x'],
                        'y': wp['y'],
                        'road_id': wp['carla_waypoint_info']['road_id'],
                        'lane_id': wp['carla_waypoint_info']['lane_id'],
                        's': wp['carla_waypoint_info']['s']
                    })
                
                # 为每个终点到每个起点绘制路径
                path_count = 0
                for end_wp in cluster_end_waypoints:
                    # 找到终点对应的quad（使用全局图）
                    end_nearest_quad, end_distance = find_nearest_quad_for_waypoint(
                        end_wp, quad_centers_3d, spatial_hash, quad_bounds)
                    
                    for start_wp in cluster_start_waypoints:
                        # 找到起点对应的quad（使用全局图）
                        start_nearest_quad, start_distance = find_nearest_quad_for_waypoint(
                            start_wp, quad_centers_3d, spatial_hash, quad_bounds)
                        
                        # 调试信息
                        if end_nearest_quad is None:
                            print(f"警告: 终点航点 ({end_wp['x']:.2f}, {end_wp['y']:.2f}) 未找到对应的quad")
                        if start_nearest_quad is None:
                            print(f"警告: 起点航点 ({start_wp['x']:.2f}, {start_wp['y']:.2f}) 未找到对应的quad")
                        
                        if end_nearest_quad is not None and start_nearest_quad is not None:
                            # 检查quad是否在图中
                            if end_nearest_quad not in adjacency_list:
                                print(f"警告: 终点quad {end_nearest_quad} 不在图中")
                                continue
                            if start_nearest_quad not in adjacency_list:
                                print(f"警告: 起点quad {start_nearest_quad} 不在图中")
                                continue
                            
                            # 使用全局Dijkstra算法找到路径
                            path_nodes, distance = find_shortest_path(
                                adjacency_list, end_nearest_quad, start_nearest_quad)
                            
                            if path_nodes:
                                # 检查路径是否在聚类内
                                path_in_cluster = True
                                for node_id in path_nodes:
                                    if node_id in quad_centers_3d:
                                        node_center = quad_centers_3d[node_id][:2]
                                        if not (min_x <= node_center[0] <= max_x and min_y <= node_center[1] <= max_y):
                                            path_in_cluster = False
                                            break
                                
                                if path_in_cluster:
                                    # 绘制路径
                                    path_x = [quad_centers_3d[node_id][0] for node_id in path_nodes]
                                    path_y = [quad_centers_3d[node_id][1] for node_id in path_nodes]
                                    
                                    # 使用不同的颜色和线型来区分不同聚类内的路径
                                    cluster_color = plt.cm.Set1(cluster_id % 8)  # 使用不同颜色
                                    cluster_color = 'black'  # 使用不同颜色
                                    ax.plot(path_x, path_y, color=cluster_color, linewidth=1, 
                                           alpha=0.6, linestyle='-', zorder=3)
                                    
                                    # 添加方向箭头
                                    if len(path_nodes) >= 2:
                                        mid_idx = len(path_nodes) // 2
                                        if mid_idx < len(path_nodes) - 1:
                                            arrow_start = quad_centers_3d[path_nodes[mid_idx]][:2]
                                            arrow_end = quad_centers_3d[path_nodes[mid_idx + 1]][:2]
                                            ax.annotate('', xy=(arrow_end[0], arrow_end[1]), 
                                                       xytext=(arrow_start[0], arrow_start[1]),
                                                       arrowprops=dict(arrowstyle='->', color=cluster_color, 
                                                                     lw=1, alpha=0.6, mutation_scale=10))
                                    

                                    # 直接使用Dijkstra路径的quad IDs
                                    path_quad_ids = path_nodes
                                    
                                    # 保存路径信息到cross_data
                                    path_info = {
                                        'from_end_waypoint': {
                                            'x': end_wp['x'],
                                            'y': end_wp['y'],
                                            'road_id': end_wp['carla_waypoint_info']['road_id'],
                                            'lane_id': end_wp['carla_waypoint_info']['lane_id'],
                                            's': end_wp['carla_waypoint_info']['s'],
                                            'from_end_waypoint_id': end_wp['waypoint_id']
                                        },
                                        'to_start_waypoint': {
                                            'x': start_wp['x'],
                                            'y': start_wp['y'],
                                            'road_id': start_wp['carla_waypoint_info']['road_id'],
                                            'lane_id': start_wp['carla_waypoint_info']['lane_id'],
                                            's': start_wp['carla_waypoint_info']['s'],
                                            'to_start_waypoint_id': start_wp['waypoint_id']
                                        },
                                        'path_quad_ids': path_quad_ids,
                                        'distance': float(distance)
                                    }
                                    cross_data[f'cross_{cluster_id}']['paths'].append(path_info)
                                    
                                    path_count += 1
                                else:
                                    print(f"路径从 {end_nearest_quad} 到 {start_nearest_quad} 不在聚类 {cluster_id} 内，跳过")
                            else:
                                print(f"无法找到从 {end_nearest_quad} 到 {start_nearest_quad} 的路径")
                print(f"聚类 {cluster_id} 内成功绘制了 {path_count} 条路径")
    else:
        print("没有未筛选的quads进行聚类")
    # 保存cross数据到JSON文件
    if cross_data:
        # 生成输出文件名
        map_name_without_ext = os.path.splitext(os.path.basename(map_data_path))[0]
        output_filename = f"cross_data_{map_name_without_ext}.json"
        output_path = os.path.join(os.path.dirname(map_data_path), output_filename)
        # 转换NumPy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 转换数据类型
        cross_data_converted = convert_numpy_types(cross_data)
        
        # 保存JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cross_data_converted, f, ensure_ascii=False, indent=2)
        
        print(f"Cross数据已保存到: {output_path}")
        # 计算实际的cross数量（排除filtered_quad_indices）
        actual_cross_count = len([k for k in cross_data.keys() if k.startswith('cross_')])
        print(f"共保存了 {actual_cross_count} 个cross的信息")
        print(f"直路部分包含 {len(cross_data['filtered_quad_indices'])} 个quads")
        for cross_id, data in cross_data.items():
            if cross_id.startswith('cross_'):
                print(f"  Cross {data['cross_id']}: {len(data['start_waypoints'])} 个起点, {len(data['end_waypoints'])} 个终点, {len(data['paths'])} 条路径")
        
        # 构建waypoint级别的有向图
        print("\n正在构建waypoint级别的有向图...")
        G, cross_waypoint_records = build_waypoint_graph(cross_data_converted)
        print(f"节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")
        
        # 将waypoint图结构写入cross_data文件
        def node_no_s(node):
            return [node[0], node[1], node[2], node[3], node[4], node[6]]
        
        # 如果已存在则先删除
        if 'waypoint_graph' in cross_data_converted:
            del cross_data_converted['waypoint_graph']
        
        waypoint_graph = {
            "nodes": [node_no_s(node) for node in G.nodes],
            "edges": [
                [node_no_s(u), node_no_s(v), G[u][v]['distance']]
                for u, v in G.edges
            ]
        }
        cross_data_converted['waypoint_graph'] = waypoint_graph
        
        # 重新保存包含waypoint_graph的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cross_data_converted, f, ensure_ascii=False, indent=2)
        print("waypoint_graph 已写入 cross_data 文件")
        
    else:
        print("没有找到任何cross数据")

    # 设置图形属性
    ax.set_title(f'visualize_lane_paths_with_dijkstra - {map_name}', fontsize=16)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def build_waypoint_graph(cross_data):
    """
    构建cross的waypoint级别的有向图
    1. 同road_id和lane_id的start_waypoints指向end_waypoints
    2. 同一个cross_id下，不同road_id或lane_id的连线仅根据paths内有记录的from_end_waypoint和to_start_waypoint建立
    3. 在每个cross_id内记录from_end_waypoint和to_start_waypoint
    """
    G = nx.DiGraph()
    cross_waypoint_records = defaultdict(lambda: {"from_end_waypoint": [], "to_start_waypoint": []})
    # 收集所有waypoint
    all_start = []
    all_end = []
    cross_start = defaultdict(list)
    cross_end = defaultdict(list)
    for key, value in cross_data.items():
        if key.startswith('cross_'):
            cross_id = value['cross_id']
            for wp in value.get('start_waypoints', []):
                wp_info = dict(wp)
                wp_info['cross_id'] = cross_id
                wp_info['type'] = 'start'
                all_start.append(wp_info)
                cross_start[cross_id].append(wp_info)
            for wp in value.get('end_waypoints', []):
                wp_info = dict(wp)
                wp_info['cross_id'] = cross_id
                wp_info['type'] = 'end'
                all_end.append(wp_info)
                cross_end[cross_id].append(wp_info)
    # 1. 同road_id和lane_id的start_waypoints指向end_waypoints
    for s_wp in all_start:
        for e_wp in all_end:
            if (s_wp['road_id'] == e_wp['road_id'] and s_wp['lane_id'] == e_wp['lane_id']):
                s_id = (s_wp['cross_id'], s_wp['road_id'], s_wp['lane_id'], s_wp['x'], s_wp['y'], s_wp['s'], 'start')
                e_id = (e_wp['cross_id'], e_wp['road_id'], e_wp['lane_id'], e_wp['x'], e_wp['y'], e_wp['s'], 'end')
                distance = abs(e_wp['s'] - s_wp['s'])
                G.add_edge(s_id, e_id, distance=distance)
    # 2. 只根据paths建立cross内部连线
    for key, value in cross_data.items():
        if key.startswith('cross_') and 'paths' in value:
            cross_id = value['cross_id']
            for path in value['paths']:
                s_wp = path['to_start_waypoint']
                e_wp = path['from_end_waypoint']
                s_id = (cross_id, s_wp['road_id'], s_wp['lane_id'], s_wp['x'], s_wp['y'], s_wp['s'], 'start')
                e_id = (cross_id, e_wp['road_id'], e_wp['lane_id'], e_wp['x'], e_wp['y'], e_wp['s'], 'end')
                distance = path['distance']
                G.add_edge(e_id, s_id, distance=distance)
                cross_waypoint_records[cross_id]['from_end_waypoint'].append(e_id)
                cross_waypoint_records[cross_id]['to_start_waypoint'].append(s_id)
    return G, cross_waypoint_records

def find_shortest_path_waypoint(G, start_id, end_id):
    """
    输入:
        G: networkx.DiGraph
        start_id: [cross_id, road_id, lane_id]
        end_id: [cross_id, road_id, lane_id]
    返回:
        路径上所有[cross_id, road_id, lane_id]的列表（不含重复）
    """
    # 找到所有起点和终点的节点（type可以是start或end）
    start_nodes = [n for n in G.nodes if list(n[:3]) == start_id]
    end_nodes = [n for n in G.nodes if list(n[:3]) == end_id]
    if not start_nodes or not end_nodes:
        print("起点或终点不存在")
        return []
    # 搜索所有组合，找最短路径
    min_path = None
    min_length = float('inf')
    for s in start_nodes:
        for e in end_nodes:
            try:
                path = nx.shortest_path(G, source=s, target=e, weight='distance')
                length = nx.shortest_path_length(G, source=s, target=e, weight='distance')
                if length < min_length:
                    min_length = length
                    min_path = path
            except nx.NetworkXNoPath:
                continue
    if min_path is None:
        print("没有可达路径")
        return []
    # 提取[cross_id, road_id, lane_id]，去重
    result = []
    seen = set()
    for n in min_path:
        key = tuple(n[:3])
        if key not in seen:
            result.append(list(key))
            seen.add(key)
    return result

def main():
    """主函数"""
    # 读取配置文件，获取地图路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取地图路径
    map_path = config.get('simulator', {}).get('map_path')
    if not map_path:
        print("错误: 配置文件中未找到 map_path")
        return
    
    # 构建完整的地图路径
    if not os.path.isabs(map_path):
        # 如果是相对路径，则相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(__file__))
        map_path_full = os.path.join(project_root, map_path)
    else:
        map_path_full = map_path
    
    # 检查地图文件是否存在
    if not os.path.exists(map_path_full):
        print(f"错误: 地图文件不存在: {map_path_full}")
        return
    
    print(f"使用地图文件: {map_path_full}")
    
    # 执行可视化
    visualize_lane_paths(map_path_full)

if __name__ == "__main__":
    main() 
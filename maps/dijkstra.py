import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import heapq
from scipy.spatial import KDTree

def build_graph(quads, neighbor_radius=3.0, direction_threshold=0.5, quad_directions=None):
    """
    根据 quad 的中心点构建一个用于路径规划的图。
    Args:
        quads (list): 地图中的 quad 数据列表。
        neighbor_radius (float): 定义邻居的最大搜索半径 (米)。
        一般设置为4m就可以跨车道

        direction_threshold (float): 方向点积阈值，用于判断是否为前进方向。范围为[0,1]
        设置为0.5:cosθ>0.5,θ<60°,即方向向量夹角小于60°

        quad_directions (dict, optional): 自定义的方向向量字典。如果为None，则自动计算。范围为[-1,1]
    Returns:
        tuple: 包含图数据结构的元组 (adjacency_list, quads_by_id, quad_centers_3d, quad_directions, poly_ids_sorted)。
    """
    #print("正在根据 quad 的中心点构建图...")
    # --- 准备 Quads 数据 (包括中心点、方向和ID映射) ---
    quads_by_id = {q['polyId']: q for q in quads}
    quad_centers_3d = {}
    
    # 如果没有提供自定义方向向量，则计算默认方向
    if quad_directions is None:
        quad_directions = {}
        for q in quads:
            poly_id = q['polyId']
            verts_3d = np.array([[v['x'], v['y'], v['z']] for v in q['vertices']])
            quad_centers_3d[poly_id] = np.mean(verts_3d, axis=0)

            # 计算方向向量（从后到前）
            verts_2d = verts_3d[:, :2]
            front_center = (verts_2d[0] + verts_2d[1]) / 2.0  # 前中心
            back_center = (verts_2d[2] + verts_2d[3]) / 2.0   # 后中心
            quad_directions[poly_id] = front_center - back_center

    else:
        # 使用提供的方向向量，但仍需要计算中心点
        for q in quads:
            poly_id = q['polyId']
            verts_3d = np.array([[v['x'], v['y'], v['z']] for v in q['vertices']])
            quad_centers_3d[poly_id] = np.mean(verts_3d, axis=0)
    
    # 将中心点组织成 numpy 数组以便快速查找
    poly_ids_sorted = sorted(quad_centers_3d.keys())
    quad_centers_array = np.array([quad_centers_3d[pid][:2] for pid in poly_ids_sorted])

    # 使用 KDTree 实现快速最近邻搜索
    centers_kdtree = KDTree(quad_centers_array)

    # --- 构建图 ---
    adjacency_list = {pid: [] for pid in poly_ids_sorted}
    
    for i, pid in enumerate(poly_ids_sorted):
        current_center = quad_centers_3d[pid][:2]
        current_direction = quad_directions[pid]
        
        # 归一化当前方向向量（只使用2D部分）
        current_direction_2d = current_direction[:2]
        # 检查方向向量模长，避免除零
        current_dir_norm = np.linalg.norm(current_direction_2d)
        if current_dir_norm < 1e-6:
            continue  # 跳过模长过小的方向向量
        norm_current_dir = current_direction_2d / current_dir_norm
        
        # 查找半径内的所有邻居中心点
        indices = centers_kdtree.query_ball_point(current_center, r=neighbor_radius)
        
        for neighbor_idx in indices:
            neighbor_quad_id = poly_ids_sorted[neighbor_idx]
            # 跳过自己
            if neighbor_quad_id == pid:
                continue

            # --- 方向检查 ---
            neighbor_center = quad_centers_3d[neighbor_quad_id][:2]
            vector_to_neighbor = neighbor_center - current_center
            # 检查向量长度，避免除以零
            if np.linalg.norm(vector_to_neighbor) < 1e-6:
                continue
            norm_vector_to_neighbor = vector_to_neighbor / np.linalg.norm(vector_to_neighbor)
            
            # 计算点积，判断是否为前进方向
            dot_product = np.dot(norm_current_dir, norm_vector_to_neighbor)
            
            # 只有当邻居在前进方向时才添加边,黄实线
            if dot_product > direction_threshold:
                # 计算距离作为边的权重
                dist = np.linalg.norm(vector_to_neighbor)
                # 避免重复添加相同的边
                existing_neighbors = [n for n, _ in adjacency_list[pid]]
                if neighbor_quad_id not in existing_neighbors:
                    adjacency_list[pid].append((neighbor_quad_id, dist)) 

    #print("图构建完成。")
    return adjacency_list, quads_by_id, quad_centers_3d, quad_directions, poly_ids_sorted

def find_shortest_path(graph, start_node, end_node):
    """
    使用 Dijkstra 算法在图中找到最短路径。

    Args:
        graph (dict): 图的邻接表。
        start_node: 起始节点 ID。
        end_node: 结束节点 ID。

    Returns:
        tuple: 包含路径节点列表和总距离的元组。
    """
    if start_node not in graph or end_node not in graph:
        #print("错误: 起点或终点不在图中。")
        return [], float('inf')
    
    # Dijkstra算法（使用quad中心点）
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    previous_nodes = {node: None for node in graph}
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        if current_node == end_node:
            break # 找到终点
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    # 回溯路径
    path = []
    node = end_node
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    if not path or path[-1] != start_node:
        return [], float('inf') # 未找到路径
    return path[::-1], distances[end_node]

def plan_path_and_visualize(config_path=None, visualize=True):
    """
    加载地图、构建图、规划路径并可选择性地进行可视化。
    Args:
        config_path (str): YAML 配置文件的路径。
        visualize (bool): 是否启动 Matplotlib 可视化。
    """
    # 1. 读取配置文件
    if config_path is None:
        # 基于文件位置解析项目根目录
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _proj_root = os.path.dirname(_this_dir)
        config_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功从以下路径加载配置: {config_path}")
    except FileNotFoundError:
        print(f"错误: 在 '{config_path}' 未找到配置文件")
        return None
    except Exception as e:
        print(f"读取或解析配置文件时出错: {e}")
        return None

    # 2. 获取 map_path 并读取地图文件
    map_path = config.get('simulator', {}).get('map_path')
    if not map_path:
        print("错误: 在配置文件中未找到 'simulator.map_path'。")
        return None
    
    # 配置文件中的路径通常是相对于项目根目录的
    print(f"从配置中获取的地图文件路径: {map_path}")

    # 3. 读取地图文件
    if not os.path.exists(map_path):
        print(f"错误: 在 '{map_path}' 未找到地图文件")
        # 尝试从项目根目录构建路径
        # 假设此脚本从项目根目录运行
        if map_path.startswith('./'):
            map_path = map_path[2:]
        
        if not os.path.exists(map_path):
            print(f"错误: 同样在 '{map_path}' 找不到地图文件")
            return None

    try:
        print(f"正在加载并解析地图文件: {map_path} ...")
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("地图文件加载完成。")
    except json.JSONDecodeError:
        print(f"错误: 无法从地图文件 '{map_path}' 解码 JSON。")
        return None
    except Exception as e:
        print(f"读取地图文件时出错: {e}")
        return None

    # 4. 提取数据并翻转Y轴 (Y轴翻转是必须的，以匹配坐标系)

    oob_points = data.get('oob_points', [])
    quads = data.get('quads', [])
    for p in oob_points:
        p['y'] = -p['y']
    for q in quads:
        for v in q['vertices']:
            v['y'] = -v['y']
    
    print(f"找到 {len(oob_points)} 个 'oob_points'。")
    print(f"找到 {len(quads)} 个 'quads'。")

    # 4. 构建图
    (adjacency_list, quads_by_id, quad_centers_3d, quad_directions, 
     poly_ids_sorted) = build_graph(
         quads, neighbor_radius=5, direction_threshold=0.5)
    
    if not visualize:
        print("可视化已禁用。返回图数据。")
        return adjacency_list, quads_by_id, quad_centers_3d, poly_ids_sorted

    # --- 从这里开始是可视化专属代码 ---
    print("可视化已启用。正在设置绘图...")
    
    # --- 新增: 找到路口区域 ---
    from carla_find_cross import find_cross
    cross_quads = find_cross(quads, convert_coordinates=False, distance_threshold=4, eps=2, min_poly_count=4)
    
    # --- 准备数据用于快速查找和绘制 ---
    quads_by_id = {q['polyId']: q for q in quads}
    
    # 计算中心点和方向
    quad_centers_3d = {}
    quad_directions = {}
    for q in quads:
        poly_id = q['polyId']
        verts_3d = np.array([[v['x'], v['y'], v['z']] for v in q['vertices']])
        quad_centers_3d[poly_id] = np.mean(verts_3d, axis=0)
        
        # 顶点顺序：前右、前左、后左、后右
        verts_2d = verts_3d[:, :2]
        front_center = (verts_2d[0] + verts_2d[1]) / 2.0
        back_center = (verts_2d[2] + verts_2d[3]) / 2.0
        quad_directions[poly_id] = front_center - back_center

    # 转换为numpy数组以便高效绘制，确保顺序一致
    poly_ids_sorted = sorted(quad_centers_3d.keys())
    all_quad_centers = np.array([quad_centers_3d[pid] for pid in poly_ids_sorted])
    all_quad_dirs = np.array([quad_directions[pid] for pid in poly_ids_sorted])
    
    oob_coords = np.array([[p['x'], p['y']] for p in oob_points]) if oob_points else np.empty((0, 2))
    
    # --- 设置绘图 ---
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Dijkstra Path Planning')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    
    # 生成路口区域颜色映射
    cross_colors = ['red' if poly_id in cross_quads else 'lightgray' for poly_id in poly_ids_sorted]
    arrow_colors = ['red' if poly_id in cross_quads else 'green' for poly_id in poly_ids_sorted]

    # 1. 绘制基础地图骨架（所有quad中心点）
    ax.scatter(all_quad_centers[:, 0], all_quad_centers[:, 1], 
               c=cross_colors, s=1, label='Quad Centers (Map Skeleton)')
    
    # 2. 绘制quad方向箭头
    ax.quiver(all_quad_centers[:, 0], all_quad_centers[:, 1], 
              all_quad_dirs[:, 0], all_quad_dirs[:, 1], 
              color=arrow_colors, alpha=0.4, width=0.002,
              headwidth=3, headlength=4, label='Quad Directions')

    # 3. 绘制全局点集
    if oob_coords.any():
        ax.plot(oob_coords[:, 0], oob_coords[:, 1], '.', color='skyblue', markersize=2, label='Global OOB Points (w_boundary source)')
        
    # 创建中心点数组和KDTree用于点击检测
    quad_centers_array = np.array([quad_centers_3d[pid][:2] for pid in poly_ids_sorted])
    centers_kdtree = KDTree(quad_centers_array)

    # 初始化变量
    start_poly_id = None
    vehicle_arrow = None
    last_path_elements = []
    vehicle_position_fixed = False  # 状态标志：True表示车辆位置已固定，False表示等待设置车辆位置

    # --- 点击事件处理器 ---
    def on_click(event):
        nonlocal start_poly_id, vehicle_arrow, last_path_elements, vehicle_position_fixed
        
        if event.inaxes != ax: 
            return

        # 清除之前的路径
        for element in last_path_elements:
            element.remove()
        last_path_elements.clear()

        # 找到最近的quad中心点
        _, closest_idx = centers_kdtree.query([event.xdata, event.ydata])
        clicked_poly_id = poly_ids_sorted[closest_idx]
        
        if not vehicle_position_fixed:
            # 第一次点击：设置并固定车辆位置
            start_poly_id = clicked_poly_id
            vehicle_center = quad_centers_3d[start_poly_id][:2]
            vehicle_direction = quad_directions[start_poly_id]
            angle_rad = np.arctan2(vehicle_direction[1], vehicle_direction[0])
            L, W = 4.3, 1.6
            arrow_vertices = np.array([[L / 2, 0], [-L / 2, W / 2], [-L / 2, -W / 2]])
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            transformed_vertices = arrow_vertices @ rotation_matrix.T + vehicle_center
            
            # 移除之前的车辆（如果存在）
            if vehicle_arrow is not None:
                vehicle_arrow.remove()
            
            vehicle_arrow = patches.Polygon(transformed_vertices, closed=True, facecolor='blue', edgecolor='b', alpha=0.9, zorder=20, label='Vehicle Position')
            ax.add_patch(vehicle_arrow)
            
            # 更新图例
            ax.legend()
            vehicle_position_fixed = True
            print(f"车辆位置已固定在 quad ID: {start_poly_id}")
            print("后续点击将设置新的目标点")
            
        else:
            # 后续点击：设置新的目标点并计算路径
            end_poly_id = clicked_poly_id
            
            if start_poly_id is not None:
                path_nodes, total_distance = find_shortest_path(adjacency_list, start_poly_id, end_poly_id)
                
                if path_nodes:
                    # 绘制路径轨迹线（连接quad中心点）
                    path_x = [quad_centers_3d[node_id][0] for node_id in path_nodes]
                    path_y = [quad_centers_3d[node_id][1] for node_id in path_nodes]
                    path_line = ax.plot(path_x, path_y, color='orange', linewidth=1, marker='>', markersize=2, zorder=10, label='Planned Path')
                    last_path_elements.extend(path_line)
                    
                    # 高亮路径经过的quad中心点
                    path_centers_x = [quad_centers_3d[node_id][0] for node_id in path_nodes]
                    path_centers_y = [quad_centers_3d[node_id][1] for node_id in path_nodes]
                    path_centers_scatter = ax.scatter(path_centers_x, path_centers_y, 
                                                    color='orange', s=20, zorder=11, alpha=0.8)
                    last_path_elements.append(path_centers_scatter)
                    print(f"路径规划完成：从固定的车辆位置 {start_poly_id} 到新的目标点 {end_poly_id}")
                    print(f"路径长度：{total_distance:.2f} 米")
                    print(f"经过的quad数量：{len(path_nodes)}")
                else:
                    print(f"无法找到从 {start_poly_id} 到 {end_poly_id} 的路径")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    plan_path_and_visualize(None, visualize=True)
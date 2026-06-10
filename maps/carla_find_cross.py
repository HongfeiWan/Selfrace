import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
#根据曲率寻找交叉路口

def detect_curve(quads, return_indices=False):
    """
    检测相邻两条边向量夹角不为90度的quad
    
    参数:
    quads: quads数据列表
    return_indices: 是否返回索引而不是quad对象
    
    返回:
    如果return_indices为True，返回索引列表；否则返回quad对象列表
    """
    # 检测相邻两条边向量夹角不为90度的grid
    non_90_angle_indices = []
    quad_polys = [np.array([[v['x'], v['y']] for v in q['vertices']]) for q in quads]
    
    for i, poly in enumerate(quad_polys):
        # 计算四个边的向量
        edges = []
        for j in range(4):
            p1 = poly[j]
            p2 = poly[(j + 1) % 4]
            edge_vector = p2 - p1
            edges.append(edge_vector)
        
        # 计算相邻边的夹角
        angles = []
        for j in range(4):
            edge1 = edges[j]
            edge2 = edges[(j + 1) % 4]    
            # 计算向量夹角
            dot_product = np.dot(edge1, edge2)
            norm1 = np.linalg.norm(edge1)
            norm2 = np.linalg.norm(edge2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = dot_product / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)        
        
        # 检查是否有夹角不等于90度（允许1度误差）
        has_non_90_angle = any(abs(angle - 90) > 0.9 for angle in angles)
        if has_non_90_angle:
            non_90_angle_indices.append(i)
    
    if return_indices:
        return non_90_angle_indices
    else:
        return [quads[i] for i in non_90_angle_indices]

def find_road_ids(quads):
    '''
    找到"路"的定义：从quads中找到所有的road_id并且放到一个set里面不重复的东西
    找到quads中所有road_id
    输入：quads，一个包含quad信息的列表
    输出：road_ids，一个包含所有road_id的列表
    '''
    road_ids = []
    # Check if the first element has road_id to decide on coloring strategy
    # 检查quads_data中的每个quad_info是否存在road_id
    has_road_ids = 'road_id' in quads[0]
    for quad_info in quads: # 遍历quads_data中的每个quad_info
        if has_road_ids:
            road_ids.append(quad_info.get('road_id'))
        else:
            print("Warning: No road_ids found in the data")
            exit(1)
    if has_road_ids and len(set(road_ids)) > 1:
        # Color by road_id if the data is available and there's more than one road.
        return road_ids

# 下面三个函数用来扩展路口区域
def get_valid_lane_keys(waypoints):
    """
    从waypoints数据中获取所有有效的(road_id, lane_id)组合
    
    Args:
    - waypoints: 地图中的waypoints数据列表
    
    Returns:
    - set: 有效的(road_id, lane_id)组合集合
    """
    valid_lanes = set()
    for wp in waypoints:
        if 'carla_waypoint_info' not in wp:
            continue
            
        info = wp['carla_waypoint_info']
        lane_key = (info['road_id'], info['lane_id'])
        valid_lanes.add(lane_key)
        
    return valid_lanes
def extend_cross_quads(quads, cross_quad_ids, valid_lanes):
    """
    扩展路口区域的quad集合，将不属于任何车道的quad也加入到路口区域中。
    
    Args:
    - quads: 地图中的所有quads数据
    - cross_quad_ids: 原始路口区域的quad ID集合
    - valid_lanes: 有效的(road_id, lane_id)组合集合
    
    Returns:
    - set: 扩展后的路口区域quad ID集合
    """
    extended_cross_quad_ids = cross_quad_ids.copy()
    
    for quad in quads:
        if 'polyId' not in quad:
            continue
            
        # 获取quad的lane_key（如果存在）
        lane_key = None if not ('road_id' in quad and 'lane_id' in quad) else (quad['road_id'], quad['lane_id'])
        
        # 如果这个quad不在有效车道上，就加入到cross_quad_ids
        if (lane_key is None or lane_key not in valid_lanes) and quad['polyId'] not in extended_cross_quad_ids:
            extended_cross_quad_ids.add(quad['polyId'])
            
    return extended_cross_quad_ids
def process_and_extend_cross_quads(waypoints, quads):
    """
    处理waypoints数据并扩展cross_quad_ids
    
    Args:
    - waypoints: 地图中的waypoints数据列表
    - quads: 地图中的所有quads数据
    
    Returns:
    - set: 扩展后的路口区域quad ID集合
    """
    # 获取有效的车道组合
    valid_lanes = get_valid_lane_keys(waypoints)
    
    # 获取原始路口区域
    cross_quad_ids = set(find_cross(quads))
    print(f"原始路口区域: {len(cross_quad_ids)} 个quad")
    
    # 扩展路口区域
    extended_cross_quad_ids = extend_cross_quads(quads, cross_quad_ids, valid_lanes)
    print(f"扩展后路口区域: {len(extended_cross_quad_ids)} 个quad")
    
    return extended_cross_quad_ids


def compute_poly_center(vertices):
    '''
    计算多边形的中心点
    输入：vertices，一个包含多边形顶点的列表
    输出：center，一个包含中心点坐标的列表
    '''
    verts = np.array(vertices)
    return np.mean(verts, axis=0)

def cluster_polys_by_distance(poly_centers, eps=10):
    """
    使用DBSCAN算法对poly中心点进行聚类
    eps: 邻域半径，决定哪些点被认为是相邻的
    """
    if len(poly_centers) == 0:
        return []
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=eps, min_samples=1).fit(poly_centers)
    labels = clustering.labels_
    
    # 将相同标签的点分组
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)
    return list(clusters.values())

def compute_cluster_bounding_box(cluster_indices, quad_polys):
    """
    计算聚类中所有poly的边界框
    返回: (min_x, min_y, max_x, max_y)
    """
    if not cluster_indices:
        return None
    
    all_vertices = []
    for idx in cluster_indices:
        poly = quad_polys[idx]
        all_vertices.extend(poly)
    
    all_vertices = np.array(all_vertices)
    min_x, min_y = np.min(all_vertices, axis=0)
    max_x, max_y = np.max(all_vertices, axis=0)
    
    return (min_x, min_y, max_x, max_y)

def boxes_intersect(box1, box2):
    """
    检测两个边界框是否相交
    box1, box2: (min_x, min_y, max_x, max_y)
    """
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2
    
    # 检查是否相交
    return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)

def boxes_distance(box1, box2):
    """
    计算两个边界框之间的最小距离
    box1, box2: (min_x, min_y, max_x, max_y)
    返回: 最小距离
    """
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2
    
    # 如果相交，距离为0
    if boxes_intersect(box1, box2):
        return 0.0
    
    # 计算X方向的距离
    if max_x1 < min_x2:
        dx = min_x2 - max_x1
    elif max_x2 < min_x1:
        dx = min_x1 - max_x2
    else:
        dx = 0.0
    
    # 计算Y方向的距离
    if max_y1 < min_y2:
        dy = min_y2 - max_y1
    elif max_y2 < min_y1:
        dy = min_y1 - max_y2
    else:
        dy = 0.0
    
    # 返回欧几里得距离
    return np.sqrt(dx*dx + dy*dy)

def merge_intersecting_boxes(clusters, quad_polys, colored_quads, distance_threshold=10.0):
    """
    合并相交或距离小于阈值的cluster框框
    返回: 合并后的cluster列表
    """
    if not clusters:
        return []
    # 计算所有cluster的边界框
    cluster_boxes = []
    for cluster in clusters:
        cluster_original_indices = [colored_quads[idx] for idx in cluster]
        bounding_box = compute_cluster_bounding_box(cluster_original_indices, quad_polys)
        if bounding_box:
            cluster_boxes.append({
                'cluster': cluster,
                'box': bounding_box
            })
    if not cluster_boxes:
        return []
    # 合并相交或距离小于阈值的框框
    merged_clusters = []
    used_indices = set()
    
    for i, cluster_box in enumerate(cluster_boxes):
        if i in used_indices:
            continue
        
        # 开始一个新的合并组
        current_group = [cluster_box]
        used_indices.add(i)
        
        # 查找所有与当前组相交或距离小于阈值的框框
        changed = True
        while changed:
            changed = False
            for j, other_cluster_box in enumerate(cluster_boxes):
                if j in used_indices:
                    continue
                
                # 检查是否与当前组中的任何框框相交或距离小于阈值
                for group_box in current_group:
                    distance = boxes_distance(group_box['box'], other_cluster_box['box'])
                    if distance <= distance_threshold:
                        current_group.append(other_cluster_box)
                        used_indices.add(j)
                        changed = True
                        break
        
        # 合并当前组中的所有cluster
        merged_cluster = []
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        
        for group_box in current_group:
            merged_cluster.extend(group_box['cluster'])
            box_min_x, box_min_y, box_max_x, box_max_y = group_box['box']
            min_x = min(min_x, box_min_x)
            min_y = min(min_y, box_min_y)
            max_x = max(max_x, box_max_x)
            max_y = max(max_y, box_max_y)
        
        merged_clusters.append({
            'cluster': merged_cluster,
            'box': (min_x, min_y, max_x, max_y)
        })
    return merged_clusters

def find_quads_in_bounding_boxes(quads, bounding_boxes):
    """
    找到所有在指定边框内的 quad_id
    参数:
    quads: quad数据列表
    bounding_boxes: 边框坐标列表，每个元素为 (min_x, min_y, max_x, max_y)
    
    返回:
    list: 在边框内的所有 quad_id 列表
    """
    quad_ids_in_boxes = set()
    
    for quad in quads:
        # 计算 quad 的中心点
        center = compute_poly_center([[v['x'], v['y']] for v in quad['vertices']])
        x, y = center[0], center[1]
        
        # 检查中心点是否在任何边框内
        for bbox in bounding_boxes:
            min_x, min_y, max_x, max_y = bbox
            if min_x <= x <= max_x and min_y <= y <= max_y:
                quad_ids_in_boxes.add(quad['polyId'])
                break
    
    return list(quad_ids_in_boxes)

def find_cross(quads, waypoints=None, eps=2.0, distance_threshold=4, min_poly_count=4, visualize=False, title="Grid Centers with Non-90° Angle Detection and Clustering", convert_coordinates=True):
    """
    检测非90度夹角的quad并进行聚类，返回在路口边框范围内的所有 quad_id
    参数:
    quads: quad数据列表
    waypoints: 可选，地图中的waypoints数据列表，用于扩展路口区域quad
    eps: DBSCAN聚类的邻域半径，目前这个4m是比较合适的
    distance_threshold: 合并cluster的距离阈值
    min_poly_count: 最小poly数量阈值，小于此数量的cluster会被过滤
    visualize: 是否显示可视化图像
    title: 可视化图像的标题
    convert_coordinates: 是否进行坐标系转换（y取反），默认为True
    返回:
    list: 在路口边框内的所有 quad_id 列表
    """
    if not quads:
        return []
    
    # 坐标系转换（y取反）- 根据参数决定是否执行
    processed_quads = []
    for q in quads:
        processed_quad = copy.deepcopy(q)
        if convert_coordinates:
            # 坐标系转换（y取反）
            for v in processed_quad['vertices']:
                v['y'] = -v['y']
        else:
            # 不进行坐标系转换
            processed_quad = q.copy()
        processed_quads.append(processed_quad)
    
    # 提取多边形和计算中心点
    quad_polys = [np.array([[v['x'], v['y']] for v in q['vertices']]) for q in processed_quads]
    quad_centers = np.array([compute_poly_center([[v['x'], v['y']] for v in q['vertices']]) for q in processed_quads])
    poly_ids = [q['polyId'] for q in processed_quads]
    
    # 使用通用函数检测非90度夹角的quad
    colored_quads = detect_curve(processed_quads, return_indices=True)
    
    # 对红色poly进行聚类
    cross_quad_ids = []
    if colored_quads:
        colored_centers = quad_centers[colored_quads]
        clusters = cluster_polys_by_distance(colored_centers, eps=eps)
        # 合并相交的cluster框框
        merged_clusters = merge_intersecting_boxes(clusters, quad_polys, colored_quads, distance_threshold)
        # 过滤掉poly数量太少的cluster(避免由于小弯道导致的小的cluster)
        final_bounding_boxes = []
        for merged_cluster in merged_clusters:
            cluster = merged_cluster['cluster']
            if len(cluster) >= min_poly_count:
                final_bounding_boxes.append(merged_cluster['box'])
        print(f"交叉路口区域(方形)数量: {len(final_bounding_boxes)} ")
        # 获取路口区域的quad_ids
        cross_quad_ids = find_quads_in_bounding_boxes(processed_quads, final_bounding_boxes)

        
        # 如果提供了waypoints数据，则扩展路口区域
        if waypoints:
            valid_lanes = get_valid_lane_keys(waypoints)
            # 将cross_quad_ids转换为集合
            cross_quad_ids = extend_cross_quads(processed_quads, set(cross_quad_ids), valid_lanes)
            # 将结果转换回列表
            cross_quad_ids = list(cross_quad_ids)
            
        # 可视化（如果需要）
        if visualize:
            visualize_clusters(quad_polys, final_bounding_boxes, cross_quad_ids, processed_quads, title)

    return cross_quad_ids

def visualize_clusters(quad_polys, final_bounding_boxes, cross_quad_ids, processed_quads, title):
    """
    可视化聚类结果
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')

    # 画所有quad（grid）
    for i, poly in enumerate(quad_polys):
        # 检查当前quad的polyId是否在cross_quad_ids中
        if i < len(processed_quads) and processed_quads[i]['polyId'] in cross_quad_ids:
            # cross_quad_ids内的quad用红色填充
            patch = patches.Polygon(poly, closed=True, facecolor='red', alpha=0.6, edgecolor='black', linewidth=0.5)
        else:
            # 其他quad用灰色
            patch = patches.Polygon(poly, closed=True, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5)
        ax.add_patch(patch)

    # 为每个边界框绘制矩形
    for i, bounding_box in enumerate(final_bounding_boxes):
        min_x, min_y, max_x, max_y = bounding_box
        width = max_x - min_x
        height = max_y - min_y
        
        # 绘制边界框
        color = 'red'
        rect = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=1, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # 添加聚类标签
        ax.text(min_x, max_y + 5, f'ROI Region {i+1}', 
               fontsize=8, color=color, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.2))

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.autoscale_view()
    plt.show()

if __name__ == "__main__":
    # 读取 JSON 文件
    map_file = './maps/processed_map_Town01_stitched.json'
    if not os.path.exists(map_file):
        print(f"找不到地图文件: {map_file}")
        exit(1)

    with open(map_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    quads = data.get('quads', [])
    waypoints = data.get('global_w_lane_waypoints', [])
    
    # 调用find_cross函数
    quad_ids = find_cross(
        quads=quads,
        waypoints=waypoints,
        eps=2.0,
        distance_threshold=4,
        min_poly_count=4,
        visualize=True,
        title='Grid Centers with Non-90° Angle Detection and Clustering'
    )

    # 根据quad_ids，找到road_ids，然后根据road_ids，找到弯路和直路
    road_ids = find_road_ids(quads) 
    curve_quads = detect_curve(quads)  # 现在默认返回quad对象列表
    curve_road_ids = find_road_ids(curve_quads)
    print(f"找到{len(set(road_ids))}条路")
    print(f"弯路数量: {len(set(curve_road_ids))} ")
    print(f"直路数量: {len(set(road_ids))-len(set(curve_road_ids))} ")

    
    

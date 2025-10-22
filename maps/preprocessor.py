import json
import numpy as np
import argparse
import os
from collections import defaultdict, deque
import sys
import heapq
from multiprocessing import Pool, cpu_count
from functools import partial
import time

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# --- 全局配置参数 ---
CONFIG = {
    "CELL_SIZE": 20.0, # 20m，用于生成3D空间网格
    "OOB_NUDGE_DISTANCE": 0.1, # 0.1m，用于生成OOB点
    "JUNCTION_CONNECT_RADIUS": 5.0, # 5m，用于连接交叉路口节点
    "W_LANE_SAMPLE_DISTANCE": 40.0,     # 40m，用于生成w_lane航点
    "W_LANE_ASSOCIATION_RADIUS": 250.0, # 250m，用于将w_lane航点与相邻航点关联
    "W_BOUNDARY_ASSOCIATION_RADIUS": 15.0,  # 15m，用于将w_lane航点与边界关联
    "PARALLEL_LANE_DOT_THRESHOLD": 0.9,  # 0.9，用于判断两条车道是否平行
    "JUNCTION_LANE_THRESHOLD": 3,  # 3条车道，用于寻找交叉路口节点
    # --- 新增: w_lane 降采样阈值 ---
    "W_LANE_DOWNSAMPLE_LONGITUDINAL_THRESHOLD": 40, # 沿道路方向距离
    "W_LANE_DOWNSAMPLE_LATERAL_THRESHOLD": 2.0,   # 垂直道路方向距离 (此版本中未使用)
}

# --- 3D几何计算工具函数 ---
def get_quad_center_3d(vertices):
    return np.mean(np.array([[p['x'], p['y'], p['z']] for p in vertices]), axis=0)

def is_point_in_quad_2d(point_2d, quad_vertices_2d):
    p = np.array(point_2d)
    v = np.array(quad_vertices_2d)
    v_edges = np.roll(v, -1, axis=0) - v
    v_points = p - v
    cross_products = v_edges[:, 0] * v_points[:, 1] - v_edges[:, 1] * v_points[:, 0]
    return np.all(cross_products >= 0) or np.all(cross_products <= 0)

# --- 核心数据结构与算法 ---
class SpatialGrid3D:
    # ... (此类无需改动) ...
    def __init__(self, quads, cell_size, name="quads"):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        all_verts_2d = np.array([v for q in quads for v in [[p['x'], p['y']] for p in q['vertices']]])
        if all_verts_2d.shape[0] == 0:
            self.min_coord, self.max_coord = np.array([0,0]), np.array([0,0])
        else:
            self.min_coord = np.min(all_verts_2d, axis=0) - cell_size
            self.max_coord = np.max(all_verts_2d, axis=0) + cell_size
        print(f"Populating 3D spatial grid for {name}...")
        for i, quad in enumerate(quads):
            sys.stdout.write(f"\rGrid processing for {name}: {i+1}/{len(quads)}")
            sys.stdout.flush()
            poly_id = quad['polyId']
            verts_2d = np.array([[p['x'], p['y']] for p in quad['vertices']])
            min_q, max_q = np.min(verts_2d, axis=0), np.max(verts_2d, axis=0)
            for i_grid in range(int((min_q[0] - self.min_coord[0]) // self.cell_size), int((max_q[0] - self.min_coord[0]) // self.cell_size) + 1):
                for j_grid in range(int((min_q[1] - self.min_coord[1]) // self.cell_size), int((max_q[1] - self.min_coord[1]) // self.cell_size) + 1):
                    self.grid[(i_grid, j_grid)].append(poly_id)
        print(f"\n{name.capitalize()} grid populated.")

    def get_candidates(self, point_2d):
        grid_x = int((point_2d[0] - self.min_coord[0]) // self.cell_size)
        grid_y = int((point_2d[1] - self.min_coord[1]) // self.cell_size)
        return self.grid.get((grid_x, grid_y), [])

class PointIndexGrid:
    # ... (此类无需改动) ...
    def __init__(self, points, cell_size, name="points"):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.points = points
        all_points_2d = np.array([[p['x'], p['y']] for p in points])
        if all_points_2d.shape[0] == 0:
            self.min_coord, self.max_coord = np.array([0,0]), np.array([0,0])
        else:
            self.min_coord = np.min(all_points_2d, axis=0) - cell_size
            self.max_coord = np.max(all_points_2d, axis=0) + cell_size
        print(f"Populating spatial grid for {name}...")
        for i, p in enumerate(points):
            sys.stdout.write(f"\r{name.capitalize()} Grid processing: {i+1}/{len(points)}")
            sys.stdout.flush()
            point_id = p.get('waypoint_id', i) # 使用 waypoint_id (如果存在)
            grid_x = int((p['x'] - self.min_coord[0]) // self.cell_size)
            grid_y = int((p['y'] - self.min_coord[1]) // self.cell_size)
            self.grid[(grid_x, grid_y)].append(point_id)
        print(f"\n{name.capitalize()} grid populated.")

    def get_nearby_ids(self, point_2d, radius):
        min_p = point_2d - radius
        max_p = point_2d + radius
        candidate_ids = set()
        start_ix = int((min_p[0] - self.min_coord[0]) // self.cell_size)
        end_ix = int((max_p[0] - self.min_coord[0]) // self.cell_size)
        start_jy = int((min_p[1] - self.min_coord[1]) // self.cell_size)
        end_jy = int((max_p[1] - self.min_coord[1]) // self.cell_size)
        for i in range(start_ix, end_ix + 1):
            for j in range(start_jy, end_jy + 1):
                candidate_ids.update(self.grid.get((i, j), []))
        nearby_ids = []
        for idx in candidate_ids:
            # 确保索引在列表范围内
            if idx < len(self.points):
                p = self.points[idx]
                dist = np.linalg.norm(point_2d - np.array([p['x'], p['y']]))
                if dist <= radius:
                    nearby_ids.append(idx)
        return nearby_ids

def run_dijkstra(graph, start_node):
    # ... (此函数无需改动) ...
    distances = {node: float('infinity') for node in graph}
    if start_node not in distances: return {}
    distances[start_node] = 0
    pq = [(0, start_node)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]: continue
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight
            if distance < distances.get(neighbor, float('infinity')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# --- 数据生成函数 ---
def generate_oob_points(quads, grid, quads_by_id):
    # ... (此函数无需改动) ...
    print("Generating Out-of-Bounds (OOB) points...")
    oob_points = []
    nudge = CONFIG["OOB_NUDGE_DISTANCE"]
    for i, quad in enumerate(quads):
        sys.stdout.write(f"\rOOB processing: {i+1}/{len(quads)}")
        sys.stdout.flush()
        verts_3d = np.array([[p['x'], p['y'], p['z']] for p in quad['vertices']])
        for i_edge in range(4):
            p1_3d, p2_3d = verts_3d[i_edge], verts_3d[(i_edge + 1) % 4]
            mid_point_3d = (p1_3d + p2_3d) / 2.0
            mid_point_2d = mid_point_3d[:2]
            edge_vec_2d = p2_3d[:2] - p1_3d[:2]
            normal_2d = np.array([edge_vec_2d[1], -edge_vec_2d[0]])
            norm = np.linalg.norm(normal_2d)
            if norm < 1e-6: continue
            normal_2d /= norm
            test_point_2d = mid_point_2d + normal_2d * 0.01
            if is_point_in_quad_2d(test_point_2d, [[v['x'], v['y']] for v in quad['vertices']]):
                normal_2d = -normal_2d
            oob_candidate_2d = mid_point_2d + normal_2d * nudge
            is_inside_any = False
            for cand_poly_id in grid.get_candidates(oob_candidate_2d):
                if is_point_in_quad_2d(oob_candidate_2d, [[v['x'], v['y']] for v in quads_by_id[cand_poly_id]['vertices']]):
                    is_inside_any = True
                    break
            if not is_inside_any:
                oob_points.append({'x': oob_candidate_2d[0], 'y': oob_candidate_2d[1], 'z': mid_point_3d[2]})
    print(f"\nGenerated {len(oob_points)} OOB points.")
    return oob_points

def build_connectivity_graph_3d(quads, quads_by_id, grid, quad_centers_3d):
    # ... (此函数无需改动) ...
    print("Building 3D road connectivity graph...")
    graph = defaultdict(list)
    lanes = defaultdict(list)
    for q in quads:
        lanes[(q['road_id'], q['lane_id'])].append(q)
    for lane_quads in lanes.values():
        sorted_quads = sorted(lane_quads, key=lambda q: q['q'])
        for i in range(len(sorted_quads) - 1):
            u, v = sorted_quads[i]['polyId'], sorted_quads[i+1]['polyId']
            dist = np.linalg.norm(quad_centers_3d[u] - quad_centers_3d[v])
            graph[u].append((v, dist))
            graph[v].append((u, dist))
    radius = CONFIG["JUNCTION_CONNECT_RADIUS"]
    for quad in quads:
        u = quad['polyId']
        center_u_3d = quad_centers_3d[u]
        u_verts = quads_by_id[u]['vertices']
        u_p_start = (np.array([u_verts[0]['x'], u_verts[0]['y'], u_verts[0]['z']]) + np.array([u_verts[1]['x'], u_verts[1]['y'], u_verts[1]['z']])) / 2
        u_p_end = (np.array([u_verts[2]['x'], u_verts[2]['y'], u_verts[2]['z']]) + np.array([u_verts[3]['x'], u_verts[3]['y'], u_verts[3]['z']])) / 2
        u_fwd = u_p_end - u_p_start
        u_fwd_norm = np.linalg.norm(u_fwd)
        if u_fwd_norm < 1e-6: continue
        u_fwd /= u_fwd_norm
        for v in grid.get_candidates(center_u_3d[:2]):
            if u == v: continue
            u_lane_id = (quads_by_id[u]['road_id'], quads_by_id[u]['lane_id'])
            v_lane_id = (quads_by_id[v]['road_id'], quads_by_id[v]['lane_id'])
            if u_lane_id == v_lane_id: continue
            if np.linalg.norm(center_u_3d - quad_centers_3d[v]) < radius:
                v_verts = quads_by_id[v]['vertices']
                v_p_start = (np.array([v_verts[0]['x'], v_verts[0]['y'], v_verts[0]['z']]) + np.array([v_verts[1]['x'], v_verts[1]['y'], v_verts[1]['z']])) / 2
                v_p_end = (np.array([v_verts[2]['x'], v_verts[2]['y'], v_verts[2]['z']]) + np.array([v_verts[3]['x'], v_verts[3]['y'], v_verts[3]['z']])) / 2
                v_fwd = v_p_end - v_p_start
                v_fwd_norm = np.linalg.norm(v_fwd)
                if v_fwd_norm < 1e-6: continue
                v_fwd /= v_fwd_norm
                if abs(np.dot(u_fwd, v_fwd)) > CONFIG["PARALLEL_LANE_DOT_THRESHOLD"]: continue
                if not any(v == neighbor_id for neighbor_id, _ in graph[u]):
                    dist_3d = np.linalg.norm(center_u_3d - quad_centers_3d[v])
                    graph[u].append((v, dist_3d))
                    graph[v].append((u, dist_3d))
    print(f"Graph built with {len(graph)} nodes.")
    return graph

def find_junction_nodes(graph, quads_by_id):
    """
    这一套寻找的不完全，因此后续只采用筛选出来的final_pois，candidate_pois弃用
    如若需要candidate_pois，可以参考find_cross_intersections函数
    寻找交叉路口节点。
    返回:
    - final_pois (list): 每个交叉路口簇的代表节点ID列表，用于路由。
    - candidate_pois (set): 所有属于交叉路口的路块ID集合，用于查找。
    """
    print("Finding junction nodes (Points of Interest)...")
    candidate_pois = set()
    for node_id in graph:
        connected_lanes = set()
        for neighbor_id, _ in graph.get(node_id, []):
            if neighbor_id in quads_by_id:
                neighbor_lane_id = (quads_by_id[neighbor_id]['road_id'], quads_by_id[neighbor_id]['lane_id'])
                connected_lanes.add(neighbor_lane_id)
        if len(connected_lanes) >= CONFIG["JUNCTION_LANE_THRESHOLD"]:
            candidate_pois.add(node_id)
    print(f"Found {len(candidate_pois)} candidate POIs for clustering.")
    
    # 使用完整的候选者集合可以确保任何在交叉路口内的点都能被识别
    if not candidate_pois:
        return [], set()

    final_pois = []
    visited_candidates = set()
    for candidate_id in sorted(list(candidate_pois)): # sort for deterministic output
        if candidate_id in visited_candidates: continue
        cluster = []
        q = deque([candidate_id])
        visited_in_cluster = {candidate_id}
        while q:
            current_id = q.popleft()
            cluster.append(current_id)
            for neighbor_id, _ in graph.get(current_id, []):
                if neighbor_id in candidate_pois and neighbor_id not in visited_in_cluster:
                    visited_in_cluster.add(neighbor_id)
                    q.append(neighbor_id)
        if cluster:
            cluster.sort()
            final_pois.append(cluster[0])
        visited_candidates.update(visited_in_cluster)
    
    print(f"Found {len(final_pois)} final junction nodes after clustering.")
    return final_pois, candidate_pois # 返回两个结果

def _run_dijkstra_for_node(node_id, graph):
    # ... (此函数无需改动) ...
    return node_id, run_dijkstra(graph, node_id)

def calculate_all_routing_data_parallel(graph, points_of_interest):
    # ... (此函数无需改动) ...
    print(f"Calculating routing data for {len(points_of_interest)} POIs (in parallel)...")
    if not points_of_interest: return {}
    process_func = partial(_run_dijkstra_for_node, graph=graph)
    routing_data = {}
    with Pool(processes=cpu_count()) as pool:
        for i, (poi_id, distances) in enumerate(pool.imap_unordered(process_func, points_of_interest, chunksize=1)):
            routing_data[poi_id] = distances
            sys.stdout.write(f"\rRouting processing: {i+1}/{len(points_of_interest)}")
            sys.stdout.flush()
    print(f"\nFinished calculating routing data.")
    return routing_data

def enhance_w_lanes_from_carla(driving_waypoints, routing_data, points_of_interest, grid, quads_by_id):
    """
    使用来自CARLA的现有航点，并为其添加方向和路由信息, 替代原有的生成逻辑。
    """
    print("\nEnhancing W_lane waypoints from CARLA data...")
    enhanced_waypoints = []
    off_road_count = 0

    total_wps = len(driving_waypoints)
    if total_wps == 0:
        print("Warning: No 'driving_waypoints' found in input data.")
        return []

    for i, wp in enumerate(driving_waypoints):
        if (i + 1) % 1000 == 0 or i + 1 == total_wps:
            sys.stdout.write(f"\rProcessing CARLA waypoint: {i+1}/{total_wps}")
            sys.stdout.flush()

        loc = wp['transform']['location']
        rot = wp['transform']['rotation']
        pos = np.array([loc['x'], loc['y'], loc['z']])
        pos_2d = pos[:2]
        
        # 找到航点所在的 quad
        containing_poly_id = None
        for cand_poly_id in grid.get_candidates(pos_2d):
            if is_point_in_quad_2d(pos_2d, [[v['x'], v['y']] for v in quads_by_id[cand_poly_id]['vertices']]):
                containing_poly_id = cand_poly_id
                break
        
        if containing_poly_id is None:
            off_road_count += 1
            continue
            
        # 从航点的 yaw 计算方向向量
        yaw_rad = np.deg2rad(rot['yaw'])
        direction_vec = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        
        point_data = {
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'direction': direction_vec.tolist(),
            'routing': {},
            # --- 新增: 保留poly_id用于后续处理 ---
            'poly_id': containing_poly_id,
            # 保留原始CARLA信息以便调试或未来使用
            'carla_waypoint_info': {
                'id': wp['id'],
                'road_id': wp['road_id'],
                'lane_id': wp['lane_id'],
                's': wp['s'],
                'is_junction': wp['is_junction']
            }
        }
        
        # 添加路由数据
        for poi_id in points_of_interest:
            dist_to_poi = routing_data.get(poi_id, {}).get(containing_poly_id, float('inf'))
            point_data['routing'][f'dist_to_{poi_id}'] = dist_to_poi
            
        enhanced_waypoints.append(point_data)

    print(f"\nFiltered out {off_road_count} off-road CARLA waypoints.")
    print(f"Initially processed {len(enhanced_waypoints)} waypoints from CARLA data.")
    return enhanced_waypoints

# --- 新增函数: 为W_lane航点寻找下一个POI ---
# 这个函数是用来为每个w_lane航点计算到其路径上遇到的下一个POI (前方) 和上一个 POI (后方) 的距离。
# 此函数内部会处理排序，确保逻辑的鲁棒性。
# 20250618 修改了逻辑，现在可以正确计算到下一个POI和上一个POI的距离
# 当进入POI后的点均设置为inf和none  
def find_closest_poi_downstream(start_poly_id, graph, poi_set, quad_centers_3d, quad_directions, direction_threshold=0.0, reverse_search=False):
    """
    从一个起始路块开始，沿着前进方向在图中进行Dijkstra搜索，找到最近的POI。
    支持POI集合和路口区域集合两种输入。
    
    Args:
        start_poly_id: 起始路块ID
        graph: 图的邻接表
        poi_set: POI集合
        quad_centers_3d: 路块中心点字典
        quad_directions: 路块方向向量字典
        direction_threshold: 方向阈值
        reverse_search: 是否进行逆向搜索
    """
    if start_poly_id in poi_set:
        return start_poly_id, 0.0
        
    # 优先级队列: (距离, 当前节点ID, 路径列表)
    pq = [(0, start_poly_id, [])] 
    visited = set()

    while pq:
        dist, current_id, path = heapq.heappop(pq)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in poi_set:
            return current_id, dist
        
        # --- 方向逻辑 ---
        if not path:
            # 对于第一个节点，使用其预定义的静态方向
            direction_vec_3d = quad_directions.get(current_id)
            if reverse_search and direction_vec_3d is not None:
                # 逆向搜索时取反方向向量
                direction_vec_3d = -direction_vec_3d
        else:
            # 对于后续节点，根据搜索模式选择方向
            if reverse_search:
                # 逆向搜索：使用传入的逆方向向量
                direction_vec_3d = quad_directions.get(current_id)
                if direction_vec_3d is not None:
                    # 逆向搜索时取反方向向量
                    direction_vec_3d = -direction_vec_3d
            else:
                # 正向搜索：使用路径中的上一步来动态计算行进方向
                prev_id = path[-1]
                direction_vec_3d = quad_centers_3d[current_id] - quad_centers_3d[prev_id]
        
        if direction_vec_3d is None: continue
        direction_vec_2d = direction_vec_3d[:2]
        dir_norm = np.linalg.norm(direction_vec_2d)
        if dir_norm < 1e-6: continue
        norm_travel_dir = direction_vec_2d / dir_norm
        
        for neighbor_id, weight in graph.get(current_id, []):
            if neighbor_id in visited:
                continue
            # 基于行进方向进行方向检查
            vec_to_neighbor = quad_centers_3d[neighbor_id][:2] - quad_centers_3d[current_id][:2]
            if np.linalg.norm(vec_to_neighbor) < 1e-6: continue
            norm_vec_to_neighbor = vec_to_neighbor / np.linalg.norm(vec_to_neighbor)
            dot_product = np.dot(norm_travel_dir, norm_vec_to_neighbor)
            if dot_product > direction_threshold:
                # 将当前节点添加到路径中，为下一次迭代做准备
                new_path = path + [current_id]
                heapq.heappush(pq, (dist + weight, neighbor_id, new_path))
    return None, float('inf')

def _detect_cross_intersections(quads_by_id, waypoints):
    """
    检测路口区域，返回路口quad集合
    
    Args:
        quads_by_id: 路块字典
        waypoints: 航点列表
        
    Returns:
        set: 路口区域quad ID集合
    """
    print("=== 检测路口区域 ===")
    try:
        from carla_find_cross import find_cross
        cross_quad_ids = find_cross(
            quads=list(quads_by_id.values()),
            waypoints=waypoints,
            eps=2.0,
            distance_threshold=4,
            min_poly_count=4,
            visualize=False
        )
        cross_quad_set = set(cross_quad_ids)
        print(f"找到 {len(cross_quad_ids)} 个路口区域quad")
        return cross_quad_set
    except Exception as e:
        print(f"检测路口区域时出错: {e}")
        return set()

def _build_directional_graphs(quads_by_id, quad_directions):
    """
    构建正向和反向图
    
    Args:
        quads_by_id: 路块字典
        quad_directions: 路块方向向量字典
        
    Returns:
        tuple: (正向图, 反向图) 或 (None, None) 如果构建失败
    """
    print("=== 构建图 ===")
    
    # 参数验证
    if not quads_by_id:
        print("Error: Empty quads_by_id dictionary")
        return None, None
    
    if not quad_directions:
        print("Error: Empty quad_directions dictionary")
        return None, None
    
    try:
        from dijkstra import build_graph
        
        # 构建正向图
        print("构建正向图...")
        adjacency_list, _, _, _, _ = build_graph(
            list(quads_by_id.values()), 
            neighbor_radius=3, 
            direction_threshold=0.5
        )
        print(f"正向图构建成功，包含 {len(adjacency_list)} 个节点")
        
        # 构建反向图：使用反向的方向向量
        print("构建反向图...")
        reversed_directions = {k: -v for k, v in quad_directions.items()}
        adjacency_list_reversed, _, _, _, _ = build_graph(
            list(quads_by_id.values()), 
            neighbor_radius=3, 
            direction_threshold=0.5, 
            quad_directions=reversed_directions
        )
        print(f"反向图构建成功，包含 {len(adjacency_list_reversed)} 个节点")
        
        # 验证图的有效性
        if len(adjacency_list) == 0 or len(adjacency_list_reversed) == 0:
            print("Warning: Empty graph built")
            return None, None
        
        return adjacency_list, adjacency_list_reversed
        
    except ImportError as e:
        print(f"Error: Failed to import dijkstra module: {e}")
        return None, None
    except Exception as e:
        print(f"构建图时出错: {e}")
        return None, None

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

def _initialize_poi_fields(waypoints_in_lane):
    """
    初始化航点的POI相关字段
    
    Args:
        waypoints_in_lane: 单个车道内的航点列表
    """
    for wp in waypoints_in_lane:
        wp.update({
            'next_poi_id': None, 
            'dist_to_next_poi': float('inf'),
            'prev_poi_id': None, 
            'dist_to_prev_poi': float('inf')
        })

def _set_poi_internal_points(waypoints_in_lane, cross_quad_set):
    """
    设置位于POI内部的航点
    
    Args:
        waypoints_in_lane: 单个车道内的航点列表
        cross_quad_set: 路口区域quad集合
    """
    for wp in waypoints_in_lane:
        if wp['poly_id'] in cross_quad_set:
            wp['next_poi_id'] = wp['poly_id']
            wp['dist_to_next_poi'] = 0.0
            wp['prev_poi_id'] = wp['poly_id']
            wp['dist_to_prev_poi'] = 0.0

def _calculate_poi_distances_with_caching(waypoints_in_lane, cross_quad_set, adjacency_list, quad_centers_3d, quad_directions, is_forward=True):
    """
    使用缓存机制计算POI距离，避免重复计算
    
    Args:
        waypoints_in_lane: 单个车道内的航点列表
        cross_quad_set: 路口区域quad集合
        adjacency_list: 图（正向或反向）
        quad_centers_3d: 路块中心点字典
        quad_directions: 路块方向向量字典
        is_forward: 是否为正向搜索
    """
    from dijkstra import find_shortest_path
    
    # 缓存已计算的路径距离
    path_cache = {}
    
    # 确定字段名称
    poi_id_field = 'next_poi_id' if is_forward else 'prev_poi_id'
    dist_field = 'dist_to_next_poi' if is_forward else 'dist_to_prev_poi'
    
    # 批量处理非POI点
    non_poi_waypoints = [wp for wp in waypoints_in_lane if wp['poly_id'] not in cross_quad_set]
    
    for wp in non_poi_waypoints:
        start_poly_id = wp['poly_id']
        
        # 检查缓存
        if start_poly_id in path_cache:
            target_poi_id, distance = path_cache[start_poly_id]
        else:
            # 找到目标POI
            target_poi_id, _ = find_closest_poi_downstream(
                start_poly_id, adjacency_list, cross_quad_set, quad_centers_3d, quad_directions
            )
            
            if target_poi_id is not None:
                # 计算实际路径距离
                _, distance = find_shortest_path(adjacency_list, start_poly_id, target_poi_id)
                # 缓存结果
                path_cache[start_poly_id] = (target_poi_id, distance)
            else:
                target_poi_id = None
                distance = float('inf')
                path_cache[start_poly_id] = (target_poi_id, distance)
        
        # 设置结果
        wp[poi_id_field] = target_poi_id
        wp[dist_field] = distance

def _process_single_lane_poi_distances(waypoints_in_lane, cross_quad_set, adjacency_list, adjacency_list_reversed, 
                                     quad_centers_3d, quad_directions):
    """
    处理单个车道的POI距离计算（优化版本）
    
    Args:
        waypoints_in_lane: 单个车道内的航点列表
        cross_quad_set: 路口区域quad集合
        adjacency_list: 正向图
        adjacency_list_reversed: 反向图
        quad_centers_3d: 路块中心点字典
        quad_directions: 路块方向向量字典
    """
    if not waypoints_in_lane:
        return
    
    # 初始化字段
    _initialize_poi_fields(waypoints_in_lane)
    
    # 设置POI内部的点
    _set_poi_internal_points(waypoints_in_lane, cross_quad_set)
    
    # 使用缓存机制计算下一个POI距离
    _calculate_poi_distances_with_caching(
        waypoints_in_lane, cross_quad_set, adjacency_list, quad_centers_3d, quad_directions, is_forward=True
    )
    
    # 使用缓存机制计算上一个POI距离
    reversed_directions = {k: -v for k, v in quad_directions.items()}
    _calculate_poi_distances_with_caching(
        waypoints_in_lane, cross_quad_set, adjacency_list_reversed, quad_centers_3d, reversed_directions, is_forward=False
    )

def _print_poi_statistics(waypoints):
    """
    打印POI距离计算的统计信息
    
    Args:
        waypoints: 航点列表
    """
    total_waypoints = len(waypoints)
    successful_next_poi = sum(1 for wp in waypoints if wp['next_poi_id'] is not None)
    successful_prev_poi = sum(1 for wp in waypoints if wp['prev_poi_id'] is not None)
    
    print(f"\n=== POI Distance Calculation Statistics ===")
    print(f"Total waypoints: {total_waypoints}")
    print(f"Successful next POI: {successful_next_poi}/{total_waypoints} ({successful_next_poi/total_waypoints*100:.1f}%)")
    print(f"Successful previous POI: {successful_prev_poi}/{total_waypoints} ({successful_prev_poi/total_waypoints*100:.1f}%)")
    
    if successful_next_poi > 0:
        distances_next = [wp['dist_to_next_poi'] for wp in waypoints 
                         if wp['next_poi_id'] is not None and wp['dist_to_next_poi'] != float('inf')]
        if distances_next:
            print(f"Next POI Distance Statistics:")
            print(f"   Average distance: {np.mean(distances_next):.2f} meters")
            print(f"   Minimum distance: {np.min(distances_next):.2f} meters")
            print(f"   Maximum distance: {np.max(distances_next):.2f} meters")
            print(f"   Median distance: {np.median(distances_next):.2f} meters")
    
    if successful_prev_poi > 0:
        distances_prev = [wp['dist_to_prev_poi'] for wp in waypoints 
                         if wp['prev_poi_id'] is not None and wp['dist_to_prev_poi'] != float('inf')]
        if distances_prev:
            print(f"Previous POI Distance Statistics:")
            print(f"   Average distance: {np.mean(distances_prev):.2f} meters")
            print(f"   Minimum distance: {np.min(distances_prev):.2f} meters")
            print(f"   Maximum distance: {np.max(distances_prev):.2f} meters")
            print(f"   Median distance: {np.median(distances_prev):.2f} meters")

            print(f"上一个POI距离统计:")
            print(f"  平均距离: {np.mean(distances_prev):.2f} 米")
            print(f"  最小距离: {np.min(distances_prev):.2f} 米")
            print(f"  最大距离: {np.max(distances_prev):.2f} 米")
            print(f"  中位数: {np.median(distances_prev):.2f} 米")

def find_next_and_prev_poi_for_wlanes(waypoints, points_of_interest, graph, quads_by_id, quad_centers_3d, quad_directions):
    """
    为每个 w_lane 航点计算到其路径上遇到的下一个 POI (前方) 和上一个 POI (后方) 的距离。
    
    Args:
        waypoints: 航点列表
        points_of_interest: POI列表（备用）
        graph: 连通图
        quads_by_id: 路块字典
        quad_centers_3d: 路块中心点字典
        quad_directions: 路块方向向量字典
    """
    print("Calculating distance to next and previous POI for all W_lane waypoints...")
    
    if not waypoints:
        print("Warning: No waypoints to process.")
        return
    
    # 1. 检测路口区域
    cross_quad_set = _detect_cross_intersections(quads_by_id, waypoints)
    if not cross_quad_set:
        # 如果检测失败，使用传入的POI
        cross_quad_set = set(points_of_interest)
        print(f"使用传入的POI集合，包含 {len(cross_quad_set)} 个POI")
    
    # 2. 构建方向图
    adjacency_list, adjacency_list_reversed = _build_directional_graphs(quads_by_id, quad_directions)
    if adjacency_list is None or adjacency_list_reversed is None:
        print("Error: Failed to build directional graphs")
        return
    
    # 3. 按车道分组航点
    lanes = _group_waypoints_by_lane(waypoints)
    
    # 4. 为每个车道计算POI距离
    print("=== 计算waypoints的POI距离 ===")
    for lane_id_tuple, wps_in_lane in lanes.items():
        _process_single_lane_poi_distances(
            wps_in_lane, cross_quad_set, adjacency_list, adjacency_list_reversed,
            quad_centers_3d, quad_directions
        )
    
    # 5. 打印统计信息
    # _print_poi_statistics(waypoints)
    
    print("Finished calculating distances to next and previous POI.")

# --- 新增函数: W_lane 降采样 ---
def downsample_w_lane_waypoints(waypoints, long_thresh):
    """
    根据用户提供的示意图进行降采样:
    - 删除所有在交叉路口内部的航点 (`is_junction == True`).
    - 在非交叉路口的路段上，大约每隔 `long_thresh` 米采样一个点.
    - 强制保留每个非交叉路口路段的第一个和最后一个航点.
    """
    print(f"Downsampling W_lane waypoints based on junction-aware logic (threshold: {long_thresh}m)...")
    if not waypoints:
        return []

    # --- 1. 按车道ID对所有航点进行分组和排序 ---
    lanes = defaultdict(list)
    # 给所有航点一个原始索引，以便之后恢复顺序
    for i, wp in enumerate(waypoints):
        wp['original_index'] = i
        info = wp['carla_waypoint_info']
        lanes[(info['road_id'], info['lane_id'])].append(wp)

    sorted_lanes = {}
    for lane_id_tuple, wps_in_lane in lanes.items():
        if not wps_in_lane:
            continue
        is_reverse_lane = wps_in_lane[0]['carla_waypoint_info']['lane_id'] < 0
        sorted_lanes[lane_id_tuple] = sorted(wps_in_lane, key=lambda w: w['carla_waypoint_info']['s'], reverse=is_reverse_lane)

    # --- 2. 执行降采样 ---
    final_waypoints_indices = set()
    for lane_id_tuple, wps_in_lane in sorted_lanes.items():
        # --- 核心逻辑: 将车道分割成非交叉路口路段 ---
        non_junction_segments = []
        current_segment = []
        for wp in wps_in_lane:
            if not wp['carla_waypoint_info']['is_junction']:
                current_segment.append(wp)
            else:
                if current_segment:
                    non_junction_segments.append(current_segment)
                    current_segment = []
        if current_segment: # Add the last segment if it exists
            non_junction_segments.append(current_segment)
        
        # --- 对每个非交叉路口路段进行采样 ---
        for segment in non_junction_segments:
            if not segment:
                continue

            # 保留路段的第一个点
            last_kept_wp = segment[0]
            final_waypoints_indices.add(last_kept_wp['original_index'])
            last_kept_pos = np.array([last_kept_wp['x'], last_kept_wp['y'], last_kept_wp['z']])

            # 沿路段采样
            for i in range(1, len(segment)):
                current_wp = segment[i]
                current_pos = np.array([current_wp['x'], current_wp['y'], current_wp['z']])
                distance_from_last = np.linalg.norm(current_pos - last_kept_pos)
                if distance_from_last >= long_thresh:
                    final_waypoints_indices.add(current_wp['original_index'])
                    last_kept_wp = current_wp
                    last_kept_pos = current_pos
            
            # 强制保留路段的最后一个点 (如果它还没被加入)
            if len(segment) > 1:
                final_waypoints_indices.add(segment[-1]['original_index'])

    # --- 3. 构建最终列表并重新分配ID ---
    simplified_waypoints = [wp for wp in waypoints if wp['original_index'] in final_waypoints_indices]
    simplified_waypoints.sort(key=lambda wp: wp['original_index']) # 保持原始顺序

    for i, wp in enumerate(simplified_waypoints):
        wp['waypoint_id'] = i
        if 'original_index' in wp:
            del wp['original_index']

    print(f"Downsampling complete. W_lane waypoints reduced from {len(waypoints)} to {len(simplified_waypoints)}.")
    return simplified_waypoints

def _associate_quad_to_points_task(poly_id, quad_centers_3d, point_grid, radius_key):
    # ... (此函数无需改动) ...
    center_3d = quad_centers_3d[poly_id]
    nearby_ids = point_grid.get_nearby_ids(center_3d[:2], CONFIG[radius_key])
    return poly_id, nearby_ids

def associate_quads_to_points_parallel(quads, quad_centers_3d, global_points, grid_name, radius_key):
    # ... (此函数无需改动) ...
    print(f"Associating quads with nearby global {grid_name} (in parallel)...")
    if not global_points:
        print(f"No global {grid_name} to associate. Returning empty map.")
        return {q['polyId']: [] for q in quads}
    point_grid = PointIndexGrid(global_points, CONFIG["CELL_SIZE"] * 2, name=grid_name)
    process_func = partial(_associate_quad_to_points_task, quad_centers_3d=quad_centers_3d, point_grid=point_grid, radius_key=radius_key)
    all_poly_ids = [q['polyId'] for q in quads]
    results = {}
    with Pool(processes=cpu_count()) as pool:
        for i, (poly_id, point_ids) in enumerate(pool.imap_unordered(process_func, all_poly_ids, chunksize=100)):
            results[poly_id] = point_ids
            sys.stdout.write(f"\r{grid_name.capitalize()} association processing: {i+1}/{len(all_poly_ids)}")
            sys.stdout.flush()
    print(f"\nFinished associating {grid_name} for {len(results)} quads.")
    return results

def associate_quads_to_adjacent_waypoints(quads, waypoints):
    """
    为每个路块计算其路径上前、后两个方向的第一个w_lane航点的ID。
    这创建了从 `polyId` 到 `waypoint_id` 的两个直接映射。
    - quad_to_next_waypoint: 映射到前方的航点
    - quad_to_prev_waypoint: 映射到后方的航点
    """
    print("Associating each road quad with its next and previous W_lane waypoints...")
    if not waypoints:
        return {}, {}

    lanes_quads = defaultdict(list)
    for q in quads:
        lanes_quads[(q['road_id'], q['lane_id'])].append(q)

    lanes_waypoints = defaultdict(list)
    for wp in waypoints:
        lanes_waypoints[(wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'])].append(wp)

    quad_to_next_wp_map = {}
    quad_to_prev_wp_map = {}
    
    for lane_id, quads_in_lane in lanes_quads.items():
        is_reverse = lane_id[1] < 0
        quads_in_lane.sort(key=lambda q: q['q'], reverse=is_reverse)
        if lane_id in lanes_waypoints:
            lanes_waypoints[lane_id].sort(key=lambda w: w['carla_waypoint_info']['s'], reverse=is_reverse)

    for lane_id, quads_in_lane in lanes_quads.items():
        waypoints_in_lane = lanes_waypoints.get(lane_id)
        if not waypoints_in_lane:
            continue

        wp_ptr = 0
        for quad in quads_in_lane:
            is_reverse = lane_id[1] < 0
            
            # 使用双指针，为当前 quad 找到最合适的航点索引
            # 将指针移动到第一个s值大于或等于quad的q值的航点
            while wp_ptr < len(waypoints_in_lane) and \
                  ((not is_reverse and waypoints_in_lane[wp_ptr]['carla_waypoint_info']['s'] < quad['q']) or \
                   (is_reverse and waypoints_in_lane[wp_ptr]['carla_waypoint_info']['s'] > quad['q'])):
                wp_ptr += 1

            # 根据指针位置确定前后航点
            if wp_ptr == 0:
                # Quad 在第一个航点之前
                next_wp_idx = 0
                prev_wp_idx = 0
            elif wp_ptr == len(waypoints_in_lane):
                # Quad 在最后一个航点之后
                next_wp_idx = len(waypoints_in_lane) - 1
                prev_wp_idx = len(waypoints_in_lane) - 1
            else:
                # Quad 在两个航点之间
                next_wp_idx = wp_ptr
                prev_wp_idx = wp_ptr - 1
            
            quad_to_next_wp_map[quad['polyId']] = waypoints_in_lane[next_wp_idx]['waypoint_id']
            quad_to_prev_wp_map[quad['polyId']] = waypoints_in_lane[prev_wp_idx]['waypoint_id']
            
    print(f"Finished association for {len(quad_to_next_wp_map)} quads.")
    return quad_to_next_wp_map, quad_to_prev_wp_map

# --- 主流程 ---
def preprocess_map(stage1_data_path, output_path):
    start_time = time.time()
    print("--- Starting Map Preprocessing (Stage 2) ---")
    
    print(f"Loading data from: {stage1_data_path}...")
    with open(stage1_data_path, 'r') as f:
        stage1_data = json.load(f)
        raw_quads_data = stage1_data.get('quads', [])
        traffic_data = stage1_data.get('traffic_controls', [])
        # --- 新增: 加载CARLA航点 ---
        driving_waypoints = stage1_data.get('driving_waypoints', [])
        if not raw_quads_data:
            print("Error: 'quads' key not found or empty in the input file.")
            sys.exit(1)

    quads_by_id = {q['polyId']: q for q in raw_quads_data}
    grid = SpatialGrid3D(raw_quads_data, CONFIG["CELL_SIZE"])
    quad_centers_3d = {q['polyId']: get_quad_center_3d(q['vertices']) for q in raw_quads_data}
    
    # --- 新增: 计算所有quad的方向向量 ---
    quad_directions = {}
    for q in raw_quads_data:
        verts = q['vertices']
        p_start = (np.array([verts[0]['x'], verts[0]['y'], verts[0]['z']]) + np.array([verts[1]['x'], verts[1]['y'], verts[1]['z']])) / 2
        p_end = (np.array([verts[2]['x'], verts[2]['y'], verts[2]['z']]) + np.array([verts[3]['x'], verts[3]['y'], verts[3]['z']])) / 2
        direction = p_start - p_end # 原始定义是从后到前
        quad_directions[q['polyId']] = direction

    oob_points = generate_oob_points(raw_quads_data, grid, quads_by_id)
    graph = build_connectivity_graph_3d(raw_quads_data, quads_by_id, grid, quad_centers_3d)
    
    # --- 更新: 获取简化POI列表和完整POI集合 ---
    points_of_interest, all_junction_quads = find_junction_nodes(graph, quads_by_id)
    #以上用于路由使用。
    
    routing_data = calculate_all_routing_data_parallel(graph, points_of_interest)
    
    # --- 核心修改: 使用CARLA航点进行增强，而非从头生成 ---
    enhanced_waypoints = enhance_w_lanes_from_carla(driving_waypoints, routing_data, points_of_interest, grid, quads_by_id)
    
    # --- 新增: 使用find_cross获取路口区域 ---
    from carla_find_cross import find_cross
    cross_quad_ids = find_cross(
        quads=raw_quads_data,
        waypoints=enhanced_waypoints,
        eps=2.0,
        distance_threshold=4,
        min_poly_count=4,
        visualize=False
    )
    cross_quad_set = set(cross_quad_ids)
    print(f"Found {len(cross_quad_set)} cross quads.")
    
    # --- 更新: 使用find_cross返回的路口区域进行计算 ---
    find_next_and_prev_poi_for_wlanes(enhanced_waypoints, cross_quad_set, graph, quads_by_id, quad_centers_3d, quad_directions)
    
    global_waypoints = downsample_w_lane_waypoints(
        enhanced_waypoints,
        CONFIG["W_LANE_DOWNSAMPLE_LONGITUDINAL_THRESHOLD"]
    )

    # --- 更新: 调用新函数，同时获取前后两个方向的映射 ---
    quad_to_next_wp_map, quad_to_prev_wp_map = associate_quads_to_adjacent_waypoints(raw_quads_data, global_waypoints)

    boundary_association_map = associate_quads_to_points_parallel(
        quads=raw_quads_data,
        quad_centers_3d=quad_centers_3d,
        global_points=oob_points,
        grid_name="oob_points",
        radius_key="W_BOUNDARY_ASSOCIATION_RADIUS"
    )
    wlane_association_map = associate_quads_to_points_parallel(
        quads=raw_quads_data,
        quad_centers_3d=quad_centers_3d,
        global_points=global_waypoints, # 使用降采样后的 w_lane 点
        grid_name="w_lane_waypoints",
        radius_key="W_LANE_ASSOCIATION_RADIUS"
    )
    
    print("Integrating all data into final format...")
    for quad in raw_quads_data:
        poly_id = quad['polyId']
        quad['w_boundary_ids'] = boundary_association_map.get(poly_id, [])
        quad['w_lane_ids'] = wlane_association_map.get(poly_id, [])

    final_processed_data = {
        'map_name': stage1_data.get('map_name', 'unknown'),
        'quads': raw_quads_data,
        'traffic_controls': traffic_data,
        'oob_points': oob_points,
        'global_w_lane_waypoints': global_waypoints, # 保存降采样后的版本
        'routing_data': routing_data,
        'points_of_interest': points_of_interest,
        'quad_to_next_waypoint': quad_to_next_wp_map, # 新增的查找表
        'quad_to_prev_waypoint': quad_to_prev_wp_map, # 新增的后向查找表
    }
    
    print(f"Saving final processed map data to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(final_processed_data, f, cls=NumpyEncoder)
    
    end_time = time.time()
    print(f"--- Preprocessing complete in {end_time - start_time:.2f} seconds. ---")
    print(f"Final file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == '__main__':
    # 使用默认配置文件中的地图路径（基于文件位置解析项目根目录）
    import yaml
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(_this_dir)
    _cfg_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
    with open(_cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    map_path = config['simulator']['map_path'] #导出路径
    stage1_path = os.path.join(_proj_root, "maps", "carla_map_data_Town01_stitched.json")
    output_path = map_path                     #导出路径
    print(f"Using default map path: {map_path}")
    print(f"Output path: {output_path}")

    if not os.path.exists(stage1_path):
        print(f"Error: Input file not found at {stage1_path}"); sys.exit(1)

    preprocess_map(stage1_path, output_path)

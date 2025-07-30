import json
import numpy as np
import os
from collections import defaultdict
import sys
from multiprocessing import Pool, cpu_count
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
    "W_LANE_ASSOCIATION_RADIUS": 200.0, # 200m，用于将w_lane航点与相邻航点关联
    "W_BOUNDARY_ASSOCIATION_RADIUS": 15.0,  # 15m，用于将w_lane航点与边界关联
    "W_LANE_DOWNSAMPLE_LONGITUDINAL_THRESHOLD": 40, # 沿道路方向距离
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

# --- 数据生成函数 ---
def generate_oob_points(quads, grid, quads_by_id):
    """优化的OOB点生成函数"""
    print("Generating Out-of-Bounds (OOB) points...")
    oob_points = []
    nudge = CONFIG["OOB_NUDGE_DISTANCE"]
    
    # 预计算所有quad的顶点数组，避免重复转换
    quad_vertices_cache = {}
    for quad in quads:
        poly_id = quad['polyId']
        quad_vertices_cache[poly_id] = np.array([[p['x'], p['y'], p['z']] for p in quad['vertices']])
    
    # 预计算所有quad的2D边界框，用于快速排除
    quad_bounds_cache = {}
    for poly_id, vertices in quad_vertices_cache.items():
        min_coords = np.min(vertices[:, :2], axis=0)
        max_coords = np.max(vertices[:, :2], axis=0)
        quad_bounds_cache[poly_id] = (min_coords, max_coords)
    
    # 预计算所有quad的2D顶点，避免重复转换
    quad_vertices_2d_cache = {}
    for poly_id, vertices in quad_vertices_cache.items():
        quad_vertices_2d_cache[poly_id] = [[v['x'], v['y']] for v in quads_by_id[poly_id]['vertices']]
    
    for i, quad in enumerate(quads):
        if (i + 1) % 100 == 0 or i + 1 == len(quads):
            sys.stdout.write(f"\rOOB processing: {i+1}/{len(quads)}")
            sys.stdout.flush()
        
        poly_id = quad['polyId']
        verts_3d = quad_vertices_cache[poly_id]
        quad_vertices_2d = quad_vertices_2d_cache[poly_id]
        
        for i_edge in range(4):
            p1_3d, p2_3d = verts_3d[i_edge], verts_3d[(i_edge + 1) % 4]
            mid_point_3d = (p1_3d + p2_3d) / 2.0
            mid_point_2d = mid_point_3d[:2]
            edge_vec_2d = p2_3d[:2] - p1_3d[:2]
            normal_2d = np.array([edge_vec_2d[1], -edge_vec_2d[0]])
            norm = np.linalg.norm(normal_2d)
            if norm < 1e-6: continue
            normal_2d /= norm
            
            # 优化：使用更小的测试距离
            test_point_2d = mid_point_2d + normal_2d * 0.005
            if is_point_in_quad_2d(test_point_2d, quad_vertices_2d):
                normal_2d = -normal_2d
            
            oob_candidate_2d = mid_point_2d + normal_2d * nudge
            
            # 优化：使用边界框快速排除
            is_inside_any = False
            candidate_poly_ids = grid.get_candidates(oob_candidate_2d)
            
            # 快速边界框检查
            for cand_poly_id in candidate_poly_ids:
                if cand_poly_id == poly_id:  # 跳过自己
                    continue
                    
                min_coords, max_coords = quad_bounds_cache[cand_poly_id]
                if (oob_candidate_2d[0] < min_coords[0] or oob_candidate_2d[0] > max_coords[0] or
                    oob_candidate_2d[1] < min_coords[1] or oob_candidate_2d[1] > max_coords[1]):
                    continue  # 点在边界框外，跳过精确检查
                
                # 只有通过边界框检查的才进行精确的quad包含检查
                cand_vertices_2d = quad_vertices_2d_cache[cand_poly_id]
                if is_point_in_quad_2d(oob_candidate_2d, cand_vertices_2d):
                    is_inside_any = True
                    break
            
            if not is_inside_any:
                oob_points.append({
                    'x': oob_candidate_2d[0], 
                    'y': oob_candidate_2d[1], 
                    'z': mid_point_3d[2]
                })
    print(f"\nGenerated {len(oob_points)} OOB points.")
    return oob_points

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
        lanes[(wp['road_id'], wp['lane_id'])].append(wp)
    # 对每个车道内的航点进行排序
    for lane_id_tuple, wps_in_lane in lanes.items():
        if wps_in_lane:
            is_reverse_lane = wps_in_lane[0]['lane_id'] < 0
            wps_in_lane.sort(key=lambda w: w['s'], reverse=is_reverse_lane)
    return lanes

def enhance_w_lanes_from_carla(driving_waypoints, grid, quads_by_id):
    """
    使用来自CARLA的现有航点，并为其添加方向和路由信息, 替代原有的生成逻辑。
    模仿filtered_quad_indices逻辑，将不在filtered_quad_indices内的carla_waypoint标记为is_junction。
    """
    print("\nEnhancing W_lane waypoints from CARLA data...")
    enhanced_waypoints = []
    off_road_count = 0

    total_wps = len(driving_waypoints)
    if total_wps == 0:
        print("Warning: No 'driving_waypoints' found in input data.")
        return []

    # 构建filtered_quad_indices逻辑
    # 按车道分组航点
    lanes = _group_waypoints_by_lane(driving_waypoints)
    
    # 过滤掉首尾waypoint距离太短的车道
    print("正在过滤太短的车道...")
    min_lane_length = 10  # 最小车道长度阈值（米）
    lanes_to_remove = []
    for lane_id, wps_in_lane in lanes.items():
        if len(wps_in_lane) >= 2:
            # 计算首尾waypoint之间的距离
            start_wp = wps_in_lane[0]
            end_wp = wps_in_lane[-1]
            # 从transform.location中获取坐标
            start_pos = start_wp['transform']['location']
            end_pos = end_wp['transform']['location']
            distance = np.sqrt((end_pos['x'] - start_pos['x'])**2 + (end_pos['y'] - start_pos['y'])**2)
            
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
    for idx, quad in enumerate(quads_by_id.values()):
        rid = quad.get('road_id', None)
        if rid is not None:
            if rid not in road_id_to_indices:
                road_id_to_indices[rid] = []
            road_id_to_indices[rid].append(idx)
    
    for wps_in_lane in lanes.values():
        for wp in wps_in_lane:
            wp_road_id = wp['road_id']
            if wp_road_id in road_id_to_indices:
                filtered_quad_indices.update(road_id_to_indices[wp_road_id])

    print(f"Found {len(filtered_quad_indices)} filtered quad indices from {len(lanes)} remaining lanes")

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
        
        # 如果航点不在任何quad内，标记为off-road并跳过
        if containing_poly_id is None:
            off_road_count += 1
            continue
            
        # 从航点的 yaw 计算方向向量
        yaw_rad = np.deg2rad(rot['yaw'])
        direction_vec = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        
        # 判断航点是否在filtered_quad_indices内
        # 找到containing_poly_id在quads_by_id中的索引
        quad_index = None
        for idx, quad in enumerate(quads_by_id.values()):
            if quad['polyId'] == containing_poly_id:
                quad_index = idx
                break
        
        # 判断是否为junction（不在filtered_quad_indices内）
        is_junction = False
        if quad_index is not None and quad_index not in filtered_quad_indices:
            is_junction = True
        
        point_data = {
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'direction': direction_vec.tolist(),
            # --- 新增: 保留poly_id用于后续处理 ---
            'poly_id': containing_poly_id,
            # 保留原始CARLA信息以便调试或未来使用
            'carla_waypoint_info': {
                'id': wp['id'],
                'road_id': wp['road_id'],
                'lane_id': wp['lane_id'],
                's': wp['s'],
                'is_junction': is_junction  # 使用新的junction判断逻辑
            }
        }
        
        enhanced_waypoints.append(point_data)

    print(f"\nFiltered out {off_road_count} off-road CARLA waypoints.")
    print(f"Initially processed {len(enhanced_waypoints)} waypoints from CARLA data.")
    
    # 统计junction航点数量
    junction_count = sum(1 for wp in enhanced_waypoints if wp['carla_waypoint_info']['is_junction'])
    print(f"Found {junction_count} junction waypoints out of {len(enhanced_waypoints)} total waypoints")
    
    return enhanced_waypoints

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

def _associate_quad_to_points_task(args):
    """多进程任务函数 - 将参数打包为单个元组"""
    poly_id, quad_centers_3d, point_grid, radius_key = args
    center_3d = quad_centers_3d[poly_id]
    nearby_ids = point_grid.get_nearby_ids(center_3d[:2], CONFIG[radius_key])
    return poly_id, nearby_ids

def associate_quads_to_points_parallel(quads, quad_centers_3d, global_points, grid_name, radius_key):
    """并行关联四元组与点"""
    print(f"Associating quads with nearby global {grid_name} (in parallel)...")
    if not global_points:
        print(f"No global {grid_name} to associate. Returning empty map.")
        return {q['polyId']: [] for q in quads}
    
    point_grid = PointIndexGrid(global_points, CONFIG["CELL_SIZE"] * 2, name=grid_name)
    all_poly_ids = [q['polyId'] for q in quads]
    # 准备参数列表
    args_list = [(poly_id, quad_centers_3d, point_grid, radius_key) for poly_id in all_poly_ids]
    results = {}
    with Pool(processes=cpu_count()) as pool:
        for i, (poly_id, point_ids) in enumerate(pool.imap_unordered(_associate_quad_to_points_task, args_list, chunksize=100)):
            results[poly_id] = point_ids
            sys.stdout.write(f"\r{grid_name.capitalize()} association processing: {i+1}/{len(all_poly_ids)}")
            sys.stdout.flush()
    print(f"\nFinished associating {grid_name} for {len(results)} quads.")
    return results

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


    oob_points = generate_oob_points(raw_quads_data, grid, quads_by_id)

    # --- 核心修改: 使用CARLA航点进行增强，而非从头生成 ---
    enhanced_waypoints = enhance_w_lanes_from_carla(driving_waypoints, grid, quads_by_id)
    
    global_waypoints = downsample_w_lane_waypoints(
        enhanced_waypoints,
        CONFIG["W_LANE_DOWNSAMPLE_LONGITUDINAL_THRESHOLD"]
    )
    # --- 更新: 调用新函数，同时获取前后两个方向的映射 ---
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
    }
    
    print(f"Saving final processed map data to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(final_processed_data, f, cls=NumpyEncoder)
    
    end_time = time.time()
    print(f"--- Preprocessing complete in {end_time - start_time:.2f} seconds. ---")
    print(f"Final file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    # 使用默认配置文件中的地图路径
    import yaml
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    map_path = config['simulator']['map_path'] #导出路径
    stage1_path = "./maps/carla_map_data_Town03_stitched.json"
    output_path = map_path                     #导出路径
    print(f"Using default map path: {map_path}")
    print(f"Output path: {output_path}")
    if not os.path.exists(stage1_path):
        print(f"Error: Input file not found at {stage1_path}"); sys.exit(1)
    preprocess_map(stage1_path, output_path)

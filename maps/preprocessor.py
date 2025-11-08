import warnings
# 抑制PyTorch CUDA的FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')
import ezdxf, math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch
import json
import os

# 导入工具模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry_utils import (
    calculate_distance, is_point_in_quad_2d, is_points_in_quads_gpu,
    normalize_angle, angle_difference, normalize_angle_degrees,
    quads_intersection_matrix_gpu
)
from utils.spatial_hash import SpatialHash
from utils.visualize_utils import visualize_map_from_json, visualize_map

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
config = load_config()

# 读取DXF文件
dxf_path = os.path.join(os.path.dirname(__file__), "town2.dxf")
doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()
# 创建matplotlib图形
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')

# 存储所有几何数据
lines_data = []
circles_data = []
arcs_data = []
# 存储所有四边形数据
polygons_data = []

# 从配置文件读取参数
TOLERANCE = config['preprocessor']['tolerance']
SAMPLE_DISTANCE = config['preprocessor']['sample_distance']
RECTANGLE_LENGTH = config['preprocessor']['rectangle_length']
RECTANGLE_WIDTH = config['preprocessor']['rectangle_width']
OOB_NUDGE_DISTANCE = config['preprocessor']['oob_nudge_distance']
CELL_SIZE = config['preprocessor']['cell_size']
W_LANE_SAMPLE_DISTANCE = config['preprocessor']['w_lane_sample_distance']

# 道路ID计数器
road_id_counter = 1
# 四边形ID计数器
poly_id_counter = 1

# GPU设备检测
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'
print(f"🚀 GPU加速状态: {'启用' if USE_GPU else '禁用'} (设备: {DEVICE})")

def increment_road_id():
    """递增道路ID并返回新ID"""
    global road_id_counter
    road_id_counter += 1
    return road_id_counter - 1

# ==================== 路网方向校正（GPU加速） ====================
def adjust_road_directions_gpu(lines_data, arcs_data, tolerance: float, device: str = DEVICE):
    """
    基于四边形分段，以树/BFS方式统一道路方向：
    - 从最小 road_id 开始；取该路当前定义的“终点”。
    - 判断此终点落入的其它 road 的四边形：若在对方的“start 四边形”，对方保持方向；若在“end 四边形”，反转对方方向；
      若落在对方中间四边形，则比较与对方 start/end 四边形中心的距离，近者为连接端（近 start 则保持，近 end 则反转）。
    - 重复直到所有可达道路方向确定。
    """
    if len(lines_data) + len(arcs_data) == 0:
        return

    # 收集 per-road 四边形（保持生成顺序）
    road_to_quads = {}
    for q in polygons_data:
        rid = q['road_id']
        road_to_quads.setdefault(rid, []).append(q)
    if not road_to_quads:
        return

    # 构建空间索引用于候选检索
    # 本地四边形索引
    quads_by_id = {q['poly_id']: q for q in polygons_data}

    # 使用 SpatialHash 构建适配器，提供 get_candidates(point_2d) 接口
    idx_to_poly_id = [q['poly_id'] for q in polygons_data]
    # 计算全局 bounds 与每个quad的AABB
    if polygons_data:
        verts_all = np.array([[v[0], v[1]] for q in polygons_data for v in q['vertices']], dtype=np.float32)
        min_bounds = torch.tensor(verts_all.min(axis=0), dtype=torch.float32, device=DEVICE)
        max_bounds = torch.tensor(verts_all.max(axis=0), dtype=torch.float32, device=DEVICE)
        # 每个quad AABB
        quad_mins = []
        quad_maxs = []
        for q in polygons_data:
            v2 = np.array([[v[0], v[1]] for v in q['vertices']], dtype=np.float32)
            quad_mins.append(v2.min(axis=0))
            quad_maxs.append(v2.max(axis=0))
        static_bounds = torch.tensor(np.stack([np.stack([quad_mins[i], quad_maxs[i]], axis=0) for i in range(len(polygons_data))], axis=0), dtype=torch.float32, device=DEVICE)
        try:
            sh = SpatialHash(cell_size=float(CELL_SIZE), min_bounds=min_bounds, max_bounds=max_bounds, device=torch.device(DEVICE))
            sh.build_static_index(static_bounds)
        except Exception:
            sh = None
    else:
        sh = None

    class _HashAdapter:
        def __init__(self, h, device, idx_to_pid):
            self.h = h
            self.device = device
            self.idx_to_pid = idx_to_pid
        def get_candidates(self, point_2d):
            if self.h is None:
                return []
            pt = torch.tensor([[float(point_2d[0]), float(point_2d[1])]], dtype=torch.float32, device=self.h.device)
            pairs = self.h.query_points(pt)
            if pairs.numel() == 0:
                return []
            item_idx = pairs[:, 1].tolist()
            return [self.idx_to_pid[i] for i in item_idx if 0 <= i < len(self.idx_to_pid)]

    grid = _HashAdapter(sh, DEVICE, idx_to_poly_id)

    # 便捷 map
    line_map = {r['road_id']: r for r in lines_data}
    arc_map = {r['road_id']: r for r in arcs_data}
    all_ids = sorted(list(road_to_quads.keys()))
    if not all_ids:
        return

    # 工具：获取路的几何终点坐标（按当前定义）
    def get_end_xy(rid):
        if rid in line_map:
            r = line_map[rid]
            return (float(r['end'][0]), float(r['end'][1]))
        elif rid in arc_map:
            r = arc_map[rid]
            cx, cy = r['center'][0], r['center'][1]
            rad = r['radius']
            ang = math.radians(r['end_angle'])
            return (float(cx + rad * math.cos(ang)), float(cy + rad * math.sin(ang)))
        return None

    # 工具：反转一条路的方向
    reversed_rids = set()

    def reverse_road(rid):
        if rid in line_map:
            r = line_map[rid]
            r['start'], r['end'] = r['end'], r['start']
        elif rid in arc_map:
            r = arc_map[rid]
            # 仅反转行驶方向，保持几何：不要更改 start/end 角度
            r['direction'] = - r.get('direction', 1)
        
        # 同时反转该道路上所有四边形的方向角度，并调整顶点定义（0<->2, 1<->3）
        if rid in road_to_quads:
            for quad in road_to_quads[rid]:
                # 方向角反转180度
                current_angle = quad.get('direction_angle', 0.0)
                quad['direction_angle'] = (current_angle + math.pi) % (2 * math.pi)
                # 顶点顺序交换以匹配左右翻转：0<->2, 1<->3
                verts = list(quad.get('vertices', []))
                if len(verts) == 4:
                    v0, v1, v2, v3 = verts
                    verts_swapped = [v2, v3, v0, v1]
                    quad['vertices'] = verts_swapped

        # 记录该道路已发生反转，用于最后调整poly_id顺序
        reversed_rids.add(rid)

    # 点是否在 quad 内（带容差：允许点到多边形最近边的距离 < tolerance 也认为命中）
    def point_hits_quad(px, py, quad):
        verts2d = [[v[0], v[1]] for v in quad['vertices']]
        if is_point_in_quad_2d([px, py], verts2d):
            return True
        # 简容差：与四条边的距离最小值
        min_d = 1e9
        for i in range(4):
            x1, y1 = verts2d[i]
            x2, y2 = verts2d[(i + 1) % 4]
            vx, vy = x2 - x1, y2 - y1
            wx, wy = px - x1, py - y1
            seg_len2 = vx * vx + vy * vy
            t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
            projx, projy = x1 + t * vx, y1 + t * vy
            d = math.hypot(px - projx, py - projy)
            if d < min_d:
                min_d = d
        return min_d <= tolerance

    # 阶段1：建立约束边（u, v, parity, weight）
    edges = []
    seen_pair = {}

    def get_end_tangent(rid):
        if rid in line_map:
            r = line_map[rid]
            vx = float(r['end'][0] - r['start'][0])
            vy = float(r['end'][1] - r['start'][1])
            n = math.hypot(vx, vy) or 1.0
            return (vx / n, vy / n)
        elif rid in arc_map:
            r = arc_map[rid]
            ang = math.radians(r['end_angle'])
            return (-math.sin(ang), math.cos(ang))
        return (1.0, 0.0)

    def neighbor_tangent_at(rid, where, quad_for_middle=None):
        if where == 'start':
            if rid in line_map:
                r = line_map[rid]
                vx = float(r['end'][0] - r['start'][0])
                vy = float(r['end'][1] - r['start'][1])
                n = math.hypot(vx, vy) or 1.0
                return (vx / n, vy / n)
            else:
                r = arc_map[rid]
                ang = math.radians(r['start_angle'])
                return (-math.sin(ang), math.cos(ang))
        elif where == 'end':
            if rid in line_map:
                r = line_map[rid]
                vx = float(r['end'][0] - r['start'][0])
                vy = float(r['end'][1] - r['start'][1])
                n = math.hypot(vx, vy) or 1.0
                return (vx / n, vy / n)
            else:
                r = arc_map[rid]
                ang = math.radians(r['end_angle'])
                return (-math.sin(ang), math.cos(ang))
        else:
            if quad_for_middle is None:
                return (1.0, 0.0)
            px, py = quad_for_middle['center']
            if rid in line_map:
                r = line_map[rid]
                vx = float(r['end'][0] - r['start'][0])
                vy = float(r['end'][1] - r['start'][1])
                n = math.hypot(vx, vy) or 1.0
                return (vx / n, vy / n)
            else:
                r = arc_map[rid]
                cx, cy = r['center'][0], r['center'][1]
                ang = math.atan2(py - cy, px - cx)
                return (-math.sin(ang), math.cos(ang))

    for cur in all_ids:
        exy = get_end_xy(cur)
        if exy is None:
            continue
        ex, ey = exy
        tx_cur, ty_cur = get_end_tangent(cur)

        cand_quads = []
        if grid is not None:
            for pid in grid.get_candidates([ex, ey]):
                cand_quads.append(quads_by_id.get(pid))
        else:
            cand_quads = polygons_data

        hits = []
        for quad in cand_quads:
            if quad is None:
                continue
            rid = quad['road_id']
            if rid == cur:
                continue
            if point_hits_quad(ex, ey, quad):
                quads = road_to_quads.get(rid, [])
                if not quads:
                    continue
                start_quad = quads[0]
                end_quad = quads[-1]
                if quad['poly_id'] == start_quad['poly_id']:
                    pos = 'start'
                elif quad['poly_id'] == end_quad['poly_id']:
                    pos = 'end'
                else:
                    pos = 'middle'
                cx, cy = quad['center']
                dc = math.hypot(ex - cx, ey - cy)
                hits.append((rid, pos, quad, dc))

        # 选每个 rid 的最佳命中
        sel = {}
        for rid, pos, quad, dc in hits:
            prio = 0 if pos == 'start' else (1 if pos == 'end' else 2)
            key = rid
            if key not in sel or (prio, dc) < sel[key][0]:
                sel[key] = ((prio, dc), (pos, quad, dc))

        for rid, (_, (pos, quad, dc)) in sel.items():
            # 计算期望奇偶（同向0/反向1）
            if pos == 'start':
                txn, tyn = neighbor_tangent_at(rid, 'start')
            elif pos == 'end':
                txn, tyn = neighbor_tangent_at(rid, 'end')
            else:
                txn, tyn = neighbor_tangent_at(rid, 'middle', quad)
            dot = tx_cur * txn + ty_cur * tyn
            parity = 0 if dot >= 0.0 else 1
            weight = dc + (-1e-3 if pos != 'middle' else 0.0)
            a, b = (cur, rid) if cur < rid else (rid, cur)
            if (a, b) not in seen_pair or weight < seen_pair[(a, b)][2]:
                seen_pair[(a, b)] = (a, b, parity, weight)

    edges = list(seen_pair.values())
    # 按权重从小到大（越可靠越先融合）
    edges.sort(key=lambda e: e[3])

    # 阶段2：并查集（带奇偶）做全局赋值
    id_to_idx = {rid: idx for idx, rid in enumerate(all_ids)}
    n = len(all_ids)
    parent = list(range(n))
    rank = [0] * n
    # parity_to_parent: 节点到父节点的奇偶
    parity_to_parent = [0] * n

    def find(x):
        if parent[x] != x:
            rx, px = find(parent[x])
            parity_to_parent[x] ^= px
            parent[x] = rx
        return parent[x], parity_to_parent[x]

    def union(x, y, p):
        rx, px = find(x)
        ry, py = find(y)
        if rx == ry:
            # 已在同一集合，检查一致性：px XOR py 应等于 p
            return (px ^ py) == p
        # 合并，维护 rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
            px, py = py, px
        parent[ry] = rx
        # 设置 ry 到 rx 的奇偶，使得 x XOR y = p
        # px XOR parity_ry XOR py = p  => parity_ry = px XOR py XOR p
        parity_to_parent[ry] = px ^ py ^ p
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for a, b, p, _w in edges:
        union(id_to_idx[a], id_to_idx[b], p)

    # 计算每个 rid 的最终 flip（相对其根）
    flip = {}
    for rid in all_ids:
        r, pr = find(id_to_idx[rid])
        flip[rid] = pr & 1

    # 应用翻转
    for rid, f in flip.items():
        if f:
            reverse_road(rid)

    # 回填 lists（line_map/arc_map 已是引用，通常不需要；保守覆盖）
    for i, item in enumerate(lines_data):
        rid = item['road_id']
        if rid in line_map:
            lines_data[i] = line_map[rid]
    for i, item in enumerate(arcs_data):
        rid = item['road_id']
        if rid in arc_map:
            arcs_data[i] = arc_map[rid]

    # 若某些道路发生了反转，为保证每条路的 poly_id 从新的 start → end 递增，
    # 将该道路的四边形顺序反转，并把 poly_id 重新赋予为原集合中的升序值
    if reversed_rids:
        for rid in reversed_rids:
            quads = road_to_quads.get(rid, [])
            if not quads:
                continue
            # 反转顺序：新的索引0对应新的start侧
            quads_reordered = list(reversed(quads))
            # 取该路原有 poly_id 集合（升序），保持全局唯一且总量不变
            old_ids_sorted = sorted([q['poly_id'] for q in quads])
            # 逐个重写 poly_id，并同步更新 quads_by_id 映射
            for idx, q in enumerate(quads_reordered):
                old_pid = q['poly_id']
                new_pid = old_ids_sorted[idx]
                if old_pid != new_pid:
                    # 删除旧键，写入新键
                    if old_pid in quads_by_id:
                        del quads_by_id[old_pid]
                    q['poly_id'] = new_pid
                    quads_by_id[new_pid] = q
                else:
                    # 键未变更也确保映射一致
                    quads_by_id[new_pid] = q
            # 回填新的顺序
            road_to_quads[rid] = quads_reordered

    print("路网方向校正完成（基于四边形连接的树式推进，poly_id已按新方向递增）")

def compute_lane_s_for_quads(polygons_data_ref):
    """
    为所有 (road_id, lane_id) 对应的车道，按当前 poly_id 顺序计算每个 quad 的弧长参数 s：
    - s[0] = 0
    - s[i] = s[i-1] + ||center_i - center_{i-1}||
    计算结果写入 q['s']。
    依赖：同一路(road_id,lane_id)内部 poly_id 已沿行驶方向递增（可先调用 ensure_lane_poly_id_monotonic）。
    """
    # 分组
    lane_groups = {}
    for q in polygons_data_ref:
        lk = (q.get('road_id'), q.get('lane_id', 1))
        lane_groups.setdefault(lk, []).append(q)
    # 每组按 poly_id 排序，保证从 start→end
    for lk in lane_groups:
        lane_groups[lk].sort(key=lambda it: it['poly_id'])
        s_val = 0.0
        prev_c = None
        for q in lane_groups[lk]:
            c = q['center']
            if prev_c is None:
                s_val = 0.0
            else:
                s_val += math.hypot(float(c[0]) - float(prev_c[0]), float(c[1]) - float(prev_c[1]))
            q['s'] = float(s_val)
            prev_c = c

def compute_lane_curvature(polygons_data_ref, eps: float = 1e-8):
    """
    为所有 (road_id, lane_id) 的车道计算每个 quad 的曲率 kappa（单位：1/m）：
    - 对中间点 i 使用三点法 (i-1, i, i+1)。
    - 两端点使用单边三点法：起点用 (0,1,2)，终点用 (n-3,n-2,n-1)。
    计算结果写入 q['curvature']（带符号，基于三点有向面积）。
    要求组内 poly_id 已按行驶方向递增。
    """
    def curvature_three_points(p0, p1, p2):
        ax, ay = float(p0[0]), float(p0[1])
        bx, by = float(p1[0]), float(p1[1])
        cx, cy = float(p2[0]), float(p2[1])
        abx, aby = bx - ax, by - ay
        bcx, bcy = cx - bx, cy - by
        acx, acy = cx - ax, cy - ay
        ab = math.hypot(abx, aby)
        bc = math.hypot(bcx, bcy)
        ac = math.hypot(acx, acy)
        # 有向面积（2*Area = cross(AB, AC)）
        cross = abx * acy - aby * acx
        denom = max(ab * bc * ac, eps)
        k = (2.0 * cross) / denom
        return k

    # 分组
    lane_groups = {}
    for q in polygons_data_ref:
        lk = (q.get('road_id'), q.get('lane_id', 1))
        lane_groups.setdefault(lk, []).append(q)

    for lk in lane_groups:
        quads = lane_groups[lk]
        if len(quads) < 2:
            for q in quads:
                q['curvature'] = 0.0
            continue
        quads.sort(key=lambda it: it['poly_id'])
        n = len(quads)
        centers = [tuple(q['center']) for q in quads]
        # 起点（单边）
        if n >= 3:
            k0 = curvature_three_points(centers[0], centers[1], centers[2])
        else:
            # 仅两点无法稳定求曲率，设0
            k0 = 0.0
        quads[0]['curvature'] = float(k0)
        # 中间
        for i in range(1, n - 1):
            k = curvature_three_points(centers[i - 1], centers[i], centers[i + 1])
            quads[i]['curvature'] = float(k)
        # 终点（单边）
        if n >= 3:
            kn = curvature_three_points(centers[n - 3], centers[n - 2], centers[n - 1])
        else:
            kn = 0.0
        quads[-1]['curvature'] = float(kn)

# ==================== GPU加速函数 ====================

def generate_line_points_gpu(start, end, sample_distance, device='cuda'):
    """GPU加速的直线采样点生成"""
    start_tensor = torch.tensor(start[:2], dtype=torch.float32, device=device)
    end_tensor = torch.tensor(end[:2], dtype=torch.float32, device=device)
    
    # 计算直线长度
    length = torch.norm(end_tensor - start_tensor).item()
    
    # 计算采样点数量
    n_points = max(1, int(length / sample_distance) + 1)
    
    # 生成参数化t值
    t_values = torch.linspace(0, 1, n_points, device=device)
    
    # 批量计算所有采样点
    points = start_tensor.unsqueeze(0) + t_values.unsqueeze(1) * (end_tensor - start_tensor).unsqueeze(0)
    
    # 将最后一个点精确设置为终点
    points[-1] = end_tensor
    
    return points.cpu().numpy(), length

def generate_circle_points_gpu(center, radius, sample_distance, device='cuda'):
    """GPU加速的圆形采样点生成"""
    center_tensor = torch.tensor(center[:2], dtype=torch.float32, device=device)
    radius_tensor = torch.tensor(radius, dtype=torch.float32, device=device)
    
    circumference = 2 * np.pi * radius
    n_points = max(1, int(circumference / sample_distance) + 1)
    
    # 生成角度
    angles = torch.linspace(0, 2 * np.pi, n_points, device=device)
    
    # 批量计算所有采样点
    points = center_tensor.unsqueeze(0) + radius_tensor * torch.stack([
        torch.cos(angles),
        torch.sin(angles)
    ], dim=1)
    
    return points.cpu().numpy(), circumference

def generate_arc_points_gpu(center, radius, start_angle, end_angle, sample_distance, device='cuda'):
    """GPU加速的圆弧采样点生成"""
    center_tensor = torch.tensor(center[:2], dtype=torch.float32, device=device)
    radius_tensor = torch.tensor(radius, dtype=torch.float32, device=device)
    
    # 计算圆弧长度
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 2 * np.pi
    
    arc_length = radius * angle_diff
    n_points = max(1, int(arc_length / sample_distance) + 1)
    
    # 生成角度
    t_values = torch.linspace(0, 1, n_points, device=device)
    angles = start_angle + t_values * angle_diff
    
    # 批量计算所有采样点
    points = center_tensor.unsqueeze(0) + radius_tensor * torch.stack([
        torch.cos(angles),
        torch.sin(angles)
    ], dim=1)
    
    return points.cpu().numpy(), arc_length

def compute_trapezoid_vertices_gpu(points, angles, rectangle_length, rectangle_width, device='cuda'):
    """
    GPU加速的梯形顶点计算
    
    参数:
    points: (N, 2) 采样点坐标
    angles: (N,) 采样点的方向角度
    rectangle_length: 梯形长度
    rectangle_width: 梯形宽度
    
    返回:
    vertices: (N, 4, 2) 每个点的四个顶点
    """
    N = len(points)
    
    # 转换为GPU tensor（先转为numpy数组避免警告）
    points_array = np.array(points)
    angles_array = np.array(angles)
    points_tensor = torch.tensor(points_array, dtype=torch.float32, device=device)  # (N, 2)
    angles_tensor = torch.tensor(angles_array, dtype=torch.float32, device=device)  # (N,)
    
    # 计算切线方向和法线方向向量
    tangent_vectors = torch.stack([
        torch.cos(angles_tensor),
        torch.sin(angles_tensor)
    ], dim=1)  # (N, 2)
    
    normal_vectors = torch.stack([
        -torch.sin(angles_tensor),
        torch.cos(angles_tensor)
    ], dim=1)  # (N, 2)
    
    # 计算参数
    bottom_half = rectangle_length / 2
    top_half = rectangle_length / 2
    height_half = rectangle_width / 2
    
    # 计算四个顶点（批量计算）
    # 上底左顶点: center + top_half * tangent - height_half * normal
    top_left = points_tensor + top_half * tangent_vectors - height_half * normal_vectors
    
    # 上底右顶点: center + top_half * tangent + height_half * normal
    top_right = points_tensor + top_half * tangent_vectors + height_half * normal_vectors
    
    # 下底右顶点: center - bottom_half * tangent + height_half * normal
    bottom_right = points_tensor - bottom_half * tangent_vectors + height_half * normal_vectors
    
    # 下底左顶点: center - bottom_half * tangent - height_half * normal
    bottom_left = points_tensor - bottom_half * tangent_vectors - height_half * normal_vectors
    
    # 组合成顶点数组 (N, 4, 2)
    vertices = torch.stack([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dim=1)
    
    return vertices.cpu().numpy()

def check_duplicates_gpu(centers, radii=None, tolerance=None, device='cuda'):
    """
    GPU加速的重复检测
    
    参数:
    centers: (N, 2) 中心点数组
    radii: (N,) 半径数组（可选）
    tolerance: 容差
    
    返回:
    is_unique: (N,) 布尔数组，True表示唯一
    """
    if tolerance is None:
        tolerance = TOLERANCE
    
    # 先转为numpy数组避免警告
    centers_array = np.array(centers)
    centers_tensor = torch.tensor(centers_array, dtype=torch.float32, device=device)  # (N, 2)
    N = len(centers)
    
    # 计算所有点对之间的距离
    # centers_tensor.unsqueeze(0): (1, N, 2)
    # centers_tensor.unsqueeze(1): (N, 1, 2)
    # dists: (N, N)
    dists = torch.norm(centers_tensor.unsqueeze(0) - centers_tensor.unsqueeze(1), dim=2)
    
    # 找出距离小于容差的点对
    # 使用torch.fill_diagonal_避免与自己比较
    mask = dists < tolerance
    
    # 标记重复点
    is_unique = ~torch.any(mask & (dists > 0), dim=1)
    
    return is_unique.cpu().numpy()

def compute_directions_gpu(points, device='cuda'):
    """
    GPU加速的方向角度计算
    
    参数:
    points: (N, 2) 采样点数组
    
    返回:
    angles: (N,) 方向角度数组
    """
    # 先转为numpy数组避免警告
    points_array = np.array(points)
    points_tensor = torch.tensor(points_array, dtype=torch.float32, device=device)  # (N, 2)
    N = len(points)
    
    if N <= 1:
        return torch.zeros(N, device=device).cpu().numpy()
    
    # 计算相邻点之间的向量
    vectors = points_tensor[1:] - points_tensor[:-1]  # (N-1, 2)
    
    # 计算角度
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # (N-1,)
    
    # 最后一个点使用前一个点的方向
    angles = torch.cat([angles, angles[-1:]])  # (N,)
    
    return angles.cpu().numpy()

def process_polygons_gpu(points, angles, road_id, rectangle_length, rectangle_width, device='cuda'):
    """
    GPU加速的批量梯形处理
    
    参数:
    points: (N, 2) 采样点数组
    angles: (N,) 方向角度数组
    road_id: 道路ID
    rectangle_length: 矩形长度
    rectangle_width: 矩形宽度
    device: GPU设备
    
    返回:
    polygons_data: 四边形数据列表
    """
    if len(points) == 0:
        return []
    
    # 转换为numpy数组
    points_np = np.array(points)
    angles_np = np.array(angles)
    
    # 计算每个点的下一个角度
    next_angles = np.concatenate([angles_np[1:], angles_np[-1:]])
    
    # 计算角度差
    angle_diffs = np.abs(np.diff(np.concatenate([angles_np, angles_np[-1:]])))
    
    # 计算上底宽度
    top_widths = rectangle_length * (1 + angle_diffs / np.pi)
    
    # 批量计算所有梯形顶点
    vertices = compute_trapezoid_vertices_gpu(
        points_np, angles_np, rectangle_length, rectangle_width, device
    )
    
    # 转换回数据格式
    polygons_data = []
    global poly_id_counter
    
    for i, (point, vertex_set) in enumerate(zip(points_np, vertices)):
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (point[0], point[1]),
            'vertices': [(v[0], v[1], 0.0) for v in vertex_set],
            'direction_angle': angles_np[i]
        })
        poly_id_counter += 1
    
    return polygons_data

# ==================== GPU加速函数结束 ====================

def create_seamless_trapezoid(center_x, center_y, tangent_angle, normal_angle, 
                             bottom_width, top_width, height, prev_vertices=None):
    """
    创建无缝连接的梯形，确保与相邻梯形紧密连接
    
    参数:
    center_x, center_y: 梯形中心点
    tangent_angle: 切线方向角度
    normal_angle: 法线方向角度
    bottom_width: 下底宽度
    top_width: 上底宽度
    height: 梯形高度
    prev_vertices: 前一个梯形的顶点（用于无缝连接）
    
    返回:
    vertices: 四个顶点坐标
    """
    if prev_vertices is None:
        # 第一个梯形，创建起始梯形
        # 标准化角度
        tangent_angle = normalize_angle(tangent_angle)
        normal_angle = normalize_angle(normal_angle)
        
        # 计算切线方向单位向量
        t_x = np.cos(tangent_angle)
        t_y = np.sin(tangent_angle)
        
        n_x = np.cos(normal_angle)
        n_y = np.sin(normal_angle)
        
        # 计算四个顶点
        bottom_half = bottom_width / 2
        top_half = top_width / 2
        height_half = height / 2
        
        # 下底两个顶点：p_i - (bottom_i / 2) * t_i ± (h_i / 2) * n_i
        bottom_left_x = center_x - bottom_half * t_x - height_half * n_x
        bottom_left_y = center_y - bottom_half * t_y - height_half * n_y
        
        bottom_right_x = center_x - bottom_half * t_x + height_half * n_x
        bottom_right_y = center_y - bottom_half * t_y + height_half * n_y
        
        # 上底两个顶点：p_i + (top_i / 2) * t_i ± (h_i / 2) * n_i
        top_left_x = center_x + top_half * t_x - height_half * n_x
        top_left_y = center_y + top_half * t_y - height_half * n_y
        
        top_right_x = center_x + top_half * t_x + height_half * n_x
        top_right_y = center_y + top_half * t_y + height_half * n_y
        
        # 四个顶点（按顺序排列）
        vertices = [
            (top_left_x, top_left_y),      # 上底左顶点
            (top_right_x, top_right_y),     # 上底右顶点
            (bottom_right_x, bottom_right_y),  # 下底右顶点
            (bottom_left_x, bottom_left_y)   # 下底左顶点
        ]
    else:
        # 后续梯形，与前一个梯形无缝连接
        prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
        prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
        
        # 计算当前梯形的下底顶点
        t_x = np.cos(tangent_angle)
        t_y = np.sin(tangent_angle)
        
        n_x = np.cos(normal_angle)
        n_y = np.sin(normal_angle)
        
        # 计算当前梯形的下底顶点
        bottom_half = bottom_width / 2
        height_half = height / 2
        
        # 下底左顶点
        bottom_left_x = center_x - bottom_half * t_x - height_half * n_x
        bottom_left_y = center_y - bottom_half * t_y - height_half * n_y
        
        # 下底右顶点
        bottom_right_x = center_x - bottom_half * t_x + height_half * n_x
        bottom_right_y = center_y - bottom_half * t_y + height_half * n_y
        
        # 四个顶点
        vertices = [
            prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
            prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
            (bottom_right_x, bottom_right_y),  # 下底右顶点
            (bottom_left_x, bottom_left_y)     # 下底左顶点
        ]
    
    return vertices

def check_duplicate_geometry(new_item, existing_items, tolerance=None):
    """检查几何体是否重复"""
    if tolerance is None:
        tolerance = TOLERANCE
        
    for existing_item in existing_items:
        if calculate_distance(new_item['center'], existing_item['center']) < tolerance:
            # 检查其他参数
            if 'radius' in new_item and 'radius' in existing_item:
                if abs(new_item['radius'] - existing_item['radius']) < tolerance:
                    # 对于圆弧，还需要检查角度
                    if 'start_angle' in new_item and 'start_angle' in existing_item:
                        start_angle_diff = abs(new_item['start_angle'] - existing_item['start_angle'])
                        end_angle_diff = abs(new_item['end_angle'] - existing_item['end_angle'])
                        
                        # 处理角度跨越0度的情况
                        if start_angle_diff > 180:
                            start_angle_diff = 360 - start_angle_diff
                        if end_angle_diff > 180:
                            end_angle_diff = 360 - end_angle_diff
                            
                        if start_angle_diff < 1.0 and end_angle_diff < 1.0:
                            return True
                    else:
                        # 圆形或线条
                        return True
            elif 'start' in new_item and 'start' in existing_item:
                # 检查起点和终点
                start_dist = calculate_distance(new_item['start'], existing_item['start'])
                end_dist = calculate_distance(new_item['end'], existing_item['end'])
                start_end_dist = calculate_distance(new_item['start'], existing_item['end'])
                end_start_dist = calculate_distance(new_item['end'], existing_item['start'])
                if ((start_dist < tolerance and end_dist < tolerance) or 
                    (start_end_dist < tolerance and end_start_dist < tolerance)):
                    return True
    return False

def sample_line_points(start, end, sample_distance):
    """为直线生成等间距采样点"""
    start_x, start_y = start[0], start[1]
    end_x, end_y = end[0], end[1]
    
    # 计算直线长度
    length = calculate_distance(start, end)
    
    # 计算采样点数量
    n_points = max(1, int(length / sample_distance) + 1)
    
    # 生成采样点
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            x, y = start_x, start_y
        elif i == n_points - 1:
            # 最后一个点，精确使用终点
            x, y = end_x, end_y
        else:
            # 中间点，使用参数化方法
            t = i / (n_points - 1)
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
        points.append((x, y))
    
    return points, length

def sample_circle_points(center, radius, sample_distance):
    """为圆形生成等间距采样点"""
    circumference = 2 * np.pi * radius
    n_points = max(1, int(circumference / sample_distance) + 1)
    
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            angle = 0
        elif i == n_points - 1:
            # 最后一个点，精确使用起点（圆形闭合）
            angle = 0
        else:
            # 中间点，使用参数化方法
            angle = 2 * np.pi * i / n_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    
    return points, circumference

def sample_arc_points(center, radius, start_angle, end_angle, sample_distance):
    """为圆弧生成等间距采样点"""
    # 计算圆弧长度
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 2 * np.pi
    
    arc_length = radius * angle_diff
    n_points = max(1, int(arc_length / sample_distance) + 1)
    
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            angle = start_angle
        elif i == n_points - 1:
            # 最后一个点，精确使用终点
            angle = end_angle
        else:
            # 中间点，使用参数化方法
            t = i / (n_points - 1)
            angle = start_angle + t * angle_diff
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    
    return points, arc_length

def generate_oob_points_cpu(quads, grid, quads_by_id, nudge_distance=OOB_NUDGE_DISTANCE):
    """CPU版本的OOB点生成（原始实现）"""
    print("Generating Out-of-Bounds (OOB) points using CPU...")
    oob_points = []
    nudge = nudge_distance
    for i, quad in enumerate(quads):
        if (i + 1) % 100 == 0 or i + 1 == len(quads):
            print(f"\rOOB processing: {i+1}/{len(quads)}", end="")
        verts_3d = np.array([[v[0], v[1], 0.0] for v in quad['vertices']])  # 添加z坐标
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
            if is_point_in_quad_2d(test_point_2d, [[v[0], v[1]] for v in quad['vertices']]):
                normal_2d = -normal_2d
            oob_candidate_2d = mid_point_2d + normal_2d * nudge
            is_inside_any = False
            for cand_poly_id in grid.get_candidates(oob_candidate_2d):
                if is_point_in_quad_2d(oob_candidate_2d, [[v[0], v[1]] for v in quads_by_id[cand_poly_id]['vertices']]):
                    is_inside_any = True
                    break
            if not is_inside_any:
                oob_points.append({'x': oob_candidate_2d[0], 'y': oob_candidate_2d[1], 'z': mid_point_3d[2]})
    print(f"\nGenerated {len(oob_points)} OOB points.")
    return oob_points

def generate_oob_points_gpu(quads, grid, quads_by_id, device='cuda', batch_size=500, nudge_distance=OOB_NUDGE_DISTANCE):
    """内存友好的GPU加速版本 - 智能分批处理避免OOM"""
    print(f"Generating Out-of-Bounds (OOB) points using memory-friendly GPU processing ({device})...")
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA不可用，回退到CPU版本")
        return generate_oob_points_cpu(quads, grid, quads_by_id, nudge_distance)
    
    device = torch.device(device)
    nudge = nudge_distance
    
    # 检查GPU内存并动态调整batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU内存: {gpu_memory:.1f}GB")
        
        # 根据GPU内存动态调整batch_size
        if gpu_memory < 4:
            batch_size = min(batch_size, 200)
        elif gpu_memory < 8:
            batch_size = min(batch_size, 500)
        else:
            batch_size = min(batch_size, 1000)
        
        print(f"调整后的batch_size: {batch_size}")
    
    # 1. 预处理：收集所有OOB候选点（CPU上完成）
    print("预处理：收集OOB候选点...")
    all_candidates = []
    candidate_metadata = []
    
    for quad in quads:
        verts = np.array([[v[0], v[1], 0.0] for v in quad['vertices']])
        
        for i_edge in range(4):
            p1_3d, p2_3d = verts[i_edge], verts[(i_edge + 1) % 4]
            mid_point_3d = (p1_3d + p2_3d) / 2.0
            mid_point_2d = mid_point_3d[:2]
            edge_vec_2d = p2_3d[:2] - p1_3d[:2]
            normal_2d = np.array([edge_vec_2d[1], -edge_vec_2d[0]])
            norm = np.linalg.norm(normal_2d)
            
            if norm < 1e-6:
                continue
                
            normal_2d /= norm
            test_point_2d = mid_point_2d + normal_2d * 0.01
            
            # 检查法向量方向
            if is_point_in_quad_2d(test_point_2d, [[v[0], v[1]] for v in quad['vertices']]):
                normal_2d = -normal_2d
                
            oob_candidate_2d = mid_point_2d + normal_2d * nudge
            
            all_candidates.append(oob_candidate_2d)
            candidate_metadata.append({
                'mid_point_3d': mid_point_3d,
                'quad_id': quad['poly_id']
            })
    
    total_candidates = len(all_candidates)
    print(f"总共 {total_candidates} 个OOB候选点需要验证")
    
    # 2. 预先计算所有四边形的2D顶点（避免重复计算）
    print("准备四边形数据...")
    quads_2d = []
    for quad in quads:
        verts_2d = [[v[0], v[1]] for v in quad['vertices']]
        quads_2d.append(verts_2d)
    
    # 3. 分批处理候选点（避免OOM）
    print("分批验证OOB候选点...")
    valid_oob_points = []
    
    # 将候选点分批处理
    for batch_start in range(0, total_candidates, batch_size):
        batch_end = min(batch_start + batch_size, total_candidates)
        batch_candidates = all_candidates[batch_start:batch_end]
        batch_metadata = candidate_metadata[batch_start:batch_end]
        
        try:
            # 将当前批次的候选点转换为GPU tensor（先转为numpy数组）
            batch_candidates_array = np.array(batch_candidates)
            candidates_tensor = torch.tensor(batch_candidates_array, dtype=torch.float32, device=device)  # (B, 2)
            
            # 将四边形数据转换为GPU tensor（先转为numpy数组）
            quads_2d_array = np.array(quads_2d)
            quads_tensor = torch.tensor(quads_2d_array, dtype=torch.float32, device=device)  # (N, 4, 2)
            
            # 检查当前批次的候选点是否在任何四边形内部
            if len(candidates_tensor) > 0 and len(quads_tensor) > 0:
                # 使用批量检测函数
                is_inside_matrix = is_points_in_quads_gpu(candidates_tensor, quads_tensor, device)  # (B, N)
                is_inside_any = torch.any(is_inside_matrix, dim=1)  # (B,)
                
                # 收集有效的OOB点
                valid_indices = (~is_inside_any).nonzero(as_tuple=True)[0]
                for idx in valid_indices:
                    metadata = batch_metadata[idx]
                    valid_oob_points.append({
                        'x': candidates_tensor[idx][0].item(),
                        'y': candidates_tensor[idx][1].item(),
                        'z': metadata['mid_point_3d'][2]
                    })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ GPU内存不足，减小batch_size从{batch_size}到{batch_size//2}")
                batch_size = batch_size // 2
                if batch_size < 50:
                    print("⚠️ batch_size过小，回退到CPU版本")
                    return generate_oob_points_cpu(quads, grid, quads_by_id, nudge_distance)
                # 重新处理当前批次
                batch_start -= batch_size
                continue
            else:
                raise e
        
        # 清理GPU内存
        if 'candidates_tensor' in locals():
            del candidates_tensor
        if 'quads_tensor' in locals():
            del quads_tensor
        if 'is_inside_matrix' in locals():
            del is_inside_matrix, is_inside_any
        torch.cuda.empty_cache()
        
        # 更新进度
        processed = batch_end
        print(f"\rOOB processing: {processed}/{total_candidates} ({processed/total_candidates*100:.1f}%)", end="")
    
    print(f"\nGenerated {len(valid_oob_points)} OOB points using memory-friendly GPU processing.")
    return valid_oob_points

def generate_oob_points(quads, grid, quads_by_id, use_gpu=True, nudge_distance=OOB_NUDGE_DISTANCE):
    """OOB点生成主函数，支持GPU加速"""
    if use_gpu and torch.cuda.is_available():
        return generate_oob_points_gpu(quads, grid, quads_by_id, nudge_distance=nudge_distance)
    else:
        return generate_oob_points_cpu(quads, grid, quads_by_id, nudge_distance)

def _compute_road_bboxes(road_to_quads):
    bboxes = {}
    for rid, quads in road_to_quads.items():
        if not quads:
            bboxes[rid] = (0, 0, 0, 0)
            continue
        xs = []
        ys = []
        for q in quads:
            for v in q['vertices']:
                xs.append(v[0])
                ys.append(v[1])
        bboxes[rid] = (min(xs), min(ys), max(xs), max(ys))
    return bboxes

def _boxes_overlap(a, b, pad=0.0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 - pad or bx2 < ax1 - pad or ay2 < by1 - pad or by2 < ay1 - pad)

def group_roads_by_overlap(polygons_data, threshold=0.8, device=DEVICE):
    # 聚合每条路的quads
    road_to_quads = {}
    for q in polygons_data:
        road_to_quads.setdefault(q['road_id'], []).append(q)
    road_ids = sorted(road_to_quads.keys())
    if len(road_ids) <= 1:
        return [[rid] for rid in road_ids]

    # 预计算每条路的quads顶点数组 (M,4,2)
    road_to_vertices = {}
    for rid, quads in road_to_quads.items():
        verts = []
        for q in quads:
            verts.append([[v[0], v[1]] for v in q['vertices']])
        road_to_vertices[rid] = np.array(verts, dtype=np.float32) if len(verts) > 0 else np.zeros((0,4,2), dtype=np.float32)

    # AABB 粗筛
    bboxes = _compute_road_bboxes(road_to_quads)

    # 建图（邻接：同组）
    rid_index = {rid: i for i, rid in enumerate(road_ids)}
    n = len(road_ids)
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = True

    for i in range(n):
        rid_i = road_ids[i]
        verts_i = road_to_vertices[rid_i]
        if verts_i.shape[0] == 0:
            continue
        for j in range(i+1, n):
            rid_j = road_ids[j]
            if not _boxes_overlap(bboxes[rid_i], bboxes[rid_j], pad=0.0):
                continue
            verts_j = road_to_vertices[rid_j]
            if verts_j.shape[0] == 0:
                continue
            # GPU 相交矩阵
            try:
                inter_mat = quads_intersection_matrix_gpu(verts_i, verts_j, device=device)
            except Exception:
                # 回退到CPU
                inter_mat = quads_intersection_matrix_gpu(verts_i, verts_j, device='cpu')
            # 计算相交比例
            # A视角：A每个quad是否与B任意quad相交
            if inter_mat.size == 0:
                continue
            hit_a = (inter_mat.any(axis=1)).astype(np.float32)
            hit_b = (inter_mat.any(axis=0)).astype(np.float32)
            ratio_a = float(hit_a.mean()) if hit_a.size > 0 else 0.0
            ratio_b = float(hit_b.mean()) if hit_b.size > 0 else 0.0
            if max(ratio_a, ratio_b) >= float(threshold):
                adj[i][j] = True
                adj[j][i] = True

    # BFS 连通分量
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        comp = [road_ids[i]]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and adj[u][v]:
                    visited[v] = True
                    queue.append(v)
                    comp.append(road_ids[v])
        groups.append(sorted(comp))
    return groups

def reassign_road_lane_ids(polygons_data, groups, w_lanes=None):
    # 组内第k条路 -> lane_id = k+1；组的road_id取该组最小旧rid
    rid_to_groupmin = {}
    rid_to_lane = {}
    for group in groups:
        base_rid = min(group)
        for idx, rid in enumerate(sorted(group)):
            rid_to_groupmin[rid] = base_rid
            rid_to_lane[rid] = idx + 1

    # 更新polygons_data
    for q in polygons_data:
        old = q['road_id']
        if old in rid_to_groupmin:
            q['road_id'] = rid_to_groupmin[old]
            q['lane_id'] = rid_to_lane[old]
        else:
            q['lane_id'] = 1
    
    # 同步更新 w_lanes 的 road_id 和 lane_id，确保与 quads 一致
    if w_lanes is not None:
        # 构建 poly_id -> quad 的映射
        poly_to_quad = {q['poly_id']: q for q in polygons_data}
        updated_count = 0
        for w in w_lanes:
            poly_id = w.get('poly_id')
            if poly_id in poly_to_quad:
                # 从对应的 quad 读取最新的 road_id 和 lane_id
                quad = poly_to_quad[poly_id]
                old_rid, old_lid = w['road_id'], w['lane_id']
                w['road_id'] = quad['road_id']
                w['lane_id'] = quad['lane_id']
                if old_rid != quad['road_id'] or old_lid != quad['lane_id']:
                    updated_count += 1
        print(f"  已同步更新 {updated_count} 个 w_lane 的 (road_id, lane_id) 以匹配对应的 quad")
    
    # 返回 polygons 以及两个映射，供几何基元层回填
    return polygons_data, rid_to_groupmin, rid_to_lane

def apply_mapping_to_geometry(lines_data, circles_data, arcs_data, rid_to_groupmin, rid_to_lane):
    """将 (旧 road_id) -> (新 road_id, lane_id) 的映射回填给几何基元层。"""
    for item in lines_data:
        old = item.get('road_id')
        if old in rid_to_groupmin:
            item['road_id'] = rid_to_groupmin[old]
            item['lane_id'] = rid_to_lane[old]
        else:
            item['lane_id'] = item.get('lane_id', 1)
    
    for item in circles_data:
        old = item.get('road_id')
        if old in rid_to_groupmin:
            item['road_id'] = rid_to_groupmin[old]
            item['lane_id'] = rid_to_lane[old]
        else:
            item['lane_id'] = item.get('lane_id', 1)
    
    for item in arcs_data:
        old = item.get('road_id')
        if old in rid_to_groupmin:
            item['road_id'] = rid_to_groupmin[old]
            item['lane_id'] = rid_to_lane[old]
        else:
            item['lane_id'] = item.get('lane_id', 1)

def attach_geometry_end_poly_ids(lines_data, circles_data, arcs_data, polygons_data):
    """为每个几何元素添加 start_poly_id / end_poly_id。
    根据 geometry 的 start/end 坐标，在相同 (road_id, lane_id) 的 polygons 中
    找到距离最近的 poly，将其 poly_id 作为 start_poly_id 和 end_poly_id。
    """
    # 构建 (road_id, lane_id) -> quads 列表，便于按坐标查找
    groups = {}
    # 同时构建 road_id -> quads 列表，作为回退
    road_groups = {}
    for q in polygons_data:
        rid = q.get('road_id')
        lid = q.get('lane_id', 1)
        groups.setdefault((rid, lid), []).append(q)
        road_groups.setdefault(rid, []).append(q)
    
    def find_nearest_poly_id(target_point_2d, candidate_quads):
        """在候选 quads 中找到 center 距离目标点最近的 poly_id"""
        if not candidate_quads:
            return None
        min_dist = float('inf')
        nearest_pid = None
        tx, ty = float(target_point_2d[0]), float(target_point_2d[1])
        for q in candidate_quads:
            cx, cy = float(q['center'][0]), float(q['center'][1])
            dist = math.hypot(tx - cx, ty - cy)
            if dist < min_dist:
                min_dist = dist
                nearest_pid = int(q['poly_id'])
        return nearest_pid
    
    # 处理 lines: 使用 start 和 end 坐标
    for item in lines_data:
        rid = item.get('road_id')
        lid = item.get('lane_id', 1)
        candidate_quads = groups.get((rid, lid), [])
        # 回退：如果按 (road_id, lane_id) 找不到，则按 road_id 查找
        if not candidate_quads:
            candidate_quads = road_groups.get(rid, [])
        
        start_point = item.get('start', [])
        end_point = item.get('end', [])
        
        if len(start_point) >= 2 and len(end_point) >= 2:
            item['start_poly_id'] = find_nearest_poly_id(start_point[:2], candidate_quads)
            item['end_poly_id'] = find_nearest_poly_id(end_point[:2], candidate_quads)
        else:
            item['start_poly_id'] = None
            item['end_poly_id'] = None
    
    # 处理 arcs: 计算起点和终点坐标
    for item in arcs_data:
        rid = item.get('road_id')
        lid = item.get('lane_id', 1)
        candidate_quads = groups.get((rid, lid), [])
        # 回退：如果按 (road_id, lane_id) 找不到，则按 road_id 查找
        if not candidate_quads:
            candidate_quads = road_groups.get(rid, [])
        
        center = item.get('center', [])
        radius = float(item.get('radius', 0))
        start_angle = float(item.get('start_angle', 0))
        end_angle = float(item.get('end_angle', 0))
        
        if len(center) >= 2 and radius > 0:
            # 计算圆弧起点和终点坐标
            cx, cy = float(center[0]), float(center[1])
            start_angle_rad = math.radians(start_angle)
            end_angle_rad = math.radians(end_angle)
            
            start_point = (cx + radius * math.cos(start_angle_rad), 
                          cy + radius * math.sin(start_angle_rad))
            end_point = (cx + radius * math.cos(end_angle_rad), 
                        cy + radius * math.sin(end_angle_rad))
            
            item['start_poly_id'] = find_nearest_poly_id(start_point, candidate_quads)
            item['end_poly_id'] = find_nearest_poly_id(end_point, candidate_quads)
        else:
            item['start_poly_id'] = None
            item['end_poly_id'] = None
    
    # 处理 circles: 圆形是闭合的，可能需要特殊处理
    # 暂时使用圆心作为参考点，但可能需要根据实际需求调整
    for item in circles_data:
        rid = item.get('road_id')
        lid = item.get('lane_id', 1)
        candidate_quads = groups.get((rid, lid), [])
        # 回退：如果按 (road_id, lane_id) 找不到，则按 road_id 查找
        if not candidate_quads:
            candidate_quads = road_groups.get(rid, [])
        
        center = item.get('center', [])
        if len(center) >= 2:
            # 对于圆形，可以找一个参考点，或者使用第一个和最后一个 poly
            # 这里先使用圆心作为参考，但可能需要根据实际需求调整
            center_point = center[:2]
            nearest_pid = find_nearest_poly_id(center_point, candidate_quads)
            item['start_poly_id'] = nearest_pid
            item['end_poly_id'] = nearest_pid  # 圆形闭合，起点和终点相同
        else:
            item['start_poly_id'] = None
            item['end_poly_id'] = None

# =========================== W_lane 采样 ===========================
# TODO: 添加W_lane的采样。
# 对于每一个road_id,lane_id对应的路，我们需要采样W_lane,每个路的开头点，结尾点一定是W_lane,在这一段路中间按照W_LANE_SAMPLE_DISTANCE=40m间隔采样W_lane
# W_lane是某个quad的center，所以可以继承部分信息：
# 0. W_lane_id
# 1. road_id
# 2. lane_id
# 3. center
# 4. direction_angle
# 5. 当前车道宽度(得算一下梯形的前面两个点和后面两个点距离的宽度平均值作为宽度)
# (以下是每次初始化的时候填入的。)
# 6. 对于下一个goal的相对归一化距离（跟别的跟坐标有关系的一起做归一化）和绝对归一化距离(需要goal出来之后迅速计算) 关于所有的goal的距离，不过每次放出来观测的是当前的下一个goal是哪一个goal的信息
def _compute_quad_width_avg(quad_vertices):
    # 顶点顺序为: [0,1,2,3]，上底(0-1)、下底(2-3)
    x0, y0 = quad_vertices[0][0], quad_vertices[0][1]
    x1, y1 = quad_vertices[1][0], quad_vertices[1][1]
    x2, y2 = quad_vertices[2][0], quad_vertices[2][1]
    x3, y3 = quad_vertices[3][0], quad_vertices[3][1]
    top_len = math.hypot(x1 - x0, y1 - y0)
    bottom_len = math.hypot(x3 - x2, y3 - y2)
    return 0.5 * (top_len + bottom_len)

def _group_quads_by_road_lane(polygons_data):
    groups = {}
    for q in polygons_data:
        rid = q.get('road_id')
        lid = q.get('lane_id', 1)
        groups.setdefault((rid, lid), []).append(q)
    # 维持生成顺序：按 poly_id 升序
    for k in groups.keys():
        groups[k].sort(key=lambda it: it['poly_id'])
    return groups

def sample_w_lanes(polygons_data, sample_distance_m=W_LANE_SAMPLE_DISTANCE):
    groups = _group_quads_by_road_lane(polygons_data)
    w_lanes = []
    w_lane_id = 1
    for (rid, lid), quads in groups.items():
        if len(quads) == 0:
            continue
        # 按quad中心构建折线长度
        centers = [tuple(q['center']) for q in quads]
        cum = [0.0]
        for i in range(1, len(centers)):
            prev = centers[i-1]
            cur = centers[i]
            cum.append(cum[-1] + math.hypot(cur[0]-prev[0], cur[1]-prev[1]))
        total_len = cum[-1] if cum else 0.0

        # 目标采样位置: 0, step, 2*step, ..., total_len
        targets = []
        t = 0.0
        if total_len <= 1e-6:
            targets = [0.0]
        else:
            while t < total_len - 1e-6:
                targets.append(t)
                t += float(sample_distance_m)
            targets.append(total_len)

        # 为每个目标找到最近的索引
        idx_used = set()
        for T in targets:
            # 二分或线性搜索最近
            # 简洁起见线性搜索（len通常不大）
            best_i = 0
            best_d = float('inf')
            for i, c in enumerate(cum):
                d = abs(c - T)
                if d < best_d:
                    best_d = d
                    best_i = i
            idx_used.add(best_i)
        
        # 将采样索引排序，并分配w_lane_id
        sorted_idx_list = sorted(idx_used)
        idx_to_w_lane_id = {}  # 记录每个采样索引对应的w_lane_id
        for idx in sorted_idx_list:
            idx_to_w_lane_id[idx] = w_lane_id
            w_lane_id += 1

        # 生成W_lane点
        for idx in sorted_idx_list:
            q = quads[idx]
            width = _compute_quad_width_avg(q['vertices'])
            current_w_lane_id = idx_to_w_lane_id[idx]
            w_lanes.append({
                'w_lane_id': current_w_lane_id,
                'road_id': rid,
                'lane_id': lid,
                'center': (float(q['center'][0]), float(q['center'][1])),
                'direction_angle': float(q.get('direction_angle', 0.0)),
                'width': float(width),
                'poly_id': q['poly_id']
            })
        
        # 为所有quads添加next_w_lane_id和prev_w_lane_id
        # prev_w_lane_id: 从当前quad往前到起点的所有w_lane_id列表
        # next_w_lane_id: 从当前quad往后到终点的所有w_lane_id列表
        for quad_idx, quad in enumerate(quads):
            if len(sorted_idx_list) == 0:
                # 没有w_lane，设置为空列表
                quad['prev_w_lane_id'] = []
                quad['next_w_lane_id'] = []
            elif quad_idx in idx_to_w_lane_id:
                # 这个quad是w_lane
                pos = sorted_idx_list.index(quad_idx)
                # prev_w_lane_id: 从起点到当前w_lane（包含当前）的所有w_lane ID
                prev_list = [idx_to_w_lane_id[sorted_idx_list[i]] for i in range(pos + 1)]
                # next_w_lane_id: 从当前w_lane（包含当前）到终点的所有w_lane ID
                next_list = [idx_to_w_lane_id[sorted_idx_list[i]] for i in range(pos, len(sorted_idx_list))]
                quad['prev_w_lane_id'] = prev_list
                quad['next_w_lane_id'] = next_list
            else:
                # 这个quad不是w_lane，在两个w_lane之间
                # 找到前面最近的w_lane（小于等于quad_idx的最大索引）
                prev_w_idx = None
                for i in range(len(sorted_idx_list) - 1, -1, -1):
                    w_idx = sorted_idx_list[i]
                    if w_idx <= quad_idx:
                        prev_w_idx = i
                        break
                
                # 找到后面最近的w_lane（大于quad_idx的最小索引）
                next_w_idx = None
                for i in range(len(sorted_idx_list)):
                    w_idx = sorted_idx_list[i]
                    if w_idx > quad_idx:
                        next_w_idx = i
                        break
                
                # prev_w_lane_id: 从起点到前面最近w_lane（包含该w_lane）的所有w_lane ID
                if prev_w_idx is not None:
                    prev_list = [idx_to_w_lane_id[sorted_idx_list[i]] for i in range(prev_w_idx + 1)]
                else:
                    # 前面没有w_lane，使用第一个w_lane
                    prev_list = [idx_to_w_lane_id[sorted_idx_list[0]]]
                quad['prev_w_lane_id'] = prev_list
                
                # next_w_lane_id: 从后面最近w_lane（包含该w_lane）到终点的所有w_lane ID
                if next_w_idx is not None:
                    next_list = [idx_to_w_lane_id[sorted_idx_list[i]] for i in range(next_w_idx, len(sorted_idx_list))]
                else:
                    # 后面没有w_lane，使用最后一个w_lane
                    next_list = [idx_to_w_lane_id[sorted_idx_list[-1]]]
                quad['next_w_lane_id'] = next_list
    
    # 确保所有quads都有prev_w_lane_id和next_w_lane_id字段（防止某些quads没有被处理）
    for quad in polygons_data:
        if 'prev_w_lane_id' not in quad:
            quad['prev_w_lane_id'] = []
        if 'next_w_lane_id' not in quad:
            quad['next_w_lane_id'] = []
    
    return w_lanes

# =========================== 数据初始化 ===========================
# 处理直路
for line in msp.query("LINE"):
    start = line.dxf.start.xyz
    end   = line.dxf.end.xyz
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 检查是否与已有线条重合
    line_item = {
        'center': ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
        'start': start,
        'end': end
    }
    
    if not check_duplicate_geometry(line_item, lines_data):
        # 存储线条数据
        lines_data.append({
            'road_id': road_id_counter,
            'layer': line.dxf.layer,
            'start': start,
            'end': end,
            'length': length,
            'center': ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        })
        increment_road_id()
# 处理环岛
for c in msp.query("CIRCLE"):
    center = c.dxf.center.xyz
    radius = c.dxf.radius

    # 圆心坐标精度规整：统一使用6位小数精度
    center_x = round(center[0], 6)
    center_y = round(center[1], 6)
    center_z = round(center[2], 6)
    center = (center_x, center_y, center_z)

    # 检查是否与已有圆形重合
    circle_item = {
        'center': (center_x, center_y),
        'radius': radius
    }
    
    if not check_duplicate_geometry(circle_item, circles_data):
        # 存储圆形数据
        circles_data.append({
            'road_id': road_id_counter,
            'center': center,
            'radius': radius
        })
        increment_road_id()   
# 处理弯道
for a in msp.query("ARC"):
    center = a.dxf.center.xyz
    radius = a.dxf.radius
    # 圆心坐标精度规整：统一使用6位小数精度
    center_x = round(center[0], 6)
    center_y = round(center[1], 6)
    center_z = round(center[2], 6)
    center = (center_x, center_y, center_z)
    # DXF 的 ARC.start_angle / end_angle 单位为度
    start_angle_deg_raw = a.dxf.start_angle
    end_angle_deg_raw = a.dxf.end_angle
    # 归一化到 [0, 360)
    start_angle = normalize_angle_degrees(start_angle_deg_raw)
    end_angle = normalize_angle_degrees(end_angle_deg_raw)
    # 检查是否与已有圆弧重合
    arc_item = {
        'center': (center_x, center_y),
        'radius': radius,
        'start_angle': start_angle,
        'end_angle': end_angle
    }
    
    if not check_duplicate_geometry(arc_item, arcs_data):
        # 转为弧度用于三角函数计算
        start_angle_rad = math.radians(start_angle)
        end_angle_rad = math.radians(end_angle)
        
        # 存储圆弧数据
        arcs_data.append({
            'road_id': road_id_counter,
            'center': center,
            'radius': radius,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'direction': 1
        })
        increment_road_id()

# =========================== 生成无缝路面（无方向） ===========================
# 为所有道路生成采样点和四边形
# 处理直线道路的采样
for line_data in lines_data:
    road_id = line_data['road_id']
    start = line_data['start']
    end = line_data['end']
    
    # 生成采样点（使用GPU加速）
    if USE_GPU:
        points_array, total_length = generate_line_points_gpu(start, end, SAMPLE_DISTANCE, DEVICE)
        points = [(p[0], p[1]) for p in points_array]
        # GPU加速计算方向角度
        angles = compute_directions_gpu(points_array, DEVICE)
    else:
        points, total_length = sample_line_points(start, end, SAMPLE_DISTANCE)
        # CPU版本计算方向角度
        angles = []
        for i, (x, y) in enumerate(points):
            if i < len(points) - 1:
                next_x, next_y = points[i + 1]
                direction_angle = math.atan2(next_y - y, next_x - x)
            else:
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
            angles.append(direction_angle)
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else None
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        # 计算梯形参数
        if next_angle is not None:
            # 根据下一个点的方向计算上底宽度
            angle_diff = abs(angle_difference(current_angle, next_angle))
            top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        else:
            # 最后一个点：使用与前一个点相同的上底宽度，确保无缝连接
            if i > 0:
                # 计算与前一个点的角度差
                prev_angle = angles[i - 1]
                angle_diff = abs(angle_difference(prev_angle, current_angle))
                top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)
            else:
                top_width = RECTANGLE_LENGTH
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': [(vx, vy, 0.0) for (vx, vy) in vertices],
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1
# 处理圆形道路的采样
for circle_data in circles_data:
    road_id = circle_data['road_id']
    center = circle_data['center']
    radius = circle_data['radius']
    
    # 生成采样点（使用GPU加速）
    if USE_GPU:
        points_array, total_length = generate_circle_points_gpu(center, radius, SAMPLE_DISTANCE, DEVICE)
        points = [(p[0], p[1]) for p in points_array]
    else:
        points, total_length = sample_circle_points(center, radius, SAMPLE_DISTANCE)
    
    
    # 计算所有点的方向角度
    angles = []
    for i, (x, y) in enumerate(points):
        # 计算切线方向（圆形切线方向）
        # 从圆心到采样点的向量，切线方向垂直于此向量
        direction_angle = math.atan2(y - center[1], x - center[0]) + math.pi / 2
        angles.append(direction_angle)
    
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else angles[0]  # 圆形道路，第一个点连接最后一个点
        
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        
        # 计算梯形参数
        angle_diff = abs(angle_difference(current_angle, next_angle))
        top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': [(vx, vy, 0.0) for (vx, vy) in vertices],
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1
# 处理圆弧道路的采样
for arc_data in arcs_data:
    road_id = arc_data['road_id']
    center = arc_data['center']
    radius = arc_data['radius']
    start_angle = arc_data['start_angle']
    end_angle = arc_data['end_angle']
    # 转换为弧度
    start_angle_rad = math.radians(start_angle)
    end_angle_rad = math.radians(end_angle)
    
    # 生成采样点（使用GPU加速）
    if USE_GPU:
        points_array, total_length = generate_arc_points_gpu(center, radius, start_angle_rad, end_angle_rad, SAMPLE_DISTANCE, DEVICE)
        points = [(p[0], p[1]) for p in points_array]
    else:
        points, total_length = sample_arc_points(center, radius, start_angle_rad, end_angle_rad, SAMPLE_DISTANCE)
    
    
    # 计算所有点的方向角度
    angles = []
    for i, (x, y) in enumerate(points):
        # 计算切线方向（圆弧切线方向）
        # 从圆心到采样点的向量，切线方向垂直于此向量
        direction_angle = math.atan2(y - center[1], x - center[0]) + math.pi / 2
        angles.append(direction_angle)
    
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else None
        
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        
        # 计算梯形参数
        if next_angle is not None:
            # 根据下一个点的方向计算上底宽度
            angle_diff = abs(angle_difference(current_angle, next_angle))
            top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        else:
            # 最后一个点：使用与前一个点相同的上底宽度，确保无缝连接
            if i > 0:
                # 计算与前一个点的角度差
                prev_angle = angles[i - 1]
                angle_diff = abs(angle_difference(prev_angle, current_angle))
                top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)
            else:
                top_width = RECTANGLE_LENGTH
        
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': [(vx, vy, 0.0) for (vx, vy) in vertices],
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1

# =========================== OOB点生成 ===========================
print("\n=== 开始生成OOB点 ===")
# 构建四边形字典，用于快速查找
quads_by_id = {quad['poly_id']: quad for quad in polygons_data}
# 创建空间哈希并适配为 get_candidates 接口
idx_to_poly_id = [q['poly_id'] for q in polygons_data]
if polygons_data:
    verts_all = np.array([[v[0], v[1]] for q in polygons_data for v in q['vertices']], dtype=np.float32)
    min_bounds = torch.tensor(verts_all.min(axis=0), dtype=torch.float32, device=DEVICE)
    max_bounds = torch.tensor(verts_all.max(axis=0), dtype=torch.float32, device=DEVICE)
    quad_mins = []
    quad_maxs = []
    for q in polygons_data:
        v2 = np.array([[v[0], v[1]] for v in q['vertices']], dtype=np.float32)
        quad_mins.append(v2.min(axis=0))
        quad_maxs.append(v2.max(axis=0))
    static_bounds = torch.tensor(np.stack([np.stack([quad_mins[i], quad_maxs[i]], axis=0) for i in range(len(polygons_data))], axis=0), dtype=torch.float32, device=DEVICE)
    spatial_hash = SpatialHash(cell_size=float(CELL_SIZE), min_bounds=min_bounds, max_bounds=max_bounds, device=torch.device(DEVICE))
    spatial_hash.build_static_index(static_bounds)
else:
    spatial_hash = None

class _HashAdapter2:
    def __init__(self, h, idx_to_pid):
        self.h = h
        self.idx_to_pid = idx_to_pid
    def get_candidates(self, point_2d):
        if self.h is None:
            return []
        pt = torch.tensor([[float(point_2d[0]), float(point_2d[1])]], dtype=torch.float32, device=self.h.device)
        pairs = self.h.query_points(pt)
        if pairs.numel() == 0:
            return []
        item_idx = pairs[:, 1].tolist()
        return [self.idx_to_pid[i] for i in item_idx if 0 <= i < len(self.idx_to_pid)]

grid = _HashAdapter2(spatial_hash, idx_to_poly_id)

# 生成OOB点（支持GPU加速）
use_gpu = torch.cuda.is_available()
print(f"GPU加速状态: {'启用' if use_gpu else '禁用'}")
oob_points = generate_oob_points(polygons_data, grid, quads_by_id, use_gpu=use_gpu)

# =========================== 路网方向校正 ===========================
print("\n=== 开始路网方向校正 ===")
adjust_road_directions_gpu(lines_data, arcs_data, tolerance=TOLERANCE, device=DEVICE)

# =========================== W_lane 采样 ===========================
print("\n=== 开始W_lane采样 ===")
w_lanes = sample_w_lanes(polygons_data, sample_distance_m=W_LANE_SAMPLE_DISTANCE)
print(f"W_lane生成: {len(w_lanes)} 个")

# 方向校正后使用matplotlib可视化
print("\n=== 方向校正后使用matplotlib可视化 ===")
visualize_map(lines_data, circles_data, arcs_data, polygons_data, oob_points, ax, w_lanes=w_lanes)
print("\n=== 已绘制图形，关闭图窗后将继续导出JSON ===")
plt.show()

# =========================== 重叠分组与ID重排 ===========================
print("\n=== 基于GPU相交检测进行道路分组与ID/LANE重排 ===")
INTERSECTION_THRESHOLD = config['preprocessor'].get('intersection_threshold')
groups = group_roads_by_overlap(polygons_data, threshold=INTERSECTION_THRESHOLD, device=DEVICE)
print(f"分到 {len(groups)} 组: {groups}")
polygons_data, rid_to_groupmin, rid_to_lane = reassign_road_lane_ids(polygons_data, groups, w_lanes=w_lanes)
apply_mapping_to_geometry(lines_data, circles_data, arcs_data, rid_to_groupmin, rid_to_lane)
attach_geometry_end_poly_ids(lines_data, circles_data, arcs_data, polygons_data)

# TODO: 添加每条路quad的s值(frenet坐标系下的s值)
# 从每条路的start开始，计算每一个quad在这条路上的s值（frenet坐标，关于start点的曲线长度）
print("\n=== 计算每条车道的弧长参数 s（分组后） ===")
compute_lane_s_for_quads(polygons_data)

# TODO: 添加曲率计算 
# 对于每一条road_id,lane_id对应的路，从start到end有n个采样点，计算quads的曲率，并写入quads的curvature字段

print("\n=== 计算每条车道的曲率 curvature ===")
compute_lane_curvature(polygons_data)


# =========================== 导出地图JSON ===========================
# 使用 dxf 文件名生成同名 json 文件（同目录）
base_name = os.path.splitext(os.path.basename(dxf_path))[0]
json_file_name = f"{base_name}.json"
json_output_path = os.path.join(os.path.dirname(dxf_path), json_file_name)

# 仅导出所需字段：poly_Id, vertices, road_id
export_quads = []
for quad in polygons_data:
    export_quad = {
        "poly_id": quad["poly_id"],
        "road_id": quad["road_id"],
        'lane_id': quad['lane_id'],
        "center": [float(quad["center"][0]), float(quad["center"][1]), 0.0],
        "vertices": [[float(v[0]), float(v[1]), float(v[2])] for v in quad["vertices"]],
        "direction_angle": float(quad["direction_angle"]),
        "s": float(quad.get("s", 0.0)),
        "curvature": float(quad.get("curvature", 0.0))
    }
    # 添加prev_w_lane_id和next_w_lane_id（所有quads在sample_w_lanes中都已添加这些字段）
    # 现在是列表格式，包含从当前quad到起点/终点的所有w_lane_id
    prev_w_lane_id = quad.get("prev_w_lane_id", [])
    next_w_lane_id = quad.get("next_w_lane_id", [])
    if prev_w_lane_id and len(prev_w_lane_id) > 0:
        export_quad["prev_w_lane_id"] = [int(w_id) for w_id in prev_w_lane_id]
    if next_w_lane_id and len(next_w_lane_id) > 0:
        export_quad["next_w_lane_id"] = [int(w_id) for w_id in next_w_lane_id]
    export_quads.append(export_quad)

# 导出OOB点数据
export_oob_points = []
for oob_point in oob_points:
    export_oob_points.append({
        "x": float(oob_point['x']),
        "y": float(oob_point['y']),
        "z": float(oob_point['z'])
    })


export_payload = {
    "map_name": json_file_name,
    "quads": export_quads,
    "oob_points": export_oob_points,
    "w_lanes": [
        {
            "w_lane_id": item['w_lane_id'],
            "road_id": item['road_id'],
            "lane_id": item['lane_id'],
            "poly_id": item['poly_id'],
            "center": [item['center'][0], item['center'][1], 0.0],
            "direction_angle": item['direction_angle'],
            "width": item['width']
        }
        for item in w_lanes
    ],
    "geometry": {
        "lines": [
            {
                "road_id": item["road_id"],
                "lane_id": int(item.get("lane_id", 1)),
                "start_poly_id": item.get("start_poly_id"),
                "end_poly_id": item.get("end_poly_id"),
                "layer": str(item.get("layer", "")),
                "start": [float(item["start"][0]), float(item["start"][1]), float(item["start"][2])],
                "end": [float(item["end"][0]), float(item["end"][1]), float(item["end"][2])],
                "length": float(item.get("length", 0.0))
            }
            for item in lines_data
        ],
        "circles": [
            {
                "road_id": item["road_id"],
                "lane_id": int(item.get("lane_id", 1)),
                "start_poly_id": item.get("start_poly_id"),
                "end_poly_id": item.get("end_poly_id"),
                "center": [float(item["center"][0]), float(item["center"][1]), float(item["center"][2])],
                "radius": float(item["radius"]) 
            }
            for item in circles_data
        ],
        "arcs": [
            {
                "road_id": item["road_id"],
                "lane_id": int(item.get("lane_id", 1)),
                "start_poly_id": item.get("start_poly_id"),
                "end_poly_id": item.get("end_poly_id"),
                "center": [float(item["center"][0]), float(item["center"][1]), float(item["center"][2])],
                "radius": float(item["radius"]),
                "start_angle": float(item["start_angle"]),
                "end_angle": float(item["end_angle"]) 
            }
            for item in arcs_data
        ]
    }
}
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(export_payload, f, ensure_ascii=False, indent=2)
print(f"已导出地图JSON: {json_output_path}")

# =========================== 统一可视化（从JSON，使用matplotlib） ===========================
print("\n=== 开始可视化（从JSON，使用matplotlib） ===")
visualize_map_from_json(json_output_path)

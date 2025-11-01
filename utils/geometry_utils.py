"""
几何计算工具函数模块
包含可复用的几何计算函数，用于地图处理和GPU加速计算
"""
import numpy as np
import torch
from typing import Tuple

def calculate_distance(point1, point2):
    """计算两点之间的距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_point_in_quad_2d(point_2d, quad_vertices_2d):
    """检查2D点是否在四边形内部（CPU版本）"""
    p = np.array(point_2d)
    v = np.array(quad_vertices_2d)
    v_edges = np.roll(v, -1, axis=0) - v
    v_points = p - v
    cross_products = v_edges[:, 0] * v_points[:, 1] - v_edges[:, 1] * v_points[:, 0]
    return np.all(cross_products >= 0) or np.all(cross_products <= 0)

def is_points_in_quads_gpu(points, quads_vertices, device):
    """批量GPU版本的点在多边形内检测 - 完全向量化"""
    # points: (M, 2) - M个点
    # quads_vertices: (N, 4, 2) - N个四边形，每个4个顶点
    # 返回: (M, N) - 每个点是否在每个四边形内部
    
    M, _ = points.shape
    N, _, _ = quads_vertices.shape
    
    # 扩展维度进行批量计算
    points_expanded = points.unsqueeze(1).expand(-1, N, -1)  # (M, N, 2)
    quads_expanded = quads_vertices.unsqueeze(0).expand(M, -1, -1, -1)  # (M, N, 4, 2)
    
    # 计算边的向量
    v_edges = torch.roll(quads_expanded, -1, dims=2) - quads_expanded  # (M, N, 4, 2)
    
    # 计算点到顶点的向量
    v_points = points_expanded.unsqueeze(2) - quads_expanded  # (M, N, 4, 2)
    
    # 计算叉积
    cross_products = v_edges[:, :, :, 0] * v_points[:, :, :, 1] - v_edges[:, :, :, 1] * v_points[:, :, :, 0]  # (M, N, 4)
    
    # 检查所有叉积是否同号
    all_positive = torch.all(cross_products >= 0, dim=2)  # (M, N)
    all_negative = torch.all(cross_products <= 0, dim=2)  # (M, N)
    
    return all_positive | all_negative  # (M, N)

def quads_intersection_matrix_gpu(quads_a, quads_b, device='cuda', batch_size=512, eps=1e-7):
    """
    使用GPU批量判断梯形/四边形是否相交。

    参数:
    - quads_a: numpy.ndarray 或 torch.Tensor，形状 (M, 4, 2)，A集合四边形顶点（按顺序）
    - quads_b: numpy.ndarray 或 torch.Tensor，形状 (N, 4, 2)，B集合四边形顶点（按顺序）
    - device: 'cuda' 或 'cpu'，默认'cuda'
    - batch_size: 分批处理的批大小，避免显存峰值
    - eps: 浮点容差
    返回:
    - intersect_matrix: numpy.ndarray，形状 (M, N)，布尔矩阵，True表示对应两四边形相交
    说明:
    - 判定规则为：任一一方顶点落入另一方内部 或 任一边对发生相交。
    - 为兼顾性能与显存占用，对 A 做分批，B 全量常驻。
    """
    # 转为 torch.Tensor(float32)
    def _to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    quads_a_t = _to_tensor(quads_a)  # (M,4,2)
    quads_b_t = _to_tensor(quads_b)  # (N,4,2)

    M = quads_a_t.shape[0]
    N = quads_b_t.shape[0]

    if M == 0 or N == 0:
        return (torch.zeros((M, N), dtype=torch.bool)).cpu().numpy()

    # 预分配结果
    result = torch.zeros((M, N), dtype=torch.bool, device=device)

    # 预备 B 的边端点 (N,4,2)
    b_e0 = quads_b_t
    b_e1 = torch.roll(quads_b_t, shifts=-1, dims=1)

    # 向量化 2D 叉积
    def cross2d(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    # 线段相交判定，返回 (Ba, Nb) 布尔矩阵
    def segments_intersect_matrix(p1, p2, q1, q2):
        # 形状: p1/p2: (Ba,2), q1/q2: (Nb,2)
        # 广播到 (Ba, Nb, 2)
        p1b = p1[:, None, :]
        p2b = p2[:, None, :]
        q1b = q1[None, :, :]
        q2b = q2[None, :, :]

        r = p2b - p1b  # (Ba,Nb,2)
        s = q2b - q1b  # (Ba,Nb,2)

        rxs = cross2d(r, s)
        q_p = q1b - p1b
        q_pxr = cross2d(q_p, r)

        # 一般位置: rxs != 0
        rxs_nz = torch.abs(rxs) > eps
        t = cross2d(q_p, s) / (rxs + (~rxs_nz) * eps)
        u = q_pxr / (rxs + (~rxs_nz) * eps)

        inter_general = (rxs_nz & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps))

        # 共线: rxs == 0 且 q_pxr == 0 → 判断投影是否重叠
        colinear = (~rxs_nz) & (torch.abs(q_pxr) <= eps)
        if torch.any(colinear):
            # 在线段投影上判断重叠: 比较在 x/y 方向的投影重叠
            p_min = torch.minimum(p1b, p2b)
            p_max = torch.maximum(p1b, p2b)
            q_min = torch.minimum(q1b, q2b)
            q_max = torch.maximum(q1b, q2b)
            overlap = (torch.minimum(p_max[..., 0], q_max[..., 0]) >= torch.maximum(p_min[..., 0], q_min[..., 0]) - eps) & \
                      (torch.minimum(p_max[..., 1], q_max[..., 1]) >= torch.maximum(p_min[..., 1], q_min[..., 1]) - eps)
            inter_colinear = colinear & overlap
        else:
            inter_colinear = colinear

        return inter_general | inter_colinear

    # 分批处理 A
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        a_batch = quads_a_t[start:end]  # (Ba,4,2)
        Ba = a_batch.shape[0]

        # 顶点包含判定：A顶点在B内
        a_pts = a_batch.reshape(Ba * 4, 2)
        in_mat_apts_in_b = is_points_in_quads_gpu(a_pts, quads_b_t, device)  # (Ba*4, N)
        in_ab = in_mat_apts_in_b.reshape(Ba, 4, N).any(dim=1)  # (Ba, N)

        # B 顶点在 A 内
        b_pts = quads_b_t.reshape(N * 4, 2)
        in_mat_bpts_in_a = is_points_in_quads_gpu(b_pts, a_batch, device)  # (N*4, Ba)
        in_ba = in_mat_bpts_in_a.reshape(N, 4, Ba).any(dim=1).transpose(0, 1)  # (Ba, N)

        contains_inter = in_ab | in_ba  # (Ba, N)

        # 提前标记
        batch_result = contains_inter.clone()

        # 边相交判定，仅对尚未确定相交的位置做
        pending_mask = ~batch_result
        if torch.any(pending_mask):
            # A 边端点
            a_e0 = a_batch
            a_e1 = torch.roll(a_batch, shifts=-1, dims=1)

            edges_inter = torch.zeros((Ba, N), dtype=torch.bool, device=device)
            # 4x4 组合
            for i in range(4):
                p1 = a_e0[:, i, :]  # (Ba,2)
                p2 = a_e1[:, i, :]  # (Ba,2)
                for j in range(4):
                    q1 = b_e0[:, j, :]  # (N,2)
                    q2 = b_e1[:, j, :]  # (N,2)
                    inter_ij = segments_intersect_matrix(p1, p2, q1, q2)  # (Ba,N)
                    edges_inter |= inter_ij

            batch_result |= edges_inter

        result[start:end] = batch_result

    return result.cpu().numpy()

def normalize_angle(angle):
    """将角度标准化到[0, 2π]范围"""
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle

def angle_difference(angle1, angle2):
    """计算两个角度之间的最小差值"""
    diff = angle1 - angle2
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff

def normalize_angle_degrees(angle):
    """将角度标准化到[0, 360)范围（度数）"""
    return angle % 360

# ===要用hash加速的版本===
def find_nearest_lanes(device, quad_centerlines: torch.Tensor, points: torch.Tensor, k: int = 1, spatial_hash=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为一批输入点找到最近的 k 个车道 (quads)。
    Args:
        device (torch.device): 计算设备。
        quad_centerlines (torch.Tensor): 形状为 (Q, 2, 2) 的中心线张量。
        points (torch.Tensor): 形状为 (N, 2) 或 (2,) 的点坐标张量。
        k (int): 为每个点返回的最近车道数量。
        spatial_hash (SpatialHash, optional): 空间哈希对象。若提供，使用其候选；否则暴力搜索。
    Returns:
        distances: (N, k)，欧氏距离
        indices:   (N, k)，最近车道索引
    """
    if points.ndim == 1:
        points = points.unsqueeze(0)
    N = points.shape[0]

    if quad_centerlines.numel() == 0:
        return (
            torch.full((N, k), float('inf'), device=device),
            torch.full((N, k), -1, dtype=torch.long, device=device),
        )

    pts = points.to(device)
    centers = quad_centerlines.to(device).mean(dim=1)  # (Q,2)

    if spatial_hash is not None:
        candidate_pairs = spatial_hash.query_points(pts)  # (num_candidates, 2) -> (point_idx, quad_idx)
        if candidate_pairs.numel() == 0:
            distances = torch.full((N, k), float('inf'), device=device)
            indices = torch.full((N, k), -1, dtype=torch.long, device=device)
            return distances, indices

        point_indices = candidate_pairs[:, 0]
        quad_indices = candidate_pairs[:, 1]

        candidate_points = pts[point_indices]               # (C,2)
        candidate_quad_centers = centers[quad_indices]       # (C,2)
        diff = candidate_points - candidate_quad_centers     # (C,2)
        candidate_distances = torch.sum(diff * diff, dim=-1) # (C,)

        distances = torch.full((N, k), float('inf'), device=device)
        indices = torch.full((N, k), -1, dtype=torch.long, device=device)

        if point_indices.numel() > 0:
            point_counts = torch.bincount(point_indices, minlength=N)
            max_cand = point_counts.max().item()
            if max_cand > 0:
                order = torch.argsort(point_indices)
                p_sorted = point_indices[order]
                d_sorted = candidate_distances[order]
                q_sorted = quad_indices[order]

                starts = torch.cumsum(torch.nn.functional.pad(point_counts, (1, 0)), dim=0)[:-1]
                mat_d = torch.full((N, max_cand), float('inf'), device=device)
                mat_i = torch.full((N, max_cand), -1, dtype=torch.long, device=device)
                pos = torch.arange(len(order), device=device) - starts[p_sorted]
                mat_d[p_sorted, pos] = d_sorted
                mat_i[p_sorted, pos] = q_sorted

                topk_d, topk_idx = torch.topk(mat_d, k=min(k, max_cand), dim=1, largest=False)
                vk = min(k, max_cand)
                distances[:, :vk] = torch.sqrt(topk_d)
                indices[:, :vk] = torch.gather(mat_i, 1, topk_idx)
        return distances, indices

    # 暴力搜索
    diff = pts.unsqueeze(1) - centers.unsqueeze(0)  # (N,Q,2)
    dist_sq = torch.sum(diff * diff, dim=-1)        # (N,Q)
    k_eff = min(k, dist_sq.shape[1])
    distances, indices = torch.topk(dist_sq, k=k_eff, dim=1, largest=False)
    if k_eff < k:
        pad = k - k_eff
        distances = torch.cat([distances, torch.full((N, pad), float('inf'), device=device)], dim=1)
        indices = torch.cat([indices, torch.full((N, pad), -1, dtype=torch.long, device=device)], dim=1)
    return torch.sqrt(distances), indices

def calculate_frenet_coordinates(device: torch.device,
                                 quad_directions: torch.Tensor,
                                 quad_centerlines: torch.Tensor,
                                 vehicle_positions: torch.Tensor,
                                 vehicle_headings: torch.Tensor,
                                 k: int = 1,
                                 spatial_hash=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算车辆在Frenet坐标系中的横向距离 d 与航向角误差 theta_f（左正右负）。
    参数：
      - device: 计算设备
      - quad_directions: (Q,2) 单位方向向量
      - quad_centerlines: (Q,2,2)
      - vehicle_positions: (B,M,2) 或 (N,2) 或 (2,)
      - vehicle_headings:  (B,M)   或 (N,)  或 ()
      - k: 最近lane数量（默认1）
      - spatial_hash: 可选的空间哈希，用于加速最近邻查询
    返回：
      - d: (B,M)
      - theta_f: (B,M)
    """
    if vehicle_positions.ndim == 1:
        vehicle_positions = vehicle_positions.unsqueeze(0).unsqueeze(0)  # (2,) -> (1,1,2)
        vehicle_headings = vehicle_headings.unsqueeze(0).unsqueeze(0)
    elif vehicle_positions.ndim == 2:
        vehicle_positions = vehicle_positions.unsqueeze(0)  # (N,2) -> (1,N,2)
        if vehicle_headings.ndim == 1:
            vehicle_headings = vehicle_headings.unsqueeze(0)
    B, M, _ = vehicle_positions.shape

    pts_flat = vehicle_positions.reshape(-1, 2)
    distances, indices = find_nearest_lanes(device, quad_centerlines, pts_flat, k=max(1, k), spatial_hash=spatial_hash)
    nearest_idx = indices[:, 0].view(B, M)

    road_dirs = quad_directions.to(device)[nearest_idx]              # (B,M,2)
    veh_dirs = torch.stack([torch.cos(vehicle_headings),
                            torch.sin(vehicle_headings)], dim=-1).to(device)  # (B,M,2)
    centerlines = quad_centerlines.to(device)[nearest_idx]           # (B,M,2,2)
    # 根据quad方向与中心线方向一致性选择起点：
    # 若 (centerlines[1]-centerlines[0]) · road_dir >= 0，则起点为centerlines[0]；否则为centerlines[1]
    v_cl = centerlines[:, :, 1, :] - centerlines[:, :, 0, :]           # (B,M,2)
    dot = torch.sum(v_cl * road_dirs, dim=-1)                          # (B,M)
    use_first = (dot >= 0).unsqueeze(-1)                               # (B,M,1)
    road_starts = torch.where(use_first, centerlines[:, :, 0, :], centerlines[:, :, 1, :])
    AP = vehicle_positions.to(device) - road_starts                  # (B,M,2)

    d = (AP[:, :, 0] * road_dirs[:, :, 1] - AP[:, :, 1] * road_dirs[:, :, 0])
    cross_heading = (veh_dirs[:, :, 0] * road_dirs[:, :, 1] - veh_dirs[:, :, 1] * road_dirs[:, :, 0])
    dot_heading = torch.sum(road_dirs * veh_dirs, dim=-1)
    theta_f = torch.atan2(cross_heading, dot_heading)
    return d, theta_f

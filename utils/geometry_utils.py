"""
几何计算工具函数模块
包含可复用的几何计算函数，用于地图处理和GPU加速计算
"""

import numpy as np
import torch
from collections import defaultdict

class SpatialGrid3D:
    """3D空间网格，用于快速查找候选四边形"""
    def __init__(self, quads, cell_size, name="quads"):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        all_verts_2d = np.array([[v[0], v[1]] for q in quads for v in q['vertices']])
        if all_verts_2d.shape[0] == 0:
            self.min_coord, self.max_coord = np.array([0,0]), np.array([0,0])
        else:
            self.min_coord = np.min(all_verts_2d, axis=0) - cell_size
            self.max_coord = np.max(all_verts_2d, axis=0) + cell_size
        print(f"Populating 3D spatial grid for {name}...")
        for i, quad in enumerate(quads):
            if (i + 1) % 100 == 0 or i + 1 == len(quads):
                print(f"\rGrid processing for {name}: {i+1}/{len(quads)}", end="")
            poly_id = quad['poly_id']
            verts_2d = np.array([[v[0], v[1]] for v in quad['vertices']])
            min_q, max_q = np.min(verts_2d, axis=0), np.max(verts_2d, axis=0)
            for i_grid in range(int((min_q[0] - self.min_coord[0]) // self.cell_size), int((max_q[0] - self.min_coord[0]) // self.cell_size) + 1):
                for j_grid in range(int((min_q[1] - self.min_coord[1]) // self.cell_size), int((max_q[1] - self.min_coord[1]) // self.cell_size) + 1):
                    self.grid[(i_grid, j_grid)].append(poly_id)
        print(f"\n{name.capitalize()} grid populated.")

    def get_candidates(self, point_2d):
        grid_x = int((point_2d[0] - self.min_coord[0]) // self.cell_size)
        grid_y = int((point_2d[1] - self.min_coord[1]) // self.cell_size)
        return self.grid.get((grid_x, grid_y), [])

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


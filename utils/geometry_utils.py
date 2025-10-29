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


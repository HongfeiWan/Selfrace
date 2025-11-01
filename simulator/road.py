import torch
import math
import json
from typing import Dict, List

class RoadNetwork:
    """
    负责加载和管理从预处理的地图数据中提取的道路网络。
    这个类将地图数据（主要是四边形路块 'quads'）加载到 PyTorch 张量中，
    以便于在 GPU 上进行高效的批量化计算（如车道中心线、边界线、航点等）。
    """
    def __init__(self, map_path: str, device: torch.device):
        """
        初始化道路网络。
        Args:
            map_path (str): 指向预处理后的地图 JSON 文件的路径。
            device (torch.device): 用于存储地图数据的计算设备 ('cpu' 或 'cuda')。
        """
        self.device = device
        # 预初始化在后续流程中会被写入/依赖的成员，便于理解可用数据结构
        # 基本拓扑/几何（放在指定device上）
        self.quads_vertices: torch.Tensor = torch.empty((0, 4, 2), dtype=torch.float32, device=self.device)
        self.quad_centerlines: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.left_boundaries: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.right_boundaries: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.quad_directions: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        # 元数据/索引
        self.quad_ids: torch.Tensor = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.lane_ids: torch.Tensor = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.road_ids: torch.Tensor = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.w_lane_ids: List[List[int]] = []
        self.w_boundary_ids: List[List[int]] = []
        # 全局航点（w_lane与OOB）
        self.global_w_lane: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self.global_w_boundary: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        # 加载和处理地图数据
        map_data = self._load_map_data(map_path)
        self._store_metadata(map_data)

    def _load_map_data(self, map_path: str) -> Dict:
        """从 JSON 文件加载地图数据。"""
        try:
            with open(map_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise

    def _store_metadata(self, map_data: Dict):
        """构建几何、存储元数据与全局航点（合并版）"""
        quads_data = map_data['quads']
        # 顶点与中心线
        TL = torch.tensor([[q['vertices'][0][0], q['vertices'][0][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        TR = torch.tensor([[q['vertices'][1][0], q['vertices'][1][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        BR = torch.tensor([[q['vertices'][2][0], q['vertices'][2][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        BL = torch.tensor([[q['vertices'][3][0], q['vertices'][3][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        self.quads_vertices = torch.stack([TL, BL, BR, TR], dim=1)
        front_center = (BL + BR) / 2.0
        back_center = (TL + TR) / 2.0
        self.quad_centerlines = torch.stack([back_center, front_center], dim=1)
        # 元数据
        self.quad_ids = torch.tensor([q['poly_id'] for q in quads_data], dtype=torch.int64, device=self.device)
        self.lane_ids = torch.tensor([q['lane_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        self.road_ids = torch.tensor([q['road_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        self.w_lane_ids = [q.get('w_lane_ids', []) for q in quads_data]
        self.w_boundary_ids = [q.get('w_boundary_ids', []) for q in quads_data]
        # 方向
        angles = torch.tensor([float(q['direction_angle']) for q in quads_data], dtype=torch.float32, device=self.device)
        self.quad_directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        # 左右边界（基于方向）
        edgeA = torch.stack([TL, BL], dim=1)
        edgeB = torch.stack([TR, BR], dim=1)
        centers = self.quad_centerlines.mean(dim=1)
        dirs = self.quad_directions
        left_normal = torch.stack([-dirs[:, 1], dirs[:, 0]], dim=1)
        midA = (TL + BL) * 0.5 - centers
        midB = (TR + BR) * 0.5 - centers
        dotA = torch.sum(midA * left_normal, dim=1)
        dotB = torch.sum(midB * left_normal, dim=1)
        mask = (dotA >= dotB).view(-1, 1, 1)
        self.left_boundaries = torch.where(mask, edgeA, edgeB)
        self.right_boundaries = torch.where(mask, edgeB, edgeA)
        # 全局航点
        w_lane_points = map_data.get('w_lanes', [])
        self.global_w_lane = (
            torch.tensor([[p['center'][0], p['center'][1]] for p in w_lane_points], dtype=torch.float32, device=self.device)
            if w_lane_points else torch.empty((0, 2), device=self.device)
        )
        w_boundary_points = map_data.get('oob_points', [])
        self.global_w_boundary = torch.tensor([[p['x'], p['y']] for p in w_boundary_points], dtype=torch.float32, device=self.device) if w_boundary_points else torch.empty((0, 2), device=self.device)
    
if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.widgets import CheckButtons
    # 简单测试接口：加载JSON并可视化基础quads与可选层
    default_json = os.path.join(os.path.dirname(__file__), "..", "maps", "town2.json")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rn = RoadNetwork(default_json, device=device)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('RoadNetwork Layers')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    artists = {
        'centerlines': [],
        'left': [],
        'right': [],
        'dirs': [],
        'w_lane': [],
        'w_boundary': [],
    }
    rendered = {k: False for k in artists.keys()}

    # 底层：仅绘制无填充颜色的四边形轮廓
    if rn.quads_vertices.numel() > 0:
        quads_np = rn.quads_vertices.detach().cpu().numpy()  # (N,4,2)
        for verts in quads_np:
            poly = Polygon(verts, closed=True, facecolor='none', edgecolor='black', linewidth=0.2)
            ax.add_patch(poly)

    # 计算中心点（用于方向箭头）
    centers_np = rn.quad_centerlines.mean(dim=1).detach().cpu().numpy() if rn.quad_centerlines.numel() > 0 else np.empty((0, 2))

    def render_centerlines():
        if rendered['centerlines']:
            return
        if rn.quad_centerlines.numel() == 0:
            rendered['centerlines'] = True
            return
        cl = rn.quad_centerlines.detach().cpu().numpy()  # (N,2,2)
        for seg in cl:
            line, = ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], color='orange', linewidth=0.4)
            artists['centerlines'].append(line)
        rendered['centerlines'] = True

    def render_left():
        if rendered['left']:
            return
        if rn.left_boundaries.numel() == 0:
            rendered['left'] = True
            return
        lb = rn.left_boundaries.detach().cpu().numpy()
        for seg in lb:
            line, = ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], color='blue', linewidth=0.4)
            artists['left'].append(line)
        rendered['left'] = True

    def render_right():
        if rendered['right']:
            return
        if rn.right_boundaries.numel() == 0:
            rendered['right'] = True
            return
        rb = rn.right_boundaries.detach().cpu().numpy()
        for seg in rb:
            line, = ax.plot([seg[0,0], seg[1,0]], [seg[0,1], seg[1,1]], color='green', linewidth=0.4)
            artists['right'].append(line)
        rendered['right'] = True

    def render_dirs():
        if rendered['dirs']:
            return
        if rn.quad_directions.numel() == 0 or centers_np.shape[0] == 0:
            rendered['dirs'] = True
            return
        dirs = rn.quad_directions.detach().cpu().numpy()
        L = 0.6  # 箭头长度
        for (cx, cy), (dx, dy) in zip(centers_np, dirs):
            ex, ey = cx + L * dx, cy + L * dy
            arr = ax.annotate('', xy=(ex, ey), xytext=(cx, cy),
                              arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.7))
            artists['dirs'].append(arr)
        rendered['dirs'] = True

    def render_w_lane():
        if rendered['w_lane']:
            return
        pts = rn.global_w_lane.detach().cpu().numpy() if rn.global_w_lane.numel() > 0 else np.empty((0, 2))
        if pts.shape[0] == 0:
            rendered['w_lane'] = True
            return
        sc = ax.scatter(pts[:,0], pts[:,1], s=10, c='red', alpha=0.8)
        artists['w_lane'].append(sc)
        rendered['w_lane'] = True

    def render_w_boundary():
        if rendered['w_boundary']:
            return
        pts = rn.global_w_boundary.detach().cpu().numpy() if rn.global_w_boundary.numel() > 0 else np.empty((0, 2))
        if pts.shape[0] == 0:
            rendered['w_boundary'] = True
            return
        sc = ax.scatter(pts[:,0], pts[:,1], s=6, c='purple', alpha=0.6)
        artists['w_boundary'].append(sc)
        rendered['w_boundary'] = True

    # 复选框
    cb_ax = fig.add_axes([0.86, 0.6, 0.12, 0.2])
    labels = ['quad_centerlines', 'left_boundaries', 'right_boundaries', 'quad_directions', 'global_w_lane', 'global_w_boundary']
    visibility = [False] * len(labels)
    check = CheckButtons(cb_ax, labels, visibility)
    cb_ax.set_title('Layers')

    def on_clicked(label):
        if label == 'quad_centerlines':
            if not rendered['centerlines']:
                render_centerlines()
            else:
                for a in artists['centerlines']:
                    a.set_visible(not a.get_visible())
        elif label == 'left_boundaries':
            if not rendered['left']:
                render_left()
            else:
                for a in artists['left']:
                    a.set_visible(not a.get_visible())
        elif label == 'right_boundaries':
            if not rendered['right']:
                render_right()
            else:
                for a in artists['right']:
                    a.set_visible(not a.get_visible())
        elif label == 'quad_directions':
            if not rendered['dirs']:
                render_dirs()
            else:
                for a in artists['dirs']:
                    vis = a.get_visible() if hasattr(a, 'get_visible') else True
                    try:
                        a.set_visible(not vis)
                    except Exception:
                        pass
        elif label == 'global_w_lane':
            if not rendered['w_lane']:
                render_w_lane()
            else:
                for a in artists['w_lane']:
                    a.set_visible(not a.get_visible())
        elif label == 'global_w_boundary':
            if not rendered['w_boundary']:
                render_w_boundary()
            else:
                for a in artists['w_boundary']:
                    a.set_visible(not a.get_visible())
        ax.figure.canvas.draw_idle()

    check.on_clicked(on_clicked)
    fig._rn_layer_check = check
    fig._rn_artists = artists
    fig._rn_rendered = rendered

    ax.autoscale()
    fig.canvas.draw_idle()
    plt.show()



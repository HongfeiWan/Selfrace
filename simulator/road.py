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

        # ===== 预先初始化 _store_metadata 中会写入的成员 =====
        # 原始数据缓存
        self.quads_raw = []
        self.w_lanes_raw = []
        # 分组与查找表
        self.lane_groups = {}
        self.quads_by_id = {}
        self.lane_start_end = {}
        # 车道索引相关
        self.lane_keys: List = []
        self.n_lanes: int = 0
        self.lane_to_idx = {}
        self.w_lane_id_to_idx = {}
        # 起终点坐标
        self.start_positions: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self.end_positions: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        # w_lane 特征
        self.w_lane_features: torch.Tensor = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        # 图结构
        self.adjacency_matrix: torch.Tensor = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        self.edge_weights: torch.Tensor = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        # poly 映射
        self.poly_id_to_lane_idx = {}
        self.poly_id_lookup: torch.Tensor = torch.empty((0,), dtype=torch.long, device=self.device)

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
        """构建几何、存储元数据与全局航点（合并版），并预计算规划所需索引"""
        quads_data = map_data['quads']
        # 保留原始数据以便上层复用
        self.quads_raw = quads_data
        self.w_lanes_raw = map_data.get('w_lanes', [])
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
        w_lane_points = self.w_lanes_raw
        self.global_w_lane = (
            torch.tensor([[p['center'][0], p['center'][1]] for p in w_lane_points], dtype=torch.float32, device=self.device)
            if w_lane_points else torch.empty((0, 2), device=self.device)
        )
        # w_lane 特征 (x, y, direction_angle)
        if self.w_lanes_raw:
            self.w_lane_features = torch.tensor(
                [
                    [float(p['center'][0]), float(p['center'][1]), float(p.get('direction_angle'))]
                    for p in self.w_lanes_raw
                ],
                dtype=torch.float32,
                device=self.device,
            )  # (n_w_lanes, 3)
        w_boundary_points = map_data.get('oob_points', [])
        self.global_w_boundary = torch.tensor([[p['x'], p['y']] for p in w_boundary_points], dtype=torch.float32, device=self.device) if w_boundary_points else torch.empty((0, 2), device=self.device)

        # ===== 预计算供 PathPlanner 复用的数据结构 =====
        # 1) (road_id, lane_id) 分组与每组起终点 w_lane_id
        from collections import defaultdict
        lane_groups = defaultdict(list)
        for wl in self.w_lanes_raw:
            lane_groups[(wl['road_id'], wl['lane_id'])].append(wl)
        self.lane_groups = lane_groups

        # quad 查找表
        quads_by_id = {q['poly_id']: q for q in quads_data}
        self.quads_by_id = quads_by_id

        # 每条车道的 start/end w_lane_id
        lane_start_end_dict = {}
        for (road_id, lane_id), w_lanes_in_lane in lane_groups.items():
            w_lanes_with_s = []
            for wl in w_lanes_in_lane:
                quad = quads_by_id.get(wl['poly_id'], {})
                s = quad.get('s', 0.0)
                w_lanes_with_s.append((wl, s))
            w_lanes_with_s.sort(key=lambda x: x[1])
            if len(w_lanes_with_s) == 0:
                continue
            start_w_lane = w_lanes_with_s[0][0]
            end_w_lane = w_lanes_with_s[-1][0]
            lane_start_end_dict[(road_id, lane_id)] = {
                'start': start_w_lane['w_lane_id'],
                'end': end_w_lane['w_lane_id'],
            }
        self.lane_start_end = lane_start_end_dict

        # 2) 车道键与索引
        self.lane_keys = sorted(lane_start_end_dict.keys())
        self.n_lanes = len(self.lane_keys)
        self.lane_to_idx = {key: idx for idx, key in enumerate(self.lane_keys)}

        # 3) w_lane_id 到索引的映射（与 global_w_lane 行对应）
        self.w_lane_id_to_idx = {wl['w_lane_id']: i for i, wl in enumerate(self.w_lanes_raw)}

        # 4) 每条 lane 的 start/end 坐标（张量）
        if self.n_lanes > 0 and self.global_w_lane.numel() > 0:
            start_indices = [self.w_lane_id_to_idx[lane_start_end_dict[k]['start']] for k in self.lane_keys]
            end_indices = [self.w_lane_id_to_idx[lane_start_end_dict[k]['end']] for k in self.lane_keys]
            self.start_positions = self.global_w_lane[torch.tensor(start_indices, dtype=torch.long, device=self.device)]
            self.end_positions = self.global_w_lane[torch.tensor(end_indices, dtype=torch.long, device=self.device)]
        else:
            self.start_positions = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self.end_positions = torch.empty((0, 2), dtype=torch.float32, device=self.device)

        # 5) 邻接矩阵和边权重（基于 end->start 距离 < 阈值）
        if self.n_lanes > 0:
            # 构建联通图：计算每个 lane 的 end 到其他 lane 的 start 的距离
            # 形成单向导通图：lane_i的end -> lane_j的start（如果距离够近）
            # self.end_positions: (n_lanes, 2)
            # self.start_positions: (n_lanes, 2)
            # 计算距离矩阵：dist[i,j] = ||end[i] - start[j]||
            # 基于 RoadNetwork 的预计算，距离矩阵无需重复计算

            # 建立邻接矩阵和边权重矩阵
            # adjacency_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
            # 邻接与边权重使用 RoadNetwork 版本

            # 图结构已构建完成
            # adjacency_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
            # 由于每个lane内部 start->end 是天然导通的，因此：
            # 从 lane_i 的 end 可以到达 lane_j 的 start，
            # 那么就可以继续到达 lane_j 的 end
            end_positions_expanded = self.end_positions.unsqueeze(1)
            start_positions_expanded = self.start_positions.unsqueeze(0)
            distances = torch.norm(end_positions_expanded - start_positions_expanded, dim=2)
            CONNECTION_THRESHOLD = 5.0
            self.adjacency_matrix = (distances < CONNECTION_THRESHOLD).float()
            INF = 1e10
            self.edge_weights = torch.where(self.adjacency_matrix > 0, distances, torch.full_like(distances, INF))
        else:
            self.adjacency_matrix = torch.empty((0, 0), dtype=torch.float32, device=self.device)
            self.edge_weights = torch.empty((0, 0), dtype=torch.float32, device=self.device)

        # 6) poly_id -> lane_idx 映射与查找表
        poly_id_to_lane_idx = {}
        max_poly_id = 0
        for q in quads_data:
            poly_id = q['poly_id']
            max_poly_id = max(max_poly_id, poly_id)
            key = (q['road_id'], q['lane_id'])
            if key in self.lane_to_idx:
                poly_id_to_lane_idx[poly_id] = self.lane_to_idx[key]
        self.poly_id_to_lane_idx = poly_id_to_lane_idx
        lookup = torch.full((max_poly_id + 1,), -1, dtype=torch.long, device=self.device)
        for pid, lidx in poly_id_to_lane_idx.items():
            lookup[pid] = lidx
        self.poly_id_lookup = lookup
    
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



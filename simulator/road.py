import torch
import math
import json
from typing import Dict, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry_utils import find_nearest_lanes

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
        self.quad_centers: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self.quad_centerlines: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.left_boundaries: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.right_boundaries: torch.Tensor = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        self.quad_directions: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self.quad_curvatures: torch.Tensor = torch.empty((0,), dtype=torch.float32, device=self.device)
        # 元数据/索引
        self.quad_ids: torch.Tensor = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.lane_ids: torch.Tensor = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.road_ids: torch.Tensor = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.poly_id_to_center_idx: torch.Tensor = torch.empty((0,), dtype=torch.long, device=self.device)
        self.w_lane_ids: List[List[int]] = []
        self.w_boundary_ids: List[List[int]] = []
        # 全局航点（w_lane与OOB）
        self.global_w_lane: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        self.global_w_boundary: torch.Tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        # 便捷/同义字段与尺寸
        self.num_quads: int = 0

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
        # 观测所需的预计算映射（quad -> 最近航点ID）
        self.quad_to_w_lanes_ids: torch.Tensor = torch.empty((0, 0), dtype=torch.long, device=self.device)
        self.quad_to_w_boundaries_ids: torch.Tensor = torch.empty((0, 0), dtype=torch.long, device=self.device)
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
        
        # 直接从 JSON 加载 quad centers（preprocessor 已经计算好的准确值）
        self.quad_centers = torch.tensor(
            [[q['center'][0], q['center'][1]] for q in quads_data], 
            dtype=torch.float32, device=self.device
        )
        
        # 顶点与中心线
        TL = torch.tensor([[q['vertices'][0][0], q['vertices'][0][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        TR = torch.tensor([[q['vertices'][1][0], q['vertices'][1][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        BR = torch.tensor([[q['vertices'][2][0], q['vertices'][2][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        BL = torch.tensor([[q['vertices'][3][0], q['vertices'][3][1]] for q in quads_data], dtype=torch.float32, device=self.device)
        self.quads_vertices = torch.stack([TL, BL, BR, TR], dim=1)
        front_center = (BL + BR) / 2.0
        back_center = (TL + TR) / 2.0
        self.quad_centerlines = torch.stack([back_center, front_center], dim=1)
        # 便捷属性
        self.num_quads = int(self.quad_centerlines.shape[0])
        # 元数据
        self.quad_ids = torch.tensor([q['poly_id'] for q in quads_data], dtype=torch.int64, device=self.device)
        self.lane_ids = torch.tensor([q['lane_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        self.road_ids = torch.tensor([q['road_id'] for q in quads_data], dtype=torch.int32, device=self.device)
        
        # 创建 poly_id 到数组索引的查找表（用于快速索引 quad_centers）
        max_poly_id = int(self.quad_ids.max().item()) if self.quad_ids.numel() > 0 else 0
        self.poly_id_to_center_idx = torch.full((max_poly_id + 1,), -1, dtype=torch.long, device=self.device)
        for idx, poly_id in enumerate(self.quad_ids.tolist()):
            self.poly_id_to_center_idx[poly_id] = idx
        self.w_lane_ids = [q.get('w_lane_ids', []) for q in quads_data]
        self.w_boundary_ids = [q.get('w_boundary_ids', []) for q in quads_data]
        # 方向
        angles = torch.tensor([float(q['direction_angle']) for q in quads_data], dtype=torch.float32, device=self.device)
        self.quad_directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        # 曲率
        self.quad_curvatures = torch.tensor([float(q.get('curvature', 0.0)) for q in quads_data], dtype=torch.float32, device=self.device)
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

        # 7) 预计算 quad -> 最近 w_lanes / w_boundaries 的索引（供 Observation 直接使用）
        try:
            quad_centers = self.quad_centerlines.mean(dim=1)  # (num_quads, 2)
            # 最近 w_lanes（最多80个，避免大内存）
            if self.global_w_lane.numel() > 0 and self.num_quads > 0:
                k_lanes = min(80, self.global_w_lane.shape[0])
                d_l = torch.cdist(quad_centers, self.global_w_lane, p=2)
                _, nn_idx_l = torch.topk(d_l, k=k_lanes, dim=1, largest=False)
                self.quad_to_w_lanes_ids = nn_idx_l.to(dtype=torch.long)
            else:
                self.quad_to_w_lanes_ids = torch.zeros(self.num_quads, 0, dtype=torch.long, device=self.device)
            # 最近 w_boundaries（最多80个）
            if self.global_w_boundary.numel() > 0 and self.num_quads > 0:
                k_bd = min(80, self.global_w_boundary.shape[0])
                d_b = torch.cdist(quad_centers, self.global_w_boundary, p=2)
                _, nn_idx_b = torch.topk(d_b, k=k_bd, dim=1, largest=False)
                self.quad_to_w_boundaries_ids = nn_idx_b.to(dtype=torch.long)
            else:
                self.quad_to_w_boundaries_ids = torch.zeros(self.num_quads, 0, dtype=torch.long, device=self.device)
        except Exception:
            # 出错时给出空结构，避免阻塞流程
            self.quad_to_w_lanes_ids = torch.zeros(self.num_quads, 0, dtype=torch.long, device=self.device)
            self.quad_to_w_boundaries_ids = torch.zeros(self.num_quads, 0, dtype=torch.long, device=self.device)
    
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

    # --- 点击交互：查找最近quad并可视化其关联航点/边界 ---
    ui = {
        'click': None,
        'lane': None,
        'bound': None,
        'poly': None,
    }

    def on_click(event):
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        px, py = float(event.xdata), float(event.ydata)
        if ui['click'] is None:
            ui['click'] = ax.scatter([px], [py], c='black', s=20, marker='x', zorder=10, label='click')
        else:
            ui['click'].set_offsets([[px, py]])

        # 最近邻查询（无需哈希）
        pts = torch.tensor([[px, py]], dtype=torch.float32, device=rn.device)
        _, idx = find_nearest_lanes(rn.device, rn.quad_centerlines, pts, k=1, spatial_hash=None)
        qidx = int(idx[0, 0].item())
        if qidx < 0 or qidx >= rn.quads_vertices.shape[0]:
            return

        # 高亮该 quad
        verts = rn.quads_vertices[qidx].detach().cpu().numpy()
        if ui['poly'] is not None:
            ui['poly'].remove()
        from matplotlib.patches import Polygon as MplPoly
        ui['poly'] = MplPoly(verts, closed=True, facecolor='none', edgecolor='crimson', linewidth=1.5, linestyle='--')
        ax.add_patch(ui['poly'])

        # 取该quad的关联航点/边界索引
        lane_ids = rn.quad_to_w_lanes_ids
        bound_ids = rn.quad_to_w_boundaries_ids
        lanes_idx = None
        bounds_idx = None
        if hasattr(rn, 'quad_to_w_lanes_ids') and lane_ids.numel() > 0 and qidx < lane_ids.shape[0]:
            lanes_idx = lane_ids[qidx].detach().cpu().numpy()
        if hasattr(rn, 'quad_to_w_boundaries_ids') and bound_ids.numel() > 0 and qidx < bound_ids.shape[0]:
            bounds_idx = bound_ids[qidx].detach().cpu().numpy()

        # 可视化 w_lanes 点
        if lanes_idx is not None and lanes_idx.size > 0 and rn.global_w_lane.numel() > 0:
            wl = rn.global_w_lane[torch.as_tensor(lanes_idx, device=rn.device)]
            lxy = wl.detach().cpu().numpy()
            if ui['lane'] is None:
                ui['lane'] = ax.scatter(lxy[:,0], lxy[:,1], c='red', s=14, alpha=0.9, label='nearest w_lanes')
            else:
                ui['lane'].set_offsets(lxy)
        elif ui['lane'] is not None:
            ui['lane'].remove(); ui['lane'] = None

        # 可视化 w_boundaries 点
        if bounds_idx is not None and bounds_idx.size > 0 and rn.global_w_boundary.numel() > 0:
            wb = rn.global_w_boundary[torch.as_tensor(bounds_idx, device=rn.device)]
            bxy = wb.detach().cpu().numpy()
            if ui['bound'] is None:
                ui['bound'] = ax.scatter(bxy[:,0], bxy[:,1], c='purple', s=10, alpha=0.8, label='nearest boundaries')
            else:
                ui['bound'].set_offsets(bxy)
        elif ui['bound'] is not None:
            ui['bound'].remove(); ui['bound'] = None

        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ax.autoscale()
    fig.canvas.draw_idle()
    plt.show()



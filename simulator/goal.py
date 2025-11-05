"""
基于 GPU 加速的 Dijkstra 路径规划系统
用于计算地图中任意两个 w_lane_id 之间的最短路径
"""
import torch
import json
import os
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

class PathPlanner:
    """
    基于 GPU 的路径规划器
    使用单向导通图计算任意两个(road_id, lane_id)之间的路径
    图结构说明：
    - 节点：每个(road_id, lane_id)表示一个lane
    - 单向边：lane_i的end可以导向lane_j的start（如果两者距离<5）
    - 自然流动：每个lane内部从start到end是自然导通的，不需要显式表示
    通过这个图结构，可以计算出从任意lane的end到任意lane的end的所有路径
    """
    def __init__(self, map_path: str, device: str = 'cuda', max_path_length: int = None):
        """
        初始化路径规划器（全部数据存储在GPU tensor上）
        Args:
            map_path: 地图 JSON 文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            max_path_length: 最大路径长度（从配置读取或指定）
        """
        # 定义无效路径标记值（使用一个足够大的负数，避免与有效索引混淆）
        self.INVALID_PATH_MARKER = -999999
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            device = 'cpu'
        self.device = torch.device(device)
        # 加载配置获取最大路径长度
        if max_path_length is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            max_path_length = config['simulator']['observation']['navigation_feature_dim']
        self.max_path_length = max_path_length
        # 加载地图数据（临时，仅用于读取，不存储）
        with open(map_path, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
        w_lanes = map_data['w_lanes']
        quads = map_data['quads']
        # 创建quad的查找表，用于快速访问
        quads_by_id = {q['poly_id']: q for q in quads}
        # 按 road_id, lane_id 分组 w_lanes，并找出每个分组的 start 和 end
        lane_groups = defaultdict(list)
        for w_lane in w_lanes:
            key = (w_lane['road_id'], w_lane['lane_id'])
            lane_groups[key].append(w_lane)
        # 为每个 road_id, lane_id 找出 start 和 end 的 w_lane
        lane_start_end_dict = {}
        for (road_id, lane_id), w_lanes_in_lane in lane_groups.items():
            # 根据 poly_id 获取对应的 quad 的 s 值
            w_lanes_with_s = []
            for w_lane in w_lanes_in_lane:
                poly_id = w_lane['poly_id']
                quad = quads_by_id[poly_id]
                s = quad.get('s', 0.0)
                w_lanes_with_s.append((w_lane, s))
            # 按 s 值排序
            w_lanes_with_s.sort(key=lambda x: x[1])
            # start 是 s 值最小的，end 是 s 值最大的
            start_w_lane = w_lanes_with_s[0][0]
            end_w_lane = w_lanes_with_s[-1][0]
            lane_start_end_dict[(road_id, lane_id)] = {
                'start': start_w_lane['w_lane_id'],
                'end': end_w_lane['w_lane_id']
            }
        # 创建 (road_id, lane_id) 到索引的映射
        self.lane_keys = sorted(lane_start_end_dict.keys())
        self.n_lanes = len(self.lane_keys)
        self.lane_to_idx = {key: idx for idx, key in enumerate(self.lane_keys)}
        # 批量构建所有w_lanes的坐标矩阵，一次性传输到GPU
        # 创建w_lane_id到索引的映射
        w_lane_id_to_idx = {w_lane['w_lane_id']: i for i, w_lane in enumerate(w_lanes)}
        # 批量创建坐标数组（先CPU numpy）
        coords_array = np.array([[w['center'][0], w['center'][1]] for w in w_lanes], dtype=np.float32)
        # 一次性转换为GPU tensor
        all_w_lane_positions = torch.from_numpy(coords_array).to(device=self.device, dtype=torch.float32)  # (n_w_lanes, 2)

        # 构建start和end位置tensor
        start_indices = [w_lane_id_to_idx[lane_start_end_dict[key]['start']] for key in self.lane_keys]
        end_indices = [w_lane_id_to_idx[lane_start_end_dict[key]['end']] for key in self.lane_keys]
        start_indices_t = torch.tensor(start_indices, dtype=torch.long, device=self.device)
        end_indices_t = torch.tensor(end_indices, dtype=torch.long, device=self.device)
        
        # 使用索引批量提取坐标
        self.start_positions = all_w_lane_positions[start_indices_t]  # (n_lanes, 2)
        self.end_positions = all_w_lane_positions[end_indices_t]  # (n_lanes, 2)
        # 3. 构建联通图：计算每个 lane 的 end 到其他 lane 的 start 的距离
        # 形成单向导通图：lane_i的end -> lane_j的start（如果距离够近）
        # self.end_positions: (n_lanes, 2)
        # self.start_positions: (n_lanes, 2)
        # 计算距离矩阵：dist[i,j] = ||end[i] - start[j]||
        end_positions_expanded = self.end_positions.unsqueeze(1)  # (n_lanes, 1, 2)
        start_positions_expanded = self.start_positions.unsqueeze(0)  # (1, n_lanes, 2)
        distances = torch.norm(end_positions_expanded - start_positions_expanded, dim=2)  # (n_lanes, n_lanes)
        
        # 4. 建立邻接矩阵和边权重矩阵
        # adjacency_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
        CONNECTION_THRESHOLD = 5.0
        self.adjacency_matrix = (distances < CONNECTION_THRESHOLD).float()  # (n_lanes, n_lanes)
        # 预计算边权重矩阵：连接存在的边权重为距离，不存在的为INF
        # 这样在Dijkstra中就不需要重复计算了
        INF = 1e10
        self.edge_weights = torch.where(self.adjacency_matrix > 0, distances, 
                                        torch.full_like(distances, INF))
        # 5. 图结构已构建完成
        # adjacency_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
        # 由于每个lane内部 start->end 是天然导通的，因此：
        # 从 lane_i 的 end 可以到达 lane_j 的 start，
        # 那么就可以继续到达 lane_j 的 end
        # 将数据保留在 CPU 上的字典中（用于调试和索引）
        self.lane_start_end = lane_start_end_dict
        
        # 创建poly_id到lane_idx的映射
        # 每个poly_id对应一个quad，每个quad有road_id和lane_id
        self.poly_id_to_lane_idx = {}
        max_poly_id = max(quads_by_id.keys()) if quads_by_id else 0
        
        # 创建GPU tensor用于快速查找
        # poly_id_lookup[i] 存储poly_id=i对应的lane_idx，如果不存在则为-1
        poly_id_lookup_cpu = np.full(max_poly_id + 1, -1, dtype=np.int64)
        for poly_id, quad in quads_by_id.items():
            road_id = quad['road_id']
            lane_id = quad['lane_id']
            key = (road_id, lane_id)
            if key in self.lane_to_idx:
                lane_idx = self.lane_to_idx[key]
                self.poly_id_to_lane_idx[poly_id] = lane_idx
                poly_id_lookup_cpu[poly_id] = lane_idx
        # 转换为GPU tensor
        self.poly_id_lookup = torch.from_numpy(poly_id_lookup_cpu).to(device=self.device, dtype=torch.long)
        
        # 设置固定waypoints长度（navigation_feature_dim / 2）
        self.waypoints_length = self.max_path_length // 2
        
        # 无效waypoint标记值（需要在_precompute_waypoints_data之前定义）
        self.INVALID_WAYPOINT_MARKER = -1e10  # 使用一个足够大的负数
        
        # 5.5. 预计算waypoints生成所需的数据结构
        print("预计算waypoints生成所需的数据结构...")
        self._precompute_waypoints_data(w_lanes, quads_by_id, lane_groups, w_lane_id_to_idx)
        
        # 6. 预计算所有起点到终点的最短路径并存储在显存中
        print(f"开始预计算所有 {self.n_lanes} x {self.n_lanes} = {self.n_lanes * self.n_lanes} 条路径...")
        self._precompute_all_paths()
        print("路径预计算完成")

    def _precompute_waypoints_data(self, w_lanes, quads_by_id, lane_groups, w_lane_id_to_idx):
        """
        预计算waypoints生成所需的所有数据结构，存储在GPU上
        直接使用JSON中的信息：w_lane有poly_id，quad有next_w_lane_id和prev_w_lane_id
        """
        n_w_lanes = len(w_lanes)
        
        # 1. 构建所有w_lanes的特征矩阵 (x, y, direction_angle)
        w_lane_features = np.zeros((n_w_lanes, 3), dtype=np.float32)
        for i, w_lane in enumerate(w_lanes):
            center = w_lane['center']
            w_lane_features[i, 0] = center[0]
            w_lane_features[i, 1] = center[1]
            w_lane_features[i, 2] = w_lane.get('direction_angle', 0.0)
        self.w_lane_features = torch.from_numpy(w_lane_features).to(
            device=self.device, dtype=torch.float32)  # (n_w_lanes, 3)
        
        # 2. 为每个lane构建排序后的waypoints列表（固定长度）
        max_w_lanes_per_lane = max(len(lane_groups[key]) for key in lane_groups.keys()) if lane_groups else 0
        max_w_lanes_per_lane = max(max_w_lanes_per_lane, 50)  # 安全余量
        
        lane_waypoints_cpu = np.full((self.n_lanes, max_w_lanes_per_lane, 3), 
                                     self.INVALID_WAYPOINT_MARKER, dtype=np.float32)
        lane_waypoints_count = np.zeros(self.n_lanes, dtype=np.int32)
        
        for lane_idx in range(self.n_lanes):
            road_id, lane_id = self.lane_keys[lane_idx]
            w_lanes_in_lane = lane_groups[(road_id, lane_id)]
            
            # 按s值排序（从quad获取）
            w_lanes_with_s = []
            for w_lane in w_lanes_in_lane:
                poly_id = w_lane['poly_id']
                quad = quads_by_id[poly_id]
                s = quad.get('s', 0.0)
                w_lanes_with_s.append((w_lane, s))
            w_lanes_with_s.sort(key=lambda x: x[1])
            
            # 填充waypoints
            for i, (w_lane, _) in enumerate(w_lanes_with_s):
                if i >= max_w_lanes_per_lane:
                    break
                center = w_lane['center']
                lane_waypoints_cpu[lane_idx, i, 0] = center[0]
                lane_waypoints_cpu[lane_idx, i, 1] = center[1]
                lane_waypoints_cpu[lane_idx, i, 2] = w_lane.get('direction_angle', 0.0)
            lane_waypoints_count[lane_idx] = min(len(w_lanes_with_s), max_w_lanes_per_lane)
        
        self.lane_waypoints = torch.from_numpy(lane_waypoints_cpu).to(
            device=self.device, dtype=torch.float32)  # (n_lanes, max_w_lanes_per_lane, 3)
        self.lane_waypoints_count = torch.from_numpy(lane_waypoints_count).to(
            device=self.device, dtype=torch.long)  # (n_lanes,)
        
        # 3. 构建poly_id到next/prev w_lane索引的映射（直接从quad获取）
        # 对于每个poly_id，找到它对应的quad，获取next_w_lane_id和prev_w_lane_id，转换为索引
        max_poly_id = max(quads_by_id.keys()) if quads_by_id else 0
        poly_id_to_next_cpu = np.full(max_poly_id + 1, -1, dtype=np.int64)
        poly_id_to_prev_cpu = np.full(max_poly_id + 1, -1, dtype=np.int64)
        
        # 构建w_lane_index到next/prev w_lane_index的直接映射（用于向量化链遍历）
        # 对于每个w_lane索引w_idx，对应的w_lane有poly_id，通过poly_id找到quad，获取next/prev_w_lane_id
        w_lane_next_idx_cpu = np.full(n_w_lanes, -1, dtype=np.int64)
        w_lane_prev_idx_cpu = np.full(n_w_lanes, -1, dtype=np.int64)
        
        # 方法1: 通过w_lanes构建w_lane索引到next/prev的映射
        for w_idx, w_lane in enumerate(w_lanes):
            poly_id = w_lane['poly_id']
            quad = quads_by_id.get(poly_id)
            if quad:
                # 从quad直接获取next_w_lane_id和prev_w_lane_id
                next_w_lane_id = quad.get('next_w_lane_id')
                prev_w_lane_id = quad.get('prev_w_lane_id')
                
                # 将w_lane_id转换为索引并存储（next/prev 支持列表，取相邻的第一个元素）
                def _as_list(val):
                    if val is None:
                        return []
                    if isinstance(val, (list, tuple)):
                        return list(val)
                    return [val]
                next_list = _as_list(next_w_lane_id)
                prev_list = _as_list(prev_w_lane_id)
                if len(next_list) > 0:
                    _nid = next_list[0]
                    if _nid in w_lane_id_to_idx:
                        w_lane_next_idx_cpu[w_idx] = w_lane_id_to_idx[_nid]
                if len(prev_list) > 0:
                    _pid = prev_list[0]
                    if _pid in w_lane_id_to_idx:
                        w_lane_prev_idx_cpu[w_idx] = w_lane_id_to_idx[_pid]
        
        # 方法2: 通过quads直接构建poly_id到next/prev的映射
        # 对于每个poly_id（可能不在w_lanes中，但在quads中），获取其next/prev_w_lane_id
        for poly_id, quad in quads_by_id.items():
            next_w_lane_id = quad.get('next_w_lane_id')
            prev_w_lane_id = quad.get('prev_w_lane_id')
            
            # poly 到相邻 next/prev（列表取第一个元素）
            next_list = next_w_lane_id if isinstance(next_w_lane_id, (list, tuple)) else ([next_w_lane_id] if next_w_lane_id is not None else [])
            prev_list = prev_w_lane_id if isinstance(prev_w_lane_id, (list, tuple)) else ([prev_w_lane_id] if prev_w_lane_id is not None else [])
            if len(next_list) > 0 and next_list[0] in w_lane_id_to_idx:
                poly_id_to_next_cpu[poly_id] = w_lane_id_to_idx[next_list[0]]
            if len(prev_list) > 0 and prev_list[0] in w_lane_id_to_idx:
                poly_id_to_prev_cpu[poly_id] = w_lane_id_to_idx[prev_list[0]]
        
        self.poly_id_to_next_w_lane = torch.from_numpy(poly_id_to_next_cpu).to(
            device=self.device, dtype=torch.long)
        self.poly_id_to_prev_w_lane = torch.from_numpy(poly_id_to_prev_cpu).to(
            device=self.device, dtype=torch.long)
        # 3.1 额外：构建poly到完整next/prev序列（CSR存储，便于直接读取整条链）
        next_seq_lists = [[] for _ in range(max_poly_id + 1)]
        prev_seq_lists = [[] for _ in range(max_poly_id + 1)]
        for poly_id, quad in quads_by_id.items():
            nxt_list = quad.get('next_w_lane_id', []) or []
            prv_list = quad.get('prev_w_lane_id', []) or []
            if not isinstance(nxt_list, (list, tuple)):
                nxt_list = [nxt_list]
            if not isinstance(prv_list, (list, tuple)):
                prv_list = [prv_list]
            nxt_idx_seq = [w_lane_id_to_idx[w_id] for w_id in nxt_list if w_id in w_lane_id_to_idx]
            prv_idx_seq = [w_lane_id_to_idx[w_id] for w_id in prv_list if w_id in w_lane_id_to_idx]
            # prev保存为 start->poly 的顺序
            prv_idx_seq = list(reversed(prv_idx_seq))
            next_seq_lists[poly_id] = nxt_idx_seq
            prev_seq_lists[poly_id] = prv_idx_seq
        def _build_csr(seq_lists):
            offsets = [0]
            flat = []
            for seq in seq_lists:
                flat.extend(seq)
                offsets.append(len(flat))
            lengths = [offsets[i+1]-offsets[i] for i in range(len(offsets)-1)]
            return (np.asarray(flat, dtype=np.int64), np.asarray(offsets[:-1], dtype=np.int64), np.asarray(lengths, dtype=np.int64))
        n_flat, n_off, n_len = _build_csr(next_seq_lists)
        p_flat, p_off, p_len = _build_csr(prev_seq_lists)
        self.poly_next_seq_flat_idx = torch.from_numpy(n_flat if len(n_flat)>0 else np.asarray([0], dtype=np.int64)).to(self.device)
        self.poly_next_seq_offsets = torch.from_numpy(n_off).to(self.device)
        self.poly_next_seq_lengths = torch.from_numpy(n_len).to(self.device)
        self.poly_prev_seq_flat_idx = torch.from_numpy(p_flat if len(p_flat)>0 else np.asarray([0], dtype=np.int64)).to(self.device)
        self.poly_prev_seq_offsets = torch.from_numpy(p_off).to(self.device)
        self.poly_prev_seq_lengths = torch.from_numpy(p_len).to(self.device)
        
        # w_lane_index直接映射（用于向量化链遍历）
        self.w_lane_next_idx = torch.from_numpy(w_lane_next_idx_cpu).to(
            device=self.device, dtype=torch.long)
        self.w_lane_prev_idx = torch.from_numpy(w_lane_prev_idx_cpu).to(
            device=self.device, dtype=torch.long)
        
        # 5. 预计算每个lane的start和end w_lane索引
        lane_start_w_lane_idx = []
        lane_end_w_lane_idx = []
        for road_id, lane_id in self.lane_keys:
            start_w_lane_id = self.lane_start_end[(road_id, lane_id)]['start']
            end_w_lane_id = self.lane_start_end[(road_id, lane_id)]['end']
            lane_start_w_lane_idx.append(w_lane_id_to_idx[start_w_lane_id])
            lane_end_w_lane_idx.append(w_lane_id_to_idx[end_w_lane_id])
        
        self.lane_start_w_lane_idx = torch.tensor(lane_start_w_lane_idx, dtype=torch.long, device=self.device)
        self.lane_end_w_lane_idx = torch.tensor(lane_end_w_lane_idx, dtype=torch.long, device=self.device)
        
        print(f"  预计算完成: {n_w_lanes} 个w_lanes, {self.n_lanes} 个lanes")
        print(f"  每个lane最多 {max_w_lanes_per_lane} 个waypoints")

    def _precompute_all_paths(self):
        """
        预计算所有起点到终点的最短路径
        两套方案：
        1. prev_matrix (O(n^2)): 前驱矩阵用于重建路径（适用于稀疏查询）
        2. path_matrix (O(n^2 * L_max)): 预先存储所有完整路径（适用于GPU批量查询）
        当空间足够时，使用path_matrix实现极致查询速度
        时间：O(n^3) + O(n^2 * L_avg * L_max)
        空间：O(n^2 * L_max)
        """
        print("  使用Floyd-Warshall算法计算前驱矩阵...")
        INF = 1e10
        # 初始化距离矩阵和前驱矩阵
        dist = self.edge_weights.clone()  # (n_lanes, n_lanes)
        prev = torch.full((self.n_lanes, self.n_lanes), -1, dtype=torch.long, device=self.device)
        # 初始化：自己到自己是0，直接连通的边设置前驱
        for i in range(self.n_lanes):
            dist[i, i] = 0
            prev[i, i] = i
            for j in range(self.n_lanes):
                if i != j and self.edge_weights[i, j] < INF:
                    prev[i, j] = i
        # Floyd-Warshall主循环
        for k in range(self.n_lanes):
            dist_ik = dist[:, k:k+1].expand(-1, self.n_lanes)
            dist_kj = dist[k:k+1, :].expand(self.n_lanes, -1)
            dist_ikj = dist_ik + dist_kj
            better_mask = dist_ikj < dist
            dist = torch.where(better_mask, dist_ikj, dist)
            prev_kj = prev[k:k+1, :].expand(self.n_lanes, -1)
            prev = torch.where(better_mask, prev_kj, prev)
        self.prev_matrix = prev
        self.dist_matrix = dist
        # 第二阶段：GPU向量化构建所有路径
        print("  预计算完整路径矩阵（GPU向量化）...")
        self.path_matrix = self._build_path_matrix_vectorized(prev)
        # 统计信息
        valid_paths_count = (prev >= 0).sum().item()
        max_dist = dist[dist < INF].max().item() if (dist < INF).any() else 0
        print(f"  有效路径数: {valid_paths_count}")
        print(f"  最大路径距离: {max_dist:.2f}")
        print(f"  路径矩阵形状: {self.path_matrix.shape}")
        print(f"  路径矩阵显存占用: {self.path_matrix.numel() * 4 / 1024**2:.2f} MB")
    
    def _build_path_matrix_vectorized(self, prev):
        """
        GPU向量化构建所有路径矩阵 - 完全向量化版本
        使用逐层扩散的方法，所有路径同时展开
        时间复杂度：O(L_max * n^2)，但所有操作都是向量化的
        """
        n = self.n_lanes
        max_len_estimate = self.max_path_length
        
        # 预分配路径矩阵: (n, n, max_len)
        print(f"    使用最大路径长度: {max_len_estimate}")
        path_matrix = torch.full((n, n, max_len_estimate), self.INVALID_PATH_MARKER, dtype=torch.long, device=self.device)
        
        # 初始化：每个路径的第一层就是终点本身
        # path_matrix[i, j, 0] = j (对于所有有效的i->j路径)
        valid_mask = prev >= 0  # (n, n)
        path_matrix[:, :, 0] = torch.where(valid_mask, 
                                           torch.arange(n, device=self.device).unsqueeze(0).expand(n, -1),
                                           torch.full((n, n), self.INVALID_PATH_MARKER, device=self.device))
        
        # 迭代展开路径：逐层扩散
        # 关键：使用高级gather进行完全向量化
        current_layer = path_matrix[:, :, 0]  # (n, n)
        # 跟踪已完成的路径（已完成的不再扩展）
        completed_paths = torch.zeros((n, n), dtype=torch.bool, device=self.device)
        
        for layer in range(1, max_len_estimate):
            # 只对未完成的路径进行扩展
            active_mask = ~completed_paths & (current_layer != self.INVALID_PATH_MARKER)
            
            # 完全向量化的下一层计算
            # 对于每个起点i和终点j，我们需要查找 prev[i, current_layer[i, j]]
            indices_2d = torch.clamp(current_layer, 0, n - 1).long()  # (n, n)
            next_layer = torch.gather(prev, 1, indices_2d)  # (n, n)
            
            # 检查哪些路径已完成（到达起点）
            # 当next_layer[i, j] == i时，路径i->j完成
            start_indices = torch.arange(n, device=self.device).unsqueeze(1).expand(n, n)  # (n, n)
            newly_completed = (next_layer == start_indices) & active_mask  # (n, n)
            
            # 检查循环：如果下一个节点等于当前节点（且不是起点），可能是循环，应该停止
            # 对于路径i->j，如果next_layer[i, j] == current_layer[i, j]且不是起点i，则可能是循环
            cycle_detected = (next_layer == current_layer) & active_mask & (next_layer != start_indices)
            
            # 更新已完成路径标记（包括到达起点和检测到循环的路径）
            completed_paths = completed_paths | newly_completed | cycle_detected
            
            # 对于完成的路径（到达起点），写入起点；对于检测到循环的路径，写入无效标记
            next_layer = torch.where(newly_completed, next_layer,  # 到达起点：写入起点
                            torch.where(cycle_detected, torch.full_like(next_layer, self.INVALID_PATH_MARKER),  # 循环：写入无效标记
                            torch.where(active_mask, next_layer, torch.full_like(next_layer, self.INVALID_PATH_MARKER))))  # 未完成：正常扩展
            
            path_matrix[:, :, layer] = next_layer
            # 检查是否所有路径都已完成
            if completed_paths.all() or (active_mask.sum() == 0):
                print(f"    所有路径完成，最大路径长度: {layer + 1}")
                # 不截断，保持max_path_length的长度，剩余层都是无效标记
                # 继续循环填充无效标记，或者直接返回（剩余层已经是无效标记了）
                break
            # 更新当前层（只考虑活跃的路径，已完成的不再更新）
            current_layer = torch.where(active_mask & ~newly_completed & ~cycle_detected, next_layer, current_layer)
        # 翻转路径（从终点到起点 -> 从起点到终点）
        print("    翻转路径方向...")
        
        # 完全向量化提取有效部分并左对齐
        # path_matrix 形状: (n, n, max_len)
        
        # 1. 创建有效路径的 mask（标记所有有效位置）
        valid_mask = (path_matrix != self.INVALID_PATH_MARKER)  # (n, n, max_len)
        
        # 2. 找到每个路径的长度（第一个无效标记的位置）
        # 使用 argmax 找到第一个无效位置
        invalid_indicator = (~valid_mask).long()  # (n, n, max_len) 1表示无效，0表示有效
        # 在末尾添加一个 1，确保全有效路径也能正确找到长度
        invalid_with_end = torch.cat([invalid_indicator, 
                                     torch.ones((n, n, 1), dtype=torch.long, device=self.device)], 
                                    dim=2)  # (n, n, max_len+1)
        first_invalid_pos = torch.argmax(invalid_with_end, dim=2)  # (n, n)
        path_lengths = first_invalid_pos  # (n, n) 每个路径的有效长度
        
        # 3. 翻转整个 path_matrix（在最后一个维度）
        flipped_matrix = torch.flip(path_matrix, [2])  # (n, n, max_len) 从后往前翻转
        
        # 4. 向量化提取并左对齐
        # 创建索引矩阵：对于每个路径，需要从翻转后的矩阵中提取有效部分
        # 由于已经翻转，我们需要从翻转后的路径中提取前 path_len 个元素（从后往前数的后 path_len 个）
        # 实际上翻转后，原来的 [0:path_len] 变成了 [max_len-path_len:max_len]
        # 但我们想要的是翻转后的前 path_len 个，即原路径的后 path_len 个翻转后的结果
        
        # 更简单的方法：对于每个路径，我们已经知道有效长度 path_len
        # 在翻转后的矩阵中，有效部分在 [max_len - path_len : max_len] 位置
        # 但我们想要左对齐，所以需要重新排列
        
        # 使用 gather 来重新排列：为每个路径创建索引
        # 对于路径 (i,j) 长度为 L：
        # - 翻转后的有效部分在 flipped_matrix[i, j, max_len-L:max_len]
        # - 需要左对齐到 final_matrix[i, j, 0:L]
        # - 即：final_matrix[i, j, k] = flipped_matrix[i, j, max_len-L+k] (0 <= k < L)
        
        # 创建索引矩阵：(n, n, max_len)
        # 对于每个路径 (i,j)，长度为 L，索引 k 对应 flipped_matrix[i, j, max_len-L+k]
        pos_indices = torch.arange(max_len_estimate, device=self.device).unsqueeze(0).unsqueeze(0).expand(n, n, -1)  # (n, n, max_len) [0, 1, 2, ..., max_len-1]
        offset = max_len_estimate - path_lengths.unsqueeze(2)  # (n, n, 1) [max_len-L, max_len-L, ...]
        gather_indices = pos_indices + offset  # (n, n, max_len) 对于路径长度为L，[max_len-L, max_len-L+1, ..., max_len-1, max_len, ...]
        
        # 创建长度 mask：对于路径长度为 L，只有 [0:L] 是有效的
        length_mask = pos_indices < path_lengths.unsqueeze(2)  # (n, n, max_len) True for k < L
        
        # 对于超出路径长度的位置（k >= L），索引无效，但我们仍然需要有效索引值
        # 限制索引范围避免越界（超出部分会被 mask 覆盖，所以值不重要）
        gather_indices = torch.clamp(gather_indices, 0, max_len_estimate - 1)
        
        # 使用 gather 提取并重新排列
        final_matrix = torch.gather(flipped_matrix, 2, gather_indices)  # (n, n, max_len)
        
        # 将超出路径长度的部分设为无效标记
        final_matrix = torch.where(length_mask, final_matrix,
                                  torch.full_like(final_matrix, self.INVALID_PATH_MARKER))
        
        return final_matrix
    
    def path_plan(self, start_poly_ids, end_poly_ids):
        """
        批量路径规划：从poly_id到poly_id的最短路径（完全GPU并行化）
        
        Args:
            start_poly_ids: (B, M) 起点poly_id的tensor
            end_poly_ids: (B, M) 终点poly_id的tensor
            
        Returns:
            torch.Tensor: (B, M, max_path_len) 批量路径结果，无路径用INVALID_PATH_MARKER填充
        """
        B, M = start_poly_ids.shape
        # 确保输入tensor在正确设备上
        start_poly_ids = start_poly_ids.to(self.device)
        end_poly_ids = end_poly_ids.to(self.device)
        # 使用GPU加速批量转换poly_id到lane_idx
        start_lane_indices = self.poly_id_lookup[start_poly_ids]  # (B, M)
        end_lane_indices = self.poly_id_lookup[end_poly_ids]  # (B, M)
        # 检查有效性并处理无效索引（替换为0，后续会检查）
        valid_mask = (start_lane_indices >= 0) & (end_lane_indices >= 0)  # (B, M)
        # 使用高级索引批量获取路径：path_matrix[start_lane_idx, end_lane_idx, :]
        # 注意：必须先将索引展平，然后使用展平后的索引查询，最后reshape回(B, M, max_path_len)
        batch_size = B * M
        start_lane_flat = start_lane_indices.flatten()  # (B*M,)
        end_lane_flat = end_lane_indices.flatten()  # (B*M,)
        # 批量查询：使用展平的索引
        # path_matrix[start_lane_idx, end_lane_idx, :] 返回 (max_path_len,)
        # 我们需要对每个(start_lane_idx, end_lane_idx)对获取完整路径
        max_path_len = self.path_matrix.shape[2]
        # 使用高级索引批量获取所有路径
        # 创建索引对：(start_idx, end_idx) -> (path_matrix维度0的索引, path_matrix维度1的索引)
        paths_flat = self.path_matrix[start_lane_flat, end_lane_flat, :]  # (B*M, max_path_len)
        # 将无效路径（第一元素为无效标记）全部设为无效标记
        valid_mask_flat = valid_mask.flatten()  # (B*M,)
        paths_flat = torch.where(valid_mask_flat.unsqueeze(1), paths_flat, 
                                 torch.full_like(paths_flat, self.INVALID_PATH_MARKER))
        # reshape回(B, M, max_path_len)
        paths = paths_flat.reshape(B, M, max_path_len)  # (B, M, max_path_len)
        return paths
    
    def collect_path_waypoints(self, paths, start_poly_ids, end_poly_ids):
        """
        批量收集路径的waypoints（GPU加速，完全向量化，无for循环）
        Args:
            paths: (B, M, max_path_len) 路径tensor，包含lane索引，无效值为INVALID_PATH_MARKER
            start_poly_ids: (B, M) 起点poly_id
            end_poly_ids: (B, M) 终点poly_id
        Returns:
            waypoints: (B, M, waypoints_length, 3) waypoints tensor，格式为(x, y, angle)
                      无效位置用INVALID_WAYPOINT_MARKER填充
        """
        B, M, max_path_len = paths.shape
        waypoints_length = self.waypoints_length
        
        # 确保输入在正确设备上
        paths = paths.to(self.device)
        start_poly_ids = start_poly_ids.to(self.device)
        end_poly_ids = end_poly_ids.to(self.device)
        
        # 初始化最终输出 (B, M, waypoints_length, 3)
        waypoints = torch.full((B, M, waypoints_length, 3), 
                              self.INVALID_WAYPOINT_MARKER, 
                              dtype=torch.float32, device=self.device)
        
        # 计算每个路径的有效长度（第一个无效标记的位置）
        invalid_mask = (paths == self.INVALID_PATH_MARKER)  # (B, M, max_path_len)
        path_lengths = torch.argmax(invalid_mask.long(), dim=2)  # (B, M) 第一个无效位置
        # 如果全有效，path_lengths会是0，需要特殊处理
        all_valid_mask = ~invalid_mask.any(dim=2)  # (B, M)
        max_path_len_t = torch.full((B, M), max_path_len, dtype=torch.long, device=self.device)
        path_lengths = torch.where(all_valid_mask, max_path_len_t, path_lengths)
        
        # 展平为批量处理 (B*M, ...)
        batch_size = B * M
        paths_flat = paths.flatten(0, 1)  # (B*M, max_path_len)
        start_poly_flat = start_poly_ids.flatten()  # (B*M,)
        end_poly_flat = end_poly_ids.flatten()  # (B*M,)
        path_lengths_flat = path_lengths.flatten()  # (B*M,)
        
        # 按照CPU版本的逻辑顺序收集waypoints：
        # 1. 起点链（从start_quad到第一条lane的end）
        # 2. 中间lanes的所有waypoints（如果path_len > 2）
        # 3. 终点链（从最后一条lane的start到终点）
        # 4. 特殊情况：只有一条lane时，只有起点链和终点链
        
        # 分块处理，避免显存爆炸 (tile over N=B*M)
        K = waypoints_length
        final_flat = waypoints.flatten(0, 1)  # (B*M, K, 3)
        # 经验 tile 大小：CUDA 下 2048，CPU 下 2048（可按需调整）
        tile = 2048
        P = max_path_len
        W = self.lane_waypoints.shape[1]
        n_lanes_total = self.n_lanes

        # 性能计时
        t_total_start = time.perf_counter()
        t_chains_total = 0.0
        t_mid_idx_total = 0.0
        t_mid_gather_total = 0.0
        t_mid_compact_total = 0.0
        t_mid_sample_total = 0.0
        t_concat_compact_total = 0.0
        t_final_write_total = 0.0
        n_tiles = (batch_size + tile - 1) // tile

        for s_idx in range(0, batch_size, tile):
            e_idx = min(s_idx + tile, batch_size)
            n_sub = e_idx - s_idx
            pf = paths_flat[s_idx:e_idx]
            sp = start_poly_flat[s_idx:e_idx]
            ep = end_poly_flat[s_idx:e_idx]
            pl = path_lengths_flat[s_idx:e_idx]
            # 起点链/终点链 (n_sub, K, 3)
            t0 = time.perf_counter()
            # 仅在第一个 tile 启用详细计时
            enable_detailed_timing = (s_idx == 0)
            chain_waypoints_start = self._get_w_lane_chain_waypoints_from_poly_vectorized(
                sp, direction='next', max_chain_len=K, enable_timing=enable_detailed_timing)
            chain_waypoints_end = self._get_w_lane_chain_waypoints_from_poly_vectorized(
                ep, direction='prev', max_chain_len=K, enable_timing=enable_detailed_timing)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_chains_total += time.perf_counter() - t0
                
            # 中间段：向量化压紧（仅对该 tile）
            t0 = time.perf_counter()
            pos_idx = torch.arange(P, device=self.device).view(1, P).expand(n_sub, -1)
            middle_pos_mask = (pos_idx > 0) & (pos_idx < (pl.unsqueeze(1) - 1))
            lane_idx_all = pf
            valid_lane_mask = middle_pos_mask & (lane_idx_all >= 0) & (lane_idx_all < n_lanes_total)
            lane_idx_safe = torch.where(valid_lane_mask, lane_idx_all, torch.zeros_like(lane_idx_all))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_mid_idx_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            lane_wps_all = self.lane_waypoints[lane_idx_safe]                # (n_sub, P, W, 3)
            lane_wps_count = self.lane_waypoints_count[lane_idx_safe]        # (n_sub, P)
            k_idx = torch.arange(W, device=self.device).view(1, 1, W).expand(n_sub, P, -1)
            valid_within_lane = (k_idx < lane_wps_count.unsqueeze(-1))       # (n_sub, P, W)
            valid_all = valid_lane_mask.unsqueeze(-1) & valid_within_lane    # (n_sub, P, W)

            order_pos = pos_idx.view(n_sub, P, 1).expand(-1, -1, W)
            order_k = k_idx
            order_linear = order_pos * (W + 1) + order_k                     # (n_sub, P, W)
            feats = lane_wps_all                                             # (n_sub, P, W, 3)
            feats_flat = feats.view(n_sub, P * W, 3)
            mask_flat = valid_all.view(n_sub, P * W)                          # (n_sub, L)
            total_mid_counts = mask_flat.sum(dim=1)                           # (n_sub,)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_mid_gather_total += time.perf_counter() - t0

            # 使用按行 cumsum 计算写入位置并 scatter 到紧凑缓冲（避免排序）
            t0 = time.perf_counter()
            Lmid = P * W
            compact_buf_all = torch.full((n_sub, Lmid, 3), self.INVALID_WAYPOINT_MARKER,
                                         dtype=torch.float32, device=self.device)
            # 位置：pos = cumsum(mask) - 1，仅在 mask 为真时有效
            pos_in_row = mask_flat.long().cumsum(dim=1) - 1                   # (n_sub, Lmid)
            # 构造批维索引
            batch_idx = torch.arange(n_sub, device=self.device).view(-1, 1).expand(n_sub, Lmid)
            valid_lin = mask_flat
            if valid_lin.any():
                compact_buf_all[batch_idx[valid_lin], pos_in_row[valid_lin], :] = feats_flat[valid_lin, :]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_mid_compact_total += time.perf_counter() - t0

            # 生成 middle_buf: (n_sub, K, 3)
            t0 = time.perf_counter()
            middle_buf = torch.full((n_sub, K, 3), self.INVALID_WAYPOINT_MARKER,
                                    dtype=torch.float32, device=self.device)
            # <=K: 直接取前缀
            small_mid = (total_mid_counts <= K)
            if small_mid.any():
                rows = torch.nonzero(small_mid, as_tuple=False).squeeze(1)
                n_rows = rows.shape[0]
                posK = torch.arange(K, device=self.device).view(1, K).expand(n_rows, -1)
                totals = total_mid_counts[rows].unsqueeze(1)
                valid_pos = posK < totals
                gathered = compact_buf_all[rows].gather(1, posK.unsqueeze(-1).expand(-1, -1, 3))
                if valid_pos.any():
                    vp = valid_pos.unsqueeze(-1).expand(-1, -1, 3)
                    middle_buf[rows][vp] = gathered[vp]
            # >K: 均匀采样
            large_mid = ~small_mid
            if large_mid.any():
                rows = torch.nonzero(large_mid, as_tuple=False).squeeze(1)
                n_rows = rows.shape[0]
                totals = total_mid_counts[rows]
                t_lin = torch.linspace(0, 1, steps=K, device=self.device).view(1, K).expand(n_rows, -1)
                sample_idx = torch.clamp((t_lin * (totals.float().unsqueeze(1) - 1.0)).round().long(), 0)
                gathered = compact_buf_all[rows].gather(1, sample_idx.unsqueeze(-1).expand(-1, -1, 3))
                middle_buf[rows, :, :] = gathered
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_mid_sample_total += time.perf_counter() - t0

            # 拼接三段并压紧到 K
            t0 = time.perf_counter()
            concat_buf = torch.cat([chain_waypoints_start, middle_buf, chain_waypoints_end], dim=1)  # (n_sub, 3K, 3)
            threeK = 3 * K
            valid_mask_all = (concat_buf[:, :, 0] != self.INVALID_WAYPOINT_MARKER)  # (n_sub, 3K)
            total_counts = valid_mask_all.sum(dim=1)                                # (n_sub,)

            # cumsum 压紧三段到 buffer3
            pos_total = valid_mask_all.long().cumsum(dim=1) - 1                     # (n_sub, 3K)
            buffer3 = torch.full((n_sub, threeK, 3), self.INVALID_WAYPOINT_MARKER,
                                 dtype=torch.float32, device=self.device)
            batch_idx3 = torch.arange(n_sub, device=self.device).view(-1, 1).expand(n_sub, threeK)
            valid3 = valid_mask_all
            if valid3.any():
                buffer3[batch_idx3[valid3], pos_total[valid3], :] = concat_buf[valid3, :]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_concat_compact_total += time.perf_counter() - t0

            # 生成该 tile 的最终结果并写入 final_flat
            t0 = time.perf_counter()
            small_mask = (total_counts <= K)
            if small_mask.any():
                rows = torch.nonzero(small_mask, as_tuple=False).squeeze(1)
                n_rows = rows.shape[0]
                pos = torch.arange(K, device=self.device).view(1, K).expand(n_rows, -1)
                totals = total_counts[rows].unsqueeze(1)
                valid_pos = pos < totals
                gathered = buffer3[rows].gather(1, pos.unsqueeze(-1).expand(-1, -1, 3))
                if valid_pos.any():
                    vp = valid_pos.unsqueeze(-1).expand(-1, -1, 3)
                    final_flat[s_idx:e_idx][rows][vp] = gathered[vp]
            large_mask = ~small_mask
            if large_mask.any():
                rows = torch.nonzero(large_mask, as_tuple=False).squeeze(1)
                n_rows = rows.shape[0]
                totals = total_counts[rows]
                t_lin = torch.linspace(0, 1, steps=K, device=self.device).view(1, K).expand(n_rows, -1)
                pos = torch.clamp((t_lin * (totals.float().unsqueeze(1) - 1.0)).round().long(), 0)
                gathered = buffer3[rows].gather(1, pos.unsqueeze(-1).expand(-1, -1, 3))
                final_flat[s_idx:e_idx][rows, :, :] = gathered
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_final_write_total += time.perf_counter() - t0
        
        # 性能统计输出
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t_total_start
        print(f"  collect_path_waypoints 性能分析 (tiles={n_tiles}, tile_size={tile}):")
        print(f"    总耗时: {t_total*1000:.2f} ms")
        print(f"    起点/终点链生成: {t_chains_total*1000:.2f} ms ({t_chains_total/t_total*100:.1f}%)")
        print(f"    中间段索引计算: {t_mid_idx_total*1000:.2f} ms ({t_mid_idx_total/t_total*100:.1f}%)")
        print(f"    中间段waypoints提取: {t_mid_gather_total*1000:.2f} ms ({t_mid_gather_total/t_total*100:.1f}%)")
        print(f"    中间段压紧: {t_mid_compact_total*1000:.2f} ms ({t_mid_compact_total/t_total*100:.1f}%)")
        print(f"    中间段采样: {t_mid_sample_total*1000:.2f} ms ({t_mid_sample_total/t_total*100:.1f}%)")
        print(f"    三段拼接压紧: {t_concat_compact_total*1000:.2f} ms ({t_concat_compact_total/t_total*100:.1f}%)")
        print(f"    最终写入: {t_final_write_total*1000:.2f} ms ({t_final_write_total/t_total*100:.1f}%)")
        
        # 单条lane特殊情形：已被三段拼接流程覆盖（start/end 链 + middle为空即可），无需额外处理
        return waypoints  # (B, M, waypoints_length, 3)
    
    def _get_w_lane_chain_waypoints_from_poly_vectorized(self, poly_ids, direction='next', max_chain_len=10, enable_timing=False):
        """
        基于poly的预存完整链（CSR）直接取序列并转为waypoints。
        Args:
            poly_ids: (B,) poly_id tensor
            direction: 'next' 或 'prev'
            max_chain_len: 取前K个
            enable_timing: 是否启用性能计时（用于调试）
        Returns:
            (B, max_chain_len, 3)
        """
        if enable_timing:
            t0 = time.perf_counter()
        B = poly_ids.shape[0]
        device = poly_ids.device
        out = torch.full((B, max_chain_len, 3), self.INVALID_WAYPOINT_MARKER, dtype=torch.float32, device=device)
        if enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"      [chain] alloc: {(t1-t0)*1000:.3f}ms")
        
        if enable_timing:
            t0 = time.perf_counter()
        if direction == 'next':
            off = self.poly_next_seq_offsets[poly_ids]
            leng = self.poly_next_seq_lengths[poly_ids]
            flat = self.poly_next_seq_flat_idx
        else:
            off = self.poly_prev_seq_offsets[poly_ids]
            leng = self.poly_prev_seq_lengths[poly_ids]
            flat = self.poly_prev_seq_flat_idx
        if enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"      [chain] lookup offsets/lengths: {(t1-t0)*1000:.3f}ms")
        
        if enable_timing:
            t0 = time.perf_counter()
        K = max_chain_len
        pos = torch.arange(K, device=device, dtype=torch.long).view(1, K).expand(B, -1)
        take = torch.minimum(leng, torch.full_like(leng, K))
        idx_in_seq = torch.minimum(pos, torch.clamp(leng.unsqueeze(1) - 1, min=0))
        base = off.unsqueeze(1).expand(B, K)
        flat_idx = base + idx_in_seq
        flat_idx = torch.clamp(flat_idx, 0, max(0, flat.shape[0]-1))
        if enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"      [chain] compute indices: {(t1-t0)*1000:.3f}ms")
        
        if enable_timing:
            t0 = time.perf_counter()
        feats = self.w_lane_features[flat[flat_idx]]  # 双重索引：先索引flat，再索引w_lane_features
        if enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"      [chain] double indexing (flat[flat_idx] -> w_lane_features): {(t1-t0)*1000:.3f}ms")
        
        if enable_timing:
            t0 = time.perf_counter()
        mask = pos < take.unsqueeze(1)
        out[mask] = feats[mask]
        if enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"      [chain] mask & write: {(t1-t0)*1000:.3f}ms")
        return out

if __name__ == '__main__':
    # ==================== 1. 初始化路径规划器 ====================
    planner = PathPlanner(map_path='maps/town2.json', device='cuda')
    print(f"总共 {len(planner.lane_start_end)} 个 (road_id, lane_id) 组合")
    print(f"邻接矩阵形状: {planner.adjacency_matrix.shape}")
    print(f"联通数量: {planner.adjacency_matrix.sum().item()}")
    
    # ==================== 2. 加载地图数据 ====================
    with open('maps/town2.json', 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    all_quads = map_data['quads']
    quads_by_id = {q['poly_id']: q for q in all_quads}
    
    # ==================== 3. 批量生成路径（使用GPU tensor） ====================
    B, M = 4800, 150
    total_paths = B * M
    print(f"\n开始批量生成 {B} x {M} = {total_paths} 条路径...")
    all_poly_ids = [quad['poly_id'] for quad in all_quads if quad['poly_id'] in planner.poly_id_to_lane_idx]
    all_poly_ids_tensor = torch.tensor(all_poly_ids, dtype=torch.long)
    
    # 随机生成起点和终点矩阵（使用GPU tensor）
    print("生成随机起点和终点 tensor...")
    # 创建包含所有可用 poly_ids 的 tensor

    n_available = len(all_poly_ids)
    # 使用 GPU 生成随机索引（如果 GPU 可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_poly_ids_tensor = all_poly_ids_tensor.to(device)
    # 生成随机索引 (B, M)
    random_indices = torch.randint(0, n_available, (B, M), device=device)
    start_poly_tensor = all_poly_ids_tensor[random_indices]  # (B, M)
    # 生成终点索引，确保与起点不同
    end_random_indices = torch.randint(0, n_available, (B, M), device=device)
    # 如果终点索引等于起点索引，重新生成（使用循环直到全部不同）
    same_mask = (end_random_indices == random_indices)
    while same_mask.any():
        # 只重新生成相同的位置
        new_indices = torch.randint(0, n_available, (B, M), device=device)
        end_random_indices = torch.where(same_mask, new_indices, end_random_indices)
        same_mask = (end_random_indices == random_indices)
    end_poly_tensor = all_poly_ids_tensor[end_random_indices]  # (B, M)
    print(f"批量查询 {total_paths} 条路径（GPU加速）...")
    all_paths = planner.path_plan(start_poly_tensor, end_poly_tensor)  # (B, M, max_path_len)
    print("路径生成完成！")
    # 不存储路径数据，直接从tensor中读取
    print(f"总共 {total_paths} 条路径（直接从tensor读取，不存储）")
    
    # ==================== 5. 基准测试 collect_path_waypoints ====================
    print("\n开始基准测试 collect_path_waypoints (B=4800, M=150)...")
    # 预热
    _ = planner.collect_path_waypoints(all_paths[:1, :1, :], start_poly_tensor[:1, :1], end_poly_tensor[:1, :1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runs = 3
    times = []
    for r in range(runs):
        t0 = time.perf_counter()
        _ = planner.collect_path_waypoints(all_paths, start_poly_tensor, end_poly_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        times.append(dt)
        print(f"  第 {r+1}/{runs} 次：{dt:.2f} ms")
    avg = sum(times) / len(times) if times else 0.0
    p95 = sorted(times)[int(0.95 * (len(times) - 1))] if times else 0.0
    print(f"collect_path_waypoints 平均耗时：{avg:.2f} ms | P95：{p95:.2f} ms")

    # ==================== 5. 可视化一条 collect_path_waypoints 结果 ====================
    print("开始可视化 collect_path_waypoints 的结果（尝试寻找一条有有效waypoints的路径）...")
    # 多次尝试，找到至少一条含有效waypoints的路径
    b = 0
    m = 0
    valid_cnt = 0
    max_try = 200
    for _ in range(max_try):
        bi = int(torch.randint(0, B, (1,), device=device).item())
        mi = int(torch.randint(0, M, (1,), device=device).item())
        paths_batch = all_paths[bi:bi+1, mi:mi+1, :]
        start_poly_batch = start_poly_tensor[bi:bi+1, mi:mi+1]
        end_poly_batch = end_poly_tensor[bi:bi+1, mi:mi+1]
        waypoints_batch = planner.collect_path_waypoints(paths_batch, start_poly_batch, end_poly_batch)
        wp = waypoints_batch[0, 0]
        wp_np = wp.detach().cpu().numpy()
        valid_cnt = int(np.sum(wp_np[:, 0] != planner.INVALID_WAYPOINT_MARKER))
        if valid_cnt > 0:
            b, m = bi, mi
            break
    # 若仍无有效点，则展示首条并提示
    if valid_cnt == 0:
        b, m = 0, 0
        paths_batch = all_paths[b:b+1, m:m+1, :]
        start_poly_batch = start_poly_tensor[b:b+1, m:m+1]
        end_poly_batch = end_poly_tensor[b:b+1, m:m+1]
        waypoints_batch = planner.collect_path_waypoints(paths_batch, start_poly_batch, end_poly_batch)
        wp = waypoints_batch[0, 0]
        wp_np = wp.detach().cpu().numpy()
    # 创建图形窗口
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        fig.canvas.manager.set_window_title('collect_path_waypoints visualization')
    except Exception:
        pass

    # 绘制地图多边形作为背景
    for quad in all_quads:
        vertices = quad['vertices']
        vertices_2d = [(v[0], v[1]) for v in vertices]
        polygon = Polygon(vertices_2d, closed=True,
                          facecolor='lightgray', edgecolor='gray',
                          alpha=0.1, linewidth=0.1)
        ax.add_patch(polygon)
    # 设置坐标轴范围为地图包围框
    xs = [v[0] for q in all_quads for v in q['vertices']]
    ys = [v[1] for q in all_quads for v in q['vertices']]
    if len(xs) > 0 and len(ys) > 0:
        margin = 10.0
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ax.set_ylim(min(ys)-margin, max(ys)+margin)
    
    # 绘制起点与终点 polygon
    start_poly_id = int(start_poly_batch[0, 0].item())
    end_poly_id = int(end_poly_batch[0, 0].item())
    start_quad = quads_by_id[start_poly_id]
    end_quad = quads_by_id[end_poly_id]
    sv = [(v[0], v[1]) for v in start_quad['vertices']]
    ev = [(v[0], v[1]) for v in end_quad['vertices']]
    ax.add_patch(Polygon(sv, closed=True, facecolor='green', edgecolor='darkgreen', alpha=0.5, linewidth=2, label='Start', zorder=5))
    ax.add_patch(Polygon(ev, closed=True, facecolor='red', edgecolor='darkred', alpha=0.5, linewidth=2, label='End', zorder=5))

    # 绘制 waypoints 箭头
    arrow_length = 10.0
    for i, p in enumerate(wp_np):
        if p[0] == planner.INVALID_WAYPOINT_MARKER:
            continue
        x, y, ang = float(p[0]), float(p[1]), float(p[2])
        dx = arrow_length * np.cos(ang)
        dy = arrow_length * np.sin(ang)
        ax.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='purple', ec='purple', alpha=0.8,
                 length_includes_head=True, zorder=6, label='Waypoint' if valid_cnt == 0 else None)
        valid_cnt += 1

    ax.set_title(f'collect_path_waypoints demo | (b={b}, m={m}) | valid waypoints: {valid_cnt} / {planner.waypoints_length}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
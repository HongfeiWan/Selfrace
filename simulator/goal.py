"""
基于 GPU 加速的 Dijkstra 路径规划系统
用于计算地图中任意两个 w_lane_id 之间的最短路径
"""
import torch
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

# TODO：解决一下collect_path_w_lane_id函数内分块的问题。
# 理论上不需要分块，创建三个空的tensor,长度为B,M,max_path_len
# 然后直接写入第一段找到的路径，最后一段找到的路径，通过_get_w_lane_chain_w_lane_id_from_poly_vectorized获得
# 然后path_plan返回的内容可以直接查询road_id,lane_id获取中间的所有w_lane_id（所以需要预先构建一个road_id,lane_id→所有w_lane_id的映射）
# 最后根据他们各自的offset，比较一下直接拼是否长度超过了max_path_len，如果超过了就降采样。然后输出也是B,M,max_path_len的一个tensor。根本没有很大显存的中间变量

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
    def __init__(self, device: str = 'cuda', road_network: "RoadNetwork" = None):
        """
        初始化路径规划器（全部数据存储在GPU tensor上）
        Args:
            map_path: 地图 JSON 文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            max_path_length: 最大路径长度（从配置读取或指定）
        """
        # 读取配置（无论 max_path_length 是否传入，都需要其他配置）
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        device_obj = torch.device(device)

        # ==================== 初始化所有 self 变量 ====================
        # 基础配置（从配置读取）
        _invalid_src = config['simulator']['observation'].get('INVALID_MARKER')
        self.INVALID_MARKER = _invalid_src
        self.INVALID_PATH_MARKER = int(_invalid_src)
        self.INVALID_w_lane_id_MARKER = float(_invalid_src)
        
        # Waypoint 对应的 w_lane_id（x, y, yaw），默认置为无效标记
        self.current_waypoint_w_lane_id = torch.full(
            (3,),
            self.INVALID_w_lane_id_MARKER,
            dtype=torch.float32,
            device=device_obj
        )

        # 路径长度相关（确保为int，且非None）
        nav_chains = config['simulator']['observation'].get('num_navigation_chains')
        self.max_path_length = int(nav_chains)
        self.w_lane_ids_length = int(self.max_path_length)
        # 起终点链长度（确保为int，且非None）
        start_end_chains = config['simulator']['observation'].get('num_start&end_chains')
        self.max_chain_len = int(start_end_chains)
        
        self.device = device_obj
        # 可选的 RoadNetwork（仅保存引用，后续可复用其张量/元数据）
        self.road_network = road_network
        if self.road_network is None:
            raise ValueError("PathPlanner 需要提供 road_network")
        
        # ==================== 开始实际初始化 ====================
        # 使用 RoadNetwork 预计算的数据
        # 图结构相关
        rn = self.road_network
        self.lane_keys = rn.lane_keys
        self.n_lanes = rn.n_lanes
        self.lane_to_idx = rn.lane_to_idx
        self.lane_start_end = rn.lane_start_end
        self.start_positions = rn.start_positions
        self.end_positions = rn.end_positions
        self.adjacency_matrix = rn.adjacency_matrix
        self.edge_weights = rn.edge_weights
        # poly_id 映射相关
        # poly 映射来自 RoadNetwork
        self.poly_id_to_lane_idx = rn.poly_id_to_lane_idx
        self.poly_id_lookup = rn.poly_id_lookup

        self.poly_id_to_next_w_lane = None
        self.poly_id_to_prev_w_lane = None
        # w_lane 相关
        self.w_lane_features = None
        self.w_lane_next_idx = None
        self.w_lane_prev_idx = None
        
        # lane 的 w_lane_ids 相关
        self.lane_w_lane_ids = None
        self.lane_w_lane_ids_count = None
        self.lane_start_w_lane_idx = None
        self.lane_end_w_lane_idx = None
        
        # poly 序列相关（CSR 格式）
        self.poly_next_seq_flat_idx = None
        self.poly_next_seq_offsets = None
        self.poly_next_seq_lengths = None
        self.poly_prev_seq_flat_idx = None
        self.poly_prev_seq_offsets = None
        self.poly_prev_seq_lengths = None
        
        # 路径预计算相关
        self.prev_matrix = None
        self.dist_matrix = None
        self.path_matrix = None
        
        # 5.5. 预计算w_lane_ids生成所需的数据结构
        print("预计算w_lane_ids生成所需的数据结构...")
        self._precompute_w_lane_ids_data(rn.w_lanes_raw, rn.quads_by_id, rn.lane_groups, rn.w_lane_id_to_idx)
        
        # 6. 预计算所有起点到终点的最短路径并存储在显存中
        print(f"开始预计算所有 {self.n_lanes} x {self.n_lanes} = {self.n_lanes * self.n_lanes} 条路径...")
        self._precompute_all_paths()
        print("路径预计算完成")

    def _precompute_w_lane_ids_data(self, w_lanes, quads_by_id, lane_groups, w_lane_id_to_idx):
        """
        预计算w_lane_ids生成所需的所有数据结构，存储在GPU上
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
        
        # 2. 为每个lane构建排序后的w_lane_ids列表（固定长度）
        max_w_lanes_per_lane = max(len(lane_groups[key]) for key in lane_groups.keys()) if lane_groups else 0
        max_w_lanes_per_lane = max(max_w_lanes_per_lane, 50)  # 安全余量
        
        lane_w_lane_ids_cpu = np.full((self.n_lanes, max_w_lanes_per_lane, 3), 
                                     self.INVALID_w_lane_id_MARKER, dtype=np.float32)
        lane_w_lane_ids_count = np.zeros(self.n_lanes, dtype=np.int32)
        
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
            
            # 填充w_lane_ids
            for i, (w_lane, _) in enumerate(w_lanes_with_s):
                if i >= max_w_lanes_per_lane:
                    break
                center = w_lane['center']
                lane_w_lane_ids_cpu[lane_idx, i, 0] = center[0]
                lane_w_lane_ids_cpu[lane_idx, i, 1] = center[1]
                lane_w_lane_ids_cpu[lane_idx, i, 2] = w_lane.get('direction_angle', 0.0)
            lane_w_lane_ids_count[lane_idx] = min(len(w_lanes_with_s), max_w_lanes_per_lane)
        
        self.lane_w_lane_ids = torch.from_numpy(lane_w_lane_ids_cpu).to(
            device=self.device, dtype=torch.float32)  # (n_lanes, max_w_lanes_per_lane, 3)
        self.lane_w_lane_ids_count = torch.from_numpy(lane_w_lane_ids_count).to(
            device=self.device, dtype=torch.long)  # (n_lanes,)
        
        # 3. 构建poly到完整next/prev序列（CSR存储，便于直接读取整条链）
        max_poly_id = max(quads_by_id.keys()) if quads_by_id else 0
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
        print(f"  每个lane最多 {max_w_lanes_per_lane} 个w_lane_ids")

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
    
    def collect_path_w_lane_ids(self, paths, start_poly_ids, end_poly_ids):
        """
        批量收集路径的w_lane_ids（GPU加速，完全向量化，无for循环，无分块，直接写入）
        Args:
            paths: (B, M, max_path_len) 路径tensor，包含lane索引，无效值为INVALID_PATH_MARKER
            start_poly_ids: (B, M) 起点poly_id
            end_poly_ids: (B, M) 终点poly_id
        Returns:
            w_lane_ids: (B, M, w_lane_ids_length, 3) w_lane_ids tensor，格式为(x, y, angle)
                      无效位置用INVALID_w_lane_id_MARKER填充
        """
        B, M, max_path_len = paths.shape
        w_lane_ids_length = self.w_lane_ids_length
        
        # 确保输入在正确设备上
        paths = paths.to(self.device)
        start_poly_ids = start_poly_ids.to(self.device)
        end_poly_ids = end_poly_ids.to(self.device)
        
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

        # ==================== 创建三个空tensor，直接写入（统一使用w_lane_ids_length） ====================
        # 1. 起点段：从start_poly获取，使用_get_w_lane_chain_w_lane_ids_from_poly_vectorized
        start_segment = self._get_w_lane_chain_w_lane_ids_from_poly_vectorized(
            start_poly_flat, direction='next', max_chain_len=self.max_chain_len)  # (B*M, max_chain_len, 3)
        
        # 2. 终点段：从end_poly获取
        end_segment = self._get_w_lane_chain_w_lane_ids_from_poly_vectorized(
            end_poly_flat, direction='prev', max_chain_len=self.max_chain_len)  # (B*M, max_chain_len, 3)
        
        # 3. 中间段：从paths的中间lane_idx获取，直接写入到middle_segment（分batchsize处理）
        middle_segment = torch.full((batch_size, w_lane_ids_length, 3), 
                                   self.INVALID_w_lane_id_MARKER,
                                   dtype=torch.float32, device=self.device)
        
        # 分块处理，避免显存爆炸
        tile = 2048  # 可以根据显存调整
        P = max_path_len
        n_lanes_total = self.n_lanes

        for s_idx in range(0, batch_size, tile):
            e_idx = min(s_idx + tile, batch_size)
            n_sub = e_idx - s_idx
            
            # 当前tile的数据
            pf = paths_flat[s_idx:e_idx]  # (n_sub, P)
            pl = path_lengths_flat[s_idx:e_idx]  # (n_sub,)
            
            # 提取中间段的lane_idx (pos > 0 and pos < path_len-1)
            pos_idx = torch.arange(P, device=self.device).view(1, P).expand(n_sub, -1)  # (n_sub, P)
            middle_pos_mask = (pos_idx > 0) & (pos_idx < (pl.unsqueeze(1) - 1))  # (n_sub, P)
            valid_lane_mask = middle_pos_mask & (pf >= 0) & (pf < n_lanes_total)  # (n_sub, P)
            
            # 只获取count，不展开数据
            lane_idx_safe = torch.where(valid_lane_mask, pf, torch.zeros_like(pf))
            lane_wps_count = self.lane_w_lane_ids_count[lane_idx_safe]  # (n_sub, P)
            lane_wps_count = torch.where(valid_lane_mask, lane_wps_count, torch.zeros_like(lane_wps_count))

            # 计算每行的写入偏移（每行内，每个lane的起始位置）- 向量化
            row_cumsum = torch.cumsum(lane_wps_count, dim=1)  # (n_sub, P) 每行累积count
            row_offsets = torch.cat([torch.zeros(n_sub, 1, dtype=torch.long, device=self.device), 
                                    row_cumsum[:, :-1]], dim=1)  # (n_sub, P) 每行的lane起始偏移
            
            # 只对有效位置处理，不展开到W维度
            # 找到所有有效的 (batch_idx, p) 组合（在当前tile内）
            valid_positions = torch.nonzero(valid_lane_mask, as_tuple=False)  # (N_valid, 2) [local_idx, p]
            
            if valid_positions.shape[0] > 0:
                local_indices = valid_positions[:, 0]  # (N_valid,) tile内的索引
                p_indices = valid_positions[:, 1]  # (N_valid,)
                
                # 转换为全局索引
                global_indices = local_indices + s_idx  # (N_valid,)
                
                # 获取对应的lane_idx, count, offset
                lane_idx_valid = lane_idx_safe[local_indices, p_indices]  # (N_valid,)
                count_valid = lane_wps_count[local_indices, p_indices]  # (N_valid,)
                offset_valid = row_offsets[local_indices, p_indices]  # (N_valid,)
                
                # 计算每个位置需要写入的数量（限制在w_lane_ids_length内）
                write_count = torch.minimum(count_valid, w_lane_ids_length - offset_valid)  # (N_valid,)
                write_count = torch.maximum(write_count, torch.zeros_like(write_count))  # 确保非负
                
                # 向量化写入：对所有有效位置批量处理
                # 使用repeat_interleave展开索引，避免嵌套循环
                repeat_counts = write_count.long()  # (N_valid,)
                total_points = repeat_counts.sum().item()  # 总点数
                
                if total_points > 0:
                    # 创建展开的batch索引和lane索引
                    batch_idx_expanded = torch.repeat_interleave(global_indices, repeat_counts)  # (total_points,)
                    lane_idx_expanded = torch.repeat_interleave(lane_idx_valid, repeat_counts)  # (total_points,)
                    offset_expanded = torch.repeat_interleave(offset_valid, repeat_counts)  # (total_points,)
                    
                    # 创建每个位置内的点索引（0, 1, 2, ..., count-1）
                    # 使用cumsum创建分组索引，然后减去偏移
                    cumsum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), 
                                              repeat_counts.cumsum(dim=0)[:-1]])  # (N_valid,)
                    point_idx_base = torch.arange(total_points, device=self.device)  # (total_points,)
                    point_idx_local = point_idx_base - torch.repeat_interleave(cumsum_counts, repeat_counts)  # (total_points,)
                    
                    # 计算全局写入位置
                    write_pos_expanded = offset_expanded + point_idx_local  # (total_points,)
                    
                    # 限制在w_lane_ids_length内
                    valid_write_mask = write_pos_expanded < w_lane_ids_length
                    if valid_write_mask.any():
                        batch_idx_final = batch_idx_expanded[valid_write_mask]
                        write_pos_final = write_pos_expanded[valid_write_mask]
                        lane_idx_final = lane_idx_expanded[valid_write_mask]
                        point_idx_final = point_idx_local[valid_write_mask]
                        
                        # 批量获取数据并写入（完全向量化）
                        w_lane_data = self.lane_w_lane_ids[lane_idx_final, point_idx_final, :]  # (N_write, 3)
                        middle_segment[batch_idx_final, write_pos_final, :] = w_lane_data
            
        # ==================== 对中间段进行采样判断和拼接 ====================
        # start_segment和end_segment永不采样，完整保留（各10个点）
        # 只对middle_segment判断是否超过 w_lane_ids_length - 10 - 10 = w_lane_ids_length - 20
        max_middle_len = w_lane_ids_length - 2*self.max_chain_len  # 中间段最大长度
        
        # 计算middle_segment的有效长度
        middle_valid_mask = (middle_segment[:, :, 0] != self.INVALID_w_lane_id_MARKER)  # (B*M, w_lane_ids_length)
        middle_counts = middle_valid_mask.sum(dim=1)  # (B*M,) 每个路径中间段的有效点数
        
        # 向量化统计需要采样的路径
        need_sample_mask = (middle_counts > max_middle_len)
        # 创建采样后的middle_segment
        middle_segment_sampled = middle_segment[:, :max_middle_len, :].clone()  # (B*M, max_middle_len, 3)
        # 对需要采样的路径进行完全向量化均匀采样
        if need_sample_mask.any():
            rows = torch.nonzero(need_sample_mask, as_tuple=False).squeeze(1)  # (N_need,)
            counts_need_sample = middle_counts[rows]  # (N_need,) 需要采样的路径的中间段长度
            
            N_need = rows.shape[0]
            if max_middle_len > 1:
                base_positions = torch.arange(max_middle_len, device=self.device, dtype=torch.float32).unsqueeze(0).expand(N_need, -1)  # (N_need, max_middle_len)
                step = (counts_need_sample.float() - 1).unsqueeze(1) / (max_middle_len - 1)
                sample_positions = base_positions * step
            else:
                sample_positions = torch.zeros((N_need, 1), device=self.device, dtype=torch.float32)
            sample_indices = torch.floor(sample_positions).long()
            upper_bound = (counts_need_sample - 1).unsqueeze(1)
            sample_indices = torch.minimum(sample_indices, upper_bound)
            sample_indices = torch.clamp(sample_indices, min=0, max=w_lane_ids_length - 1)
            sample_indices[:, -1] = torch.minimum(upper_bound.squeeze(1), torch.tensor(w_lane_ids_length - 1, device=self.device))

            # 使用gather进行向量化采样
            # middle_segment[rows]: (N_need, w_lane_ids_length, 3)
            # sample_indices: (N_need, max_middle_len)
            # 需要将sample_indices扩展到3维以匹配gather的要求
            sample_indices_expanded = sample_indices.unsqueeze(2).expand(-1, -1, 3)  # (N_need, max_middle_len, 3)
            
            # 使用gather进行采样
            sampled_data = torch.gather(
                middle_segment[rows],  # (N_need, w_lane_ids_length, 3)
                dim=1,
                index=sample_indices_expanded  # (N_need, max_middle_len, 3)
            )  # (N_need, max_middle_len, 3)
            
            # 写入到采样后的middle_segment
            middle_segment_sampled[rows] = sampled_data
        
        # 直接拼接三段：start(max_chain_len) + middle(max_middle_len) + end(max_chain_len) = w_lane_ids_length
        # 注意：end_segment 需要反转顺序，因为它是 [终点, 前1, ..., 前9]，需要变成 [前9, ..., 前1, 终点]
        end_segment_flipped = torch.flip(end_segment, dims=[1])  # 沿着序列维度反转

        w_lane_ids_flat = torch.cat([
            start_segment,  # (B*M, max_chain_len, 3) - [起点, 后1, ..., 后9]
            middle_segment_sampled,  # (B*M, max_middle_len, 3) - [中间点...]
            end_segment_flipped  # (B*M, max_chain_len, 3) - [前9, ..., 前1, 终点]
        ], dim=1)  # (B*M, w_lane_ids_length, 3)
        # reshape回 (B, M, w_lane_ids_length, 3)
        w_lane_ids = w_lane_ids_flat.reshape(B, M, w_lane_ids_length, 3)
        
        return w_lane_ids
    
    def _get_w_lane_chain_w_lane_ids_from_poly_vectorized(self, poly_ids, direction='next', max_chain_len=10):
        """
        基于poly的预存完整链（CSR）直接取序列并转为w_lane_ids。
        Args:
            poly_ids: (B,) poly_id tensor
            direction: 'next' 或 'prev'
            max_chain_len: 取前K个
        Returns:
            (B, max_chain_len, 3)
        """
        B = poly_ids.shape[0]
        device = poly_ids.device
        out = torch.full((B, max_chain_len, 3), self.INVALID_w_lane_id_MARKER, dtype=torch.float32, device=device)
        
        if direction == 'next':
            off = self.poly_next_seq_offsets[poly_ids]
            leng = self.poly_next_seq_lengths[poly_ids]
            flat = self.poly_next_seq_flat_idx
        else:
            off = self.poly_prev_seq_offsets[poly_ids]
            leng = self.poly_prev_seq_lengths[poly_ids]
            flat = self.poly_prev_seq_flat_idx
        
        K = max_chain_len
        pos = torch.arange(K, device=device, dtype=torch.long).view(1, K).expand(B, -1)
        take = torch.minimum(leng, torch.full_like(leng, K))
        
        # 无论 next 还是 prev，都从序列开头取前 K 个点
        # 因为在构建时，prev 已经 reversed，所以两者的顺序都是从近到远
        idx_in_seq = torch.minimum(pos, torch.clamp(leng.unsqueeze(1) - 1, min=0))
        
        base = off.unsqueeze(1).expand(B, K)
        flat_idx = base + idx_in_seq
        flat_idx = torch.clamp(flat_idx, 0, max(0, flat.shape[0]-1))
        feats = self.w_lane_features[flat[flat_idx]]  # 双重索引：先索引flat，再索引w_lane_features
        mask = pos < take.unsqueeze(1)
        out[mask] = feats[mask]
        return out

if __name__ == '__main__':
    # ==================== 1. 初始化 RoadNetwork 和 Planner ====================
    from road import RoadNetwork
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rn = RoadNetwork(map_path=os.path.join(os.path.dirname(__file__), '..', 'maps', 'town2.json'), device=torch.device(device))
    planner = PathPlanner(device=device, road_network=rn)

    # ==================== 2. 生成测试数据 ====================
    all_poly_ids_tensor = rn.quad_ids
    quads_vertices_np = rn.quads_vertices.detach().cpu().numpy()
    quads_by_id = rn.quads_by_id

    B, M = 4800, 150
    valid_poly_ids_tensor = all_poly_ids_tensor
    if all_poly_ids_tensor.numel() > 0:
        max_poly_id = all_poly_ids_tensor.max().item()
        if max_poly_id < planner.poly_id_lookup.shape[0]:
            valid_mask = (planner.poly_id_lookup[all_poly_ids_tensor] >= 0)
            valid_poly_ids_tensor = all_poly_ids_tensor[valid_mask]
    n_available = int(valid_poly_ids_tensor.shape[0])

    if n_available == 0:
        raise RuntimeError('No valid poly ids available for demo')
    random_indices = torch.randint(0, n_available, (B, M), device=device)
    start_poly_tensor = valid_poly_ids_tensor[random_indices]
    end_random_indices = torch.randint(0, n_available, (B, M), device=device)
    same_mask = (end_random_indices == random_indices)
    while same_mask.any():
        new_indices = torch.randint(0, n_available, (B, M), device=device)
        end_random_indices = torch.where(same_mask, new_indices, end_random_indices)
        same_mask = (end_random_indices == random_indices)
    end_poly_tensor = valid_poly_ids_tensor[end_random_indices]

    all_paths = planner.path_plan(start_poly_tensor, end_poly_tensor)
    all_w_lane_ids = planner.collect_path_w_lane_ids(all_paths, start_poly_tensor, end_poly_tensor)

    print("all_w_lane_ids shape:", tuple(all_w_lane_ids.shape))
    sample_b, sample_m = 0, 0
    if all_w_lane_ids.numel() > 0:
        sample_path = all_w_lane_ids[sample_b, sample_m].detach().cpu().tolist()
        print(f"Sample path for B={sample_b}, M={sample_m}:")
        for idx, waypoint in enumerate(sample_path):
            if waypoint[0] == planner.INVALID_w_lane_id_MARKER:
                print(f"  [{idx:02d}] INVALID")
            else:
                x, y, angle = waypoint
                print(f"  [{idx:02d}] x={x:.4f}, y={y:.4f}, angle={angle:.4f}")
    else:
        print("all_w_lane_ids is empty.")

    # ==================== 3. 基于 Pygame + OpenGL 可视化 ====================
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import glClearColor, glClear, GL_COLOR_BUFFER_BIT, glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity, glOrtho, glBegin, glEnd, glVertex2f, glColor3f, GL_QUADS, GL_LINES, GL_POINTS, glPointSize
    except Exception:
        print('请安装 pygame 和 PyOpenGL: pip install pygame PyOpenGL')
        raise

    # 坐标范围
    if quads_vertices_np.shape[0] > 0:
        xs = quads_vertices_np[:, :, 0].reshape(-1)
        ys = quads_vertices_np[:, :, 1].reshape(-1)
        margin = 20.0
        x_min, x_max = float(xs.min() - margin), float(xs.max() + margin)
        y_min, y_max = float(ys.min() - margin), float(ys.max() + margin)
    else:
        x_min, x_max, y_min, y_max = -100.0, 100.0, -100.0, 100.0

    pygame.init()
    screen = pygame.display.set_mode((1080, 800), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('collect_path_w_lane_ids (SPACE: next path, ESC: quit)')
    
    # 初始化字体
    try:
        font = pygame.font.Font(None, 36)
    except:
        font = pygame.font.SysFont('arial', 36)

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(x_min, x_max, y_min, y_max, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    # 可用路径索引列表
    flat_counts = (all_w_lane_ids[..., 0] != planner.INVALID_w_lane_id_MARKER).sum(dim=2).view(-1)
    indices = torch.nonzero(flat_counts > 0, as_tuple=False).flatten().tolist()
    if not indices:
        indices = [0]
    cur_idx = 0

    def draw_quads():
        glColor3f(0.85, 0.85, 0.85)
        for verts in quads_vertices_np:
            glBegin(GL_QUADS)
            glVertex2f(float(verts[0, 0]), float(verts[0, 1]))
            glVertex2f(float(verts[1, 0]), float(verts[1, 1]))
            glVertex2f(float(verts[2, 0]), float(verts[2, 1]))
            glVertex2f(float(verts[3, 0]), float(verts[3, 1]))
            glEnd()

    def draw_point_xy(x, y, r, g, b, size=8.0):
        glColor3f(r, g, b)
        glPointSize(float(size))
        glBegin(GL_POINTS)
        glVertex2f(float(x), float(y))
        glEnd()

    def draw_path(wp_np_local):
        glColor3f(0.5, 0.1, 0.7)
        L = 10.0
        for p in wp_np_local:
            if p[0] == planner.INVALID_w_lane_id_MARKER:
                continue
            x, y, ang = float(p[0]), float(p[1]), float(p[2])
            dx = L * np.cos(ang)
            dy = L * np.sin(ang)
            glBegin(GL_LINES)
            glVertex2f(x, y)
            glVertex2f(x + dx, y + dy)
            glEnd()
    
    def draw_text(text, x, y, r=0.0, g=0.0, b=0.0):
        """在OpenGL窗口中渲染文本"""
        from OpenGL.GL import glEnable, GL_TEXTURE_2D, glBindTexture, glTexParameteri, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST, glTexImage2D, GL_RGBA, GL_UNSIGNED_BYTE, glDisable, glPushMatrix, glPopMatrix, glTranslatef, glScalef, GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE
        from OpenGL.GL import GL_RGB
        
        # 渲染文本到surface
        text_surface = font.render(text, True, (int(r*255), int(g*255), int(b*255)), (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        width, height = text_surface.get_size()
        
        # 创建纹理并显示
        glEnable(GL_TEXTURE_2D)
        texture_id = glBindTexture(GL_TEXTURE_2D, 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # 绘制纹理矩形
        glPushMatrix()
        glTranslatef(x, y, 0)
        glScalef(width, height, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()
        glPopMatrix()
        
        glDisable(GL_TEXTURE_2D)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    cur_idx = (cur_idx + 1) % len(indices)

        flat_i = indices[cur_idx]
        b = flat_i // M
        m = flat_i % M
        wp = all_w_lane_ids[b, m].detach().cpu().numpy()
        start_poly_id = int(start_poly_tensor[b, m].item())
        end_poly_id = int(end_poly_tensor[b, m].item())
        
        # 更新窗口标题显示B和M编号
        pygame.display.set_caption(f'collect_path_w_lane_ids - B={b}, M={m} (SPACE: next path, ESC: quit)')
        
        # 起点：优先使用路径的第一个有效 w_lane 点，否则使用 start_poly_id 的 center
        valid = wp[:, 0] != planner.INVALID_w_lane_id_MARKER
        if valid.any():
            first_idx = int(np.argmax(valid))
            sx, sy = float(wp[first_idx, 0]), float(wp[first_idx, 1])
        else:
            # 兜底：使用起点 quad 的 center
            start_quad = quads_by_id.get(start_poly_id, {})
            center = start_quad.get('center', (0.0, 0.0))
            sx, sy = float(center[0]), float(center[1])
        
        # 终点：始终使用 end_poly_id 对应的 quad 的质心（显示 end_poly_tensor 的位置）
        end_quad = quads_by_id.get(end_poly_id, {})
        center = end_quad.get('center', (0.0, 0.0))
        ex, ey = float(center[0]), float(center[1])

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        draw_quads()
        draw_point_xy(sx, sy, 0.2, 0.8, 0.2, size=5.0)
        draw_point_xy(ex, ey, 0.9, 0.2, 0.2, size=10.0)
        draw_path(wp)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
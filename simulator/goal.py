"""
基于 GPU 加速的 Dijkstra 路径规划系统
用于计算地图中任意两个 w_lane_id 之间的最短路径
"""
import torch
import json
import os
from collections import defaultdict
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
        self.map_path = map_path
        
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
        
        # 存储quads数据以便后续使用
        self.quads_by_id = quads_by_id
        # 按 road_id, lane_id 分组 w_lanes，并找出每个分组的 start 和 end
        lane_groups = defaultdict(list)
        for w_lane in w_lanes:
            key = (w_lane['road_id'], w_lane['lane_id'])
            lane_groups[key].append(w_lane)
        # 为每个 road_id, lane_id 找出 start 和 end 的 w_lane
        lane_start_end_dict = {}
        w_lane_id_to_pos = {}  # 映射 w_lane_id 到 position
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
        self.end_to_start_matrix = (distances < CONNECTION_THRESHOLD).float()  # (n_lanes, n_lanes)
        # 预计算边权重矩阵：连接存在的边权重为距离，不存在的为INF
        # 这样在Dijkstra中就不需要重复计算了
        INF = 1e10
        self.edge_weights = torch.where(self.end_to_start_matrix > 0, distances, 
                                        torch.full_like(distances, INF))
        # 5. 图结构已构建完成
        # end_to_start_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
        # 由于每个lane内部 start->end 是天然导通的，因此：
        # 从 lane_i 的 end 可以到达 lane_j 的 start，
        # 那么就可以继续到达 lane_j 的 end
        # 将数据保留在 CPU 上的字典中（用于调试和索引）
        self.lane_start_end = lane_start_end_dict
        # 为了兼容BFS接口，将邻接矩阵统一命名为adjacency_matrix
        self.adjacency_matrix = self.end_to_start_matrix
        
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
        
        # 6. 预计算所有起点到终点的最短路径并存储在显存中
        print(f"开始预计算所有 {self.n_lanes} x {self.n_lanes} = {self.n_lanes * self.n_lanes} 条路径...")
        self._precompute_all_paths()
        print("路径预计算完成")

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
    
if __name__ == '__main__':
    import random
    
    # ==================== 1. 初始化路径规划器 ====================
    planner = PathPlanner(map_path='maps/town2.json', device='cuda')
    print(f"总共 {len(planner.lane_start_end)} 个 (road_id, lane_id) 组合")
    print(f"邻接矩阵形状: {planner.adjacency_matrix.shape}")
    print(f"联通数量: {planner.adjacency_matrix.sum().item()}")
    
    # ==================== 2. 加载地图数据 ====================
    with open('maps/town2.json', 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    all_quads = map_data['quads']
    all_w_lanes = map_data['w_lanes']
    quads_by_id = {q['poly_id']: q for q in all_quads}
    w_lanes_by_id = {w_lane['w_lane_id']: w_lane for w_lane in all_w_lanes}
    lane_groups = defaultdict(list)
    for w_lane in all_w_lanes:
        key = (w_lane['road_id'], w_lane['lane_id'])
        lane_groups[key].append(w_lane)
    
    # ==================== 3. 辅助函数：提取有效路径 ====================
    def extract_valid_path(path_np, planner):
        """从路径张量中提取有效的lane索引列表"""
        valid_path_indices = []
        for val in path_np:
            if val == planner.INVALID_PATH_MARKER:
                break
            if 0 <= val < planner.n_lanes:
                valid_path_indices.append(int(val))
            else:
                break
        return valid_path_indices
    
    # ==================== 4. 批量生成路径（使用GPU tensor） ====================
    B, M = 4800, 150
    total_paths = B * M
    print(f"\n开始批量生成 {B} x {M} = {total_paths} 条路径...")
    
    all_poly_ids = [quad['poly_id'] for quad in all_quads if quad['poly_id'] in planner.poly_id_to_lane_idx]
    
    # 随机生成起点和终点矩阵（使用GPU tensor）
    print("生成随机起点和终点 tensor...")
    start_poly_ids_list = []
    end_poly_ids_list = []
    
    for b in range(B):
        for m in range(M):
            start_poly_id = random.choice(all_poly_ids)
            end_poly_id = random.choice(all_poly_ids)
            while end_poly_id == start_poly_id:
                end_poly_id = random.choice(all_poly_ids)
            start_poly_ids_list.append(start_poly_id)
            end_poly_ids_list.append(end_poly_id)
    
    # 创建 (B, M) 形状的 tensor，直接批量查询
    start_poly_tensor = torch.tensor(start_poly_ids_list, dtype=torch.long).reshape(B, M)
    end_poly_tensor = torch.tensor(end_poly_ids_list, dtype=torch.long).reshape(B, M)
    
    print(f"批量查询 {total_paths} 条路径（GPU加速）...")
    all_paths = planner.path_plan(start_poly_tensor, end_poly_tensor)  # (B, M, max_path_len)
    print("路径生成完成！")
    
    # 不存储路径数据，直接从tensor中读取
    print(f"总共 {total_paths} 条路径（直接从tensor读取，不存储）")
    
    # ==================== 8. 辅助函数：按s值排序w_lanes ====================
    def get_sorted_w_lanes(lane_idx, lane_groups, quads_by_id, planner):
        """获取指定lane的所有w_lanes，按s值排序"""
        road_id, lane_id = planner.lane_keys[lane_idx]
        w_lanes_in_lane = lane_groups[(road_id, lane_id)]
        w_lanes_with_s = []
        for w_lane in w_lanes_in_lane:
            poly_id = w_lane['poly_id']
            quad = quads_by_id[poly_id]
            s = quad.get('s', 0.0)
            w_lanes_with_s.append((w_lane, s))
        w_lanes_with_s.sort(key=lambda x: x[1])
        return [w for w, s in w_lanes_with_s]
    
    # ==================== 9. 辅助函数：遍历w_lane_id链 ====================
    def traverse_w_lane_chain(start_w_lane_id, end_w_lane_id, w_lanes_by_id, quads_by_id, 
                              next_key='next_w_lane_id'):
        """
        遍历w_lane_id链，收集中间的waypoints
        next_key: 'next_w_lane_id' 或 'prev_w_lane_id'
        """
        if start_w_lane_id is None:
            return []
        
        waypoints = []
        current_w_lane_id = start_w_lane_id
        visited = set()
        
        while current_w_lane_id is not None and current_w_lane_id != end_w_lane_id:
            if current_w_lane_id in visited:
                break
            visited.add(current_w_lane_id)
            
            if current_w_lane_id not in w_lanes_by_id:
                break
            
            w_lane = w_lanes_by_id[current_w_lane_id]
            waypoints.append({
                'center': w_lane['center'],
                'direction_angle': w_lane.get('direction_angle', 0.0)
            })
            
            # 获取下一个w_lane_id
            quad_for_w_lane = quads_by_id.get(w_lane['poly_id'])
            if quad_for_w_lane and quad_for_w_lane.get(next_key) is not None:
                current_w_lane_id = quad_for_w_lane[next_key]
            else:
                break
        return waypoints
    
    # ==================== 5. 辅助函数：收集单个路径的waypoints ====================
    def collect_path_waypoints(valid_path, start_poly_id, end_poly_id, lane_groups, quads_by_id, 
                               w_lanes_by_id, planner):
        """收集单个路径的所有waypoints"""
        start_quad = quads_by_id[start_poly_id]
        end_quad = quads_by_id[end_poly_id]
        waypoints = []
        
        # 从start_quad遍历到第一条lane的end
        if len(valid_path) > 0:
            first_lane_w_lanes = get_sorted_w_lanes(valid_path[0], lane_groups, quads_by_id, planner)
            if first_lane_w_lanes:
                end_w_lane_id = first_lane_w_lanes[-1]['w_lane_id']
                start_waypoints = traverse_w_lane_chain(
                    start_quad.get('next_w_lane_id'), end_w_lane_id,
                    w_lanes_by_id, quads_by_id, 'next_w_lane_id'
                )
                waypoints.extend(start_waypoints)
        
        # 中间lanes的所有waypoints
        if len(valid_path) > 2:
            for lane_idx in valid_path[1:-1]:
                lane_w_lanes = get_sorted_w_lanes(lane_idx, lane_groups, quads_by_id, planner)
                for w_lane in lane_w_lanes:
                    waypoints.append({
                        'center': w_lane['center'],
                        'direction_angle': w_lane.get('direction_angle', 0.0)
                    })
        
        # 从end_quad遍历到最后一条lane的start
        if len(valid_path) > 1:
            last_lane_w_lanes = get_sorted_w_lanes(valid_path[-1], lane_groups, quads_by_id, planner)
            if last_lane_w_lanes:
                start_w_lane_id = last_lane_w_lanes[0]['w_lane_id']
                end_waypoints = traverse_w_lane_chain(
                    end_quad.get('prev_w_lane_id'), start_w_lane_id,
                    w_lanes_by_id, quads_by_id, 'prev_w_lane_id'
                )
                waypoints.extend(end_waypoints)
        
        # 特殊情况：只有一条lane
        elif len(valid_path) == 1:
            lane_w_lanes = get_sorted_w_lanes(valid_path[0], lane_groups, quads_by_id, planner)
            if lane_w_lanes:
                start_w_lane_id = lane_w_lanes[0]['w_lane_id']
                end_w_lane_id = lane_w_lanes[-1]['w_lane_id']
                
                start_waypoints = traverse_w_lane_chain(
                    start_quad.get('next_w_lane_id'), end_w_lane_id,
                    w_lanes_by_id, quads_by_id, 'next_w_lane_id'
                )
                end_waypoints = traverse_w_lane_chain(
                    end_quad.get('prev_w_lane_id'), start_w_lane_id,
                    w_lanes_by_id, quads_by_id, 'prev_w_lane_id'
                )
                waypoints = start_waypoints + end_waypoints
        
        return waypoints, start_quad, end_quad
    
    # ==================== 6. 交互式可视化（直接从tensor读取） ====================
    print(f"\n开始交互式可视化 ({total_paths} 条路径)...")
    print("提示: 按空格键查看下一张路径，关闭窗口退出")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        fig.canvas.manager.set_window_title('Navigation Path Viewer - Press SPACE for next')
    except:
        pass  # 某些后端可能不支持set_window_title
    
    # 当前显示的路径索引（使用列表以便在闭包中修改）
    current_idx = [0]
    
    def draw_path(idx):
        """绘制指定索引的路径（直接从tensor读取）"""
        ax.clear()
        
        # 绘制地图背景
        for quad in all_quads:
            vertices = quad['vertices']
            vertices_2d = [(v[0], v[1]) for v in vertices]
            polygon = Polygon(vertices_2d, closed=True,
                         facecolor='lightgray', edgecolor='gray',
                         alpha=0.1, linewidth=0.1)
            ax.add_patch(polygon)
        
        # 计算对应的(b, m)索引
        if idx >= total_paths:
            idx = idx % total_paths  # 循环处理
        
        b = idx // M
        m = idx % M
        
        # 从tensor中直接提取路径数据
        path_np = all_paths[b, m, :].cpu().numpy()
        valid_path = extract_valid_path(path_np, planner)
        
        # 获取对应的poly_id
        start_poly_id = start_poly_ids_list[idx]
        end_poly_id = end_poly_ids_list[idx]
        
        # 如果路径无效，显示提示
        if len(valid_path) == 0:
            ax.text(0.5, 0.5, f'Invalid Path\nB={b}, M={m}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'Path {idx+1} / {total_paths} | Invalid Path')
            ax.set_aspect('equal')
            plt.tight_layout()
            return
        
        # 收集waypoints
        waypoints, start_quad, end_quad = collect_path_waypoints(
            valid_path, start_poly_id, end_poly_id, 
            lane_groups, quads_by_id, w_lanes_by_id, planner
        )
        
        # 绘制waypoints箭头
        if len(waypoints) > 0:
            arrow_length = 10.0
            for i, wp in enumerate(waypoints):
                center = wp['center']
                direction_angle = wp['direction_angle']
                x, y = center[0], center[1]
                dx = arrow_length * np.cos(direction_angle)
                dy = arrow_length * np.sin(direction_angle)
                ax.arrow(x, y, dx, dy,
                        head_width=2, head_length=3,
                        fc='purple', ec='purple', alpha=0.7,
                        length_includes_head=True, zorder=3, 
                        label='Path Waypoints' if i == 0 else '')
        
        # 绘制起点和终点polygon
        start_vertices = start_quad['vertices']
        start_vertices_2d = [(v[0], v[1]) for v in start_vertices]
        start_polygon = Polygon(start_vertices_2d, closed=True,
                               facecolor='green', edgecolor='darkgreen',
                               alpha=0.5, linewidth=2, label='Start', zorder=6)
        ax.add_patch(start_polygon)
        
        end_vertices = end_quad['vertices']
        end_vertices_2d = [(v[0], v[1]) for v in end_vertices]
        end_polygon = Polygon(end_vertices_2d, closed=True,
                             facecolor='red', edgecolor='darkred',
                             alpha=0.5, linewidth=2, label='Target', zorder=6)
        ax.add_patch(end_polygon)
        
        # 绘制起点和终点waypoint标记
        start_lane_idx = planner.poly_id_to_lane_idx[start_poly_id]
        target_lane_idx = planner.poly_id_to_lane_idx[end_poly_id]
        start_pos = planner.start_positions.cpu().numpy()
        end_pos = planner.end_positions.cpu().numpy()
        ax.scatter(end_pos[start_lane_idx, 0], end_pos[start_lane_idx, 1], 
                  c='green', s=10, alpha=0.9, label='Start Waypoint', zorder=2, 
                  edgecolors='black', linewidth=2, marker='o')
        ax.scatter(start_pos[target_lane_idx, 0], start_pos[target_lane_idx, 1], 
                  c='red', s=10, alpha=0.9, label='Target Waypoint', zorder=2, 
                  edgecolors='black', linewidth=2, marker='o')
        
        # 设置图表属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Path {idx+1} / {total_paths} (B={b}, M={m}) | '
                    f'Start: poly_id {start_poly_id} | End: poly_id {end_poly_id} | '
                    f'Lanes: {len(valid_path)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
    
    def on_key_press(event):
        """键盘事件处理"""
        if event.key == ' ':  # 空格键
            current_idx[0] += 1
            if current_idx[0] >= total_paths:
                current_idx[0] = 0  # 循环回到开头
            draw_path(current_idx[0])
            fig.canvas.draw()
    
    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # 绘制第一张路径
    draw_path(current_idx[0])
    
    print(f"显示路径 1 / {total_paths} (按空格键查看下一张)")
    plt.show()
    
    
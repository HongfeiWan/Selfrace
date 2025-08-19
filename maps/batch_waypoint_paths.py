import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import time

class WaypointGraphGPU:
    """
    将 waypoint_graph 预处理为 GPU 常驻的稀疏入边表，并提供固定长度的批量最短路径生成功能。
    """
    def __init__(self, json_path: str, device: Optional[str] = None):
        # 1) 读取 JSON 并解析 waypoint_graph
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'waypoint_graph' not in data:
            raise ValueError(f"文件中未找到 'waypoint_graph': {json_path}")
        wg = data['waypoint_graph']
        raw_nodes: List[List[Any]] = wg['nodes']
        raw_edges: List[List[Any]] = wg['edges']

        # 2) 向量化解析 nodes
        nodes_arr = np.array(raw_nodes, dtype=object)  # [N,6]
        N = int(nodes_arr.shape[0])
        cross_col = nodes_arr[:, 0].astype(np.int64)
        road_col = nodes_arr[:, 1].astype(np.int64)
        lane_col = nodes_arr[:, 2].astype(np.int64)
        x_col = nodes_arr[:, 3].astype(np.float64)
        y_col = nodes_arr[:, 4].astype(np.float64)
        type_col = nodes_arr[:, 5].astype(str)

        # 三元组张量 [N,3]
        node_triplets_np = np.stack([cross_col, road_col, lane_col], axis=1)
        # 构造节点唯一键（字符串），用于与边端点对齐
        # 使用固定小数格式保证稳定性
        x_str = np.char.mod('%.6f', x_col)
        y_str = np.char.mod('%.6f', y_col)
        key_nodes = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        cross_col.astype(str), '|'), road_col.astype(str)), '|'), lane_col.astype(str)), '|')
        key_nodes = np.char.add(np.char.add(np.char.add(np.char.add(key_nodes, x_str), '|'), y_str), '|')
        key_nodes = np.char.add(key_nodes, type_col)

        # 3) 向量化解析 edges
        edges_arr = np.array(raw_edges, dtype=object)  # [E,3]
        # 提取 u/v 节点矩阵 [E,6]
        u_arr = np.stack(edges_arr[:, 0])
        v_arr = np.stack(edges_arr[:, 1])
        w_arr = edges_arr[:, 2].astype(np.float32)
        # 为 u/v 节点生成键
        u_key = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        u_arr[:, 0].astype(str), '|'), u_arr[:, 1].astype(str)), '|'), u_arr[:, 2].astype(str)), '|')
        u_x_str = np.char.mod('%.6f', u_arr[:, 3].astype(np.float64))
        u_y_str = np.char.mod('%.6f', u_arr[:, 4].astype(np.float64))
        u_key = np.char.add(np.char.add(np.char.add(np.char.add(u_key, u_x_str), '|'), u_y_str), '|')
        u_key = np.char.add(u_key, u_arr[:, 5].astype(str))
        v_key = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        v_arr[:, 0].astype(str), '|'), v_arr[:, 1].astype(str)), '|'), v_arr[:, 2].astype(str)), '|')
        v_x_str = np.char.mod('%.6f', v_arr[:, 3].astype(np.float64))
        v_y_str = np.char.mod('%.6f', v_arr[:, 4].astype(np.float64))
        v_key = np.char.add(np.char.add(np.char.add(np.char.add(v_key, v_x_str), '|'), v_y_str), '|')
        v_key = np.char.add(v_key, v_arr[:, 5].astype(str))

        # 4) 用排序 + searchsorted 将 u/v 键映射到节点索引（避免 Python 字典循环）
        order_nodes = np.argsort(key_nodes)
        key_nodes_sorted = key_nodes[order_nodes]
        pos_u = np.searchsorted(key_nodes_sorted, u_key)
        pos_v = np.searchsorted(key_nodes_sorted, v_key)
        u_idx = order_nodes[pos_u].astype(np.int64)
        v_idx = order_nodes[pos_v].astype(np.int64)

        # 5) 基于 v_idx 分组构建稠密入边表 [N, max_in_deg]
        deg = np.bincount(v_idx, minlength=N)
        max_in_deg = int(deg.max()) if N > 0 else 1
        if max_in_deg == 0:
            max_in_deg = 1
        # 无效位置用0占位，配合 inf 权重在松弛时不会被选中
        incoming_src_idx_np = np.full((N, max_in_deg), 0, dtype=np.int64)
        incoming_w_np = np.full((N, max_in_deg), np.inf, dtype=np.float32)
        if max_in_deg > 0 and v_idx.size > 0:
            order_e = np.argsort(v_idx, kind='stable')
            v_sorted = v_idx[order_e]
            u_sorted = u_idx[order_e]
            w_sorted = w_arr[order_e]
            # 每个节点的起始偏移（按排序后）
            starts = np.cumsum(np.concatenate(([0], deg[:-1])))
            # 逐节点切片填充（这个环节循环的是节点数而非边数，且纯切片赋值，已较轻）
            nz_nodes = np.nonzero(deg)[0]
            for v in nz_nodes:
                s = starts[v]
                d = deg[v]
                incoming_src_idx_np[v, :d] = u_sorted[s:s + d]
                incoming_w_np[v, :d] = w_sorted[s:s + d]

        # 3) 迁移到设备并保存
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.incoming_src_idx = torch.from_numpy(incoming_src_idx_np).to(device)  # [N, M]
        self.incoming_w = torch.from_numpy(incoming_w_np).to(device)              # [N, M]
        self.node_triplets = torch.from_numpy(node_triplets_np.astype(np.int64)).to(device)

        # 同步构建出边表 [N, max_out_deg]（用于终点反向DP+前向next跳转）
        deg_out = np.bincount(u_idx, minlength=N)
        max_out_deg = int(deg_out.max()) if N > 0 else 1
        if max_out_deg == 0:
            max_out_deg = 1
        out_tgt_idx_np = np.full((N, max_out_deg), 0, dtype=np.int64)
        out_w_np = np.full((N, max_out_deg), np.inf, dtype=np.float32)
        if max_out_deg > 0 and u_idx.size > 0:
            order_e2 = np.argsort(u_idx, kind='stable')
            u_sorted2 = u_idx[order_e2]
            v_sorted2 = v_idx[order_e2]
            w_sorted2 = w_arr[order_e2]
            starts_out = np.cumsum(np.concatenate(([0], deg_out[:-1])))
            nz_u = np.nonzero(deg_out)[0]
            for u in nz_u:
                s = starts_out[u]
                d = deg_out[u]
                out_tgt_idx_np[u, :d] = v_sorted2[s:s + d]
                out_w_np[u, :d] = w_sorted2[s:s + d]
        self.outgoing_tgt_idx = torch.from_numpy(out_tgt_idx_np).to(device)  # [N, M_out]
        self.outgoing_w = torch.from_numpy(out_w_np).to(device)              # [N, M_out]


        # 4) 纯GPU三元组分组结构（唯一键、分组起始/大小、排序后的节点索引）
        mins = self.node_triplets.min(dim=0).values.to(torch.int64)
        maxs = self.node_triplets.max(dim=0).values.to(torch.int64)
        ranges = (maxs - mins + 1).to(torch.int64)
        mul_cross = (ranges[1] * ranges[2]).to(torch.int64)
        mul_road = ranges[2].to(torch.int64)
        off = (self.node_triplets - mins)
        keys_nodes = off[:, 0] * mul_cross + off[:, 1] * mul_road + off[:, 2]
        keys_sorted, order_nodes2 = torch.sort(keys_nodes)
        uniq_keys_t, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.int64), counts[:-1]]), dim=0)
        self._triplet_mins = mins
        self._triplet_mul_cross = mul_cross
        self._triplet_mul_road = mul_road
        self.triplet_unique_keys = uniq_keys_t
        self.triplet_group_starts = starts
        self.triplet_group_counts = counts
        self.nodes_sorted_by_triplet = order_nodes2
        
        # 缓存（offset位掩码、终点树）
        self._offset_masks_cache = {}
        
        # 预计算所有终点组的最短路径树，使用张量存储
        self._precompute_all_end_trees_tensor()

    def _precompute_all_end_trees_tensor(self):
        """为每个终点组预计算最短路径树，使用张量存储"""
        device = self.device
        U = self.triplet_unique_keys.numel()
        N = self.outgoing_tgt_idx.size(0)
        
        print(f"预计算 {U} 个终点组的最短路径树（张量存储）...")
        start_time = time.time()
        
        # 预分配张量存储所有终点组的结果
        # dist_tensor: [U, N] - 每个终点组到所有节点的最短距离
        # next_tensor: [U, N] - 每个终点组的最短路径下一跳
        dist_tensor = torch.full((U, N), float('inf'), dtype=torch.float32, device=device)
        next_tensor = torch.full((U, N), -1, dtype=torch.long, device=device)
        
        # 批量构建所有终点组的最短路径树
        for g in range(U):
            dist_g, next_g = self._build_end_tree(g)
            dist_tensor[g] = dist_g
            next_tensor[g] = next_g
        
        self.end_dist_tensor = dist_tensor  # [U, N]
        self.end_next_tensor = next_tensor  # [U, N]
        
        end_time = time.time()
        print(f"终点组预计算完成，耗时: {end_time - start_time:.4f}秒")

    def _build_end_tree(self, end_group_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """为给定终点组构建最短路径树"""
        device = self.device
        N = self.outgoing_tgt_idx.size(0)
        
        # 终点组内所有节点
        start = self.triplet_group_starts[end_group_id]
        count = self.triplet_group_counts[end_group_id]
        nodes = self.nodes_sorted_by_triplet[start:start + count]
        
        dist = torch.full((N,), float('inf'), dtype=torch.float32, device=device)
        next_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        if count > 0:
            dist.index_fill_(0, nodes, 0.0)
        
        # Bellman-Ford 松弛
        for _ in range(max(1, N - 1)):
            tgt_d = dist[self.outgoing_tgt_idx]           # [N, M_out]
            cand = tgt_d + self.outgoing_w                # [N, M_out]
            new_d, min_pos = torch.min(cand, dim=1)       # [N]
            improved = new_d < dist
            if not torch.any(improved):
                break
            dist = torch.minimum(dist, new_d)
            best_v = self.outgoing_tgt_idx.gather(1, min_pos.view(-1, 1)).squeeze(1)  # [N]
            next_idx[improved] = best_v[improved]
        
        return dist, next_idx

    def _get_offset_masks(self, L: int) -> torch.Tensor:
        """
        返回形状 [K, L] 的 bool 掩码，第 k 行表示 offset 中第 k 个比特是否为 1。
        缓存到 (L, device) 键。
        """
        device = self.device
        L = int(L)
        K = int(max(1, (L - 1).bit_length()))
        key = (L, device)
        cached = self._offset_masks_cache.get(key, None)
        if cached is not None and cached.shape == (K, L):
            return cached
        offsets = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)  # [1,L]
        bits = torch.arange(K, device=device, dtype=torch.long).unsqueeze(1)     # [K,1]
        use = ((offsets >> bits) & 1).to(torch.bool)                              # [K,L]
        self._offset_masks_cache[key] = use
        return use

    def batch_shortest_paths_fixed_len(self,
                                   start_ids: torch.Tensor,
                                   end_ids: torch.Tensor,
                                   fixed_len: int = 100,
                                   pad_value: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (paths, mask)
            paths: [B, L, 3] long，三元组 (cross, road, lane)，用 pad_value 填充
            mask:  [B, L] bool，True 表示该步有效
        Args:
            start_ids: [B, 3] long tensor，起点三元组 (cross, road, lane)
            end_ids: [B, 3] long tensor，终点三元组 (cross, road, lane)
        """
        if start_ids.shape != end_ids.shape:
            raise ValueError(f"start_ids 与 end_ids 形状必须一致，got {start_ids.shape} vs {end_ids.shape}")
        if start_ids.dim() != 2 or start_ids.size(1) != 3:
            raise ValueError(f"start_ids 必须是 [B, 3] 形状，got {start_ids.shape}")
        
        device = self.device
        N = self.outgoing_tgt_idx.size(0)
        U = self.end_dist_tensor.size(0)
        B = start_ids.size(0)
        L = int(fixed_len)
        
        # 确保张量在正确设备上
        start_tensor = start_ids.to(device=device, dtype=torch.long)
        end_tensor = end_ids.to(device=device, dtype=torch.long)
        
        # 输出张量
        triplets = torch.full((B, L, 3), pad_value, dtype=torch.long, device=device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        
        # 将三元组映射为组id
        s_off = (start_tensor - self._triplet_mins)
        s_keys = s_off[:, 0] * self._triplet_mul_cross + s_off[:, 1] * self._triplet_mul_road + s_off[:, 2]
        pos_s = torch.searchsorted(self.triplet_unique_keys, s_keys)
        pos_s = torch.clamp(pos_s, 0, U - 1)
        matched_s = self.triplet_unique_keys[pos_s] == s_keys  # [B]

        e_off = (end_tensor - self._triplet_mins)
        e_keys = e_off[:, 0] * self._triplet_mul_cross + e_off[:, 1] * self._triplet_mul_road + e_off[:, 2]
        pos_e = torch.searchsorted(self.triplet_unique_keys, e_keys)
        pos_e = torch.clamp(pos_e, 0, U - 1)
        matched_e = self.triplet_unique_keys[pos_e] == e_keys  # [B]

        # 预取 offsets 掩码
        use_masks = self._get_offset_masks(L)
        K = use_masks.shape[0]

        # 向量化处理所有批次
        if torch.any(matched_e):
            # 获取所有有效的终点组
            valid_b = torch.nonzero(matched_e, as_tuple=False).squeeze(1)
            end_groups = pos_e[valid_b]  # [valid_B]
            
            # 获取对应的距离和下一跳张量
            dist_groups = self.end_dist_tensor[end_groups]  # [valid_B, N]
            next_groups = self.end_next_tensor[end_groups]  # [valid_B, N]
            
            # 为每个批次选择起点：从其 start 组内选 dist_g 最小的节点
            start_pos_sel = pos_s[valid_b]  # [valid_B]
            start_matched_sel = matched_s[valid_b]  # [valid_B]
            
            # 初始化起点节点索引
            start_node_idx = torch.full((valid_b.numel(),), -1, dtype=torch.long, device=device)
            
            if torch.any(start_matched_sel):
                # 找到匹配的起点组
                matched_start_b = torch.nonzero(start_matched_sel, as_tuple=False).squeeze(1)  # [matched_B]
                matched_start_groups = start_pos_sel[matched_start_b]  # [matched_B]
                
                # 获取这些起点组的信息
                g_starts = self.triplet_group_starts[matched_start_groups]  # [matched_B]
                g_counts = self.triplet_group_counts[matched_start_groups]  # [matched_B]
                max_c = int(g_counts.max().item())
                
                if max_c > 0:
                    # 构建网格索引
                    ar = torch.arange(max_c, device=device, dtype=torch.int64)
                    grid = g_starts.unsqueeze(1) + ar.unsqueeze(0)  # [matched_B, max_c]
                    valid = ar.unsqueeze(0) < g_counts.unsqueeze(1)  # [matched_B, max_c]
                    
                    # 安全索引
                    grid_clamped = torch.clamp(grid, 0, self.nodes_sorted_by_triplet.numel() - 1)
                    nodes_mat = self.nodes_sorted_by_triplet[grid_clamped]  # [matched_B, max_c]
                    
                    # 获取对应的距离
                    dist_mat = dist_groups[matched_start_b.unsqueeze(1), nodes_mat]  # [matched_B, max_c]
                    dist_mat = dist_mat.masked_fill(~valid, float('inf'))
                    
                    # 找到最小距离的节点
                    min_vals, min_pos = torch.min(dist_mat, dim=1)  # [matched_B]
                    chosen = nodes_mat.gather(1, min_pos.unsqueeze(1)).squeeze(1)  # [matched_B]
                    ok = torch.isfinite(min_vals)
                    
                    # 更新起点节点索引
                    start_node_idx[matched_start_b[ok]] = chosen[ok]
            
            # 向量化二进制跳转
            valid_count = valid_b.numel()
            if valid_count > 0:
                # 准备跳转张量
                v = start_node_idx.view(valid_count, 1).expand(-1, L)  # [valid_B, L]
                cur = next_groups  # [valid_B, N]
                
                # 二进制跳转
                for bit in range(K):
                    use = use_masks[bit].view(1, L).expand(valid_count, -1)  # [valid_B, L]
                    if torch.any(use):
                        safe_v = torch.clamp(v, 0, N - 1)
                        jumped = cur.gather(1, safe_v)  # [valid_B, L]
                        jumped = torch.where(v >= 0, jumped, torch.full_like(v, -1))
                        v = torch.where(use, jumped, v)
                    
                    # 自合成：cur = cur ∘ cur
                    last = cur
                    safe_idx = torch.clamp(last, 0, N - 1)
                    nxt = last.gather(1, safe_idx)
                    cur = torch.where(last >= 0, nxt, torch.full_like(last, -1))
                
                # 构建路径
                path_idx = v  # [valid_B, L] - 已是从起点到终点方向
                local_mask = path_idx >= 0  # [valid_B, L]
                
                # 左对齐（保持列内顺序）
                if torch.any(local_mask):
                    col_idx = torch.arange(L, device=device).view(1, L).expand(valid_count, -1)
                    order_score = (~local_mask).to(torch.long) * L + col_idx
                    order = torch.argsort(order_score, dim=1, stable=True)
                    path_idx = path_idx.gather(1, order)
                    local_mask = local_mask.gather(1, order)
    
                # 映射到三元组
                tris = torch.full((valid_count, L, 3), pad_value, dtype=torch.long, device=device)
                if torch.any(local_mask):
                    flat_idx = path_idx[local_mask]
                    tris_flat = self.node_triplets.index_select(0, flat_idx)
                    tris[local_mask] = tris_flat
                # 回填到全局输出
                triplets[valid_b] = tris
                mask[valid_b] = local_mask

        return triplets, mask

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='随机批量测试 WaypointGraphGPU 寻路')
    parser.add_argument('--graph', type=str, default='maps/cross_data_processed_map_Town01_stitched.json', help='包含 waypoint_graph 的 cross_data_* JSON 路径')
    parser.add_argument('--num_pairs', type=int, default=360000, help='随机生成的 (start,end) 对数')
    parser.add_argument('--fixed_len', type=int, default=100, help='固定路径长度 L')
    parser.add_argument('--device', type=str, default=None, help='cuda 或 cpu，默认自动')
    args = parser.parse_args()
    
    # 构建图（驻留 GPU）
    graph = WaypointGraphGPU(args.graph, device=args.device)
    # 基于 GPU 的 triplet 唯一键随机采样与解码
    U = int(graph.triplet_unique_keys.numel())
    if U == 0:
        raise RuntimeError('waypoint_graph 中没有可用的 (cross, road, lane) 唯一键')
    device = graph.device
    idx_start = torch.randint(low=0, high=U, size=(args.num_pairs,), device=device, dtype=torch.int64)
    idx_end = torch.randint(low=0, high=U, size=(args.num_pairs,), device=device, dtype=torch.int64)
    keys_start = graph.triplet_unique_keys[idx_start]
    keys_end = graph.triplet_unique_keys[idx_end]

    def decode_keys(keys: torch.Tensor) -> torch.Tensor:
        mul_cross = graph._triplet_mul_cross
        mul_road = graph._triplet_mul_road
        mins = graph._triplet_mins
        off_cross = keys // mul_cross
        rem = keys % mul_cross
        off_road = rem // mul_road
        off_lane = rem % mul_road
        return torch.stack([off_cross + mins[0], off_road + mins[1], off_lane + mins[2]], dim=1).to(torch.long)

    start_ids = decode_keys(keys_start)  # [B,3] long (device)
    end_ids = decode_keys(keys_end)      # [B,3] long (device)

    # 批量寻路（现在直接传入张量）
    start_time = time.time()
    paths, mask = graph.batch_shortest_paths_fixed_len(
        start_ids=start_ids,
        end_ids=end_ids,
        fixed_len=args.fixed_len,
    )
    end_time = time.time()
    print(f"batch_shortest_paths_fixed_len 耗时: {end_time - start_time:.4f}秒")
    
    # 统计可达 vs 不可达（以最后一个有效步存在为准）
    valid_len_vec = mask.sum(dim=1)
    num_reach = (valid_len_vec > 0).sum()
    total = mask.shape[0]
    print(f'Total pairs: {total}, Reachable: {int(num_reach.item())}, Unreachable: {total - int(num_reach.item())}')

    # 打印前3条路径示例
    show_k = min(3, total)
    for i in range(show_k):
        valid_len = int(mask[i].sum().item())
        seq = paths[i, :valid_len]
        print(f'Pair {i}: start={start_ids[i]}, end={end_ids[i]}, length={valid_len}')
        print(f'  path={seq}')
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

    def batch_shortest_paths_fixed_len(self,
                                   start_ids: torch.Tensor,
                                   end_ids: torch.Tensor,
                                   fixed_len: int = 100,
                                   max_iter: Optional[int] = None,
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
        N, M = self.incoming_src_idx.shape
        B = start_ids.size(0)
        
        # 确保张量在正确设备上
        start_tensor = start_ids.to(device=device, dtype=torch.long)
        end_tensor = end_ids.to(device=device, dtype=torch.long)
        
        dist = torch.full((B, N), float('inf'), dtype=torch.float32, device=device)
        prev = torch.full((B, N), -1, dtype=torch.long, device=device)
        
        # 批量映射 start_ids -> 节点集合
        s_off = (start_tensor - self._triplet_mins)
        s_keys = s_off[:, 0] * self._triplet_mul_cross + s_off[:, 1] * self._triplet_mul_road + s_off[:, 2]
        pos = torch.searchsorted(self.triplet_unique_keys, s_keys)
        pos = torch.clamp(pos, 0, self.triplet_unique_keys.numel() - 1)
        matched = self.triplet_unique_keys[pos] == s_keys
        if torch.any(matched):
            b_idx = torch.nonzero(matched, as_tuple=False).squeeze(1)
            g_starts = self.triplet_group_starts[pos[b_idx]]
            g_counts = self.triplet_group_counts[pos[b_idx]]
            max_c = int(g_counts.max().item())
            if max_c > 0:
                ar = torch.arange(max_c, device=device, dtype=torch.int64)
                grid = g_starts.unsqueeze(1) + ar.unsqueeze(0)
                valid = ar.unsqueeze(0) < g_counts.unsqueeze(1)
                # 避免非法索引：先将 grid clamp 到合法范围，再结合 valid 掩码使用
                grid_clamped = torch.clamp(grid, 0, self.nodes_sorted_by_triplet.numel() - 1)
                nodes_mat = self.nodes_sorted_by_triplet[grid_clamped]
                node_idx_flat = nodes_mat[valid]
                b_grid = b_idx.unsqueeze(1).expand(-1, max_c)
                b_flat = b_grid[valid]
                dist[b_flat, node_idx_flat] = 0.0

        if max_iter is None:
            max_iter = max(1, N - 1)
        gather_index = self.incoming_src_idx.unsqueeze(0).expand(B, -1, -1)
        w_broadcast = self.incoming_w.unsqueeze(0).expand(B, -1, -1)

        loop1_time = time.time()
        for _ in range(max_iter):
            # dist: [B, N]; gather 索引为 [B, N, M]，先在 dim=1 上扩展为 [B, N, N]
            dist_expanded = dist.unsqueeze(1).expand(B, N, -1)
            src_dists = dist_expanded.gather(2, gather_index)     # [B, N, M]
            cand = src_dists + w_broadcast                        # [B, N, M]
            new_dist_v, min_src_j = torch.min(cand, dim=2)        # [B, N]
            improved = new_dist_v < dist
            if not torch.any(improved):
                break
            # 更新距离
            dist = torch.minimum(dist, new_dist_v)
            # 计算该最小候选对应的真实 src 索引
            chosen_src = self.incoming_src_idx.unsqueeze(0).expand(B, -1, -1)
            chosen_src = chosen_src.gather(2, min_src_j.unsqueeze(2)).squeeze(2)  # [B, N]
            # 仅在改进处更新 prev
            prev[improved] = chosen_src[improved]
        
        loop1_time = time.time() - loop1_time
        print(f"loop1_time: {loop1_time:.4f}秒")

        # 选择每个 b 的终点节点（在 end_ids 对应集合中距离最小者）
        end_node_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        e_off = (end_tensor - self._triplet_mins)
        e_keys = e_off[:, 0] * self._triplet_mul_cross + e_off[:, 1] * self._triplet_mul_road + e_off[:, 2]
        pos_e = torch.searchsorted(self.triplet_unique_keys, e_keys)
        pos_e = torch.clamp(pos_e, 0, self.triplet_unique_keys.numel() - 1)
        matched_e = self.triplet_unique_keys[pos_e] == e_keys
        if torch.any(matched_e):
            b_idx_e = torch.nonzero(matched_e, as_tuple=False).squeeze(1)
            g_starts_e = self.triplet_group_starts[pos_e[b_idx_e]]
            g_counts_e = self.triplet_group_counts[pos_e[b_idx_e]]
            max_c_e = int(g_counts_e.max().item())
            if max_c_e > 0:
                ar = torch.arange(max_c_e, device=device, dtype=torch.int64)
                grid = g_starts_e.unsqueeze(1) + ar.unsqueeze(0)
                valid = ar.unsqueeze(0) < g_counts_e.unsqueeze(1)
                grid_clamped = torch.clamp(grid, 0, self.nodes_sorted_by_triplet.numel() - 1)
                nodes_mat = self.nodes_sorted_by_triplet[grid_clamped]
                d_mat = dist[b_idx_e.unsqueeze(1), nodes_mat]
                d_mat = d_mat.masked_fill(~valid, float('inf'))
                min_vals, min_pos = torch.min(d_mat, dim=1)
                chosen = nodes_mat.gather(1, min_pos.unsqueeze(1)).squeeze(1)
                ok = torch.isfinite(min_vals)
                end_node_idx[b_idx_e[ok]] = chosen[ok]

        # 回溯固定步长 L：用二进制 lifting 去掉逐步回溯循环
        L = int(fixed_len)
        K = int(max(1, (L - 1).bit_length()))

        loop2_time = time.time()
        # 构建 jump 表: ancestors[k] 表示沿 prev 前进 2^k 步的映射
        ancestors = [prev]
        for _k in range(1, K):
            last = ancestors[-1]
            safe_idx = torch.clamp(last, 0, N - 1)
            nxt = last.gather(1, safe_idx)
            nxt = torch.where(last >= 0, nxt, torch.full_like(last, -1))
            ancestors.append(nxt)
        loop2_time = time.time() - loop2_time
        print(f"loop2_time: {loop2_time:.4f}秒")

        loop3_time = time.time()
        # 为每个列偏移 0..L-1 计算对应的祖先，向量化处理
        offsets = torch.arange(L, device=device, dtype=torch.long).view(1, L).expand(B, -1)  # [B,L]
        v = end_node_idx.view(B, 1).expand(-1, L)  # [B,L]
        for bit in range(K):
            step = 1 << bit
            use = ((offsets >> bit) & 1).to(torch.bool)
            if torch.any(use):
                safe_v = torch.clamp(v, 0, N - 1)
                jumped = ancestors[bit].gather(1, safe_v)
                jumped = torch.where(v >= 0, jumped, torch.full_like(v, -1))
                v = torch.where(use, jumped, v)
        loop3_time = time.time() - loop3_time
        print(f"loop3_time: {loop3_time:.4f}秒")

        path_idx = v  # [B,L] 从终点回溯 offset 步的节点索引（倒序）
        # 源->终点顺序与有效掩码
        path_idx = torch.flip(path_idx, dims=[1])
        mask = path_idx >= 0
        # 将有效位置左对齐（保持列内顺序稳定）
        if torch.any(mask):
            col_idx = torch.arange(L, device=device).view(1, L).expand(B, -1)
            order_score = (~mask).to(torch.long) * L + col_idx
            order = torch.argsort(order_score, dim=1, stable=True)
            path_idx = path_idx.gather(1, order)
            mask = mask.gather(1, order)
        # 将节点索引映射为 (cross, road, lane)
        triplets = torch.full((B, L, 3), pad_value, dtype=torch.long, device=device)
        valid_mask = mask
        if torch.any(valid_mask):
            r = valid_mask
            # 扁平 gather
            flat_idx = path_idx[r]
            triplets_flat = self.node_triplets.index_select(0, flat_idx)
            triplets[r] = triplets_flat
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
        max_iter=None,
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
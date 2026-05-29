import torch
from typing import Tuple, Optional, Dict
import time

class SpatialHash:
    """
    一个统一的、基于GPU加速的批量化空间哈希，用于高效查询几何图元。
    该类支持两种核心功能：
    1. 索引和查询静态几何体（如路面多边形）。
    2. 对动态物体（如智能体）执行高效的临近对查询。
    """
    def __init__(self, cell_size: float, min_bounds: torch.Tensor, max_bounds: torch.Tensor, device: torch.device):
        """
        初始化 SpatialHash 网格。
        Args:
            cell_size (float): 每个网格单元的宽度和高度。
            min_bounds (torch.Tensor): 网格左下角的 (x, y) 坐标。
            max_bounds (torch.Tensor): 网格右上角的 (x, y) 坐标。
            device (torch.device): 用于存储张量的设备。
        """
        self.cell_size = cell_size
        self.min_bounds = min_bounds.to(device)
        self.max_bounds = max_bounds.to(device)
        self.device = device
        grid_dim = torch.ceil((self.max_bounds - self.min_bounds) / self.cell_size).long()
        self.grid_size = torch.max(grid_dim, torch.tensor([1, 1], device=device, dtype=torch.long))
        self.grid_total_cells = self.grid_size[0] * self.grid_size[1]
        # 用于静态几何体索引的属性
        self.static_sorted_items = torch.empty((0,), dtype=torch.long, device=self.device)
        self.static_cell_starts = torch.empty((0,), dtype=torch.long, device=self.device)
        self.static_max_candidates_per_cell = 0
        #print(f"SpatialHash initialized with grid {self.grid_size.cpu().numpy()} and cell size {self.cell_size:.2f}m.")

    def get_cell_idx(self, points: torch.Tensor) -> torch.Tensor:
        """将世界坐标批量转换为网格单元索引。"""
        indices = torch.floor((points - self.min_bounds) / self.cell_size).long()
        indices[:, 0].clamp_(0, self.grid_size[0] - 1)
        indices[:, 1].clamp_(0, self.grid_size[1] - 1)
        return indices

    def build_static_index(self, static_items_bounds: torch.Tensor):
        """
        为静态几何体（如多边形）构建一个持久化的哈希索引。
        Args:
            static_items_bounds (torch.Tensor): 静态物体的AABB，形状 (num_items, 2, 2) for (min, max).
        """
        num_items = static_items_bounds.shape[0]
        if num_items == 0:
            self.static_cell_starts = torch.zeros(self.grid_total_cells + 1, dtype=torch.long, device=self.device)
            self.static_max_candidates_per_cell = 0
            return

        item_min_bounds = static_items_bounds[:, 0]
        item_max_bounds = static_items_bounds[:, 1]
        
        start_cells = self.get_cell_idx(item_min_bounds)
        end_cells = self.get_cell_idx(item_max_bounds)

        item_ids = torch.arange(num_items, device=self.device)
        all_pairs = []
        # Этот цикл выполняется на CPU, но только один раз при инициализации
        for i in range(num_items):
            for x in range(start_cells[i, 0].item(), end_cells[i, 0].item() + 1):
                for y in range(start_cells[i, 1].item(), end_cells[i, 1].item() + 1):
                    cell_idx_flat = x * self.grid_size[1] + y
                    all_pairs.append([item_ids[i], cell_idx_flat])
        
        if not all_pairs:
            item_cell_pairs = torch.empty((0, 2), dtype=torch.long, device=self.device)
        else:
            item_cell_pairs = torch.tensor(all_pairs, dtype=torch.long, device=self.device)

        if item_cell_pairs.numel() == 0:
            self.static_sorted_items = torch.empty(0, dtype=torch.long, device=self.device)
            self.static_cell_starts = torch.zeros(self.grid_total_cells + 1, dtype=torch.long, device=self.device)
            self.static_max_candidates_per_cell = 0
            return

        sorted_pairs = item_cell_pairs[item_cell_pairs[:, 1].argsort()]
        self.static_sorted_items = sorted_pairs[:, 0].contiguous()
        self.static_cell_starts = torch.zeros(self.grid_total_cells + 1, dtype=torch.long, device=self.device)
        unique_cells, counts = torch.unique_consecutive(sorted_pairs[:, 1], return_counts=True)
        self.static_cell_starts[unique_cells + 1] = counts
        self.static_cell_starts = self.static_cell_starts.cumsum_(0)
        self.static_max_candidates_per_cell = int(counts.max().item()) if counts.numel() > 0 else 0
        #print(f"Built static index for {num_items} items.")

    def query_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        批量查询点，返回每个点对应的候选静态物体ID。
        """
        if self.static_sorted_items.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        cell_indices_2d = self.get_cell_idx(points)
        cell_indices_flat = cell_indices_2d[:, 0] * self.grid_size[1] + cell_indices_2d[:, 1]
        starts = self.static_cell_starts[cell_indices_flat]
        ends = self.static_cell_starts[cell_indices_flat + 1]
        num_candidates_per_point = ends - starts
        max_candidates = self.static_max_candidates_per_cell
        if max_candidates <= 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        point_indices_out = torch.arange(len(points), device=self.device).repeat_interleave(num_candidates_per_point)
        # GPU加速版本：避免for循环
        # 使用高级索引操作替代torch.cat和for循环
        # 创建索引偏移矩阵
        offsets = torch.arange(max_candidates, device=self.device).unsqueeze(0)  # (1, max_candidates)
        # 扩展starts以匹配最大候选数
        starts_expanded = starts.unsqueeze(1) + offsets  # (num_points, max_candidates)
        # 创建有效掩码
        valid_mask = offsets < num_candidates_per_point.unsqueeze(1)  # (num_points, max_candidates)
        # 应用掩码并展平
        valid_starts = starts_expanded[valid_mask]
        # 使用高级索引获取item_indices
        item_indices_out = self.static_sorted_items[valid_starts]

        return torch.stack([point_indices_out, item_indices_out], dim=1)

    def query_dynamic_pairs(self, B: int, M: int, active_mask: torch.Tensor,
                            verts_t0: torch.Tensor, verts_t1: torch.Tensor,
                            max_neighbors: int, debug: bool = False, debug_env_idx: int = 0) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        为一批动态物体执行高效的临近对查询。
        此方法使用稀疏矩阵乘法，非常适合于智能体间的碰撞检测宽阶段。
        优化版本：使用批处理稀疏矩阵操作减少循环。
        """
        all_verts = torch.cat([verts_t0, verts_t1], dim=-2)
        min_coords, _ = torch.min(all_verts, dim=-2)
        max_coords, _ = torch.max(all_verts, dim=-2)
        
        min_grid_idx = self.get_cell_idx(min_coords.view(-1, 2)).view(B, M, 2)
        max_grid_idx = self.get_cell_idx(max_coords.view(-1, 2)).view(B, M, 2)
        
        span_x = (max_grid_idx[..., 0] - min_grid_idx[..., 0] + 1).clamp(min=1)
        span_y = (max_grid_idx[..., 1] - min_grid_idx[..., 1] + 1).clamp(min=1)
        max_cells_per_agent = (span_x * span_y).max().item()

        ix_x = torch.arange(max_cells_per_agent, device=self.device) % span_x.view(-1, 1)
        ix_y = torch.arange(max_cells_per_agent, device=self.device) // span_x.view(-1, 1)

        agent_grid_x = min_grid_idx[..., 0].view(-1, 1) + ix_x
        agent_grid_y = min_grid_idx[..., 1].view(-1, 1) + ix_y

        valid_cell_mask = (ix_y < span_y.view(-1, 1)) & active_mask.view(-1, 1)
        
        cell_ids = agent_grid_x * self.grid_size[1] + agent_grid_y

        valid_flat_mask = valid_cell_mask.flatten()
        
        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(-1, M).flatten().unsqueeze(-1).expand(-1, max_cells_per_agent).flatten()[valid_flat_mask]
        agent_indices = torch.arange(M, device=self.device).view(1, M).expand(B, -1).flatten().unsqueeze(-1).expand(-1, max_cells_per_agent).flatten()[valid_flat_mask]
        cell_indices = cell_ids.flatten()[valid_flat_mask]

        debug_info = None
        if debug:
            env_mask = (batch_indices == debug_env_idx)
            occupied_cell_indices = cell_indices[env_mask]
            debug_info = {'occupied_cell_ids': torch.unique(occupied_cell_indices)}

        # 向量化分组与配对：避免 per-batch 循环和巨大矩阵
        num_neighbors = min(max_neighbors, M - 1)
        if num_neighbors <= 0:
            return torch.full((B, M, 0), -1, dtype=torch.long, device=self.device), debug_info

        # 1) 将占用条目按 (batch, cell) 分组
        group_keys = batch_indices.long() * self.grid_total_cells + cell_indices.long()
        sort_idx = torch.argsort(group_keys)
        sorted_keys = group_keys[sort_idx]
        sorted_batches = batch_indices[sort_idx]
        sorted_agents = agent_indices[sort_idx]

        if sorted_keys.numel() == 0:
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        all_keys, all_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        valid_groups_mask = all_counts >= 2
        if not valid_groups_mask.any():
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        group_counts = all_counts[valid_groups_mask]
        group_starts_all = torch.cumsum(torch.nn.functional.pad(all_counts, (1, 0)), dim=0)[:-1]
        group_starts = group_starts_all[valid_groups_mask]

        # 2) 为每个分组生成无序对 (i<j)，完全向量化
        max_group_size = int(group_counts.max().item())
        tri = torch.triu_indices(max_group_size, max_group_size, offset=1, device=self.device)
        tri_i, tri_j = tri[0], tri[1]
        valid_pair_mask = (tri_i.unsqueeze(0) < group_counts.unsqueeze(1)) & (tri_j.unsqueeze(0) < group_counts.unsqueeze(1))
        if not valid_pair_mask.any():
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        pos_i = group_starts.unsqueeze(1) + tri_i.unsqueeze(0)
        pos_j = group_starts.unsqueeze(1) + tri_j.unsqueeze(0)
        flat_mask = valid_pair_mask.flatten()
        flat_i = pos_i.flatten()[flat_mask]
        flat_j = pos_j.flatten()[flat_mask]

        pair_batches = sorted_batches[flat_i]
        agents_i = sorted_agents[flat_i]
        agents_j = sorted_agents[flat_j]

        # 3) 对对儿进行归一化并去重（同一对可出现在多个cell）
        a_min = torch.minimum(agents_i, agents_j)
        a_max = torch.maximum(agents_i, agents_j)
        pair_key = pair_batches.long() * (M * M) + a_min.long() * M + a_max.long()
        # 旧版PyTorch不支持 return_index，改为排序+unique_consecutive 获取首个索引
        perm_pairs = torch.argsort(pair_key)
        sorted_pair_key = pair_key[perm_pairs]
        _, counts_pairs = torch.unique_consecutive(sorted_pair_key, return_counts=True)
        starts_pairs = torch.cumsum(torch.nn.functional.pad(counts_pairs, (1, 0)), dim=0)[:-1]
        unique_idx_sorted = perm_pairs[starts_pairs]
        pair_batches = pair_batches[unique_idx_sorted]
        a_min = a_min[unique_idx_sorted]
        a_max = a_max[unique_idx_sorted]

        # 4) 生成有向邻接 (src->dst 和 dst->src)
        neigh_batches = torch.cat([pair_batches, pair_batches], dim=0)
        src_agents = torch.cat([a_min, a_max], dim=0)
        dst_agents = torch.cat([a_max, a_min], dim=0)

        # 5) 对每个 (batch, src) 选择前K个（按 dst 升序保证确定性）
        group_id = neigh_batches.long() * M + src_agents.long()
        lex_key = group_id * M + dst_agents.long()
        perm = torch.argsort(lex_key)
        gid_sorted = group_id[perm]
        dst_sorted = dst_agents[perm]

        uniq_gid, gid_counts = torch.unique_consecutive(gid_sorted, return_counts=True)
        gid_starts = torch.cumsum(torch.nn.functional.pad(gid_counts, (1, 0)), dim=0)[:-1]
        repeated_starts = gid_starts.repeat_interleave(gid_counts)
        positions = torch.arange(dst_sorted.numel(), device=self.device) - repeated_starts
        keep = positions < num_neighbors

        sel_gid = gid_sorted[keep]
        sel_pos = positions[keep].long()
        sel_dst = dst_sorted[keep]

        candidate_pairs = torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device)
        sel_batch = sel_gid // M
        sel_src = sel_gid % M
        candidate_pairs[sel_batch, sel_src, sel_pos] = sel_dst

        return candidate_pairs, debug_info
    

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
        self.grid_total_cells_int = int(self.grid_total_cells.item())
        self._arange_cache: Dict[int, torch.Tensor] = {}
        # 用于静态几何体索引的属性
        self.static_sorted_items = torch.empty((0,), dtype=torch.long, device=self.device)
        self.static_cell_starts = torch.empty((0,), dtype=torch.long, device=self.device)
        self.static_max_candidates_per_cell = 0
        self.static_cell_items = torch.empty((0, 0), dtype=torch.long, device=self.device)
        self.static_cell_counts = torch.empty((0,), dtype=torch.long, device=self.device)
        #print(f"SpatialHash initialized with grid {self.grid_size.cpu().numpy()} and cell size {self.cell_size:.2f}m.")

    def _cached_arange(self, size: int) -> torch.Tensor:
        size = int(size)
        cached = self._arange_cache.get(size)
        if cached is None:
            cached = torch.arange(size, device=self.device)
            self._arange_cache[size] = cached
        return cached

    def _profile_now(self, cuda_sync: bool = False) -> float:
        if cuda_sync and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.time()

    def _profile_record(self, profile: Optional[Dict], name: str, start_time: float,
                        cuda_sync: bool = False) -> float:
        if profile is None:
            return start_time
        now = self._profile_now(cuda_sync)
        profile[name] = (now - start_time) * 1000.0
        return now

    def _profile_set_default_counts(self, profile: Optional[Dict]):
        if profile is None:
            return
        profile.setdefault('cell_build_ms', 0.0)
        profile.setdefault('sort_group_ms', 0.0)
        profile.setdefault('pair_gen_ms', 0.0)
        profile.setdefault('dedup_ms', 0.0)
        profile.setdefault('occupancy_entries', 0)
        profile.setdefault('num_cell_groups', 0)
        profile.setdefault('max_group_size', 0)
        profile.setdefault('pairs_before_dedup', 0)
        profile.setdefault('unique_pairs', 0)

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
        grid_total_cells = self.grid_total_cells_int
        if num_items == 0:
            self.static_cell_starts = torch.zeros(grid_total_cells + 1, dtype=torch.long, device=self.device)
            self.static_max_candidates_per_cell = 0
            self.static_sorted_items = torch.empty(0, dtype=torch.long, device=self.device)
            self.static_cell_counts = torch.zeros(grid_total_cells, dtype=torch.long, device=self.device)
            self.static_cell_items = torch.empty((grid_total_cells, 0), dtype=torch.long, device=self.device)
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
            self.static_cell_starts = torch.zeros(grid_total_cells + 1, dtype=torch.long, device=self.device)
            self.static_max_candidates_per_cell = 0
            self.static_cell_counts = torch.zeros(grid_total_cells, dtype=torch.long, device=self.device)
            self.static_cell_items = torch.empty((grid_total_cells, 0), dtype=torch.long, device=self.device)
            return

        sorted_pairs = item_cell_pairs[item_cell_pairs[:, 1].argsort()]
        self.static_sorted_items = sorted_pairs[:, 0].contiguous()
        self.static_cell_starts = torch.zeros(grid_total_cells + 1, dtype=torch.long, device=self.device)
        unique_cells, counts = torch.unique_consecutive(sorted_pairs[:, 1], return_counts=True)
        self.static_cell_starts[unique_cells + 1] = counts
        self.static_cell_starts = self.static_cell_starts.cumsum_(0)
        self.static_max_candidates_per_cell = int(counts.max().item()) if counts.numel() > 0 else 0
        self.static_cell_counts = torch.zeros(grid_total_cells, dtype=torch.long, device=self.device)
        self.static_cell_counts[unique_cells] = counts

        self.static_cell_items = torch.full(
            (grid_total_cells, self.static_max_candidates_per_cell),
            -1,
            dtype=torch.long,
            device=self.device
        )
        if self.static_max_candidates_per_cell > 0:
            sorted_cells = sorted_pairs[:, 1]
            starts = self.static_cell_starts[sorted_cells]
            positions = torch.arange(sorted_pairs.shape[0], device=self.device) - starts
            self.static_cell_items[sorted_cells, positions] = sorted_pairs[:, 0]
        #print(f"Built static index for {num_items} items.")

    def query_points_padded(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量查询点，返回每个点所在 cell 的 padded 静态候选表。

        Returns:
            candidate_item_ids: [N, C]，无效位置为 -1。
            valid_mask: [N, C]，候选是否有效。
        """
        N = points.shape[0]
        max_candidates = int(self.static_max_candidates_per_cell)
        if N == 0 or max_candidates <= 0 or self.static_cell_items.numel() == 0:
            empty_items = torch.empty((N, 0), dtype=torch.long, device=self.device)
            empty_mask = torch.empty((N, 0), dtype=torch.bool, device=self.device)
            return empty_items, empty_mask

        cell_indices_2d = self.get_cell_idx(points)
        cell_indices_flat = cell_indices_2d[:, 0] * self.grid_size[1] + cell_indices_2d[:, 1]
        candidate_item_ids = self.static_cell_items[cell_indices_flat]
        candidate_counts = self.static_cell_counts[cell_indices_flat]
        candidate_positions = torch.arange(max_candidates, device=self.device).unsqueeze(0)
        valid_mask = (candidate_positions < candidate_counts.unsqueeze(1)) & (candidate_item_ids >= 0)
        return candidate_item_ids, valid_mask

    def query_dynamic_pair_list(self, B: int, M: int, active_mask: torch.Tensor,
                                verts_t0: torch.Tensor, verts_t1: torch.Tensor,
                                debug: bool = False, debug_env_idx: int = 0,
                                profile: Optional[Dict] = None,
                                profile_cuda_sync: bool = False,
                                pair_gen_chunk_pairs: int = 1048576) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        返回动态物体的 sparse unordered pair list: (batch, agent_i, agent_j)。
        agent_i < agent_j，且不会按 max_neighbors 截断。
        """
        self._profile_set_default_counts(profile)
        empty = torch.empty((0,), dtype=torch.long, device=self.device)
        if B == 0 or M <= 1:
            debug_info = {'occupied_cell_ids': empty} if debug else None
            return empty, empty, empty, debug_info

        cell_build_start = self._profile_now(profile_cuda_sync) if profile is not None else 0.0
        all_verts = torch.cat([verts_t0, verts_t1], dim=-2)
        min_coords, _ = torch.min(all_verts, dim=-2)
        max_coords, _ = torch.max(all_verts, dim=-2)
        
        min_grid_idx = self.get_cell_idx(min_coords.view(-1, 2)).view(B, M, 2)
        max_grid_idx = self.get_cell_idx(max_coords.view(-1, 2)).view(B, M, 2)
        
        span_x = (max_grid_idx[..., 0] - min_grid_idx[..., 0] + 1).clamp(min=1)
        span_y = (max_grid_idx[..., 1] - min_grid_idx[..., 1] + 1).clamp(min=1)
        max_cells_per_agent = (span_x * span_y).max().item()

        cell_offsets = self._cached_arange(max_cells_per_agent).view(1, -1)
        ix_x = cell_offsets % span_x.view(-1, 1)
        ix_y = cell_offsets // span_x.view(-1, 1)

        agent_grid_x = min_grid_idx[..., 0].view(-1, 1) + ix_x
        agent_grid_y = min_grid_idx[..., 1].view(-1, 1) + ix_y

        valid_cell_mask = (ix_y < span_y.view(-1, 1)) & active_mask.view(-1, 1)
        
        cell_ids = agent_grid_x * self.grid_size[1] + agent_grid_y

        valid_flat_indices = valid_cell_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
        flat_agent_positions = torch.div(valid_flat_indices, max_cells_per_agent, rounding_mode='floor')
        batch_indices = torch.div(flat_agent_positions, M, rounding_mode='floor')
        agent_indices = flat_agent_positions - batch_indices * M
        cell_indices = cell_ids.flatten()[valid_flat_indices]
        self._profile_record(profile, 'cell_build_ms', cell_build_start, profile_cuda_sync)
        if profile is not None:
            profile['occupancy_entries'] = int(cell_indices.numel())

        debug_info = None
        if debug:
            env_mask = (batch_indices == debug_env_idx)
            occupied_cell_indices = cell_indices[env_mask]
            debug_info = {'occupied_cell_ids': torch.unique(occupied_cell_indices)}

        # 向量化分组与配对：避免 per-batch 循环和巨大矩阵
        # 1) 将占用条目按 (batch, cell) 分组
        sort_group_start = self._profile_now(profile_cuda_sync) if profile is not None else 0.0
        group_keys = batch_indices.long() * self.grid_total_cells_int + cell_indices.long()
        sort_idx = torch.argsort(group_keys)
        sorted_keys = group_keys[sort_idx]
        sorted_batches = batch_indices[sort_idx]
        sorted_agents = agent_indices[sort_idx]

        if sorted_keys.numel() == 0:
            self._profile_record(profile, 'sort_group_ms', sort_group_start, profile_cuda_sync)
            return empty, empty, empty, debug_info

        all_keys, all_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        if profile is not None:
            profile['num_cell_groups'] = int(all_counts.numel())
        valid_group_indices = (all_counts >= 2).nonzero(as_tuple=False).squeeze(-1)
        if valid_group_indices.numel() == 0:
            self._profile_record(profile, 'sort_group_ms', sort_group_start, profile_cuda_sync)
            return empty, empty, empty, debug_info

        group_counts = all_counts[valid_group_indices]
        group_starts_all = torch.cumsum(torch.nn.functional.pad(all_counts, (1, 0)), dim=0)[:-1]
        group_starts = group_starts_all[valid_group_indices]

        # 2) 为每个分组生成无序对 (i<j)，完全向量化
        max_group_size = int(group_counts.max().item())
        if profile is not None:
            profile['max_group_size'] = max_group_size
        self._profile_record(profile, 'sort_group_ms', sort_group_start, profile_cuda_sync)

        pair_gen_start = self._profile_now(profile_cuda_sync) if profile is not None else 0.0
        tri = torch.triu_indices(max_group_size, max_group_size, offset=1, device=self.device)
        tri_i, tri_j = tri[0], tri[1]
        tri_pair_count = int(tri_i.numel())
        if tri_pair_count == 0:
            self._profile_record(profile, 'pair_gen_ms', pair_gen_start, profile_cuda_sync)
            return empty, empty, empty, debug_info

        target_pairs_per_chunk = max(1, int(pair_gen_chunk_pairs))
        groups_per_chunk = max(1, target_pairs_per_chunk // max(1, tri_pair_count))
        pair_batches_chunks = []
        agents_i_chunks = []
        agents_j_chunks = []
        num_valid_groups = int(group_counts.numel())

        for group_start_idx in range(0, num_valid_groups, groups_per_chunk):
            group_end_idx = min(group_start_idx + groups_per_chunk, num_valid_groups)
            counts_chunk = group_counts[group_start_idx:group_end_idx]
            starts_chunk = group_starts[group_start_idx:group_end_idx]
            valid_pair_mask = (
                (tri_i.unsqueeze(0) < counts_chunk.unsqueeze(1))
                & (tri_j.unsqueeze(0) < counts_chunk.unsqueeze(1))
            )
            valid_pair_positions = valid_pair_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
            if valid_pair_positions.numel() == 0:
                continue

            local_group_idx = torch.div(valid_pair_positions, tri_pair_count, rounding_mode='floor')
            local_pair_idx = valid_pair_positions - local_group_idx * tri_pair_count
            flat_i = starts_chunk[local_group_idx] + tri_i[local_pair_idx]
            flat_j = starts_chunk[local_group_idx] + tri_j[local_pair_idx]

            pair_batches_chunks.append(sorted_batches[flat_i])
            agents_i_chunks.append(sorted_agents[flat_i])
            agents_j_chunks.append(sorted_agents[flat_j])

        if not pair_batches_chunks:
            self._profile_record(profile, 'pair_gen_ms', pair_gen_start, profile_cuda_sync)
            return empty, empty, empty, debug_info

        pair_batches = torch.cat(pair_batches_chunks, dim=0)
        agents_i = torch.cat(agents_i_chunks, dim=0)
        agents_j = torch.cat(agents_j_chunks, dim=0)
        if profile is not None:
            profile['pairs_before_dedup'] = int(pair_batches.numel())
        self._profile_record(profile, 'pair_gen_ms', pair_gen_start, profile_cuda_sync)

        # 3) 对对儿进行归一化并去重（同一对可出现在多个cell）
        dedup_start = self._profile_now(profile_cuda_sync) if profile is not None else 0.0
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
        if profile is not None:
            profile['unique_pairs'] = int(pair_batches.numel())
        self._profile_record(profile, 'dedup_ms', dedup_start, profile_cuda_sync)

        return pair_batches, a_min, a_max, debug_info
    

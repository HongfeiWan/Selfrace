import torch
from typing import Tuple, Optional, Dict

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
            return

        sorted_pairs = item_cell_pairs[item_cell_pairs[:, 1].argsort()]
        self.static_sorted_items = sorted_pairs[:, 0].contiguous()

        self.static_cell_starts = torch.zeros(self.grid_total_cells + 1, dtype=torch.long, device=self.device)
        unique_cells, counts = torch.unique_consecutive(sorted_pairs[:, 1], return_counts=True)
        self.static_cell_starts[unique_cells + 1] = counts
        self.static_cell_starts = self.static_cell_starts.cumsum_(0)
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
        if num_candidates_per_point.sum() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
            
        point_indices_out = torch.arange(len(points), device=self.device).repeat_interleave(num_candidates_per_point)
        item_indices_out = torch.cat([self.static_sorted_items[s:e] for s, e in zip(starts, ends)])
        
        return torch.stack([point_indices_out, item_indices_out], dim=1)

    def query_dynamic_pairs(self, B: int, M: int, active_mask: torch.Tensor,
                            verts_t0: torch.Tensor, verts_t1: torch.Tensor,
                            max_neighbors: int, debug: bool = False, debug_env_idx: int = 0) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        为一批动态物体执行高效的临近对查询。
        此方法使用稀疏矩阵乘法，非常适合于智能体间的碰撞检测宽阶段。
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

        adjacency_list = []
        for i in range(B):
            batch_mask = (batch_indices == i)
            if not batch_mask.any():
                adjacency_list.append(torch.zeros((M, M), dtype=torch.bool, device=self.device))
                continue

            b_agent_indices = agent_indices[batch_mask]
            b_cell_indices = cell_indices[batch_mask]
            
            indices = torch.stack([b_agent_indices, b_cell_indices], dim=0)
            values = torch.ones(indices.shape[1], dtype=torch.float32, device=self.device)
            sparse_occupancy = torch.sparse_coo_tensor(indices, values, (M, self.grid_total_cells), is_coalesced=False).coalesce()
            
            adjacency_i = torch.sparse.mm(sparse_occupancy, sparse_occupancy.T).to_dense() > 0
            adjacency_list.append(adjacency_i)
        
        adjacency = torch.stack(adjacency_list, dim=0)

        adjacency.diagonal(dim1=-2, dim2=-1).fill_(False)
        sorter = torch.full_like(adjacency, -1e9, dtype=torch.float32)
        sorter[adjacency] = 1.0
        sorter += torch.rand_like(sorter) * 0.1
        
        num_neighbors = min(max_neighbors, M - 1)
        if num_neighbors <= 0:
            return torch.full((B, M, 0), -1, dtype=torch.long, device=self.device), debug_info

        _, candidate_pairs = torch.topk(sorter, k=num_neighbors, dim=-1)
        candidate_pairs[~adjacency.gather(2, candidate_pairs)] = -1
        
        return candidate_pairs, debug_info 
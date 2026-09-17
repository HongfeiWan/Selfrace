from typing import Tuple

import torch


class SpatialHash:
    """GPU spatial hash for persistent static geometry queries."""

    def __init__(
        self,
        cell_size: float,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
        device: torch.device,
    ):
        self.cell_size = cell_size
        self.min_bounds = min_bounds.to(device)
        self.max_bounds = max_bounds.to(device)
        self.device = device
        grid_dim = torch.ceil(
            (self.max_bounds - self.min_bounds) / self.cell_size
        ).long()
        self.grid_size = torch.maximum(
            grid_dim,
            torch.tensor([1, 1], device=device, dtype=torch.long),
        )
        self.grid_total_cells = self.grid_size[0] * self.grid_size[1]
        self.grid_total_cells_int = int(self.grid_total_cells.item())

        self.static_sorted_items = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        self.static_cell_starts = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        self.static_max_candidates_per_cell = 0
        self.static_cell_items = torch.empty(
            (0, 0), dtype=torch.long, device=self.device
        )
        self.static_cell_counts = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )

    def get_cell_idx(self, points: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to clamped grid-cell coordinates."""
        indices = torch.floor((points - self.min_bounds) / self.cell_size).long()
        indices[:, 0].clamp_(0, self.grid_size[0] - 1)
        indices[:, 1].clamp_(0, self.grid_size[1] - 1)
        return indices

    def _clear_static_index(self, grid_total_cells: int) -> None:
        self.static_cell_starts = torch.zeros(
            grid_total_cells + 1, dtype=torch.long, device=self.device
        )
        self.static_max_candidates_per_cell = 0
        self.static_sorted_items = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self.static_cell_counts = torch.zeros(
            grid_total_cells, dtype=torch.long, device=self.device
        )
        self.static_cell_items = torch.empty(
            (grid_total_cells, 0), dtype=torch.long, device=self.device
        )

    def build_static_index(self, static_items_bounds: torch.Tensor) -> None:
        """Build the persistent cell-to-static-item lookup table."""
        num_items = static_items_bounds.shape[0]
        grid_total_cells = self.grid_total_cells_int
        if num_items == 0:
            self._clear_static_index(grid_total_cells)
            return

        start_cells = self.get_cell_idx(static_items_bounds[:, 0])
        end_cells = self.get_cell_idx(static_items_bounds[:, 1])
        item_ids = torch.arange(num_items, device=self.device)
        all_pairs = []

        # Static geometry is indexed once during initialization.
        for item_index in range(num_items):
            for x in range(
                start_cells[item_index, 0].item(),
                end_cells[item_index, 0].item() + 1,
            ):
                for y in range(
                    start_cells[item_index, 1].item(),
                    end_cells[item_index, 1].item() + 1,
                ):
                    cell_idx_flat = x * self.grid_size[1] + y
                    all_pairs.append([item_ids[item_index], cell_idx_flat])

        if not all_pairs:
            self._clear_static_index(grid_total_cells)
            return

        item_cell_pairs = torch.tensor(
            all_pairs, dtype=torch.long, device=self.device
        )
        sorted_pairs = item_cell_pairs[item_cell_pairs[:, 1].argsort()]
        self.static_sorted_items = sorted_pairs[:, 0].contiguous()
        self.static_cell_starts = torch.zeros(
            grid_total_cells + 1, dtype=torch.long, device=self.device
        )
        unique_cells, counts = torch.unique_consecutive(
            sorted_pairs[:, 1], return_counts=True
        )
        self.static_cell_starts[unique_cells + 1] = counts
        self.static_cell_starts.cumsum_(0)
        self.static_max_candidates_per_cell = int(counts.max().item())
        self.static_cell_counts = torch.zeros(
            grid_total_cells, dtype=torch.long, device=self.device
        )
        self.static_cell_counts[unique_cells] = counts

        self.static_cell_items = torch.full(
            (grid_total_cells, self.static_max_candidates_per_cell),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        sorted_cells = sorted_pairs[:, 1]
        starts = self.static_cell_starts[sorted_cells]
        positions = torch.arange(
            sorted_pairs.shape[0], device=self.device
        ) - starts
        self.static_cell_items[sorted_cells, positions] = sorted_pairs[:, 0]

    def query_points_padded(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the padded static candidates for every query point."""
        num_points = points.shape[0]
        max_candidates = int(self.static_max_candidates_per_cell)
        if (
            num_points == 0
            or max_candidates <= 0
            or self.static_cell_items.numel() == 0
        ):
            empty_items = torch.empty(
                (num_points, 0), dtype=torch.long, device=self.device
            )
            empty_mask = torch.empty(
                (num_points, 0), dtype=torch.bool, device=self.device
            )
            return empty_items, empty_mask

        cell_indices_2d = self.get_cell_idx(points)
        cell_indices_flat = (
            cell_indices_2d[:, 0] * self.grid_size[1] + cell_indices_2d[:, 1]
        )
        candidate_item_ids = self.static_cell_items[cell_indices_flat]
        candidate_counts = self.static_cell_counts[cell_indices_flat]
        candidate_positions = torch.arange(
            max_candidates, device=self.device
        ).unsqueeze(0)
        valid_mask = (
            (candidate_positions < candidate_counts.unsqueeze(1))
            & (candidate_item_ids >= 0)
        )
        return candidate_item_ids, valid_mask

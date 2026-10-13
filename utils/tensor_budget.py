"""Bounded scratch space for geometry (outputs are accounted for separately)."""

import torch


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def workspace_bytes(device, maximum_mb: float = 64.0) -> int:
    """Sample free memory at construction, never in a frame's hot path."""
    limit = max(4096, int(float(maximum_mb) * 1024**2))
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        limit = min(limit, max(4096, free // 16))
    return limit


def nearest_indices(query, reference, k, budget_bytes=64 * 1024**2,
                    query_chunk=1024):
    """Exact two-dimensional tiled kNN, never an N-by-map distance matrix.

    Ties may choose any equally distant point, as with torch.topk. Scratch is
    bounded independently of map size; the persistent N-by-k result is not
    scratch. OOM retries only this pure query, not a partially advanced world.
    """
    k = int(k)
    out = torch.full((query.shape[0], k), -1, device=query.device, dtype=torch.long)
    effective = min(k, reference.shape[0])
    if effective == 0 or query.shape[0] == 0:
        return out
    columns = min(reference.shape[0], 2048, max(1, int(budget_bytes) // 64 - 2 * k))
    rows = max(1, min(int(query_chunk), int(budget_bytes) // (32 * (columns + 2 * k))))
    start = 0
    while start < query.shape[0]:
        stop = min(start + rows, query.shape[0])
        try:
            out[start:stop, :effective] = _nearest_tile(query[start:stop], reference, effective, columns)
            start = stop
        except RuntimeError as exc:
            if query.device.type != "cuda" or not is_cuda_oom(exc) or (rows == 1 and columns == 1):
                raise
            # Bound retry work and retain the learned smaller tile for this call.
            if rows > 1:
                rows = max(1, rows // 2)
            else:
                columns = max(1, columns // 2)
    return out


def _nearest_tile(query, reference, k, columns):
    # A separate pure frame releases all failed scratch before the outer retry.
    best_d = query.new_full((query.shape[0], k), float('inf'))
    best_i = torch.full(best_d.shape, -1, device=query.device, dtype=torch.long)
    for left in range(0, reference.shape[0], columns):
        distance = torch.cdist(query, reference[left:left + columns], compute_mode='donot_use_mm_for_euclid_dist')
        local_d, local_i = distance.topk(min(k, distance.shape[1]), largest=False, dim=1)
        merged_d = torch.cat((best_d, local_d), dim=1)
        merged_i = torch.cat((best_i, local_i + left), dim=1)
        best_d, positions = merged_d.topk(k, largest=False, dim=1)
        best_i = merged_i.gather(1, positions)
    return best_i

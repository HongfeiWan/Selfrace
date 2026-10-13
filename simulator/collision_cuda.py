"""CUDA cell-list narrow phase. Imported only when Triton is available.

Every vehicle has one sorted cell entry. Normal swept AABBs fit inside one
cell-width, so overlapping pairs have centers in the same/8 adjacent cells.
Large/fast/out-of-range vehicles take an exact all-slot fallback ON DEVICE.
Cell occupancy is not capped: crowded cells are streamed in 32-pair tiles.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pose(states, index, mask):
    x = tl.load(states + index * 7, mask, 0)
    y = tl.load(states + index * 7 + 1, mask, 0)
    yaw = tl.load(states + index * 7 + 2, mask, 0)
    length = tl.load(states + index * 7 + 4, mask, 0) * 0.5
    width = tl.load(states + index * 7 + 5, mask, 0) * 0.5
    return x, y, tl.cos(yaw), tl.sin(yaw), length, width


@triton.jit
def _sat(states, i, j, mask):
    ax, ay, ac, ass, al, aw = _pose(states, i, mask)
    bx, by, bc, bs, bl, bw = _pose(states, j, mask)
    dx, dy = bx - ax, by - ay
    c = tl.abs(ac * bc + ass * bs)
    s = tl.abs(ass * bc - ac * bs)
    return ((tl.abs(dx * ac + dy * ass) < al + bl * c + bw * s)
            & (tl.abs(-dx * ass + dy * ac) < aw + bl * s + bw * c)
            & (tl.abs(dx * bc + dy * bs) < bl + al * c + aw * s)
            & (tl.abs(-dx * bs + dy * bc) < bw + al * s + aw * c))


@triton.jit
def _one_way(s0, s1, v0, v1, ref, moving, mask):
    x0, y0, c0, sn0, length, width = _pose(s0, ref, mask)
    x1, y1, c1, sn1, _, _ = _pose(s1, ref, mask)
    hit = tl.full(mask.shape, False, tl.int1)
    for corner in tl.static_range(4):
        dx0 = tl.load(v0 + moving * 8 + corner * 2, mask, 0) - x0
        dy0 = tl.load(v0 + moving * 8 + corner * 2 + 1, mask, 0) - y0
        dx1 = tl.load(v1 + moving * 8 + corner * 2, mask, 0) - x1
        dy1 = tl.load(v1 + moving * 8 + corner * 2 + 1, mask, 0) - y1
        px0, py0 = dx0 * c0 + dy0 * sn0, -dx0 * sn0 + dy0 * c0
        px1, py1 = dx1 * c1 + dy1 * sn1, -dx1 * sn1 + dy1 * c1
        vx, vy = px1 - px0, py1 - py0
        invx = 1.0 / (vx + tl.where(vx >= 0, 1.e-8, -1.e-8))
        invy = 1.0 / (vy + tl.where(vy >= 0, 1.e-8, -1.e-8))
        tx0, tx1 = (-length - px0) * invx, (length - px0) * invx
        ty0, ty1 = (-width - py0) * invy, (width - py0) * invy
        enter = tl.maximum(tl.minimum(tx0, tx1), tl.minimum(ty0, ty1))
        leave = tl.minimum(tl.maximum(tx0, tx1), tl.maximum(ty0, ty1))
        hit |= (enter < leave) & (leave > 0) & (enter < 1)
    return hit


@triton.jit
def _cell_kernel(s0, s1, v0, v1, bounds_min, bounds_max, active, regular, order, starts, ends, out,
                 M: tl.constexpr, TILE: tl.constexpr):
    world = tl.program_id(0)
    agent = tl.program_id(1)
    i = world * M + agent
    enabled = tl.load(active + i)
    if enabled:
        ordinary = tl.load(regular + i)
        cells = tl.arange(0, 16)
        lower = tl.load(starts + i * 9 + cells, cells < 9, 0)
        upper = tl.load(ends + i * 9 + cells, cells < 9, 0)
        count = upper - lower
        cumulative = tl.cumsum(count)
        total = tl.where(ordinary, tl.sum(count), M)
        for base in range(0, total, TILE):
            rank = base + tl.arange(0, TILE)
            valid = rank < total
            if ordinary:
                cell = tl.sum(((rank[:, None] >= cumulative[None, :]) & (cells[None, :] < 9)).to(tl.int32), 1)
                before = tl.sum(tl.where(cells[None, :] < cell[:, None], count[None, :], 0), 1)
                begin = tl.sum(tl.where(cells[None, :] == cell[:, None], lower[None, :], 0), 1)
                jslot = tl.load(order + world * M + begin + rank - before, valid, 0)
                valid &= jslot > agent
            else:
                # No overflow is dropped and no candidate count goes to CPU.
                jslot = rank
                valid &= jslot != agent
            j = world * M + jslot
            valid &= tl.load(active + j, valid, 0)
            for axis in tl.static_range(2):
                ilo = tl.load(bounds_min + i * 2 + axis)
                ihi = tl.load(bounds_max + i * 2 + axis)
                jlo = tl.load(bounds_min + j * 2 + axis, valid, 0)
                jhi = tl.load(bounds_max + j * 2 + axis, valid, 0)
                valid &= (ilo <= jhi) & (jlo <= ihi)
            if tl.sum(valid.to(tl.int32)) > 0:
                pair_i = i + tl.zeros((TILE,), tl.int32)
                hit = (_sat(s0, pair_i, j, valid) | _sat(s1, pair_i, j, valid)
                       | _one_way(s0, s1, v0, v1, pair_i, j, valid)
                       | _one_way(s0, s1, v0, v1, j, pair_i, valid)) & valid
                tl.atomic_or(out + j, hit.to(tl.int32), valid, sem="relaxed")
                tl.atomic_or(out + i, (tl.sum(hit.to(tl.int32)) > 0).to(tl.int32), sem="relaxed")


def cell_collisions(states_t0, states_t1, active, verts_t0, verts_t1, cell_size):
    """One bounded world tile. Shapes, not occupancy, determine allocations."""
    worlds, agents = active.shape
    if agents == 0 or worlds == 0:
        return torch.zeros_like(active)
    swept_min = torch.minimum(verts_t0.amin(-2), verts_t1.amin(-2))
    swept_max = torch.maximum(verts_t0.amax(-2), verts_t1.amax(-2))
    cell = ((swept_min + swept_max) * (0.5 / cell_size)).floor().long()
    # Encoding is injective within this range. Outside uses the exact fallback.
    offset, stride = 1 << 20, 1 << 21
    ordinary = ((swept_max - swept_min) <= cell_size).all(-1)
    ordinary &= ((cell > -offset + 1) & (cell < offset - 1)).all(-1) & active
    keys = (cell[..., 0] + offset) * stride + cell[..., 1] + offset
    keys = keys.masked_fill(~ordinary, torch.iinfo(torch.int64).max)
    sorted_keys, order = keys.sort(dim=1)
    offsets = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]], device=cell.device)
    adjacent = cell[:, :, None, :] + offsets
    queries = ((adjacent[..., 0] + offset) * stride + adjacent[..., 1] + offset).flatten(1).contiguous()
    starts = torch.searchsorted(sorted_keys, queries).view(worlds, agents, 9)
    ends = torch.searchsorted(sorted_keys, queries, right=True).view(worlds, agents, 9)
    out = torch.zeros_like(active, dtype=torch.int32)
    _cell_kernel[(worlds, agents)](
        states_t0.contiguous(), states_t1.contiguous(), verts_t0.contiguous(), verts_t1.contiguous(),
        swept_min, swept_max, active.contiguous(), ordinary, order, starts, ends, out, agents, 32,
        num_warps=4, enable_fp_fusion=False,
    )
    return out.bool()

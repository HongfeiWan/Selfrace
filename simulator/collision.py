import importlib.util
import os
import sys
import time
import warnings
from typing import Dict, Optional, Tuple

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 依赖于 spatial_hash
# 添加utils目录到路径
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from spatial_hash import SpatialHash


def _stream_transform_points(points: torch.Tensor, ref_state: torch.Tensor) -> torch.Tensor:
    """World-to-local transform used by the bounded collision kernel."""
    center = ref_state[..., :2].unsqueeze(-2)
    yaw = ref_state[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    translated = points - center
    local_x = (
        translated[..., 0] * cos_yaw.unsqueeze(-1)
        + translated[..., 1] * sin_yaw.unsqueeze(-1)
    )
    local_y = (
        -translated[..., 0] * sin_yaw.unsqueeze(-1)
        + translated[..., 1] * cos_yaw.unsqueeze(-1)
    )
    return torch.stack((local_x, local_y), dim=-1)


def _stream_line_aabb_intersection(p0, p1, aabb_min, aabb_max) -> torch.Tensor:
    eps = 1e-8
    direction = p1 - p0
    inv_direction = 1.0 / (
        direction + torch.copysign(torch.full_like(direction, eps), direction)
    )
    t_plane1 = (aabb_min - p0) * inv_direction
    t_plane2 = (aabb_max - p0) * inv_direction
    t_enter = torch.minimum(t_plane1, t_plane2).amax(dim=-1)
    t_exit = torch.maximum(t_plane1, t_plane2).amin(dim=-1)
    return ~((t_enter >= t_exit) | (t_exit <= 0) | (t_enter >= 1))


def _stream_one_way_collision(ref_t0, ref_t1, moving_verts_t0, moving_verts_t1):
    local_t0 = _stream_transform_points(moving_verts_t0, ref_t0)
    local_t1 = _stream_transform_points(moving_verts_t1, ref_t1)
    half_dims = ref_t0[..., 4:6] / 2.0
    intersections = _stream_line_aabb_intersection(
        local_t0,
        local_t1,
        -half_dims.unsqueeze(-2),
        half_dims.unsqueeze(-2),
    )
    return intersections.any(dim=-1)


def _bounded_collision_world_chunk(
    states_t0: torch.Tensor,
    states_t1: torch.Tensor,
    active_mask: torch.Tensor,
    verts_t0: torch.Tensor,
    verts_t1: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    """Broad phase, exact narrow phase, and scatter for a bounded world chunk.

    The data-dependent tensor is bounded by ``worlds * len(pair_i)``, which the
    caller caps with ``collision_stream_pair_budget``. Under ``torch.compile``
    its nonzero and consumers stay in the device graph, so Python never reads a
    candidate count.
    """
    worlds, max_agents = active_mask.shape
    swept_min = torch.minimum(verts_t0.amin(dim=-2), verts_t1.amin(dim=-2))
    swept_max = torch.maximum(verts_t0.amax(dim=-2), verts_t1.amax(dim=-2))

    broad = (
        (swept_min[:, pair_i, 0] <= swept_max[:, pair_j, 0])
        & (swept_min[:, pair_j, 0] <= swept_max[:, pair_i, 0])
        & (swept_min[:, pair_i, 1] <= swept_max[:, pair_j, 1])
        & (swept_min[:, pair_j, 1] <= swept_max[:, pair_i, 1])
        & active_mask[:, pair_i]
        & active_mask[:, pair_j]
    )
    candidates = torch.nonzero(broad, as_tuple=False)
    world_idx = candidates[:, 0]
    pair_idx = candidates[:, 1]
    agent_i = pair_i[pair_idx]
    agent_j = pair_j[pair_idx]

    pair_collisions = _stream_one_way_collision(
        states_t0[world_idx, agent_i],
        states_t1[world_idx, agent_i],
        verts_t0[world_idx, agent_j],
        verts_t1[world_idx, agent_j],
    ) | _stream_one_way_collision(
        states_t0[world_idx, agent_j],
        states_t1[world_idx, agent_j],
        verts_t0[world_idx, agent_i],
        verts_t1[world_idx, agent_i],
    )

    collision_bits = torch.zeros(
        worlds * max_agents,
        dtype=torch.int32,
        device=states_t0.device,
    )
    hit_bits = pair_collisions.to(torch.int32)
    collision_bits.scatter_reduce_(
        0,
        world_idx * max_agents + agent_i,
        hit_bits,
        reduce="amax",
        include_self=True,
    )
    collision_bits.scatter_reduce_(
        0,
        world_idx * max_agents + agent_j,
        hit_bits,
        reduce="amax",
        include_self=True,
    )
    return collision_bits.view(worlds, max_agents).bool()


def _bounded_collision_world_chunk_dense(
    states_t0: torch.Tensor,
    states_t1: torch.Tensor,
    active_mask: torch.Tensor,
    verts_t0: torch.Tensor,
    verts_t1: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    """Synchronization-free eager fallback with the same fixed pair bound."""
    worlds, max_agents = active_mask.shape
    swept_min = torch.minimum(verts_t0.amin(dim=-2), verts_t1.amin(dim=-2))
    swept_max = torch.maximum(verts_t0.amax(dim=-2), verts_t1.amax(dim=-2))
    broad = (
        (swept_min[:, pair_i, 0] <= swept_max[:, pair_j, 0])
        & (swept_min[:, pair_j, 0] <= swept_max[:, pair_i, 0])
        & (swept_min[:, pair_i, 1] <= swept_max[:, pair_j, 1])
        & (swept_min[:, pair_j, 1] <= swept_max[:, pair_i, 1])
        & active_mask[:, pair_i]
        & active_mask[:, pair_j]
    )
    pair_collisions = broad & (
        _stream_one_way_collision(
            states_t0[:, pair_i],
            states_t1[:, pair_i],
            verts_t0[:, pair_j],
            verts_t1[:, pair_j],
        )
        | _stream_one_way_collision(
            states_t0[:, pair_j],
            states_t1[:, pair_j],
            verts_t0[:, pair_i],
            verts_t1[:, pair_i],
        )
    )

    world_idx = torch.arange(worlds, device=states_t0.device).unsqueeze(1)
    collision_bits = torch.zeros(
        worlds * max_agents,
        dtype=torch.int32,
        device=states_t0.device,
    )
    hit_bits = pair_collisions.to(torch.int32).flatten()
    collision_bits.scatter_reduce_(
        0,
        (world_idx * max_agents + pair_i).flatten(),
        hit_bits,
        reduce="amax",
        include_self=True,
    )
    collision_bits.scatter_reduce_(
        0,
        (world_idx * max_agents + pair_j).flatten(),
        hit_bits,
        reduce="amax",
        include_self=True,
    )
    return collision_bits.view(worlds, max_agents).bool()


class CollisionChecker:
    """Continuous collision checking with a hard-bounded pair stream.

    Python only iterates over shape-derived world/pair ranges. Candidate counts
    remain on-device, so the CUDA hot path never calls ``item()`` or branches on
    a tensor value. Linux CUDA compiles the sparse narrow phase; the eager CUDA
    fallback evaluates a fixed dense pair block to preserve that guarantee.
    """

    def __init__(self, config: Dict, spatial_hash: SpatialHash):
        """Initialize device, stream capacity, and lazy compiled kernels."""
        simulator_config = config.get('simulator', config)
        self.device = torch.device(spatial_hash.device)
        self.cell_size = float(spatial_hash.cell_size)
        self.stream_pair_budget = max(
            1, int(simulator_config.get('collision_stream_pair_budget', 524288))
        )
        self.stream_compile = bool(
            simulator_config.get('collision_stream_compile', True)
        )
        self.stream_compile_mode = str(
            simulator_config.get('collision_stream_compile_mode', 'default')
        )
        self._compile_backend_available = (
            os.name != 'nt'
            and hasattr(torch, 'compile')
            and importlib.util.find_spec('triton') is not None
        )
        self._pair_index_cache = {}
        self._compiled_stream_chunk = None
        self._stream_compile_failed = False
        self._stream_compile_warning_emitted = False
        profile_config = config.get('training', {}).get('profile', {})
        self.profile_cuda_sync = bool(profile_config.get('cuda_sync', False))
        
        print(f"Collision hash cell size: {self.cell_size:.2f}m")
        print(f"Collision stream pair budget: {self.stream_pair_budget:,}")

    def _profile_now(self) -> float:
        if self.profile_cuda_sync and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return time.time()

    def _profile_record(
        self, profile: Optional[Dict], name: str, start_time: float
    ) -> float:
        if profile is None:
            return start_time
        now = self._profile_now()
        profile[name] = (now - start_time) * 1000.0
        return now

    def _pair_indices(self, max_agents: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._pair_index_cache.get(int(max_agents))
        if cached is None:
            pairs = torch.triu_indices(
                max_agents,
                max_agents,
                offset=1,
                device=self.device,
            )
            cached = (pairs[0].contiguous(), pairs[1].contiguous())
            self._pair_index_cache[int(max_agents)] = cached
        return cached

    def _can_compile_stream(self) -> bool:
        return (
            self.stream_compile
            and not self._stream_compile_failed
            and self.device.type == 'cuda'
            and self._compile_backend_available
        )

    def _run_stream_chunk(self, *args) -> torch.Tensor:
        if not self._can_compile_stream():
            fallback = (
                _bounded_collision_world_chunk_dense
                if self.device.type == 'cuda'
                else _bounded_collision_world_chunk
            )
            return fallback(*args)
        try:
            if self._compiled_stream_chunk is None:
                self._compiled_stream_chunk = torch.compile(
                    _bounded_collision_world_chunk,
                    mode=self.stream_compile_mode,
                    dynamic=True,
                    fullgraph=True,
                )
            dynamo = getattr(torch, '_dynamo', None)
            patch_values = {}
            if dynamo is not None:
                if hasattr(dynamo.config, 'capture_dynamic_output_shape_ops'):
                    patch_values['capture_dynamic_output_shape_ops'] = True
                if hasattr(dynamo.config, 'capture_scalar_outputs'):
                    patch_values['capture_scalar_outputs'] = True
            if patch_values:
                with dynamo.config.patch(**patch_values):
                    return self._compiled_stream_chunk(*args)
            return self._compiled_stream_chunk(*args)
        except Exception as exc:
            if 'out of memory' in str(exc).lower():
                raise
            self._stream_compile_failed = True
            self._compiled_stream_chunk = None
            if not self._stream_compile_warning_emitted:
                warnings.warn(
                    f"collision torch.compile disabled after backend failure: {exc}",
                    RuntimeWarning,
                )
                self._stream_compile_warning_emitted = True
            return _bounded_collision_world_chunk_dense(*args)

    def _streaming_dynamic_collisions(
        self,
        states_t0: torch.Tensor,
        states_t1: torch.Tensor,
        active_mask: torch.Tensor,
        verts_t0: torch.Tensor,
        verts_t1: torch.Tensor,
        profile: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Process fixed-capacity pair/world tiles and immediately merge results."""
        batch_size, max_agents = active_mask.shape
        collisions = torch.zeros(
            (batch_size, max_agents), dtype=torch.bool, device=self.device
        )
        if batch_size == 0 or max_agents <= 1:
            return collisions

        pair_i, pair_j = self._pair_indices(max_agents)
        pairs_per_world = int(pair_i.numel())
        stream_start = self._profile_now() if profile is not None else 0.0
        pairs_per_chunk = min(pairs_per_world, self.stream_pair_budget)
        for pair_start in range(0, pairs_per_world, pairs_per_chunk):
            pair_end = min(pair_start + pairs_per_chunk, pairs_per_world)
            pair_i_chunk = pair_i[pair_start:pair_end]
            pair_j_chunk = pair_j[pair_start:pair_end]
            worlds_per_chunk = max(
                1, self.stream_pair_budget // (pair_end - pair_start)
            )
            for world_start in range(0, batch_size, worlds_per_chunk):
                world_end = min(world_start + worlds_per_chunk, batch_size)
                chunk_collisions = self._run_stream_chunk(
                    states_t0[world_start:world_end],
                    states_t1[world_start:world_end],
                    active_mask[world_start:world_end],
                    verts_t0[world_start:world_end],
                    verts_t1[world_start:world_end],
                    pair_i_chunk,
                    pair_j_chunk,
                )
                collisions[world_start:world_end].logical_or_(chunk_collisions)
        self._profile_record(profile, 'collision_stream_ms', stream_start)
        return collisions

    def check(
        self,
        states_t0: torch.Tensor,
        states_t1: torch.Tensor,
        static_obstacles: Optional[torch.Tensor] = None,
        debug: bool = False,
        debug_env_idx: int = 0,
        active_mask_override: Optional[torch.Tensor] = None,
        profile: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        目的: 作为主入口函数，对一批智能体的状态进行完整的碰撞检测。

        逻辑:
        1.  从 t1 时刻的状态中提取出当前处于激活状态的智能体。
        2.  计算所有智能体在 t0 和 t1 时刻的边界框顶点。
        3.  按固定 pair 预算流式执行宽阶段和连续窄阶段检测。
        4.  每个块立刻散播回最终结果，不保留全局候选列表。
        5.  如果提供了静态障碍物，则调用静态碰撞检测方法。
        6.  合并动态和静态碰撞结果，并用激活掩码过滤，返回最终结果。
        7.  (Debug) 如果开启debug模式，额外返回用于可视化的调试信息。
        """
        # 确保输入张量在正确的设备上
        states_t0 = states_t0.to(self.device)
        states_t1 = states_t1.to(self.device)
        
        if active_mask_override is None:
            active_mask = states_t1[..., 6] > 0.5
        else:
            active_mask = active_mask_override.to(self.device, dtype=torch.bool)
        vertex_start = self._profile_now() if profile is not None else 0.0
        verts_t0 = self._get_world_vertices(states_t0)
        verts_t1 = self._get_world_vertices(states_t1)
        self._profile_record(profile, 'vertex_ms', vertex_start)

        dynamic_collisions = self._streaming_dynamic_collisions(
            states_t0,
            states_t1,
            active_mask,
            verts_t0,
            verts_t1,
            profile=profile,
        )

        final_collisions = dynamic_collisions
        
        if static_obstacles is not None:
            static_collisions = self._check_static_collisions(
                states_t0, states_t1, static_obstacles
            )
            final_collisions = torch.logical_or(final_collisions, static_collisions)

        final_collisions = final_collisions & active_mask

        if debug:
            debug_data = {
                'broad_phase': {
                    'algorithm': 'bounded_pair_stream',
                    'pair_budget': self.stream_pair_budget,
                }
            }
            return final_collisions, debug_data
        
        return final_collisions

    def _get_world_vertices(self, states: torch.Tensor) -> torch.Tensor:
        """
        目的: 从智能体的状态向量中计算其矩形边界框的四个顶点在世界坐标系下的坐标。

        逻辑:
        - 解包状态向量获取中心点、偏航角和尺寸。
        - 根据偏航角计算出车身坐标系的基向量（x轴和y轴方向）。
        - 从中心点出发，通过基向量和半长/半宽，计算出四个顶点的坐标。
        - 所有操作都是并行的张量运算。
        """
        center_x, center_y, yaw, _, length, width, _ = states.unbind(-1)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        vec_x = torch.stack([cos_yaw, sin_yaw], dim=-1)
        vec_y = torch.stack([-sin_yaw, cos_yaw], dim=-1)
        half_l, half_w = length.unsqueeze(-1) / 2, width.unsqueeze(-1) / 2
        center = states[..., :2]
        v1 = center + half_l * vec_x + half_w * vec_y
        v2 = center - half_l * vec_x + half_w * vec_y
        v3 = center - half_l * vec_x - half_w * vec_y
        v4 = center + half_l * vec_x - half_w * vec_y
        return torch.stack([v1, v2, v3, v4], dim=-2)

    def _check_static_collisions(
        self,
        states_t0: torch.Tensor,
        states_t1: torch.Tensor,
        static_obstacles: torch.Tensor,
    ) -> torch.Tensor:
        """
        目的: 批量检测所有动态智能体与所有静态障碍物之间的碰撞。

        逻辑:
        - 将静态障碍物视为一种特殊的智能体，其在 t0 和 t1 时刻的状态完全相同。
        - 利用广播机制，将动态智能体（作为移动方）与所有静态障碍物（作为参考方）进行配对。
        - 调用向量化的连续单向检测，一次性完成所有动态-静态对的碰撞检测。
        - 如果一个智能体与任何一个障碍物发生碰撞，则将其标记为碰撞。
        """
        O, _, _ = static_obstacles.shape
        all_verts_t0 = self._get_world_vertices(states_t0)
        all_verts_t1 = self._get_world_vertices(states_t1)
        obs_centers = torch.mean(static_obstacles, dim=1)
        obs_states = torch.zeros(O, 7, device=self.device)
        obs_states[:, :2] = obs_centers
        obs_states[:, 6] = 1.0
        mov_verts_t0 = all_verts_t0.unsqueeze(2)
        mov_verts_t1 = all_verts_t1.unsqueeze(2)
        ref_states = obs_states.view(1, 1, O, 7)
        collisions = _stream_one_way_collision(
            ref_states, ref_states, mov_verts_t0, mov_verts_t1
        )
        return collisions.any(dim=-1)

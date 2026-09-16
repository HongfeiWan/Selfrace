import os
import sys
import json
import socket
import math
import random
import shutil
import gc
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加simulator目录到路径
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)

from simulator import TeraflowSimulator
from adaptive_batch import AdaptiveBatchSizer
from experiment_logging import initialize_swanlab
from feature_schema import FEATURE_PAD_VALUE, FeatureSchema
from network import create_network

'''
验证 NVLink: nvidia-smi topo -m 查看拓扑,NCCL_DEBUG=INFO 输出里会显示使用 NVLink 的通道。
'''

# ============================== 全tensor GAE ==============================
# 全tensor GAE: rewards[T, ...], values[T+1, ...], dones[T, ...] (0/1)
# 返回 advantages[T, ...], returns[T, ...]
def gae_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, gae_lambda: float):
	T = rewards.shape[0]
	done_mask = dones.to(rewards.dtype)
	advantages = torch.zeros_like(rewards)
	gae = torch.zeros_like(rewards[0])
	for t in range(T - 1, -1, -1):
		delta = rewards[t] + gamma * values[t + 1] * (1.0 - done_mask[t]) - values[t]
		gae = delta + gamma * gae_lambda * (1.0 - done_mask[t]) * gae
		advantages[t] = gae
	returns = advantages + values[:-1]
	#returns = advantages + V(s)
    #= A(s,a) + V(s)  
    #= (Q(s,a) - V(s)) + V(s)
    #= Q(s,a)
	# advantages = (advantages-advantages.mean())/advantages.std()
	return advantages, returns #即返回A(s,a), Q(s,a)

# ============================== 模型检查点保存 ==============================
CHECKPOINT_FORMAT_VERSION = 3


def capture_rng_state(device: torch.device) -> dict:
	state = {
		'python': random.getstate(),
		'torch_cpu': torch.get_rng_state(),
	}
	if device.type == 'cuda':
		state['torch_cuda'] = torch.cuda.get_rng_state(device).cpu()
	return state


def restore_rng_state(state: dict, device: torch.device):
	random.setstate(state['python'])
	torch.set_rng_state(state['torch_cpu'].cpu())
	if device.type == 'cuda' and 'torch_cuda' in state:
		torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def save_checkpoint(
	model,
	policy_optimizer,
	value_optimizer,
	policy_scheduler,
	value_scheduler,
	amp_scaler,
	batch_sizers: dict,
	progress: dict,
	a_max_ewma,
	checkpoint_dir: str,
	device: torch.device,
	rank: int = 0,
):
	"""Atomically save one complete optimizer-update boundary."""
	rank_runtime = {
		'rng_state': capture_rng_state(device),
		'adaptive_batch_state': {
			name: sizer.state_dict() for name, sizer in batch_sizers.items()
		},
		'a_max_ewma': None if a_max_ewma is None else a_max_ewma.detach().cpu(),
	}
	if dist.is_available() and dist.is_initialized():
		gathered_runtime = [None] * dist.get_world_size() if rank == 0 else None
		dist.gather_object(rank_runtime, gathered_runtime, dst=0)
	else:
		gathered_runtime = [rank_runtime]
	if rank != 0:
		return None

	os.makedirs(checkpoint_dir, exist_ok=True)
	save_model = model.module if hasattr(model, 'module') else model
	update_step = int(progress['update_step'])
	state = {
		'format_version': CHECKPOINT_FORMAT_VERSION,
		'progress': dict(progress),
		'model_state_dict': save_model.state_dict(),
		'policy_optimizer_state_dict': policy_optimizer.state_dict(),
		'value_optimizer_state_dict': value_optimizer.state_dict(),
		'policy_scheduler_state_dict': policy_scheduler.state_dict(),
		'value_scheduler_state_dict': value_scheduler.state_dict(),
		'grad_scaler_state_dict': amp_scaler.state_dict() if amp_scaler is not None else None,
		'rank_runtime_states': gathered_runtime,
	}
	ckpt_path = os.path.join(checkpoint_dir, f'ckpt_update_{update_step}.pt')
	ckpt_tmp_path = f"{ckpt_path}.tmp"
	torch.save(state, ckpt_tmp_path)
	os.replace(ckpt_tmp_path, ckpt_path)
	latest_path = os.path.join(checkpoint_dir, 'latest.pt')
	latest_tmp_path = f"{latest_path}.tmp"
	shutil.copyfile(ckpt_path, latest_tmp_path)
	os.replace(latest_tmp_path, latest_path)
	return ckpt_path


def load_checkpoint(
	model,
	policy_optimizer,
	value_optimizer,
	policy_scheduler,
	value_scheduler,
	amp_scaler,
	batch_sizers: dict,
	checkpoint_path: str,
	device: torch.device,
	rank: int = 0,
) -> dict:
	"""Restore a versioned optimizer-update checkpoint without legacy migration."""
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
	state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	version = int(state.get('format_version', -1))
	if version != CHECKPOINT_FORMAT_VERSION:
		raise ValueError(
			f"unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}"
		)
	load_model = model.module if hasattr(model, 'module') else model
	load_model.load_state_dict(state['model_state_dict'], strict=True)
	policy_optimizer.load_state_dict(state['policy_optimizer_state_dict'])
	value_optimizer.load_state_dict(state['value_optimizer_state_dict'])
	policy_scheduler.load_state_dict(state['policy_scheduler_state_dict'])
	value_scheduler.load_state_dict(state['value_scheduler_state_dict'])
	if amp_scaler is not None and state['grad_scaler_state_dict'] is not None:
		amp_scaler.load_state_dict(state['grad_scaler_state_dict'])
	rank_runtime_states = state['rank_runtime_states']
	if rank >= len(rank_runtime_states):
		raise ValueError(
			f"checkpoint contains {len(rank_runtime_states)} rank states, cannot restore rank {rank}"
		)
	rank_runtime = rank_runtime_states[rank]
	for name, sizer in batch_sizers.items():
		if name not in rank_runtime['adaptive_batch_state']:
			raise ValueError(f"checkpoint is missing adaptive batch state: {name}")
		sizer.load_state_dict(rank_runtime['adaptive_batch_state'][name])
	restore_rng_state(rank_runtime['rng_state'], device)
	progress = dict(state['progress'])
	progress['a_max_ewma'] = (
		None
		if rank_runtime['a_max_ewma'] is None
		else rank_runtime['a_max_ewma'].to(device=device)
	)
	return progress


def create_lr_scheduler(optimizer, training_cfg, total_updates: int):
	"""Build the paper-configured learning-rate schedule."""
	schedule = str(getattr(training_cfg, 'lr_schedule', 'cosine')).strip().lower()
	if schedule == 'cosine':
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=int(total_updates),
			eta_min=0.0,
		)
	if schedule in {'constant', 'none'}:
		return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
	raise ValueError(f"unsupported training.lr_schedule: {schedule!r}")


def reconcile_lr_scheduler_horizon(scheduler, total_updates: int):
	"""Apply a resumed run's final update target to a loaded cosine scheduler."""
	total_updates = int(total_updates)
	if total_updates <= 0:
		raise ValueError("training.total_updates must be positive")
	if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
		return
	completed_steps = max(0, int(scheduler.last_epoch))
	if completed_steps > total_updates:
		raise ValueError(
			f"scheduler already completed {completed_steps} optimizer steps, "
			f"which exceeds training.total_updates={total_updates}"
		)
	scheduler.T_max = total_updates
	last_lrs = [
		scheduler.eta_min
		+ (base_lr - scheduler.eta_min)
		* (1.0 + math.cos(math.pi * completed_steps / total_updates))
		/ 2.0
		for base_lr in scheduler.base_lrs
	]
	for group, learning_rate in zip(scheduler.optimizer.param_groups, last_lrs):
		group['lr'] = learning_rate
	scheduler._last_lr = last_lrs

# ============================== 构建网络输入特征 ==============================


def normalize_to_minus1_1(x: torch.Tensor, min_val, max_val) -> torch.Tensor:
    """
    x输入可以是B,M,1批量数据
	将输入按区间 [min_val, max_val] 线性映射到 [-1, 1] 并裁剪：
    - x <= min_val -> -1
    - x >= max_val -> 1
    - 其余线性映射到 (-1, 1)
    当 max_val == min_val 时（退化区间）：
    - x < min_val -> -1, x > max_val -> 1, 等于 -> 0

    min_val / max_val 可为标量或与 x 可广播的张量。
    """
    min_t = torch.as_tensor(min_val, dtype=x.dtype, device=x.device)
    max_t = torch.as_tensor(max_val, dtype=x.dtype, device=x.device)
    denom = max_t - min_t
    # 避免除零：仅对非退化位置执行标准线性映射
    denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)
    y = (x - min_t) / denom_safe
    y = y * 2 - 1
    # 裁剪到 [-1, 1]
    y = torch.clamp(y, -1.0, 1.0)
    # 退化处理：max==min。这里避免 Python if 读取 GPU 标量造成同步。
    deg_mask = (denom == 0)
    y_degen = torch.where(
        x > max_t, torch.ones_like(x),
        torch.where(x < min_t, -torch.ones_like(x), torch.zeros_like(x))
    )
    return torch.where(deg_mask, y_degen, y)

def config_get(container, name: str, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def normalize_route_distance(distance: torch.Tensor, max_distance: float, mode: str = "log") -> torch.Tensor:
    """Normalize non-negative route distances while preserving long-range ordering."""
    max_distance = max(float(max_distance), 1e-6)
    distance = torch.clamp(distance, min=0.0)
    mode = str(mode or "log").lower()
    if mode == "linear":
        return normalize_to_minus1_1(distance, 0.0, max_distance)
    if mode == "sqrt":
        scaled = torch.sqrt(distance) / math.sqrt(max_distance)
    else:
        scaled = torch.log1p(distance) / math.log1p(max_distance)
    return torch.clamp(scaled * 2.0 - 1.0, -1.0, 1.0)

class FeatureBuildWorkspace:
    """Long-lived scratch/cache for feature construction hot paths."""

    def __init__(self, config: SimpleNamespace = None):
        self._bounds_cache = {}
        self._arange_cache = {}
        self._scratch = {}
        self.feature_chunks = 0
        if config is not None:
            self.configure(config)

    def configure(self, config: SimpleNamespace):
        self.schema = FeatureSchema.from_config(config)
        self.total_input_dim = self.schema.total_input_dim

    def reset_counters(self):
        self.feature_chunks = 0

    def clear_scratch(self):
        """Release shape-dependent buffers after a CUDA OOM backoff."""
        self._scratch.clear()
        self._arange_cache.clear()

    def mark_feature_chunk(self):
        self.feature_chunks += 1

    def bounds(self, name: str, dtype: torch.dtype, device: torch.device):
        key = (name, dtype, device)
        cached = self._bounds_cache.get(key)
        if cached is not None:
            return cached
        if name == 's13':
            bounds = (
                (-5.0, 5.0), (-math.pi, math.pi), (-0.2, 0.2), (-2.0, 30.0), (0.0, 30.0),
                (-0.7, 0.7), (-5.0, 5.0), (-4.0, 4.0),
                (1 / 1.5, 1.5), (1 / 1.25, 1.25), (1 / 1.25, 1.25),
                (0.8, 7.0), (0.8, 3.0),
            )
        elif name == 'reward':
            bounds = (
                (2, 12), (0, 3), (0, 3), (0, 0.1), (0.00025, 0.025),
                (0, 1), (0.00025, 0.0075), (-0.5, 0.5), (0.0025, 0.0025),
                (0.00025, 0.0075), (0, 1), (0.000025, 0.000025),
            )
        elif name == 'style':
            bounds = ((1 / 1.25, 1.25), (1 / 1.25, 1.25), (1 / 1.5, 1.5), (1 / 1.5, 1.5))
        else:
            raise KeyError(f"unknown feature bounds: {name}")
        min_t = torch.tensor([lo for lo, _ in bounds], device=device, dtype=dtype)
        max_t = torch.tensor([hi for _, hi in bounds], device=device, dtype=dtype)
        self._bounds_cache[key] = (min_t, max_t)
        return min_t, max_t

    def arange(self, n: int, device: torch.device, dtype: torch.dtype = torch.long):
        n = int(n)
        if n < 0:
            raise ValueError("arange length must be non-negative")
        key = (device, dtype)
        cached = self._arange_cache.get(key)
        if cached is None or cached.numel() < n:
            cached = torch.arange(n, device=device, dtype=dtype)
            self._arange_cache[key] = cached
        return cached[:n]

    def scratch(self, name: str, shape, device: torch.device, dtype: torch.dtype, fill_value=None) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        cached = self._scratch.get(name)
        reuse = (
            cached is not None
            and cached.device == device
            and cached.dtype == dtype
            and cached.dim() == len(shape)
            and all(int(cached.shape[i]) >= shape[i] for i in range(len(shape)))
        )
        if not reuse:
            cached = torch.empty(shape, device=device, dtype=dtype)
            self._scratch[name] = cached
        view = cached[tuple(slice(0, dim) for dim in shape)]
        if fill_value is not None:
            view.fill_(fill_value)
        return view


def _default_workspace(config: SimpleNamespace = None) -> FeatureBuildWorkspace:
    workspace = FeatureBuildWorkspace(config) if config is not None else FeatureBuildWorkspace()
    return workspace


def pad_or_truncate_flat(flat: torch.Tensor, target_size: int, pad_value: float = 0.0,
                         out: torch.Tensor = None) -> torch.Tensor:
    B, M, D = flat.shape
    if out is None:
        out = torch.full((B, M, target_size), pad_value, device=flat.device, dtype=flat.dtype)
    else:
        out.fill_(pad_value)
    copy_size = min(D, target_size)
    if copy_size > 0:
        out[:, :, :copy_size] = flat[:, :, :copy_size]
    return out

def normalize_point_set(points: torch.Tensor, target_size: int, element_dim: int = 2,
                        min_val: float = -100.0, max_val: float = 100.0,
                        out: torch.Tensor = None) -> torch.Tensor:
    if points is None or points.numel() == 0:
        if out is not None:
            out.fill_(FEATURE_PAD_VALUE)
            return out
        return None
    if points.dim() == 3:
        B, M, N = points.shape
        elements = points.view(B, M, N // element_dim, element_dim)
    else:
        elements = points
    B, M, num_elements, _ = elements.shape
    if out is None:
        out = torch.empty((B, M, target_size), device=elements.device, dtype=elements.dtype)
    out.fill_(FEATURE_PAD_VALUE)
    max_elements = min(num_elements, target_size // max(1, element_dim))
    if max_elements <= 0:
        return out
    elements = elements[:, :, :max_elements]
    valid = torch.isfinite(elements).all(dim=-1) & (elements.abs().sum(dim=-1) > 1e-6)
    normalized = normalize_to_minus1_1(torch.nan_to_num(elements, nan=0.0, posinf=max_val, neginf=min_val), min_val, max_val)
    target = out[:, :, :max_elements * element_dim].view(B, M, max_elements, element_dim)
    target.copy_(torch.where(valid.unsqueeze(-1), normalized, torch.full_like(normalized, FEATURE_PAD_VALUE)))
    return out

def normalize_s_features(s_t: torch.Tensor, target_size: int, vehicle_style: torch.Tensor = None,
                         control_state: torch.Tensor = None, workspace: FeatureBuildWorkspace = None,
                         out: torch.Tensor = None) -> torch.Tensor:
    """原文式 S(t): c,theta,kappa,v,v_lim,phi,a_long,a_lat,Cacc,Cthrottle,Csteer,l,w。"""
    B, M, _ = s_t.shape
    if out is None:
        out = torch.zeros(B, M, target_size, device=s_t.device, dtype=s_t.dtype)
    else:
        out.zero_()
    copy = min(s_t.shape[-1], target_size)
    if copy > 0:
        out[:, :, :copy] = s_t[:, :, :copy]
    if control_state is not None and target_size >= 8:
        control_state = control_state.to(device=s_t.device, dtype=s_t.dtype)
        out[:, :, 5:8] = control_state[:, :, :3]
    if vehicle_style is not None and target_size >= 11:
        vehicle_style = vehicle_style.to(device=s_t.device, dtype=s_t.dtype)
        out[:, :, 8] = vehicle_style[:, :, 2]   # Cacc
        out[:, :, 9] = vehicle_style[:, :, 0]   # Cthrottle
        out[:, :, 10] = vehicle_style[:, :, 1]  # Csteer

    if target_size >= 13:
        normalized = out.clone() if out is not None else torch.empty_like(out)
        workspace = workspace or _default_workspace()
        bounds_min, bounds_max = workspace.bounds('s13', out.dtype, out.device)
        copy_bounds = min(target_size, bounds_min.numel())
        normalized[:, :, :copy_bounds] = normalize_to_minus1_1(
            out[:, :, :copy_bounds],
            bounds_min[:copy_bounds],
            bounds_max[:copy_bounds],
        )
        if target_size > copy_bounds:
            normalized[:, :, copy_bounds:] = out[:, :, copy_bounds:]
        return normalized

    # 兼容旧7维 local state。
    normalized = torch.zeros_like(out)
    if target_size > 0:
        normalized[:, :, 0] = normalize_to_minus1_1(out[:, :, 0], -100, 100)
    if target_size > 1:
        normalized[:, :, 1] = out[:, :, 1]
    if target_size > 2:
        normalized[:, :, 2] = out[:, :, 2]
    if target_size > 3:
        normalized[:, :, 3] = normalize_to_minus1_1(out[:, :, 3], -2, 20)
    if target_size > 4:
        normalized[:, :, 4] = normalize_to_minus1_1(out[:, :, 4], 0.8, 7)
    if target_size > 5:
        normalized[:, :, 5] = normalize_to_minus1_1(out[:, :, 5], 0.8, 3)
    if target_size > 6:
        normalized[:, :, 6] = out[:, :, 6]
    return normalized

def _is_navigation_packet(navigation: torch.Tensor) -> bool:
    return navigation is not None and navigation.dim() == 4 and navigation.shape[-1] >= 3


def build_lane_map_features(w_lanes_local: torch.Tensor, navigation: torch.Tensor,
                            target_size: int, element_dim: int = 7,
                            goal_slots: int = 0, out: torch.Tensor = None,
                            route_distance_norm: str = "log",
                            route_abs_distance_max: float = 12000.0,
                            route_rel_distance_max: float = 12000.0) -> torch.Tensor:
    """构造原文式 W_lane: 位置、车道方向、车道宽度、到下一目标的绝对/相对距离。"""
    if w_lanes_local is None or w_lanes_local.numel() == 0:
        if out is not None:
            out.fill_(FEATURE_PAD_VALUE)
            return out
        return None
    lanes = w_lanes_local
    if lanes.dim() == 3:
        B, M, N = lanes.shape
        if N % 5 == 0:
            raw_dim = 5
        elif N % 2 == 0:
            raw_dim = 2
        else:
            raw_dim = lanes.shape[-1]
        lanes = lanes.view(B, M, N // raw_dim, raw_dim)
    B, M, K, raw_dim = lanes.shape
    if out is None:
        out = torch.empty((B, M, target_size), device=lanes.device, dtype=lanes.dtype)
    out.fill_(FEATURE_PAD_VALUE)
    K_eff = min(K, target_size // max(1, element_dim))
    if K_eff <= 0:
        return out
    lanes = lanes[:, :, :K_eff]
    K = K_eff
    lane_xy = lanes[..., :2]
    valid = torch.isfinite(lane_xy).all(dim=-1)

    if raw_dim >= 4:
        lane_dir = lanes[..., 2:4].clamp(-1.0, 1.0)
        valid = valid & torch.isfinite(lanes[..., 2:4]).all(dim=-1)
    else:
        lane_dir = torch.zeros(B, M, K, 2, device=lanes.device, dtype=lanes.dtype)
    if raw_dim >= 5:
        lane_width = lanes[..., 4]
        valid = valid & torch.isfinite(lane_width) & (lane_width > 0)
    else:
        lane_width = torch.zeros(B, M, K, device=lanes.device, dtype=lanes.dtype)
        valid = valid & (lane_xy.abs().sum(dim=-1) > 1e-6)

    nav_packet = _is_navigation_packet(navigation)
    route_abs = route_rel = route_valid = None
    if nav_packet and goal_slots <= 0 and navigation.shape[2] >= K:
        goal_slots = max(0, navigation.shape[2] - K)
    if nav_packet and navigation.shape[2] >= goal_slots + K:
        nav = navigation.to(device=lanes.device, dtype=lanes.dtype)
        route_rows = nav[:, :, goal_slots:goal_slots + K, :]
        route_abs = route_rows[..., 0]
        route_rel = route_rows[..., 1]
        route_valid = (route_rows[..., 2] > 0.5) & torch.isfinite(route_abs) & torch.isfinite(route_rel)

    if nav_packet and goal_slots > 0 and navigation.shape[2] >= goal_slots:
        nav = navigation.to(device=lanes.device, dtype=lanes.dtype)
        goal_local = nav[:, :, 0, :2]
        has_goal = nav[:, :, 0, 2] > 0.5
    else:
        goal_local = torch.zeros(B, M, 2, device=lanes.device, dtype=lanes.dtype)
        has_goal = torch.zeros(B, M, dtype=torch.bool, device=lanes.device)

    if route_abs is not None:
        distance_valid = valid & route_valid
        goal_dist = torch.where(distance_valid, route_abs, torch.zeros_like(route_abs))
        rel_goal_dist = torch.where(distance_valid, route_rel, torch.zeros_like(route_rel))
    else:
        # 兼容旧输入：没有 W_lane 图距离 packet 时才退回到显式目标的欧氏距离。
        euclidean_goal_dist = torch.norm(lane_xy - goal_local.unsqueeze(2), dim=-1)
        distance_valid = valid & has_goal.unsqueeze(-1)
        goal_dist = euclidean_goal_dist
        goal_dist = torch.where(distance_valid, goal_dist, torch.zeros_like(goal_dist))
        masked_goal_dist = goal_dist.masked_fill(~distance_valid, float('inf'))
        min_goal_dist = masked_goal_dist.amin(dim=2)
        min_goal_dist = torch.where(torch.isfinite(min_goal_dist), min_goal_dist, torch.zeros_like(min_goal_dist))
        rel_goal_dist = goal_dist - min_goal_dist.unsqueeze(-1)

    lane_out = out[:, :, :K * element_dim].view(B, M, K, element_dim)
    if element_dim > 0:
        lane_out[..., 0] = normalize_to_minus1_1(lane_xy[..., 0], -200, 200)
    if element_dim > 1:
        lane_out[..., 1] = normalize_to_minus1_1(lane_xy[..., 1], -200, 200)
    if element_dim > 2:
        dir_end = min(element_dim, 4)
        lane_out[..., 2:dir_end] = lane_dir[..., :dir_end - 2]
    if element_dim > 4:
        lane_out[..., 4] = normalize_to_minus1_1(lane_width, 0.0, 8.0)
    if element_dim > 5:
        goal_dist_norm = normalize_route_distance(
            goal_dist,
            route_abs_distance_max,
            mode=route_distance_norm,
        )
        lane_out[..., 5] = torch.where(distance_valid, goal_dist_norm, torch.full_like(goal_dist_norm, FEATURE_PAD_VALUE))
    if element_dim > 6:
        rel_goal_dist_norm = normalize_route_distance(
            rel_goal_dist,
            route_rel_distance_max,
            mode=route_distance_norm,
        )
        lane_out[..., 6] = torch.where(distance_valid, rel_goal_dist_norm, torch.full_like(rel_goal_dist_norm, FEATURE_PAD_VALUE))
    lane_out.masked_fill_(~valid.unsqueeze(-1), FEATURE_PAD_VALUE)
    return out

def build_network_features(agents_state: torch.Tensor, 
                           neighbors_local: torch.Tensor, 
                           w_lanes_local: torch.Tensor, 
                           w_boundaries_local: torch.Tensor,
                           navigation: torch.Tensor,
                           stop_lines: torch.Tensor,
                           reward_coef: torch.Tensor,
                           config: SimpleNamespace,
                           vehicle_style: torch.Tensor = None,
                           control_state: torch.Tensor = None,
                           map_dropout: dict = None,
                           workspace: FeatureBuildWorkspace = None,
                           out: torch.Tensor = None) -> torch.Tensor:
    """
    将拆解后的观测组件构建为网络输入的特征张量
    Args:
        agents_state: (B, M, S_dim) - 原文式 S(t) 局部状态
        neighbors_local: (B, M, K, neighbor_dim) - 邻居相对状态，active 位于最后一维
        w_lanes_local: (B, M, N_lanes, lane_dim) - map lane raw feature
        w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标
        navigation: (B, M, goal_slots + lane_slots, 3) - 显式 G(t) 与 W_lane 图路由距离
        stop_lines: (B, M, num_stop_lines * 4) - 每条停止线的两个局部坐标端点
        reward_coef: (B, M, 12) - 原文式 reward conditioning
        config: 配置对象
    Returns:
        torch.Tensor: 形状为 (B, M, total_input_dim) 的网络输入特征张量
    """
    batch_size, max_agents, _ = agents_state.shape
    if workspace is None:
        workspace = _default_workspace(config)
    elif not hasattr(workspace, 'total_input_dim'):
        workspace.configure(config)
    w_lanes_local, w_boundaries_local = apply_map_dropout_components(
        w_lanes_local,
        w_boundaries_local,
        map_dropout,
        inplace=True,
    )
    
    schema = workspace.schema
    total_input_dim = schema.total_input_dim
    
    # 初始化输出张量
    if out is None:
        features_tensor = workspace.scratch(
            'network_features',
            (batch_size, max_agents, total_input_dim),
            agents_state.device,
            agents_state.dtype,
            fill_value=0.0,
        )
    else:
        features_tensor = out
        features_tensor.zero_()
    
    # 1. 构建简单特征：S(t), 显式 G(t), reward 参数和车辆风格参数。
    goal_slots = 0
    
    # S(t): c, theta, kappa, v, v_lim, phi, a_long, a_lat, Cacc, Cthrottle, Csteer, l, w.
    state_group = schema.group('state')
    state_slice = schema.flat_slice('state')
    features_tensor[:, :, state_slice] = normalize_s_features(
        agents_state,
        state_group.flat_dim,
        vehicle_style=vehicle_style,
        control_state=control_state,
        workspace=workspace,
        out=features_tensor[:, :, state_slice],
    )
    
    goal_group = schema.group('goal')
    goal_slice = schema.flat_slice('goal')
    if navigation is None:
        goal_vector = features_tensor[:, :, goal_slice]
        goal_vector.zero_()
    elif _is_navigation_packet(navigation):
        nav = navigation.to(device=agents_state.device, dtype=agents_state.dtype)
        goal_slots = max(1, goal_group.flat_dim // 2)
        goal_rows = nav[:, :, :goal_slots, :]
        goal_valid = goal_rows[..., 2] > 0.5
        goal_xy = torch.where(goal_valid.unsqueeze(-1), goal_rows[..., :2], torch.zeros_like(goal_rows[..., :2]))
        goal_vector = pad_or_truncate_flat(
            goal_xy.flatten(start_dim=2),
            goal_group.flat_dim,
            pad_value=0.0,
            out=features_tensor[:, :, goal_slice],
        )
    else:
        goal_vector = features_tensor[:, :, goal_slice]
        goal_vector.zero_()
    features_tensor[:, :, goal_slice] = normalize_to_minus1_1(goal_vector, -200, 200)

    # reward系数: 原文式12维 C_reward，顺序对应 RewardParameterSampler.sample_all_parameters。
    reward_group = schema.group('reward')
    reward_slice = schema.flat_slice('reward')
    reward_coef = reward_coef.to(device=agents_state.device, dtype=agents_state.dtype)
    reward_out = features_tensor[:, :, reward_slice]
    reward_out.zero_()
    reward_min, reward_max = workspace.bounds('reward', agents_state.dtype, agents_state.device)
    copy_reward = min(reward_coef.shape[-1], reward_group.flat_dim, reward_min.numel())
    if copy_reward > 0:
        reward_out[:, :, :copy_reward] = normalize_to_minus1_1(
            reward_coef[:, :, :copy_reward],
            reward_min[:copy_reward],
            reward_max[:copy_reward],
        )

    # 车辆风格参数: 4维 - 从agents_state中提取
    style_group = schema.group('vehicle_style')
    style_slice = schema.flat_slice('vehicle_style')
    if vehicle_style is None:
        vehicle_style = torch.ones(
            batch_size,
            max_agents,
            style_group.flat_dim,
            device=agents_state.device,
            dtype=agents_state.dtype,
        )
    else:
        vehicle_style = vehicle_style.to(device=agents_state.device, dtype=agents_state.dtype)
    style_out = features_tensor[:, :, style_slice]
    style_out.zero_()
    style_min, style_max = workspace.bounds('style', agents_state.dtype, agents_state.device)
    copy_style = min(vehicle_style.shape[-1], style_group.flat_dim, style_min.numel())
    if copy_style > 0:
        style_out[:, :, :copy_style] = normalize_to_minus1_1(
            vehicle_style[:, :, :copy_style],
            style_min[:copy_style],
            style_max[:copy_style],
        )
    
    # road_boundary: 原文使用最近80个boundary coarse features
    boundary_group = schema.group('road_boundary')
    boundary_out = features_tensor[:, :, schema.flat_slice('road_boundary')]
    w_boundaries_flat = normalize_point_set(
        w_boundaries_local,
        boundary_group.flat_dim,
        min_val=-200.0,
        max_val=200.0,
        out=boundary_out,
    )
    if w_boundaries_flat is None:
        boundary_out.fill_(FEATURE_PAD_VALUE)
    
    # lane_points: 原文式 map lane feature，每个元素包含位置、方向、宽度、目标距离。
    lane_group = schema.group('lane_points')
    training_cfg = config_get(config, 'training', SimpleNamespace())
    navigation_cfg = config_get(training_cfg, 'navigation', SimpleNamespace())
    route_distance_norm = config_get(navigation_cfg, 'route_distance_norm', 'log')
    route_abs_distance_max = float(config_get(navigation_cfg, 'route_abs_distance_max', 12000.0))
    route_rel_distance_max = float(config_get(navigation_cfg, 'route_rel_distance_max', 12000.0))
    lane_out = features_tensor[:, :, schema.flat_slice('lane_points')]
    w_lanes_flat = build_lane_map_features(
        w_lanes_local,
        navigation,
        lane_group.flat_dim,
        lane_group.element_dim,
        goal_slots=goal_slots,
        route_distance_norm=route_distance_norm,
        route_abs_distance_max=route_abs_distance_max,
        route_rel_distance_max=route_rel_distance_max,
        out=lane_out,
    )
    if w_lanes_flat is None:
        lane_out.fill_(FEATURE_PAD_VALUE)
    
    # stop_lines: 20维 - 使用停止线信息
    stop_group = schema.group('stop_lines')
    stop_out = features_tensor[:, :, schema.flat_slice('stop_lines')]
    if stop_lines is not None and stop_lines.numel() > 0:
        stop_lines_flat = normalize_point_set(
            stop_lines.to(device=agents_state.device, dtype=agents_state.dtype),
            stop_group.flat_dim,
            out=stop_out,
        )
        if stop_lines_flat is None:
            stop_out.fill_(FEATURE_PAD_VALUE)
    else:
        stop_out.fill_(FEATURE_PAD_VALUE)
    
    # other_agents: 使用邻居位置、朝向、速度、尺寸、z 与 active mask
    other_group = schema.group('other_agents')
    # 将邻居信息按通道做归一化后再展平并填充，active=0 的 padding 不参与网络 maxpool。
    neighbors_local = neighbors_local.to(device=agents_state.device, dtype=agents_state.dtype)
    neighbor_dim = neighbors_local.shape[-1]
    if neighbor_dim != other_group.element_dim:
        raise ValueError(
            f"other_agents source width={neighbor_dim} does not match schema "
            f"element_dim={other_group.element_dim}"
        )
    other_out = features_tensor[:, :, schema.flat_slice('other_agents')]
    other_out.fill_(FEATURE_PAD_VALUE)
    neighbor_slots = min(neighbors_local.shape[2], other_group.flat_dim // max(1, neighbor_dim))
    neighbors_proc = other_out[:, :, :neighbor_slots * neighbor_dim].view(batch_size, max_agents, neighbor_slots, neighbor_dim)
    neighbors_src = neighbors_local[:, :, :neighbor_slots]
    if neighbor_dim >= 10:
        neighbors_proc[:, :, :, 0] = normalize_to_minus1_1(neighbors_src[:, :, :, 0], -200, 200)
        neighbors_proc[:, :, :, 1] = normalize_to_minus1_1(neighbors_src[:, :, :, 1], -200, 200)
        neighbors_proc[:, :, :, 2] = torch.clamp(neighbors_src[:, :, :, 2], -1.0, 1.0)
        neighbors_proc[:, :, :, 3] = torch.clamp(neighbors_src[:, :, :, 3], -1.0, 1.0)
        neighbors_proc[:, :, :, 4] = normalize_to_minus1_1(neighbors_src[:, :, :, 4], -60, 60)
        neighbors_proc[:, :, :, 5] = normalize_to_minus1_1(neighbors_src[:, :, :, 5], -60, 60)
        neighbors_proc[:, :, :, 6] = normalize_to_minus1_1(neighbors_src[:, :, :, 6], 0.8, 7)
        neighbors_proc[:, :, :, 7] = normalize_to_minus1_1(neighbors_src[:, :, :, 7], 0.8, 3)
        neighbors_proc[:, :, :, 8] = normalize_to_minus1_1(neighbors_src[:, :, :, 8], -10, 10)
        neighbors_proc[:, :, :, 9] = neighbors_src[:, :, :, 9]
    else:
        neighbors_proc[:, :, :, 0] = normalize_to_minus1_1(neighbors_src[:, :, :, 0], -100, 100)
        neighbors_proc[:, :, :, 1] = normalize_to_minus1_1(neighbors_src[:, :, :, 1], -100, 100)
        if neighbor_dim > 2:
            neighbors_proc[:, :, :, 2] = normalize_to_minus1_1(neighbors_src[:, :, :, 2], -60, 60)
        if neighbor_dim > 3:
            neighbors_proc[:, :, :, 3] = normalize_to_minus1_1(neighbors_src[:, :, :, 3], -60, 60)
        if neighbor_dim > 4:
            neighbors_proc[:, :, :, 4] = normalize_to_minus1_1(neighbors_src[:, :, :, 4], 0.8, 7)
        if neighbor_dim > 5:
            neighbors_proc[:, :, :, 5] = normalize_to_minus1_1(neighbors_src[:, :, :, 5], 0.8, 3)
        if neighbor_dim > 6:
            neighbors_proc[:, :, :, 6] = neighbors_src[:, :, :, 6]
    
    return features_tensor

# ============================== 检查GPU信息 ==============================
def check_gpu_info(print_info: bool = True):
	"""
	检查GPU信息和CUDA支持情况

	Args:
		print_info: 是否打印函数内部的日志（默认True）。
	"""
	def log(*args, **kws):
		if print_info:
			print(*args, **kws)

	log("🔍 GPU 信息检测...")
	# 检查CUDA是否可用
	if torch.cuda.is_available():
		log("✅ CUDA 可用")
		# 获取CUDA版本
		cuda_version = torch.version.cuda
		log(f"📋 CUDA 版本: {cuda_version}")
		# 获取GPU数量
		gpu_count = torch.cuda.device_count()
		log(f"🎮 GPU 数量: {gpu_count}")
		# 获取当前GPU设备
		current_device = torch.cuda.current_device()
		log(f"🎯 当前GPU设备: {current_device}")
		# 获取GPU名称
		gpu_name = torch.cuda.get_device_name(current_device)
		log(f"🏷️  GPU名称: {gpu_name}")
		# 获取GPU内存信息
		gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
		gpu_memory_gb = gpu_memory / (1024**3)
		log(f"💾 GPU内存: {gpu_memory_gb:.2f} GB")
		# 检查分布式训练支持
		if dist.is_available():
			log("✅ PyTorch分布式训练支持可用")
			# 检查NCCL后端
			if dist.is_nccl_available():
				log("✅ NCCL后端可用")
			else:
				log("❌ NCCL后端不可用")
			# 检查GLOO后端
			if dist.is_gloo_available():
				log("✅ GLOO后端可用")
			else:
				log("❌ GLOO后端不可用")
		else:
			log("❌ PyTorch分布式训练支持不可用")
		# 显示所有GPU的详细信息
		log("\n📊 所有GPU详细信息:")
		for i in range(gpu_count):
			props = torch.cuda.get_device_properties(i)
			log(f"  GPU {i}: {props.name}")
			log(f"    内存: {props.total_memory / (1024**3):.2f} GB")
			log(f"    计算能力: {props.major}.{props.minor}")
			log(f"    多处理器数量: {props.multi_processor_count}")
		# 返回CUDA rank列表
		cuda_ranks = list(range(gpu_count))
		return True, cuda_ranks
	else:
		log("❌ CUDA 不可用")
		log("📋 PyTorch版本:", torch.__version__)
		log("💡 请确保已正确安装CUDA和对应版本的PyTorch")
		return False, []

def unwrap_model(model):
	return model.module if hasattr(model, 'module') else model

def get_policy_parameters(model):
	base = unwrap_model(model)
	if hasattr(base, 'policy_parameters'):
		return list(base.policy_parameters())
	return list(base.policy_network.parameters())

def get_value_parameters(model):
	base = unwrap_model(model)
	if hasattr(base, 'value_parameters'):
		return list(base.value_parameters())
	return list(base.value_network.parameters())

def forward_model(model, features_tensor, mode="both", chunk_agents: int = None):
	if (
		chunk_agents is None
		or chunk_agents <= 0
		or features_tensor is None
		or features_tensor.dim() != 3
	):
		return model(features_tensor, mode=mode)

	B, M, D = features_tensor.shape
	total_agents = B * M
	if total_agents <= int(chunk_agents):
		return model(features_tensor, mode=mode)

	flat_features = features_tensor.reshape(total_agents, D)
	chunk_logits = []
	chunk_values = []
	chunk_outputs = []
	for start in range(0, total_agents, int(chunk_agents)):
		chunk = flat_features[start:start + int(chunk_agents)].view(-1, 1, D)
		out = model(chunk, mode=mode)
		if isinstance(out, tuple):
			logits, values = out
			chunk_logits.append(logits.reshape(logits.shape[0], *logits.shape[2:]))
			chunk_values.append(values.reshape(values.shape[0], *values.shape[2:]) if values.dim() > 2 else values.reshape(values.shape[0]))
		else:
			chunk_outputs.append(out.reshape(out.shape[0], *out.shape[2:]) if out.dim() > 2 else out.reshape(out.shape[0]))

	if chunk_logits:
		logits = torch.cat(chunk_logits, dim=0).view(B, M, -1)
		values = torch.cat(chunk_values, dim=0).view(B, M)
		return logits, values
	output = torch.cat(chunk_outputs, dim=0)
	if output.dim() == 1:
		return output.view(B, M)
	return output.view(B, M, *output.shape[1:])

def make_autocast_context(device: torch.device, precision: str):
	use_amp = device.type == 'cuda' and str(precision).lower() in {"16-bit", "fp16", "float16", "amp"}
	if not use_amp:
		return nullcontext()
	if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
		return torch.amp.autocast(device_type='cuda', enabled=True)
	return torch.cuda.amp.autocast(enabled=True)

def make_grad_scaler(device: torch.device, precision: str):
	use_amp = device.type == 'cuda' and str(precision).lower() in {"16-bit", "fp16", "float16", "amp"}
	if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
		try:
			return torch.amp.GradScaler('cuda', enabled=use_amp)
		except TypeError:
			return torch.amp.GradScaler(enabled=use_amp)
	return torch.cuda.amp.GradScaler(enabled=use_amp)

def cuda_memory_stats(device: torch.device) -> dict:
	if device.type != 'cuda' or not torch.cuda.is_available():
		return {
			'max_memory_allocated_mb': 0.0,
			'max_memory_reserved_mb': 0.0,
		}
	return {
		'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
		'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
	}

def make_update_stats(reason: str = "", device: torch.device = None) -> dict:
	stats = {
		'did_optimizer_step': False,
		'skip_reason': reason,
		'num_candidates': 0,
		'num_selected': 0,
		'ppo_microbatch_size': 0,
		'ppo_microbatch_count': 0,
		'ppo_microbatch_retries': 0,
		'ppo_memory_adapted': False,
		'num_epochs': 0,
		'policy_loss': None,
		'value_loss': None,
		'entropy': None,
		'approx_kl': None,
		'old_approx_kl': None,
		'clip_frac': None,
		'ratio_mean': None,
		'ratio_min': None,
		'ratio_max': None,
		'max_action_prob_mean': None,
		'ppo_update_time_s': 0.0,
		'ppo_feature_cache_build_ms': 0.0,
		'ppo_feature_cache_location': None,
		'ppo_feature_cache_oom_retries': 0,
		'ppo_feature_cache_offloads': 0,
		'ddp_gradient_buckets': 0,
		'max_memory_allocated_mb': 0.0,
		'max_memory_reserved_mb': 0.0,
	}
	if device is not None:
		stats.update(cuda_memory_stats(device))
	return stats

def get_profile_cfg(config):
	training_cfg = getattr(config, 'training', SimpleNamespace())
	profile_cfg = getattr(training_cfg, 'profile', SimpleNamespace())
	return profile_cfg

def profile_enabled(config) -> bool:
	return bool(getattr(get_profile_cfg(config), 'enabled', False))

def profile_cuda_sync(config) -> bool:
	return bool(getattr(get_profile_cfg(config), 'cuda_sync', False))

def profile_log_interval(config) -> int:
	return int(getattr(get_profile_cfg(config), 'log_interval', 10))

def maybe_cuda_sync(device: torch.device, config):
	if device.type == 'cuda' and profile_cuda_sync(config):
		torch.cuda.synchronize(device)

def profile_timer_start(device: torch.device, config) -> float:
	maybe_cuda_sync(device, config)
	return time.time()

def profile_elapsed_ms(start_time: float, device: torch.device, config) -> float:
	maybe_cuda_sync(device, config)
	return (time.time() - start_time) * 1000.0

def format_profile(profile_dict: dict) -> str:
	if not profile_dict:
		return ""
	return ", ".join(f"{key}={value:.2f}ms" for key, value in profile_dict.items())

REWARD_COMPONENT_NAMES = (
	'goal',
	'collision',
	'offroad',
	'comfort',
	'lane_align',
	'lane_center',
	'velocity',
	'reverse',
	'stop_line',
	'timestep',
)

REWARD_COMPONENT_LABELS = {
	'goal': 'goal',
	'collision': 'coll_r',
	'offroad': 'off_r',
	'comfort': 'comfort',
	'lane_align': 'align',
	'lane_center': 'center',
	'velocity': 'vel',
	'reverse': 'rev',
	'stop_line': 'stop',
	'timestep': 'time',
}

def get_diagnostics_cfg(config):
	training_cfg = getattr(config, 'training', SimpleNamespace())
	return getattr(training_cfg, 'diagnostics', SimpleNamespace())

def diagnostics_enabled(config, name: str = None) -> bool:
	diag_cfg = get_diagnostics_cfg(config)
	if not bool(getattr(diag_cfg, 'enabled', False)):
		return False
	if name is None:
		return True
	return bool(getattr(diag_cfg, name, True))

def scalar_item(value, default=0.0):
	if value is None:
		return default
	if torch.is_tensor(value):
		return float(value.detach().cpu().item())
	return float(value)

def tensor_stats(tensor: torch.Tensor) -> dict:
	if tensor is None or tensor.numel() == 0:
		return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
	t = tensor.detach().float()
	return {
		'mean': scalar_item(t.mean()),
		'std': scalar_item(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
		'min': scalar_item(t.min()),
		'max': scalar_item(t.max()),
	}

def action_label(action_values: torch.Tensor, action_idx: int) -> str:
	if torch.is_tensor(action_values) and 0 <= int(action_idx) < int(action_values.shape[0]):
		val = action_values[int(action_idx)].detach().cpu().tolist()
		if len(val) >= 2:
			return f"{int(action_idx)}[{val[0]:.0f},{val[1]:.0f}]"
	return str(int(action_idx))

def top_action_summary(counts: torch.Tensor, action_values: torch.Tensor = None, limit: int = 6,
					   numerators: dict = None) -> str:
	if counts is None or counts.numel() == 0:
		return "none"
	counts_cpu = counts.detach().to('cpu', dtype=torch.float64)
	total = float(counts_cpu.sum().item())
	if total <= 0:
		return "none"
	limit = min(int(limit), int(counts_cpu.numel()))
	top_vals, top_idx = torch.topk(counts_cpu, k=limit)
	parts = []
	for c, idx in zip(top_vals.tolist(), top_idx.tolist()):
		if c <= 0:
			continue
		extras = []
		if numerators:
			for name, values in numerators.items():
				if values is None:
					continue
				num = float(values.detach().to('cpu', dtype=torch.float64)[int(idx)].item())
				extras.append(f"{name}={num / max(c, 1.0):.4f}")
		extra_text = ("," + ",".join(extras)) if extras else ""
		parts.append(f"{action_label(action_values, int(idx))}:{int(c)}({100.0*c/total:.1f}%{extra_text})")
	return "; ".join(parts) if parts else "none"

def reward_component_summary(component_sums: dict, samples: float) -> str:
	parts = []
	for name in REWARD_COMPONENT_NAMES:
		value = component_sums.get(name) if component_sums else None
		if value is None:
			continue
		label = REWARD_COMPONENT_LABELS.get(name, name)
		parts.append(f"{label}={scalar_item(value) / max(samples, 1.0):.5f}")
	return ", ".join(parts) if parts else "none"

def top_action_component_summary(counts: torch.Tensor, action_component_sums: dict,
								 action_values: torch.Tensor = None, limit: int = 3) -> str:
	if counts is None or counts.numel() == 0 or not action_component_sums:
		return "none"
	counts_cpu = counts.detach().to('cpu', dtype=torch.float64)
	total = float(counts_cpu.sum().item())
	if total <= 0:
		return "none"
	limit = min(int(limit), int(counts_cpu.numel()))
	top_vals, top_idx = torch.topk(counts_cpu, k=limit)
	parts = []
	for c, idx in zip(top_vals.tolist(), top_idx.tolist()):
		if c <= 0:
			continue
		component_parts = []
		for name in REWARD_COMPONENT_NAMES:
			values = action_component_sums.get(name)
			if values is None:
				continue
			label = REWARD_COMPONENT_LABELS.get(name, name)
			num = float(values.detach().to('cpu', dtype=torch.float64)[int(idx)].item())
			component_parts.append(f"{label}={num / max(c, 1.0):.4f}")
		parts.append(f"{action_label(action_values, int(idx))}:(" + ",".join(component_parts) + ")")
	return "; ".join(parts) if parts else "none"

def init_rollout_diagnostics(config, device: torch.device):
	if not diagnostics_enabled(config, 'rollout'):
		return None
	num_actions = int(getattr(getattr(config, 'training').network, 'num_actions', 12))
	return {
		'steps': torch.zeros((), device=device, dtype=torch.long),
		'samples': torch.zeros((), device=device, dtype=torch.long),
		'reward_sum': torch.zeros((), device=device, dtype=torch.float32),
		'reward_sq_sum': torch.zeros((), device=device, dtype=torch.float32),
		'reward_min': torch.full((), float('inf'), device=device, dtype=torch.float32),
		'reward_max': torch.full((), float('-inf'), device=device, dtype=torch.float32),
		'negative_rewards': torch.zeros((), device=device, dtype=torch.long),
		'dones': torch.zeros((), device=device, dtype=torch.long),
		'collisions': torch.zeros((), device=device, dtype=torch.long),
		'offroads': torch.zeros((), device=device, dtype=torch.long),
		'final_goals': torch.zeros((), device=device, dtype=torch.long),
		'intermediate_goals': torch.zeros((), device=device, dtype=torch.long),
		'action_counts': torch.zeros((num_actions,), device=device, dtype=torch.long),
		'action_reward_sum': torch.zeros((num_actions,), device=device, dtype=torch.float32),
		'action_done_counts': torch.zeros((num_actions,), device=device, dtype=torch.long),
		'action_collision_counts': torch.zeros((num_actions,), device=device, dtype=torch.long),
		'action_offroad_counts': torch.zeros((num_actions,), device=device, dtype=torch.long),
		'reward_component_sums': {
			name: torch.zeros((), device=device, dtype=torch.float32)
			for name in REWARD_COMPONENT_NAMES
		},
		'action_component_sums': {
			name: torch.zeros((num_actions,), device=device, dtype=torch.float32)
			for name in REWARD_COMPONENT_NAMES
		},
	}

def update_rollout_diagnostics(diag: dict, simulator, alive_mask: torch.Tensor,
							   reward: torch.Tensor, done: torch.Tensor, actions: torch.Tensor):
	if diag is None:
		return
	info = getattr(simulator, 'last_step_train_info', None) or {}
	mask = info.get('effective_mask', alive_mask)
	mask = mask.to(device=reward.device, dtype=torch.bool)
	if mask.numel() == 0:
		return
	reward_value = reward.detach().float()
	mask_float = mask.to(dtype=reward_value.dtype)
	reward_masked = reward_value * mask_float
	diag['steps'] += 1
	diag['samples'] += mask.sum()
	diag['reward_sum'] += reward_masked.sum()
	diag['reward_sq_sum'] += (reward_masked * reward_masked).sum()
	diag['reward_min'] = torch.minimum(
		diag['reward_min'], reward_value.masked_fill(~mask, float('inf')).min()
	)
	diag['reward_max'] = torch.maximum(
		diag['reward_max'], reward_value.masked_fill(~mask, float('-inf')).max()
	)
	diag['negative_rewards'] += ((reward_value < 0) & mask).sum()

	def add_mask_count(name: str, fallback: torch.Tensor = None):
		mask_value = info.get(name, fallback)
		if mask_value is not None:
			return (mask_value.to(device=reward.device, dtype=torch.bool) & mask).sum()
		return torch.zeros((), device=reward.device, dtype=torch.long)

	done_mask = add_mask_count('done_mask', done)
	collision_mask = add_mask_count('collision_mask')
	offroad_mask = add_mask_count('offroad_mask')
	final_goal_mask = add_mask_count('final_goal_mask')
	intermediate_goal_mask = add_mask_count('intermediate_goal_mask')
	diag['dones'] += done_mask
	diag['collisions'] += collision_mask
	diag['offroads'] += offroad_mask
	diag['final_goals'] += final_goal_mask
	diag['intermediate_goals'] += intermediate_goal_mask

	num_actions = int(diag['action_counts'].shape[0])
	action_flat = actions.detach().to(torch.long).reshape(-1).clamp(0, num_actions - 1)
	mask_flat = mask.reshape(-1)
	diag['action_counts'].scatter_add_(0, action_flat, mask_flat.to(torch.long))
	diag['action_reward_sum'].scatter_add_(0, action_flat, reward_masked.reshape(-1))
	reward_components = info.get('reward_components', {}) or {}
	for name, component in reward_components.items():
		if name not in diag['reward_component_sums']:
			continue
		component_masked = component.detach().to(
			device=reward.device, dtype=torch.float32
		) * mask_float
		diag['reward_component_sums'][name] += component_masked.sum()
		diag['action_component_sums'][name].scatter_add_(
			0, action_flat, component_masked.reshape(-1)
		)
	for name, key in (
		('action_done_counts', 'done_mask'),
		('action_collision_counts', 'collision_mask'),
		('action_offroad_counts', 'offroad_mask'),
	):
		event_mask = info.get(key, done if key == 'done_mask' else None)
		if event_mask is None:
			continue
		event_flat = (
			event_mask.to(device=reward.device, dtype=torch.bool) & mask
		).reshape(-1)
		diag[name].scatter_add_(0, action_flat, event_flat.to(torch.long))

def rollout_diagnostic_metrics(diag: dict) -> dict:
	"""Convert accumulated rollout diagnostics into scalar dashboard metrics."""
	if diag is None:
		return {}
	samples = scalar_item(diag['samples'])
	if samples <= 0:
		return {}
	reward_mean = scalar_item(diag['reward_sum']) / samples
	reward_var = max(0.0, scalar_item(diag['reward_sq_sum']) / samples - reward_mean * reward_mean)
	metrics = {
		'rollout/steps': int(scalar_item(diag['steps'])),
		'rollout/samples': int(samples),
		'rollout/reward_mean': reward_mean,
		'rollout/reward_std': math.sqrt(reward_var),
		'rollout/reward_min': scalar_item(diag['reward_min']) if torch.isfinite(diag['reward_min']) else 0.0,
		'rollout/reward_max': scalar_item(diag['reward_max']) if torch.isfinite(diag['reward_max']) else 0.0,
		'rollout/negative_rate': scalar_item(diag['negative_rewards']) / samples,
		'rollout/done_rate': scalar_item(diag['dones']) / samples,
		'rollout/collision_rate': scalar_item(diag['collisions']) / samples,
		'rollout/offroad_rate': scalar_item(diag['offroads']) / samples,
		'rollout/final_goal_rate': scalar_item(diag['final_goals']) / samples,
		'rollout/intermediate_goal_rate': scalar_item(diag['intermediate_goals']) / samples,
	}
	for name, value in diag.get('reward_component_sums', {}).items():
		metrics[f'reward/{name}_mean'] = scalar_item(value) / samples
	return metrics

def print_rollout_diagnostics(diag: dict, action_values: torch.Tensor = None, prefix: str = "📈 rollout诊断"):
	if diag is None:
		return
	samples = max(1.0, scalar_item(diag['samples']))
	steps = int(scalar_item(diag['steps']))
	reward_mean = scalar_item(diag['reward_sum']) / samples
	reward_var = max(0.0, scalar_item(diag['reward_sq_sum']) / samples - reward_mean * reward_mean)
	reward_std = math.sqrt(reward_var)
	reward_min = scalar_item(diag['reward_min']) if torch.isfinite(diag['reward_min']) else 0.0
	reward_max = scalar_item(diag['reward_max']) if torch.isfinite(diag['reward_max']) else 0.0
	print(
		f"{prefix}: steps={steps}, samples={int(samples)}, reward_mean={reward_mean:.5f}, "
		f"reward_std={reward_std:.5f}, reward_min={reward_min:.4f}, reward_max={reward_max:.4f}, "
		f"neg%={100.0 * scalar_item(diag['negative_rewards']) / samples:.2f}, "
		f"done%={100.0 * scalar_item(diag['dones']) / samples:.2f}, "
		f"collision%={100.0 * scalar_item(diag['collisions']) / samples:.2f}, "
		f"offroad%={100.0 * scalar_item(diag['offroads']) / samples:.2f}, "
		f"final_goal%={100.0 * scalar_item(diag['final_goals']) / samples:.2f}, "
		f"waypoint%={100.0 * scalar_item(diag['intermediate_goals']) / samples:.2f}"
	)
	print(
		f"{prefix}动作top: "
		+ top_action_summary(
			diag['action_counts'],
			action_values=action_values,
			numerators={
				'r': diag['action_reward_sum'],
				'done': diag['action_done_counts'].to(torch.float64),
				'coll': diag['action_collision_counts'].to(torch.float64),
				'off': diag['action_offroad_counts'].to(torch.float64),
			},
		)
	)
	print(f"{prefix}奖励拆分: {reward_component_summary(diag.get('reward_component_sums', {}), samples)}")
	print(
		f"{prefix}动作奖励拆分: "
		+ top_action_component_summary(
			diag['action_counts'],
			diag.get('action_component_sums', {}),
			action_values=action_values,
		)
	)

def print_selected_batch_diagnostics(actions: torch.Tensor, raw_adv: torch.Tensor,
									 returns: torch.Tensor, old_logp: torch.Tensor,
									 num_actions: int, action_values: torch.Tensor = None,
									 prefix: str = "📊 PPO样本诊断"):
	if actions.numel() == 0:
		return
	actions = actions.detach().to(torch.long).clamp(0, num_actions - 1)
	raw_adv = raw_adv.detach().float()
	returns = returns.detach().float()
	old_logp = old_logp.detach().float()
	adv_s = tensor_stats(raw_adv)
	ret_s = tensor_stats(returns)
	counts = torch.bincount(actions, minlength=num_actions)
	adv_sum = torch.zeros((num_actions,), device=raw_adv.device, dtype=torch.float64)
	abs_adv_sum = torch.zeros((num_actions,), device=raw_adv.device, dtype=torch.float64)
	pos_sum = torch.zeros((num_actions,), device=raw_adv.device, dtype=torch.float64)
	logp_sum = torch.zeros((num_actions,), device=raw_adv.device, dtype=torch.float64)
	adv_sum.scatter_add_(0, actions, raw_adv.to(torch.float64))
	abs_adv_sum.scatter_add_(0, actions, raw_adv.abs().to(torch.float64))
	pos_sum.scatter_add_(0, actions, (raw_adv > 0).to(torch.float64))
	logp_sum.scatter_add_(0, actions, old_logp.to(torch.float64))
	print(
		f"{prefix}: adv_mean={adv_s['mean']:.5f}, adv_std={adv_s['std']:.5f}, "
		f"adv_min={adv_s['min']:.5f}, adv_max={adv_s['max']:.5f}, "
		f"adv_pos%={100.0 * scalar_item((raw_adv > 0).float().mean()):.2f}, "
		f"ret_mean={ret_s['mean']:.5f}, ret_std={ret_s['std']:.5f}, old_logp_mean={scalar_item(old_logp.mean()):.5f}"
	)
	print(
		f"{prefix}动作top: "
		+ top_action_summary(
			counts,
			action_values=action_values,
			numerators={
				'adv': adv_sum,
				'|adv|': abs_adv_sum,
				'pos': pos_sum,
				'oldlp': logp_sum,
			},
		)
	)

def step_schedulers_if_updated(policy_scheduler, value_scheduler, update_stats_accum: dict):
	if update_stats_accum.get('did_optimizer_step', False):
		policy_scheduler.step()
		value_scheduler.step()
		return True
	return False

def current_navigation(simulator, agents_state: torch.Tensor = None, route_state: dict = None,
					   w_lane_keep_mask: torch.Tensor = None, w_lane_ids: torch.Tensor = None,
					   out: torch.Tensor = None):
	"""返回显式 G(t) + W_lane 图路由距离的导航包。"""
	if hasattr(simulator, 'get_navigation_observation'):
		return simulator.get_navigation_observation(
			agents_state=agents_state,
			route_state=route_state,
			w_lane_keep_mask=w_lane_keep_mask,
			w_lane_ids=w_lane_ids,
			out=out,
		)
	return None


def feature_build_chunk_agents(config, max_agents: int) -> int:
	training_cfg = getattr(config, 'training', SimpleNamespace())
	chunk_agents = int(getattr(
		training_cfg,
		'feature_build_chunk_agents',
		getattr(training_cfg, 'network_forward_chunk_agents', 32768),
	))
	if chunk_agents <= 0:
		return 0
	return max(int(max_agents), chunk_agents)


def feature_build_env_chunk_size(config, max_agents: int) -> int:
	chunk_agents = feature_build_chunk_agents(config, max_agents)
	if chunk_agents <= 0:
		return 0
	return max(1, chunk_agents // max(1, int(max_agents)))


def slice_env_tensor(value, start: int, end: int, total_envs: int):
	if value is None:
		return None
	if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(total_envs):
		return value[start:end]
	return value


def slice_route_state_env(route_state: dict, start: int, end: int, total_envs: int) -> dict:
	if not route_state:
		return {}
	out = {}
	for key, value in route_state.items():
		out[key] = slice_env_tensor(value, start, end, total_envs)
	return out


def snapshot_condition_state(simulator) -> dict:
	"""Hold immutable rollout-level condition views; reset happens only after PPO."""
	reward_coef = getattr(getattr(simulator, 'reward_calculator', None), 'sampled_params', None)
	vehicle_style = getattr(simulator, 'driving_style_params', None)
	traffic_light_states = getattr(simulator, 'traffic_light_states', None)
	return {
		'reward_coef': reward_coef.detach() if torch.is_tensor(reward_coef) else None,
		'vehicle_style': vehicle_style.detach() if torch.is_tensor(vehicle_style) else None,
		'traffic_light_states': traffic_light_states.detach() if torch.is_tensor(traffic_light_states) else None,
	}


def slice_condition_state_env(condition_state: dict, start: int, end: int, total_envs: int) -> dict:
	if not condition_state:
		return {}
	return {
		key: slice_env_tensor(value, start, end, total_envs)
		for key, value in condition_state.items()
	}


def stop_lines_from_condition(simulator, agents_state: torch.Tensor, condition_state: dict,
							  out: torch.Tensor = None) -> torch.Tensor:
	if not hasattr(simulator, '_compute_stop_line_observation'):
		return None
	traffic_light_states = condition_state.get('traffic_light_states') if condition_state else None
	return simulator._compute_stop_line_observation(
		agents_state,
		traffic_light_states=traffic_light_states,
		out=out,
	)


class RolloutTensorBuffer:
	"""Preallocated rollout storage. PPO reads tensor views, avoiding list + stack duplication."""

	def __init__(self, capacity: int):
		self.capacity = max(1, int(capacity))
		self.length = 0
		self.states = None
		self.route_state = {}
		self.rewards = None
		self.dones = None
		self.values = None
		self.old_log_probs = None
		self.actions = None
		self.time_indices = None
		self._pre_step_pending = False

	def __len__(self):
		return self.length

	@property
	def device(self):
		if self.states is not None:
			return self.states.device
		for tensor in self.route_state.values():
			return tensor.device
		return torch.device('cpu')

	def _ensure_tensor(self, name: str, value: torch.Tensor) -> torch.Tensor:
		buffer = getattr(self, name)
		if buffer is None:
			dtype = self._storage_dtype(name, value)
			buffer = torch.empty(
				(self.capacity, *value.shape),
				device=value.device,
				dtype=dtype,
			)
			setattr(self, name, buffer)
		elif buffer.shape[1:] != value.shape:
			raise ValueError(f"{name} shape mismatch: {buffer.shape[1:]} != {value.shape}")
		elif buffer.device != value.device:
			raise ValueError(f"{name} device mismatch: {buffer.device} != {value.device}")
		return buffer

	def _ensure_tensor_shape(self, name: str, shape, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		buffer = getattr(self, name)
		shape = tuple(int(dim) for dim in shape)
		if buffer is None:
			buffer = torch.empty((self.capacity, *shape), device=device, dtype=dtype)
			setattr(self, name, buffer)
		elif buffer.shape[1:] != shape:
			raise ValueError(f"{name} shape mismatch: {buffer.shape[1:]} != {shape}")
		elif buffer.device != device:
			raise ValueError(f"{name} device mismatch: {buffer.device} != {device}")
		elif buffer.dtype != dtype:
			raise ValueError(f"{name} dtype mismatch: {buffer.dtype} != {dtype}")
		return buffer

	def _storage_dtype(self, name: str, value: torch.Tensor) -> torch.dtype:
		if name == 'actions' and value.dtype in (
			torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long,
		):
			if value.numel() == 0:
				return torch.uint8
			min_action = int(value.min().item())
			max_action = int(value.max().item())
			if 0 <= min_action and max_action <= 255:
				return torch.uint8
		return value.dtype

	def _route_storage_dtype(self, key: str, value: torch.Tensor) -> torch.dtype:
		if key == 'route_quad_ids':
			if value.numel() == 0:
				return torch.int16
			min_id = int(value.min().item())
			max_id = int(value.max().item())
			if -32768 <= min_id and max_id <= 32767:
				return torch.int16
			return torch.int32
		if key in ('target_count', 'current_idx'):
			return torch.int16
		return value.dtype

	def _ensure_route_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
		buffer = self.route_state.get(key)
		if buffer is None:
			dtype = self._route_storage_dtype(key, value)
			buffer = torch.empty(
				(self.capacity, *value.shape),
				device=value.device,
				dtype=dtype,
			)
			self.route_state[key] = buffer
		elif buffer.shape[1:] != value.shape:
			raise ValueError(f"route_state[{key}] shape mismatch: {buffer.shape[1:]} != {value.shape}")
		elif buffer.device != value.device:
			raise ValueError(f"route_state[{key}] device mismatch: {buffer.device} != {value.device}")
		return buffer

	def _write_time_index(self, state: torch.Tensor, time_index):
		B = int(state.shape[0])
		value = torch.as_tensor(time_index, device=state.device, dtype=torch.long)
		if value.dim() == 0:
			value = value.expand(B)
		else:
			value = value.reshape(B)
		storage = self._ensure_tensor_shape(
			'time_indices', (B,), state.device, torch.long
		)
		storage[self.length].copy_(value)

	def write_pre_step(self, state: torch.Tensor, route_state: dict, time_index: int = 0):
		if self.length >= self.capacity:
			raise RuntimeError(f"RolloutTensorBuffer full: length={self.length}, capacity={self.capacity}")
		if self._pre_step_pending:
			raise RuntimeError("write_pre_step called twice before write_post_step")
		self._ensure_tensor('states', state)[self.length].copy_(state.detach())
		self._write_time_index(state, time_index)
		for key, value in (route_state or {}).items():
			if torch.is_tensor(value):
				dst = self._ensure_route_tensor(key, value)[self.length]
				dst.copy_(value.detach().to(dtype=dst.dtype))
		self._pre_step_pending = True

	def write_pre_step_from_simulator(self, simulator, alive_mask: torch.Tensor, route_state: dict, time_index: int = 0):
		if self.length >= self.capacity:
			raise RuntimeError(f"RolloutTensorBuffer full: length={self.length}, capacity={self.capacity}")
		if self._pre_step_pending:
			raise RuntimeError("write_pre_step_from_simulator called twice before write_post_step")
		state = simulator.agents_state.detach()
		B, M = state.shape[:2]
		dst = self._ensure_tensor_shape('states', (B, M, 10), state.device, state.dtype)[self.length]
		dst.zero_()
		copy_state = min(7, state.shape[-1])
		if copy_state > 0:
			dst[..., :copy_state].copy_(state[..., :copy_state])
		dst[..., 6] = alive_mask.to(device=state.device, dtype=state.dtype)
		control = current_control_state(simulator).detach()
		dst[..., 7:10].copy_(control.to(device=state.device, dtype=state.dtype) * dst[..., 6:7])
		self._write_time_index(state, time_index)
		for key, value in (route_state or {}).items():
			if torch.is_tensor(value):
				route_dst = self._ensure_route_tensor(key, value)[self.length]
				route_dst.copy_(value.detach().to(dtype=route_dst.dtype))
		self._pre_step_pending = True

	def write_post_step(self, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor,
						old_log_prob: torch.Tensor, action: torch.Tensor):
		if not self._pre_step_pending:
			raise RuntimeError("write_post_step called before write_pre_step")
		done_bool = done.detach().bool()
		self._ensure_tensor('rewards', reward)[self.length].copy_(reward.detach())
		self._ensure_tensor('dones', done_bool)[self.length].copy_(done_bool)
		self._ensure_tensor('values', value)[self.length].copy_(value.detach())
		self._ensure_tensor('old_log_probs', old_log_prob)[self.length].copy_(old_log_prob.detach())
		action_dst = self._ensure_tensor('actions', action)[self.length]
		action_dst.copy_(action.detach().to(dtype=action_dst.dtype))
		self.length += 1
		self._pre_step_pending = False

	def clear(self):
		self.length = 0
		self._pre_step_pending = False

	def view(self, tensor: torch.Tensor):
		return tensor[:self.length] if tensor is not None else None

	def route_state_view(self) -> dict:
		return {key: value[:self.length] for key, value in self.route_state.items()}

	def validate(self, condition_state: dict):
		if self.length == 0:
			return
		if self._pre_step_pending:
			raise ValueError("RolloutTensorBuffer has a pending pre-step without post-step")
		required = {
			'states': self.states,
			'rewards': self.rewards,
			'dones': self.dones,
			'values': self.values,
			'old_log_probs': self.old_log_probs,
			'actions': self.actions,
			'time_indices': self.time_indices,
		}
		missing = [name for name, tensor in required.items() if tensor is None]
		if missing:
			raise ValueError(f"RolloutTensorBuffer missing tensors: {missing}")
		base_shape = self.states.shape[1:]
		base_device = self.states.device
		if len(base_shape) != 3 or base_shape[-1] < 7:
			raise ValueError(f"states expected (T,B,M,S>=7), got {(self.length, *base_shape)}")
		B, M, _ = base_shape
		validate_condition_state(condition_state, B, M, base_device)
		for key in ('route_quad_ids', 'target_count', 'current_idx'):
			if key not in self.route_state:
				raise ValueError(f"route_state missing key: {key}")
			tensor = self.route_state[key]
			if tensor.device != base_device:
				raise ValueError(f"route_state[{key}] device mismatch: {tensor.device} != {base_device}")
			if tensor.shape[1] != B or tensor.shape[2] != M:
				raise ValueError(f"route_state[{key}] leading shape mismatch: {tensor.shape[1:3]} != {(B, M)}")
		for name, tensor in required.items():
			if name == 'time_indices':
				if tensor.shape[1:] != (B,):
					raise ValueError(
						f"time_indices shape mismatch: {tensor.shape[1:]} != {(B,)}"
					)
				continue
			if tensor.device != base_device:
				raise ValueError(f"{name} device mismatch: {tensor.device} != {base_device}")
			if tensor.shape[1:3] != (B, M):
				raise ValueError(f"{name} shape mismatch: {tensor.shape[1:3]} != {(B, M)}")


def gather_route_state_selected(route_state_tensor: dict, t_idx: torch.Tensor,
								b_idx: torch.Tensor, agent_idx: torch.Tensor) -> dict:
	if not route_state_tensor:
		return {}
	out = {}
	for key, value in route_state_tensor.items():
		selected = value[t_idx, b_idx, agent_idx]
		out[key] = selected.unsqueeze(1)
	return out


def gather_condition_state_selected(condition_state: dict, env_idx: torch.Tensor,
									agent_idx: torch.Tensor) -> dict:
	if not condition_state:
		return {}
	out = {}
	for key, value in condition_state.items():
		if not torch.is_tensor(value) or value.dim() == 0:
			out[key] = value
		elif key in ('reward_coef', 'vehicle_style') and value.dim() >= 3:
			out[key] = value[env_idx, agent_idx].unsqueeze(1)
		elif value.shape[0] == condition_state.get('reward_coef', value).shape[0]:
			out[key] = value[env_idx]
		else:
			out[key] = value
	return out

def current_control_state(simulator):
	"""返回当前动力学控制状态 [phi, a_long, a_lat]，形状 (B, M, 3)。"""
	if hasattr(simulator, '_current_control_state'):
		return simulator._current_control_state()
	states = simulator.agents_state
	B, M = states.shape[:2]
	control = torch.zeros(B, M, 3, device=states.device, dtype=states.dtype)
	dynamics = getattr(simulator, 'dynamics_model', None)
	if dynamics is None:
		return control
	for idx, name in enumerate(('current_steering_angle', 'current_along', 'current_alat')):
		value = getattr(dynamics, name, None)
		if value is not None and value.numel() == B * M:
			control[:, :, idx] = value.to(device=states.device, dtype=states.dtype).view(B, M)
	return control

def observation_state_from_buffer(world_state: torch.Tensor) -> torch.Tensor:
	"""PPO buffer 可额外拼接控制状态；重建 observation 时只使用世界状态前7维。"""
	return world_state[..., :7]

def control_from_buffer_state(world_state: torch.Tensor) -> torch.Tensor:
	if world_state.shape[-1] >= 10:
		return world_state[..., 7:10]
	return torch.zeros(*world_state.shape[:2], 3, device=world_state.device, dtype=world_state.dtype)

def get_map_dropout_probs(config):
	training_cfg = getattr(config, 'training', SimpleNamespace())
	lane_p = float(getattr(training_cfg, 'w_lane_dropout_prob', 0.0))
	boundary_p = float(getattr(training_cfg, 'w_boundary_dropout_prob', 0.0))
	return min(max(lane_p, 0.0), 1.0), min(max(boundary_p, 0.0), 1.0)


def _index_grid(value, B: int, M: int, device: torch.device, default_kind: str,
                workspace: FeatureBuildWorkspace = None):
	if value is None:
		if default_kind == 'env':
			arange = workspace.arange(B, device) if workspace is not None else torch.arange(B, device=device, dtype=torch.long)
			return arange.view(B, 1, 1)
		if default_kind == 'agent':
			arange = workspace.arange(M, device) if workspace is not None else torch.arange(M, device=device, dtype=torch.long)
			return arange.view(1, M, 1)
		return torch.zeros((1, 1, 1), device=device, dtype=torch.long)
	tensor = torch.as_tensor(value, device=device, dtype=torch.long)
	if tensor.dim() == 0:
		return tensor.view(1, 1, 1)
	if tensor.numel() == B * M:
		return tensor.view(B, M, 1)
	if tensor.numel() == B:
		return tensor.view(B, 1, 1)
	if tensor.numel() == M:
		return tensor.view(1, M, 1)
	return tensor.reshape(B, M, 1)


def deterministic_element_keep_mask(B: int, M: int, num_elements: int, drop_prob: float,
									device: torch.device, time_idx=None, env_idx=None,
									agent_idx=None, salt: int = 0,
									workspace: FeatureBuildWorkspace = None):
	if num_elements <= 0:
		return None
	if drop_prob <= 0.0:
		return None
	if drop_prob >= 1.0:
		return torch.zeros((B, M, num_elements), device=device, dtype=torch.bool)
	keep_threshold = int(round((1.0 - drop_prob) * 10000))
	time_grid = _index_grid(time_idx, B, M, device, 'zero', workspace=workspace)
	env_grid = _index_grid(env_idx, B, M, device, 'env', workspace=workspace)
	agent_grid = _index_grid(agent_idx, B, M, device, 'agent', workspace=workspace)
	elem_arange = workspace.arange(num_elements, device) if workspace is not None else torch.arange(num_elements, device=device, dtype=torch.long)
	elem_grid = elem_arange.view(1, 1, num_elements)
	seed = (
		time_grid * 1000003
		+ env_grid * 19349663
		+ agent_grid * 83492791
		+ elem_grid * 47899981
		+ int(salt)
	)
	seed = torch.remainder(seed, 2147483647)
	hashed = torch.remainder(seed * 48271 + 12345, 2147483647)
	return torch.remainder(hashed, 10000) < keep_threshold


def make_map_dropout_masks(config, B: int, M: int, lane_count: int, boundary_count: int,
						   device: torch.device, time_idx=None, env_idx=None, agent_idx=None,
						   workspace: FeatureBuildWorkspace = None):
	lane_drop, boundary_drop = get_map_dropout_probs(config)
	return {
		'lane_keep': deterministic_element_keep_mask(
			B, M, lane_count, lane_drop, device,
			time_idx=time_idx, env_idx=env_idx, agent_idx=agent_idx, salt=17,
			workspace=workspace,
		),
		'boundary_keep': deterministic_element_keep_mask(
			B, M, boundary_count, boundary_drop, device,
			time_idx=time_idx, env_idx=env_idx, agent_idx=agent_idx, salt=29,
			workspace=workspace,
		),
	}


def apply_map_dropout_components(w_lanes_local: torch.Tensor, w_boundaries_local: torch.Tensor,
								 map_dropout: dict = None, inplace: bool = False):
	if not map_dropout:
		return w_lanes_local, w_boundaries_local
	lane_keep = map_dropout.get('lane_keep')
	if lane_keep is not None and w_lanes_local is not None:
		if inplace:
			w_lanes_local.masked_fill_(~lane_keep.unsqueeze(-1), 0.0)
		else:
			w_lanes_local = torch.where(lane_keep.unsqueeze(-1), w_lanes_local, torch.zeros_like(w_lanes_local))
	boundary_keep = map_dropout.get('boundary_keep')
	if boundary_keep is not None and w_boundaries_local is not None:
		if inplace:
			w_boundaries_local.masked_fill_(~boundary_keep.unsqueeze(-1), 0.0)
		else:
			w_boundaries_local = torch.where(boundary_keep.unsqueeze(-1), w_boundaries_local, torch.zeros_like(w_boundaries_local))
	return w_lanes_local, w_boundaries_local


def _policy_observation_state_chunk(simulator, start: int, end: int, alive_mask: torch.Tensor = None):
	obs_state = simulator.agents_state[start:end]
	if alive_mask is not None:
		obs_state = obs_state.clone()
		obs_state[..., 6] = alive_mask[start:end].to(device=obs_state.device, dtype=obs_state.dtype)
	elif getattr(simulator, 'last_done', None) is not None:
		last_done = simulator.last_done[start:end].to(obs_state.device)
		obs_state = obs_state.clone()
		obs_state[..., 6] = torch.where(last_done, 0.0, obs_state[..., 6])
	return obs_state


def build_features_from_components(agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
								   simulator, config, route_state, condition_state,
								   control_state=None, map_dropout=None, world_agents_state=None,
								   map_metadata: dict = None, workspace: FeatureBuildWorkspace = None,
								   out: torch.Tensor = None):
	world_agents_state = agents_state if world_agents_state is None else world_agents_state
	lane_keep = map_dropout.get('lane_keep') if map_dropout else None
	w_lane_ids = map_metadata.get('w_lane_ids') if map_metadata else None
	workspace = workspace or _default_workspace(config)
	goal_slots = int(getattr(simulator, 'max_route_targets', 0))
	lane_slots = int(getattr(simulator.observation_generator, 'num_w_lanes', 0))
	nav_out = workspace.scratch(
		'navigation',
		(world_agents_state.shape[0], world_agents_state.shape[1], goal_slots + lane_slots, 3),
		world_agents_state.device,
		world_agents_state.dtype,
	)
	navigation = current_navigation(
		simulator,
		agents_state=world_agents_state,
		route_state=route_state,
		w_lane_keep_mask=lane_keep,
		w_lane_ids=w_lane_ids,
		out=nav_out,
	)
	stop_dim = int(getattr(simulator, 'stop_line_feature_dim', 0))
	stop_out = workspace.scratch(
		'stop_lines',
		(world_agents_state.shape[0], world_agents_state.shape[1], stop_dim),
		world_agents_state.device,
		world_agents_state.dtype,
	)
	stop_lines = stop_lines_from_condition(simulator, world_agents_state, condition_state, out=stop_out)
	workspace.mark_feature_chunk()
	return build_network_features(
		agents_state,
		neighbors_local,
		w_lanes_local,
		w_boundaries_local,
		navigation,
		stop_lines,
		condition_state.get('reward_coef'),
		config,
		vehicle_style=condition_state.get('vehicle_style'),
		control_state=control_state,
		map_dropout=map_dropout,
		workspace=workspace,
		out=out,
	)


def build_features_from_simulator_env_slice(
	simulator,
	config,
	start: int,
	end: int,
	alive_mask: torch.Tensor = None,
	condition_state: dict = None,
	dropout_step=0,
	workspace: FeatureBuildWorkspace = None,
	control_state_all: torch.Tensor = None,
	route_state_all: dict = None,
	out: torch.Tensor = None,
) -> torch.Tensor:
	B, M = simulator.agents_state.shape[:2]
	workspace = workspace or _default_workspace(config)
	condition_state = condition_state if condition_state is not None else snapshot_condition_state(simulator)
	route_state_all = (
		route_state_all
		if route_state_all is not None
		else (simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {})
	)
	control_state_all = (
		control_state_all if control_state_all is not None else current_control_state(simulator)
	)
	lane_count = int(getattr(simulator.observation_generator, 'num_w_lanes', 0))
	boundary_count = int(getattr(simulator.observation_generator, 'num_w_boundaries', 0))
	obs_state = _policy_observation_state_chunk(simulator, start, end, alive_mask=alive_mask)
	control_chunk = slice_env_tensor(control_state_all, start, end, B)
	condition_chunk = slice_condition_state_env(condition_state, start, end, B)
	route_chunk = slice_route_state_env(route_state_all, start, end, B)
	local_state, neighbors_local, w_lanes_local, w_boundaries_local, map_metadata = simulator.observation_generator.generate_components(
		obs_state,
		control_state=control_chunk,
		driving_style_params=condition_chunk.get('vehicle_style'),
		return_map_ids=True,
	)
	env_idx = workspace.arange(B, obs_state.device)[start:end]
	map_dropout = make_map_dropout_masks(
		config,
		end - start,
		M,
		lane_count,
		boundary_count,
		obs_state.device,
		time_idx=slice_env_tensor(dropout_step, start, end, B),
		env_idx=env_idx,
		workspace=workspace,
	)
	return build_features_from_components(
		local_state,
		neighbors_local,
		w_lanes_local,
		w_boundaries_local,
		simulator,
		config,
		route_chunk,
		condition_chunk,
		control_state=control_chunk,
		map_dropout=map_dropout,
		world_agents_state=obs_state,
		map_metadata=map_metadata,
		workspace=workspace,
		out=out,
	)


def build_features_from_simulator_state(simulator, config, alive_mask: torch.Tensor = None,
										condition_state: dict = None, dropout_step=0,
										workspace: FeatureBuildWorkspace = None) -> torch.Tensor:
	B, M = simulator.agents_state.shape[:2]
	workspace = workspace or _default_workspace(config)
	env_chunk = feature_build_env_chunk_size(config, M) or B
	features = torch.empty(
		B,
		M,
		workspace.total_input_dim,
		device=simulator.agents_state.device,
		dtype=simulator.agents_state.dtype,
	)
	condition_state = condition_state if condition_state is not None else snapshot_condition_state(simulator)
	route_state = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
	control_state_all = current_control_state(simulator)
	for start in range(0, B, env_chunk):
		end = min(start + env_chunk, B)
		build_features_from_simulator_env_slice(
			simulator,
			config,
			start,
			end,
			alive_mask=alive_mask,
			condition_state=condition_state,
			dropout_step=dropout_step,
			workspace=workspace,
			control_state_all=control_state_all,
			route_state_all=route_state,
			out=features[start:end],
		)
	return features


def build_features_for_selected_agents(world_states: torch.Tensor, simulator, config,
									   route_state: dict, condition_state: dict,
									   agent_indices: torch.Tensor, time_indices: torch.Tensor = None,
									   env_indices: torch.Tensor = None,
									   workspace: FeatureBuildWorkspace = None,
									   out: torch.Tensor = None) -> torch.Tensor:
	B = world_states.shape[0]
	workspace = workspace or _default_workspace(config)
	obs_state = observation_state_from_buffer(world_states)
	control_state = control_from_buffer_state(world_states)
	agent_indices = agent_indices.to(device=world_states.device, dtype=torch.long).view(B)
	batch_idx = workspace.arange(B, world_states.device)
	ego_obs_state = obs_state[batch_idx, agent_indices].unsqueeze(1)
	local_state, neighbors_local, w_lanes_local, w_boundaries_local, map_metadata = simulator.observation_generator.generate_selected_components(
		obs_state,
		agent_indices,
		control_state=control_state,
		driving_style_params=condition_state.get('vehicle_style'),
		return_map_ids=True,
	)
	lane_count = int(w_lanes_local.shape[2]) if w_lanes_local is not None else 0
	boundary_count = int(w_boundaries_local.shape[2]) if w_boundaries_local is not None else 0
	map_dropout = make_map_dropout_masks(
		config,
		B,
		1,
		lane_count,
		boundary_count,
		world_states.device,
		time_idx=time_indices,
		env_idx=env_indices,
		agent_idx=agent_indices,
		workspace=workspace,
	)
	return build_features_from_components(
		local_state,
		neighbors_local,
		w_lanes_local,
		w_boundaries_local,
		simulator,
		config,
		route_state,
		condition_state,
		control_state=control_state.gather(1, agent_indices.view(B, 1, 1).expand(-1, -1, control_state.shape[-1])),
		map_dropout=map_dropout,
		world_agents_state=ego_obs_state,
		map_metadata=map_metadata,
		workspace=workspace,
		out=out,
	)

def sync_bool_across_ranks(value: bool, device: torch.device, op=dist.ReduceOp.MIN) -> bool:
	if not dist.is_available() or not dist.is_initialized():
		return value
	t = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
	dist.all_reduce(t, op=op)
	return bool(t.item())


def sync_min_int_across_ranks(value: int, device: torch.device) -> int:
	if not dist.is_available() or not dist.is_initialized():
		return int(value)
	tensor = torch.tensor([int(value)], dtype=torch.int32, device=device)
	dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
	return int(tensor.item())


def sum_int_across_ranks(value: int, device: torch.device) -> int:
	"""Sum a run-level integer once; this is not used in rollout hot paths."""
	if not dist.is_available() or not dist.is_initialized():
		return int(value)
	tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
	dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
	return int(tensor.item())


def is_cuda_oom_error(exc: RuntimeError, device: torch.device) -> bool:
	if device.type != 'cuda':
		return False
	oom_type = getattr(torch.cuda, 'OutOfMemoryError', ())
	return (oom_type and isinstance(exc, oom_type)) or 'out of memory' in str(exc).lower()


def clear_optimizer_gradients(policy_optimizer, value_optimizer, device: torch.device):
	policy_optimizer.zero_grad(set_to_none=True)
	value_optimizer.zero_grad(set_to_none=True)
	if device.type == 'cuda':
		torch.cuda.empty_cache()


def average_gradients_across_ranks(
	model,
	local_samples: int,
	device: torch.device,
	bucket_cap_mb: float = 25.0,
) -> int:
	"""Sample-weight gradients using one collective per contiguous-size bucket."""
	if not dist.is_available() or not dist.is_initialized():
		return 0
	total_samples = torch.tensor([float(local_samples)], dtype=torch.float32, device=device)
	dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
	local_weight = total_samples.new_tensor(float(local_samples)) / total_samples
	bucket_cap_bytes = max(1, int(float(bucket_cap_mb) * 1024 ** 2))
	buckets = []
	current = []
	current_bytes = 0
	current_key = None
	for parameter in unwrap_model(model).parameters():
		if parameter.grad is None:
			parameter.grad = torch.zeros_like(parameter)
		grad = parameter.grad
		key = (grad.device, grad.dtype)
		grad_bytes = grad.numel() * grad.element_size()
		if current and (key != current_key or current_bytes + grad_bytes > bucket_cap_bytes):
			buckets.append(current)
			current = []
			current_bytes = 0
		current.append(grad)
		current_bytes += grad_bytes
		current_key = key
	if current:
		buckets.append(current)

	for bucket in buckets:
		flat = torch.cat([grad.reshape(-1) for grad in bucket])
		flat.mul_(local_weight)
		dist.all_reduce(flat, op=dist.ReduceOp.SUM)
		offset = 0
		for grad in bucket:
			numel = grad.numel()
			grad.copy_(flat[offset:offset + numel].view_as(grad))
			offset += numel
	return len(buckets)

def validate_condition_state(condition_state: dict, B: int, M: int, device: torch.device):
	if not isinstance(condition_state, dict):
		raise ValueError(f"condition_state expected dict, got {type(condition_state)}")
	for key in ('reward_coef', 'vehicle_style'):
		tensor = condition_state.get(key)
		if tensor is None:
			raise ValueError(f"condition_state missing {key}")
		if tensor.device != device:
			raise ValueError(f"condition_state[{key}] device mismatch: {tensor.device} != {device}")
		if tensor.shape[0] != B or tensor.shape[1] != M:
			raise ValueError(f"condition_state[{key}] leading shape mismatch: {tensor.shape[:2]} != {(B, M)}")
	traffic_light_states = condition_state.get('traffic_light_states')
	if traffic_light_states is not None:
		if traffic_light_states.device != device:
			raise ValueError(f"condition_state[traffic_light_states] device mismatch: {traffic_light_states.device} != {device}")
		if traffic_light_states.dim() > 0 and traffic_light_states.shape[0] != B:
			raise ValueError(f"condition_state[traffic_light_states] leading shape mismatch: {traffic_light_states.shape[:1]} != {(B,)}")


def validate_rollout_buffer(rollout_buffer: RolloutTensorBuffer, condition_state: dict):
	if not isinstance(rollout_buffer, RolloutTensorBuffer):
		raise ValueError(f"rollout_buffer expected RolloutTensorBuffer, got {type(rollout_buffer)}")
	rollout_buffer.validate(condition_state)

def rollout_alive_mask(simulator, cumulative_done_all=None) -> torch.Tensor:
	"""返回本 rollout 当前仍应参与采样/训练的 agent mask。"""
	states = simulator.agents_state
	active_mask = states[..., 6] > 0.5
	if cumulative_done_all is None:
		return active_mask
	return active_mask & (~cumulative_done_all.to(active_mask.device))

def rollout_forward_alive_agents(model, simulator, config, alive_mask: torch.Tensor,
								 condition_state: dict, dropout_step: int,
								 precision: str, forward_chunk_agents: int,
								 sample_actions: bool = True,
								 feature_workspace: FeatureBuildWorkspace = None,
								 batch_sizer: AdaptiveBatchSizer = None):
	"""
	Online rollout forward for active agents only.

	The returned action/logp/value tensors keep the full (B,M) shape required by
	the simulator and rollout buffer, but feature construction and network
	forward only run on alive selected agents.
	"""
	states = simulator.agents_state
	B, M = states.shape[:2]
	device = states.device
	actions = torch.zeros((B, M), dtype=torch.long, device=device)
	old_log_probs = torch.zeros((B, M), dtype=states.dtype, device=device)
	value_pred = torch.zeros((B, M), dtype=states.dtype, device=device)
	profile = {
		'feature_ms': 0.0,
		'policy_ms': 0.0,
		'num_selected': B * M,
		'feature_chunks': 0,
		'path': 'dense_stream',
		'oom_retries': 0,
	}
	feature_workspace = feature_workspace or _default_workspace(config)
	feature_workspace.reset_counters()
	alive_mask = alive_mask.to(device=device, dtype=torch.bool)
	if batch_sizer is None:
		maximum = max(M, feature_build_chunk_agents(config, M))
		batch_sizer = AdaptiveBatchSizer(maximum=maximum, minimum=M)
	chunk_agents = batch_sizer.choose(B * M)
	env_chunk_size = max(1, chunk_agents // M)
	route_state_all = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
	control_state_all = current_control_state(simulator)
	profile_on = profile_enabled(config)
	start = 0
	while start < B:
		end = min(start + env_chunk_size, B)
		features_chunk = None
		action_logits = None
		values_chunk = None
		try:
			if profile_on:
				feature_start = profile_timer_start(device, config)
			features_chunk = build_features_from_simulator_env_slice(
				simulator,
				config,
				start,
				end,
				alive_mask=alive_mask,
				condition_state=condition_state,
				dropout_step=dropout_step,
				workspace=feature_workspace,
				control_state_all=control_state_all,
				route_state_all=route_state_all,
			)
			if profile_on:
				profile['feature_ms'] += profile_elapsed_ms(feature_start, device, config)
			if profile_on:
				policy_start = profile_timer_start(device, config)
			with torch.inference_mode(), make_autocast_context(device, precision):
				if sample_actions:
					action_logits, values_chunk = forward_model(
						model,
						features_chunk,
						mode="both",
						chunk_agents=forward_chunk_agents,
					)
					distribution = torch.distributions.Categorical(logits=action_logits)
					action_chunk = distribution.sample()
					log_prob_chunk = distribution.log_prob(action_chunk).to(old_log_probs.dtype)
					chunk_alive = alive_mask[start:end]
					actions[start:end] = torch.where(
						chunk_alive, action_chunk, actions[start:end]
					)
					old_log_probs[start:end] = torch.where(
						chunk_alive, log_prob_chunk, old_log_probs[start:end]
					)
				else:
					values_chunk = forward_model(
						model,
						features_chunk,
						mode="value",
						chunk_agents=forward_chunk_agents,
					)
			if values_chunk.dim() == 3 and values_chunk.shape[-1] == 1:
				values_chunk = values_chunk.squeeze(-1)
			value_pred[start:end] = torch.where(
				alive_mask[start:end],
				values_chunk.to(value_pred.dtype),
				value_pred[start:end],
			)
			if profile_on:
				profile['policy_ms'] += profile_elapsed_ms(policy_start, device, config)
			start = end
		except RuntimeError as exc:
			if not is_cuda_oom_error(exc, device):
				raise
			profile['oom_retries'] += 1
			features_chunk = None
			action_logits = None
			values_chunk = None
			feature_workspace.clear_scratch()
			torch.cuda.empty_cache()
			attempted_agents = max(M, (end - start) * M)
			next_agents = batch_sizer.backoff(attempted_agents)
			if next_agents is None:
				raise RuntimeError(
					f"CUDA OOM in rollout at minimum chunk of one world ({M} agents)"
				) from exc
			next_agents = max(M, (next_agents // M) * M)
			batch_sizer.current = next_agents
			env_chunk_size = max(1, next_agents // M)
	profile['feature_chunks'] = feature_workspace.feature_chunks
	batch_sizer.record_success(env_chunk_size * M, B * M)
	return actions, old_log_probs, value_pred, profile


def bootstrap_values_for_alive_agents(model, simulator, config, alive_mask: torch.Tensor,
									  condition_state: dict, dropout_step: int,
									  precision: str, forward_chunk_agents: int,
									  feature_workspace: FeatureBuildWorkspace = None,
									  batch_sizer: AdaptiveBatchSizer = None):
	_, _, value_pred, _ = rollout_forward_alive_agents(
		model,
		simulator,
		config,
		alive_mask,
		condition_state,
		dropout_step,
		precision,
		forward_chunk_agents,
		sample_actions=False,
		feature_workspace=feature_workspace,
		batch_sizer=batch_sizer,
	)
	return value_pred


def current_rollout_bootstrap_value(model, simulator, config, cumulative_done_all,
									condition_state: dict, dropout_step: int,
									precision: str, forward_chunk_agents: int,
									feature_workspace: FeatureBuildWorkspace = None,
									batch_sizer: AdaptiveBatchSizer = None):
	alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
	return bootstrap_values_for_alive_agents(
		model,
		simulator,
		config,
		alive_mask,
		condition_state,
		dropout_step,
		precision,
		forward_chunk_agents,
		feature_workspace=feature_workspace,
		batch_sizer=batch_sizer,
	)


def resolve_ppo_samples_per_rank(training_cfg) -> int:
	"""Resolve the local sample target from the paper's global/per-GPU batch settings."""
	world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
	per_rank = getattr(training_cfg, 'batch_size_per_gpu', None)
	global_batch = getattr(training_cfg, 'batch_size', None)
	if per_rank is not None:
		return int(per_rank)
	if global_batch is not None:
		return max(1, math.ceil(int(global_batch) / max(1, world_size)))
	return 2000


def create_ppo_batch_sizer(training_cfg) -> AdaptiveBatchSizer:
	"""Create the persistent OOM-feedback controller for PPO microbatches."""
	target = resolve_ppo_samples_per_rank(training_cfg)
	configured_cap = getattr(training_cfg, 'ppo_microbatch_max', None)
	maximum = target if configured_cap is None else min(target, int(configured_cap))
	configured_initial = getattr(training_cfg, 'ppo_microbatch_initial', None)
	initial = maximum if configured_initial is None else min(maximum, int(configured_initial))
	minimum = int(getattr(training_cfg, 'ppo_min_microbatch_size', 256))
	growth_interval = int(getattr(training_cfg, 'ppo_microbatch_growth_interval', 20))
	return AdaptiveBatchSizer(
		maximum=maximum,
		minimum=minimum,
		growth_interval=growth_interval,
		initial=initial,
	)


def create_feature_batch_sizer(training_cfg) -> AdaptiveBatchSizer:
	"""Create the persistent controller used while materializing PPO features."""
	maximum = resolve_ppo_samples_per_rank(training_cfg)
	configured_initial = getattr(training_cfg, 'ppo_feature_build_initial', None)
	initial = maximum if configured_initial is None else min(maximum, int(configured_initial))
	minimum = int(getattr(training_cfg, 'ppo_feature_build_min', 256))
	growth_interval = int(getattr(training_cfg, 'ppo_microbatch_growth_interval', 20))
	return AdaptiveBatchSizer(
		maximum=maximum,
		minimum=minimum,
		growth_interval=growth_interval,
		initial=initial,
	)


def create_rollout_batch_sizer(config) -> AdaptiveBatchSizer:
	training_cfg = config.training
	simulator_cfg = config.simulator
	max_agents = int(simulator_cfg.max_agents_num)
	maximum = int(simulator_cfg.num_envs) * max_agents
	configured_initial = getattr(training_cfg, 'rollout_chunk_initial_agents', None)
	initial = (
		min(maximum, int(configured_initial))
		if configured_initial is not None
		else min(maximum, feature_build_chunk_agents(config, max_agents))
	)
	initial = max(max_agents, (initial // max_agents) * max_agents)
	return AdaptiveBatchSizer(
		maximum=maximum,
		minimum=max_agents,
		growth_interval=int(getattr(training_cfg, 'rollout_chunk_growth_interval', 100)),
		initial=initial,
	)


class SelectedFeatureCache:
	"""One immutable feature tensor reused by every PPO epoch."""

	def __init__(self, storage: torch.Tensor, compute_device: torch.device):
		self.storage = storage
		self.compute_device = compute_device

	@property
	def location(self) -> str:
		return self.storage.device.type

	def get(self, start: int, end: int) -> torch.Tensor:
		chunk = self.storage[start:end]
		if chunk.device == self.compute_device:
			return chunk
		return chunk.to(self.compute_device, non_blocking=chunk.is_pinned())

	def offload_to_cpu(self) -> bool:
		if self.storage.device.type != 'cuda':
			return False
		try:
			cpu_storage = torch.empty_like(
				self.storage, device='cpu', pin_memory=True
			)
		except RuntimeError:
			cpu_storage = torch.empty_like(self.storage, device='cpu')
		cpu_storage.copy_(self.storage)
		self.storage = cpu_storage
		torch.cuda.empty_cache()
		return True


def ppo_feature_cache_dtype(precision: str) -> torch.dtype:
	precision = str(precision).strip().lower()
	if precision in {'16', '16-bit', 'fp16', 'float16'}:
		return torch.float16
	if precision in {'bf16', 'bfloat16'}:
		return torch.bfloat16
	return torch.float32


def allocate_selected_feature_cache(
	count: int,
	feature_dim: int,
	device: torch.device,
	dtype: torch.dtype,
	training_cfg,
) -> SelectedFeatureCache:
	location = str(getattr(training_cfg, 'ppo_feature_cache_location', 'auto')).lower()
	if location not in {'auto', 'cuda', 'cpu'}:
		raise ValueError(
			"training.ppo_feature_cache_location must be one of: auto, cuda, cpu"
		)
	shape = (int(count), 1, int(feature_dim))
	use_cuda = device.type == 'cuda' and location != 'cpu'
	if use_cuda and location == 'auto':
		free_bytes, _ = torch.cuda.mem_get_info(device)
		memory_cfg = getattr(training_cfg, 'memory_adaptation', SimpleNamespace())
		reserve_mb = float(getattr(memory_cfg, 'reserve_mb', 1024.0))
		needed_bytes = count * feature_dim * torch.empty((), dtype=dtype).element_size()
		use_cuda = needed_bytes <= max(0, free_bytes - int(reserve_mb * 1024 ** 2))
	if use_cuda:
		try:
			return SelectedFeatureCache(torch.empty(shape, device=device, dtype=dtype), device)
		except RuntimeError as exc:
			if location == 'cuda' or not is_cuda_oom_error(exc, device):
				raise
			torch.cuda.empty_cache()
	try:
		storage = torch.empty(shape, device='cpu', dtype=dtype, pin_memory=device.type == 'cuda')
	except RuntimeError:
		storage = torch.empty(shape, device='cpu', dtype=dtype)
	return SelectedFeatureCache(storage, device)


def build_selected_feature_cache(
	states_tensor: torch.Tensor,
	route_state_tensor: dict,
	condition_state: dict,
	selected_t: torch.Tensor,
	selected_b: torch.Tensor,
	selected_m: torch.Tensor,
	time_indices_tensor: torch.Tensor,
	simulator,
	config,
	precision: str,
	workspace: FeatureBuildWorkspace,
	batch_sizer: AdaptiveBatchSizer,
) -> tuple[SelectedFeatureCache, int, int]:
	"""Build selected observations once, with persistent CUDA-OOM backoff."""
	count = int(selected_t.shape[0])
	training_cfg = config.training
	cache = allocate_selected_feature_cache(
		count,
		workspace.total_input_dim,
		states_tensor.device,
		ppo_feature_cache_dtype(precision),
		training_cfg,
	)
	chunk_size = batch_sizer.choose(count)
	oom_retries = 0
	offloads = 0
	start = 0
	while start < count:
		end = min(start + chunk_size, count)
		try:
			chunk_t = selected_t[start:end]
			chunk_b = selected_b[start:end]
			chunk_m = selected_m[start:end]
			world_states = states_tensor[chunk_t, chunk_b]
			route_state = gather_route_state_selected(
				route_state_tensor, chunk_t, chunk_b, chunk_m
			)
			conditions = gather_condition_state_selected(condition_state, chunk_b, chunk_m)
			features = build_features_for_selected_agents(
				world_states,
				simulator,
				config,
				route_state,
				conditions,
				chunk_m,
				time_indices=time_indices_tensor[chunk_t, chunk_b],
				env_indices=chunk_b,
				workspace=workspace,
			)
			cache.storage[start:end].copy_(
				features.to(dtype=cache.storage.dtype),
				non_blocking=cache.storage.is_pinned(),
			)
			start = end
		except RuntimeError as exc:
			if not is_cuda_oom_error(exc, states_tensor.device):
				raise
			oom_retries += 1
			features = None
			world_states = None
			route_state = None
			conditions = None
			workspace.clear_scratch()
			torch.cuda.empty_cache()
			if (
				str(getattr(training_cfg, 'ppo_feature_cache_location', 'auto')).lower() == 'auto'
				and cache.offload_to_cpu()
			):
				offloads += 1
				continue
			next_size = batch_sizer.backoff(chunk_size)
			if next_size is None:
				raise RuntimeError(
					f"CUDA OOM while building PPO feature cache at minimum chunk {chunk_size}"
				) from exc
			chunk_size = next_size
	batch_sizer.record_success(chunk_size, count)
	return cache, oom_retries, offloads


# ============================== PPO更新函数 ==============================
def perform_ppo_update(model, policy_optimizer, value_optimizer,
					   rollout_buffer, condition_state,
					   simulator, config, update_step, rank=None,
					   a_max_ewma=None, amp_scaler=None, bootstrap_value=None,
					   feature_workspace: FeatureBuildWorkspace = None,
					   batch_sizer: AdaptiveBatchSizer = None,
					   feature_batch_sizer: AdaptiveBatchSizer = None):
	"""执行 PPO 更新。buffer 保存世界状态/route state，条件特征在 minibatch 内重建。"""
	is_rank0 = (rank is None or rank == 0)
	update_start_time = time.time()
	if len(rollout_buffer) == 0:
		device = simulator.device
		if is_rank0:
			print("⚠️ Buffer为空，无法进行PPO更新")
		return a_max_ewma, make_update_stats("empty_buffer", device)
	if is_rank0:
		print(f"🎯 开始经验采样训练，Buffer长度: {len(rollout_buffer)}")

	training_cfg = getattr(config, 'training')
	gamma = getattr(training_cfg, 'gamma', 0.999)
	gae_lambda = getattr(training_cfg, 'gae_lambda', 0.95)
	ppo_epochs = int(getattr(training_cfg, 'ppo_epochs', 3))
	clip_ratio = float(getattr(training_cfg, 'clip_ratio', 0.2))
	entropy_coef = float(getattr(training_cfg, 'entropy_coef', 0.01))
	value_loss_coef = float(getattr(training_cfg, 'value_loss_coef', 0.5))
	configured_value_clip = getattr(training_cfg, 'value_clip_ratio', None)
	value_clip_ratio = None if configured_value_clip is None else float(configured_value_clip)
	max_grad_norm = float(getattr(training_cfg, 'max_grad_norm', 1.0))
	batch_size_per_gpu = resolve_ppo_samples_per_rank(training_cfg)
	advantage_filter_threshold = float(getattr(training_cfg, 'advantage_filter_threshold', 0.01))
	beta = float(getattr(training_cfg, 'advantage_filter_beta', 0.25))
	advantage_filter_max_drop_fraction = float(getattr(training_cfg, 'advantage_filter_max_drop_fraction', 1.0))
	advantage_filter_max_drop_fraction = min(max(advantage_filter_max_drop_fraction, 0.0), 1.0)
	min_ppo_samples = max(0, int(getattr(training_cfg, 'min_ppo_samples', 0)))
	precision = getattr(training_cfg, 'precision', '32-bit')
	forward_chunk_agents = int(getattr(training_cfg, 'network_forward_chunk_agents', 32768))
	feature_workspace = feature_workspace or _default_workspace(config)
	batch_sizer = batch_sizer or create_ppo_batch_sizer(training_cfg)
	feature_batch_sizer = feature_batch_sizer or create_feature_batch_sizer(training_cfg)
	feature_workspace.reset_counters()
	validate_rollout_buffer(rollout_buffer, condition_state)
	device = rollout_buffer.states.device
	if device.type == 'cuda':
		torch.cuda.reset_peak_memory_stats(device)

	states_tensor = rollout_buffer.view(rollout_buffer.states)
	route_state_tensor = rollout_buffer.route_state_view()
	rewards_tensor = rollout_buffer.view(rollout_buffer.rewards)
	dones_tensor = rollout_buffer.view(rollout_buffer.dones).bool()
	values_tensor = rollout_buffer.view(rollout_buffer.values)
	old_log_probs_tensor = rollout_buffer.view(rollout_buffer.old_log_probs)
	actions_tensor = rollout_buffer.view(rollout_buffer.actions)
	time_indices_tensor = rollout_buffer.view(rollout_buffer.time_indices)

	if bootstrap_value is not None:
		last_value_pred = bootstrap_value.to(device=device, dtype=values_tensor.dtype)
		if last_value_pred.dim() == 3 and last_value_pred.shape[-1] == 1:
			last_value_pred = last_value_pred.squeeze(-1)
	else:
		last_value_pred = torch.zeros_like(values_tensor[-1])
	values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)

	seen_done_inclusive = torch.cumsum(dones_tensor, dim=0, dtype=torch.int32) > 0
	advantages, returns = gae_advantages(
		rewards_tensor, values_tp1, seen_done_inclusive, gamma, gae_lambda
	)

	A_max_tensor = torch.max(torch.abs(advantages)).detach()
	a_max_ewma = A_max_tensor if a_max_ewma is None else (beta * A_max_tensor + (1.0 - beta) * a_max_ewma.to(device))
	eta = advantage_filter_threshold * a_max_ewma
	abs_advantages = torch.abs(advantages)

	seen_done_prev = torch.roll(seen_done_inclusive, shifts=1, dims=0)
	seen_done_prev[0] = False
	first_done_step = dones_tensor & (~seen_done_prev)
	post_done_mask = seen_done_inclusive & (~first_done_step)
	active_sample_mask = states_tensor[..., 6] > 0.5
	eligible_mask = active_sample_mask & (~post_done_mask)
	threshold_keep_mask = eligible_mask & (abs_advantages >= eta)
	eligible_count = int(eligible_mask.sum().item())
	threshold_count = int(threshold_keep_mask.sum().item())
	min_keep_by_drop_cap = math.ceil(eligible_count * (1.0 - advantage_filter_max_drop_fraction)) if eligible_count > 0 else 0
	min_keep = min(eligible_count, max(min_keep_by_drop_cap, min_ppo_samples))
	protection_applied = threshold_count < min_keep
	effective_candidate_count = max(threshold_count, min_keep)

	if is_rank0:
		print(f"PPO update {update_step}: max |A|={A_max_tensor.item():.4f}, threshold={eta.item():.4f}")
		print(
			f"📊 eligible: {eligible_count}, threshold后: {threshold_count}, "
			f"保护后候选池: {effective_candidate_count}, min_keep: {min_keep} "
			f"(最多过滤 {advantage_filter_max_drop_fraction:.0%}, min_ppo_samples={min_ppo_samples})"
		)

	local_has_samples = effective_candidate_count > 0 and eligible_count > 0 and (min_ppo_samples <= 0 or eligible_count >= min_ppo_samples)
	if not sync_bool_across_ranks(local_has_samples, device, op=dist.ReduceOp.MIN):
		if is_rank0:
			print("⚠️ 至少一个rank eligible样本不足，本轮跳过以避免少样本PPO/DDP不同步")
		stats = make_update_stats("too_few_eligible_samples", device)
		stats['num_candidates'] = int(effective_candidate_count)
		stats['num_eligible'] = int(eligible_count)
		stats['num_threshold_candidates'] = int(threshold_count)
		stats['num_min_keep'] = int(min_keep)
		stats['filter_protection_applied'] = bool(protection_applied)
		stats['ppo_update_time_s'] = time.time() - update_start_time
		return a_max_ewma.detach(), stats

	def sample_mask_indices(mask: torch.Tensor, k: int, mask_count: int) -> torch.Tensor:
		"""Sample up to k indices from a [T, B, M] boolean mask without materializing huge pools."""
		if k <= 0 or mask_count <= 0:
			return torch.empty((0, 3), dtype=torch.long, device=device)
		if k >= mask_count or mask_count <= 1_000_000:
			idx = mask.nonzero(as_tuple=False)
			if idx.shape[0] <= k:
				return idx
			rand_pos = torch.randperm(idx.shape[0], device=device)[:k]
			return idx[rand_pos]

		flat_mask = mask.reshape(-1)
		total = flat_mask.numel()
		density = max(mask_count / max(total, 1), 1e-8)
		chunks = []
		collected = 0
		for _ in range(8):
			need = k - collected
			if need <= 0:
				break
			draw_count = min(total, max(k * 2, int(math.ceil(need / density * 1.5))))
			flat_idx = torch.randint(total, (draw_count,), device=device)
			flat_idx = flat_idx[flat_mask[flat_idx]]
			if flat_idx.numel() == 0:
				continue
			chunks.append(flat_idx)
			collected += int(flat_idx.numel())

		if chunks:
			flat_selected = torch.unique(torch.cat(chunks))
			if flat_selected.numel() >= k:
				flat_selected = flat_selected[torch.randperm(flat_selected.numel(), device=device)[:k]]
				B = mask.shape[1]
				M = mask.shape[2]
				t = flat_selected // (B * M)
				rem = flat_selected % (B * M)
				b = rem // M
				m = rem % M
				return torch.stack((t, b, m), dim=1)

		idx = mask.nonzero(as_tuple=False)
		if idx.shape[0] <= k:
			return idx
		rand_pos = torch.randperm(idx.shape[0], device=device)[:k]
		return idx[rand_pos]

	N = int(effective_candidate_count)
	K_target = batch_size_per_gpu if batch_size_per_gpu > 0 else N
	K = min(K_target, N)
	selected_from_threshold = 0
	selected_from_protection = 0
	if protection_applied:
		if N <= 1_000_000:
			pool_pos = torch.randperm(N, device=device)[:K]
		else:
			pool_pos = torch.randint(N, (K,), device=device)
		selected_from_threshold = int((pool_pos < threshold_count).sum().item()) if threshold_count > 0 else 0
		selected_from_protection = K - selected_from_threshold
		selected_parts = []
		if selected_from_threshold > 0:
			selected_parts.append(sample_mask_indices(threshold_keep_mask, selected_from_threshold, threshold_count))
		if selected_from_protection > 0:
			extra_mask = eligible_mask & (~threshold_keep_mask)
			selected_parts.append(sample_mask_indices(extra_mask, selected_from_protection, eligible_count - threshold_count))
		selected_idx = torch.cat(selected_parts, dim=0) if selected_parts else torch.empty((0, 3), dtype=torch.long, device=device)
		if selected_idx.shape[0] > K:
			selected_idx = selected_idx[torch.randperm(selected_idx.shape[0], device=device)[:K]]
	elif threshold_count >= K:
		selected_idx = sample_mask_indices(threshold_keep_mask, K, threshold_count)
		selected_from_threshold = K
	else:
		threshold_idx = threshold_keep_mask.nonzero(as_tuple=False)
		remaining = K - int(threshold_idx.shape[0])
		extra_idx = sample_mask_indices(eligible_mask & (~threshold_keep_mask), remaining, eligible_count - threshold_count)
		selected_idx = torch.cat([threshold_idx, extra_idx], dim=0)
		if selected_idx.shape[0] > K:
			selected_idx = selected_idx[torch.randperm(selected_idx.shape[0], device=device)[:K]]
		selected_from_threshold = int(min(threshold_count, K))
		selected_from_protection = max(0, K - selected_from_threshold)
	K = int(selected_idx.shape[0])
	if K <= 0:
		if is_rank0:
			print("⚠️ 过滤保护后仍无可用样本，本轮跳过")
		stats = make_update_stats("no_samples_after_filter", device)
		stats['num_candidates'] = int(effective_candidate_count)
		stats['num_eligible'] = int(eligible_count)
		stats['num_threshold_candidates'] = int(threshold_count)
		stats['num_min_keep'] = int(min_keep)
		stats['filter_protection_applied'] = bool(protection_applied)
		stats['ppo_update_time_s'] = time.time() - update_start_time
		return a_max_ewma.detach(), stats
	selected_t = selected_idx[:, 0]
	selected_b = selected_idx[:, 1]
	selected_m = selected_idx[:, 2]
	if is_rank0:
		protection_msg = "，已触发过滤保护" if protection_applied else ""
		print(
			f"🎯 随机选取 {K} 个样本用于更新（候选池 {N}, 目标 {K_target}{protection_msg}, "
			f"阈值样本 {selected_from_threshold}, 保护补样 {selected_from_protection}）"
		)

	old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
	old_values_batch = values_tensor[selected_t, selected_b, selected_m].view(-1)
	raw_advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
	returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
	actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1).to(torch.long)
	if is_rank0 and diagnostics_enabled(config, 'ppo'):
		action_values = None
		try:
			action_values = simulator.dynamics_model.discrete_action_space.get_all_actions()
		except Exception:
			action_values = None
		num_actions = int(getattr(getattr(training_cfg, 'network', SimpleNamespace()), 'num_actions', 12))
		print_selected_batch_diagnostics(
			actions_batch,
			raw_advantages_batch,
			returns_batch,
			old_log_probs_batch,
			num_actions,
			action_values=action_values,
		)
	advantages_batch = (raw_advantages_batch - raw_advantages_batch.mean()) / (raw_advantages_batch.std(unbiased=False) + 1e-8)
	profile_on = profile_enabled(config)
	feature_cache_start = profile_timer_start(device, config) if profile_on else time.time()
	feature_cache, feature_cache_oom_retries, feature_cache_offloads = build_selected_feature_cache(
		states_tensor,
		route_state_tensor,
		condition_state,
		selected_t,
		selected_b,
		selected_m,
		time_indices_tensor,
		simulator,
		config,
		precision,
		feature_workspace,
		feature_batch_sizer,
	)
	ppo_feature_cache_build_ms = (
		profile_elapsed_ms(feature_cache_start, device, config)
		if profile_on
		else (time.time() - feature_cache_start) * 1000.0
	)

	policy_params = get_policy_parameters(model)
	value_params = get_value_parameters(model)
	model.train()
	did_optimizer_step = False
	completed_epochs = 0
	memory_adapted = False
	microbatch_retries = 0
	ddp_gradient_buckets = 0
	abort_reason = ""
	last_policy_loss = None
	last_value_loss = None
	last_entropy = None
	last_approx_kl = None
	last_old_approx_kl = None
	last_clip_frac = None
	last_ratio_mean = None
	last_ratio_min = None
	last_ratio_max = None
	last_max_action_prob_mean = None
	microbatch_size = sync_min_int_across_ranks(batch_sizer.choose(K), device)
	if is_rank0:
		print(
			f"🧠 PPO effective batch={K}, initial microbatch={microbatch_size}, "
			f"adaptive ceiling={batch_sizer.maximum}"
		)

	for epoch in range(ppo_epochs):
		while True:
			policy_optimizer.zero_grad(set_to_none=True)
			value_optimizer.zero_grad(set_to_none=True)
			epoch_policy_loss = torch.zeros((), device=device, dtype=torch.float32)
			epoch_value_loss = torch.zeros((), device=device, dtype=torch.float32)
			epoch_entropy = torch.zeros((), device=device, dtype=torch.float32)
			epoch_approx_kl = torch.zeros((), device=device, dtype=torch.float32)
			epoch_old_approx_kl = torch.zeros((), device=device, dtype=torch.float32)
			epoch_clip_frac = torch.zeros((), device=device, dtype=torch.float32)
			epoch_ratio_mean = torch.zeros((), device=device, dtype=torch.float32)
			epoch_max_action_prob = torch.zeros((), device=device, dtype=torch.float32)
			epoch_ratio_min = None
			epoch_ratio_max = None
			local_oom = False
			oom_detail = ""
			features_chunk = None
			action_logits = None
			value_pred_full = None
			total_loss = None

			# Suppress DDP's per-microbatch reductions.  Gradients are averaged once
			# after every rank has completed all local chunks, which also makes an
			# OOM on one rank recoverable without leaving peers in a collective.
			no_sync_context = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
			try:
				with no_sync_context:
					for start in range(0, K, microbatch_size):
						end = min(start + microbatch_size, K)
						features_chunk = feature_cache.get(start, end)

						chunk_old_logp = old_log_probs_batch[start:end]
						chunk_old_value = old_values_batch[start:end]
						chunk_adv = advantages_batch[start:end]
						chunk_ret = returns_batch[start:end]
						chunk_actions = actions_batch[start:end]
						chunk_weight = (end - start) / float(K)

						with make_autocast_context(device, precision):
							action_logits, value_pred_full = forward_model(
								model,
								features_chunk,
								mode="both",
								chunk_agents=forward_chunk_agents,
							)
							logits_selected = action_logits[:, 0]
							dist_selected = torch.distributions.Categorical(logits=logits_selected)
							new_log_probs = dist_selected.log_prob(chunk_actions)
							log_ratio = new_log_probs - chunk_old_logp
							ratio = torch.exp(log_ratio)
							surr1 = ratio * chunk_adv
							surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * chunk_adv
							policy_loss = -torch.min(surr1, surr2).mean()
							entropy = dist_selected.entropy().mean()
							approx_kl = ((ratio - 1.0) - log_ratio).mean()
							old_approx_kl = (-log_ratio).mean()
							clip_frac = ((ratio - 1.0).abs() > clip_ratio).to(torch.float32).mean()
							ratio_mean = ratio.mean()
							ratio_min = ratio.min()
							ratio_max = ratio.max()
							max_action_prob_mean = dist_selected.probs.max(dim=-1).values.mean()
							value_pred = value_pred_full[:, 0]
							if value_clip_ratio is None:
								value_loss = (value_pred - chunk_ret).pow(2).mean()
							else:
								value_pred_clipped = chunk_old_value + torch.clamp(
									value_pred - chunk_old_value,
									-value_clip_ratio,
									value_clip_ratio,
								)
								value_loss = torch.maximum(
									(value_pred - chunk_ret).pow(2),
									(value_pred_clipped - chunk_ret).pow(2),
								).mean()
							total_loss = (
								policy_loss - entropy_coef * entropy + value_loss_coef * value_loss
							) * chunk_weight

						if amp_scaler is not None and getattr(amp_scaler, 'is_enabled', lambda: False)():
							amp_scaler.scale(total_loss).backward()
						else:
							total_loss.backward()

						epoch_policy_loss += policy_loss.detach().float() * chunk_weight
						epoch_value_loss += value_loss.detach().float() * chunk_weight
						epoch_entropy += entropy.detach().float() * chunk_weight
						epoch_approx_kl += approx_kl.detach().float() * chunk_weight
						epoch_old_approx_kl += old_approx_kl.detach().float() * chunk_weight
						epoch_clip_frac += clip_frac.detach().float() * chunk_weight
						epoch_ratio_mean += ratio_mean.detach().float() * chunk_weight
						epoch_max_action_prob += max_action_prob_mean.detach().float() * chunk_weight
						ratio_min_detached = ratio_min.detach().float()
						ratio_max_detached = ratio_max.detach().float()
						epoch_ratio_min = (
							ratio_min_detached
							if epoch_ratio_min is None
							else torch.minimum(epoch_ratio_min, ratio_min_detached)
						)
						epoch_ratio_max = (
							ratio_max_detached
							if epoch_ratio_max is None
							else torch.maximum(epoch_ratio_max, ratio_max_detached)
						)
						features_chunk = None
						action_logits = None
						value_pred_full = None
						total_loss = None
			except RuntimeError as exc:
				if not is_cuda_oom_error(exc, device):
					raise
				local_oom = True
				oom_detail = str(exc).splitlines()[0]
				oom_rank = 0 if rank is None else rank
				print(
					f"[Rank {oom_rank}] CUDA OOM at PPO microbatch={microbatch_size}: "
					f"{oom_detail}",
					flush=True,
				)
				features_chunk = None
				action_logits = None
				value_pred_full = None
				total_loss = None
				feature_workspace.clear_scratch()
				clear_optimizer_gradients(policy_optimizer, value_optimizer, device)

			all_ranks_succeeded = sync_bool_across_ranks(not local_oom, device, op=dist.ReduceOp.MIN)
			if not all_ranks_succeeded:
				memory_adapted = True
				microbatch_retries += 1
				feature_workspace.clear_scratch()
				clear_optimizer_gradients(policy_optimizer, value_optimizer, device)
				offloaded_here = False
				if str(getattr(training_cfg, 'ppo_feature_cache_location', 'auto')).lower() == 'auto':
					offloaded_here = feature_cache.offload_to_cpu()
				any_cache_offloaded = sync_bool_across_ranks(
					offloaded_here, device, op=dist.ReduceOp.MAX
				)
				if any_cache_offloaded:
					feature_cache_offloads += int(offloaded_here)
					if is_rank0:
						print("PPO feature cache offloaded to CPU after CUDA OOM")
					continue
				next_microbatch = batch_sizer.backoff(microbatch_size)
				if next_microbatch is None:
					abort_reason = "cuda_oom_at_min_microbatch"
					if is_rank0:
						print(
							f"⚠️ PPO OOM at minimum microbatch={microbatch_size}; "
							f"stopping remaining epochs. {oom_detail}"
						)
					break
				microbatch_size = sync_min_int_across_ranks(next_microbatch, device)
				if is_rank0:
					print(
						f"♻️ PPO CUDA OOM; retrying epoch {epoch + 1} "
						f"with microbatch={microbatch_size}"
					)
				continue

			# Gradients are still AMP-scaled here.  Reducing them first propagates
			# any non-finite value to every rank so GradScaler stays synchronized.
			ddp_gradient_buckets = average_gradients_across_ranks(
				model,
				K,
				device,
				bucket_cap_mb=float(getattr(training_cfg, 'ddp_gradient_bucket_mb', 25.0)),
			)
			if amp_scaler is not None and getattr(amp_scaler, 'is_enabled', lambda: False)():
				old_scale = amp_scaler.get_scale()
				amp_scaler.unscale_(policy_optimizer)
				amp_scaler.unscale_(value_optimizer)
				torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
				torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
				amp_scaler.step(policy_optimizer)
				amp_scaler.step(value_optimizer)
				amp_scaler.update()
				did_optimizer_step = did_optimizer_step or (amp_scaler.get_scale() >= old_scale)
			else:
				torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
				torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
				policy_optimizer.step()
				value_optimizer.step()
				did_optimizer_step = True

			completed_epochs += 1
			epoch_summary = torch.stack((
				epoch_policy_loss,
				epoch_value_loss,
				epoch_entropy,
				epoch_approx_kl,
				epoch_old_approx_kl,
				epoch_clip_frac,
				epoch_ratio_mean,
				epoch_ratio_min,
				epoch_ratio_max,
				epoch_max_action_prob,
			)).detach().cpu().tolist()
			(
				last_policy_loss,
				last_value_loss,
				last_entropy,
				last_approx_kl,
				last_old_approx_kl,
				last_clip_frac,
				last_ratio_mean,
				last_ratio_min,
				last_ratio_max,
				last_max_action_prob_mean,
			) = (float(value) for value in epoch_summary)
			break

		if abort_reason:
			break

		if is_rank0:
			print(
				f"   Epoch {epoch+1}/{ppo_epochs}: Policy Loss: {last_policy_loss:.6f}, "
				f"Value Loss: {last_value_loss:.6f}, Entropy: {last_entropy:.6f}, "
				f"KL: {last_approx_kl:.6f}, oldKL: {last_old_approx_kl:.6f}, "
				f"clip_frac: {last_clip_frac:.3f}, ratio: {last_ratio_mean:.3f}/"
				f"{last_ratio_min:.3f}-{last_ratio_max:.3f}, maxp: {last_max_action_prob_mean:.3f}"
			)

	if not abort_reason:
		batch_sizer.record_success(microbatch_size, K)

	model.eval()
	if is_rank0:
		if abort_reason:
			print(f"PPO update {update_step} stopped early: {abort_reason}")
		else:
			print(f"PPO update {update_step} complete")
	stats = make_update_stats(abort_reason, device)
	stats.update({
		'did_optimizer_step': did_optimizer_step,
		'num_candidates': int(N),
		'num_selected': int(K),
		'ppo_microbatch_size': int(microbatch_size),
		'ppo_microbatch_count': int(math.ceil(K / max(1, microbatch_size))),
		'ppo_microbatch_retries': int(microbatch_retries),
		'ppo_memory_adapted': bool(memory_adapted),
		'num_eligible': int(eligible_count),
		'num_threshold_candidates': int(threshold_count),
		'num_min_keep': int(min_keep),
		'num_selected_threshold': int(selected_from_threshold),
		'num_selected_protection': int(selected_from_protection),
		'filter_protection_applied': bool(protection_applied),
		'num_epochs': int(completed_epochs),
		'policy_loss': last_policy_loss,
		'value_loss': last_value_loss,
		'entropy': last_entropy,
		'approx_kl': last_approx_kl,
		'old_approx_kl': last_old_approx_kl,
		'clip_frac': last_clip_frac,
		'ratio_mean': last_ratio_mean,
		'ratio_min': last_ratio_min,
		'ratio_max': last_ratio_max,
		'max_action_prob_mean': last_max_action_prob_mean,
		'ppo_update_time_s': time.time() - update_start_time,
		'ppo_feature_cache_build_ms': ppo_feature_cache_build_ms,
		'ppo_feature_cache_location': feature_cache.location,
		'ppo_feature_cache_oom_retries': int(feature_cache_oom_retries),
		'ppo_feature_cache_offloads': int(feature_cache_offloads),
		'ddp_gradient_buckets': int(ddp_gradient_buckets),
	})
	stats.update(cuda_memory_stats(device))
	if is_rank0 and device.type == 'cuda':
		print(
			f"🧠 PPO峰值显存: allocated={stats['max_memory_allocated_mb']:.1f}MB, "
			f"reserved={stats['max_memory_reserved_mb']:.1f}MB"
		)
	return a_max_ewma.detach(), stats

@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device
    store: object = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
		
    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def update_rank(self):
        return self.rank if self.is_distributed else None

    def all_true(self, value: bool) -> bool:
        if not self.is_distributed:
            return value
        return sync_bool_across_ranks(value, self.device, op=dist.ReduceOp.MIN)

    def log(self, message: str, *, all_ranks: bool = False):
        if all_ranks or self.is_primary:
            print(f"[Rank {self.rank}] {message}", flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

	
def initialize_distributed_context(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> DistributedContext:
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
		
    store = None
    if world_size > 1:
        if os.name == 'nt':
            os.environ['GLOO_DEVICE_TRANSPORT'] = 'tcp'
            os.environ.pop('GLOO_SOCKET_IFNAME', None)
        store = dist.TCPStore(
            master_addr,
            master_port,
            world_size,
            rank == 0,
            timeout=timedelta(seconds=180),
        )
        backend = 'nccl' if (os.name != 'nt' and torch.cuda.is_available()) else 'gloo'
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=timedelta(seconds=180),
        )
    return DistributedContext(rank=rank, world_size=world_size, device=device, store=store)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
		

def validate_runtime_contract(config, simulator, context: DistributedContext):
    schema = FeatureSchema.from_config(config)
    configured_actions = int(getattr(config.training.network, 'num_actions', 12))
    simulator_actions = int(simulator.dynamics_model.discrete_action_space.num_actions)
    if configured_actions != simulator_actions:
        raise ValueError(
            f"network num_actions={configured_actions} does not match simulator action space={simulator_actions}"
        )

    per_rank = resolve_ppo_samples_per_rank(config.training)
    global_target = getattr(config.training, 'batch_size', None)
    effective_target = per_rank * context.world_size
    target_text = f", paper global target={int(global_target)}" if global_target is not None else ""
    context.log(
        f"feature width={schema.total_input_dim}, PPO samples/rank={per_rank}, "
        f"effective target={effective_target}{target_text}"
    )


def adapt_num_envs_to_memory(config_dict: dict, device: torch.device) -> int:
    """Cap worlds/rank from free VRAM using rollout and PPO working-set bytes."""
    simulator_cfg = config_dict['simulator']
    training_cfg = config_dict['training']
    configured = int(simulator_cfg['num_envs'])
    if device.type != 'cuda' or not bool(training_cfg.get('memory_adaptation', {}).get('enabled', True)):
        return configured

    memory_cfg = training_cfg.get('memory_adaptation', {})
    target_fraction = min(0.98, max(0.1, float(memory_cfg.get('target_fraction', 0.85))))
    reserve_bytes = int(float(memory_cfg.get('reserve_mb', 1024.0)) * 1024 ** 2)
    minimum = max(1, int(memory_cfg.get('min_num_envs', 8)))
    free_bytes, _ = torch.cuda.mem_get_info(device)
    usable_bytes = max(0, int(free_bytes * target_fraction) - reserve_bytes)

    max_agents = int(simulator_cfg['max_agents_num'])
    rollout_length = int(training_cfg['rollout_length'])
    route_targets = 4
    rollout_bytes_per_agent_step = (
        10 * 4                         # buffered state + control
        + route_targets * 2 + 2 + 2   # compact route state
        + 3 * 4                       # reward, value, old log-prob
        + 1 + 1                       # done and uint8 action
    )
    ppo_work_bytes_per_agent_step = 32
    simulator_bytes_per_agent = 256
    bytes_per_world = (
        max_agents
        * (
            rollout_length
            * (rollout_bytes_per_agent_step + ppo_work_bytes_per_agent_step)
            + simulator_bytes_per_agent
        )
        + rollout_length * 8
    )
    budget_cap = max(1, usable_bytes // max(1, bytes_per_world))
    effective = min(configured, max(minimum, int(budget_cap)))
    simulator_cfg['num_envs'] = effective
    return effective


def preallocate_rollout_buffer(simulator, rollout_length: int) -> RolloutTensorBuffer:
    """Allocate the complete persistent rollout footprint before training starts."""
    buffer = RolloutTensorBuffer(rollout_length)
    state = simulator.agents_state
    B, M = state.shape[:2]
    alive_mask = state[..., 6] > 0.5
    route_state = simulator.get_route_state(clone=False)
    buffer.write_pre_step_from_simulator(
        simulator,
        alive_mask,
        route_state,
        time_index=torch.zeros(B, dtype=torch.long, device=state.device),
    )
    zeros = torch.zeros((B, M), dtype=state.dtype, device=state.device)
    buffer.write_post_step(
        zeros,
        torch.zeros((B, M), dtype=torch.bool, device=state.device),
        zeros,
        zeros,
        torch.zeros((B, M), dtype=torch.long, device=state.device),
    )
    buffer.clear()
    return buffer


def create_simulator_and_rollout_buffer_with_backoff(
    config_dict: dict,
    device: torch.device,
):
    """Probe real simulator/reset/buffer allocations and halve worlds on CUDA OOM."""
    memory_cfg = config_dict['training'].get('memory_adaptation', {})
    minimum = max(1, int(memory_cfg.get('min_num_envs', 8)))
    rollout_length = int(config_dict['training']['rollout_length'])
    candidate = int(config_dict['simulator']['num_envs'])
    while True:
        simulator = None
        rollout_buffer = None
        try:
            config_dict['simulator']['num_envs'] = candidate
            simulator = TeraflowSimulator(config=config_dict, device=device)
            simulator.reset(return_observation=False)
            rollout_buffer = preallocate_rollout_buffer(simulator, rollout_length)
            return simulator, rollout_buffer
        except RuntimeError as exc:
            if not is_cuda_oom_error(exc, device) or candidate <= minimum:
                raise
            next_candidate = max(minimum, candidate // 2)
            print(
                f"CUDA OOM during simulator/rollout preflight; "
                f"retrying num_envs {candidate} -> {next_candidate}",
                flush=True,
            )
            simulator = None
            rollout_buffer = None
            gc.collect()
            torch.cuda.empty_cache()
            candidate = next_candidate

		
def run_training_loop(
    context: DistributedContext,
    model,
    simulator,
    config,
    policy_optimizer,
    value_optimizer,
    policy_scheduler,
    value_scheduler,
    progress: dict,
    total_updates: int,
    amp_scaler,
    batch_sizers: dict,
    rollout_buffer: RolloutTensorBuffer = None,
    experiment_tracker=None,
):
    training_cfg = config.training
    max_episode_length = int(training_cfg.max_episode_length)
    checkpoint_interval = int(getattr(training_cfg, 'checkpoint_interval', 100))
    checkpoint_dir = getattr(training_cfg, 'checkpoint_dir')
    log_interval = int(getattr(training_cfg, 'log_interval', 10))
    rollout_length = int(getattr(training_cfg, 'rollout_length', 128))
    precision = getattr(training_cfg, 'precision', '32-bit')
    forward_chunk_agents = int(getattr(training_cfg, 'network_forward_chunk_agents', 32768))
    profile_on = profile_enabled(config)
    feature_workspace = FeatureBuildWorkspace(config)
    ppo_batch_sizer = batch_sizers['ppo']
    rollout_batch_sizer = batch_sizers['rollout']
    a_max_ewma = progress.pop('a_max_ewma', None)
    progress.setdefault('update_step', 0)
    progress.setdefault('environment_steps', 0)
    progress.setdefault('rank0_completed_world_episodes', 0)
    global_worlds_per_step = sum_int_across_ranks(simulator.num_envs, context.device)

    try:
        diag_action_values = simulator.dynamics_model.discrete_action_space.get_all_actions()
    except Exception:
        diag_action_values = None

    if int(progress['update_step']) >= total_updates:
        context.log(
            f"checkpoint already reached update {progress['update_step']}/{total_updates}"
        )
        return progress

    if max_episode_length % rollout_length != 0:
        raise ValueError(
            "training.max_episode_length must be divisible by training.rollout_length "
            "when masked resets occur at rollout boundaries"
        )
    if simulator.agents_state is None:
        if profile_on:
            reset_profile_start = profile_timer_start(context.device, config)
        simulator.reset(return_observation=False)
        if profile_on and context.is_primary:
            reset_ms = profile_elapsed_ms(reset_profile_start, context.device, config)
            print(f"reset={reset_ms:.2f}ms, initial_feature_build=0.00ms")

    condition_state = snapshot_condition_state(simulator)
    rollout_buffer = rollout_buffer or RolloutTensorBuffer(rollout_length)
    cumulative_done_all = None
    episode_steps = torch.zeros(
        simulator.num_envs, dtype=torch.long, device=context.device
    )
    last_checkpoint_update = 0

    for update_number in range(int(progress['update_step']) + 1, total_updates + 1):
        context.log(f"starting update {update_number}/{total_updates}")
        update_start_time = time.time()
        rollout_buffer.clear()
        rollout_diag = init_rollout_diagnostics(config, context.device) if context.is_primary else None
        rollout_target = rollout_length

        while len(rollout_buffer) < rollout_target:
            alive_mask = rollout_alive_mask(simulator, cumulative_done_all)

            step_start_time = time.time()
            actions, old_log_probs, value_pred, rollout_profile = rollout_forward_alive_agents(
                model,
                simulator,
                config,
                alive_mask,
                condition_state,
                dropout_step=episode_steps,
                precision=precision,
                forward_chunk_agents=forward_chunk_agents,
                sample_actions=True,
                feature_workspace=feature_workspace,
                batch_sizer=rollout_batch_sizer,
            )
            policy_forward_ms = rollout_profile['policy_ms']
            feature_build_ms = rollout_profile['feature_ms']
				
            pre_route_state = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
            rollout_buffer.write_pre_step_from_simulator(
                simulator,
                alive_mask,
                pre_route_state,
                time_index=episode_steps,
            )

            if profile_on:
                env_profile_start = profile_timer_start(context.device, config)
            reward, done = simulator.step(actions, return_observation=False)
            if profile_on:
                env_step_ms = profile_elapsed_ms(env_profile_start, context.device, config)

            rollout_buffer.write_post_step(reward, done, value_pred, old_log_probs, actions)
            update_rollout_diagnostics(rollout_diag, simulator, alive_mask, reward, done, actions)

            current_done_all = done.detach().bool()
            cumulative_done_all = (
                current_done_all.clone()
                if cumulative_done_all is None
                else cumulative_done_all | current_done_all
            )
            episode_steps.add_(1)
            progress['environment_steps'] += global_worlds_per_step
            if context.is_primary and len(rollout_buffer) % log_interval == 0:
                print(
                    f"rollout step {len(rollout_buffer)}/{rollout_target}: "
                    f"{time.time() - step_start_time:.4f}s"
                )
            if (
                context.is_primary
                and profile_on
                and len(rollout_buffer) % profile_log_interval(config) == 0
            ):
                step_profile = format_profile(getattr(simulator, 'last_step_profile', {}))
                rollout_path = rollout_profile.get('path', 'none')
                rollout_selected = int(rollout_profile.get('num_selected', 0) or 0)
                rollout_chunks = int(rollout_profile.get('feature_chunks', 0) or 0)
                print(
                    f"profile step={len(rollout_buffer)}: policy={policy_forward_ms:.2f}ms, "
                    f"env={env_step_ms:.2f}ms, feature={feature_build_ms:.2f}ms, "
                    f"path={rollout_path}, selected={rollout_selected}, "
                    f"feature_chunks={rollout_chunks}"
                    + (f", {step_profile}" if step_profile else "")
                )

        bootstrap_value = current_rollout_bootstrap_value(
            model,
            simulator,
            config,
            cumulative_done_all,
            condition_state,
            episode_steps,
            precision,
            forward_chunk_agents,
            feature_workspace=feature_workspace,
            batch_sizer=rollout_batch_sizer,
        )
        if context.is_primary:
            print_rollout_diagnostics(
                rollout_diag,
                action_values=diag_action_values,
                prefix=f"rollout diagnostics (update {update_number}, len {len(rollout_buffer)})",
            )

        a_max_ewma, update_stats = perform_ppo_update(
            model,
            policy_optimizer,
            value_optimizer,
            rollout_buffer,
            condition_state,
            simulator,
            config,
            update_number,
            rank=context.update_rank,
            a_max_ewma=a_max_ewma,
            amp_scaler=amp_scaler,
            bootstrap_value=bootstrap_value,
            feature_workspace=feature_workspace,
            batch_sizer=ppo_batch_sizer,
            feature_batch_sizer=batch_sizers['feature'],
        )

        progress['update_step'] = update_number
        scheduler_stepped = step_schedulers_if_updated(
            policy_scheduler, value_scheduler, update_stats
        )
        if context.is_primary and profile_on:
            print(
                f"update profile: scheduler_step={scheduler_stepped}, "
                f"samples={update_stats.get('num_selected', 0)}, "
                f"ppo={update_stats.get('ppo_update_time_s', 0.0):.3f}s, "
                f"ppo_feature_cache={update_stats.get('ppo_feature_cache_build_ms', 0.0):.2f}ms, "
                f"mem_alloc={update_stats.get('max_memory_allocated_mb', 0.0):.1f}MB, "
                f"oom_retries={update_stats.get('ppo_microbatch_retries', 0)}, "
                f"skip={update_stats.get('skip_reason', '')}"
            )

        world_alive = rollout_alive_mask(simulator, cumulative_done_all).any(dim=1)
        reset_world_mask = (~world_alive) | (episode_steps >= max_episode_length)
        local_completed = int(reset_world_mask.sum().item())
        if context.is_primary:
            progress['rank0_completed_world_episodes'] += local_completed
        if local_completed > 0:
            simulator.reset_worlds(reset_world_mask, return_observation=False)
            episode_steps.masked_fill_(reset_world_mask, 0)
            cumulative_done_all = cumulative_done_all & (~reset_world_mask.unsqueeze(1))
            condition_state = snapshot_condition_state(simulator)

        if context.is_primary:
            update_time_s = time.time() - update_start_time
            print(f"update time: {update_time_s:.4f}s")
            if experiment_tracker is not None:
                metrics = {
                    'training/update_step': update_number,
                    'training/environment_steps': progress['environment_steps'],
                    'training/rank0_completed_world_episodes': progress['rank0_completed_world_episodes'],
                    'training/update_time_s': update_time_s,
                    'training/policy_learning_rate': policy_optimizer.param_groups[0]['lr'],
                    'training/value_learning_rate': value_optimizer.param_groups[0]['lr'],
                    'ppo/optimizer_step': update_stats.get('did_optimizer_step', False),
                    'ppo/candidates': update_stats.get('num_candidates', 0),
                    'ppo/selected_samples': update_stats.get('num_selected', 0),
                    'ppo/update_time_s': update_stats.get('ppo_update_time_s', 0.0),
                    'ppo/feature_cache_build_ms': update_stats.get('ppo_feature_cache_build_ms', 0.0),
                    'ppo/feature_cache_location': update_stats.get('ppo_feature_cache_location'),
                    'ppo/feature_cache_on_gpu': (
                        update_stats.get('ppo_feature_cache_location') == 'cuda'
                    ),
                    'ppo/feature_cache_oom_retries': update_stats.get('ppo_feature_cache_oom_retries', 0),
                    'ppo/feature_cache_offloads': update_stats.get('ppo_feature_cache_offloads', 0),
                    'system/ddp_gradient_buckets': update_stats.get('ddp_gradient_buckets', 0),
                    'ppo/oom_retries': update_stats.get('ppo_microbatch_retries', 0),
                    'ppo/memory_adapted': update_stats.get('ppo_memory_adapted', False),
                    'system/max_memory_allocated_mb': update_stats.get('max_memory_allocated_mb', 0.0),
                    'system/max_memory_reserved_mb': update_stats.get('max_memory_reserved_mb', 0.0),
                }
                for key in (
                    'ppo_microbatch_size',
                    'ppo_microbatch_count',
                    'num_epochs',
                    'policy_loss',
                    'value_loss',
                    'entropy',
                    'approx_kl',
                    'old_approx_kl',
                    'clip_frac',
                    'ratio_mean',
                    'ratio_min',
                    'ratio_max',
                    'max_action_prob_mean',
                ):
                    metrics[f"ppo/{key}"] = update_stats.get(key)
                metrics.update(rollout_diagnostic_metrics(rollout_diag))
                experiment_tracker.log(metrics, update_number)

        if update_number % checkpoint_interval == 0:
            save_checkpoint(
                model,
                policy_optimizer,
                value_optimizer,
                policy_scheduler,
                value_scheduler,
                amp_scaler,
                batch_sizers,
                progress,
                a_max_ewma,
                checkpoint_dir,
                context.device,
				rank=context.rank,
            )
            last_checkpoint_update = update_number

    if int(progress['update_step']) != last_checkpoint_update:
        save_checkpoint(
            model,
            policy_optimizer,
            value_optimizer,
            policy_scheduler,
            value_scheduler,
            amp_scaler,
            batch_sizers,
            progress,
            a_max_ewma,
            checkpoint_dir,
            context.device,
			rank=context.rank,
        )

    context.log("training complete")
    return progress

					
def ddppo_worker(
    rank: int,
    gpu_count: int,
    config_dict: dict,
    master_addr: str,
    master_port: int,
):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(
                encoding='utf-8',
                errors='replace',
                line_buffering=True,
                write_through=True,
            )
        except Exception:
            pass

    context = None
    experiment_tracker = None
    try:
        context = initialize_distributed_context(
            rank,
            gpu_count,
            master_addr,
            master_port,
        )
        context.log(
            f"worker start: {'DDP' if context.is_distributed else 'single GPU'} on {context.device}",
            all_ranks=context.is_distributed,
        )
        local_config_dict = json.loads(json.dumps(config_dict))
        config = json.loads(
            json.dumps(local_config_dict), object_hook=lambda value: SimpleNamespace(**value)
        )
        model = create_network(config=config, network_type="independent").to(context.device)
        if context.is_distributed:
            if context.device.type == 'cuda':
                model = DDP(
                    model,
                    device_ids=[rank],
                    output_device=rank,
                    broadcast_buffers=False,
                )
            else:
                model = DDP(model, broadcast_buffers=False)

        training_cfg = config.training
        learning_rate = float(getattr(training_cfg, 'learning_rate', 5e-4))
        total_updates = int(getattr(training_cfg, 'total_updates'))
        policy_optimizer = optim.Adam(get_policy_parameters(model), lr=learning_rate)
        value_optimizer = optim.Adam(get_value_parameters(model), lr=learning_rate)
        policy_scheduler = create_lr_scheduler(policy_optimizer, training_cfg, total_updates)
        value_scheduler = create_lr_scheduler(value_optimizer, training_cfg, total_updates)
        amp_scaler = make_grad_scaler(context.device, getattr(training_cfg, 'precision', '32-bit'))
        batch_sizers = {
            'ppo': create_ppo_batch_sizer(training_cfg),
            'feature': create_feature_batch_sizer(training_cfg),
            'rollout': create_rollout_batch_sizer(config),
        }

        progress = {
            'update_step': 0,
            'environment_steps': 0,
            'rank0_completed_world_episodes': 0,
        }
        resume_from = getattr(training_cfg, 'resume_from', None)
        if resume_from:
            progress = load_checkpoint(
                model,
                policy_optimizer,
                value_optimizer,
                policy_scheduler,
                value_scheduler,
                amp_scaler,
                batch_sizers,
                resume_from,
                context.device,
                rank=context.rank,
            )
            context.log(
                f"resumed checkpoint {resume_from}, update_step={progress['update_step']}"
            )
            if int(progress['update_step']) > total_updates:
                raise ValueError(
                    f"checkpoint update_step={progress['update_step']} exceeds "
                    f"training.total_updates={total_updates}"
                )
            reconcile_lr_scheduler_horizon(policy_scheduler, total_updates)
            reconcile_lr_scheduler_horizon(value_scheduler, total_updates)

        configured_envs = int(local_config_dict['simulator']['num_envs'])
        effective_envs = adapt_num_envs_to_memory(local_config_dict, context.device)
        if effective_envs != configured_envs:
            context.log(
                f"memory budget adjusted num_envs/rank {configured_envs} -> {effective_envs}",
                all_ranks=context.is_distributed,
            )
        config = json.loads(
            json.dumps(local_config_dict), object_hook=lambda value: SimpleNamespace(**value)
        )
        rollout_sizer_state = batch_sizers['rollout'].state_dict()
        batch_sizers['rollout'] = create_rollout_batch_sizer(config)
        batch_sizers['rollout'].load_state_dict(rollout_sizer_state)

        simulator, rollout_buffer = create_simulator_and_rollout_buffer_with_backoff(
            local_config_dict, context.device
        )
        if int(config.simulator.num_envs) != int(local_config_dict['simulator']['num_envs']):
            config = json.loads(
                json.dumps(local_config_dict), object_hook=lambda value: SimpleNamespace(**value)
            )
            rollout_sizer_state = batch_sizers['rollout'].state_dict()
            batch_sizers['rollout'] = create_rollout_batch_sizer(config)
            batch_sizers['rollout'].load_state_dict(rollout_sizer_state)

        validate_runtime_contract(config, simulator, context)

        tracker_error = None
        if context.is_primary:
            try:
                experiment_tracker = initialize_swanlab(local_config_dict)
            except Exception as exc:
                tracker_error = exc
        if not context.all_true(tracker_error is None):
            if tracker_error is not None:
                raise tracker_error
            raise RuntimeError("rank 0 failed to initialize SwanLab")

        run_training_loop(
            context,
            model,
            simulator,
            config,
            policy_optimizer,
            value_optimizer,
            policy_scheduler,
            value_scheduler,
            progress,
            total_updates,
            amp_scaler,
            batch_sizers,
            rollout_buffer=rollout_buffer,
            experiment_tracker=experiment_tracker,
        )
    except Exception as exc:
        print(f"[Rank {rank}] training failed: {exc}", flush=True)
        raise
    finally:
        if experiment_tracker is not None:
            experiment_tracker.finish()
        cleanup_ddp()


def run_distributed_ddppo(config_dict: dict, cuda_ranks: list[int]):
    if not cuda_ranks:
        raise RuntimeError("no CUDA device is available")
	
    gpu_count = len(cuda_ranks)
    master_addr = '127.0.0.1'
    master_port = _find_free_port()

    mp.set_start_method('spawn', force=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(rank) for rank in cuda_ranks)
    context = mp.get_context('spawn')
    processes = [
        context.Process(
            target=ddppo_worker,
            args=(rank, gpu_count, config_dict, master_addr, master_port),
        )
        for rank in range(gpu_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    failed = [
        (rank, process.exitcode)
        for rank, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"training workers failed: {failed}")

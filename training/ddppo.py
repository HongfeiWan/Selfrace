import os
import sys
import json
import socket
import math
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
def save_checkpoint(model, policy_optimizer, value_optimizer, step: int, checkpoint_dir: str):
	"""保存模型与优化器状态字典"""
	try:
		os.makedirs(checkpoint_dir, exist_ok=True)
		# 兼容 DDP 包裹
		save_model = model.module if hasattr(model, 'module') else model
		state = {
			'step': step,
			'model_state_dict': save_model.state_dict(),
			'policy_state_dict': save_model.policy_network.state_dict(),
			'value_state_dict': save_model.value_network.state_dict(),
			'policy_feature_encoder_state_dict': save_model.policy_feature_encoder.state_dict(),
			'value_feature_encoder_state_dict': save_model.value_feature_encoder.state_dict(),
			'policy_optim_state_dict': policy_optimizer.state_dict(),
			'value_optim_state_dict': value_optimizer.state_dict(),
		}
		ckpt_path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pt')
		torch.save(state, ckpt_path)
	except Exception as e:
		print(f"⚠️ 保存检查点失败: {e}")

def load_checkpoint(model, policy_optimizer, value_optimizer, checkpoint_path: str, device: torch.device) -> int:
	"""从检查点恢复模型和优化器，返回已完成的 iteration step。"""
	if not checkpoint_path:
		return 0
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
	state = torch.load(checkpoint_path, map_location=device)
	load_model = model.module if hasattr(model, 'module') else model
	if 'model_state_dict' in state:
		load_model.load_state_dict(state['model_state_dict'], strict=True)
	else:
		load_model.policy_network.load_state_dict(state['policy_state_dict'], strict=True)
		load_model.value_network.load_state_dict(state['value_state_dict'], strict=True)
		if 'policy_feature_encoder_state_dict' in state:
			load_model.policy_feature_encoder.load_state_dict(state['policy_feature_encoder_state_dict'], strict=True)
		if 'value_feature_encoder_state_dict' in state:
			load_model.value_feature_encoder.load_state_dict(state['value_feature_encoder_state_dict'], strict=True)
	if 'policy_optim_state_dict' in state:
		policy_optimizer.load_state_dict(state['policy_optim_state_dict'])
	if 'value_optim_state_dict' in state:
		value_optimizer.load_state_dict(state['value_optim_state_dict'])
	return int(state.get('step', 0))

def advance_scheduler_to_iteration(policy_scheduler, value_scheduler, completed_iterations: int):
	"""旧 checkpoint 未保存 scheduler 状态，这里按已完成 iteration 近似推进余弦调度器。"""
	completed_iterations = max(0, int(completed_iterations))
	for scheduler in (policy_scheduler, value_scheduler):
		if hasattr(scheduler, 'T_max') and hasattr(scheduler, 'eta_min'):
			scheduler.last_epoch = completed_iterations
			next_lrs = []
			for base_lr, param_group in zip(scheduler.base_lrs, scheduler.optimizer.param_groups):
				lr = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
					1 + math.cos(math.pi * completed_iterations / scheduler.T_max)
				) / 2
				param_group['lr'] = lr
				next_lrs.append(lr)
			scheduler._last_lr = next_lrs
		else:
			for _ in range(completed_iterations):
				scheduler.step()

# ============================== 观测数据拆解 ==============================
def decompose_observation(observation: torch.Tensor, config: SimpleNamespace) -> tuple:
    """
    将initial_observation拆解为网络需要的各个组件
    
    Args:
        observation: 形状为 (B, M, total_obs_dim) 的观测张量
        config: 配置对象
    
    Returns:
        tuple: (agents_state, neighbors_local, w_lanes_local, w_boundaries_local)
            - agents_state: (B, M, S_dim) - 原文式 S(t) 局部状态
            - neighbors_local: (B, M, K, neighbor_dim) - 邻居相对状态，active 位于最后一维
            - w_lanes_local: (B, M, N_lanes, lane_dim) - W_lane raw features
            - w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标 [dx, dy]
    """
    batch_size, max_agents, total_obs_dim = observation.shape
    
    # 从配置中获取维度信息
    simulator_config = config.simulator
    local_state_dim = simulator_config.observation.local_state_dim
    neighbor_feature_dim = simulator_config.observation.neighbor_feature_dim
    waypoint_feature_dim = simulator_config.observation.waypoint_feature_dim
    boundary_feature_dim = simulator_config.observation.boundary_feature_dim  # 2
    num_neighbors = simulator_config.observation.num_neighbors  # 20
    num_w_lanes = simulator_config.observation.num_w_lanes
    num_w_boundaries = simulator_config.observation.num_w_boundaries
    
    # 计算各部分在观测向量中的位置
    local_state_size = local_state_dim
    neighbors_size = num_neighbors * neighbor_feature_dim
    w_lanes_size = num_w_lanes * waypoint_feature_dim
    w_boundaries_size = num_w_boundaries * boundary_feature_dim
    
    # 1. 提取agents_state (前7个维度)
    agents_state = observation[:, :, :local_state_dim]
    
    # 2. 提取neighbors_local
    neighbors_start = local_state_size
    neighbors_end = neighbors_start + neighbors_size
    neighbors_flat = observation[:, :, neighbors_start:neighbors_end]
    neighbors_local = neighbors_flat.view(batch_size, max_agents, num_neighbors, neighbor_feature_dim)
    
    # 3. 提取w_lanes_local
    w_lanes_start = neighbors_end
    w_lanes_end = w_lanes_start + w_lanes_size
    w_lanes_flat = observation[:, :, w_lanes_start:w_lanes_end]  # (B, M, N_lanes*2)
    w_lanes_local = w_lanes_flat.view(batch_size, max_agents, num_w_lanes, waypoint_feature_dim)  # (B, M, N_lanes, 2)
    
    # 4. 提取w_boundaries_local
    w_boundaries_start = w_lanes_end
    w_boundaries_flat = observation[:, :, w_boundaries_start:]  # (B, M, N_boundaries*2)
    w_boundaries_local = w_boundaries_flat.view(batch_size, max_agents, num_w_boundaries, boundary_feature_dim)  # (B, M, N_boundaries, 2)
    
    return agents_state, neighbors_local, w_lanes_local, w_boundaries_local

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

FEATURE_PAD_VALUE = -2.0

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
        network_config = config.training.network
        self.simple_feature_dims = list(network_config.simple_feature_dims)
        self.permutation_feature_dims = list(network_config.permutation_feature_dims)
        self.permutation_element_dims = list(getattr(network_config, 'permutation_element_dims', [2, 7, 2, 7]))
        self.simple_end = sum(self.simple_feature_dims)
        self.total_input_dim = self.simple_end + sum(self.permutation_feature_dims)

    def reset_counters(self):
        self.feature_chunks = 0

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
                (0, 1), (0.00025, 0.0075), (-0.5, 0.5), (0.00025, 0.0075), (0, 1),
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
        key = (int(n), device, dtype)
        cached = self._arange_cache.get(key)
        if cached is None:
            cached = torch.arange(int(n), device=device, dtype=dtype)
            self._arange_cache[key] = cached
        return cached

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
                            goal_slots: int = 0, out: torch.Tensor = None) -> torch.Tensor:
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
        goal_dist_norm = normalize_to_minus1_1(goal_dist, 0.0, 400.0)
        lane_out[..., 5] = torch.where(distance_valid, goal_dist_norm, torch.full_like(goal_dist_norm, FEATURE_PAD_VALUE))
    if element_dim > 6:
        rel_goal_dist_norm = normalize_to_minus1_1(rel_goal_dist, 0.0, 200.0)
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
        stop_lines: (B, M, num_stop_lines, 20) - 停止线点
        reward_coef: (B, M, 10) - 奖励系数
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
    
    # 从配置中获取网络需要的特征维度
    simple_feature_dims = workspace.simple_feature_dims
    permutation_feature_dims = workspace.permutation_feature_dims
    permutation_element_dims = workspace.permutation_element_dims
    
    # 计算总输入维度
    total_input_dim = workspace.total_input_dim
    
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
    simple_end = sum(simple_feature_dims)
    has_dense_goal_vector = len(simple_feature_dims) >= 4
    goal_slots = 0
    simple_offset = 0
    
    # S(t): c, theta, kappa, v, v_lim, phi, a_long, a_lat, Cacc, Cthrottle, Csteer, l, w.
    s_t_size = simple_feature_dims[simple_offset]
    s_t_start = 0
    s_t_end = s_t_start + s_t_size
    features_tensor[:, :, s_t_start:s_t_end] = normalize_s_features(
        agents_state,
        s_t_size,
        vehicle_style=vehicle_style,
        control_state=control_state,
        workspace=workspace,
        out=features_tensor[:, :, s_t_start:s_t_end],
    )
    simple_offset += 1
    feature_cursor = s_t_end
    
    if has_dense_goal_vector:
        g_t_size = simple_feature_dims[simple_offset]
        g_t_start = feature_cursor
        g_t_end = g_t_start + g_t_size
        if navigation is None:
            goal_vector = features_tensor[:, :, g_t_start:g_t_end]
            goal_vector.zero_()
        elif _is_navigation_packet(navigation):
            nav = navigation.to(device=agents_state.device, dtype=agents_state.dtype)
            goal_slots = max(1, g_t_size // 2)
            goal_rows = nav[:, :, :goal_slots, :]
            goal_valid = goal_rows[..., 2] > 0.5
            goal_xy = torch.where(goal_valid.unsqueeze(-1), goal_rows[..., :2], torch.zeros_like(goal_rows[..., :2]))
            goal_vector = pad_or_truncate_flat(
                goal_xy.flatten(start_dim=2),
                g_t_size,
                pad_value=0.0,
                out=features_tensor[:, :, g_t_start:g_t_end],
            )
        else:
            goal_vector = features_tensor[:, :, g_t_start:g_t_end]
            goal_vector.zero_()
        goal_vector = normalize_to_minus1_1(goal_vector, -200, 200)
        features_tensor[:, :, g_t_start:g_t_end] = goal_vector
        simple_offset += 1
        feature_cursor = g_t_end

    # reward系数: 10维 - 使用传入的采样参数
    reward_coef_size = simple_feature_dims[simple_offset]
    reward_coef_start = feature_cursor
    reward_coef_end = reward_coef_start + reward_coef_size

    reward_coef = reward_coef.to(device=agents_state.device, dtype=agents_state.dtype)
    reward_out = features_tensor[:, :, reward_coef_start:reward_coef_end]
    reward_out.zero_()
    reward_min, reward_max = workspace.bounds('reward', agents_state.dtype, agents_state.device)
    copy_reward = min(reward_coef.shape[-1], reward_coef_size, reward_min.numel())
    if copy_reward > 0:
        reward_out[:, :, :copy_reward] = normalize_to_minus1_1(
            reward_coef[:, :, :copy_reward],
            reward_min[:copy_reward],
            reward_max[:copy_reward],
        )
    simple_offset += 1
    feature_cursor = reward_coef_end

    # 车辆风格参数: 4维 - 从agents_state中提取
    vehicle_style_size = simple_feature_dims[simple_offset]
    vehicle_style_start = feature_cursor
    vehicle_style_end = vehicle_style_start + vehicle_style_size
    if vehicle_style is None:
        vehicle_style = torch.ones(batch_size, max_agents, vehicle_style_size, device=agents_state.device, dtype=agents_state.dtype)
    else:
        vehicle_style = vehicle_style.to(device=agents_state.device, dtype=agents_state.dtype)
    style_out = features_tensor[:, :, vehicle_style_start:vehicle_style_end]
    style_out.zero_()
    style_min, style_max = workspace.bounds('style', agents_state.dtype, agents_state.device)
    copy_style = min(vehicle_style.shape[-1], vehicle_style_size, style_min.numel())
    if copy_style > 0:
        style_out[:, :, :copy_style] = normalize_to_minus1_1(
            vehicle_style[:, :, :copy_style],
            style_min[:copy_style],
            style_max[:copy_style],
        )
    
    # 2. 构建排列不变特征 (road_boundary, lane_points, stop_lines, other_agents)
    permutation_start = simple_end
    
    # road_boundary: 原文使用最近80个boundary coarse features
    road_boundary_size = permutation_feature_dims[0]
    road_boundary_start = permutation_start
    road_boundary_end = road_boundary_start + road_boundary_size
    
    boundary_out = features_tensor[:, :, road_boundary_start:road_boundary_end]
    w_boundaries_flat = normalize_point_set(
        w_boundaries_local,
        road_boundary_size,
        min_val=-200.0,
        max_val=200.0,
        out=boundary_out,
    )
    if w_boundaries_flat is None:
        boundary_out.fill_(FEATURE_PAD_VALUE)
    
    # lane_points: 原文式 map lane feature，每个元素包含位置、方向、宽度、目标距离。
    lane_points_size = permutation_feature_dims[1]
    lane_points_start = road_boundary_end
    lane_points_end = lane_points_start + lane_points_size
    
    lane_element_dim = permutation_element_dims[1] if len(permutation_element_dims) > 1 else 7
    lane_out = features_tensor[:, :, lane_points_start:lane_points_end]
    w_lanes_flat = build_lane_map_features(
        w_lanes_local,
        navigation,
        lane_points_size,
        lane_element_dim,
        goal_slots=goal_slots,
        out=lane_out,
    )
    if w_lanes_flat is None:
        lane_out.fill_(FEATURE_PAD_VALUE)
    
    # stop_lines: 20维 - 使用停止线信息
    stop_lines_size = permutation_feature_dims[2]  # 20
    stop_lines_start = lane_points_end
    stop_lines_end = stop_lines_start + stop_lines_size
    
    stop_out = features_tensor[:, :, stop_lines_start:stop_lines_end]
    if stop_lines is not None and stop_lines.numel() > 0:
        stop_lines_flat = normalize_point_set(
            stop_lines.to(device=agents_state.device, dtype=agents_state.dtype),
            stop_lines_size,
            out=stop_out,
        )
        if stop_lines_flat is None:
            stop_out.fill_(FEATURE_PAD_VALUE)
    else:
        stop_out.fill_(FEATURE_PAD_VALUE)
    
    # other_agents: 使用邻居位置、朝向、速度、尺寸、z 与 active mask
    other_agents_size = permutation_feature_dims[3]
    other_agents_start = stop_lines_end
    other_agents_end = other_agents_start + other_agents_size
    
    # 将邻居信息按通道做归一化后再展平并填充，active=0 的 padding 不参与网络 maxpool。
    neighbors_local = neighbors_local.to(device=agents_state.device, dtype=agents_state.dtype)
    neighbor_dim = neighbors_local.shape[-1]
    other_out = features_tensor[:, :, other_agents_start:other_agents_end]
    other_out.fill_(FEATURE_PAD_VALUE)
    neighbor_slots = min(neighbors_local.shape[2], other_agents_size // max(1, neighbor_dim))
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
def check_gpu_info(print_info: bool = True, **kwargs):
	"""
	检查GPU信息和CUDA支持情况

	Args:
		print_info: 是否打印函数内部的日志（默认True）。
		Print: 别名，兼容传入 Print=False 的调用方式。
	"""
	# 兼容别名参数 Print=False 的用法
	if 'Print' in kwargs:
		try:
			print_info = bool(kwargs['Print'])
		except Exception:
			pass

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
		'num_epochs': 0,
		'policy_loss': None,
		'value_loss': None,
		'entropy': None,
		'ppo_update_time_s': 0.0,
		'ppo_feature_rebuild_ms': 0.0,
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


def snapshot_route_state(simulator) -> dict:
	"""Clone compact route tensors for the rollout buffer."""
	if not hasattr(simulator, 'get_route_state'):
		return {}
	return simulator.get_route_state(clone=True)


def snapshot_condition_state(simulator) -> dict:
	"""Clone rollout-level condition tensors; expanded stop-line features are rebuilt on demand."""
	reward_coef = getattr(getattr(simulator, 'reward_calculator', None), 'sampled_params', None)
	vehicle_style = getattr(simulator, 'driving_style_params', None)
	traffic_light_states = getattr(simulator, 'traffic_light_states', None)
	return {
		'reward_coef': reward_coef.detach().clone() if torch.is_tensor(reward_coef) else None,
		'vehicle_style': vehicle_style.detach().clone() if torch.is_tensor(vehicle_style) else None,
		'traffic_light_states': traffic_light_states.detach().clone() if torch.is_tensor(traffic_light_states) else None,
	}


def slice_condition_state_env(condition_state: dict, start: int, end: int, total_envs: int) -> dict:
	if not condition_state:
		return {}
	return {
		key: slice_env_tensor(value, start, end, total_envs)
		for key, value in condition_state.items()
	}


def gather_condition_state_env(condition_state: dict, env_idx: torch.Tensor) -> dict:
	if not condition_state:
		return {}
	out = {}
	for key, value in condition_state.items():
		out[key] = value[env_idx] if torch.is_tensor(value) and value.dim() > 0 else value
	return out


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

	def write_pre_step(self, state: torch.Tensor, route_state: dict, time_index: int = 0):
		if self.length >= self.capacity:
			raise RuntimeError(f"RolloutTensorBuffer full: length={self.length}, capacity={self.capacity}")
		if self._pre_step_pending:
			raise RuntimeError("write_pre_step called twice before write_post_step")
		self._ensure_tensor('states', state)[self.length].copy_(state.detach())
		if self.time_indices is None:
			self.time_indices = torch.empty((self.capacity,), device=state.device, dtype=torch.long)
		self.time_indices[self.length] = int(time_index)
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
		if self.time_indices is None:
			self.time_indices = torch.empty((self.capacity,), device=state.device, dtype=torch.long)
		self.time_indices[self.length] = int(time_index)
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


def gather_current_route_state_selected(route_state_tensor: dict, env_idx: torch.Tensor,
										agent_idx: torch.Tensor) -> dict:
	if not route_state_tensor:
		return {}
	out = {}
	for key, value in route_state_tensor.items():
		selected = value[env_idx, agent_idx]
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


def build_features_from_simulator_state(simulator, config, alive_mask: torch.Tensor = None,
										condition_state: dict = None, dropout_step=0,
										workspace: FeatureBuildWorkspace = None) -> torch.Tensor:
	B, M = simulator.agents_state.shape[:2]
	workspace = workspace or _default_workspace(config)
	env_chunk = feature_build_env_chunk_size(config, M)
	total_input_dim = workspace.total_input_dim
	features = torch.empty(B, M, total_input_dim, device=simulator.agents_state.device, dtype=simulator.agents_state.dtype)
	if env_chunk <= 0:
		env_chunk = B

	condition_state = condition_state if condition_state is not None else snapshot_condition_state(simulator)
	route_state = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
	control_state_all = current_control_state(simulator)
	lane_count = int(getattr(simulator.observation_generator, 'num_w_lanes', 0))
	boundary_count = int(getattr(simulator.observation_generator, 'num_w_boundaries', 0))

	for start in range(0, B, env_chunk):
		end = min(start + env_chunk, B)
		obs_state = _policy_observation_state_chunk(simulator, start, end, alive_mask=alive_mask)
		control_chunk = slice_env_tensor(control_state_all, start, end, B)
		condition_chunk = slice_condition_state_env(condition_state, start, end, B)
		route_chunk = slice_route_state_env(route_state, start, end, B)
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
			time_idx=dropout_step,
			env_idx=env_idx,
			workspace=workspace,
		)
		build_features_from_components(
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
			out=features[start:end],
		)
	return features


def build_features_for_selected_agents(world_states: torch.Tensor, simulator, config,
									   route_state: dict, condition_state: dict,
									   agent_indices: torch.Tensor, time_indices: torch.Tensor = None,
									   env_indices: torch.Tensor = None,
									   workspace: FeatureBuildWorkspace = None,
									   out: torch.Tensor = None) -> torch.Tensor:
	B, M = world_states.shape[:2]
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

def merge_update_stats(accum: dict, update_stats: dict):
	if update_stats is None:
		return accum
	accum['did_optimizer_step'] = accum.get('did_optimizer_step', False) or update_stats.get('did_optimizer_step', False)
	accum['num_candidates'] = accum.get('num_candidates', 0) + int(update_stats.get('num_candidates', 0) or 0)
	accum['num_selected'] = accum.get('num_selected', 0) + int(update_stats.get('num_selected', 0) or 0)
	accum['ppo_update_time_s'] = accum.get('ppo_update_time_s', 0.0) + float(update_stats.get('ppo_update_time_s', 0.0) or 0.0)
	accum['ppo_feature_rebuild_ms'] = accum.get('ppo_feature_rebuild_ms', 0.0) + float(update_stats.get('ppo_feature_rebuild_ms', 0.0) or 0.0)
	for key in ('max_memory_allocated_mb', 'max_memory_reserved_mb'):
		accum[key] = max(float(accum.get(key, 0.0) or 0.0), float(update_stats.get(key, 0.0) or 0.0))
	accum['last_update'] = update_stats
	return accum

def clear_rollout_buffers(*buffers):
	for buf in buffers:
		buf.clear()

# ============================== 检查是否所有世界无存活agent ==============================
def all_worlds_no_alive_agents(simulator, cumulative_done_all=None) -> bool:
	"""
	检查是否所有世界都没有存活的智能体。
	与game.py的_check_all_worlds_no_alive_agents逻辑完全一致。
	"""
	try:
		states = simulator.agents_state  # (B, M, S)
		active_mask = states[..., 6] > 0.5
		if cumulative_done_all is None:
			alive_mask = active_mask
		else:
			alive_mask = active_mask & (~cumulative_done_all.to(active_mask.device))
		return not bool(alive_mask.any().item())
	except Exception:
		return True

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
								 feature_workspace: FeatureBuildWorkspace = None):
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
	profile = {'feature_ms': 0.0, 'policy_ms': 0.0, 'num_selected': 0, 'feature_chunks': 0, 'path': 'none'}
	feature_workspace = feature_workspace or _default_workspace(config)
	feature_workspace.reset_counters()

	alive_mask = alive_mask.to(device=device, dtype=torch.bool)
	env_idx, agent_idx = alive_mask.nonzero(as_tuple=True)
	total_selected = int(env_idx.numel())
	if total_selected <= 0:
		return actions, old_log_probs, value_pred, profile
	profile['num_selected'] = total_selected
	training_cfg = getattr(config, 'training')
	dense_alive_fraction = float(getattr(training_cfg, 'rollout_dense_alive_fraction', 0.0))
	if dense_alive_fraction > 0.0 and (total_selected / max(1, B * M)) >= dense_alive_fraction:
		profile['path'] = 'dense'
		profile_on = profile_enabled(config)
		if profile_on:
			feature_start = profile_timer_start(device, config)
		features_all = build_features_from_simulator_state(
			simulator,
			config,
			alive_mask=alive_mask,
			condition_state=condition_state,
			dropout_step=dropout_step,
			workspace=feature_workspace,
		)
		if profile_on:
			profile['feature_ms'] += profile_elapsed_ms(feature_start, device, config)
		profile['feature_chunks'] = feature_workspace.feature_chunks

		if profile_on:
			policy_start = profile_timer_start(device, config)
		with torch.inference_mode(), make_autocast_context(device, precision):
			if sample_actions:
				action_logits, values_all = forward_model(
					model,
					features_all,
					mode="both",
					chunk_agents=forward_chunk_agents,
				)
				dist_all = torch.distributions.Categorical(logits=action_logits)
				actions_all = dist_all.sample()
				log_probs_all = dist_all.log_prob(actions_all).to(old_log_probs.dtype)
				actions = torch.where(alive_mask, actions_all, actions)
				old_log_probs = torch.where(alive_mask, log_probs_all, old_log_probs)
			else:
				values_all = forward_model(
					model,
					features_all,
					mode="value",
					chunk_agents=forward_chunk_agents,
				)
		if values_all.dim() == 3 and values_all.shape[-1] == 1:
			values_all = values_all.squeeze(-1)
		value_pred = torch.where(alive_mask, values_all.to(value_pred.dtype), value_pred)
		if profile_on:
			profile['policy_ms'] += profile_elapsed_ms(policy_start, device, config)
		del features_all
		return actions, old_log_probs, value_pred, profile

	selected_chunk = feature_build_chunk_agents(config, M)
	if selected_chunk <= 0:
		selected_chunk = total_selected
	profile['path'] = 'selected'

	route_state_all = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
	control_state_all = current_control_state(simulator)
	profile_on = profile_enabled(config)

	for start in range(0, total_selected, selected_chunk):
		end = min(start + selected_chunk, total_selected)
		env_chunk = env_idx[start:end]
		agent_chunk = agent_idx[start:end]

		if profile_on:
			feature_start = profile_timer_start(device, config)
		obs_state_chunk = states[env_chunk].clone()
		obs_state_chunk[..., 6] = alive_mask[env_chunk].to(dtype=obs_state_chunk.dtype)
		control_chunk = control_state_all[env_chunk]
		world_state_chunk = torch.cat([obs_state_chunk, control_chunk], dim=-1)
		route_chunk = gather_current_route_state_selected(route_state_all, env_chunk, agent_chunk)
		condition_chunk = gather_condition_state_selected(condition_state, env_chunk, agent_chunk)
		features_chunk = build_features_for_selected_agents(
			world_state_chunk,
			simulator,
			config,
			route_chunk,
			condition_chunk,
			agent_chunk,
			time_indices=torch.as_tensor(dropout_step, device=device, dtype=torch.long),
			env_indices=env_chunk,
			workspace=feature_workspace,
		)
		if profile_on:
			profile['feature_ms'] += profile_elapsed_ms(feature_start, device, config)
		profile['feature_chunks'] = feature_workspace.feature_chunks

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
				logits_selected = action_logits[:, 0]
				dist_selected = torch.distributions.Categorical(logits=logits_selected)
				actions_chunk = dist_selected.sample()
				actions[env_chunk, agent_chunk] = actions_chunk
				old_log_probs[env_chunk, agent_chunk] = dist_selected.log_prob(actions_chunk).to(old_log_probs.dtype)
			else:
				values_chunk = forward_model(
					model,
					features_chunk,
					mode="value",
					chunk_agents=forward_chunk_agents,
				)
		if values_chunk.dim() == 3 and values_chunk.shape[-1] == 1:
			values_chunk = values_chunk.squeeze(-1)
		if values_chunk.dim() == 2:
			values_selected = values_chunk[:, 0]
		else:
			values_selected = values_chunk.reshape(-1)
		value_pred[env_chunk, agent_chunk] = values_selected.to(value_pred.dtype)
		if profile_on:
			profile['policy_ms'] += profile_elapsed_ms(policy_start, device, config)

		del features_chunk, world_state_chunk, obs_state_chunk

	return actions, old_log_probs, value_pred, profile


def bootstrap_values_for_alive_agents(model, simulator, config, alive_mask: torch.Tensor,
									  condition_state: dict, dropout_step: int,
									  precision: str, forward_chunk_agents: int,
									  feature_workspace: FeatureBuildWorkspace = None):
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
	)
	return value_pred


def current_rollout_bootstrap_value(model, simulator, config, cumulative_done_all,
									condition_state: dict, dropout_step: int,
									precision: str, forward_chunk_agents: int,
									feature_workspace: FeatureBuildWorkspace = None):
	alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
	if not bool(alive_mask.any().item()):
		return None
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
	)

# ============================== PPO更新函数 ==============================
def perform_ppo_update(model, policy_optimizer, value_optimizer,
					   rollout_buffer, condition_state,
					   features_tensor, simulator, config, iteration, rank=None,
					   a_max_ewma=None, amp_scaler=None, bootstrap_value=None,
					   feature_workspace: FeatureBuildWorkspace = None):
	"""执行 PPO 更新。buffer 保存世界状态/route state，条件特征在 minibatch 内重建。"""
	is_rank0 = (rank is None or rank == 0)
	update_start_time = time.time()
	if len(rollout_buffer) == 0:
		device = features_tensor.device if isinstance(features_tensor, torch.Tensor) else torch.device('cpu')
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
	max_grad_norm = float(getattr(training_cfg, 'max_grad_norm', 1.0))
	batch_size_per_gpu = int(getattr(training_cfg, 'batch_size_per_gpu', 2000))
	advantage_filter_threshold = float(getattr(training_cfg, 'advantage_filter_threshold', 0.01))
	beta = float(getattr(training_cfg, 'advantage_filter_beta', 0.25))
	precision = getattr(training_cfg, 'precision', '32-bit')
	forward_chunk_agents = int(getattr(training_cfg, 'network_forward_chunk_agents', 32768))
	feature_workspace = feature_workspace or _default_workspace(config)
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
	elif features_tensor is None:
		last_value_pred = torch.zeros_like(values_tensor[-1])
	else:
		with torch.inference_mode(), make_autocast_context(device, precision):
			_, last_value_pred = forward_model(model, features_tensor, mode="both", chunk_agents=forward_chunk_agents)
		if last_value_pred.dim() == 3 and last_value_pred.shape[-1] == 1:
			last_value_pred = last_value_pred.squeeze(-1)
	values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)

	dones_accum = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	advantages, returns = gae_advantages(rewards_tensor, values_tp1, dones_accum, gamma, gae_lambda)

	A_max_tensor = torch.max(torch.abs(advantages)).detach()
	a_max_ewma = A_max_tensor if a_max_ewma is None else (beta * A_max_tensor + (1.0 - beta) * a_max_ewma.to(device))
	eta = advantage_filter_threshold * a_max_ewma
	keep_mask = (torch.abs(advantages) >= eta)

	seen_done_inclusive = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	seen_done_prev = torch.roll(seen_done_inclusive, shifts=1, dims=0)
	seen_done_prev[0] = False
	first_done_step = dones_tensor & (~seen_done_prev)
	post_done_mask = seen_done_inclusive & (~first_done_step)
	keep_mask = keep_mask & (~post_done_mask)
	active_sample_mask = states_tensor[..., 6] > 0.5
	keep_mask = keep_mask & active_sample_mask
	cand_idx = keep_mask.nonzero(as_tuple=False)

	if is_rank0:
		print(f"🎯 第 {iteration} 个iteration - 最大|A|: {A_max_tensor.item():.4f}, 阈值: {eta.item():.4f}")
		print(f"📊 过滤前: {keep_mask.numel()}, 过滤后: {keep_mask.sum().item()}")

	local_has_samples = cand_idx.numel() > 0
	if not sync_bool_across_ranks(local_has_samples, device, op=dist.ReduceOp.MIN):
		if is_rank0:
			print("⚠️ 至少一个rank无可用样本，本轮跳过以避免DDP不同步")
		stats = make_update_stats("no_samples_after_filter", device)
		stats['num_candidates'] = int(cand_idx.shape[0])
		stats['ppo_update_time_s'] = time.time() - update_start_time
		return a_max_ewma.detach(), stats

	N = cand_idx.shape[0]
	K_target = batch_size_per_gpu if batch_size_per_gpu > 0 else N
	K = min(K_target, N)
	rand_pos = torch.randperm(N, device=device)[:K]
	selected_idx = cand_idx[rand_pos]
	selected_t = selected_idx[:, 0]
	selected_b = selected_idx[:, 1]
	selected_m = selected_idx[:, 2]
	if is_rank0:
		print(f"🎯 随机选取 {K} 个样本用于更新（候选 {N}, 目标 {K_target}）")

	old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
	returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
	actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1).to(torch.long)
	advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std(unbiased=False) + 1e-8)
	batch_N = old_log_probs_batch.shape[0]

	route_state_mb = gather_route_state_selected(route_state_tensor, selected_t, selected_b, selected_m)
	condition_state_mb = gather_condition_state_selected(condition_state, selected_b, selected_m)
	world_states_mb = states_tensor[selected_t, selected_b]
	ppo_feature_rebuild_ms = 0.0
	profile_on = profile_enabled(config)
	if profile_on:
		feature_rebuild_start = profile_timer_start(device, config)
	mb_features = build_features_for_selected_agents(
		world_states_mb,
		simulator,
		config,
		route_state_mb,
		condition_state_mb,
		selected_m,
		time_indices=time_indices_tensor[selected_t],
		env_indices=selected_b,
		workspace=feature_workspace,
	)
	if profile_on:
		ppo_feature_rebuild_ms = profile_elapsed_ms(feature_rebuild_start, device, config)

	policy_params = get_policy_parameters(model)
	value_params = get_value_parameters(model)
	model.train()
	did_optimizer_step = False
	last_policy_loss = None
	last_value_loss = None
	last_entropy = None
	mb_old_logp = old_log_probs_batch
	mb_adv = advantages_batch
	mb_ret = returns_batch
	mb_actions = actions_batch
	for epoch in range(ppo_epochs):
		policy_optimizer.zero_grad(set_to_none=True)
		value_optimizer.zero_grad(set_to_none=True)
		with make_autocast_context(device, precision):
			action_logits, value_pred_full = forward_model(model, mb_features, mode="both", chunk_agents=forward_chunk_agents)
			logits_selected = action_logits[:, 0]
			dist_selected = torch.distributions.Categorical(logits=logits_selected)
			new_log_probs = dist_selected.log_prob(mb_actions)
			ratio = torch.exp(new_log_probs - mb_old_logp)
			surr1 = ratio * mb_adv
			surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
			policy_loss = -torch.min(surr1, surr2).mean()
			entropy = dist_selected.entropy().mean()
			value_pred = value_pred_full[:, 0]
			value_loss = (value_pred - mb_ret).pow(2).mean()
			total_loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

		if amp_scaler is not None and getattr(amp_scaler, 'is_enabled', lambda: False)():
			old_scale = amp_scaler.get_scale()
			amp_scaler.scale(total_loss).backward()
			amp_scaler.unscale_(policy_optimizer)
			amp_scaler.unscale_(value_optimizer)
			torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
			torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
			amp_scaler.step(policy_optimizer)
			amp_scaler.step(value_optimizer)
			amp_scaler.update()
			did_optimizer_step = did_optimizer_step or (amp_scaler.get_scale() >= old_scale)
		else:
			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
			torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
			policy_optimizer.step()
			value_optimizer.step()
			did_optimizer_step = True

		last_policy_loss = float(policy_loss.detach().item())
		last_value_loss = float(value_loss.detach().item())
		last_entropy = float(entropy.detach().item())

		if is_rank0:
			print(f"   Epoch {epoch+1}/{ppo_epochs}: Policy Loss: {last_policy_loss:.6f}, Value Loss: {last_value_loss:.6f}, Entropy: {last_entropy:.6f}")

	model.eval()
	if is_rank0:
		print(f"✅ 第 {iteration} 个iteration - 经验采样训练完成")
	stats = make_update_stats("", device)
	stats.update({
		'did_optimizer_step': did_optimizer_step,
		'num_candidates': int(N),
		'num_selected': int(K),
		'num_epochs': int(ppo_epochs),
		'policy_loss': last_policy_loss,
		'value_loss': last_value_loss,
		'entropy': last_entropy,
		'ppo_update_time_s': time.time() - update_start_time,
		'ppo_feature_rebuild_ms': ppo_feature_rebuild_ms,
	})
	stats.update(cuda_memory_stats(device))
	if is_rank0 and device.type == 'cuda':
		print(
			f"🧠 PPO峰值显存: allocated={stats['max_memory_allocated_mb']:.1f}MB, "
			f"reserved={stats['max_memory_reserved_mb']:.1f}MB"
		)
	return a_max_ewma.detach(), stats

def perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer,
								 rollout_buffer, condition_state,
								 features_tensor, simulator, config, iteration, a_max_ewma=None,
								 amp_scaler=None, bootstrap_value=None,
								 feature_workspace: FeatureBuildWorkspace = None):
	return perform_ppo_update(
		model, policy_optimizer, value_optimizer,
		rollout_buffer, condition_state,
		features_tensor, simulator, config, iteration,
		rank=None, a_max_ewma=a_max_ewma, amp_scaler=amp_scaler,
		bootstrap_value=bootstrap_value,
		feature_workspace=feature_workspace,
	)

def perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer,
								rollout_buffer, condition_state,
								features_tensor, simulator, config, iteration, rank, a_max_ewma=None,
								amp_scaler=None, bootstrap_value=None,
								feature_workspace: FeatureBuildWorkspace = None):
	return perform_ppo_update(
		model, policy_optimizer, value_optimizer,
		rollout_buffer, condition_state,
		features_tensor, simulator, config, iteration,
		rank=rank, a_max_ewma=a_max_ewma, amp_scaler=amp_scaler,
		bootstrap_value=bootstrap_value,
		feature_workspace=feature_workspace,
	)
		
# ============================== 寻找空闲端口 ==============================
def _find_free_port() -> int:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("127.0.0.1", 0))
	addr, port = s.getsockname()
	s.close()
	return port

# ============================== 设置DDP环境 ==============================
def setup_ddp_env(rank: int, gpu_count: int, master_addr: str, master_port: int):
	os.environ['MASTER_ADDR'] = master_addr
	os.environ['MASTER_PORT'] = str(master_port)
	os.environ['gpu_count'] = str(gpu_count)
	os.environ['RANK'] = str(rank)
	# Windows/本地优先gloo网卡
	if os.name == 'nt':
		# 不设置 lo，避免 Windows 找不到接口
		os.environ['GLOO_DEVICE_TRANSPORT'] = 'tcp'
		os.environ.pop('GLOO_SOCKET_IFNAME', None)

# ============================== 清理DDP环境 ==============================
def cleanup_ddp():
	if dist.is_initialized():
		dist.destroy_process_group()

# ============================== DDPPO训练 ==============================
def ddppo_worker(rank: int, gpu_count: int, config_dict: dict, master_addr: str, master_port: int, store_port: int):
	for stream in (sys.stdout, sys.stderr):
		try:
			stream.reconfigure(line_buffering=True, write_through=True)
		except Exception:
			pass

	def worker_log(message: str):
		print(f"[Rank {rank}] {message}", flush=True)

	if gpu_count == 1:
		#TODO:这里写单卡训练代码，用于调试
		worker_log("worker start: single-gpu path")
		device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		torch.cuda.set_device(device) if device.type == 'cuda' else None
		config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
		worker_log("creating network")
		model = create_network(config=config, network_type="independent")
		model = model.to(device)
		worker_log("network ready")
		worker_log("creating simulator")
		simulator = TeraflowSimulator(config=config_dict, device=device)
		worker_log("simulator ready")

		sim_cfg = getattr(config, 'simulator')
		training_cfg = getattr(config, 'training')
		learning_rate = getattr(training_cfg, 'learning_rate')
		num_iterations = getattr(training_cfg, 'iteration')
		max_episode_length = getattr(training_cfg,'max_episode_length')
		ppo_epochs = getattr(training_cfg, 'ppo_epochs')
		gamma = getattr(training_cfg, 'gamma')
		gae_lambda = getattr(training_cfg, 'gae_lambda')
		clip_ratio = getattr(training_cfg, 'clip_ratio')
		entropy_coef = getattr(training_cfg, 'entropy_coef')
		value_loss_coef = getattr(training_cfg, 'value_loss_coef')
		max_grad_norm = getattr(training_cfg, 'max_grad_norm')
		checkpoint_interval = getattr(training_cfg, 'checkpoint_interval')
		checkpoint_dir = getattr(training_cfg, 'checkpoint_dir')
		log_interval = getattr(training_cfg, 'log_interval', 10)
		
		# 分别创建策略网络和价值网络的优化器
		policy_optimizer = optim.Adam(get_policy_parameters(model), lr=learning_rate)
		value_optimizer = optim.Adam(get_value_parameters(model), lr=learning_rate)

		# 分别创建策略网络和价值网络的调度器
		policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_iterations, eta_min=0.0)
		value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_iterations, eta_min=0.0)
		resume_from = getattr(training_cfg, 'resume_from', None)
		start_iteration = 0
		if resume_from:
			start_iteration = load_checkpoint(model, policy_optimizer, value_optimizer, resume_from, device)
			advance_scheduler_to_iteration(policy_scheduler, value_scheduler, start_iteration)
			print(f"✅ 从 checkpoint 恢复: {resume_from}, start_iteration={start_iteration}")

		# 优势过滤参数
		beta = getattr(training_cfg, 'advantage_filter_beta', 0.25)	# EWMA衰减参数
		advantage_filter_threshold = getattr(training_cfg, 'advantage_filter_threshold', 0.01)	# 优势过滤阈值
		A_max_ewma = None 		# EWMA of max absolute advantage
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 2000)  # 每GPU的batch size
		rollout_length = getattr(training_cfg, 'rollout_length', 128)  # rollout长度
		precision = getattr(training_cfg, 'precision', '32-bit')
		forward_chunk_agents = int(getattr(training_cfg, 'network_forward_chunk_agents', 32768))
		amp_scaler = make_grad_scaler(device, precision)
		profile_on = profile_enabled(config)
		feature_workspace = FeatureBuildWorkspace(config)
		
		for k in range(start_iteration, num_iterations):
			print(f"🔄 开始第 {k+1}/{num_iterations} 轮迭代")

			episode_start_time = time.time()
			# ============================== 采样（初始化） ==============================
			if profile_on:
				reset_profile_start = profile_timer_start(device, config)
			worker_log(f"iteration {k+1}: reset start")
			simulator.reset(return_observation=False)
			worker_log(f"iteration {k+1}: reset done")
			if profile_on:
				reset_ms = profile_elapsed_ms(reset_profile_start, device, config)
			condition_state = snapshot_condition_state(simulator)
			features_tensor = None
			worker_log(f"iteration {k+1}: rollout feature build deferred to alive-agent chunks")
			if profile_on:
				print(f"\t⏱️ reset={reset_ms:.2f}ms, initial_feature_build=0.00ms")
			
			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = simulator.agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			rollout_buffer = RolloutTensorBuffer(rollout_length)
			buffer_step_count = 0
			iteration_update_stats = make_update_stats("no_update", device)
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
				if not bool(alive_mask.any().item()):
					if buffer_step_count > 0:
						print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							rollout_buffer, condition_state,
							None, simulator, config, k+1, A_max_ewma, amp_scaler,
							bootstrap_value=None,
							feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(rollout_buffer)
					else:
						print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				debug_step = step_count < 3
				if debug_step:
					worker_log(f"step {step_count + 1}: policy start")
				actions, old_log_probs, value_pred, rollout_profile = rollout_forward_alive_agents(
					model,
					simulator,
					config,
					alive_mask,
					condition_state,
					dropout_step=step_count,
					precision=precision,
					forward_chunk_agents=forward_chunk_agents,
					sample_actions=True,
					feature_workspace=feature_workspace,
				)
				policy_forward_ms = rollout_profile['policy_ms']
				feature_build_ms = rollout_profile['feature_ms']
				if debug_step:
					worker_log(f"step {step_count + 1}: policy done")
				
				# 在推进环境前缓存当前状态
				pre_route_state = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
				rollout_buffer.write_pre_step_from_simulator(simulator, alive_mask, pre_route_state, time_index=step_count)
				
				# 环境步进
				if profile_on:
					env_profile_start = profile_timer_start(device, config)
				if debug_step:
					worker_log(f"step {step_count + 1}: env start")
				reward, done = simulator.step(actions, return_observation=False)
				if debug_step:
					worker_log(f"step {step_count + 1}: env done")
				if profile_on:
					env_step_ms = profile_elapsed_ms(env_profile_start, device, config)
				
				# 写入训练buffer（与game.py一致）
				rollout_buffer.write_post_step(reward, done, value_pred, old_log_probs, actions)
				buffer_step_count = len(rollout_buffer)
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				no_alive_after_step = not bool(rollout_alive_mask(simulator, cumulative_done_all).any().item())
				
				# 下一步 feature 不再整批预构造；下个循环会按 alive agent chunk 即时生成。
				features_tensor = None
				
				step_count += 1
				if step_count % log_interval == 0:
					print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				if profile_on and step_count % profile_log_interval(config) == 0:
					step_profile = format_profile(getattr(simulator, 'last_step_profile', {}))
					rollout_path = rollout_profile.get('path', 'none')
					rollout_selected = int(rollout_profile.get('num_selected', 0) or 0)
					rollout_chunks = int(rollout_profile.get('feature_chunks', 0) or 0)
					print(f"\t⏱️ profile step={step_count}: policy={policy_forward_ms:.2f}ms, env={env_step_ms:.2f}ms, feature={feature_build_ms:.2f}ms"
						  + f", path={rollout_path}, selected={rollout_selected}, feature_chunks={rollout_chunks}"
						  + (f", {step_profile}" if step_profile else ""))

				if no_alive_after_step:
					print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
					A_max_ewma, update_stats = perform_ppo_update_single_gpu(
						model, policy_optimizer, value_optimizer,
						rollout_buffer, condition_state,
						None, simulator, config, k+1, A_max_ewma, amp_scaler,
						bootstrap_value=None,
						feature_workspace=feature_workspace)
					merge_update_stats(iteration_update_stats, update_stats)
					clear_rollout_buffers(rollout_buffer)
					break
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						bootstrap_value = current_rollout_bootstrap_value(
							model, simulator, config, cumulative_done_all,
							condition_state, step_count, precision, forward_chunk_agents,
							feature_workspace=feature_workspace)
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							rollout_buffer, condition_state,
							None, simulator, config, k+1, A_max_ewma, amp_scaler,
							bootstrap_value=bootstrap_value,
							feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(rollout_buffer)
						print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						bootstrap_value = current_rollout_bootstrap_value(
							model, simulator, config, cumulative_done_all,
							condition_state, step_count, precision, forward_chunk_agents,
							feature_workspace=feature_workspace)
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							rollout_buffer, condition_state,
							None, simulator, config, k+1, A_max_ewma, amp_scaler,
							bootstrap_value=bootstrap_value,
							feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)

						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						if all_worlds_no_alive_agents(simulator, cumulative_done_all):
							print("🔄 所有世界都没有存活agents，开启新iteration...")
							clear_rollout_buffers(rollout_buffer)
							break
						else:
							print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							clear_rollout_buffers(rollout_buffer)
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性


			# 只有真实完成 optimizer step 后才推进学习率，避免空样本或 AMP skip 时跳过首个LR。
			scheduler_stepped = step_schedulers_if_updated(policy_scheduler, value_scheduler, iteration_update_stats)
			if profile_on:
				last_update = iteration_update_stats.get('last_update', {})
				print(f"\t⏱️ update profile: scheduler_step={scheduler_stepped}, "
					  f"samples={iteration_update_stats.get('num_selected', 0)}, "
					  f"ppo={iteration_update_stats.get('ppo_update_time_s', 0.0):.3f}s, "
					  f"ppo_feature_rebuild={iteration_update_stats.get('ppo_feature_rebuild_ms', 0.0):.2f}ms, "
					  f"mem_alloc={iteration_update_stats.get('max_memory_allocated_mb', 0.0):.1f}MB, "
					  f"skip={last_update.get('skip_reason', '')}")
			# 保存检查点
			if (k + 1) % checkpoint_interval == 0:
				save_checkpoint(model, policy_optimizer, value_optimizer, k + 1, checkpoint_dir)
			print(f"🎯 本轮总步数耗时: {time.time()-episode_start_time:.4f}秒")

		print('train done!')
		return 0
	
	try:
		worker_log("worker start: ddp path")
		device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		torch.cuda.set_device(device) if device.type == 'cuda' else None
		# 调试打印：确认设备映射
		print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
		print(f"[Rank {rank}] device={device}")
		print(f"[Rank {rank}] torch.cuda.current_device()={torch.cuda.current_device()}, name={torch.cuda.get_device_name(torch.cuda.current_device())}")
		print(f"[Rank {rank}] torch.cuda.device_count()={torch.cuda.device_count()}")
		# 设置环境变量
		setup_ddp_env(rank, gpu_count, master_addr, master_port)
		# TCPStore: rank0为主节点
		is_master = (rank == 0)
		
		store = dist.TCPStore(master_addr, store_port, gpu_count, is_master, timeout=timedelta(seconds=180))

		# 使用store初始化进程组（按原文示例）
		backend = 'nccl' if (os.name != 'nt' and torch.cuda.is_available()) else 'gloo'
		dist.init_process_group(backend=backend, world_size=gpu_count, rank=rank, store=store, timeout=timedelta(seconds=180))
		# 用PrefixStore追踪完成数量
		num_workers_done = dist.PrefixStore("num_workers_done", store)

		# 载入配置并建模
		config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
		worker_log("creating network")
		model = create_network(config=config, network_type="independent")
		model = model.to(device)
		worker_log("network ready")
		
		# 按原文示例的DDP签名（等价于传入本地rank）
		if device.type == 'cuda':
			model = DDP(model, device_ids=[rank], output_device=rank)
		else:
			model = DDP(model)

		# ==== 与单卡保持一致的模拟器与超参数初始化 ====
		worker_log("creating simulator")
		simulator = TeraflowSimulator(config=config_dict, device=device)
		worker_log("simulator ready")
		sim_cfg = getattr(config, 'simulator', SimpleNamespace())
		training_cfg = getattr(config, 'training', SimpleNamespace())
		learning_rate = getattr(training_cfg, 'learning_rate', 3e-4)
		num_iterations = getattr(training_cfg, 'iteration')
		max_episode_length = getattr(training_cfg, 'max_episode_length', 1024)
		ppo_epochs = getattr(training_cfg, 'ppo_epochs', 2)
		gamma = getattr(training_cfg, 'gamma', 0.999)
		gae_lambda = getattr(training_cfg, 'gae_lambda', 0.95)
		clip_ratio = getattr(training_cfg, 'clip_ratio', 0.2)
		entropy_coef = getattr(training_cfg, 'entropy_coef', 0.01)
		value_loss_coef = getattr(training_cfg, 'value_loss_coef', 0.5)
		max_grad_norm = getattr(training_cfg, 'max_grad_norm', 1.0)
		checkpoint_interval = getattr(training_cfg, 'checkpoint_interval', 1)
		checkpoint_dir = getattr(training_cfg, 'checkpoint_dir')
		log_interval = getattr(training_cfg, 'log_interval', 10)
		# 优势过滤参数
		beta = getattr(training_cfg, 'advantage_filter_beta', 0.25)
		advantage_filter_threshold = getattr(training_cfg, 'advantage_filter_threshold', 0.01)
		A_max_ewma = None
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 2000)
		rollout_length = getattr(training_cfg, 'rollout_length', 128)

		precision = getattr(training_cfg, 'precision', '32-bit')
		forward_chunk_agents = int(getattr(training_cfg, 'network_forward_chunk_agents', 32768))
		amp_scaler = make_grad_scaler(device, precision)
		profile_on = profile_enabled(config)
		feature_workspace = FeatureBuildWorkspace(config)

		# 分别创建策略网络和价值网络的优化器，包含 encoder + head
		policy_optimizer = optim.Adam(get_policy_parameters(model), lr=learning_rate)
		value_optimizer = optim.Adam(get_value_parameters(model), lr=learning_rate)
		policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_iterations, eta_min=0.0)
		value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_iterations, eta_min=0.0)
		resume_from = getattr(training_cfg, 'resume_from', None)
		start_iteration = 0
		if resume_from:
			start_iteration = load_checkpoint(model, policy_optimizer, value_optimizer, resume_from, device)
			advance_scheduler_to_iteration(policy_scheduler, value_scheduler, start_iteration)
			if rank == 0:
				print(f"✅ 从 checkpoint 恢复: {resume_from}, start_iteration={start_iteration}")
		
		# 每一轮迭代（步进式训练：与game.py完全一致）
		for k in range(start_iteration, num_iterations):
			# 2) 本轮开始：重置完成计数（保持与原多卡同步逻辑一致）
			num_workers_done.set("done", b"0")

			# 采样初始化
			if profile_on:
				reset_profile_start = profile_timer_start(device, config)
			worker_log(f"iteration {k+1}: reset start")
			simulator.reset(return_observation=False)
			worker_log(f"iteration {k+1}: reset done")
			if profile_on:
				reset_ms = profile_elapsed_ms(reset_profile_start, device, config)
			condition_state = snapshot_condition_state(simulator)
			features_tensor = None
			worker_log(f"iteration {k+1}: rollout feature build deferred to alive-agent chunks")
			if rank == 0 and profile_on:
				print(f"\t⏱️ reset={reset_ms:.2f}ms, initial_feature_build=0.00ms")

			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = simulator.agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			rollout_buffer = RolloutTensorBuffer(rollout_length)
			buffer_step_count = 0
			iteration_update_stats = make_update_stats("no_update", device)
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
				local_no_alive = not bool(alive_mask.any().item())
				if sync_bool_across_ranks(local_no_alive, device, op=dist.ReduceOp.MIN):
					if buffer_step_count > 0:
						if rank == 0:
							print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
								rollout_buffer, condition_state,
								None, simulator, config, k+1, rank, A_max_ewma, amp_scaler,
								bootstrap_value=None,
								feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(rollout_buffer)
					else:
						if rank == 0:
							print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				debug_step = step_count < 3
				if debug_step:
					worker_log(f"step {step_count + 1}: policy start")
				actions, old_log_probs, value_pred, rollout_profile = rollout_forward_alive_agents(
					model,
					simulator,
					config,
					alive_mask,
					condition_state,
					dropout_step=step_count,
					precision=precision,
					forward_chunk_agents=forward_chunk_agents,
					sample_actions=True,
					feature_workspace=feature_workspace,
				)
				policy_forward_ms = rollout_profile['policy_ms']
				feature_build_ms = rollout_profile['feature_ms']
				if debug_step:
					worker_log(f"step {step_count + 1}: policy done")
				
				# 在推进环境前缓存当前状态
				pre_route_state = simulator.get_route_state(clone=False) if hasattr(simulator, 'get_route_state') else {}
				rollout_buffer.write_pre_step_from_simulator(simulator, alive_mask, pre_route_state, time_index=step_count)
				
				# 环境步进
				if profile_on:
					env_profile_start = profile_timer_start(device, config)
				if debug_step:
					worker_log(f"step {step_count + 1}: env start")
				reward, done = simulator.step(actions, return_observation=False)
				if debug_step:
					worker_log(f"step {step_count + 1}: env done")
				if profile_on:
					env_step_ms = profile_elapsed_ms(env_profile_start, device, config)
				
				# 写入训练buffer（与game.py一致）
				rollout_buffer.write_post_step(reward, done, value_pred, old_log_probs, actions)
				buffer_step_count = len(rollout_buffer)
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				local_no_alive_after_step = not bool(rollout_alive_mask(simulator, cumulative_done_all).any().item())
				no_alive_after_step = sync_bool_across_ranks(local_no_alive_after_step, device, op=dist.ReduceOp.MIN)
				
				# 下一步 feature 不再整批预构造；下个循环会按 alive agent chunk 即时生成。
				features_tensor = None
				
				step_count += 1
				if rank == 0 and step_count % log_interval == 0:
					print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				if rank == 0 and profile_on and step_count % profile_log_interval(config) == 0:
					step_profile = format_profile(getattr(simulator, 'last_step_profile', {}))
					rollout_path = rollout_profile.get('path', 'none')
					rollout_selected = int(rollout_profile.get('num_selected', 0) or 0)
					rollout_chunks = int(rollout_profile.get('feature_chunks', 0) or 0)
					print(f"\t⏱️ profile step={step_count}: policy={policy_forward_ms:.2f}ms, env={env_step_ms:.2f}ms, feature={feature_build_ms:.2f}ms"
						  + f", path={rollout_path}, selected={rollout_selected}, feature_chunks={rollout_chunks}"
						  + (f", {step_profile}" if step_profile else ""))

				if no_alive_after_step:
					if rank == 0:
						print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
					A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
						model, policy_optimizer, value_optimizer,
						rollout_buffer, condition_state,
						None, simulator, config, k+1, rank, A_max_ewma, amp_scaler,
						bootstrap_value=None,
						feature_workspace=feature_workspace)
					merge_update_stats(iteration_update_stats, update_stats)
					clear_rollout_buffers(rollout_buffer)
					break
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						bootstrap_value = current_rollout_bootstrap_value(
							model, simulator, config, cumulative_done_all,
							condition_state, step_count, precision, forward_chunk_agents,
							feature_workspace=feature_workspace)
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
							rollout_buffer, condition_state,
							None, simulator, config, k+1, rank, A_max_ewma, amp_scaler,
							bootstrap_value=bootstrap_value,
							feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(rollout_buffer)
						if rank == 0:
							print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						bootstrap_value = current_rollout_bootstrap_value(
							model, simulator, config, cumulative_done_all,
							condition_state, step_count, precision, forward_chunk_agents,
							feature_workspace=feature_workspace)
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
							rollout_buffer, condition_state,
							None, simulator, config, k+1, rank, A_max_ewma, amp_scaler,
							bootstrap_value=bootstrap_value,
							feature_workspace=feature_workspace)
						merge_update_stats(iteration_update_stats, update_stats)

						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						local_no_alive = all_worlds_no_alive_agents(simulator, cumulative_done_all)
						if sync_bool_across_ranks(local_no_alive, device, op=dist.ReduceOp.MIN):
							if rank == 0:
								print("🔄 所有世界都没有存活agents，开启新iteration...")
							clear_rollout_buffers(rollout_buffer)
							break
						else:
							if rank == 0:
								print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							clear_rollout_buffers(rollout_buffer)
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性

			# 7) 只有真实完成 optimizer step 后才推进学习率，避免空样本或 AMP skip 时跳过首个LR。
			scheduler_stepped = step_schedulers_if_updated(policy_scheduler, value_scheduler, iteration_update_stats)
			if rank == 0 and profile_on:
				last_update = iteration_update_stats.get('last_update', {})
				print(f"\t⏱️ update profile: scheduler_step={scheduler_stepped}, "
					  f"samples={iteration_update_stats.get('num_selected', 0)}, "
					  f"ppo={iteration_update_stats.get('ppo_update_time_s', 0.0):.3f}s, "
					  f"ppo_feature_rebuild={iteration_update_stats.get('ppo_feature_rebuild_ms', 0.0):.2f}ms, "
					  f"mem_alloc={iteration_update_stats.get('max_memory_allocated_mb', 0.0):.1f}MB, "
					  f"skip={last_update.get('skip_reason', '')}")

			# 标记本worker完成（保留原计数器结构）
			try:
				if hasattr(num_workers_done, 'add'):
					num_workers_done.add("done", 1)
				else:
					curr = int(num_workers_done.get("done").decode())
					num_workers_done.set("done", str(curr + 1).encode())
			except Exception:
				pass

			# 仅在主进程保存检查点并打印轮次进度
			if is_master:
				try:
					num_done = int(num_workers_done.get("done").decode())
				except Exception:
					num_done = -1
				print(f"[Round {k}] finished={num_done}/{gpu_count}")
				if (k + 1) % checkpoint_interval == 0:
					try:
						save_checkpoint(model, policy_optimizer, value_optimizer, k + 1, checkpoint_dir)
					except Exception:
						pass
					
	except Exception as e:
		print(f"[Rank {rank}] 训练异常: {e}")
	finally:
		cleanup_ddp()

# ============================== 运行分布式DDPPO ==============================
def run_distributed_ddppo(config_dict: dict, cuda_ranks: list[int]):
	if not cuda_ranks:
		raise RuntimeError("没有可用的CUDA设备")
	
	gpu_count = len(cuda_ranks)
	master_addr = '127.0.0.1'
	master_port = _find_free_port()
	store_port = _find_free_port()

	# Windows需要spawn
	mp.set_start_method('spawn', force=True)

	# 将rank映射到CUDA设备
	os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(r) for r in cuda_ranks)
	ctx = mp.get_context('spawn')

	processes = []
	for rank in range(gpu_count):
		p = ctx.Process(target=ddppo_worker, args=(rank, gpu_count, config_dict, master_addr, master_port, store_port))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()

if __name__ == "__main__":
	# 读取配置并运行一个简化示例
	import yaml
	# 静默检测GPU
	ok, ranks = check_gpu_info(print_info=False)
	print(f"CUDA可用: {ok}, Ranks: {ranks}")
	if not ok or not ranks:
		raise SystemExit("无可用GPU，退出")
	# 默认使用全部可用卡
	# 基于文件位置解析项目根目录，避免依赖当前工作目录
	_this_dir = os.path.dirname(os.path.abspath(__file__))
	_proj_root = os.path.dirname(_this_dir)
	_config_path = os.path.join(_proj_root, 'configs', 'default_config.yaml')
	with open(_config_path, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	run_distributed_ddppo(cfg, ranks)

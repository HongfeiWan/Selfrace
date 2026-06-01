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

def pad_or_truncate_flat(flat: torch.Tensor, target_size: int, pad_value: float = 0.0) -> torch.Tensor:
    B, M, D = flat.shape
    out = torch.full((B, M, target_size), pad_value, device=flat.device, dtype=flat.dtype)
    copy_size = min(D, target_size)
    if copy_size > 0:
        out[:, :, :copy_size] = flat[:, :, :copy_size]
    return out

def normalize_point_set(points: torch.Tensor, target_size: int, element_dim: int = 2,
                        min_val: float = -100.0, max_val: float = 100.0) -> torch.Tensor:
    if points is None or points.numel() == 0:
        return None
    if points.dim() == 3:
        B, M, N = points.shape
        elements = points.view(B, M, N // element_dim, element_dim)
    else:
        elements = points
    valid = torch.isfinite(elements).all(dim=-1) & (elements.abs().sum(dim=-1) > 1e-6)
    normalized = normalize_to_minus1_1(torch.nan_to_num(elements, nan=0.0, posinf=max_val, neginf=min_val), min_val, max_val)
    normalized = torch.where(valid.unsqueeze(-1), normalized, torch.full_like(normalized, FEATURE_PAD_VALUE))
    return pad_or_truncate_flat(normalized.flatten(start_dim=2), target_size, FEATURE_PAD_VALUE)

def normalize_s_features(s_t: torch.Tensor, target_size: int, vehicle_style: torch.Tensor = None,
                         control_state: torch.Tensor = None) -> torch.Tensor:
    """原文式 S(t): c,theta,kappa,v,v_lim,phi,a_long,a_lat,Cacc,Cthrottle,Csteer,l,w。"""
    B, M, _ = s_t.shape
    out = torch.zeros(B, M, target_size, device=s_t.device, dtype=s_t.dtype)
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
        normalized = torch.empty_like(out)
        bounds = (
            (-5.0, 5.0), (-math.pi, math.pi), (-0.2, 0.2), (-2.0, 30.0), (0.0, 30.0),
            (-0.7, 0.7), (-5.0, 5.0), (-4.0, 4.0),
            (1 / 1.5, 1.5), (1 / 1.25, 1.25), (1 / 1.25, 1.25),
            (0.8, 7.0), (0.8, 3.0),
        )
        for i in range(target_size):
            if i < len(bounds):
                normalized[:, :, i] = normalize_to_minus1_1(out[:, :, i], *bounds[i])
            else:
                normalized[:, :, i] = out[:, :, i]
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

def build_lane_map_features(w_lanes_local: torch.Tensor, path_plan: torch.Tensor,
                            target_size: int, element_dim: int = 7) -> torch.Tensor:
    """构造原文式 W_lane: 位置、车道方向、车道宽度、到下一目标的绝对/相对距离。"""
    if w_lanes_local is None or w_lanes_local.numel() == 0:
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

    if path_plan is not None and path_plan.numel() > 0:
        path = path_plan.to(device=lanes.device, dtype=lanes.dtype)
        path_valid = torch.isfinite(path).all(dim=-1) & ~((path[..., 0] == -1.0) & (path[..., 1] == -1.0))
        L = path.shape[2]
        idx = torch.arange(L, device=lanes.device).view(1, 1, L)
        last_idx = torch.where(path_valid, idx, torch.zeros_like(idx)).amax(dim=2)
        has_goal = path_valid.any(dim=2)
        b_idx = torch.arange(B, device=lanes.device).view(B, 1).expand(B, M)
        m_idx = torch.arange(M, device=lanes.device).view(1, M).expand(B, M)
        goal_local = path[b_idx, m_idx, last_idx]
        goal_local = torch.where(has_goal.unsqueeze(-1), goal_local, torch.zeros_like(goal_local))
    else:
        goal_local = torch.zeros(B, M, 2, device=lanes.device, dtype=lanes.dtype)
        has_goal = torch.zeros(B, M, dtype=torch.bool, device=lanes.device)

    goal_dist = torch.norm(lane_xy - goal_local.unsqueeze(2), dim=-1)
    goal_dist = torch.where(valid & has_goal.unsqueeze(-1), goal_dist, torch.zeros_like(goal_dist))
    masked_goal_dist = goal_dist.masked_fill(~(valid & has_goal.unsqueeze(-1)), float('inf'))
    min_goal_dist = masked_goal_dist.amin(dim=2)
    min_goal_dist = torch.where(torch.isfinite(min_goal_dist), min_goal_dist, torch.zeros_like(min_goal_dist))
    rel_goal_dist = goal_dist - min_goal_dist.unsqueeze(-1)

    out = torch.full((B, M, K, element_dim), FEATURE_PAD_VALUE, device=lanes.device, dtype=lanes.dtype)
    if element_dim > 0:
        out[..., 0] = normalize_to_minus1_1(lane_xy[..., 0], -200, 200)
    if element_dim > 1:
        out[..., 1] = normalize_to_minus1_1(lane_xy[..., 1], -200, 200)
    if element_dim > 2:
        dir_end = min(element_dim, 4)
        out[..., 2:dir_end] = lane_dir[..., :dir_end - 2]
    if element_dim > 4:
        out[..., 4] = normalize_to_minus1_1(lane_width, 0.0, 8.0)
    if element_dim > 5:
        out[..., 5] = normalize_to_minus1_1(goal_dist, 0.0, 400.0)
    if element_dim > 6:
        out[..., 6] = normalize_to_minus1_1(rel_goal_dist, 0.0, 200.0)
    out = torch.where(valid.unsqueeze(-1), out, torch.full_like(out, FEATURE_PAD_VALUE))
    return pad_or_truncate_flat(out.flatten(start_dim=2), target_size, FEATURE_PAD_VALUE)

def build_network_features(agents_state: torch.Tensor, 
                          neighbors_local: torch.Tensor, 
                          w_lanes_local: torch.Tensor, 
                          w_boundaries_local: torch.Tensor,
                          path_plan: torch.Tensor,
                          stop_lines: torch.Tensor,
                          reward_coef: torch.Tensor,
                          config: SimpleNamespace,
                          vehicle_style: torch.Tensor = None,
                          control_state: torch.Tensor = None) -> torch.Tensor:
    """
    将拆解后的观测组件构建为网络输入的特征张量
    Args:
        agents_state: (B, M, S_dim) - 原文式 S(t) 局部状态
        neighbors_local: (B, M, K, neighbor_dim) - 邻居相对状态，active 位于最后一维
        w_lanes_local: (B, M, N_lanes, lane_dim) - map lane raw feature
        w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标
        path_plan: (B, M, path_length, 2) - 路径规划点
        stop_lines: (B, M, num_stop_lines, 20) - 停止线点
        reward_coef: (B, M, 10) - 奖励系数
        config: 配置对象
    Returns:
        torch.Tensor: 形状为 (B, M, total_input_dim) 的网络输入特征张量
    """
    batch_size, max_agents, _ = agents_state.shape
    
    # 从配置中获取网络需要的特征维度
    network_config = config.training.network
    simple_feature_dims = network_config.simple_feature_dims
    permutation_feature_dims = network_config.permutation_feature_dims
    permutation_element_dims = getattr(network_config, 'permutation_element_dims', [2, 7, 2, 7])
    
    # 计算总输入维度
    total_input_dim = sum(simple_feature_dims) + sum(permutation_feature_dims)
    
    # 初始化输出张量
    features_tensor = torch.zeros(batch_size, max_agents, total_input_dim, device=agents_state.device, dtype=agents_state.dtype)
    
    # 1. 构建简单特征。新配置不再包含单独的 G(t) dense path vector；
    #    routing 信息只通过 W_lane 的 goal-distance 特征进入网络。
    simple_end = sum(simple_feature_dims)
    has_dense_goal_vector = len(simple_feature_dims) >= 4
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
    )
    simple_offset += 1
    feature_cursor = s_t_end
    
    if has_dense_goal_vector:
        g_t_size = simple_feature_dims[simple_offset]
        g_t_start = feature_cursor
        g_t_end = g_t_start + g_t_size
        if path_plan is None:
            path_plan_stable = torch.zeros(batch_size, max_agents, g_t_size, device=agents_state.device, dtype=agents_state.dtype)
        else:
            path_plan_stable = path_plan.to(device=agents_state.device, dtype=agents_state.dtype).flatten(start_dim=2)
            path_plan_stable = pad_or_truncate_flat(path_plan_stable, g_t_size, pad_value=0.0)
        path_plan_stable = normalize_to_minus1_1(path_plan_stable, -200, 200)
        features_tensor[:, :, g_t_start:g_t_end] = path_plan_stable
        simple_offset += 1
        feature_cursor = g_t_end

    # reward系数: 10维 - 使用传入的采样参数
    reward_coef_size = simple_feature_dims[simple_offset]
    reward_coef_start = feature_cursor
    reward_coef_end = reward_coef_start + reward_coef_size

    reward_coef = reward_coef.to(device=agents_state.device, dtype=agents_state.dtype)
    reward_coef_stable = torch.zeros(batch_size, max_agents, reward_coef_size, device=agents_state.device, dtype=agents_state.dtype)
    reward_bounds = (
        (2, 12), (0, 3), (0, 3), (0, 0.1), (0.00025, 0.025),
        (0, 1), (0.00025, 0.0075), (-0.5, 0.5), (0.00025, 0.0075), (0, 1),
    )
    copy_reward = min(reward_coef.shape[-1], reward_coef_size, len(reward_bounds))
    for i in range(copy_reward):
        reward_coef_stable[:, :, i] = normalize_to_minus1_1(reward_coef[:, :, i], *reward_bounds[i])
    features_tensor[:, :, reward_coef_start:reward_coef_end] = reward_coef_stable
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
    vehicle_style_stable = torch.zeros(batch_size, max_agents, vehicle_style_size, device=agents_state.device, dtype=agents_state.dtype)
    style_bounds = ((1 / 1.25, 1.25), (1 / 1.25, 1.25), (1 / 1.5, 1.5), (1 / 1.5, 1.5))
    copy_style = min(vehicle_style.shape[-1], vehicle_style_size, len(style_bounds))
    for i in range(copy_style):
        vehicle_style_stable[:, :, i] = normalize_to_minus1_1(vehicle_style[:, :, i], *style_bounds[i])
    features_tensor[:, :, vehicle_style_start:vehicle_style_end] = vehicle_style_stable
    
    # 2. 构建排列不变特征 (road_boundary, lane_points, stop_lines, other_agents)
    permutation_start = simple_end
    
    # road_boundary: 原文使用最近80个boundary coarse features
    road_boundary_size = permutation_feature_dims[0]
    road_boundary_start = permutation_start
    road_boundary_end = road_boundary_start + road_boundary_size
    
    w_boundaries_flat = normalize_point_set(w_boundaries_local, road_boundary_size, min_val=-200.0, max_val=200.0)
    if w_boundaries_flat is not None:
        features_tensor[:, :, road_boundary_start:road_boundary_end] = w_boundaries_flat
    else:
        features_tensor[:, :, road_boundary_start:road_boundary_end] = FEATURE_PAD_VALUE
    
    # lane_points: 原文式 map lane feature，每个元素包含位置、方向、宽度、目标距离。
    lane_points_size = permutation_feature_dims[1]
    lane_points_start = road_boundary_end
    lane_points_end = lane_points_start + lane_points_size
    
    lane_element_dim = permutation_element_dims[1] if len(permutation_element_dims) > 1 else 7
    w_lanes_flat = build_lane_map_features(w_lanes_local, path_plan, lane_points_size, lane_element_dim)
    if w_lanes_flat is not None:
        features_tensor[:, :, lane_points_start:lane_points_end] = w_lanes_flat
    else:
        features_tensor[:, :, lane_points_start:lane_points_end] = FEATURE_PAD_VALUE
    
    # stop_lines: 20维 - 使用停止线信息
    stop_lines_size = permutation_feature_dims[2]  # 20
    stop_lines_start = lane_points_end
    stop_lines_end = stop_lines_start + stop_lines_size
    
    if stop_lines is not None and stop_lines.numel() > 0:
        stop_lines_flat = normalize_point_set(stop_lines.to(device=agents_state.device, dtype=agents_state.dtype), stop_lines_size)
        if stop_lines_flat is not None:
            features_tensor[:, :, stop_lines_start:stop_lines_end] = stop_lines_flat
        else:
            features_tensor[:, :, stop_lines_start:stop_lines_end] = FEATURE_PAD_VALUE
    else:
        features_tensor[:, :, stop_lines_start:stop_lines_end] = FEATURE_PAD_VALUE
    
    # other_agents: 使用邻居位置、朝向、速度、尺寸、z 与 active mask
    other_agents_size = permutation_feature_dims[3]
    other_agents_start = stop_lines_end
    other_agents_end = other_agents_start + other_agents_size
    
    # 将邻居信息按通道做归一化后再展平并填充，active=0 的 padding 不参与网络 maxpool。
    neighbors_local = neighbors_local.to(device=agents_state.device, dtype=agents_state.dtype)
    neighbors_proc = torch.full_like(neighbors_local, FEATURE_PAD_VALUE)
    neighbor_dim = neighbors_local.shape[-1]
    if neighbor_dim >= 10:
        neighbors_proc[:, :, :, 0] = normalize_to_minus1_1(neighbors_local[:, :, :, 0], -200, 200)
        neighbors_proc[:, :, :, 1] = normalize_to_minus1_1(neighbors_local[:, :, :, 1], -200, 200)
        neighbors_proc[:, :, :, 2] = torch.clamp(neighbors_local[:, :, :, 2], -1.0, 1.0)
        neighbors_proc[:, :, :, 3] = torch.clamp(neighbors_local[:, :, :, 3], -1.0, 1.0)
        neighbors_proc[:, :, :, 4] = normalize_to_minus1_1(neighbors_local[:, :, :, 4], -60, 60)
        neighbors_proc[:, :, :, 5] = normalize_to_minus1_1(neighbors_local[:, :, :, 5], -60, 60)
        neighbors_proc[:, :, :, 6] = normalize_to_minus1_1(neighbors_local[:, :, :, 6], 0.8, 7)
        neighbors_proc[:, :, :, 7] = normalize_to_minus1_1(neighbors_local[:, :, :, 7], 0.8, 3)
        neighbors_proc[:, :, :, 8] = normalize_to_minus1_1(neighbors_local[:, :, :, 8], -10, 10)
        neighbors_proc[:, :, :, 9] = neighbors_local[:, :, :, 9]
    else:
        neighbors_proc[:, :, :, 0] = normalize_to_minus1_1(neighbors_local[:, :, :, 0], -100, 100)
        neighbors_proc[:, :, :, 1] = normalize_to_minus1_1(neighbors_local[:, :, :, 1], -100, 100)
        if neighbor_dim > 2:
            neighbors_proc[:, :, :, 2] = normalize_to_minus1_1(neighbors_local[:, :, :, 2], -60, 60)
        if neighbor_dim > 3:
            neighbors_proc[:, :, :, 3] = normalize_to_minus1_1(neighbors_local[:, :, :, 3], -60, 60)
        if neighbor_dim > 4:
            neighbors_proc[:, :, :, 4] = normalize_to_minus1_1(neighbors_local[:, :, :, 4], 0.8, 7)
        if neighbor_dim > 5:
            neighbors_proc[:, :, :, 5] = normalize_to_minus1_1(neighbors_local[:, :, :, 5], 0.8, 3)
        if neighbor_dim > 6:
            neighbors_proc[:, :, :, 6] = neighbors_local[:, :, :, 6]
    neighbors_flat = neighbors_proc.flatten(start_dim=2)
    features_tensor[:, :, other_agents_start:other_agents_end] = pad_or_truncate_flat(
        neighbors_flat,
        other_agents_size,
        pad_value=FEATURE_PAD_VALUE,
    )
    
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

def forward_model(model, features_tensor, mode="both"):
	return model(features_tensor, mode=mode)

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

def current_path_plan(simulator):
	"""返回用于 W_lane routing distance 的当前目标点，形状 (B, M, 1, 2)，局部坐标。"""
	goal_positions = getattr(simulator, 'goal_positions', None)
	agents_state = getattr(simulator, 'agents_state', None)
	if goal_positions is not None and agents_state is not None:
		ego_pos = agents_state[..., :2]
		ego_yaw = agents_state[..., 2]
		cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
		rot_matrix = torch.stack([
			torch.stack([cos_yaw, -sin_yaw], dim=-1),
			torch.stack([sin_yaw, cos_yaw], dim=-1)
		], dim=-2)
		B, M, _ = agents_state.shape
		rel_goal = (goal_positions.to(device=agents_state.device, dtype=agents_state.dtype) - ego_pos).view(B * M, 1, 2)
		goal_local = torch.bmm(rel_goal, rot_matrix.view(B * M, 2, 2)).view(B, M, 1, 2)
		active = agents_state[..., 6] > 0.5
		return torch.where(active.unsqueeze(-1).unsqueeze(-1), goal_local, torch.full_like(goal_local, -1.0))
	return simulator.agents_path_plans_local if getattr(simulator, 'agents_path_plans_local', None) is not None else simulator.agents_path_plans

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

def state_with_control_for_buffer(simulator, alive_mask: torch.Tensor) -> torch.Tensor:
	pre_state = simulator.agents_state.detach().clone()
	pre_state[..., 6] = alive_mask.to(dtype=pre_state.dtype)
	pre_control_state = current_control_state(simulator).detach().clone()
	pre_control_state = pre_control_state * alive_mask.unsqueeze(-1).to(dtype=pre_control_state.dtype)
	return torch.cat([pre_state, pre_control_state], dim=-1)

def build_features_from_observation(observation, simulator, config):
	agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
	return build_network_features(
		agents_state,
		neighbors_local,
		w_lanes_local,
		w_boundaries_local,
		current_path_plan(simulator),
		getattr(simulator, 'stop_lines', None),
		simulator.reward_calculator.sampled_params,
		config,
		vehicle_style=getattr(simulator, 'driving_style_params', None),
		control_state=current_control_state(simulator),
	)

def sync_bool_across_ranks(value: bool, device: torch.device, op=dist.ReduceOp.MIN) -> bool:
	if not dist.is_available() or not dist.is_initialized():
		return value
	t = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
	dist.all_reduce(t, op=op)
	return bool(t.item())

def validate_rollout_buffers(states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							 rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer):
	T = len(states_buffer)
	buffer_lengths = [
		len(path_plan_buffer), len(reward_coef_buffer), len(vehicle_style_buffer), len(stop_lines_buffer),
		len(rewards_buffer), len(dones_buffer), len(values_buffer), len(old_log_probs_buffer), len(actions_buffer)
	]
	if any(length != T for length in buffer_lengths):
		raise ValueError(f"Rollout buffer length mismatch: states={T}, others={buffer_lengths}")
	if T == 0:
		return
	base_shape = states_buffer[0].shape
	base_device = states_buffer[0].device
	if len(base_shape) != 3 or base_shape[-1] < 7:
		raise ValueError(f"states_buffer expected (T,B,M,S>=7), got {base_shape}")
	B, M, _ = base_shape
	for name, buf in (
		('states', states_buffer),
		('path_plan', path_plan_buffer),
		('reward_coef', reward_coef_buffer),
		('vehicle_style', vehicle_style_buffer),
		('stop_lines', stop_lines_buffer),
	):
		for i, tensor in enumerate(buf):
			if tensor.device != base_device:
				raise ValueError(f"{name}[{i}] device mismatch: {tensor.device} != {base_device}")
			if tensor.shape[0] != B or tensor.shape[1] != M:
				raise ValueError(f"{name}[{i}] leading shape mismatch: {tensor.shape[:2]} != {(B, M)}")
	for name, buf in (
		('rewards', rewards_buffer),
		('dones', dones_buffer),
		('values', values_buffer),
		('old_log_probs', old_log_probs_buffer),
		('actions', actions_buffer),
	):
		for i, tensor in enumerate(buf):
			if tensor.device != base_device:
				raise ValueError(f"{name}[{i}] device mismatch: {tensor.device} != {base_device}")
			if tensor.shape[:2] != (B, M):
				raise ValueError(f"{name}[{i}] shape mismatch: {tensor.shape[:2]} != {(B, M)}")

def merge_update_stats(accum: dict, update_stats: dict):
	if update_stats is None:
		return accum
	accum['did_optimizer_step'] = accum.get('did_optimizer_step', False) or update_stats.get('did_optimizer_step', False)
	accum['num_candidates'] = accum.get('num_candidates', 0) + int(update_stats.get('num_candidates', 0) or 0)
	accum['num_selected'] = accum.get('num_selected', 0) + int(update_stats.get('num_selected', 0) or 0)
	accum['ppo_update_time_s'] = accum.get('ppo_update_time_s', 0.0) + float(update_stats.get('ppo_update_time_s', 0.0) or 0.0)
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

def sample_actions_for_alive(action_logits: torch.Tensor, alive_mask: torch.Tensor):
	"""只为仍存活的 agent 采样动作，已 done 的 slot 填 dummy 动作与零 logprob。"""
	actions = torch.zeros(action_logits.shape[:2], dtype=torch.long, device=action_logits.device)
	old_log_probs = torch.zeros(action_logits.shape[:2], dtype=action_logits.dtype, device=action_logits.device)
	if bool(alive_mask.any().item()):
		dist_alive = torch.distributions.Categorical(logits=action_logits[alive_mask])
		actions_alive = dist_alive.sample()
		actions[alive_mask] = actions_alive
		old_log_probs[alive_mask] = dist_alive.log_prob(actions_alive).to(old_log_probs.dtype)
	return actions, old_log_probs

# ============================== PPO更新函数 ==============================
def perform_ppo_update(model, policy_optimizer, value_optimizer,
					   states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
					   rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
					   features_tensor, simulator, config, iteration, rank=None, a_max_ewma=None, amp_scaler=None):
	"""执行 PPO 更新。buffer 中保存真实世界状态和对应时刻的条件特征。"""
	is_rank0 = (rank is None or rank == 0)
	update_start_time = time.time()
	if len(states_buffer) == 0:
		device = features_tensor.device if isinstance(features_tensor, torch.Tensor) else torch.device('cpu')
		if is_rank0:
			print("⚠️ Buffer为空，无法进行PPO更新")
		return a_max_ewma, make_update_stats("empty_buffer", device)
	if is_rank0:
		print(f"🎯 开始经验采样训练，Buffer长度: {len(states_buffer)}")

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
	device = states_buffer[0].device
	validate_rollout_buffers(
		states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
		rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
	)
	if device.type == 'cuda':
		torch.cuda.reset_peak_memory_stats(device)

	states_tensor = torch.stack(states_buffer, dim=0)  # (T, B, M, 7)
	path_plan_tensor = torch.stack(path_plan_buffer, dim=0) if path_plan_buffer else None
	reward_coef_tensor = torch.stack(reward_coef_buffer, dim=0) if reward_coef_buffer else None
	vehicle_style_tensor = torch.stack(vehicle_style_buffer, dim=0) if vehicle_style_buffer else None
	stop_lines_tensor = torch.stack(stop_lines_buffer, dim=0) if stop_lines_buffer else None
	rewards_tensor = torch.stack(rewards_buffer, dim=0)
	dones_tensor = torch.stack(dones_buffer, dim=0).bool()
	values_tensor = torch.stack(values_buffer, dim=0)
	old_log_probs_tensor = torch.stack(old_log_probs_buffer, dim=0)
	actions_tensor = torch.stack(actions_buffer, dim=0)

	if features_tensor is None:
		last_value_pred = torch.zeros_like(values_tensor[-1])
	else:
		with torch.inference_mode(), make_autocast_context(device, precision):
			_, last_value_pred = forward_model(model, features_tensor, mode="both")
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
	K = batch_size_per_gpu if batch_size_per_gpu > 0 else N
	if N >= K:
		rand_pos = torch.randperm(N, device=device)[:K]
	else:
		rand_pos = torch.randint(0, N, (K,), device=device)
	selected_idx = cand_idx[rand_pos]
	selected_t = selected_idx[:, 0]
	selected_b = selected_idx[:, 1]
	selected_m = selected_idx[:, 2]
	if is_rank0:
		print(f"🎯 随机选取 {K} 个样本用于更新（候选 {N}）")

	agent_indices_batch = selected_m.to(device)
	old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
	returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
	actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std(unbiased=False) + 1e-8)
	batch_N = old_log_probs_batch.shape[0]

	uniq_tb, inverse_mb = torch.unique(torch.stack([selected_t, selected_b], dim=1), dim=0, return_inverse=True)
	t_u_mb = uniq_tb[:, 0]
	b_u_mb = uniq_tb[:, 1]
	path_plan_mb = path_plan_tensor[t_u_mb, b_u_mb] if path_plan_tensor is not None else None
	stop_lines_mb = stop_lines_tensor[t_u_mb, b_u_mb] if stop_lines_tensor is not None else None
	reward_coef_mb = reward_coef_tensor[t_u_mb, b_u_mb] if reward_coef_tensor is not None else simulator.reward_calculator.sampled_params[b_u_mb]
	if vehicle_style_tensor is not None:
		vehicle_style_mb = vehicle_style_tensor[t_u_mb, b_u_mb]
	else:
		current_style = getattr(simulator, 'driving_style_params', None)
		vehicle_style_mb = current_style[b_u_mb] if current_style is not None else None
	world_states_mb = states_tensor[t_u_mb, b_u_mb]
	control_state_mb = control_from_buffer_state(world_states_mb)
	obs_mb = simulator.observation_generator.generate(
		observation_state_from_buffer(world_states_mb),
		control_state=control_state_mb,
		driving_style_params=vehicle_style_mb,
	)
	agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, config)
	features_u_mb = build_network_features(
		agents_state_dec_mb,
		neighbors_local_mb,
		w_lanes_local_mb,
		w_boundaries_local_mb,
		path_plan_mb,
		stop_lines_mb,
		reward_coef_mb,
		config,
		vehicle_style=vehicle_style_mb,
		control_state=control_state_mb,
	)
	mb_features = features_u_mb[inverse_mb.to(device), agent_indices_batch].unsqueeze(1)

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
			action_logits, value_pred_full = forward_model(model, mb_features, mode="both")
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
	})
	stats.update(cuda_memory_stats(device))
	return a_max_ewma.detach(), stats

def perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer,
								 states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								 rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
								 features_tensor, simulator, config, iteration, a_max_ewma=None, amp_scaler=None):
	return perform_ppo_update(
		model, policy_optimizer, value_optimizer,
		states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
		rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
		features_tensor, simulator, config, iteration,
		rank=None, a_max_ewma=a_max_ewma, amp_scaler=amp_scaler,
	)

def perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer,
								states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
								features_tensor, simulator, config, iteration, rank, a_max_ewma=None, amp_scaler=None):
	return perform_ppo_update(
		model, policy_optimizer, value_optimizer,
		states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
		rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
		features_tensor, simulator, config, iteration,
		rank=rank, a_max_ewma=a_max_ewma, amp_scaler=amp_scaler,
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
	if gpu_count == 1:
		#TODO:这里写单卡训练代码，用于调试
		device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		torch.cuda.set_device(device) if device.type == 'cuda' else None
		config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
		model = create_network(config=config, network_type="independent")
		model = model.to(device)
		simulator = TeraflowSimulator(config=config_dict, device=device)

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
		amp_scaler = make_grad_scaler(device, precision)
		profile_on = profile_enabled(config)
		
		for k in range(start_iteration, num_iterations):
			print(f"🔄 开始第 {k+1}/{num_iterations} 轮迭代")

			episode_start_time = time.time()
			# ============================== 采样（初始化） ==============================
			if profile_on:
				reset_profile_start = profile_timer_start(device, config)
			initial_observation = simulator.reset()
			if profile_on:
				reset_ms = profile_elapsed_ms(reset_profile_start, device, config)
				feature_profile_start = profile_timer_start(device, config)
			features_tensor = build_features_from_observation(initial_observation, simulator, config)
			if profile_on:
				initial_feature_ms = profile_elapsed_ms(feature_profile_start, device, config)
				print(f"\t⏱️ reset={reset_ms:.2f}ms, initial_feature_build={initial_feature_ms:.2f}ms")
			
			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = simulator.agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			states_buffer = []
			path_plan_buffer = []
			reward_coef_buffer = []
			vehicle_style_buffer = []
			stop_lines_buffer = []
			rewards_buffer = []
			dones_buffer = []
			values_buffer = []
			old_log_probs_buffer = []
			actions_buffer = []
			buffer_step_count = 0
			iteration_update_stats = make_update_stats("no_update", device)
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				if all_worlds_no_alive_agents(simulator, cumulative_done_all):
					if buffer_step_count > 0:
						print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						)
					else:
						print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				if profile_on:
					policy_profile_start = profile_timer_start(device, config)
				with torch.inference_mode(), make_autocast_context(device, precision):
					action_logits, value_pred = forward_model(model, features_tensor, mode="both")
					if value_pred.dim() == 3 and value_pred.shape[-1] == 1:
						value_pred = value_pred.squeeze(-1)
					value_pred = torch.where(alive_mask, value_pred, torch.zeros_like(value_pred))
					actions, old_log_probs = sample_actions_for_alive(action_logits, alive_mask)
				if profile_on:
					policy_forward_ms = profile_elapsed_ms(policy_profile_start, device, config)
				del action_logits
				
				# 在推进环境前缓存当前状态
				pre_state = state_with_control_for_buffer(simulator, alive_mask)
				pre_path_plan = current_path_plan(simulator).detach().clone()
				pre_reward_coef = simulator.reward_calculator.sampled_params.detach().clone()
				pre_vehicle_style = simulator.driving_style_params.detach().clone()
				pre_stop_lines = simulator.stop_lines.detach().clone()
				
				# 环境步进
				if profile_on:
					env_profile_start = profile_timer_start(device, config)
				observation, reward, done = simulator.step(actions)
				if profile_on:
					env_step_ms = profile_elapsed_ms(env_profile_start, device, config)
				
				# 写入训练buffer（与game.py一致）
				states_buffer.append(pre_state)
				path_plan_buffer.append(pre_path_plan)
				reward_coef_buffer.append(pre_reward_coef)
				vehicle_style_buffer.append(pre_vehicle_style)
				stop_lines_buffer.append(pre_stop_lines)
				rewards_buffer.append(reward.detach().clone())
				dones_buffer.append(done.detach().clone())
				values_buffer.append(value_pred.detach().clone())
				old_log_probs_buffer.append(old_log_probs.detach().clone())
				actions_buffer.append(actions.detach().clone())
				buffer_step_count += 1
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				no_alive_after_step = all_worlds_no_alive_agents(simulator, cumulative_done_all)
				
				# 更新观测与特征
				if no_alive_after_step:
					features_tensor = None
					feature_build_ms = 0.0
				else:
					if profile_on:
						feature_profile_start = profile_timer_start(device, config)
					features_tensor = build_features_from_observation(observation, simulator, config)
					if profile_on:
						feature_build_ms = profile_elapsed_ms(feature_profile_start, device, config)
				
				step_count += 1
				if step_count % log_interval == 0:
					print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				if profile_on and step_count % profile_log_interval(config) == 0:
					step_profile = format_profile(getattr(simulator, 'last_step_profile', {}))
					print(f"\t⏱️ profile step={step_count}: policy={policy_forward_ms:.2f}ms, env={env_step_ms:.2f}ms, feature={feature_build_ms:.2f}ms"
						  + (f", {step_profile}" if step_profile else ""))

				if no_alive_after_step:
					print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
					A_max_ewma, update_stats = perform_ppo_update_single_gpu(
						model, policy_optimizer, value_optimizer,
						states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
						rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						features_tensor, simulator, config, k+1, A_max_ewma, amp_scaler)
					merge_update_stats(iteration_update_stats, update_stats)
					clear_rollout_buffers(
						states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
						rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
					)
					break
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						)
						print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						A_max_ewma, update_stats = perform_ppo_update_single_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						
						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						if all_worlds_no_alive_agents(simulator, cumulative_done_all):
							print("🔄 所有世界都没有存活agents，开启新iteration...")
							clear_rollout_buffers(
								states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							)
							break
						else:
							print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							clear_rollout_buffers(
								states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							)
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性


			# 只有真实完成 optimizer step 后才推进学习率，避免空样本或 AMP skip 时跳过首个LR。
			scheduler_stepped = step_schedulers_if_updated(policy_scheduler, value_scheduler, iteration_update_stats)
			if profile_on:
				last_update = iteration_update_stats.get('last_update', {})
				print(f"\t⏱️ update profile: scheduler_step={scheduler_stepped}, "
					  f"samples={iteration_update_stats.get('num_selected', 0)}, "
					  f"ppo={iteration_update_stats.get('ppo_update_time_s', 0.0):.3f}s, "
					  f"mem_alloc={iteration_update_stats.get('max_memory_allocated_mb', 0.0):.1f}MB, "
					  f"skip={last_update.get('skip_reason', '')}")
			# 保存检查点
			if (k + 1) % checkpoint_interval == 0:
				save_checkpoint(model, policy_optimizer, value_optimizer, k + 1, checkpoint_dir)
			print(f"🎯 本轮总步数耗时: {time.time()-episode_start_time:.4f}秒")

		print('train done!')
		return 0
	
	try:
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
		model = create_network(config=config, network_type="independent")
		model = model.to(device)
		
		# 按原文示例的DDP签名（等价于传入本地rank）
		if device.type == 'cuda':
			model = DDP(model, device_ids=[rank], output_device=rank)
		else:
			model = DDP(model)

		# ==== 与单卡保持一致的模拟器与超参数初始化 ====
		simulator = TeraflowSimulator(config=config_dict, device=device)
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
		amp_scaler = make_grad_scaler(device, precision)
		profile_on = profile_enabled(config)

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
			initial_observation = simulator.reset()
			if profile_on:
				reset_ms = profile_elapsed_ms(reset_profile_start, device, config)
				feature_profile_start = profile_timer_start(device, config)
			features_tensor = build_features_from_observation(initial_observation, simulator, config)
			if profile_on:
				initial_feature_ms = profile_elapsed_ms(feature_profile_start, device, config)
			if rank == 0 and profile_on:
				print(f"\t⏱️ reset={reset_ms:.2f}ms, initial_feature_build={initial_feature_ms:.2f}ms")

			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = simulator.agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			states_buffer = []
			path_plan_buffer = []
			reward_coef_buffer = []
			vehicle_style_buffer = []
			stop_lines_buffer = []
			rewards_buffer = []
			dones_buffer = []
			values_buffer = []
			old_log_probs_buffer = []
			actions_buffer = []
			buffer_step_count = 0
			iteration_update_stats = make_update_stats("no_update", device)
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				local_no_alive = all_worlds_no_alive_agents(simulator, cumulative_done_all)
				if sync_bool_across_ranks(local_no_alive, device, op=dist.ReduceOp.MIN):
					if buffer_step_count > 0:
						if rank == 0:
							print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, rank, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						)
					else:
						if rank == 0:
							print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				alive_mask = rollout_alive_mask(simulator, cumulative_done_all)
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				if profile_on:
					policy_profile_start = profile_timer_start(device, config)
				with torch.inference_mode(), make_autocast_context(device, precision):
					action_logits, value_pred = forward_model(model, features_tensor, mode="both")
					if value_pred.dim() == 3 and value_pred.shape[-1] == 1:
						value_pred = value_pred.squeeze(-1)
					value_pred = torch.where(alive_mask, value_pred, torch.zeros_like(value_pred))
					actions, old_log_probs = sample_actions_for_alive(action_logits, alive_mask)
				if profile_on:
					policy_forward_ms = profile_elapsed_ms(policy_profile_start, device, config)
				del action_logits
				
				# 在推进环境前缓存当前状态
				pre_state = state_with_control_for_buffer(simulator, alive_mask)
				pre_path_plan = current_path_plan(simulator).detach().clone()
				pre_reward_coef = simulator.reward_calculator.sampled_params.detach().clone()
				pre_vehicle_style = simulator.driving_style_params.detach().clone()
				pre_stop_lines = simulator.stop_lines.detach().clone()
				
				# 环境步进
				if profile_on:
					env_profile_start = profile_timer_start(device, config)
				observation, reward, done = simulator.step(actions)
				if profile_on:
					env_step_ms = profile_elapsed_ms(env_profile_start, device, config)
				
				# 写入训练buffer（与game.py一致）
				states_buffer.append(pre_state)
				path_plan_buffer.append(pre_path_plan)
				reward_coef_buffer.append(pre_reward_coef)
				vehicle_style_buffer.append(pre_vehicle_style)
				stop_lines_buffer.append(pre_stop_lines)
				rewards_buffer.append(reward.detach().clone())
				dones_buffer.append(done.detach().clone())
				values_buffer.append(value_pred.detach().clone())
				old_log_probs_buffer.append(old_log_probs.detach().clone())
				actions_buffer.append(actions.detach().clone())
				buffer_step_count += 1
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				local_no_alive_after_step = all_worlds_no_alive_agents(simulator, cumulative_done_all)
				no_alive_after_step = sync_bool_across_ranks(local_no_alive_after_step, device, op=dist.ReduceOp.MIN)
				
				# 更新观测与特征
				if no_alive_after_step:
					features_tensor = None
					feature_build_ms = 0.0
				else:
					if profile_on:
						feature_profile_start = profile_timer_start(device, config)
					features_tensor = build_features_from_observation(observation, simulator, config)
					if profile_on:
						feature_build_ms = profile_elapsed_ms(feature_profile_start, device, config)
				
				step_count += 1
				if rank == 0 and step_count % log_interval == 0:
					print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				if rank == 0 and profile_on and step_count % profile_log_interval(config) == 0:
					step_profile = format_profile(getattr(simulator, 'last_step_profile', {}))
					print(f"\t⏱️ profile step={step_count}: policy={policy_forward_ms:.2f}ms, env={env_step_ms:.2f}ms, feature={feature_build_ms:.2f}ms"
						  + (f", {step_profile}" if step_profile else ""))

				if no_alive_after_step:
					if rank == 0:
						print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
					A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
						model, policy_optimizer, value_optimizer,
						states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
						rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						features_tensor, simulator, config, k+1, rank, A_max_ewma, amp_scaler)
					merge_update_stats(iteration_update_stats, update_stats)
					clear_rollout_buffers(
						states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
						rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
					)
					break
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, rank, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						clear_rollout_buffers(
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
						)
						if rank == 0:
							print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						A_max_ewma, update_stats = perform_ppo_update_multi_gpu(
							model, policy_optimizer, value_optimizer,
							states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
							rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							features_tensor, simulator, config, k+1, rank, A_max_ewma, amp_scaler)
						merge_update_stats(iteration_update_stats, update_stats)
						
						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						local_no_alive = all_worlds_no_alive_agents(simulator, cumulative_done_all)
						if sync_bool_across_ranks(local_no_alive, device, op=dist.ReduceOp.MIN):
							if rank == 0:
								print("🔄 所有世界都没有存活agents，开启新iteration...")
							clear_rollout_buffers(
								states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							)
							break
						else:
							if rank == 0:
								print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							clear_rollout_buffers(
								states_buffer, path_plan_buffer, reward_coef_buffer, vehicle_style_buffer, stop_lines_buffer,
								rewards_buffer, dones_buffer, values_buffer, old_log_probs_buffer, actions_buffer,
							)
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性

			# 7) 只有真实完成 optimizer step 后才推进学习率，避免空样本或 AMP skip 时跳过首个LR。
			scheduler_stepped = step_schedulers_if_updated(policy_scheduler, value_scheduler, iteration_update_stats)
			if rank == 0 and profile_on:
				last_update = iteration_update_stats.get('last_update', {})
				print(f"\t⏱️ update profile: scheduler_step={scheduler_stepped}, "
					  f"samples={iteration_update_stats.get('num_selected', 0)}, "
					  f"ppo={iteration_update_stats.get('ppo_update_time_s', 0.0):.3f}s, "
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

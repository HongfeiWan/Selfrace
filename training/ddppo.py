import os
import sys
import json
import socket
import math
from datetime import timedelta
from types import SimpleNamespace
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
			'policy_state_dict': save_model.policy_network.state_dict(),
			'value_state_dict': save_model.value_network.state_dict(),
			'policy_optim_state_dict': policy_optimizer.state_dict(),
			'value_optim_state_dict': value_optimizer.state_dict(),
		}
		ckpt_path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pt')
		torch.save(state, ckpt_path)
	except Exception as e:
		print(f"⚠️ 保存检查点失败: {e}")

# ============================== 观测数据拆解 ==============================
def decompose_observation(observation: torch.Tensor, config: SimpleNamespace) -> tuple:
    """
    将initial_observation拆解为网络需要的各个组件
    
    Args:
        observation: 形状为 (B, M, total_obs_dim) 的观测张量
        config: 配置对象
    
    Returns:
        tuple: (agents_state, neighbors_local, w_lanes_local, w_boundaries_local)
            - agents_state: (B, M, 7) - 智能体状态 [x, y, yaw, speed, length, width, active]
            - neighbors_local: (B, M, K, 7) - 邻居相对状态 [dx, dy, vx, vy, length, width, active]
            - w_lanes_local: (B, M, N_lanes, 2) - 车道线相对坐标 [dx, dy]
            - w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标 [dx, dy]
    """
    batch_size, max_agents, total_obs_dim = observation.shape
    
    # 从配置中获取维度信息
    simulator_config = config.simulator
    local_state_dim = simulator_config.observation.local_state_dim  # 7
    neighbor_feature_dim = simulator_config.observation.neighbor_feature_dim  # 7
    waypoint_feature_dim = simulator_config.observation.waypoint_feature_dim  # 2
    boundary_feature_dim = simulator_config.observation.boundary_feature_dim  # 2
    num_neighbors = simulator_config.observation.num_neighbors  # 20
    num_w_lanes = simulator_config.observation.num_w_lanes  # 25
    num_w_boundaries = simulator_config.observation.num_w_boundaries  # 26
    
    # 计算各部分在观测向量中的位置
    local_state_size = local_state_dim
    neighbors_size = num_neighbors * neighbor_feature_dim
    w_lanes_size = num_w_lanes * waypoint_feature_dim
    w_boundaries_size = num_w_boundaries * boundary_feature_dim
    
    # 1. 提取agents_state (前7个维度)
    agents_state = observation[:, :, :local_state_dim]  # (B, M, 7)
    
    # 2. 提取neighbors_local
    neighbors_start = local_state_size
    neighbors_end = neighbors_start + neighbors_size
    neighbors_flat = observation[:, :, neighbors_start:neighbors_end]  # (B, M, K*7)
    neighbors_local = neighbors_flat.view(batch_size, max_agents, num_neighbors, neighbor_feature_dim)  # (B, M, K, 7)
    
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
    # 退化处理：max==min
    deg_mask = (denom == 0)
    if torch.any(deg_mask):
        y_degen = torch.where(
            x > max_t, torch.ones_like(x),
            torch.where(x < min_t, -torch.ones_like(x), torch.zeros_like(x))
        )
        y = torch.where(deg_mask, y_degen, y)
    return y

def build_network_features(agents_state: torch.Tensor, 
                          neighbors_local: torch.Tensor, 
                          w_lanes_local: torch.Tensor, 
                          w_boundaries_local: torch.Tensor,
                          path_plan: torch.Tensor,
                          stop_lines: torch.Tensor,
                          reward_coef: torch.Tensor,
                          config: SimpleNamespace) -> torch.Tensor:
    """
    将拆解后的观测组件构建为网络输入的特征张量
    Args:
        agents_state: (B, M, 7) - 智能体状态
        neighbors_local: (B, M, K, 7) - 邻居相对状态
        w_lanes_local: (B, M, N_lanes, 2) - 车道线相对坐标
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
    simple_feature_dims = network_config.simple_feature_dims  # [7, 256, 10, 4]
    permutation_feature_dims = network_config.permutation_feature_dims  # [52, 50, 20, 140]
    
    # 计算总输入维度
    total_input_dim = sum(simple_feature_dims) + sum(permutation_feature_dims)
    
    # 初始化输出张量
    features_tensor = torch.zeros(batch_size, max_agents, total_input_dim, device=agents_state.device)
    
    # 1. 构建简单特征 (S(t), G(t), reward系数, 车辆风格参数)
    simple_end = sum(simple_feature_dims)
    
    # S(t): 7维 - 使用agents_state，但将 yaw 替换为 cos(yaw) 保持数值稳定
    s_t_size = simple_feature_dims[0]  # 7
    agents_state_stable = agents_state.clone()
    try:
        agents_state_stable[:, :, 2] = torch.cos(agents_state_stable[:, :, 2])
        agents_state_stable[:, :, 1] = torch.sin(agents_state_stable[:, :, 2]) # 占用一个位置写入sin(yaw)进入网络
        # TODO:后续这里的最大最小最好改为从config中获取
        agents_state_stable[:, :, 3] = normalize_to_minus1_1(agents_state_stable[:, :, 3], -2, 20)
        agents_state_stable[:, :, 4] = normalize_to_minus1_1(agents_state_stable[:, :, 4], 0.8, 3)
        agents_state_stable[:, :, 5] = normalize_to_minus1_1(agents_state_stable[:, :, 5], 0.8, 7)
    except Exception:
        # 若维度不符则回退为原值
        print('error')
        pass 
    features_tensor[:, :, :s_t_size] = agents_state_stable
    
    # G(t): 256维 - 使用路径规划信息
    g_t_size = simple_feature_dims[1]  # 256
    g_t_start = s_t_size
    g_t_end = g_t_start + g_t_size
    path_plan_stable = path_plan.clone().flatten(start_dim=2) #复制，展平
    path_plan_stable = normalize_to_minus1_1(path_plan_stable, -100, 100) #归一化
    features_tensor[:, :, g_t_start:g_t_end] = path_plan_stable

    # reward系数: 10维 - 使用传入的采样参数
    reward_coef_size = simple_feature_dims[2]  # 10
    reward_coef_start = g_t_start + g_t_size
    reward_coef_end = reward_coef_start + reward_coef_size

    reward_coef_stable = reward_coef.clone()

    reward_coef_stable[:, :, 0] = normalize_to_minus1_1(reward_coef_stable[:, :, 0], 2, 12)
    reward_coef_stable[:, :, 1] = normalize_to_minus1_1(reward_coef_stable[:, :, 1], 0, 3)
    reward_coef_stable[:, :, 2] = normalize_to_minus1_1(reward_coef_stable[:, :, 2], 0, 3)
    reward_coef_stable[:, :, 3] = normalize_to_minus1_1(reward_coef_stable[:, :, 3], 0, 0.1)
    reward_coef_stable[:, :, 4] = normalize_to_minus1_1(reward_coef_stable[:, :, 4], 0.00025, 0.025)
    reward_coef_stable[:, :, 5] = normalize_to_minus1_1(reward_coef_stable[:, :, 5], 0, 1)
    reward_coef_stable[:, :, 6] = normalize_to_minus1_1(reward_coef_stable[:, :, 6], 0.00025, 0.0075)
    reward_coef_stable[:, :, 7] = normalize_to_minus1_1(reward_coef_stable[:, :, 7], -0.5, 0.5)
    reward_coef_stable[:, :, 8] = normalize_to_minus1_1(reward_coef_stable[:, :, 8], 0.00025, 0.0075)
    reward_coef_stable[:, :, 9] = normalize_to_minus1_1(reward_coef_stable[:, :, 9], 0, 1)
    features_tensor[:, :, reward_coef_start:reward_coef_end] = reward_coef_stable

    # 车辆风格参数: 4维 - 从agents_state中提取
    vehicle_style_size = simple_feature_dims[3]  # 4
    vehicle_style_start = reward_coef_start + reward_coef_size
    vehicle_style_end = vehicle_style_start + vehicle_style_size
    # TODO:车辆风格暂置为四个0，后续改为传入的风格参数且要做归一化
    vehicle_style = torch.zeros(batch_size, max_agents, vehicle_style_size, device=agents_state.device, dtype=agents_state.dtype)
    features_tensor[:, :, vehicle_style_start:vehicle_style_end] = vehicle_style
    
    # 2. 构建排列不变特征 (road_boundary, lane_points, stop_lines, other_agents)
    permutation_start = simple_end
    
    # road_boundary: 52维 - 使用边界线信息
    road_boundary_size = permutation_feature_dims[0]  # 52
    road_boundary_start = permutation_start
    road_boundary_end = road_boundary_start + road_boundary_size
    
    # 将边界线展平并填充
    w_boundaries_flat = w_boundaries_local.flatten(start_dim=2)  # (B, M, N_boundaries*2)
    w_boundaries_flat = normalize_to_minus1_1(w_boundaries_flat, -100, 100) #归一化
    if w_boundaries_flat.shape[2] <= road_boundary_size:
        features_tensor[:, :, road_boundary_start:road_boundary_start + w_boundaries_flat.shape[2]] = w_boundaries_flat
    else:
        features_tensor[:, :, road_boundary_start:road_boundary_end] = w_boundaries_flat[:, :, :road_boundary_size]
    
    # lane_points: 50维 - 使用车道线信息
    lane_points_size = permutation_feature_dims[1]  # 50
    lane_points_start = road_boundary_end
    lane_points_end = lane_points_start + lane_points_size
    
    # 将车道线展平并填充
    w_lanes_flat = w_lanes_local.flatten(start_dim=2)  # (B, M, N_lanes*2)
    w_lanes_flat = normalize_to_minus1_1(w_lanes_flat, -100, 100) #归一化
    if w_lanes_flat.shape[2] <= lane_points_size:
        features_tensor[:, :, lane_points_start:lane_points_start + w_lanes_flat.shape[2]] = w_lanes_flat
    else:
        features_tensor[:, :, lane_points_start:lane_points_end] = w_lanes_flat[:, :, :lane_points_size]
    
    # stop_lines: 20维 - 使用停止线信息
    stop_lines_size = permutation_feature_dims[2]  # 20
    stop_lines_start = lane_points_end
    stop_lines_end = stop_lines_start + stop_lines_size
    
    # 将停止线展平并填充
    if stop_lines is not None and stop_lines.numel() > 0:
        stop_lines_flat = stop_lines.flatten(start_dim=2)  # (B, M, num_stop_lines*2)
        stop_lines_flat = normalize_to_minus1_1(stop_lines_flat, -100, 100) #归一化
        if stop_lines_flat.shape[2] <= stop_lines_size:
            features_tensor[:, :, stop_lines_start:stop_lines_start + stop_lines_flat.shape[2]] = stop_lines_flat
        else:
            features_tensor[:, :, stop_lines_start:stop_lines_end] = stop_lines_flat[:, :, :stop_lines_size]
    else:
        # 如果没有停止线信息，使用零填充
        features_tensor[:, :, stop_lines_start:stop_lines_end] = 0.0
    
    # other_agents: 140维 - 使用邻居信息
    other_agents_size = permutation_feature_dims[3]  # 140
    other_agents_start = stop_lines_end
    other_agents_end = other_agents_start + other_agents_size
    
    # 将邻居信息按通道做归一化后再展平并填充
    neighbors_proc = neighbors_local.clone()
    try:
        neighbors_proc[:, :, :, 0] = normalize_to_minus1_1(neighbors_proc[:, :, :, 0], -100, 100)
        neighbors_proc[:, :, :, 1] = normalize_to_minus1_1(neighbors_proc[:, :, :, 1], -100, 100)
        neighbors_proc[:, :, :, 2] = normalize_to_minus1_1(neighbors_proc[:, :, :, 2], -40, 40)
        neighbors_proc[:, :, :, 3] = normalize_to_minus1_1(neighbors_proc[:, :, :, 3], -40, 40)
        neighbors_proc[:, :, :, 4] = normalize_to_minus1_1(neighbors_proc[:, :, :, 4], 0.8, 3)
        neighbors_proc[:, :, :, 5] = normalize_to_minus1_1(neighbors_proc[:, :, :, 5], 0.8, 7)
        # 通道 6 为 active 标志，保持原样
    except Exception:
        pass
    neighbors_flat = neighbors_proc.flatten(start_dim=2)  # (B, M, K*7)
    if neighbors_flat.shape[2] <= other_agents_size:
        features_tensor[:, :, other_agents_start:other_agents_start + neighbors_flat.shape[2]] = neighbors_flat
    else:
        features_tensor[:, :, other_agents_start:other_agents_end] = neighbors_flat[:, :, :other_agents_size]
    
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

# ============================== 检查是否所有世界无存活agent ==============================
def all_worlds_no_alive_agents(simulator, cumulative_done_all=None) -> bool:
	"""
	检查是否所有世界都没有存活的智能体。
	与game.py的_check_all_worlds_no_alive_agents逻辑完全一致。
	"""
	try:
		states = simulator.agents_state  # (B, M, S)
		B, M, S = states.shape
		
		# 如果没有done状态记录，检查是否有active的agents
		if cumulative_done_all is None:
			# 初始化时，只要有active的agents就认为有存活的
			for b in range(B):
				world_agents = states[b]  # (M, S)
				active_mask = world_agents[:, 6] > 0.5  # active状态
				if active_mask.any():
					return False  # 有active的agents，认为有存活的
			return True  # 没有active的agents
		
		# 检查每个世界是否有存活的agents
		for b in range(B):
			world_agents = states[b]  # (M, S)
			active_mask = world_agents[:, 6] > 0.5  # active状态
			# 如果有active的agents，检查是否有存活的
			if active_mask.any():
				world_done = cumulative_done_all[b].to(active_mask.device)  # 使用累积done状态
				alive_mask = active_mask & (~world_done)
				if alive_mask.any():
					return False  # 这个世界还有存活的agents
		return True  # 所有世界都没有存活的agents
	except Exception:
		return True

# ============================== 单卡PPO更新函数 ==============================
def perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer, 
								 states_buffer, rewards_buffer, dones_buffer, 
								 values_buffer, old_log_probs_buffer, actions_buffer,
								 features_tensor, simulator, config, iteration):
	"""执行PPO更新，模仿game.py的逻辑"""
	if len(states_buffer) == 0:
		print("⚠️ Buffer为空，无法进行PPO更新")
		return
	print(f"🎯 开始经验采样训练，Buffer长度: {len(states_buffer)}")
	
	# 将buffer转换为tensor
	T = len(states_buffer)
	B, M, S = states_buffer[0].shape
	
	# 构建tensor buffer
	states_tensor = torch.stack(states_buffer, dim=0)  # (T, B, M, S)
	rewards_tensor = torch.stack(rewards_buffer, dim=0)  # (T, B, M)
	dones_tensor = torch.stack(dones_buffer, dim=0)  # (T, B, M)
	values_tensor = torch.stack(values_buffer, dim=0)  # (T, B, M)
	old_log_probs_tensor = torch.stack(old_log_probs_buffer, dim=0)  # (T, B, M)
	actions_tensor = torch.stack(actions_buffer, dim=0)  # (T, B, M)
	
	# 计算最后一个状态的价值（bootstrap）
	with torch.no_grad():
		last_value_pred = model.forward(features_tensor, mode="value")
	
	# 构建values_tp1用于GAE计算
	if last_value_pred.dim() == 3 and last_value_pred.shape[-1] == 1:
		last_value_pred = last_value_pred.squeeze(-1)  # (B, M)
	values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)
	
	# 计算GAE优势（使用段内前缀 OR 的累计 done 掩码，符合PPO定义）
	dones_accum = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	advantages, returns = gae_advantages(rewards_tensor, values_tp1, dones_accum, 0.999, 0.95)
	
	# 优势过滤
	A_max = torch.max(torch.abs(advantages)).item()
	eta = 0.01 * A_max
	keep_mask = (torch.abs(advantages) >= eta)
	
	# 额外剔除：每个(B,M)智能体在第一次 done 之后的所有时间步
	seen_done_inclusive = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	seen_done_prev = torch.roll(seen_done_inclusive, shifts=1, dims=0)
	seen_done_prev[0] = False
	first_done_step = dones_tensor & (~seen_done_prev)
	post_done_mask = seen_done_inclusive & (~first_done_step)
	keep_mask = keep_mask & (~post_done_mask)
	cand_idx = keep_mask.nonzero(as_tuple=False)
	
	print(f"🎯 第 {iteration} 个iteration - 最大|A|: {A_max:.4f}, 阈值: {eta:.4f}")
	print(f"📊 过滤前: {keep_mask.numel()}, 过滤后: {keep_mask.sum().item()}")
	if cand_idx.numel() == 0:
		print("⚠️ 无可用样本，跳过更新")
		return
	
	# 随机选择样本进行更新
	N = cand_idx.shape[0]
	K = min(2000, N)  # batch_size_per_gpu
	if N >= K:
		rand_pos = torch.randperm(N, device=states_tensor.device)[:K]
		selected_idx = cand_idx[rand_pos]
	else:
		rand_pos = torch.randint(0, N, (K,), device=states_tensor.device)
		selected_idx = cand_idx[rand_pos]
	
	selected_t = selected_idx[:, 0]
	selected_b = selected_idx[:, 1]
	selected_m = selected_idx[:, 2]
	print(f"🎯 随机选取 {K} 个样本用于更新（候选 {N}）")
	
	# 提取选中的样本
	agent_indices_batch = selected_m.to(states_tensor.device)
	old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
	returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
	actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
	
	batch_N = old_log_probs_batch.shape[0]
	print(f"🎯 开始PPO更新，样本数量: {batch_N}")
	
	# 训练模式
	model.train()
	# PPO更新循环
	for epoch in range(3):  # ppo_epochs
		# 重新生成观测和特征
		mb_idx = torch.arange(batch_N, device=states_tensor.device)
		mb_old_logp = old_log_probs_batch[mb_idx]
		mb_adv = advantages_batch[mb_idx]
		mb_ret = returns_batch[mb_idx]
		mb_agent_idx = agent_indices_batch[mb_idx]
		mb_actions = actions_batch[mb_idx]
		
		# 基于选中的样本重建特征
		mb_t = selected_t[mb_idx]
		mb_b = selected_b[mb_idx]
		mb_m = selected_m[mb_idx]
		
		# 获取唯一的状态组合
		uniq_tbm, inverse_mb = torch.unique(torch.stack([mb_t, mb_b, mb_m], dim=1), dim=0, return_inverse=True)
		t_u_mb = uniq_tbm[:, 0]
		b_u_mb = uniq_tbm[:, 1]
		m_u_mb = uniq_tbm[:, 2]
		
		# 为每个唯一的(t,b,m)组合生成观测
		agents_states_mb = states_tensor[t_u_mb, b_u_mb]  # (unique_samples, M, S)
		
		# 生成观测
		obs_mb = simulator.observation_generator.generate(agents_states_mb)
		agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, config)
		
		# 构建特征
		path_plan_mb = simulator.agents_path_plans[b_u_mb]
		stop_lines_mb = simulator.stop_lines[b_u_mb] if (simulator.stop_lines is not None and simulator.stop_lines.numel() > 0) else simulator.stop_lines
		reward_coef_mb = simulator.reward_calculator.sampled_params[b_u_mb]
		
		features_u_mb = build_network_features(
			agents_state_dec_mb,
			neighbors_local_mb,
			w_lanes_local_mb,
			w_boundaries_local_mb,
			path_plan_mb,
			stop_lines_mb,
			reward_coef_mb,
			config
		)
		u_idx_mb = inverse_mb.to(states_tensor.device)
		mb_features = features_u_mb[u_idx_mb]
		
		# 策略更新
		action_logits = model.forward(mb_features, mode="policy")
		row_idx = torch.arange(mb_actions.shape[0], device=states_tensor.device)
		logits_selected = action_logits[row_idx, mb_agent_idx]
		dist_selected = torch.distributions.Categorical(logits=logits_selected)
		new_log_probs = dist_selected.log_prob(mb_actions)
		
		# 计算比率和损失
		ratio = torch.exp(new_log_probs - mb_old_logp)
		surr1 = ratio * mb_adv
		surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * mb_adv
		policy_loss = -torch.min(surr1, surr2).mean()
		
		# 熵损失
		entropy = dist_selected.entropy().mean()
		policy_total_loss = policy_loss - 0.01 * entropy
		
		# 策略网络更新
		policy_optimizer.zero_grad()
		policy_total_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.policy_network.parameters(), 1.0)
		policy_optimizer.step()
		
		# 价值网络更新
		value_pred_full = model.forward(mb_features, mode="value").squeeze(-1)
		value_pred = value_pred_full[row_idx, mb_agent_idx]
		value_loss = (value_pred - mb_ret).pow(2).mean()
		value_loss = 0.5 * value_loss
		
		value_optimizer.zero_grad()
		value_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.value_network.parameters(), 1.0)
		value_optimizer.step()
		
		print(f"   Epoch {epoch+1}/2: Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}, Entropy: {entropy.item():.6f}")
	
	# 切回评估模式
	model.eval()
	print(f"✅ 第 {iteration} 个iteration - 经验采样训练完成")

# ============================== 多卡PPO更新函数 ==============================
def perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer, 
								states_buffer, rewards_buffer, dones_buffer, 
								values_buffer, old_log_probs_buffer, actions_buffer,
								features_tensor, simulator, config, iteration, rank):
	"""执行多卡PPO更新，模仿game.py的逻辑"""
	if len(states_buffer) == 0:
		if rank == 0:
			print("⚠️ Buffer为空，无法进行PPO更新")
		return
	if rank == 0:
		print(f"🎯 开始经验采样训练，Buffer长度: {len(states_buffer)}")
	
	# 将buffer转换为tensor
	T = len(states_buffer)
	B, M, S = states_buffer[0].shape
	
	# 构建tensor buffer
	states_tensor = torch.stack(states_buffer, dim=0)  # (T, B, M, S)
	rewards_tensor = torch.stack(rewards_buffer, dim=0)  # (T, B, M)
	dones_tensor = torch.stack(dones_buffer, dim=0)  # (T, B, M)
	values_tensor = torch.stack(values_buffer, dim=0)  # (T, B, M)
	old_log_probs_tensor = torch.stack(old_log_probs_buffer, dim=0)  # (T, B, M)
	actions_tensor = torch.stack(actions_buffer, dim=0)  # (T, B, M)
	
	# 计算最后一个状态的价值（bootstrap）
	with torch.no_grad():
		last_value_pred = model.module.forward(features_tensor, mode="value")
	
	# 构建values_tp1用于GAE计算
	if last_value_pred.dim() == 3 and last_value_pred.shape[-1] == 1:
		last_value_pred = last_value_pred.squeeze(-1)  # (B, M)
	values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)
	
	# 计算GAE优势（使用段内前缀 OR 的累计 done 掩码，符合PPO定义）
	dones_accum = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	advantages, returns = gae_advantages(rewards_tensor, values_tp1, dones_accum, 0.999, 0.95)
	
	# 优势过滤
	A_max = torch.max(torch.abs(advantages)).item()
	eta = 0.01 * A_max
	keep_mask = (torch.abs(advantages) >= eta)
	
	# 额外剔除：每个(B,M)智能体在第一次 done 之后的所有时间步
	seen_done_inclusive = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
	seen_done_prev = torch.roll(seen_done_inclusive, shifts=1, dims=0)
	seen_done_prev[0] = False
	first_done_step = dones_tensor & (~seen_done_prev)
	post_done_mask = seen_done_inclusive & (~first_done_step)
	keep_mask = keep_mask & (~post_done_mask)
	cand_idx = keep_mask.nonzero(as_tuple=False)
	
	if rank == 0:
		print(f"🎯 第 {iteration} 个iteration - 最大|A|: {A_max:.4f}, 阈值: {eta:.4f}")
		print(f"📊 过滤前: {keep_mask.numel()}, 过滤后: {keep_mask.sum().item()}")
	if cand_idx.numel() == 0:
		if rank == 0:
			print("⚠️ 无可用样本，跳过更新")
		return
	
	# 随机选择样本进行更新
	N = cand_idx.shape[0]
	K = min(2000, N)  # batch_size_per_gpu
	if N >= K:
		rand_pos = torch.randperm(N, device=states_tensor.device)[:K]
		selected_idx = cand_idx[rand_pos]
	else:
		rand_pos = torch.randint(0, N, (K,), device=states_tensor.device)
		selected_idx = cand_idx[rand_pos]
	
	selected_t = selected_idx[:, 0]
	selected_b = selected_idx[:, 1]
	selected_m = selected_idx[:, 2]
	if rank == 0:
		print(f"🎯 随机选取 {K} 个样本用于更新（候选 {N}）")
	
	# 提取选中的样本
	agent_indices_batch = selected_m.to(states_tensor.device)
	old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
	returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
	actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1)
	advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
	
	batch_N = old_log_probs_batch.shape[0]
	if rank == 0:
		print(f"🎯 开始PPO更新，样本数量: {batch_N}")
	
	# 训练模式
	model.train()
	# PPO更新循环
	for epoch in range(3):  # ppo_epochs
		# 重新生成观测和特征
		mb_idx = torch.arange(batch_N, device=states_tensor.device)
		mb_old_logp = old_log_probs_batch[mb_idx]
		mb_adv = advantages_batch[mb_idx]
		mb_ret = returns_batch[mb_idx]
		mb_agent_idx = agent_indices_batch[mb_idx]
		mb_actions = actions_batch[mb_idx]
		
		# 基于选中的样本重建特征
		mb_t = selected_t[mb_idx]
		mb_b = selected_b[mb_idx]
		mb_m = selected_m[mb_idx]
		
		# 获取唯一的状态组合
		uniq_tbm, inverse_mb = torch.unique(torch.stack([mb_t, mb_b, mb_m], dim=1), dim=0, return_inverse=True)
		t_u_mb = uniq_tbm[:, 0]
		b_u_mb = uniq_tbm[:, 1]
		m_u_mb = uniq_tbm[:, 2]
		
		# 为每个唯一的(t,b,m)组合生成观测
		agents_states_mb = states_tensor[t_u_mb, b_u_mb]  # (unique_samples, M, S)
		
		# 生成观测
		obs_mb = simulator.observation_generator.generate(agents_states_mb)
		agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, config)
		
		# 构建特征
		path_plan_mb = simulator.agents_path_plans[b_u_mb]
		stop_lines_mb = simulator.stop_lines[b_u_mb] if (simulator.stop_lines is not None and simulator.stop_lines.numel() > 0) else simulator.stop_lines
		reward_coef_mb = simulator.reward_calculator.sampled_params[b_u_mb]
		
		features_u_mb = build_network_features(
			agents_state_dec_mb,
			neighbors_local_mb,
			w_lanes_local_mb,
			w_boundaries_local_mb,
			path_plan_mb,
			stop_lines_mb,
			reward_coef_mb,
			config
		)
		u_idx_mb = inverse_mb.to(states_tensor.device)
		mb_features = features_u_mb[u_idx_mb]
		
		# 策略更新（DDP自动同步）
		action_logits = model.module.forward(mb_features, mode="policy")
		row_idx = torch.arange(mb_actions.shape[0], device=states_tensor.device)
		logits_selected = action_logits[row_idx, mb_agent_idx]
		dist_selected = torch.distributions.Categorical(logits=logits_selected)
		new_log_probs = dist_selected.log_prob(mb_actions)
		
		# 计算比率和损失
		ratio = torch.exp(new_log_probs - mb_old_logp)
		surr1 = ratio * mb_adv
		surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * mb_adv
		policy_loss = -torch.min(surr1, surr2).mean()
		
		# 熵损失
		entropy = dist_selected.entropy().mean()
		policy_total_loss = policy_loss - 0.01 * entropy
		
		# 策略网络更新
		policy_optimizer.zero_grad()
		policy_total_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.module.policy_network.parameters(), 1.0)
		policy_optimizer.step()
		
		# 价值网络更新（DDP自动同步）
		value_pred_full = model.module.forward(mb_features, mode="value").squeeze(-1)
		value_pred = value_pred_full[row_idx, mb_agent_idx]
		value_loss = (value_pred - mb_ret).pow(2).mean()
		value_loss = 0.5 * value_loss
		
		value_optimizer.zero_grad()
		value_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.module.value_network.parameters(), 1.0)
		value_optimizer.step()
		
		if rank == 0:
			print(f"   Epoch {epoch+1}/2: Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}, Entropy: {entropy.item():.6f}")
	
	# 切回评估模式
	model.eval()
	if rank == 0:
		print(f"✅ 第 {iteration} 个iteration - 经验采样训练完成")
		
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
		
		# 分别创建策略网络和价值网络的优化器
		policy_optimizer = optim.Adam(model.policy_network.parameters(), lr=learning_rate)
		value_optimizer = optim.Adam(model.value_network.parameters(), lr=learning_rate)

		# 分别创建策略网络和价值网络的调度器
		policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_iterations, eta_min=0.0)
		value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_iterations, eta_min=0.0)
		
		# 优势过滤参数
		beta = getattr(training_cfg, 'advantage_filter_beta', 0.25)	# EWMA衰减参数
		advantage_filter_threshold = getattr(training_cfg, 'advantage_filter_threshold', 0.01)	# 优势过滤阈值
		A_max_ewma = None 		# EWMA of max absolute advantage
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 2000)  # 每GPU的batch size
		rollout_length = getattr(training_cfg, 'rollout_length', 128)  # rollout长度
		
		for k in range(num_iterations):
			print(f"🔄 开始第 {k+1}/{num_iterations} 轮迭代")

			episode_start_time = time.time()
			# ============================== 采样（初始化） ==============================
			initial_observation = simulator.reset()
			path_plan = simulator.agents_path_plans
			stop_lines = simulator.stop_lines
			# 拆解initial_observation为网络需要的组件
			agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, config)
			# 构建网络输入特征
			features_tensor = build_network_features(
				agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
				path_plan, stop_lines, simulator.reward_calculator.sampled_params, config)
			
			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			states_buffer = []
			rewards_buffer = []
			dones_buffer = []
			values_buffer = []
			old_log_probs_buffer = []
			actions_buffer = []
			buffer_step_count = 0
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				if all_worlds_no_alive_agents(simulator, cumulative_done_all):
					if buffer_step_count > 0:
						print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1)
					else:
						print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				with torch.no_grad():
					action_logits = model.forward(features_tensor, mode="policy")
					value_pred = model.forward(features_tensor, mode="value")
				action_dist = torch.distributions.Categorical(logits=action_logits)
				actions = action_dist.sample()
				
				# 在推进环境前缓存当前状态
				pre_state = agents_state.clone()
				
				# 环境步进
				observation, reward, done = simulator.step(actions)
				
				# 写入训练buffer（与game.py一致）
				states_buffer.append(pre_state)
				rewards_buffer.append(reward.clone())
				dones_buffer.append(done.clone())
				values_buffer.append(value_pred.clone())
				old_log_probs_buffer.append(action_dist.log_prob(actions).detach().clone())
				actions_buffer.append(actions.clone())
				buffer_step_count += 1
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().to('cpu').bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				
				# 更新观测与特征
				agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
				features_tensor = build_network_features(
					agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
					path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
				)
				
				step_count += 1
				print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1)
						print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						perform_ppo_update_single_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1)
						
						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						if all_worlds_no_alive_agents(simulator):
							print("🔄 所有世界都没有存活agents，开启新iteration...")
							break
						else:
							print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							states_buffer = []
							rewards_buffer = []
							dones_buffer = []
							values_buffer = []
							old_log_probs_buffer = []
							actions_buffer = []
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性


			# 分别更新学习率调度器
			policy_scheduler.step()
			value_scheduler.step()
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
		# 优势过滤参数
		beta = getattr(training_cfg, 'advantage_filter_beta', 0.25)
		advantage_filter_threshold = getattr(training_cfg, 'advantage_filter_threshold', 0.01)
		A_max_ewma = None
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 2000)
		rollout_length = getattr(training_cfg, 'rollout_length', 128)

		# 分别创建策略网络和价值网络的优化器（DDP下需访问 module）
		policy_optimizer = optim.Adam(model.module.policy_network.parameters(), lr=learning_rate)
		value_optimizer = optim.Adam(model.module.value_network.parameters(), lr=learning_rate)
		policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_iterations, eta_min=0.0)
		value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_iterations, eta_min=0.0)
		
		# 每一轮迭代（步进式训练：与game.py完全一致）
		for k in range(num_iterations):
			# 2) 本轮开始：重置完成计数（保持与原多卡同步逻辑一致）
			num_workers_done.set("done", b"0")

			# 采样初始化
			initial_observation = simulator.reset()
			path_plan = simulator.agents_path_plans
			stop_lines = simulator.stop_lines
			agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, config)
			features_tensor = build_network_features(
				agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
				path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
			)

			# =========================== 步进式训练：与game.py完全一致 ==============================
			B, M, S = agents_state.shape
			step_count = 0
			
			# 初始化全局buffer（与game.py一致）
			states_buffer = []
			rewards_buffer = []
			dones_buffer = []
			values_buffer = []
			old_log_probs_buffer = []
			actions_buffer = []
			buffer_step_count = 0
			
			# 初始化累积done状态（与game.py一致）
			cumulative_done_all = None

			while step_count < max_episode_length:
				# 全局死亡检测：如果所有世界都没有存活agents，执行PPO更新后开始新iteration
				if all_worlds_no_alive_agents(simulator, cumulative_done_all):
					if buffer_step_count > 0:
						if rank == 0:
							print(f"🔄 所有agents死亡，执行PPO更新后开始新iteration")
						perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1, rank)
					else:
						if rank == 0:
							print(f"🔄 所有agents死亡，无buffer数据，直接开始新iteration")
					break
				
				# 单步训练（与game.py的update_game_state一致）
				step_start_time = time.time()
				with torch.no_grad():
					action_logits = model.module.forward(features_tensor, mode="policy")
					value_pred = model.module.forward(features_tensor, mode="value")
				action_dist = torch.distributions.Categorical(logits=action_logits)
				actions = action_dist.sample()
				
				# 在推进环境前缓存当前状态
				pre_state = agents_state.clone()
				
				# 环境步进
				observation, reward, done = simulator.step(actions)
				
				# 写入训练buffer（与game.py一致）
				states_buffer.append(pre_state)
				rewards_buffer.append(reward.clone())
				dones_buffer.append(done.clone())
				values_buffer.append(value_pred.clone())
				old_log_probs_buffer.append(action_dist.log_prob(actions).detach().clone())
				actions_buffer.append(actions.clone())
				buffer_step_count += 1
				
				# 累积done状态，记录这一轮iteration中done过的车辆（与game.py一致）
				current_done_all = done.detach().to('cpu').bool()  # (B, M)
				if cumulative_done_all is None:
					cumulative_done_all = current_done_all.clone()
				else:
					cumulative_done_all = cumulative_done_all | current_done_all
				
				# 更新观测与特征
				agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
				features_tensor = build_network_features(
					agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
					path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
				)
				
				step_count += 1
				if rank == 0:
					print(f"\t📍 第 {step_count}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
				
				# 检查是否需要PPO更新（与game.py一致）
				if buffer_step_count >= rollout_length or step_count >= max_episode_length:
					if step_count >= max_episode_length:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到最大步数 {max_episode_length}，强制开始PPO更新...")
						perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1, rank)
						if rank == 0:
							print("🔄 达到最大步数，强制开启新iteration...")
						break
					else:
						if rank == 0:
							print(f"🎯 第 {k+1} 个iteration - 达到rollout长度 {rollout_length}，开始PPO更新...")
						perform_ppo_update_multi_gpu(model, policy_optimizer, value_optimizer, 
													states_buffer, rewards_buffer, dones_buffer, 
													values_buffer, old_log_probs_buffer, actions_buffer,
													features_tensor, simulator, config, k+1, rank)
						
						# 检查是否所有世界都没有存活agents，如果是则开启新iteration
						if all_worlds_no_alive_agents(simulator, cumulative_done_all):
							if rank == 0:
								print("🔄 所有世界都没有存活agents，开启新iteration...")
							break
						else:
							if rank == 0:
								print("✅ 仍有世界有存活agents，继续下一个128step...")
							# 仅清空采样buffer，保留累积的dones用于可视化与死亡着色（与game.py一致）
							states_buffer = []
							rewards_buffer = []
							dones_buffer = []
							values_buffer = []
							old_log_probs_buffer = []
							actions_buffer = []
							buffer_step_count = 0
							# 注意：不重置cumulative_done_all，保持跨rollout的一致性

			# 7) 在优化器更新后调用学习率调度器（与单卡一致）
			policy_scheduler.step()
			value_scheduler.step()

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

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
    simple_feature_dims = network_config.simple_feature_dims  # [10, 1024, 10, 4]
    permutation_feature_dims = network_config.permutation_feature_dims  # [52, 50, 20, 140]
    
    # 计算总输入维度
    total_input_dim = sum(simple_feature_dims) + sum(permutation_feature_dims)
    
    # 初始化输出张量
    features_tensor = torch.zeros(batch_size, max_agents, total_input_dim, device=agents_state.device)
    
    # 1. 构建简单特征 (S(t), G(t), reward系数, 车辆风格参数)
    simple_end = sum(simple_feature_dims)
    
    # S(t): 7维 - 直接使用agents_state
    s_t_size = simple_feature_dims[0]  # 7
    features_tensor[:, :, :s_t_size] = agents_state
    
    # G(t): 256维 - 使用路径规划信息
    g_t_size = simple_feature_dims[1]  # 256
    g_t_start = s_t_size
    g_t_end = g_t_start + g_t_size
    features_tensor[:, :, g_t_start:g_t_end] = path_plan.flatten(start_dim=2)  
        
    # reward系数: 10维 - 使用传入的采样参数
    reward_coef_size = simple_feature_dims[2]  # 10
    reward_coef_start = g_t_start + g_t_size
    reward_coef_end = reward_coef_start + reward_coef_size
    features_tensor[:, :, reward_coef_start:reward_coef_end] = reward_coef
    
    # 车辆风格参数: 4维 - 从agents_state中提取
    vehicle_style_size = simple_feature_dims[3]  # 4
    vehicle_style_start = reward_coef_start + reward_coef_size
    vehicle_style_end = vehicle_style_start + vehicle_style_size
    # 使用车辆的长度、宽度、速度和活跃状态
    vehicle_style = torch.stack([
        agents_state[:, :, 4],  # length
        agents_state[:, :, 5],  # width
        agents_state[:, :, 3],  # speed
        agents_state[:, :, 6]   # active
    ], dim=2)
    features_tensor[:, :, vehicle_style_start:vehicle_style_end] = vehicle_style
    
    # 2. 构建排列不变特征 (road_boundary, lane_points, stop_lines, other_agents)
    permutation_start = simple_end
    
    # road_boundary: 52维 - 使用边界线信息
    road_boundary_size = permutation_feature_dims[0]  # 52
    road_boundary_start = permutation_start
    road_boundary_end = road_boundary_start + road_boundary_size
    
    # 将边界线展平并填充
    w_boundaries_flat = w_boundaries_local.flatten(start_dim=2)  # (B, M, N_boundaries*2)
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
    
    # 将邻居信息展平并填充
    neighbors_flat = neighbors_local.flatten(start_dim=2)  # (B, M, K*7)
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
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 32000)  # 每GPU的batch size
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
				path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
			)

			# =========================== 分段rollout：每段 rollout_length 后更新 ==============================
			B, M, S = agents_state.shape
			t_global = 0
			while t_global < max_episode_length:
				segment_steps = min(rollout_length, max_episode_length - t_global)
				# 本段buffer
				states_buffer = torch.empty((segment_steps + 1, B, M, S), device=agents_state.device, dtype=agents_state.dtype)
				rewards_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				dones_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				values_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				old_log_probs_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				states_buffer[0].copy_(agents_state)
				# 段内采样
				for t in range(segment_steps):
					step_start_time = time.time()
					with torch.no_grad():
						action_logits = model.forward_policy(features_tensor)
						value_pred = model.forward(features_tensor, mode="value")
					action_dist = torch.distributions.Categorical(logits=action_logits)
					actions = action_dist.sample()
					# 环境步进
					observation, reward, done = simulator.step(actions)
					# 更新观测与特征
					agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
					features_tensor = build_network_features(
						agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
						path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
					)
					# 写入buffer
					states_buffer[t + 1].copy_(agents_state)
					values_buffer[t] = value_pred.detach()
					rewards_buffer[t] = reward
					dones_buffer[t] = done
					old_log_probs_buffer[t] = action_dist.log_prob(actions).detach()
					print(f"\t📍 第 {t_global + t + 1}/{segment_steps} 步耗时: {time.time()-step_start_time:.4f}秒")

				# 段末bootstrap并计算GAE
				with torch.no_grad():
					last_value_pred = model.forward(features_tensor, mode="value")
				values_tp1 = torch.cat([values_buffer, last_value_pred.unsqueeze(0)], dim=0)
				advantages, returns = gae_advantages(rewards_buffer, values_tp1, dones_buffer, gamma, gae_lambda)

				# 优势过滤（基于段内）
				A_max = torch.max(torch.abs(advantages)).item()
				if A_max_ewma is None:
					A_max_ewma = A_max
				else:
					A_max_ewma = beta * A_max + (1 - beta) * A_max_ewma
				eta = advantage_filter_threshold * A_max_ewma
				advantage_mask = torch.abs(advantages) < eta
				print(f"🎯 第{k+1}轮 - 段[{t_global},{t_global+segment_steps}) 最大|A|: {A_max:.4f}, EWMA: {A_max_ewma:.4f}, 阈值: {eta:.4f}")
				print(f"📊 过滤前: {advantage_mask.numel()}, 被过滤: {advantage_mask.sum().item()}")

				# 随机采样 batch_size_per_gpu 个样本并进行PPO更新
				keep_mask = (torch.abs(advantages) >= eta)
				cand_idx = keep_mask.nonzero(as_tuple=False)
				if cand_idx.numel() == 0:
					print("⚠️ 本段无可用样本，跳过更新")
					t_global += segment_steps
					continue
				# 随机选择 batch_size_per_gpu 个样本（不足则放回采样）
				N = cand_idx.shape[0]
				K = batch_size_per_gpu
				if N >= K:
					rand_pos = torch.randperm(N, device=device)[:K]
					selected_idx = cand_idx[rand_pos]
				else:
					rand_pos = torch.randint(0, N, (K,), device=device)
					selected_idx = cand_idx[rand_pos]
				selected_t = selected_idx[:, 0]
				selected_b = selected_idx[:, 1]
				selected_m = selected_idx[:, 2]
				print(f"🎯 本段随机选取 {K} 个样本用于更新（候选 {N}）")
				agent_indices_batch = selected_m.to(device)
				old_log_probs_batch = old_log_probs_buffer[selected_t, selected_b, selected_m].view(-1)
				advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
				returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
				advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
				batch_N = old_log_probs_batch.shape[0]

				segment_start_time = time.time()
				for _ in range(ppo_epochs):
					# 直接使用整批（不再分mini-batch）
					mb_idx = torch.arange(batch_N, device=device)
					mb_old_logp = old_log_probs_batch[mb_idx]
					mb_adv = advantages_batch[mb_idx]
					mb_ret = returns_batch[mb_idx]
					mb_agent_idx = agent_indices_batch[mb_idx]

					# 基于本段状态重建整批特征
					mb_t = selected_t[mb_idx]
					mb_b = selected_b[mb_idx]
					mb_m = selected_m[mb_idx]
					uniq_tb_mb, inverse_mb = torch.unique(torch.stack([mb_t, mb_b], dim=1), dim=0, return_inverse=True)
					t_u_mb = uniq_tb_mb[:, 0]
					b_u_mb = uniq_tb_mb[:, 1]
					agents_states_mb = states_buffer[t_u_mb, b_u_mb]
					obs_mb = simulator.observation_generator.generate(agents_states_mb)
					agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, config)
					path_plan_mb = path_plan[b_u_mb]
					stop_lines_mb = stop_lines[b_u_mb] if (stop_lines is not None and stop_lines.numel() > 0) else stop_lines
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
					u_idx_mb = inverse_mb.to(device)
					mb_features = features_u_mb[u_idx_mb]

					# 策略更新
					print(mb_features.shape)
					action_logits = model.forward(mb_features, mode="policy")
					action_dist = torch.distributions.Categorical(logits=action_logits)
					actions_sampled = action_dist.sample()
					row_idx = torch.arange(actions_sampled.shape[0], device=device)
					new_log_probs = action_dist.log_prob(actions_sampled)[row_idx, mb_agent_idx]
					ratio = torch.exp(torch.clamp(new_log_probs - mb_old_logp, -1, 1))
					surr1 = ratio * mb_adv
					surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
					policy_loss = -torch.min(surr1, surr2).mean()
					entropy = action_dist.entropy()[row_idx, mb_agent_idx].mean()
					policy_total_loss = policy_loss - entropy_coef * entropy
					policy_total_loss = torch.clamp(policy_total_loss, -100, 100)
					policy_optimizer.zero_grad(set_to_none=True)
					policy_total_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.policy_network.parameters(), max_grad_norm)
					policy_optimizer.step()

					# 价值更新
					value_pred_full = model.forward(mb_features, mode="value").squeeze(-1)
					value_pred = value_pred_full[row_idx, mb_agent_idx]
					value_loss = (value_pred - mb_ret).pow(2).mean()
					value_loss = value_loss_coef * value_loss
					value_loss = torch.clamp(value_loss, -100, 100)
					value_optimizer.zero_grad(set_to_none=True)
					value_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.value_network.parameters(), max_grad_norm)
					value_optimizer.step()

					# 打印损失信息
					print(f"🎯 第{k+1}轮 - 段[{t_global},{t_global+segment_steps}) - PPO更新:")
					print(f"   Policy Loss: {policy_loss.item():.6f}, Entropy: {entropy.item():.6f}")
					print(f"   Value Loss: {value_loss.item():.6f}, Total Policy Loss: {policy_total_loss.item():.6f}")
					print(f"   Ratio Mean: {ratio.mean().item():.4f}, Ratio Std: {ratio.std().item():.4f}")
					print(f"   Advantage Mean: {mb_adv.mean().item():.4f}, Advantage Std: {mb_adv.std().item():.4f}")

				# 推进到下一段
				t_global += segment_steps
				print(f"🎯 第{k+1}轮 - 段[{t_global},{t_global+segment_steps}) 更新完成,耗时: {time.time()-segment_start_time:.4f}秒")

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
		batch_size_per_gpu = getattr(training_cfg, 'batch_size_per_gpu', 32000)
		rollout_length = getattr(training_cfg, 'rollout_length', 128)

		# 分别创建策略网络和价值网络的优化器（DDP下需访问 module）
		policy_optimizer = optim.Adam(model.module.policy_network.parameters(), lr=learning_rate)
		value_optimizer = optim.Adam(model.module.value_network.parameters(), lr=learning_rate)
		policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=num_iterations, eta_min=0.0)
		value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=num_iterations, eta_min=0.0)
		
		# 每一轮迭代（分段rollout + 随机批更新）
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

			B, M, S = agents_state.shape
			t_global = 0
			while t_global < max_episode_length:
				segment_steps = min(rollout_length, max_episode_length - t_global)
				states_buffer = torch.empty((segment_steps + 1, B, M, S), device=agents_state.device, dtype=agents_state.dtype)
				rewards_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				dones_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				values_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				old_log_probs_buffer = torch.zeros(segment_steps, B, M, device=agents_state.device)
				states_buffer[0].copy_(agents_state)

				for t in range(segment_steps):
					with torch.no_grad():
						action_logits = model.module.forward_policy(features_tensor)
						value_pred = model.module.forward(features_tensor, mode="value")
					action_dist = torch.distributions.Categorical(logits=action_logits)
					actions = action_dist.sample()
					observation, reward, done = simulator.step(actions)
					agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
					features_tensor = build_network_features(
						agents_state, neighbors_local, w_lanes_local, w_boundaries_local,
						path_plan, stop_lines, simulator.reward_calculator.sampled_params, config
					)
					states_buffer[t + 1].copy_(agents_state)
					values_buffer[t] = value_pred.detach()
					rewards_buffer[t] = reward
					dones_buffer[t] = done
					old_log_probs_buffer[t] = action_dist.log_prob(actions).detach()

				# 本段GAE
				with torch.no_grad():
					last_value_pred = model.module.forward(features_tensor, mode="value")
				values_tp1 = torch.cat([values_buffer, last_value_pred.unsqueeze(0)], dim=0)
				advantages, returns = gae_advantages(rewards_buffer, values_tp1, dones_buffer, gamma, gae_lambda)

				# 优势过滤
				A_max = torch.max(torch.abs(advantages)).item()
				if A_max_ewma is None:
					A_max_ewma = A_max
				else:
					A_max_ewma = beta * A_max + (1 - beta) * A_max_ewma
				eta = advantage_filter_threshold * A_max_ewma
				keep_mask = (torch.abs(advantages) >= eta)
				cand_idx = keep_mask.nonzero(as_tuple=False)
				if cand_idx.numel() == 0:
					# 本段无可用样本，推进到下一段
					t_global += segment_steps
					continue

				# 从候选中随机抽取 batch_size_per_gpu 个样本（不足放回）
				N = cand_idx.shape[0]
				K = batch_size_per_gpu
				if N >= K:
					rand_pos = torch.randperm(N, device=device)[:K]
					selected_idx = cand_idx[rand_pos]
				else:
					rand_pos = torch.randint(0, N, (K,), device=device)
					selected_idx = cand_idx[rand_pos]
				selected_t = selected_idx[:, 0]
				selected_b = selected_idx[:, 1]
				selected_m = selected_idx[:, 2]

				agent_indices_batch = selected_m.to(device)
				old_log_probs_batch = old_log_probs_buffer[selected_t, selected_b, selected_m].view(-1)
				advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
				returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
				advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

				batch_N = old_log_probs_batch.shape[0]
				segment_start_time = time.time()
				for _ in range(ppo_epochs):
					# 整批更新
					mb_idx = torch.arange(batch_N, device=device)
					mb_old_logp = old_log_probs_batch[mb_idx]
					mb_adv = advantages_batch[mb_idx]
					mb_ret = returns_batch[mb_idx]
					mb_agent_idx = agent_indices_batch[mb_idx]

					# 重建整批特征（基于本段缓存）
					mb_t = selected_t[mb_idx]
					mb_b = selected_b[mb_idx]
					mb_m = selected_m[mb_idx]
					uniq_tb_mb, inverse_mb = torch.unique(torch.stack([mb_t, mb_b], dim=1), dim=0, return_inverse=True)
					t_u_mb = uniq_tb_mb[:, 0]
					b_u_mb = uniq_tb_mb[:, 1]
					agents_states_mb = states_buffer[t_u_mb, b_u_mb]
					obs_mb = simulator.observation_generator.generate(agents_states_mb)
					agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, config)
					path_plan_mb = path_plan[b_u_mb]
					stop_lines_mb = stop_lines[b_u_mb] if (stop_lines is not None and stop_lines.numel() > 0) else stop_lines
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
					u_idx_mb = inverse_mb.to(device)
					mb_features = features_u_mb[u_idx_mb]

					# 策略更新（DDP自动同步）
					action_logits = model.module.forward(mb_features, mode="policy")
					action_dist = torch.distributions.Categorical(logits=action_logits)
					actions_sampled = action_dist.sample()
					row_idx = torch.arange(actions_sampled.shape[0], device=device)
					new_log_probs = action_dist.log_prob(actions_sampled)[row_idx, mb_agent_idx]
					ratio = torch.exp(torch.clamp(new_log_probs - mb_old_logp, -1, 1))
					surr1 = ratio * mb_adv
					surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
					policy_loss = -torch.min(surr1, surr2).mean()
					entropy = action_dist.entropy()[row_idx, mb_agent_idx].mean()
					policy_total_loss = policy_loss - entropy_coef * entropy
					policy_total_loss = torch.clamp(policy_total_loss, -100, 100)
					policy_optimizer.zero_grad(set_to_none=True)
					policy_total_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.module.policy_network.parameters(), max_grad_norm)
					policy_optimizer.step()

					# 价值更新（DDP自动同步）
					value_pred_full = model.module.forward(mb_features, mode="value").squeeze(-1)
					value_pred = value_pred_full[row_idx, mb_agent_idx]
					value_loss = (value_pred - mb_ret).pow(2).mean()
					value_loss = value_loss_coef * value_loss
					value_loss = torch.clamp(value_loss, -100, 100)
					value_optimizer.zero_grad(set_to_none=True)
					value_loss.backward()
					torch.nn.utils.clip_grad_norm_(model.module.value_network.parameters(), max_grad_norm)
					value_optimizer.step()

					# 打印损失信息（多卡）
					if rank == 0:
						print(f"[Rank {rank}] 第{k+1}轮 - 段[{t_global},{t_global+segment_steps}) - PPO更新:")
						print(f"   Policy Loss: {policy_loss.item():.6f}, Entropy: {entropy.item():.6f}")
						print(f"   Value Loss: {value_loss.item():.6f}, Total Policy Loss: {policy_total_loss.item():.6f}")
						print(f"   Ratio Mean: {ratio.mean().item():.4f}, Ratio Std: {ratio.std().item():.4f}")
						print(f"   Advantage Mean: {mb_adv.mean().item():.4f}, Advantage Std: {mb_adv.std().item():.4f}")

				# 推进到下一段
				t_global += segment_steps
				if rank == 0:
					print(f"[Round {k}] 段完成, 用时 {time.time()-segment_start_time:.4f}s")

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
	with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	run_distributed_ddppo(cfg, ranks)

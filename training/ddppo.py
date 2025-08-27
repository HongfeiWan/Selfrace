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
	advantages = (advantages-advantages.mean())/advantages.std()
	return advantages, returns #即返回A(s,a), Q(s,a)

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

		model = create_network(config=config, network_type="shared")
		model = model.to(device)
		simulator = TeraflowSimulator(config=config_dict, device=device)
		sim_cfg = getattr(config, 'simulator')
		training_cfg = getattr(config, 'training')
		learning_rate = getattr(training_cfg, 'learning_rate')
		num_iterations = getattr(training_cfg, 'batch_size_per_gpu')
		max_episode_length = getattr(training_cfg,'max_episode_length')
		ppo_epochs = getattr(training_cfg, 'ppo_epochs')
		rollout_length = getattr(training_cfg, 'rollout_length')

		gamma = getattr(training_cfg, 'gamma')
		gae_lambda = getattr(training_cfg, 'gae_lambda')

		clip_ratio = getattr(training_cfg, 'clip_ratio')
		entropy_coef = getattr(training_cfg, 'entropy_coef')
		value_loss_coef = getattr(training_cfg, 'value_loss_coef')

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=0.0)
		

		for k in range(num_iterations):
			print(f"🔄 开始第 {k+1}/{num_iterations} 轮迭代")
			episode_start_time = time.time()

			# ============================== 采样 ==============================
			# simulator初始化
			initial_observation = simulator.reset()
			path_plan = simulator.agents_path_plans
			stop_lines = simulator.stop_lines
			# 拆解initial_observation为网络需要的组件
			agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, config)
			# 构建网络输入特征
			features_tensor = build_network_features(agents_state, neighbors_local, w_lanes_local, w_boundaries_local, path_plan, stop_lines, simulator.reward_calculator.sampled_params, config)
			# =========================== 初始化本轮buffer（存每步所有agents的状态） ==============================
			# 形状: [T+1, B, M, 7]
			B, M, S = agents_state.shape
			episode_states_buffer = torch.empty((max_episode_length+1, B, M, S), device=agents_state.device, dtype=agents_state.dtype)
			episode_states_buffer[0] = agents_state
			rewards_buffer = torch.zeros(max_episode_length, B, M, device=agents_state.device)
			dones_buffer = torch.zeros(max_episode_length, B, M, device=agents_state.device)
			values_buffer = torch.zeros(max_episode_length, B, M, device=agents_state.device)
			old_log_probs_buffer = torch.zeros(max_episode_length, B, M, device=agents_state.device)

			for t in range(max_episode_length):
				step_start_time = time.time()
				# ============================== 执行环境步进 ==============================
				# 使用网络进行前向传播
				with torch.no_grad():
					action_logits, value_pred = model(features_tensor)
				dist = torch.distributions.Categorical(logits=action_logits)
				actions = dist.sample()  # 根据策略分布采样动作索引 (0-11)
				# 执行环境步进
				observation, reward, done = simulator.step(actions)
				# 更新观测数据用于下一步
				agents_state, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
				features_tensor = build_network_features(agents_state, neighbors_local, w_lanes_local, w_boundaries_local, path_plan, stop_lines, simulator.reward_calculator.sampled_params, config)
				# ============================== 收集经验 ==================================
				# 记录本步所有agents的状态到buffer
				episode_states_buffer[t+1].copy_(agents_state)
				values_buffer[t] = value_pred.detach()
				rewards_buffer[t] = reward
				dones_buffer[t] = done
				old_log_probs_buffer[t] = dist.log_prob(actions).detach()
				print(f"  📍 第 {t+1}/{max_episode_length} 步耗时: {time.time()-step_start_time:.4f}秒")
			# ============================== 计算优势 ==============================
			with torch.no_grad():
				_, last_value_pred = model(features_tensor)  # 最后一个状态的V_{T}
			# 拼接为 [T+1, B, M]
			values_tp1 = torch.cat([values_buffer, last_value_pred.unsqueeze(0)], dim=0)
			advantages, returns = gae_advantages(rewards_buffer, values_tp1, dones_buffer, gamma, gae_lambda)
			# ============================== 更新网络 ==============================
			for _ in range(ppo_epochs):
				indices = torch.randperm(max_episode_length, device=device)[:rollout_length]
				# 重新构建网络输入特征以节约显存
				features_batch = []
				for idx in indices:
					# 对每个时间步重新生成观测和特征
					agents_state = episode_states_buffer[idx]  # [B, M, 7]
					# 通过simulator重新生成观测
					observation = simulator.observation_generator.generate(agents_state)
					# 拆解观测并构建特征
					agents_state_decomp, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config)
					features = build_network_features(agents_state_decomp, neighbors_local, w_lanes_local, w_boundaries_local, path_plan, stop_lines, simulator.reward_calculator.sampled_params, config)
					features_batch.append(features)
					
				features_tensor = torch.stack(features_batch, dim=0)  # [rollout_length, B, M, total_input_dim]
				# 重塑为网络期望的输入形状 [B*rollout_length, M, total_input_dim]
				features_tensor = features_tensor.transpose(0, 1).contiguous().view(-1, features_tensor.shape[2], features_tensor.shape[3])
				
				old_log_probs_batch = old_log_probs_buffer[indices].transpose(0, 1).contiguous().view(-1)  # [B*rollout_length*M]
				advantages_batch = advantages[indices].transpose(0, 1).contiguous().view(-1)  # [B*rollout_length*M]
				returns_batch = returns[indices].transpose(0, 1).contiguous().view(-1)  # [B*rollout_length*M]
				action_logits, value_pred = model(features_tensor)
				dist = torch.distributions.Categorical(logits=action_logits)
				actions_batch = dist.sample()
				new_log_probs = dist.log_prob(actions_batch)
				# 确保维度匹配
				new_log_probs = new_log_probs.view(-1)  # [B*rollout_length*M]
				value_pred = value_pred.view(-1)  # [B*rollout_length*M]
				ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs_batch, -1, 1))
				surr1 = ratio * advantages_batch
				surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_batch
				policy_loss = -torch.min(surr1, surr2).mean()
				entropy = dist.entropy().mean()
				value_loss = (value_pred - returns_batch).pow(2).mean()
				loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
				loss = torch.clamp(loss, -100, 100)
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()
			scheduler.step()
			print(f"🎯 本轮总步数耗时: {time.time()-episode_start_time:.4f}秒")
		print('train done!')
		return 0
	
	try:
		device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
		torch.cuda.set_device(device) if device.type == 'cuda' else None
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
		model = create_network(config=config, network_type="shared")
		model = model.to(device)
		
		# 按原文示例的DDP签名（等价于传入本地rank）
		if device.type == 'cuda':
			model = DDP(model, device_ids=[rank], output_device=rank)
		else:
			model = DDP(model)

		# 训练超参从配置读取（含回退到现有键名）
		sim_cfg = getattr(config, 'simulator', SimpleNamespace())
		training_cfg = getattr(config, 'training', SimpleNamespace())
		learning_rate = getattr(training_cfg, 'learning_rate', 3e-4)
		num_iterations = getattr(training_cfg, 'num_iterations', 10)
		# M 维度：优先使用 training.batch_M，否则回退到 simulator.max_agents_num，再否则用150
		batch_M = getattr(training_cfg, 'batch_M', getattr(sim_cfg, 'max_agents_num', 150))
		preempt_ratio = getattr(training_cfg, 'preempt_ratio', 0.6)
		# 每轮采样步数：优先 training.max_experience_steps，否则回退到 training.rollout_length
		max_experience_steps = getattr(training_cfg, 'max_experience_steps', getattr(training_cfg, 'rollout_length', 1024))
		min_steps_fraction = getattr(training_cfg, 'min_steps_fraction', 0.25)
		# PPO 迭代：优先 training.n_ppo_epochs，否则回退到 training.ppo_epochs
		n_ppo_epochs = getattr(training_cfg, 'n_ppo_epochs', getattr(training_cfg, 'ppo_epochs', 2))
		n_ppo_batch = getattr(training_cfg, 'n_ppo_batch', 4)
		ppo_batch_size = getattr(training_cfg, 'ppo_batch_size', 32)
		gamma = getattr(training_cfg, 'gamma', 0.999)
		gae_lambda = getattr(training_cfg, 'gae_lambda', 0.95)

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #TODO:这里把采样换入simulator的输入、ppo换成已经写好的test_ppo就OK了！！
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=0.0)
		
		# 每一轮迭代
		for k in range(num_iterations):
			# 注意：学习率调度器应该在优化器更新后调用
			# 这里先不调用scheduler.step()，等优化器更新后再调用

			# 2) 本轮开始：重置完成计数
			num_workers_done.set("done", b"0")

			# 3) rollout 收集，带抢占
			min_steps = max(1, int(max_experience_steps * min_steps_fraction))
			preempt_threshold = max(1, int(math.ceil(preempt_ratio * gpu_count)))
			collected_steps = 0
			input_dim = model.module.feature_encoder.total_input_dim
			values_steps = torch.zeros(max_experience_steps, batch_M, device=device)
			last_features = None
			for step in range(max_experience_steps):
				# collect_step(model): 这里用随机张量模拟一次环境交互
				features_tensor = torch.randn(1, batch_M, input_dim, device=device)
				with torch.no_grad():
					action_logits, values = model(features_tensor)
				# 记录 V_t: [B]（去掉[1, B, 1]的两端维度）
				values_steps[step] = values.squeeze(-1).squeeze(0)
				last_features = features_tensor
				collected_steps += 1

				# 抢占慢worker（满足：其他完成数达到阈值 且 本worker已达到最少步数）
				try:
					num_done = int(num_workers_done.get("done").decode())
				except Exception:
					num_done = 0
				if (num_done >= preempt_threshold) and (collected_steps >= min_steps):
					break

			# 4) 标记本worker完成采样
			try:
				if hasattr(num_workers_done, 'add'):
					num_workers_done.add("done", 1)
				else:
					# 退化：读-改-写
					curr = int(num_workers_done.get("done").decode())
					num_workers_done.set("done", str(curr + 1).encode())
			except Exception:
				pass

			# 5) 计算GAE（全tensor、同device）
			T = collected_steps
			if T > 0:
				# bootstrap V_{T}
				with torch.no_grad():
					_, v_boot = model(last_features)
				v_boot = v_boot.squeeze(-1).squeeze(0)  # [B]
				values_tensor = torch.stack(values_steps[:T] + [v_boot], dim=0)  # [T+1, B]
				rewards_tensor = torch.zeros(T, values_tensor.shape[1], device=device)
				dones_tensor = torch.zeros(T, values_tensor.shape[1], device=device)
				advantages, returns = gae_advantages(rewards_tensor, values_tensor, dones_tensor, gamma, gae_lambda)


			# 6) 使用PPO进行更新（占位实现）
			for _ in range(n_ppo_epochs):
				for _ in range(n_ppo_batch):
					# get_batch(): 用随机批次代替
					batch = torch.randn(ppo_batch_size, batch_M, input_dim, device=device)
					action_logits, values = model(batch)
					loss = (action_logits.float().mean() - values.float().mean())
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					# DDP在backward中会自动AllReduce梯度
					optimizer.step()
			
			# 7) 在优化器更新后调用学习率调度器
			scheduler.step()

			if is_master:
				try:
					num_done = int(num_workers_done.get("done").decode())
				except Exception:
					num_done = -1
				print(f"[Round {k}] finished={num_done}/{gpu_count}, collected_steps(rank{rank})={collected_steps}")
				
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

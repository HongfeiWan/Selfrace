import os
import json

import socket

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from network import create_network

'''
推荐环境变量（无 IB 时）：
NCCL_IB_DISABLE=1,NCCL_P2P_DISABLE=0(启用 P2P)
可加 NCCL_DEBUG=INFO 验证是否走 NVLink
需要时可设 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
验证 NVLink: nvidia-smi topo -m 查看拓扑,NCCL_DEBUG=INFO 输出里会显示使用 NVLink 的通道。
'''

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

def _find_free_port() -> int:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("127.0.0.1", 0))
	addr, port = s.getsockname()
	s.close()
	return port

def setup_ddp_env(rank: int, gpu_count: int, master_addr: str, master_port: int):
	os.environ['MASTER_ADDR'] = master_addr
	os.environ['MASTER_PORT'] = str(master_port)
	os.environ['gpu_count'] = str(gpu_count)
	os.environ['RANK'] = str(rank)
	# Windows/本地优先gloo网卡
	if os.name == 'nt':
		os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')

def cleanup_ddp():
	if dist.is_initialized():
		dist.destroy_process_group()

def ddppo_worker(rank: int, gpu_count: int, config_dict: dict, master_addr: str, master_port: int, store_port: int):
	if gpu_count == 1:
		#TODO:这里写单卡训练代码，用于调试
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
		dist.init_process_group(backend=backend, gpu_count=gpu_count, rank=rank, store=store, timeout=timedelta(seconds=180))
		
        # 用PrefixStore追踪完成数量
		num_workers_done = dist.PrefixStore("num_workers_done", store)

		# 载入配置并放入policy网络模型
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
		# M 维度：优先使用 training.batch_M，否则回退到 simulator.max_agents_num，再否则用 8
		batch_M = getattr(training_cfg, 'batch_M', getattr(sim_cfg, 'max_agents_num', 150))
		preempt_ratio = getattr(training_cfg, 'preempt_ratio', 0.6)
		# 每轮采样步数：优先 training.max_experience_steps，否则回退到 training.rollout_length
		max_experience_steps = getattr(training_cfg, 'max_experience_steps', getattr(training_cfg, 'rollout_length', 1024))
		min_steps_fraction = getattr(training_cfg, 'min_steps_fraction', 0.25)
		# PPO 迭代：优先 training.n_ppo_epochs，否则回退到 training.ppo_epochs
		n_ppo_epochs = getattr(training_cfg, 'n_ppo_epochs', getattr(training_cfg, 'ppo_epochs', 2))
		n_ppo_batch = getattr(training_cfg, 'n_ppo_batch', 4)
		ppo_batch_size = getattr(training_cfg, 'ppo_batch_size', 32)

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #TODO:这里把采样换入simulator的输入、ppo换成已经写好的test_ppo就OK了！！
		
		# 每一轮迭代
		for k in range(num_iterations):
			# 本轮开始：重置完成计数
			num_workers_done.set("done", b"0")
			# rollout 收集，带抢占
			min_steps = max(1, int(max_experience_steps * min_steps_fraction))
			preempt_threshold = preempt_ratio * gpu_count
			collected_steps = 0
			input_dim = model.module.feature_encoder.total_input_dim
			for step in range(max_experience_steps):
				# collect_step(model): 这里用随机张量模拟一次环境交互
				features_tensor = torch.randn(1, batch_M, input_dim, device=device)
				with torch.no_grad():
					_ = model(features_tensor)
				collected_steps += 1
				# 抢占慢worker
				try:
					num_done = int(num_workers_done.get("done").decode())
				except Exception:
					num_done = 0
				if (num_done > preempt_threshold) and (step >= max_experience_steps / 4):
					break

			# 标记本worker完成采样
			try:
				if hasattr(num_workers_done, 'add'):
					num_workers_done.add("done", 1)
				else:
					# 退化：读-改-写
					curr = int(num_workers_done.get("done").decode())
					num_workers_done.set("done", str(curr + 1).encode())
			except Exception:
				pass

			# 使用PPO进行更新（占位实现）
			input_dim = model.module.feature_encoder.total_input_dim
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



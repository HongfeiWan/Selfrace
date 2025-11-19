import os
import sys
import json
import socket
import tempfile
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# 将 ../ 目录下的所有文件夹添加到 sys.path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
for item in os.listdir(parent_dir):
    item_path = os.path.join(parent_dir, item)
    if os.path.isdir(item_path):
        sys.path.append(item_path)

from simulator import TeraflowSimulator
from network import (
    WBoundaryNet,
    GoalsNet,
    WlaneNet,
    OtherAgentsNet,
    ConditionNet,
    VehicleStateNet,
    MLP_policy,
    MLP_value
)

class ddppo:
    @staticmethod
    def _find_free_port():
        """找到一个可用的端口号"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @staticmethod
    def setup_ddp_env(rank: int, world_size: int, master_addr: str, master_port: int):
        """设置分布式训练环境变量（必需）"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        # Windows/本地优先gloo网卡（使用 GLOO 后端时必需）
        if os.name == 'nt':
            os.environ['GLOO_DEVICE_TRANSPORT'] = 'tcp'
            os.environ.pop('GLOO_SOCKET_IFNAME', None)

    @staticmethod
    def _create_store(is_master: bool, world_size: int, master_addr: str, store_port: int):
        """
        创建分布式通信所需的 store。
        默认优先 TCPStore；若当前 PyTorch 没有 libuv 支持，则自动回退到 FileStore。
        """
        timeout = timedelta(seconds=180)
        try:
            return dist.TCPStore(master_addr, store_port, world_size, is_master, timeout=timeout)
        except RuntimeError as err:
            if "libuv" not in str(err):
                raise
            # Windows / 无 libuv 支持时使用 FileStore
            tmp_root = os.path.join(tempfile.gettempdir(), f"ddppo_store_{store_port}")
            if is_master:
                # 需要保证目录存在且为空
                if os.path.exists(tmp_root):
                    shutil.rmtree(tmp_root, ignore_errors=True)
                os.makedirs(tmp_root, exist_ok=True)
            return dist.FileStore(os.path.join(tmp_root, "filestore"), world_size)

    def ddppo_worker(self, rank, world_size, master_addr, master_port, store_port):
        """分布式训练的工作进程函数（多卡驾驶经验采集器）"""
        # TODO: 实现分布式训练逻辑
        print(f"Worker {rank}/{world_size} started on GPU {rank}")
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        # Windows 上不使用分布式，只使用单卡
        if os.name == 'nt':
            is_master = True
            store = None
            num_workers_done = None
            use_distributed = False
        else:
            # Linux 上使用分布式训练
            # 设置分布式训练环境变量
            ddppo.setup_ddp_env(rank, world_size, master_addr, master_port)
            is_master = (rank == 0)
            store = ddppo._create_store(is_master, world_size, master_addr, store_port)

            # 使用store初始化进程组
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, world_size=world_size, rank=rank, store=store, timeout=timedelta(seconds=180))

            # 用PrefixStore追踪完成数量
            num_workers_done = dist.PrefixStore("num_workers_done", store)
            use_distributed = True

        # 初始化所有神经网络并转移到设备
        w_boundary_net = WBoundaryNet(config=self.config).to(device)
        goals_net = GoalsNet(config=self.config).to(device)
        w_lane_net = WlaneNet(config=self.config).to(device)
        other_agents_net = OtherAgentsNet(config=self.config).to(device)
        condition_net = ConditionNet(config=self.config).to(device)
        vehicle_state_net = VehicleStateNet(config=self.config).to(device)
        mlp_policy = MLP_policy(config=self.config).to(device)
        mlp_value = MLP_value(config=self.config).to(device)
        
        # 使用 DDP 包装网络（仅在分布式模式下）
        if use_distributed:
            w_boundary_net = DDP(w_boundary_net, device_ids=[rank], output_device=rank)
            goals_net = DDP(goals_net, device_ids=[rank], output_device=rank)
            w_lane_net = DDP(w_lane_net, device_ids=[rank], output_device=rank)
            other_agents_net = DDP(other_agents_net, device_ids=[rank], output_device=rank)
            condition_net = DDP(condition_net, device_ids=[rank], output_device=rank)
            vehicle_state_net = DDP(vehicle_state_net, device_ids=[rank], output_device=rank)
            mlp_policy = DDP(mlp_policy, device_ids=[rank], output_device=rank)
            mlp_value = DDP(mlp_value, device_ids=[rank], output_device=rank)
        if is_master:
            print(f"All networks initialized and moved to device {device}")
        simulator = TeraflowSimulator(config=self.config, device=device)
        
        # 收集 policy 网络的参数（特征提取网络 + policy 网络）
        policy_params = []
        policy_params.extend(w_boundary_net.module.parameters() if isinstance(w_boundary_net, DDP) else w_boundary_net.parameters())
        policy_params.extend(goals_net.module.parameters() if isinstance(goals_net, DDP) else goals_net.parameters())
        policy_params.extend(w_lane_net.module.parameters() if isinstance(w_lane_net, DDP) else w_lane_net.parameters())
        policy_params.extend(other_agents_net.module.parameters() if isinstance(other_agents_net, DDP) else other_agents_net.parameters())
        policy_params.extend(condition_net.module.parameters() if isinstance(condition_net, DDP) else condition_net.parameters())
        policy_params.extend(vehicle_state_net.module.parameters() if isinstance(vehicle_state_net, DDP) else vehicle_state_net.parameters())
        policy_params.extend(mlp_policy.module.parameters() if isinstance(mlp_policy, DDP) else mlp_policy.parameters())
        
        # 收集 value 网络的参数（特征提取网络 + value 网络）
        value_params = []
        value_params.extend(w_boundary_net.module.parameters() if isinstance(w_boundary_net, DDP) else w_boundary_net.parameters())
        value_params.extend(goals_net.module.parameters() if isinstance(goals_net, DDP) else goals_net.parameters())
        value_params.extend(w_lane_net.module.parameters() if isinstance(w_lane_net, DDP) else w_lane_net.parameters())
        value_params.extend(other_agents_net.module.parameters() if isinstance(other_agents_net, DDP) else other_agents_net.parameters())
        value_params.extend(condition_net.module.parameters() if isinstance(condition_net, DDP) else condition_net.parameters())
        value_params.extend(vehicle_state_net.module.parameters() if isinstance(vehicle_state_net, DDP) else vehicle_state_net.parameters())
        value_params.extend(mlp_value.module.parameters() if isinstance(mlp_value, DDP) else mlp_value.parameters())
        
        # 创建分离的优化器和调度器
        optimizer_policy = optim.Adam(policy_params, lr=self.learning_rate)
        optimizer_value = optim.Adam(value_params, lr=self.learning_rate)
        scheduler_policy = optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=self.num_iterations, eta_min=0.0)
        scheduler_value = optim.lr_scheduler.CosineAnnealingLR(optimizer_value, T_max=self.num_iterations, eta_min=0.0)
        
        if is_master:
            print(f"Policy optimizer created with learning_rate={self.learning_rate}, num_iterations={self.num_iterations}")
            print(f"Value optimizer created with learning_rate={self.learning_rate}, num_iterations={self.num_iterations}")
            print(f"Policy parameters: {sum(p.numel() for p in policy_params):,}")
            print(f"Value parameters: {sum(p.numel() for p in value_params):,}")
        
        for k in range(self.num_iterations):
            print(f"Iteration {k} started")
            # 重置完成计数（仅在 Linux 分布式模式下）
            if use_distributed and num_workers_done is not None:
                num_workers_done.set("done", b"0")
            
            # 初始化buffer
            initial_observation, d, theta_f = simulator.reset()
            # 初始化本轮 rollout buffer（长度 T = rollout_steps），用于缓存采样到的数据
            buffer_T = self.rollout_steps
            states_buffer = [None] * buffer_T
            rewards_buffer = [None] * buffer_T
            dones_buffer = [None] * buffer_T
            values_buffer = [None] * buffer_T
            old_log_probs_buffer = [None] * buffer_T
            actions_buffer = [None] * buffer_T
            buffer_step_count = 0  # 当前已写入的时间步计数

            if buffer_T > 0:
                states_buffer[buffer_step_count] = initial_observation
            
            # 从 simulator 获取网络需要的各个输入
            obs_gen = simulator.observation_generator
            # 1. 从 observation_generator 的 self 属性中直接读取（已在 reset() 的 generate() 中计算并保存）
            w_boundaries_local = obs_gen.last_w_boundaries_local  # (B, M, K, 2) - 用于 WBoundaryNet
            local_state = obs_gen.last_local_state  # (B, M, 7) - 用于 VehicleStateNet
            neighbors_local = obs_gen.last_neighbors_local  # (B, M, K, 7) - 用于 OtherAgentsNet
            # 2. 获取 w_lanes_local_with_goal_distances (B, M, K, 3) - 用于 WlaneNet
            w_lanes_local_with_goal_distances = simulator.w_lanes_local_with_goal_distances
            # 3. 获取 agents_path_plans 的世界坐标 (B, M, L, 3) - 用于 GoalsNet
            # 从 path_planner 获取路径点的世界坐标
            path_centers = simulator.path_planner.get_w_lane_centers_by_id(simulator.agents_path_plans)  # (B, M, L, 2)
            # 添加一个零的 z 维度使其成为 (B, M, L, 3)
            agents_path_plans_world = torch.cat([
                path_centers,
                torch.zeros_like(path_centers[..., :1])
            ], dim=-1)
            # 4. 获取 curvature (B, M) - 用于 ConditionNet
            curvature = obs_gen.curvature if hasattr(obs_gen, 'curvature') else torch.zeros(
                (simulator.num_envs, simulator.max_agents), device=device
            )
            # 5. 获取 reward_params (B, M, R) - 用于 ConditionNet
            reward_params = simulator.reward_calculator.sampled_params  # (B, M, R)
            # 6. 获取 wheelbase (B*M,) - 用于 ConditionNet
            # 从 dynamics_model 获取
            if hasattr(simulator.dynamics_model, 'vehicle_params') and 'wheelbase' in simulator.dynamics_model.vehicle_params:
                wheelbase_flat = simulator.dynamics_model.vehicle_params['wheelbase']  # (B*M,)
                B, M = simulator.agents_state.shape[:2]
                wheelbase = wheelbase_flat.view(B, M)  # (B, M)
            else:
                B, M = simulator.agents_state.shape[:2]
                wheelbase = torch.zeros((B, M), device=device)
            # 7. 获取 c_throttle, c_steer, c_acc, c_vel - 用于 ConditionNet
            # 这些参数通常从 reward_calculator 或 dynamics_model 获取
            # 如果不存在，使用零值
            B, M = simulator.agents_state.shape[:2]
            c_throttle = torch.zeros((B, M), device=device)
            c_steer = torch.zeros((B, M), device=device)
            c_acc = torch.zeros((B, M), device=device)
            c_vel = torch.zeros((B, M), device=device)
            # 如果有 extend_state，可以从那里获取一些信息
            if hasattr(simulator, 'extend_state') and simulator.extend_state is not None:
                # extend_state: (B, M, 10) [x, y, heading, speed, along, alat, along_jerk, alat_jerk, theta_f, d]
                c_acc = simulator.extend_state[..., 4]  # along 加速度
                # c_vel 可以从 speed 获取
                c_vel = simulator.extend_state[..., 3]  # speed
            # 现在将所有输入传入网络
            with torch.no_grad():
                # WBoundaryNet
                w_boundary_features = w_boundary_net(w_boundaries_local)  # (B, M, encoded_dim)
                # GoalsNet
                goals_features = goals_net(simulator.agents_state, agents_path_plans_world)  # (B, M, encoded_dim)
                # WlaneNet
                w_lane_features = w_lane_net(w_lanes_local_with_goal_distances)  # (B, M, encoded_dim)
                # OtherAgentsNet
                other_agents_features = other_agents_net(neighbors_local)  # (B, M, encoded_dim)
                # ConditionNet
                condition_features = condition_net(
                    curvature, c_throttle, c_steer, c_acc, c_vel, reward_params, wheelbase
                )  # (B, M, encoded_dim)
                # VehicleStateNet
                vehicle_state_features = vehicle_state_net(local_state)  # (B, M, encoded_dim)
                # MLP_policy - 输出动作概率分布
                action_probs = mlp_policy(
                    w_boundary_features,
                    goals_features,
                    w_lane_features,
                    other_agents_features,
                    condition_features,
                    vehicle_state_features,
                )  # (B, M, action_dim)
                # MLP_value - 输出价值估计
                values = mlp_value(
                    w_boundary_features,
                    goals_features,
                    w_lane_features,
                    other_agents_features,
                    condition_features,
                    vehicle_state_features,
                )  # (B, M, 1)
            if is_master:
                print(f"Iteration {k}: Networks forward pass completed")
                print(f"  Action probs shape: {action_probs.shape}, Values shape: {values.shape}")

    def __init__(self, config_path):
        # 加载配置文件
        if isinstance(config_path, str):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = config_path
        
        # 从 training 配置中读取参数
        training_config = self.config.get('training', {})
        self.gamma = float(training_config.get('gamma'))
        self.gae_lambda = float(training_config.get('lambda'))
        self.learning_rate = float(training_config.get('learning_rate'))
        self.num_iterations = int(training_config.get('num_iterations'))
        # 每轮采样的时间步长度（若未配置则使用 32）
        self.rollout_steps = int(training_config.get('rollout_steps'))
        self.max_episode_steps = int(training_config.get('max_episode_steps'))
        
        # 检查 GPU 信息并启动分布式进程
        cuda_available, cuda_ranks = self.check_gpu_info(print_info=True)
        
        if not cuda_available or not cuda_ranks:
            raise RuntimeError("没有可用的CUDA设备")

        gpu_count = len(cuda_ranks)
        master_addr = '127.0.0.1'
        master_port = self._find_free_port()
        store_port = self._find_free_port()
        
        # Windows需要spawn
        mp.set_start_method('spawn', force=True)
        
        # 将rank映射到CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(r) for r in cuda_ranks)
        ctx = mp.get_context('spawn')
        
        processes = []
        for rank in range(gpu_count):
            p = ctx.Process(
                target=self.ddppo_worker,
                args=(rank, gpu_count, master_addr, master_port, store_port))
            p.start()
            processes.append(p)
        self.processes = processes
        self.gpu_count = gpu_count
        self.cuda_ranks = cuda_ranks
        
        # 等待所有进程完成
        for p in processes:
            p.join()

    def check_gpu_info(self, print_info: bool = True, **kwargs):
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
        if torch.cuda.is_available():
            # CUDA版本
            cuda_version = torch.version.cuda
            log(f"CUDA 版本: {cuda_version}")
            # GPU数量（格式：cuda:0,1,...）
            gpu_count = torch.cuda.device_count()
            gpu_list = ','.join([f"cuda:{i}" for i in range(gpu_count)])
            log(f"GPU 数量: {gpu_list}")
            # 显存总大小
            total_memory = 0
            for i in range(gpu_count):
                total_memory += torch.cuda.get_device_properties(i).total_memory
            total_memory_gb = total_memory / (1024**3)
            log(f"显存总大小: {total_memory_gb:.2f} GB")
            return True, list(range(gpu_count))
        else:
            log("CUDA 不可用")
            return False, []

    def GAE_calculate(self, rewards, values, dones, gamma=None, gae_lambda=None):
        """
        使用向量化操作计算 GAE (Generalized Advantage Estimation)，适合 GPU 并行计算。
        Args:
            rewards: (B, M, T) 奖励张量，T 是时间步数，B 是批次大小，M 是智能体数
            values: (B, M, T+1) 价值函数估计，包含最后一个状态的 bootstrap 值
            dones: (B, M, T) done 标志
            gamma: 折扣因子，如果为 None 则从 config 的 training 部分读取
            gae_lambda: GAE lambda 参数，如果为 None 则从 config 的 training 部分读取
        Returns:
            advantages: (B, M, T) 优势函数值
        """
        if gamma is None:
            gamma = self.gamma
        if gae_lambda is None:
            gae_lambda = self.gae_lambda
        B, M, T = rewards.shape
        # 提取当前和下一个状态的价值
        values_t = values[..., :-1]  # (B, M, T)
        next_values_t = values[..., 1:]  # (B, M, T)
        # 计算 TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        td_errors = rewards + gamma * next_values_t * (1 - dones.float()) - values_t
        # 计算 GAE: A_t = δ_t + (γ * λ) * A_{t+1} * (1 - done_t)
        # 使用向量化的反向累积实现
        gae_factor = gamma * gae_lambda
        # 翻转 TD errors 和 dones，以便从后往前处理
        td_flipped = torch.flip(td_errors, dims=[-1])  # (B, M, T)
        dones_flipped = torch.flip(dones.float(), dims=[-1])  # (B, M, T)
        # 累积折扣因子：每个时间步的 (1 - done) * (γ * λ)
        discount_factors = (1 - dones_flipped) * gae_factor  # (B, M, T)
        # 从后往前累积 GAE
        advantages_flipped = torch.zeros_like(td_flipped)
        advantages_flipped[..., -1] = td_flipped[..., -1]  # 最后一个时间步
        # 从倒数第二个开始，向量化累积
        for t in range(T - 2, -1, -1):
            advantages_flipped[..., t] = td_flipped[..., t] + discount_factors[..., t] * advantages_flipped[..., t + 1]
        # 翻转回原始顺序
        advantages = torch.flip(advantages_flipped, dims=[-1])
        return advantages

if __name__ == "__main__":
    trainer = ddppo(config_path="configs/default_config.json")
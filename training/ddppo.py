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
import torch.nn.functional as F
from torch.distributions import Categorical
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
            current_observation = initial_observation

            for episode_step in range(self.max_episode_steps):
                components = self._collect_observation_components(simulator, current_observation, device)
                snapshot = self._clone_components(components)
                if buffer_step_count >= buffer_T:
                    raise RuntimeError("Buffer overflow detected before PPO update.")
                states_buffer[buffer_step_count] = snapshot

                with torch.no_grad():
                    encoded_features = self._encode_features(
                        components,
                        w_boundary_net,
                        goals_net,
                        w_lane_net,
                        other_agents_net,
                        condition_net,
                        vehicle_state_net,
                    )
                    action_probs = mlp_policy(*encoded_features)
                    values = mlp_value(*encoded_features)

                actions, log_probs, _ = self._sample_actions(action_probs)
                next_observation, rewards, dones, _ = simulator.step(actions.unsqueeze(-1))

                rewards_buffer[buffer_step_count] = rewards.detach().clone()
                dones_buffer[buffer_step_count] = dones.detach().clone()
                values_buffer[buffer_step_count] = values.detach().clone()
                old_log_probs_buffer[buffer_step_count] = log_probs.detach().clone()
                actions_buffer[buffer_step_count] = actions.detach().clone()
                buffer_step_count += 1
                current_observation = next_observation

                should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)

                if should_update and buffer_step_count > 0:
                    valid_mask = torch.zeros(buffer_T, dtype=torch.bool, device=device)
                    valid_mask[:buffer_step_count] = True
                    bootstrap_components = self._collect_observation_components(simulator, current_observation, device)
                    with torch.no_grad():
                        bootstrap_encoded = self._encode_features(
                            bootstrap_components,
                            w_boundary_net,
                            goals_net,
                            w_lane_net,
                            other_agents_net,
                            condition_net,
                            vehicle_state_net,
                        )
                        bootstrap_value = mlp_value(*bootstrap_encoded).detach()

                    self._ppo_update_from_buffer(
                        states_buffer[:buffer_step_count],
                        rewards_buffer[:buffer_step_count],
                        dones_buffer[:buffer_step_count],
                        values_buffer[:buffer_step_count],
                        old_log_probs_buffer[:buffer_step_count],
                        actions_buffer[:buffer_step_count],
                        valid_mask,
                        bootstrap_value,
                        w_boundary_net,
                        goals_net,
                        w_lane_net,
                        other_agents_net,
                        condition_net,
                        vehicle_state_net,
                        mlp_policy,
                        mlp_value,
                        optimizer_policy,
                        optimizer_value,
                        device,
                        is_master,
                    )

                    states_buffer = [None] * buffer_T
                    rewards_buffer = [None] * buffer_T
                    dones_buffer = [None] * buffer_T
                    values_buffer = [None] * buffer_T
                    old_log_probs_buffer = [None] * buffer_T
                    actions_buffer = [None] * buffer_T
                    buffer_step_count = 0

            scheduler_policy.step()
            scheduler_value.step()

    def _collect_observation_components(self, simulator, observation, device):
        """收集当前环境状态下构建网络输入所需的全部数据。"""
        components = {}
        obs_gen = simulator.observation_generator
        B, M = simulator.agents_state.shape[:2]

        def _ensure_tensor(tensor, shape=None, fill_value=0.0):
            if tensor is None:
                if shape is None:
                    raise RuntimeError("Missing observation component without fallback shape.")
                return torch.full(shape, fill_value, device=device)
            return tensor.to(device)

        components["observation"] = observation.to(device) if observation is not None else None
        components["w_boundaries_local"] = _ensure_tensor(obs_gen.last_w_boundaries_local)
        components["local_state"] = _ensure_tensor(obs_gen.last_local_state)
        components["neighbors_local"] = _ensure_tensor(obs_gen.last_neighbors_local)
        components["w_lanes_local_with_goal_distances"] = _ensure_tensor(simulator.w_lanes_local_with_goal_distances)
        components["agents_state"] = simulator.agents_state.to(device)

        path_centers = simulator.path_planner.get_w_lane_centers_by_id(simulator.agents_path_plans)
        agents_path_plans_world = torch.cat(
            [path_centers, torch.zeros_like(path_centers[..., :1])],
            dim=-1,
        )
        components["agents_path_plans_world"] = agents_path_plans_world.to(device)

        curvature = getattr(obs_gen, "curvature", None)
        components["curvature"] = _ensure_tensor(curvature, (B, M))
        components["reward_params"] = _ensure_tensor(simulator.reward_calculator.sampled_params)

        wheelbase = torch.zeros((B, M), device=device)
        if hasattr(simulator.dynamics_model, "vehicle_params"):
            vehicle_params = simulator.dynamics_model.vehicle_params
            if isinstance(vehicle_params, dict) and "wheelbase" in vehicle_params:
                wheelbase = vehicle_params["wheelbase"].view(B, M).to(device)
        components["wheelbase"] = wheelbase

        c_throttle = torch.zeros((B, M), device=device)
        c_steer = torch.zeros((B, M), device=device)
        c_acc = torch.zeros((B, M), device=device)
        c_vel = torch.zeros((B, M), device=device)
        if hasattr(simulator, "extend_state") and simulator.extend_state is not None:
            c_acc = simulator.extend_state[..., 4].to(device)
            c_vel = simulator.extend_state[..., 3].to(device)
        components["c_throttle"] = c_throttle
        components["c_steer"] = c_steer
        components["c_acc"] = c_acc
        components["c_vel"] = c_vel
        return components

    @staticmethod
    def _clone_components(components):
        snapshot = {}
        for key, value in components.items():
            if torch.is_tensor(value):
                snapshot[key] = value.detach().clone()
            else:
                snapshot[key] = value
        return snapshot

    def _encode_features(
        self,
        components,
        w_boundary_net,
        goals_net,
        w_lane_net,
        other_agents_net,
        condition_net,
        vehicle_state_net):
        w_boundary_features = w_boundary_net(components["w_boundaries_local"])
        goals_features = goals_net(components["agents_state"], components["agents_path_plans_world"])
        w_lane_features = w_lane_net(components["w_lanes_local_with_goal_distances"])
        other_agents_features = other_agents_net(components["neighbors_local"])
        condition_features = condition_net(
            components["curvature"],
            components["c_throttle"],
            components["c_steer"],
            components["c_acc"],
            components["c_vel"],
            components["reward_params"],
            components["wheelbase"],
        )
        vehicle_state_features = vehicle_state_net(components["local_state"])
        return (
            w_boundary_features,
            goals_features,
            w_lane_features,
            other_agents_features,
            condition_features,
            vehicle_state_features,
        )

    def _sample_actions(self, action_probs):
        """依据策略分布采样动作，并返回动作及对应的 log_prob 与熵。"""
        B, M, A = action_probs.shape
        probs = torch.clamp(action_probs, min=1e-10).view(B * M, A)
        dist = Categorical(probs)
        actions_flat = dist.sample()
        log_probs = dist.log_prob(actions_flat).view(B, M)
        entropy = dist.entropy().view(B, M)
        actions = actions_flat.view(B, M)
        return actions, log_probs, entropy

    def _compute_log_probs(self, action_probs, actions):
        """根据给定的动作重新计算 log_prob 与熵。"""
        B, M, A = action_probs.shape
        probs = torch.clamp(action_probs, min=1e-10).view(B * M, A)
        dist = Categorical(probs)
        actions_flat = actions.view(B * M)
        log_probs = dist.log_prob(actions_flat).view(B, M)
        entropy = dist.entropy().view(B, M)
        return log_probs, entropy

    def _ppo_update_from_buffer(
        self,
        states_buffer,
        rewards_buffer,
        dones_buffer,
        values_buffer,
        old_log_probs_buffer,
        actions_buffer,
        valid_mask,
        bootstrap_value,
        w_boundary_net,
        goals_net,
        w_lane_net,
        other_agents_net,
        condition_net,
        vehicle_state_net,
        mlp_policy,
        mlp_value,
        optimizer_policy,
        optimizer_value,
        device,
        is_master):
        """使用缓冲区中的 rollout 数据执行一次 PPO 更新。"""
        if valid_mask is None or valid_mask.numel() == 0:
            return

        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return
        idx_list = valid_indices.tolist()

        states_buffer = [states_buffer[i] for i in idx_list]
        rewards_buffer = [rewards_buffer[i] for i in idx_list]
        dones_buffer = [dones_buffer[i] for i in idx_list]
        values_buffer = [values_buffer[i] for i in idx_list]
        old_log_probs_buffer = [old_log_probs_buffer[i] for i in idx_list]
        actions_buffer = [actions_buffer[i] for i in idx_list]

        rewards = torch.stack(rewards_buffer, dim=0).to(device)
        dones = torch.stack(dones_buffer, dim=0).to(device).float()
        values = torch.stack(values_buffer, dim=0).to(device).squeeze(-1)
        old_log_probs = torch.stack(old_log_probs_buffer, dim=0).to(device)
        actions = torch.stack(actions_buffer, dim=0).to(device)

        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=device)
        returns = torch.zeros_like(rewards, device=device)

        rewards_bmt = rewards.permute(1, 2, 0)              # (B, M, T)
        values_bmt = values.permute(1, 2, 0)                # (B, M, T)
        dones_bmt = dones.permute(1, 2, 0)                  # (B, M, T)
        bootstrap_bmt = bootstrap_value.to(device).squeeze(-1)  # (B, M)
        values_with_bootstrap = torch.cat(
            [values_bmt, bootstrap_bmt.unsqueeze(-1)],
            dim=-1,
        )  # (B, M, T+1)

        advantages_bmt = self.GAE_calculate(
            rewards_bmt,
            values_with_bootstrap,
            dones_bmt,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )  # (B, M, T)

        advantages = advantages_bmt.permute(2, 0, 1)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        new_log_probs_list = []
        new_values_list = []
        entropy_list = []
        
        for snapshot, actions_t in zip(states_buffer, actions):
            components = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in snapshot.items()}
            encoded = self._encode_features(
                components,
                w_boundary_net,
                goals_net,
                w_lane_net,
                other_agents_net,
                condition_net,
                vehicle_state_net,
            )
            action_probs = mlp_policy(*encoded)
            current_values = mlp_value(*encoded).squeeze(-1)
            log_probs_now, entropy_now = self._compute_log_probs(action_probs, actions_t)
            new_log_probs_list.append(log_probs_now)
            new_values_list.append(current_values)
            entropy_list.append(entropy_now)

        new_log_probs = torch.stack(new_log_probs_list, dim=0)
        new_values = torch.stack(new_values_list, dim=0)
        entropies = torch.stack(entropy_list, dim=0)

        ratio = torch.exp(new_log_probs - old_log_probs)
        advantages_detached = advantages.detach()
        surr1 = ratio * advantages_detached
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages_detached
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        entropy_bonus = entropies.mean()
        total_policy_loss = policy_loss - self.entropy_coef * entropy_bonus

        optimizer_policy.zero_grad()
        total_policy_loss.backward()
        optimizer_policy.step()

        value_loss = F.mse_loss(new_values, returns) * self.value_coef
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        if is_master:
            print(
                f"PPO update -> policy_loss: {policy_loss.item():.6f}, "
                f"value_loss: {value_loss.item():.6f}, entropy: {entropy_bonus.item():.6f}"
            )

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
        # 每轮采样的时间步长度（默认 128），以及单个 episode 的最大步数
        self.rollout_steps = int(training_config.get('rollout_steps', 128))
        self.max_episode_steps = int(training_config.get('max_episode_steps', 1024))
        self.ppo_clip = float(training_config.get('ppo_clip', 0.2))
        self.entropy_coef = float(training_config.get('entropy_coef', 0.0))
        self.value_coef = float(training_config.get('value_coef', 0.5))
        
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
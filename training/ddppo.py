import os
import sys
import json
import socket
import tempfile
import shutil
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
import swanlab

# 将 ../ 目录下的所有文件夹添加到 sys.path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
for item in os.listdir(parent_dir):
    item_path = os.path.join(parent_dir, item)
    if os.path.isdir(item_path):
        sys.path.append(item_path)

from simulator import TeraflowSimulator
from network import (
    SimplePolicyNet,
    SimpleValueNet,
    convert_path_world_to_ego)

class ddppo:
    @staticmethod
    def decompose_observation(observation: torch.Tensor, config) -> tuple:
        """
        将展平的 observation 拆解为网络需要的各个组件（纯函数，只做 view 操作，不复制数据）
        
        Args:
            observation: 形状为 (B, M, total_obs_dim) 的观测张量
            config: 配置对象（可以是 dict 或 SimpleNamespace）
        
        Returns:
            tuple: (agents_state, neighbors_local, w_lanes_local, w_boundaries_local)
                - agents_state: (B, M, 7) - 智能体状态 [x, y, yaw, speed, length, width, active]
                - neighbors_local: (B, M, K, 7) - 邻居相对状态 [dx, dy, vx, vy, length, width, active]
                - w_lanes_local: (B, M, N_lanes, 2) - 车道线相对坐标 [dx, dy]
                - w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标 [dx, dy]
        """
        batch_size, max_agents, total_obs_dim = observation.shape
        
        # 从配置中获取维度信息（支持 dict 和 SimpleNamespace）
        if isinstance(config, dict):
            simulator_config = config['simulator']
            obs_config = simulator_config['observation']
            local_state_dim = obs_config['local_state_dim']  # 7
            neighbor_feature_dim = obs_config['neighbor_feature_dim']  # 7
            # 注意：w_lanes_local 和 w_boundaries_local 在展平时都是 2 维 (dx, dy)
            # 但配置中 w_lane_feature_dim 可能是 3（包含 angle），实际展平时是 2
            w_lane_feature_dim = 2  # 实际展平时的维度是 2 (dx, dy)
            boundary_feature_dim = obs_config['boundary_feature_dim']  # 2
            num_neighbors = obs_config['num_neighbors']  # 20
            num_w_lanes = obs_config['num_w_lanes']  # 80
            num_w_boundaries = obs_config['num_w_boundaries']  # 80
        else:
            simulator_config = config.simulator
            local_state_dim = simulator_config.observation.local_state_dim  # 7
            neighbor_feature_dim = simulator_config.observation.neighbor_feature_dim  # 7
            # 注意：w_lanes_local 和 w_boundaries_local 在展平时都是 2 维 (dx, dy)
            w_lane_feature_dim = 2  # 实际展平时的维度是 2 (dx, dy)
            boundary_feature_dim = simulator_config.observation.boundary_feature_dim  # 2
            num_neighbors = simulator_config.observation.num_neighbors  # 20
            num_w_lanes = simulator_config.observation.num_w_lanes  # 80
            num_w_boundaries = simulator_config.observation.num_w_boundaries  # 80
        
        # 计算各部分在观测向量中的位置
        local_state_size = local_state_dim
        neighbors_size = num_neighbors * neighbor_feature_dim
        w_lanes_size = num_w_lanes * w_lane_feature_dim  # 使用实际展平时的维度 2
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
        w_lanes_local = w_lanes_flat.view(batch_size, max_agents, num_w_lanes, w_lane_feature_dim)  # (B, M, N_lanes, 2)
        
        # 4. 提取w_boundaries_local
        w_boundaries_start = w_lanes_end
        w_boundaries_flat = observation[:, :, w_boundaries_start:]  # (B, M, N_boundaries*2)
        w_boundaries_local = w_boundaries_flat.view(batch_size, max_agents, num_w_boundaries, boundary_feature_dim)  # (B, M, N_boundaries, 2)
        
        return agents_state, neighbors_local, w_lanes_local, w_boundaries_local

    def _pad_or_trim(self, tensor, target_dim: int, B: int, M: int, device, dtype) -> torch.Tensor:
        if target_dim <= 0:
            return torch.zeros(B, M, 0, device=device, dtype=dtype)
        if tensor is None:
            return torch.zeros(B, M, target_dim, device=device, dtype=dtype)
        if tensor.dim() == 2:
            tensor = tensor.view(B, M, -1)
        elif tensor.dim() == 3 and tensor.shape[0] == B and tensor.shape[1] == M:
            pass
        else:
            tensor = tensor.reshape(B, M, -1)
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            return tensor[..., :target_dim]
        pad = torch.zeros(
            B,
            M,
            target_dim - current_dim,
            device=device,
            dtype=dtype,
        )
        return torch.cat([tensor, pad], dim=-1)

    def _reshape_reward_params(self, reward_params, B, M, device, dtype):
        if reward_params is None or (hasattr(reward_params, "numel") and reward_params.numel() == 0):
            tensor = torch.zeros(B, M, 10, device=device, dtype=dtype)
            return self._normalize_reward_params(tensor)
        tensor = reward_params.to(device)
        if tensor.dim() == 2 and tensor.shape[0] == B * M:
            tensor = tensor.view(B, M, -1)
        elif tensor.dim() == 2 and tensor.shape[0] == B and tensor.shape[1] == 10:
            tensor = tensor.unsqueeze(1).expand(-1, M, -1)
        elif tensor.dim() == 3 and tensor.shape[0] == B and tensor.shape[1] == M:
            pass
        else:
            tensor = tensor.view(B, M, -1)
        if tensor.shape[-1] < 10:
            pad = torch.zeros(B, M, 10 - tensor.shape[-1], device=device, dtype=dtype)
            tensor = torch.cat([tensor, pad], dim=-1)
        return self._normalize_reward_params(tensor[..., :10])

    def _normalize_reward_params(self, reward_params: torch.Tensor) -> torch.Tensor:
        B, M, _ = reward_params.shape
        mins = self.reward_param_mins.view(1, 1, -1).to(reward_params.device).expand(B, M, -1)
        maxs = self.reward_param_maxs.view(1, 1, -1).to(reward_params.device).expand(B, M, -1)
        denoms = torch.clamp(maxs - mins, min=1e-6)
        normalized = (reward_params - mins) / denoms
        center_bias_idx = 7
        center_bias = torch.clamp(normalized[..., center_bias_idx:center_bias_idx+1] * 2.0 - 1.0, -1.0, 1.0)
        before = torch.clamp(normalized[..., :center_bias_idx], 0.0, 1.0)
        after = torch.clamp(normalized[..., center_bias_idx+1:], 0.0, 1.0)
        return torch.cat([before, center_bias, after], dim=-1)

    def _normalize_relative(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor / self.observation_horizon, min=-1.0, max=1.0)

    def _normalize_speed(self, speed: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(speed)
        mask_neg = (speed >= self.speed_min) & (speed < self.speed_mid)
        normalized = torch.where(
            mask_neg,
            (speed - self.speed_mid) / (self.speed_mid - self.speed_min + 1e-6),
            normalized,
        )
        mask_pos = (speed >= self.speed_mid) & (speed <= self.speed_max)
        normalized = torch.where(
            mask_pos,
            (speed - self.speed_mid) / (self.speed_max - self.speed_mid + 1e-6),
            normalized,
        )
        return torch.clamp(normalized, min=-1.0, max=1.0)

    def _normalize_dimension(self, value: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        denom = max(v_max - v_min, 1e-6)
        normalized = (value - v_min) / denom
        return torch.clamp(normalized, min=0.0, max=1.0)

    def _build_local_state_features(self, agents_state: torch.Tensor) -> torch.Tensor:
        yaw = agents_state[..., 2]
        speed = agents_state[..., 3]
        length = agents_state[..., 4]
        width = agents_state[..., 5]
        active = agents_state[..., 6]
        zeros = torch.zeros_like(agents_state[..., :2])
        cos_yaw = torch.cos(yaw).unsqueeze(-1)
        sin_yaw = torch.sin(yaw).unsqueeze(-1)
        speed_norm = self._normalize_speed(speed).unsqueeze(-1)
        length_norm = self._normalize_dimension(length, self.l_min, self.l_max).unsqueeze(-1)
        width_norm = self._normalize_dimension(width, self.w_min, self.w_max).unsqueeze(-1)
        active = active.unsqueeze(-1)
        return torch.cat([zeros, cos_yaw, sin_yaw, speed_norm, width_norm, length_norm, active], dim=-1)

    def _normalize_neighbor_features(self, neighbors: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(neighbors)
        normalized[..., :2] = self._normalize_relative(neighbors[..., :2])
        dvx = neighbors[..., 2]
        dvy = neighbors[..., 3]
        mask_neg = (dvx >= self.dvx_min) & (dvx < self.dvx_mid)
        normalized[..., 2] = torch.where(
            mask_neg,
            (dvx - self.dvx_mid) / (self.dvx_mid - self.dvx_min + 1e-6),
            torch.zeros_like(dvx),
        )
        mask_pos = (dvx >= self.dvx_mid) & (dvx <= self.dvx_max)
        normalized[..., 2] = torch.where(
            mask_pos,
            (dvx - self.dvx_mid) / (self.dvx_max - self.dvx_mid + 1e-6),
            normalized[..., 2],
        )
        normalized[..., 2] = torch.clamp(normalized[..., 2], min=-1.0, max=1.0)

        mask_neg = (dvy >= self.dvx_min) & (dvy < self.dvx_mid)
        normalized[..., 3] = torch.where(
            mask_neg,
            (dvy - self.dvx_mid) / (self.dvx_mid - self.dvx_min + 1e-6),
            torch.zeros_like(dvy),
        )
        mask_pos = (dvy >= self.dvx_mid) & (dvy <= self.dvx_max)
        normalized[..., 3] = torch.where(
            mask_pos,
            (dvy - self.dvx_mid) / (self.dvx_max - self.dvx_mid + 1e-6),
            normalized[..., 3],
        )
        normalized[..., 3] = torch.clamp(normalized[..., 3], min=-1.0, max=1.0)

        normalized[..., 4] = self._normalize_dimension(neighbors[..., 4], self.w_min, self.w_max)
        normalized[..., 5] = self._normalize_dimension(neighbors[..., 5], self.l_min, self.l_max)
        normalized[..., 6] = neighbors[..., 6]
        return normalized

    def _compose_feature_tensor(
        self,
        agents_state: torch.Tensor,
        neighbors_local: torch.Tensor,
        w_lanes_local: torch.Tensor,
        w_boundaries_local: torch.Tensor,
        path_plan_features: torch.Tensor,
        reward_params,
        stop_lines) -> torch.Tensor:
        B, M, _ = agents_state.shape
        device = agents_state.device
        dtype = agents_state.dtype

        if neighbors_local.device != device:
            neighbors_local = neighbors_local.to(device)
        if w_lanes_local.device != device:
            w_lanes_local = w_lanes_local.to(device)
        if w_boundaries_local.device != device:
            w_boundaries_local = w_boundaries_local.to(device)

        segments = []

        local_feats = self._build_local_state_features(agents_state)
        segments.append(self._pad_or_trim(local_feats, self.local_state_feature_dim, B, M, device, dtype))

        path_flat = None
        if self.path_plan_feature_dim > 0 and path_plan_features is not None and path_plan_features.numel() > 0:
            path_world = path_plan_features.to(device)
            path_local = convert_path_world_to_ego(agents_state, path_world, self.observation_horizon)
            path_local = self._normalize_relative(path_local)
            path_flat = path_local.view(B, M, -1)
        segments.append(self._pad_or_trim(path_flat, self.path_plan_feature_dim, B, M, device, dtype))

        reward_tensor = self._reshape_reward_params(reward_params, B, M, device, dtype)
        segments.append(self._pad_or_trim(reward_tensor, self.reward_param_dim, B, M, device, dtype))

        boundary_norm = self._normalize_relative(w_boundaries_local)
        boundary_tensor = boundary_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(boundary_tensor, self.w_boundary_feature_dim, B, M, device, dtype))

        w_lanes_norm = self._normalize_relative(w_lanes_local)
        w_lanes_tensor = w_lanes_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(w_lanes_tensor, self.w_lane_feature_dim, B, M, device, dtype))

        if stop_lines is not None:
            stop_lines_flat = stop_lines.to(device=device, dtype=dtype).reshape(B, M, -1)
        else:
            stop_lines_flat = None
        segments.append(self._pad_or_trim(stop_lines_flat, self.stop_line_feature_dim, B, M, device, dtype))

        neighbors_norm = self._normalize_neighbor_features(neighbors_local)
        neighbors_tensor = neighbors_norm.reshape(B, M, -1)
        segments.append(self._pad_or_trim(neighbors_tensor, self.neighbor_feature_dim, B, M, device, dtype))

        features = torch.cat(segments, dim=-1)
        if features.shape[-1] != self.total_feature_dim:
            raise ValueError(
                f"组合后的特征维度 {features.shape[-1]} 与预期 {self.total_feature_dim} 不一致，请检查配置"
            )
        return features

    def _save_checkpoint(
        self,
        iteration: int,
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        scheduler_policy,
        scheduler_value,
    ) -> None:
        if self.checkpoint_interval <= 0:
            return
        policy_state = policy_net.module.state_dict() if isinstance(policy_net, DDP) else policy_net.state_dict()
        value_state = value_net.module.state_dict() if isinstance(value_net, DDP) else value_net.state_dict()
        checkpoint = {
            "iteration": iteration,
            "policy_state": policy_state,
            "value_state": value_state,
            "optimizer_policy": optimizer_policy.state_dict(),
            "optimizer_value": optimizer_value.state_dict(),
            "scheduler_policy": scheduler_policy.state_dict(),
            "scheduler_value": scheduler_value.state_dict(),
            "config": self.config,
        }
        ckpt_path = os.path.join(self.checkpoint_dir, f"iter_{iteration:06d}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved iteration {iteration} to {ckpt_path}")

    def _init_swanlab(self):
        if not self.enable_swanlab or self._swan_initialized:
            return
        swanlab.init(
            project=self.swanlab_project,
            experiment_name=self.swanlab_run_name,
            config=self.config,
        )
        self._swan_initialized = True

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

        # 初始化轻量策略与价值网络
        policy_net = SimplePolicyNet(input_dim=self.total_feature_dim, config=self.config).to(device)
        value_net = SimpleValueNet(input_dim=self.total_feature_dim, config=self.config).to(device)
        
        # 使用 DDP 包装网络（仅在分布式模式下）
        if use_distributed:
            policy_net = DDP(policy_net, device_ids=[rank], output_device=rank)
            value_net = DDP(value_net, device_ids=[rank], output_device=rank)
        if is_master:
            print(f"All networks initialized and moved to device {device}")
            if self.enable_swanlab:
                self._init_swanlab()
        simulator = TeraflowSimulator(config=self.config, device=device)
        
        # 收集网络参数
        policy_params = list(policy_net.module.parameters() if isinstance(policy_net, DDP) else policy_net.parameters())
        value_params = list(value_net.module.parameters() if isinstance(value_net, DDP) else value_net.parameters())
        
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
        
        # 初始化上一次的平均奖励，用于跟踪 path_length 的动态调整
        # 记录上一次是否已经因为 avg_reward > 0.5 而增加过 path_length
        path_length_increased_for_current_threshold = False  # 标记当前阈值（0.5）是否已经增加过
        
        for k in range(self.num_iterations):
            print(f"Iteration {k} started")
            # 重置完成计数（仅在 Linux 分布式模式下）
            if use_distributed and num_workers_done is not None:
                num_workers_done.set("done", b"0")
            # 初始化buffer
            initial_observation, d, theta_f = simulator.reset()  # 获取初始观测（返回 tuple）
            # 初始化本轮 rollout buffer（长度 T = rollout_steps），用于缓存采样到的数据
            buffer_T = self.rollout_steps
            states_buffer = [None] * buffer_T
            rewards_buffer = [None] * buffer_T
            dones_buffer = [None] * buffer_T
            values_buffer = [None] * buffer_T
            old_log_probs_buffer = [None] * buffer_T
            actions_buffer = [None] * buffer_T
            buffer_step_count = 0  # 当前已写入的时间步计数
            
            # 使用初始观测构建特征（第一次迭代）
            current_observation = initial_observation
            # 获取 stop_lines（保持引用）
            stop_lines = getattr(simulator, 'stop_lines', None)

            start_time = time.time()
            for episode_step in range(self.max_episode_steps):
                print(f"step花费时间={time.time()-start_time}")
                # 获取当前路径（w_lane id）对应的世界特征 (x, y, angle)
                path_plan_ids = simulator.agents_path_plans
                if path_plan_ids is None:
                    path_plan_features = None
                else:
                    path_plan_features = simulator.path_planner.get_w_lane_features_by_id(path_plan_ids)
                # 直接从 observation 中提取组件（纯函数，只做 view 操作，不复制数据）
                agents_state, neighbors_local, w_lanes_local, w_boundaries_local = self.decompose_observation(
                    current_observation, self.config
                )
                # 构建轻量网络所需的扁平特征
                features_tensor = self._compose_feature_tensor(
                    agents_state,
                    neighbors_local,
                    w_lanes_local,
                    w_boundaries_local,
                    path_plan_features,
                    simulator.reward_calculator.sampled_params,
                    stop_lines,
                )
                snapshot = self._capture_agent_state_snapshot(simulator)
                if buffer_step_count >= buffer_T:
                    raise RuntimeError("Buffer overflow detected before PPO update.")
                states_buffer[buffer_step_count] = snapshot
                with torch.no_grad():
                    logits = policy_net.module.forward(features_tensor) if isinstance(policy_net, DDP) else policy_net.forward(features_tensor)
                    action_probs = torch.softmax(logits, dim=-1)
                    values = value_net.module.forward(features_tensor) if isinstance(value_net, DDP) else value_net.forward(features_tensor)
                    del features_tensor, logits
                    # 释放解包后的组件（它们只是 view，但为了明确释放引用）
                    del agents_state, neighbors_local, w_lanes_local, w_boundaries_local
                    # 立即保存到 buffer 并释放，减少显存占用
                    values_detached = values.detach().clone()
                    del values
                actions, log_probs, _ = self._sample_actions(action_probs)
                # 直接使用 simulator.step() 返回的 observation，避免重建
                current_observation, rewards, dones = simulator.step(actions.unsqueeze(-1))
                # 释放 action_probs，因为已经得到 actions 和 log_probs
                del action_probs

                rewards_buffer[buffer_step_count] = rewards.detach().clone()
                dones_buffer[buffer_step_count] = dones.detach().clone()
                values_buffer[buffer_step_count] = values_detached
                old_log_probs_buffer[buffer_step_count] = log_probs.detach().clone()
                actions_buffer[buffer_step_count] = actions.detach().clone()
                buffer_step_count += 1
                
                # 检查所有 active 的 agent 是否都 done 了
                active_mask = simulator.agents_state[..., 6] > 0.5  # (B, M) - 所有 active 的 agent
                if hasattr(simulator, 'cumulative_done_mask') and simulator.cumulative_done_mask is not None:
                    done_mask = simulator.cumulative_done_mask  # (B, M) - 所有 done 的 agent
                    # 检查是否有 active 的 agent，且所有 active 的 agent 都 done 了
                    has_active = active_mask.any()
                    if has_active:
                        # 所有 active 的 agent 都 done 了
                        all_active_done = (active_mask & done_mask).all()
                        if all_active_done:
                            if is_master:
                                print(f"所有 active 的 agent 都 done 了，提前结束 episode (step {episode_step + 1}/{self.max_episode_steps})")
                            # 标记需要更新和重置
                            should_update = True
                            should_reset = True
                        else:
                            should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)
                            should_reset = False
                    else:
                        # 没有 active 的 agent，正常检查
                        should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)
                        should_reset = False
                else:
                    # 如果没有 cumulative_done_mask，使用默认逻辑
                    should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)
                    should_reset = False

                if should_update and buffer_step_count > 0:
                    valid_mask = torch.zeros(buffer_T, dtype=torch.bool, device=device)
                    valid_mask[:buffer_step_count] = True
                    # 直接使用当前的 observation 构建 bootstrap features（参考脚本方式）
                    bootstrap_agents_state, bootstrap_neighbors_local, bootstrap_w_lanes_local, bootstrap_w_boundaries_local = self.decompose_observation(
                        current_observation, self.config
                    )
                    bootstrap_path_ids = simulator.agents_path_plans
                    if bootstrap_path_ids is None:
                        bootstrap_path_features = None
                    else:
                        bootstrap_path_features = simulator.path_planner.get_w_lane_features_by_id(bootstrap_path_ids)
                    bootstrap_features = self._compose_feature_tensor(
                        bootstrap_agents_state,
                        bootstrap_neighbors_local,
                        bootstrap_w_lanes_local,
                        bootstrap_w_boundaries_local,
                        bootstrap_path_features,
                        simulator.reward_calculator.sampled_params,
                        stop_lines,
                    )
                    with torch.no_grad():
                        bootstrap_value = value_net.module.forward(bootstrap_features) if isinstance(value_net, DDP) else value_net.forward(bootstrap_features)
                        bootstrap_value = bootstrap_value.detach()
                        del bootstrap_features
                        del bootstrap_agents_state, bootstrap_neighbors_local, bootstrap_w_lanes_local, bootstrap_w_boundaries_local

                    # 不再预先计算 avg_reward，因为大部分样本会被过滤掉
                    # 实际用于更新的平均奖励会在 _ppo_update_from_buffer 中计算
                    update_metrics = self._ppo_update_from_buffer(
                        states_buffer[:buffer_step_count],
                        rewards_buffer[:buffer_step_count],
                        dones_buffer[:buffer_step_count],
                        values_buffer[:buffer_step_count],
                        old_log_probs_buffer[:buffer_step_count],
                        actions_buffer[:buffer_step_count],
                        valid_mask,
                        bootstrap_value,
                        simulator,
                        policy_net,
                        value_net,
                        optimizer_policy,
                        optimizer_value,
                        device,
                        is_master,
                        extra_metrics=None,  # 不再传入预先计算的 avg_reward
                    )
                    if update_metrics is not None:
                        # 检查是否需要增加 path_length（当 avg_reward > 0.5 时）
                        current_avg_reward = update_metrics.get("avg_reward", float('-inf'))
                        if current_avg_reward > 0.5:
                            # avg_reward 大于 0.5，如果还没有因为当前阈值增加过，就增加 path_length
                            if not path_length_increased_for_current_threshold:
                                if simulator.update_path_length(increment=1):
                                    if is_master:
                                        print(f"avg_reward = {current_avg_reward:.4f} > 0.5，path_length 已更新为 {simulator.path_length}")
                                else:
                                    if is_master:
                                        print(f"avg_reward = {current_avg_reward:.4f} > 0.5，但 path_length 已达到最大值 {simulator.max_path_length}")
                                path_length_increased_for_current_threshold = True
                        else:
                            # avg_reward <= 0.5，重置标记，以便下次大于 0.5 时可以再次增加
                            path_length_increased_for_current_threshold = False
                        
                        if (
                            is_master
                            and self.enable_swanlab
                            and self._swan_initialized
                        ):
                            self.global_update_step += 1
                            log_data = dict(update_metrics)
                            log_data["iteration"] = k + 1
                            log_data["update_step"] = self.global_update_step
                            log_data["learning_rate_policy"] = scheduler_policy.get_last_lr()[0]
                            log_data["learning_rate_value"] = scheduler_value.get_last_lr()[0]
                            log_data["path_length"] = simulator.path_length  # 记录当前的 path_length
                            swanlab.log(log_data, step=self.global_update_step)
                    states_buffer = [None] * buffer_T
                    rewards_buffer = [None] * buffer_T
                    dones_buffer = [None] * buffer_T
                    values_buffer = [None] * buffer_T
                    old_log_probs_buffer = [None] * buffer_T
                    actions_buffer = [None] * buffer_T
                    buffer_step_count = 0
                    # 清理显存缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 如果所有 active 的 agent 都 done 了，重置并提前结束当前 episode
                    if should_reset:
                        if is_master:
                            print(f"重置环境并进入下一个 iteration")
                        initial_observation, d, theta_f = simulator.reset()
                        current_observation = initial_observation
                        stop_lines = getattr(simulator, 'stop_lines', None)
                        break  # 提前结束当前 episode 循环
                    
            scheduler_policy.step()
            scheduler_value.step()
            # 保存ckpt
            if is_master and self.checkpoint_interval > 0 and (k + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    iteration=k + 1,
                    policy_net=policy_net,
                    value_net=value_net,
                    optimizer_policy=optimizer_policy,
                    optimizer_value=optimizer_value,
                    scheduler_policy=scheduler_policy,
                    scheduler_value=scheduler_value,
                )
        if is_master and self.enable_swanlab and self._swan_initialized:
            swanlab.finish()
            self._swan_initialized = False

    def _capture_agent_state_snapshot(self, simulator):
        snapshot = {
            "agents_state": simulator.agents_state.detach().to("cpu"),
        }
        if simulator.agents_path_plans is not None:
            snapshot["agents_path_plans"] = simulator.agents_path_plans.detach().to("cpu")
        else:
            snapshot["agents_path_plans"] = None
        if simulator.agents_path_plan_goal_distances is not None:
            snapshot["agents_path_plan_goal_distances"] = simulator.agents_path_plan_goal_distances.detach().to("cpu")
        else:
            snapshot["agents_path_plan_goal_distances"] = None
        reward_params = getattr(simulator.reward_calculator, "sampled_params", None)
        if reward_params is not None:
            snapshot["reward_params"] = reward_params.detach().to("cpu")
        else:
            snapshot["reward_params"] = None
        return snapshot

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
        simulator,
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        device,
        is_master,
        extra_metrics=None):
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
        dones_tensor = torch.stack(dones_buffer, dim=0).to(device)
        dones_bool = dones_tensor.bool()
        dones = dones_tensor.float()
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
        raw_advantages = advantages.clone()
        returns = raw_advantages + values

        dones_bmt_bool = dones_bool.permute(1, 2, 0)
        done_seen = torch.cumsum(dones_bmt_bool.long(), dim=-1)
        post_done_mask = torch.zeros_like(dones_bmt_bool, dtype=torch.bool)
        post_done_mask[..., 1:] = done_seen[..., :-1] > 0
        keep_mask_done = (~post_done_mask).permute(2, 0, 1)

        abs_adv = raw_advantages.abs()
        A_max = abs_adv.max().item()
        # 使用跨迭代 EWMA 的 A_max（参考 Advantage filtering 算法）
        if A_max <= 0:
            # 当前 batch 没有有效优势，直接不过滤
            keep_mask_adv = torch.ones_like(raw_advantages, dtype=torch.bool)
            eta_value = 0.0
        else:
            # 初始化或更新跨迭代的 A_max 平滑值
            if getattr(self, "advantage_Amax_ewma", None) is None:
                # 首次迭代：直接使用当前 A_max
                self.advantage_Amax_ewma = A_max
            else:
                beta = self.advantage_filter_beta
                self.advantage_Amax_ewma = (
                    beta * A_max + (1.0 - beta) * self.advantage_Amax_ewma
                )
            # 按论文：eta = c * \bar{A}_max，这里 c 由 advantage_filter_threshold 控制（例如 0.01）
            eta_value = self.advantage_filter_threshold * self.advantage_Amax_ewma
            keep_mask_adv = abs_adv >= eta_value

        keep_mask = keep_mask_done & keep_mask_adv
        if not keep_mask.any():
            keep_mask = keep_mask_done
        if not keep_mask.any():
            keep_mask = torch.ones_like(keep_mask_done, dtype=torch.bool)

        kept_positions = keep_mask.nonzero(as_tuple=False)
        total_kept = kept_positions.shape[0]
        if total_kept == 0:
            return

        if total_kept > self.batch_size:
            perm = torch.randperm(total_kept, device=device)
            kept_positions = kept_positions[perm[:self.batch_size]]

        sel_t = kept_positions[:, 0]
        sel_b = kept_positions[:, 1]
        sel_m = kept_positions[:, 2]

        advantages_flat = raw_advantages[sel_t, sel_b, sel_m]
        returns_flat = returns[sel_t, sel_b, sel_m]
        old_log_probs_flat = old_log_probs[sel_t, sel_b, sel_m]
        # 提取实际用于更新的奖励（只计算进入 PPO 更新的样本）
        rewards_flat = rewards[sel_t, sel_b, sel_m]

        num_samples = kept_positions.shape[0]
        if num_samples == 0:
            return

        features_flat = torch.empty((num_samples, self.total_feature_dim), device=device)

        unique_times = torch.unique(sel_t, sorted=True)
        for t_idx in unique_times:
            time_mask = (sel_t == t_idx)
            sample_indices = time_mask.nonzero(as_tuple=False).squeeze(-1)
            b_indices = sel_b[sample_indices]
            m_indices = sel_m[sample_indices]

            snapshot = states_buffer[int(t_idx)]
            if snapshot is None or sample_indices.numel() == 0:
                continue

            env_ids, env_inverse = torch.unique(b_indices, sorted=False, return_inverse=True)
            env_ids_cpu = env_ids.to("cpu")
            env_ids_gpu = env_ids.to(simulator.device)
            agents_state_env = snapshot["agents_state"][env_ids_cpu].to(simulator.device)
            observation, _, _ = simulator.observation_generator.generate(agents_state_env)
            agents_state_local, neighbors_local, w_lanes_local, w_boundaries_local = self.decompose_observation(
                observation, self.config
            )

            path_plan_snapshot = snapshot.get("agents_path_plans")
            if path_plan_snapshot is not None:
                path_plan_env = path_plan_snapshot[env_ids_cpu].to(simulator.device)
                path_plan_features_env = simulator.path_planner.get_w_lane_features_by_id(path_plan_env)
            else:
                path_plan_features_env = None

            stop_lines_global = getattr(simulator, "stop_lines", None)
            if stop_lines_global is not None and stop_lines_global.numel() > 0:
                stop_lines_env = stop_lines_global[env_ids_gpu]
            else:
                stop_lines_env = None

            reward_params_snapshot = snapshot.get("reward_params")
            if reward_params_snapshot is not None:
                reward_params_env = reward_params_snapshot[env_ids_cpu].to(simulator.device)
            else:
                reward_params_env = simulator.reward_calculator.sampled_params

            features_env = self._compose_feature_tensor(
                agents_state_local,
                neighbors_local,
                w_lanes_local,
                w_boundaries_local,
                path_plan_features_env,
                reward_params_env,
                stop_lines_env,
            )
            features_selected = features_env[env_inverse, m_indices].view(sample_indices.numel(), -1)
            features_flat[sample_indices] = features_selected

            del (
                agents_state_env,
                observation,
                agents_state_local,
                neighbors_local,
                w_lanes_local,
                w_boundaries_local,
            )

        advantages_mean = advantages_flat.mean()
        advantages_std = advantages_flat.std(unbiased=False)
        if advantages_std.item() == 0:
            advantages_std = advantages_std + 1e-8
        advantages_norm = (advantages_flat - advantages_mean) / (advantages_std + 1e-8)
        advantages_detached = advantages_norm.detach()

        num_samples = advantages_norm.shape[0]
        batch_size = min(self.batch_size, num_samples)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        total_weight = 0

        if num_samples == 0:
            return
        for epoch in range(max(1, self.ppo_epochs)):
            if num_samples <= batch_size:
                perm = torch.arange(num_samples, device=device)
            else:
                perm = torch.randperm(num_samples, device=device)
            start = 0
            while start < num_samples:
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                batch_len = idx.numel()
                if batch_len == 0:
                    break
                feature_batch = features_flat[idx].view(batch_len, 1, self.total_feature_dim)
                logits_batch = policy_net.module.forward(feature_batch) if isinstance(policy_net, DDP) else policy_net.forward(feature_batch)
                action_probs_batch = torch.softmax(logits_batch, dim=-1)
                actions_batch = actions[sel_t[idx], sel_b[idx], sel_m[idx]].view(batch_len, 1)
                new_log_probs_batch, entropy_batch = self._compute_log_probs(action_probs_batch, actions_batch)
                values_batch = value_net.module.forward(feature_batch) if isinstance(value_net, DDP) else value_net.forward(feature_batch)
                values_batch = values_batch.view(batch_len)
                ratio = torch.exp(new_log_probs_batch.view(-1) - old_log_probs_flat[idx])
                advantages_batch = advantages_norm[idx]
                advantages_detached_batch = advantages_detached[idx]
                surr1 = ratio * advantages_detached_batch
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages_detached_batch
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                entropy_bonus = entropy_batch.mean()
                total_policy_loss = policy_loss - self.entropy_coef * entropy_bonus

                optimizer_policy.zero_grad()
                total_policy_loss.backward()
                optimizer_policy.step()

                value_loss = F.mse_loss(values_batch, returns_flat[idx]) * self.value_coef
                optimizer_value.zero_grad()
                value_loss.backward()
                optimizer_value.step()

                policy_loss_sum += policy_loss.item() * batch_len
                value_loss_sum += value_loss.item() * batch_len
                entropy_sum += entropy_bonus.item() * batch_len
                total_weight += batch_len
                start = end

        if total_weight == 0:
            return None

        avg_policy_loss = policy_loss_sum / total_weight
        avg_value_loss = value_loss_sum / total_weight
        avg_entropy = entropy_sum / total_weight
        # 计算实际用于更新的样本的平均奖励（只包括通过过滤的样本）
        # 只统计 active 且未 done 的样本的奖励（奖励非零的样本）
        # 这样可以避免非 active 的 agent（奖励为 0）影响平均值
        active_reward_mask = rewards_flat != 0
        if active_reward_mask.any():
            avg_reward_used = rewards_flat[active_reward_mask].mean().item()
        else:
            # 如果所有样本的奖励都是 0，返回 0
            avg_reward_used = 0.0
        metrics = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "eta": eta_value,
            "A_max": float(A_max),
            "A_max_ewma": float(getattr(self, "advantage_Amax_ewma", A_max)),
            "kept_steps": float(num_samples),
            "updates": float(total_weight),
            "avg_reward": avg_reward_used,  # 实际用于更新的样本的平均奖励
        }
        if extra_metrics:
            # 覆盖 extra_metrics 中的 avg_reward，使用实际用于更新的值
            extra_metrics_copy = extra_metrics.copy()
            extra_metrics_copy.pop("avg_reward", None)  # 移除预先计算的 avg_reward
            metrics.update(extra_metrics_copy)

        if is_master:
            print(
                f"PPO update -> policy_loss: {avg_policy_loss:.6f}, "
                f"value_loss: {avg_value_loss:.6f}, entropy: {avg_entropy:.6f}, "
                f"eta: {eta_value:.6f}, kept_steps: {num_samples}, "
                f"updates: {total_weight} (epochs={max(1, self.ppo_epochs)})"
            )
        return metrics

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
        self.rollout_steps = int(training_config.get('rollout_steps'))
        self.max_episode_steps = int(training_config.get('max_episode_steps'))
        self.batch_size = int(training_config.get('batch_size'))
        self.ppo_epochs = int(training_config.get('ppo_epochs'))
        self.ppo_clip = float(training_config.get('ppo_clip'))
        self.entropy_coef = float(training_config.get('entropy_coef'))
        self.value_coef = float(training_config.get('value_coef'))
        self.advantage_filter_threshold = float(training_config.get('advantage_filter_threshold'))
        self.advantage_filter_beta = float(training_config.get('advantage_filter_beta'))
        self.checkpoint_interval = int(training_config.get('checkpoint_interval', 0))
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.checkpoint_dir = os.path.join(base_dir, 'ckpt')
        if self.checkpoint_interval > 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.enable_swanlab = bool(training_config.get('swanlab_enable', True))
        self.swanlab_project = training_config.get('swanlab_project', 'selfrace-ddppo')
        default_run_name = training_config.get('swanlab_run_name')
        self.swanlab_run_name = default_run_name or f"ddppo_{int(time.time())}"
        self._swan_initialized = False
        self.global_update_step = 0

        simulator_config = self.config.get('simulator', {})
        obs_config = simulator_config.get('observation', {})
        dynamics_config = simulator_config.get('dynamics', {})
        reward_config = simulator_config.get('reward', {})
        self.local_state_feature_dim = 8  # [x,y,cos,sin,speed,length,width,active]
        self.path_plan_length = int(obs_config.get('num_navigation_chains', 0))
        self.path_plan_feature_dim = max(self.path_plan_length, 0) * 2  # 仅使用 dx, dy
        self.reward_param_dim = 10
        num_w_boundaries = int(obs_config.get('num_w_boundaries', 0))
        boundary_feat_dim = int(obs_config.get('boundary_feature_dim', 2))
        self.w_boundary_feature_dim = num_w_boundaries * boundary_feat_dim
        num_w_lanes = int(obs_config.get('num_w_lanes', 0))
        self.w_lane_feature_dim = num_w_lanes * 2  # dx, dy
        num_neighbors = int(obs_config.get('num_neighbors', 0))
        neighbor_feat_dim = int(obs_config.get('neighbor_feature_dim', 0))
        self.neighbor_feature_dim = num_neighbors * neighbor_feat_dim
        self.stop_line_feature_dim = int(simulator_config.get('stop_line_feature_dim', 20))
        self.total_feature_dim = (
            self.local_state_feature_dim
            + self.path_plan_feature_dim
            + self.reward_param_dim
            + self.w_boundary_feature_dim
            + self.w_lane_feature_dim
            + self.stop_line_feature_dim
            + self.neighbor_feature_dim
        )
        self.speed_min = float(dynamics_config.get('min_velocity', -2.0))
        self.speed_mid = 0.0
        self.speed_max = float(dynamics_config.get('max_velocity', 20.0))
        self.w_min = float(dynamics_config.get('vehicle_width_min', 0.8))
        self.w_max = float(dynamics_config.get('vehicle_width_max', 3.0))
        self.l_min = float(dynamics_config.get('vehicle_length_min', 0.8))
        self.l_max = float(dynamics_config.get('vehicle_length_max', 7.0))
        self.dvx_min = -2.0
        self.dvx_mid = 0.0
        self.dvx_max = 20.0
        reward_param_mins = torch.tensor([
            float(reward_config.get("delta_goal_min", 2.0)),
            float(reward_config.get("collision_alpha_min", 0.0)),
            float(reward_config.get("boundary_alpha_min", 0.0)),
            float(reward_config.get("comfort_alpha_min", 0.0)),
            float(reward_config.get("l_align_alpha_min", 2.5e-4)),
            float(reward_config.get("vel_align_alpha_min", 0.0)),
            float(reward_config.get("l_center_alpha_min", 2.5e-4)),
            float(reward_config.get("center_bias_alpha_min", -0.5)),
            float(reward_config.get("reverse_alpha_min", 2.5e-4)),
            float(reward_config.get("stop_line_alpha_min", 0.0)),
        ], dtype=torch.float32)
        reward_param_maxs = torch.tensor([
            float(reward_config.get("delta_goal_max", 12.0)),
            float(reward_config.get("collision_alpha_max", 3.0)),
            float(reward_config.get("boundary_alpha_max", 3.0)),
            float(reward_config.get("comfort_alpha_max", 0.1)),
            float(reward_config.get("l_align_alpha_max", 2.5e-2)),
            float(reward_config.get("vel_align_alpha_max", 1.0)),
            float(reward_config.get("l_center_alpha_max", 7.5e-3)),
            float(reward_config.get("center_bias_alpha_max", 0.5)),
            float(reward_config.get("reverse_alpha_max", 7.5e-3)),
            float(reward_config.get("stop_line_alpha_max", 1.0)),
        ], dtype=torch.float32)
        self.reward_param_mins = reward_param_mins
        self.reward_param_maxs = reward_param_maxs

        simulator_config = self.config.get('simulator', {})
        obs_config = simulator_config.get('observation', {})
        self.observation_horizon = float(obs_config.get('horizon', 200.0))
        
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
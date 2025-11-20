import os
import sys
import json
import socket
import tempfile
import shutil
from datetime import timedelta
from typing import Dict

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
    CompletePolicyNet,
    CompleteValueNet)

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

    @staticmethod
    def build_network_features(
        agents_state: torch.Tensor, 
        neighbors_local: torch.Tensor, 
        w_lanes_local: torch.Tensor, 
        w_boundaries_local: torch.Tensor,
        path_plan_features: torch.Tensor,
        stop_lines: torch.Tensor,
        reward_coef: torch.Tensor,
        config) -> Dict[str, torch.Tensor]:
        """
        将拆解后的观测组件构建为网络输入的特征张量
        Args:
            agents_state: (B, M, 7) - 智能体状态
            neighbors_local: (B, M, K, 7) - 邻居相对状态
            w_lanes_local: (B, M, N_lanes, 2) - 车道线相对坐标
            w_boundaries_local: (B, M, N_boundaries, 2) - 边界线相对坐标
            path_plan_features: (B, M, path_length, 3) - 路径规划特征 (x, y, angle)
            stop_lines: (B, M, num_stop_lines, 20) - 停止线点
            reward_coef: (B, M, 10) - 奖励系数
            config: 配置对象（可以是 dict 或 SimpleNamespace）
        Returns:
            Dict[str, torch.Tensor]: 满足 CompletePolicy/ValueNet 输入需求的组件字典
        """
        batch_size, max_agents, _ = agents_state.shape
        device = agents_state.device

        components: Dict[str, torch.Tensor] = {}

        # w_boundaries_local is already (B, M, K, 2)
        components["w_boundaries_local"] = w_boundaries_local.to(device)

        # local_state 与 agents_state 等价（本地坐标）
        components["local_state"] = agents_state.to(device)
        components["agents_state"] = agents_state.to(device)

        # w_lanes_local_with_goal_distances: 需要 (B, M, K, 4) => [dx, dy, angle_local, Δs]
        if w_lanes_local is None or w_lanes_local.numel() == 0:
            components["w_lanes_local_with_goal_distances"] = torch.zeros(
                batch_size, max_agents, 0, 4, device=device, dtype=agents_state.dtype
            )
        else:
            angles = torch.zeros(
                w_lanes_local.shape[:-1] + (1,), device=device, dtype=w_lanes_local.dtype
            )
            delta = torch.zeros_like(angles)
            components["w_lanes_local_with_goal_distances"] = torch.cat(
                [w_lanes_local.to(device), angles, delta], dim=-1
            )

        components["neighbors_local"] = neighbors_local.to(device)

        # agents_path_plans_world: (B, M, L, 3) -> 使用 path_plan_features (x, y, angle)
        if path_plan_features is None or path_plan_features.numel() == 0:
            components["agents_path_plans_world"] = torch.zeros(
                batch_size, max_agents, 0, 3, device=device, dtype=agents_state.dtype
            )
        else:
            components["agents_path_plans_world"] = path_plan_features.to(device)

        # Curvature / c_* / wheelbase 暂时使用 0（缺省条件）
        components["curvature"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)
        components["c_throttle"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)
        components["c_steer"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)
        components["c_acc"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)
        components["c_vel"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)

        if reward_coef is None:
            components["reward_params"] = torch.zeros(
                batch_size, max_agents, 10, device=device, dtype=agents_state.dtype
            )
        else:
            if reward_coef.shape[-1] < 10:
                pad = torch.zeros(
                    batch_size, max_agents, 10 - reward_coef.shape[-1],
                    device=reward_coef.device, dtype=reward_coef.dtype
                )
                reward_padded = torch.cat([reward_coef, pad], dim=-1)
            else:
                reward_padded = reward_coef
            components["reward_params"] = reward_padded[..., :10].to(device)

        components["wheelbase"] = torch.zeros(batch_size, max_agents, device=device, dtype=agents_state.dtype)

        return components

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

        # 初始化完整的策略网络和价值网络（包含所有编码网络和 MLP）
        complete_policy_net = CompletePolicyNet(config=self.config).to(device)
        complete_value_net = CompleteValueNet(config=self.config).to(device)
        
        # 使用 DDP 包装网络（仅在分布式模式下）
        if use_distributed:
            complete_policy_net = DDP(complete_policy_net, device_ids=[rank], output_device=rank)
            complete_value_net = DDP(complete_value_net, device_ids=[rank], output_device=rank)
        if is_master:
            print(f"All networks initialized and moved to device {device}")
        simulator = TeraflowSimulator(config=self.config, device=device)
        
        # 收集 policy 网络的参数（完整网络包含所有编码网络和 policy MLP）
        policy_params = list(complete_policy_net.module.parameters() if isinstance(complete_policy_net, DDP) else complete_policy_net.parameters())
        
        # 收集 value 网络的参数（完整网络包含所有编码网络和 value MLP）
        value_params = list(complete_value_net.module.parameters() if isinstance(complete_value_net, DDP) else complete_value_net.parameters())
        
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
            
            for episode_step in range(self.max_episode_steps):
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
                # 构建网络输入组件（满足 CompletePolicy/ValueNet 需求）
                components = self.build_network_features(
                    agents_state,
                    neighbors_local,
                    w_lanes_local,
                    w_boundaries_local,
                    path_plan_features,
                    stop_lines,
                    simulator.reward_calculator.sampled_params,
                    self.config,
                )
                snapshot = self._capture_agent_state_snapshot(simulator)
                if buffer_step_count >= buffer_T:
                    raise RuntimeError("Buffer overflow detected before PPO update.")
                states_buffer[buffer_step_count] = snapshot
                with torch.no_grad():
                    # 直接使用完整网络，避免产生大量中间激活值
                    # 在 no_grad 模式下，编码网络的中间激活值不会保存，达到节约显存的效果
                    action_probs = complete_policy_net.module.forward(components) if isinstance(complete_policy_net, DDP) else complete_policy_net.forward(components)
                    values = complete_value_net.module.forward(components) if isinstance(complete_value_net, DDP) else complete_value_net.forward(components)
                    # 立即释放 components，减少显存占用
                    del components
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
                should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)

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
                    bootstrap_components = self.build_network_features(
                        bootstrap_agents_state,
                        bootstrap_neighbors_local,
                        bootstrap_w_lanes_local,
                        bootstrap_w_boundaries_local,
                        bootstrap_path_features,
                        stop_lines,
                        simulator.reward_calculator.sampled_params,
                        self.config,
                    )
                    with torch.no_grad():
                        # 直接使用完整网络，避免产生大量中间激活值
                        # 注意：当前网络架构使用 components 字典，需要适配
                        # TODO: 修改网络架构使其接收 features_tensor
                        bootstrap_value = complete_value_net.module.forward(bootstrap_components) if isinstance(complete_value_net, DDP) else complete_value_net.forward(bootstrap_components)
                        bootstrap_value = bootstrap_value.detach()
                        # 立即释放中间激活值
                        del bootstrap_components
                        del bootstrap_agents_state, bootstrap_neighbors_local, bootstrap_w_lanes_local, bootstrap_w_boundaries_local

                    self._ppo_update_from_buffer(
                        states_buffer[:buffer_step_count],
                        rewards_buffer[:buffer_step_count],
                        dones_buffer[:buffer_step_count],
                        values_buffer[:buffer_step_count],
                        old_log_probs_buffer[:buffer_step_count],
                        actions_buffer[:buffer_step_count],
                        valid_mask,
                        bootstrap_value,
                        simulator,
                        complete_policy_net,
                        complete_value_net,
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
                    # 清理显存缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            scheduler_policy.step()
            scheduler_value.step()

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
        complete_policy_net,
        complete_value_net,
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
        if A_max <= 0:
            keep_mask_adv = torch.ones_like(raw_advantages, dtype=torch.bool)
            eta_value = 0.0
        else:
            eta_value = self.advantage_filter_threshold * A_max
            keep_mask_adv = abs_adv >= eta_value

        keep_mask = keep_mask_done & keep_mask_adv
        if not keep_mask.any():
            keep_mask = keep_mask_done
        if not keep_mask.any():
            keep_mask = torch.ones_like(keep_mask_done, dtype=torch.bool)

        mask_flat = keep_mask.view(-1)
        if not mask_flat.any():
            return

        time_keep_mask = keep_mask.any(dim=(1, 2))
        time_indices = time_keep_mask.nonzero(as_tuple=True)[0].tolist()
        if not time_indices:
            return

        # 只为需要的时间步创建张量，而不是为所有时间步创建
        new_log_probs_full = torch.zeros_like(old_log_probs)
        new_values_full = torch.zeros_like(values)
        entropies_full = torch.zeros_like(old_log_probs)

        for t_idx in time_indices:
            snapshot = states_buffer[t_idx]
            if snapshot is None:
                continue
            with torch.no_grad():
                # 在 no_grad 下重建观测和组件（参考脚本方式）
                agents_state_snapshot = snapshot["agents_state"].to(simulator.device)
                observation, d, theta_f = simulator.observation_generator.generate(agents_state_snapshot)
                agents_state, neighbors_local, w_lanes_local, w_boundaries_local = self.decompose_observation(
                    observation, self.config
                )
                path_plan_snapshot = snapshot.get("agents_path_plans")
                if path_plan_snapshot is not None:
                    path_plan_snapshot = path_plan_snapshot.to(simulator.device)
                    path_plan_features_snapshot = simulator.path_planner.get_w_lane_features_by_id(path_plan_snapshot)
                else:
                    path_plan_features_snapshot = None
                stop_lines_snapshot = getattr(simulator, 'stop_lines', None)
                reward_params_snapshot = snapshot.get("reward_params")
                if reward_params_snapshot is not None:
                    reward_params_snapshot = reward_params_snapshot.to(simulator.device)
                else:
                    reward_params_snapshot = simulator.reward_calculator.sampled_params
                
                # 构建网络输入组件（参考脚本方式）
                components = self.build_network_features(
                    agents_state,
                    neighbors_local,
                    w_lanes_local,
                    w_boundaries_local,
                    path_plan_features_snapshot,
                    stop_lines_snapshot,
                    reward_params_snapshot,
                    self.config,
                )
                # 直接使用完整网络，避免产生大量中间激活值
                action_probs = complete_policy_net.module.forward(components) if isinstance(complete_policy_net, DDP) else complete_policy_net.forward(components)
                current_values = complete_value_net.module.forward(components) if isinstance(complete_value_net, DDP) else complete_value_net.forward(components)
                current_values = current_values.squeeze(-1)
                log_probs_now, entropy_now = self._compute_log_probs(action_probs, actions[t_idx])
                # 立即保存并释放，减少显存占用
                new_log_probs_full[t_idx] = log_probs_now.detach()
                new_values_full[t_idx] = current_values.detach()
                entropies_full[t_idx] = entropy_now.detach()
                # 清理临时张量，释放显存
                del components, action_probs, current_values, log_probs_now, entropy_now
                del observation, agents_state, neighbors_local, w_lanes_local, w_boundaries_local

        advantages_flat = raw_advantages.view(-1)[mask_flat]
        returns_flat = returns.view(-1)[mask_flat]
        old_log_probs_flat = old_log_probs.view(-1)[mask_flat]
        new_log_probs_flat = new_log_probs_full.view(-1)[mask_flat]
        new_values_flat = new_values_full.view(-1)[mask_flat]
        entropies_flat = entropies_full.view(-1)[mask_flat]

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
                ratio = torch.exp(new_log_probs_flat[idx] - old_log_probs_flat[idx])
                advantages_batch = advantages_norm[idx]
                advantages_detached_batch = advantages_detached[idx]
                surr1 = ratio * advantages_detached_batch
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages_detached_batch
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                entropy_bonus = entropies_flat[idx].mean()
                total_policy_loss = policy_loss - self.entropy_coef * entropy_bonus

                optimizer_policy.zero_grad()
                total_policy_loss.backward()
                optimizer_policy.step()

                value_loss = F.mse_loss(new_values_flat[idx], returns_flat[idx]) * self.value_coef
                optimizer_value.zero_grad()
                value_loss.backward()
                optimizer_value.step()

                policy_loss_sum += policy_loss.item() * batch_len
                value_loss_sum += value_loss.item() * batch_len
                entropy_sum += entropy_bonus.item() * batch_len
                total_weight += batch_len
                start = end

        if total_weight == 0:
            return

        if is_master:
            avg_policy_loss = policy_loss_sum / total_weight
            avg_value_loss = value_loss_sum / total_weight
            avg_entropy = entropy_sum / total_weight
            print(
                f"PPO update -> policy_loss: {avg_policy_loss:.6f}, "
                f"value_loss: {avg_value_loss:.6f}, entropy: {avg_entropy:.6f}, "
                f"eta: {eta_value:.6f}, kept_steps: {num_samples}, "
                f"updates: {total_weight} (epochs={max(1, self.ppo_epochs)})"
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
        self.rollout_steps = int(training_config.get('rollout_steps'))
        self.max_episode_steps = int(training_config.get('max_episode_steps'))
        self.batch_size = int(training_config.get('batch_size'))
        self.ppo_epochs = int(training_config.get('ppo_epochs'))
        self.ppo_clip = float(training_config.get('ppo_clip'))
        self.entropy_coef = float(training_config.get('entropy_coef'))
        self.value_coef = float(training_config.get('value_coef'))
        self.advantage_filter_threshold = float(training_config.get('advantage_filter_threshold'))
        self.advantage_filter_beta = float(training_config.get('advantage_filter_beta'))
        
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
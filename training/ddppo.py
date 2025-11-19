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
            simulator.reset()
            # 初始化本轮 rollout buffer（长度 T = rollout_steps），用于缓存采样到的数据
            buffer_T = self.rollout_steps
            states_buffer = [None] * buffer_T
            rewards_buffer = [None] * buffer_T
            dones_buffer = [None] * buffer_T
            values_buffer = [None] * buffer_T
            old_log_probs_buffer = [None] * buffer_T
            actions_buffer = [None] * buffer_T
            buffer_step_count = 0  # 当前已写入的时间步计数
            for episode_step in range(self.max_episode_steps):
                components = self._build_components_from_data(
                    simulator,
                    simulator.agents_state,
                    simulator.agents_path_plans,
                    simulator.agents_path_plan_goal_distances,
                    simulator.reward_calculator.sampled_params,
                    device,
                )
                snapshot = self._capture_agent_state_snapshot(simulator)
                if buffer_step_count >= buffer_T:
                    raise RuntimeError("Buffer overflow detected before PPO update.")
                states_buffer[buffer_step_count] = snapshot

                with torch.no_grad():
                    encoded_features = self._encode_features(
                        components,
                        simulator,
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
                _, rewards, dones = simulator.step(actions.unsqueeze(-1))

                rewards_buffer[buffer_step_count] = rewards.detach().clone()
                dones_buffer[buffer_step_count] = dones.detach().clone()
                values_buffer[buffer_step_count] = values.detach().clone()
                old_log_probs_buffer[buffer_step_count] = log_probs.detach().clone()
                actions_buffer[buffer_step_count] = actions.detach().clone()
                buffer_step_count += 1
                should_update = (buffer_step_count == buffer_T) or (episode_step == self.max_episode_steps - 1)

                if should_update and buffer_step_count > 0:
                    valid_mask = torch.zeros(buffer_T, dtype=torch.bool, device=device)
                    valid_mask[:buffer_step_count] = True
                    bootstrap_components = self._build_components_from_data(
                        simulator,
                        simulator.agents_state,
                        simulator.agents_path_plans,
                        simulator.agents_path_plan_goal_distances,
                        simulator.reward_calculator.sampled_params,
                        device,
                    )
                    with torch.no_grad():
                        bootstrap_encoded = self._encode_features(
                            bootstrap_components,
                            simulator,
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
                        simulator,
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

    def _build_components_from_data(
        self,
        simulator,
        agents_state,
        path_plans,
        path_plan_goal_distances,
        reward_params,
        device):
        """根据给定状态信息即时生成观测组件。"""
        obs_gen = simulator.observation_generator
        sim_device = simulator.device
        agents_state_dev = agents_state.to(sim_device)

        observation, d, theta_f = obs_gen.generate(agents_state_dev)
        simulator.frenet_d = d
        simulator.frenet_theta_f = theta_f

        if path_plans is not None:
            path_plans_dev = path_plans.to(sim_device)
        else:
            path_plans_dev = None
        if path_plan_goal_distances is not None:
            path_plan_goal_distances_dev = path_plan_goal_distances.to(sim_device)
        else:
            path_plan_goal_distances_dev = None

        w_lane_features = self._compose_w_lane_features(
            simulator,
            agents_state_dev,
            path_plans_dev,
            path_plan_goal_distances_dev,
        )

        if path_plans_dev is not None:
            path_centers = simulator.path_planner.get_w_lane_centers_by_id(path_plans_dev)
            agents_path_plans_world = torch.cat(
                [path_centers, torch.zeros_like(path_centers[..., :1])],
                dim=-1,
            )
        else:
            B, M = agents_state_dev.shape[:2]
            agents_path_plans_world = torch.zeros((B, M, 0, 3), device=sim_device)

        B, M = agents_state_dev.shape[:2]
        target_device = device

        def _ensure_tensor(tensor, shape=None, fill_value=0.0):
            if tensor is None:
                if shape is None:
                    raise RuntimeError("Missing tensor for components.")
                return torch.full(shape, fill_value, device=target_device)
            return tensor.to(target_device)

        components = {}
        components["observation"] = observation.to(target_device)
        components["w_boundaries_local"] = _ensure_tensor(obs_gen.last_w_boundaries_local)
        components["local_state"] = _ensure_tensor(obs_gen.last_local_state)
        components["neighbors_local"] = _ensure_tensor(obs_gen.last_neighbors_local)
        components["w_lanes_local_with_goal_distances"] = w_lane_features.to(target_device)
        components["agents_state"] = agents_state_dev.to(target_device)
        components["agents_path_plans_world"] = agents_path_plans_world.to(target_device)

        curvature = getattr(obs_gen, "curvature", None)
        components["curvature"] = _ensure_tensor(curvature, (B, M))

        if reward_params is None:
            reward_params = simulator.reward_calculator.sampled_params
        components["reward_params"] = reward_params.to(target_device)

        wheelbase = torch.zeros((B, M), device=target_device)
        if hasattr(simulator.dynamics_model, "vehicle_params"):
            vehicle_params = simulator.dynamics_model.vehicle_params
            if isinstance(vehicle_params, dict) and "wheelbase" in vehicle_params:
                wheelbase = vehicle_params["wheelbase"].view(B, M).to(target_device)
        components["wheelbase"] = wheelbase

        c_throttle = torch.zeros((B, M), device=target_device)
        c_steer = torch.zeros((B, M), device=target_device)
        c_acc = torch.zeros((B, M), device=target_device)
        c_vel = torch.zeros((B, M), device=target_device)
        components["c_throttle"] = c_throttle
        components["c_steer"] = c_steer
        components["c_acc"] = c_acc
        components["c_vel"] = c_vel
        return components

    def _compose_w_lane_features(
        self,
        simulator,
        agents_state,
        path_plans,
        path_plan_goal_distances):
        obs_gen = simulator.observation_generator
        w_lanes_local = obs_gen.last_w_lanes_local
        w_lane_ids = obs_gen.last_w_lanes_ids
        if w_lanes_local is None or w_lane_ids is None:
            B, M = agents_state.shape[:2]
            return torch.zeros((B, M, 0, 4), device=agents_state.device)

        w_lanes_local = w_lanes_local.to(agents_state.device)
        w_lane_ids = w_lane_ids.to(agents_state.device)
        B, M, K, _ = w_lanes_local.shape
        dtype = w_lanes_local.dtype
        planner = simulator.path_planner

        if path_plans is None or path_plan_goal_distances is None:
            invalid_value = float(planner.INVALID_MARKER)
            return torch.full((B, M, K, 4), invalid_value, dtype=dtype, device=agents_state.device)

        w_lane_goal_full = self._build_w_lane_goal_full(
            simulator,
            path_plans,
            path_plan_goal_distances,
        )
        total_w_lanes = w_lane_goal_full.shape[-1]

        idx = planner.map_w_lane_ids_to_indices(w_lane_ids)
        invalid_value = float(planner.INVALID_MARKER)
        delta = torch.full((B, M, K), invalid_value, dtype=dtype, device=agents_state.device)
        valid = (idx >= 0) & (idx < total_w_lanes)
        if valid.any():
            batch_idx = torch.arange(B, device=agents_state.device).view(B, 1, 1).expand_as(idx)
            agent_idx = torch.arange(M, device=agents_state.device).view(1, M, 1).expand_as(idx)
            delta_vals = w_lane_goal_full[batch_idx[valid], agent_idx[valid], idx[valid]]
            delta[valid] = delta_vals

        w_lane_features_world = planner.get_w_lane_features_by_id(w_lane_ids)
        angles_world = w_lane_features_world[..., 2]
        ego_yaw = agents_state[..., 2].unsqueeze(-1)
        angles_local = angles_world - ego_yaw
        angles_local = torch.atan2(torch.sin(angles_local), torch.cos(angles_local))

        return torch.cat(
            [
                w_lanes_local,
                angles_local.unsqueeze(-1),
                delta.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _build_w_lane_goal_full(
        self,
        simulator,
        path_plans,
        path_plan_goal_distances):
        planner = simulator.path_planner
        total_w_lanes = planner.w_lane_features.shape[0]
        B, M, _ = path_plans.shape
        if total_w_lanes == 0:
            return torch.empty((B, M, 0), dtype=path_plan_goal_distances.dtype, device=path_plans.device)

        idx = planner.map_w_lane_ids_to_indices(path_plans)
        invalid_value = float(planner.INVALID_MARKER)
        idx_mask = (idx >= 0) & (idx < total_w_lanes)

        if not idx_mask.any():
            return torch.full(
                (B, M, total_w_lanes),
                invalid_value,
                dtype=path_plan_goal_distances.dtype,
                device=path_plans.device,
            )

        b_idx, m_idx, l_idx = idx_mask.nonzero(as_tuple=True)
        target_idx = idx[b_idx, m_idx, l_idx]
        src_vals = path_plan_goal_distances[b_idx, m_idx, l_idx]
        flat_size = B * M
        inf_value = torch.tensor(float("inf"), dtype=path_plan_goal_distances.dtype, device=path_plans.device)
        flat_full = torch.full(
            (flat_size * total_w_lanes,),
            inf_value,
            dtype=path_plan_goal_distances.dtype,
            device=path_plans.device,
        )
        flat_row = b_idx * M + m_idx
        linear_idx = flat_row * total_w_lanes + target_idx
        flat_full.scatter_reduce_(
            dim=0,
            index=linear_idx,
            src=src_vals,
            reduce="amin",
            include_self=True,
        )
        full = flat_full.view(B, M, total_w_lanes)
        return torch.where(torch.isinf(full), torch.full_like(full, invalid_value), full)

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

    def _encode_features(
        self,
        components,
        simulator,
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
        simulator,
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

        new_log_probs_full = torch.zeros_like(old_log_probs)
        new_values_full = torch.zeros_like(values)
        entropies_full = torch.zeros_like(old_log_probs)

        for t_idx in time_indices:
            snapshot = states_buffer[t_idx]
            if snapshot is None:
                continue
            components = self._build_components_from_data(
                simulator,
                snapshot["agents_state"],
                snapshot.get("agents_path_plans"),
                snapshot.get("agents_path_plan_goal_distances"),
                snapshot.get("reward_params"),
                device,
            )
            encoded = self._encode_features(
                components,
                simulator,
                w_boundary_net,
                goals_net,
                w_lane_net,
                other_agents_net,
                condition_net,
                vehicle_state_net,
            )
            action_probs = mlp_policy(*encoded)
            current_values = mlp_value(*encoded).squeeze(-1)
            log_probs_now, entropy_now = self._compute_log_probs(action_probs, actions[t_idx])
            new_log_probs_full[t_idx] = log_probs_now
            new_values_full[t_idx] = current_values
            entropies_full[t_idx] = entropy_now

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
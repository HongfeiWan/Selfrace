import os
import sys
import math
import json
from types import SimpleNamespace
from typing import Dict, List, Tuple
import random
import yaml
import torch
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加simulator目录到路径
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
# 添加training目录到路径
training_dir = os.path.join(parent_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
    
from simulator import TeraflowSimulator
from ddppo import decompose_observation, build_network_features
from network import create_network

class CarGame:
    def __init__(self):
        # 加载默认配置（不需要传入路径）
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml') 
        self.config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        # 初始化pygame（可视化）
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("selfrace可视化训练")
        # 设备
        if not torch.cuda.is_available():
            raise RuntimeError("需要CUDA环境，请在GPU上运行。")
        self.device = torch.device('cuda:0')

        # 接入 TeraflowSimulator（统一管理路网/车辆/奖励/碰撞/离路等）
        if 'simulator' not in self.config:
            self.config['simulator'] = {}
        self.simulator = TeraflowSimulator(self.config, self.device)
        initial_observation = self.simulator.reset()
        # 构建网络并初始化特征
        self.config_ns = json.loads(json.dumps(self.config), object_hook=lambda d: SimpleNamespace(**d))
        self.model = create_network(config=self.config_ns, network_type="independent").to(self.device)
        self.model.eval() # 设置为推理模式
        
        # 初始化特征（与训练保持一致）
        self.path_plan = self.simulator.agents_path_plans
        self.stop_lines = self.simulator.stop_lines

        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.path_plan,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )

        # 游戏状态
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.episode_reward = 0.0
        self.step_count = 0
        self.iteration_count = 0  # 添加iteration计数器

        # 从配置读取最大步数：training.iteration
        self.max_steps = int(self.config.get('training').get('max_episode_length'))
        
        # 模仿ddppo.py的buffer和rollout设置
        self.rollout_length = 128  # 与ddppo.py保持一致
        self.ppo_epochs = int(self.config.get('training').get('ppo_epochs', 2))
        self.gamma = float(self.config.get('training').get('gamma', 0.999))
        self.gae_lambda = float(self.config.get('training').get('gae_lambda', 0.95))
        self.clip_ratio = float(self.config.get('training').get('clip_ratio', 0.2))
        self.entropy_coef = float(self.config.get('training').get('entropy_coef', 0.01))
        self.value_loss_coef = float(self.config.get('training').get('value_loss_coef', 0.5))
        self.max_grad_norm = float(self.config.get('training').get('max_grad_norm', 1.0))
        self.learning_rate = float(self.config.get('training').get('learning_rate', 3e-4))
        
        # 优势过滤参数（模仿ddppo.py）
        self.beta = float(self.config.get('training').get('advantage_filter_beta', 0.25))
        self.advantage_filter_threshold = float(self.config.get('training').get('advantage_filter_threshold', 0.01))
        self.A_max_ewma = None
        self.batch_size_per_gpu = int(self.config.get('training').get('batch_size_per_gpu', 32000))
        
        # 初始化优化器
        self.policy_optimizer = torch.optim.Adam(self.model.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.model.value_network.parameters(), lr=self.learning_rate)
        
        # 初始化buffer
        self.reset_buffers()

        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # 颜色定义
        self.colors = {
            'road': (0, 0, 0),
            'grass': (255, 255, 255),
            'car': (255, 0, 0),
            'other_car': (0, 0, 255),  # 蓝色表示其他车辆
            'goal': (255, 255, 0),
            'text': (0, 0, 0),
            'lane_markings': (0, 255, 0),
            'dead_car': (255, 215, 0)
        }
    
    def reset_buffers(self):
        """重置所有buffer，模仿ddppo.py"""
        self.states_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.old_log_probs_buffer = []
        self.actions_buffer = []
        self.buffer_step_count = 0
    
    def gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, gae_lambda: float):
        """GAE优势计算，与ddppo.py保持一致"""
        T = rewards.shape[0]
        done_mask = dones.to(rewards.dtype)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - done_mask[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - done_mask[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _check_all_vehicles_dead(self) -> bool:
        """检查是否所有车辆都已死亡（只检查第一个世界，用于显示控制）。"""
        if self.simulator is None:
            return True
        try:
            env0_states = self.simulator.agents_state[0]
            active_mask = (env0_states[:, 6] > 0.5).detach().to('cpu')
            if self.last_done_env0 is not None and self.last_done_env0.numel() == active_mask.numel():
                # 所有车辆都死亡：active=False 或 done=True
                all_dead = torch.all(self.last_done_env0 | (~active_mask))
                return bool(all_dead.item())
            return False
        except Exception:
            return False
    
    def _check_all_worlds_done(self) -> bool:
        """检查是否所有世界都已完成（用于训练控制）。"""
        if self.simulator is None:
            return True
        try:
            # 检查所有世界的done状态
            all_dones = self.simulator.done if hasattr(self.simulator, 'done') else None
            if all_dones is not None:
                # 所有世界都完成：所有batch的done都为True
                return bool(torch.all(all_dones).item())
            return False
        except Exception:
            return False
    
    def _check_all_worlds_no_alive_agents(self) -> bool:
        """检查是否所有世界都没有存活agents（用于决定是否开启新iteration）。"""
        if self.simulator is None:
            return True
        try:
            # 检查所有世界的agents状态
            all_agents_state = self.simulator.agents_state  # (B, M, S)
            B, M, S = all_agents_state.shape
            
            print(f"🔍 检查所有世界存活状态: B={B}, M={M}")
            
            # 检查每个世界是否有存活的agents
            for b in range(B):
                world_agents = all_agents_state[b]  # (M, S)
                active_mask = world_agents[:, 6] > 0.5  # active状态
                active_count = active_mask.sum().item()
                
                print(f"  世界 {b}: active agents = {active_count}")
                
                # 如果有active的agents，检查是否有存活的
                if active_mask.any():
                    # 使用记录的所有世界的done状态
                    if hasattr(self, 'last_done_all_worlds') and self.last_done_all_worlds is not None:
                        world_done = self.last_done_all_worlds[b].to(active_mask.device)  # 确保设备一致
                        done_count = world_done.sum().item()
                        alive_mask = active_mask & (~world_done)
                        alive_count = alive_mask.sum().item()
                        
                        print(f"    累积done状态: {done_count}, 存活agents: {alive_count}")
                        
                        if alive_mask.any():
                            print(f"    ✅ 世界 {b} 还有存活agents，不开启新iteration")
                            return False  # 这个世界还有存活的agents
                    else:
                        # 没有done状态记录，只要有active的agents就认为有存活的
                        print(f"    ✅ 世界 {b} 有active agents且无done记录，不开启新iteration")
                        return False
            
            print(f"    ❌ 所有世界都没有存活agents，开启新iteration")
            return True  # 所有世界都没有存活的agents
        except Exception as e:
            print(f"    ⚠️ 检查存活状态时出错: {e}")
            return True

    def _find_alive_agent_in_any_world(self) -> Tuple[int, int]:
        """在所有世界中找到第一个存活的智能体。
        Returns:
            Tuple[int, int]: (world_index, agent_index)，如果没找到返回(-1, -1)
        """
        if self.simulator is None:
            return -1, -1
        
        all_agents_state = self.simulator.agents_state  # (B, M, S)
        B, M, S = all_agents_state.shape
        
        for b in range(B):
            world_agents = all_agents_state[b]  # (M, S)
            active_mask = world_agents[:, 6] > 0.5  # active状态
            
            if active_mask.any():
                # 检查是否有存活的agents
                if hasattr(self, 'last_done_all_worlds') and self.last_done_all_worlds is not None:
                    world_done = self.last_done_all_worlds[b].to(active_mask.device)
                    alive_mask = active_mask & (~world_done)
                    if alive_mask.any():
                        idx = torch.nonzero(alive_mask, as_tuple=False)
                        if idx.numel() > 0:
                            return b, idx[0, 0].item()
                else:
                    # 没有done状态记录，只要有active的agents就认为有存活的
                    idx = torch.nonzero(active_mask, as_tuple=False)
                    if idx.numel() > 0:
                        return b, idx[0, 0].item()
        
        return -1, -1  # 没有找到存活的智能体

    def _get_player_agent_index(self) -> int:
        """获取当前玩家智能体的索引（优先世界0，如果世界0全部done则切换到其他存活世界）。"""
        # 首先尝试世界0
        states = self.simulator.agents_state
        env0 = states[0]
        active_mask = env0[:, 6] > 0.5
        alive_mask = active_mask
        if hasattr(self, 'dead_mask_env0') and self.dead_mask_env0 is not None:
            alive_mask = active_mask & (~self.dead_mask_env0.to(active_mask.device))
        idx = torch.nonzero(alive_mask, as_tuple=False)
        if idx.numel() > 0:
            return idx[0, 0].item()
        
        # 世界0没有存活智能体，尝试其他世界
        world_idx, agent_idx = self._find_alive_agent_in_any_world()
        if world_idx >= 0 and agent_idx >= 0:
            # 记录当前视角来自哪个世界
            self._current_view_world = world_idx
            return agent_idx
        
        return -1  # 没有存活的智能体

    def _get_player_pose(self) -> Tuple[float, float, float]:
        """选择第一个仍然 alive 的智能体（active 且未 dead）的位姿；若无则沿用上一次视角或返回(0,0,0)。"""
        player_idx = self._get_player_agent_index()
        if player_idx >= 0:
            # 确定从哪个世界获取位姿
            if hasattr(self, '_current_view_world'):
                world_idx = self._current_view_world
            else:
                world_idx = 0  # 默认世界0
            
            world_agents = self.simulator.agents_state[world_idx]
            x = world_agents[player_idx, 0].item()
            y = world_agents[player_idx, 1].item()
            yaw = world_agents[player_idx, 2].item()
            self._last_camera_pose = (x, y, yaw)
            return x, y, yaw
        if hasattr(self, '_last_camera_pose'):
            return self._last_camera_pose
        return 0.0, 0.0, 0.0

    def update_game_state(self):
        """使用 TeraflowSimulator 推进一步，并更新显示状态（env0）。模仿ddppo.py的buffer和rollout逻辑"""
        if self.simulator is None:
            return
        
        # 检查是否第一个世界的车辆都已死亡，如果是则暂停显示但继续训练
        if self._check_all_vehicles_dead():
            if not hasattr(self, 'first_world_dead'):
                print("🔄 第一个世界所有车辆死亡，暂停显示但继续训练...")
                self.first_world_dead = True
            # 不return，继续执行训练逻辑

        # 使用网络输出动作分布并采样动作（与训练一致）
        with torch.no_grad():
            action_logits = self.model.forward(self.features_tensor, mode="policy")
            value_pred = self.model.forward(self.features_tensor, mode="value")
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        
        # 保存本步动作以便UI显示
        self.last_actions = actions.detach().to('cpu')
        
        # 推进环境
        obs, reward, done = self.simulator.step(actions)
        
        # 将数据存入buffer（模仿ddppo.py）
        self.states_buffer.append(self.simulator.agents_state.clone())
        self.rewards_buffer.append(reward.clone())
        self.dones_buffer.append(done.clone())
        self.values_buffer.append(value_pred.clone())
        self.old_log_probs_buffer.append(action_dist.log_prob(actions).clone())
        self.actions_buffer.append(actions.clone())
        self.buffer_step_count += 1
        
        # 更新特征以供下一个step使用
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(obs, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.path_plan,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )

        # 保存本step来自simulator的原始reward（env0，逐agent）
        self.last_reward_env0 = reward[0].detach().to('cpu')
        # 记录所有世界的done状态用于判断是否所有世界都没有存活agents
        try:
            current_done_all = done.detach().to('cpu').bool()  # (B, M)
            if hasattr(self, 'last_done_all_worlds') and self.last_done_all_worlds is not None:
                # 累积done状态：一旦done=True，就保持True
                self.last_done_all_worlds = self.last_done_all_worlds | current_done_all
            else:
                self.last_done_all_worlds = current_done_all
        except Exception:
            self.last_done_all_worlds = None
        
        # 记录第一个世界的done状态用于显示控制
        try:
            current_done_env0 = done[0].detach().to('cpu').bool()
            if hasattr(self, 'last_done_env0') and self.last_done_env0 is not None:
                # 累积done状态：一旦done=True，就保持True
                self.last_done_env0 = self.last_done_env0 | current_done_env0
            else:
                self.last_done_env0 = current_done_env0
        except Exception:
            self.last_done_env0 = None

        # 死亡（done）处理：标记并保存死亡时姿态（所有世界）
        all_done = done.detach().to('cpu').bool()  # (B, M)
        all_states_now = self.simulator.agents_state.detach().to('cpu')  # (B, M, S)
        B, M, S = all_states_now.shape
        
        # 初始化所有世界的死亡状态记录
        if not hasattr(self, 'dead_mask_all_worlds') or self.dead_mask_all_worlds is None or self.dead_mask_all_worlds.shape != (B, M):
            self.dead_mask_all_worlds = torch.zeros((B, M), dtype=torch.bool)
            self.dead_pose_all_worlds = {}  # {(world_idx, agent_idx): (x, y, yaw, length, width)}
        
        # 处理所有世界的死亡状态
        for b in range(B):
            world_done = all_done[b]  # (M,)
            world_states = all_states_now[b]  # (M, S)
            world_dead_mask = self.dead_mask_all_worlds[b]  # (M,)
            
            newly_dead = (~world_dead_mask) & world_done
            if newly_dead.any():
                dead_indices = torch.nonzero(newly_dead, as_tuple=False).squeeze(-1).tolist()
                for idx in dead_indices:
                    x = float(world_states[idx, 0].item())
                    y = float(world_states[idx, 1].item())
                    yaw = float(world_states[idx, 2].item())
                    length = float(world_states[idx, 4].item())
                    width = float(world_states[idx, 5].item())
                    self.dead_pose_all_worlds[(b, idx)] = (x, y, yaw, length, width)
            
            # 更新死亡掩码
            self.dead_mask_all_worlds[b] = world_dead_mask | world_done
        
        # 保持向后兼容性：世界0的死亡状态
        self.dead_mask_env0 = self.dead_mask_all_worlds[0]
        self.dead_pose_env0 = {}
        for (world_idx, agent_idx), pose in self.dead_pose_all_worlds.items():
            if world_idx == 0:
                self.dead_pose_env0[agent_idx] = pose
        
        # 取当前玩家智能体的reward作为当前step显示值
        player_idx = self._get_player_agent_index()
        if player_idx >= 0:
            # 确定从哪个世界获取reward
            if hasattr(self, '_current_view_world'):
                world_idx = self._current_view_world
                # 从对应世界获取reward
                world_reward = reward[world_idx]
                self.current_step_reward = float(world_reward[player_idx].item())
            else:
                # 默认从世界0获取reward
                self.current_step_reward = float(self.last_reward_env0[player_idx].item())
        else:
            # 没有存活的智能体，尝试显示第一个智能体的reward（包括已死亡的）
            if self.last_reward_env0 is not None and self.last_reward_env0.numel() > 0:
                # 显示第一个智能体的reward（可能是死亡时的reward）
                self.current_step_reward = float(self.last_reward_env0[0].item())
            else:
                self.current_step_reward = 0.0
        self.step_count += 1
        
        # 检查是否需要进行rollout更新（模仿ddppo.py）
        if self.buffer_step_count >= self.rollout_length or self.step_count >= self.max_steps:
            if self.step_count >= self.max_steps:
                print(f"🎯 第 {self.iteration_count} 个iteration - 达到最大步数 {self.max_steps}，强制开始PPO更新...")
                # 达到最大步数，训练完毕后直接进入新iteration
                self.perform_ppo_update()
                print("🔄 达到最大步数，强制开启新iteration...")
                self.reset_simulator()
            else:
                print(f"🎯 第 {self.iteration_count} 个iteration - 达到rollout长度 {self.rollout_length}，开始PPO更新...")
                self.perform_ppo_update()
                
                # 检查是否所有世界都没有存活agents，如果是则开启新iteration
                if self._check_all_worlds_no_alive_agents():
                    print("🔄 所有世界都没有存活agents，开启新iteration...")
                    self.reset_simulator()
                else:
                    print("✅ 仍有世界有存活agents，继续下一个128step...")
                    # 重置累积的done状态，因为我们要开始新的128步收集
                    self.last_done_all_worlds = None
                    self.reset_buffers()
    
    def reset_simulator(self):
        """重置simulator，模仿ddppo.py - 相当于进入新的iteration"""
        self.iteration_count += 1
        print(f"🔄 开始第 {self.iteration_count} 个iteration，重置simulator...")
        
        initial_observation = self.simulator.reset()
        self.path_plan = self.simulator.agents_path_plans
        self.stop_lines = self.simulator.stop_lines
        
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.path_plan,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )
        
        # 重置死亡状态
        self.dead_mask_env0 = None
        self.dead_pose_env0 = {}
        self.dead_mask_all_worlds = None
        self.dead_pose_all_worlds = {}
        self.last_done_env0 = None
        self.last_done_all_worlds = None
        
        # 重置第一个世界死亡标志
        self.first_world_dead = False
        
        # 重置视角世界（新iteration从世界0开始）
        self._current_view_world = 0
        
        # 重置step计数（新iteration从头开始）
        self.step_count = 0
        self.episode_reward = 0.0
        
        # 清空所有步骤相关的记录
        self.current_step_reward = 0.0
        self.last_reward_env0 = None
        
        # 重置所有buffer（重要：新iteration必须清空buffer）
        self.reset_buffers()
        
        print(f"✅ 第 {self.iteration_count} 个iteration开始，Simulator重置完成")
    
    def perform_ppo_update(self):
        """执行PPO更新，模仿ddppo.py的逻辑 - 所有世界完成后的经验采样训练"""
        if len(self.states_buffer) == 0:
            print("⚠️ Buffer为空，无法进行PPO更新")
            return
        
        print(f"🎯 开始经验采样训练，Buffer长度: {len(self.states_buffer)}")
        
        # 将buffer转换为tensor
        T = len(self.states_buffer)
        B, M, S = self.states_buffer[0].shape
        
        # 构建tensor buffer
        states_tensor = torch.stack(self.states_buffer, dim=0)  # (T, B, M, S)
        rewards_tensor = torch.stack(self.rewards_buffer, dim=0)  # (T, B, M)
        dones_tensor = torch.stack(self.dones_buffer, dim=0)  # (T, B, M)
        values_tensor = torch.stack(self.values_buffer, dim=0)  # (T, B, M)
        old_log_probs_tensor = torch.stack(self.old_log_probs_buffer, dim=0)  # (T, B, M)
        actions_tensor = torch.stack(self.actions_buffer, dim=0)  # (T, B, M)
        
        # 计算最后一个状态的价值（bootstrap）
        with torch.no_grad():
            last_value_pred = self.model.forward(self.features_tensor, mode="value")
        
        # 构建values_tp1用于GAE计算
        values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)
        
        # 计算GAE优势
        advantages, returns = self.gae_advantages(rewards_tensor, values_tp1, dones_tensor, self.gamma, self.gae_lambda)
        
        # 优势过滤（模仿ddppo.py）
        A_max = torch.max(torch.abs(advantages)).item()
        if self.A_max_ewma is None:
            self.A_max_ewma = A_max
        else:
            self.A_max_ewma = self.beta * A_max + (1 - self.beta) * self.A_max_ewma
        eta = self.advantage_filter_threshold * self.A_max_ewma
        keep_mask = (torch.abs(advantages) >= eta)
        cand_idx = keep_mask.nonzero(as_tuple=False)
        
        print(f"🎯 第 {self.iteration_count} 个iteration - 最大|A|: {A_max:.4f}, EWMA: {self.A_max_ewma:.4f}, 阈值: {eta:.4f}")
        print(f"📊 过滤前: {keep_mask.numel()}, 被过滤: {keep_mask.sum().item()}")
        
        if cand_idx.numel() == 0:
            print("⚠️ 无可用样本，跳过更新")
            return
        
        # 随机选择 batch_size_per_gpu 个样本（不足则放回采样）
        N = cand_idx.shape[0]
        K = self.batch_size_per_gpu
        if N >= K:
            rand_pos = torch.randperm(N, device=self.device)[:K]
            selected_idx = cand_idx[rand_pos]
        else:
            rand_pos = torch.randint(0, N, (K,), device=self.device)
            selected_idx = cand_idx[rand_pos]
        
        selected_t = selected_idx[:, 0]
        selected_b = selected_idx[:, 1]
        selected_m = selected_idx[:, 2]
        print(f"🎯 随机选取 {K} 个样本用于更新（候选 {N}）")
        
        # 提取选中的样本
        agent_indices_batch = selected_m.to(self.device)
        old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m].view(-1)
        advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
        returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1)
        
        batch_N = old_log_probs_batch.shape[0]
        print(f"🎯 开始PPO更新，样本数量: {batch_N}")
        
        # PPO更新循环
        for epoch in range(self.ppo_epochs):
            # 重新生成观测和特征
            mb_idx = torch.arange(batch_N, device=self.device)
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
            uniq_tb_mb, inverse_mb = torch.unique(torch.stack([mb_t, mb_b], dim=1), dim=0, return_inverse=True)
            t_u_mb = uniq_tb_mb[:, 0]
            b_u_mb = uniq_tb_mb[:, 1]
            agents_states_mb = states_tensor[t_u_mb, b_u_mb]
            
            # 生成观测
            obs_mb = self.simulator.observation_generator.generate(agents_states_mb)
            agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, self.config_ns)
            
            # 构建特征 - 确保batch size匹配
            batch_size_mb = agents_state_dec_mb.shape[0]
            path_plan_mb = self.path_plan[b_u_mb]  # 形状应该是 (unique_batches, M, L, 2)
            stop_lines_mb = self.stop_lines[b_u_mb] if (self.stop_lines is not None and self.stop_lines.numel() > 0) else self.stop_lines
            reward_coef_mb = self.simulator.reward_calculator.sampled_params[b_u_mb]
            
            # 确保所有tensor的batch size一致
            if path_plan_mb.shape[0] != batch_size_mb:
                # 如果batch size不匹配，使用expand来匹配
                path_plan_mb = path_plan_mb.expand(batch_size_mb, -1, -1, -1)
            if stop_lines_mb is not None and stop_lines_mb.shape[0] != batch_size_mb:
                stop_lines_mb = stop_lines_mb.expand(batch_size_mb, -1, -1)
            if reward_coef_mb.shape[0] != batch_size_mb:
                reward_coef_mb = reward_coef_mb.expand(batch_size_mb, -1, -1)
            
            features_u_mb = build_network_features(
                agents_state_dec_mb,
                neighbors_local_mb,
                w_lanes_local_mb,
                w_boundaries_local_mb,
                path_plan_mb,
                stop_lines_mb,
                reward_coef_mb,
                self.config_ns
            )
            u_idx_mb = inverse_mb.to(self.device)
            mb_features = features_u_mb[u_idx_mb]
            
            # 策略更新
            action_logits = self.model.forward(mb_features, mode="policy")
            action_dist = torch.distributions.Categorical(logits=action_logits)
            
            # 创建与logits形状匹配的actions tensor
            # mb_actions形状: [3000], 需要reshape为 [3000, 1] 然后expand为 [3000, 150]
            actions_for_log_prob = mb_actions.unsqueeze(1).expand(-1, action_logits.shape[1])
            new_log_probs = action_dist.log_prob(actions_for_log_prob)
            
            # 获取对应agents的log概率
            row_idx = torch.arange(mb_actions.shape[0], device=self.device)
            new_log_probs = new_log_probs[row_idx, mb_agent_idx]
            
            # 计算比率和损失
            ratio = torch.exp(torch.clamp(new_log_probs - mb_old_logp, -1, 1))
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy = action_dist.entropy()[row_idx, mb_agent_idx].mean()
            policy_total_loss = policy_loss - self.entropy_coef * entropy
            
            # 策略网络更新
            self.policy_optimizer.zero_grad()
            policy_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy_network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # 价值网络更新
            value_pred_full = self.model.forward(mb_features, mode="value").squeeze(-1)
            value_pred = value_pred_full[row_idx, mb_agent_idx]
            value_loss = (value_pred - mb_ret).pow(2).mean()
            value_loss = self.value_loss_coef * value_loss
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.value_network.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            print(f"   Epoch {epoch+1}/{self.ppo_epochs}: Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}, Entropy: {entropy.item():.6f}")
        
        print(f"✅ 第 {self.iteration_count} 个iteration - 经验采样训练完成")

# ========================= 绘图 =================================

    def draw(self):
        """绘制游戏画面（从 TeraflowSimulator 读取状态与道路）。"""
        # 清屏
        self.screen.fill(self.colors['grass'])
        # 绘制道路
        self.draw_road_from_network()
        # 绘制目标（来自 simulator 的 agents_goal_quad_ids -> quad centers）
        try:
            goal_ids = getattr(self.simulator, 'agents_goal_quad_ids', None)
            if goal_ids is not None:
                # 确定当前视角来自哪个世界
                if hasattr(self, '_current_view_world'):
                    world_idx = self._current_view_world
                else:
                    world_idx = 0  # 默认世界0
                
                world_goal_ids = goal_ids[world_idx]
                # 取当前玩家智能体的目标
                player_idx = self._get_player_agent_index()
                if player_idx >= 0:
                    goal_quad_id = int(world_goal_ids[player_idx].item())
                    if goal_quad_id >= 0:
                        quad_centers = self.simulator.road_network.quad_centerlines.mean(dim=1)
                        gx, gy = quad_centers[goal_quad_id]
                        goal_screen_pos = self.convert_world_to_screen(torch.tensor([[gx, gy]], dtype=torch.float32, device=self.device))
                        if goal_screen_pos:
                            pygame.draw.circle(self.screen, self.colors['goal'], goal_screen_pos[0], int(10.0))
                        else:
                            self.draw_goal_indicator()
        except Exception:
            pass
        
        # 绘制玩家路径（模仿simulator.py中的路径绘制）
        self.draw_player_path()
        
        # 绘制来自模拟器的智能体（当前视角世界内激活体）；死亡的车辆变色并停在死亡时姿态
        # 确定当前视角来自哪个世界
        if hasattr(self, '_current_view_world'):
            view_world_idx = self._current_view_world
        else:
            view_world_idx = 0  # 默认世界0
        
        current_world = self.simulator.agents_state[view_world_idx]
        active_mask = current_world[:, 6] > 0.5
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1).tolist()
        
        for order, agent_idx in enumerate(active_indices):
            # 检查是否死亡（检查当前视角世界的死亡状态）
            is_dead = False
            if hasattr(self, 'dead_mask_all_worlds') and self.dead_mask_all_worlds is not None:
                if view_world_idx < self.dead_mask_all_worlds.shape[0] and agent_idx < self.dead_mask_all_worlds.shape[1]:
                    is_dead = bool(self.dead_mask_all_worlds[view_world_idx, agent_idx].item())
            
            if is_dead and hasattr(self, 'dead_pose_all_worlds') and (view_world_idx, agent_idx) in self.dead_pose_all_worlds:
                # 直接在此处按缓存姿态绘制（避免额外函数）
                x, y, yaw, length, width = self.dead_pose_all_worlds[(view_world_idx, agent_idx)]
                half_l = max(0.1, length * 0.5)
                half_w = max(0.1, width * 0.5)
                local = torch.tensor([
                    [-half_l, -half_w],
                    [ half_l, -half_w],
                    [ half_l,  half_w],
                    [-half_l,  half_w],
                ], dtype=torch.float32, device=self.device)
                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
                world = local @ rot.T + torch.tensor([x, y], dtype=torch.float32, device=self.device)
                pts = self.convert_world_to_screen(world)
                if len(pts) >= 4:
                    pygame.draw.polygon(self.screen, self.colors['dead_car'], pts, width=0)
            else:
                self.draw_car(agent_idx, is_player=(order == 0), color_override=None, world_idx=view_world_idx)
        # 绘制UI
        self.draw_ui()
        # 更新显示
        pygame.display.flip()
    
    def draw_road_from_network(self):
        """使用RoadNetwork数据绘制道路（带视野限制）"""
        # 获取汽车视野范围内的边界点
        visible_boundary_points = self.get_visible_boundary_points()
        if visible_boundary_points.shape[0] == 0:
            # 如果没有边界点，不绘制任何内容
            return
        # 将边界点转换为屏幕坐标
        screen_points = self.convert_world_to_screen(visible_boundary_points)
        # 绘制边界点作为散点
        self.draw_boundary_points(screen_points)
    
    def get_visible_boundary_points(self) -> torch.Tensor:
        """获取汽车视野范围内的边界点（优化版本）"""
        try:
            # 缓存视野点，避免每帧重复计算
            if not hasattr(self, '_cached_visible_points') or not hasattr(self, '_last_car_pos'):
                self._cached_visible_points = None
                self._last_car_pos = None
            # 检查汽车位置是否发生显著变化
            px, py, _ = self._get_player_pose()
            current_car_pos = (px, py)
            if (self._last_car_pos is not None and 
                abs(current_car_pos[0] - self._last_car_pos[0]) < 10.0 and 
                abs(current_car_pos[1] - self._last_car_pos[1]) < 10.0 and
                self._cached_visible_points is not None):
                return self._cached_visible_points
            # 获取所有边界点
            all_boundary_points = self.simulator.road_network.global_w_boundary_points
            if all_boundary_points.shape[0] == 0:
                return all_boundary_points
            # 使用更高效的距离计算
            car_pos = torch.tensor([[current_car_pos[0], current_car_pos[1]]], dtype=torch.float32, device=self.device)
            # 视野范围（米）
            vision_radius = 100.0

            # 使用更高效的距离计算和筛选
            diff = all_boundary_points - car_pos.squeeze(0)
            distances_squared = torch.sum(diff * diff, dim=1)
            visible_mask = distances_squared <= (vision_radius * vision_radius)
            visible_points = all_boundary_points[visible_mask]
            
            # 缓存结果
            self._cached_visible_points = visible_points
            self._last_car_pos = current_car_pos
            return visible_points
        
        except Exception as e:
            print(f"警告: 获取视野边界点失败: {e}")
            return torch.empty((0, 2), device=self.device)
    
    def draw_player_path(self):
        """绘制当前玩家智能体的路径（模仿simulator.py中的路径绘制）"""
        try:
            # 获取路径规划数据
            if not hasattr(self.simulator, 'agents_path_plans') or self.simulator.agents_path_plans is None:
                return
            
            # 确定当前视角来自哪个世界
            if hasattr(self, '_current_view_world'):
                world_idx = self._current_view_world
            else:
                world_idx = 0  # 默认世界0
            
            # 获取当前玩家智能体的索引（在当前视角世界中）
            player_idx = self._get_player_agent_index()
            if player_idx < 0:
                return
            
            # 获取当前玩家的路径 (B, M, L, 2) -> 取 (world_idx, player_idx, :, :)
            path_plan = self.simulator.agents_path_plans[world_idx, player_idx]  # (L, 2)
            
            # 过滤有效路径点（x, y != -1）
            valid_mask = (path_plan[:, 0] != -1) & (path_plan[:, 1] != -1)
            if not valid_mask.any():
                return
            
            valid_path = path_plan[valid_mask]  # (N_valid, 2)
            
            # 将路径点转换为屏幕坐标
            screen_path_points = []
            for point in valid_path:
                world_point = torch.tensor([[point[0].item(), point[1].item()]], dtype=torch.float32, device=self.device)
                screen_points = self.convert_world_to_screen(world_point)
                if screen_points:
                    screen_path_points.append(screen_points[0])
            
            # 绘制路径线（红色虚线，模仿simulator.py）
            if len(screen_path_points) >= 2:
                for i in range(len(screen_path_points) - 1):
                    start_point = screen_path_points[i]
                    end_point = screen_path_points[i + 1]
                    # 使用虚线效果（每隔几个像素画一个点）
                    self.draw_dashed_line(start_point, end_point, (255, 0, 0), 2)
            
            # 绘制起点（红色圆点）
            if screen_path_points:
                pygame.draw.circle(self.screen, (255, 0, 0), screen_path_points[0], 5)
            
            # 绘制终点（红色叉号）
            if len(screen_path_points) >= 2:
                end_point = screen_path_points[-1]
                # 绘制叉号
                cross_size = 2
                pygame.draw.line(self.screen, (255, 0, 0), 
                               (end_point[0] - cross_size, end_point[1] - cross_size),
                               (end_point[0] + cross_size, end_point[1] + cross_size), 3)
                pygame.draw.line(self.screen, (255, 0, 0), 
                               (end_point[0] - cross_size, end_point[1] + cross_size),
                               (end_point[0] + cross_size, end_point[1] - cross_size), 3)
                
        except Exception as e:
            print(f"警告: 绘制玩家路径失败: {e}")
    
    def draw_dashed_line(self, start_point, end_point, color, width):
        """绘制虚线"""
        try:
            x1, y1 = start_point
            x2, y2 = end_point
            
            # 计算距离和方向
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance == 0:
                return
            
            # 虚线参数
            dash_length = 8
            gap_length = 4
            
            # 单位方向向量
            unit_x = dx / distance
            unit_y = dy / distance
            
            # 绘制虚线
            current_distance = 0
            while current_distance < distance:
                # 计算当前段的起点和终点
                start_dist = current_distance
                end_dist = min(current_distance + dash_length, distance)
                
                start_x = int(x1 + start_dist * unit_x)
                start_y = int(y1 + start_dist * unit_y)
                end_x = int(x1 + end_dist * unit_x)
                end_y = int(y1 + end_dist * unit_y)
                
                # 绘制线段
                pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), width)
                
                # 移动到下一个段
                current_distance += dash_length + gap_length
                
        except Exception as e:
            print(f"警告: 绘制虚线失败: {e}")

    def draw_goal_indicator(self):
        """在屏幕边缘绘制目标指示器"""
        try:
            # 从模拟器获取玩家当前位置与目标位置
            px, py, _ = self._get_player_pose()
            goal_ids = getattr(self.simulator, 'agents_goal_quad_ids', None)
            
            # 确定当前视角来自哪个世界
            if hasattr(self, '_current_view_world'):
                world_idx = self._current_view_world
            else:
                world_idx = 0  # 默认世界0
            
            world_goal_ids = goal_ids[world_idx]
            world_states = self.simulator.agents_state[world_idx]
            active_mask = world_states[:, 6] > 0.5
            
            first_idx = int(torch.nonzero(active_mask, as_tuple=False)[0].item())
            goal_quad_id = int(world_goal_ids[first_idx].item())
            
            quad_centers = self.simulator.road_network.quad_centerlines.mean(dim=1)
            gx, gy = quad_centers[goal_quad_id]
            dx = float(gx.item() - px)
            dy = float(gy.item() - py)
            # 计算方向角度
            angle = math.atan2(dy, dx)
            # 在屏幕边缘绘制指示器
            screen_center_x = self.width // 2
            screen_center_y = self.height // 2
            indicator_radius = min(self.width, self.height) // 2 - 30
            indicator_x = screen_center_x + int(indicator_radius * math.cos(angle))
            indicator_y = screen_center_y + int(indicator_radius * math.sin(angle))
            # 绘制指示器
            pygame.draw.circle(self.screen, (255, 255, 0), (indicator_x, indicator_y), 5)
        except Exception as e:
            print(f"警告: 绘制目标指示器失败: {e}")
    
    def draw_boundary_points(self, screen_points: List[Tuple[int, int]]):
        """绘制边界点作为散点（优化版本）"""
        if len(screen_points) == 0:
            return
        # 限制绘制的点数量以提高性能
        max_points = 500
        if len(screen_points) > max_points:
            # 均匀采样点
            step = len(screen_points) // max_points
            screen_points = screen_points[::step]
        # 绘制每个边界点
        point_radius = 1  # 点的大小
        point_color = self.colors['road']  # 道路颜色（黑色）
        for point in screen_points:
            pygame.draw.circle(self.screen, point_color, point, point_radius)
        
    def convert_world_to_screen(self, world_points: torch.Tensor) -> List[Tuple[int, int]]:
        """将世界坐标转换为屏幕坐标（基于汽车视野，优化版本）"""
        if world_points.shape[0] == 0:
            return []
        # 缓存转换参数，避免重复计算
        if not hasattr(self, '_cached_scale_params') or not hasattr(self, '_last_scale_car_pos'):
            self._cached_scale_params = None
            self._last_scale_car_pos = None
        # 检查汽车位置是否发生显著变化
        current_car_pos = self._get_player_pose()[:2]
        if (self._last_scale_car_pos is not None and 
            abs(current_car_pos[0] - self._last_scale_car_pos[0]) < 5.0 and 
            abs(current_car_pos[1] - self._last_scale_car_pos[1]) < 5.0 and
            self._cached_scale_params is not None):
            scale_x, scale_y, car_x, car_y = self._cached_scale_params
        else:
            # 获取汽车位置作为视野中心
            car_x, car_y, _ = self._get_player_pose()

            # 视野范围（米）
            vision_radius = 100.0
            
            # 计算缩放比例，留出边距
            margin = 50
            scale_x = (self.width - 2 * margin) / (2 * vision_radius)
            scale_y = (self.height - 2 * margin) / (2 * vision_radius)
            # 使用较小的缩放比例以保持宽高比
            scale = min(scale_x, scale_y)
            scale_x = scale_y = scale
            # 缓存参数
            self._cached_scale_params = (scale_x, scale_y, car_x, car_y)
            self._last_scale_car_pos = current_car_pos
        
        # 转换坐标（使用向量化操作）不翻转地图（世界y向上 => 屏幕y减小）
        screen_points = []
        for point in world_points:
            # 将世界坐标相对于汽车位置进行偏移
            relative_x = point[0].item() - car_x
            relative_y = point[1].item() - car_y
            # 转换为屏幕坐标
            screen_x = int(relative_x * scale_x + self.width // 2)
            screen_y = int(-relative_y * scale_y + self.height // 2)
            # 检查是否在屏幕范围内
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                screen_points.append((screen_x, screen_y))
        return screen_points
    
    def draw_car(self, agent_idx: int, is_player: bool = True, color_override: tuple = None, world_idx: int = 0):
        """根据 simulator.agents_state 的第5、6字段(长度/宽度)按朝向绘制车辆。"""
        world_agents = self.simulator.agents_state[world_idx]
        x = float(world_agents[agent_idx, 0].item())
        y = float(world_agents[agent_idx, 1].item())
        yaw = float(world_agents[agent_idx, 2].item())
        length = float(world_agents[agent_idx, 4].item())
        width = float(world_agents[agent_idx, 5].item())
        half_l = max(0.1, length * 0.5)
        half_w = max(0.1, width * 0.5)
        # 车辆局部坐标四角点
        local = torch.tensor([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w],
        ], dtype=torch.float32, device=self.device)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
        world = local @ rot.T + torch.tensor([x, y], dtype=torch.float32, device=self.device)
        pts = self.convert_world_to_screen(world)
        if len(pts) < 4:
            return
        color = color_override if color_override is not None else (self.colors['car'] if is_player else self.colors['other_car'])
        pygame.draw.polygon(self.screen, color, pts, width=0)
        # 朝向线
        center_screen = self.convert_world_to_screen(torch.tensor([[x, y]], dtype=torch.float32, device=self.device))
        front_world = torch.tensor([[x + half_l * cos_y, y + half_l * sin_y]], dtype=torch.float32, device=self.device)
        front_screen = self.convert_world_to_screen(front_world)
        if center_screen and front_screen:
            pygame.draw.line(self.screen, (255, 255, 0), center_screen[0], front_screen[0], 2)
  
    def draw_ui(self):
        """绘制用户界面"""
        # 根据第一个世界状态显示不同的UI
        if hasattr(self, 'first_world_dead') and self.first_world_dead:
            # 第一个世界死亡时，显示训练继续状态
            training_text = self.font.render("FIRST WORLD DEAD - Training continues...", True, (255, 165, 0))
            step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
            iteration_text = self.font.render(f"Iteration: {self.iteration_count}", True, self.colors['text'])
            buffer_text = self.font.render(f"Buffer steps: {self.buffer_step_count}/{self.rollout_length}", True, self.colors['text'])
            self.screen.blit(training_text, (10, 10))
            self.screen.blit(step_text, (10, 50))
            self.screen.blit(iteration_text, (10, 90))
            self.screen.blit(buffer_text, (10, 130))
        else:
            # 游戏进行中，显示正常的reward和step信息
            step_reward = getattr(self, 'current_step_reward', 0.0)
            reward_text = self.font.render(f"step reward: {step_reward:.4f}", True, self.colors['text'])
            step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
            iteration_text = self.font.render(f"Iteration: {self.iteration_count}", True, self.colors['text'])
            buffer_text = self.font.render(f"Buffer steps: {self.buffer_step_count}/{self.rollout_length}", True, self.colors['text'])
            self.screen.blit(reward_text, (10, 10))
            self.screen.blit(step_text, (10, 50))
            self.screen.blit(iteration_text, (10, 90))
            self.screen.blit(buffer_text, (10, 130))

        # 显示玩家车辆状态
        px, py, pyaw = self._get_player_pose()
        if hasattr(self, 'first_world_dead') and self.first_world_dead:
            # 第一个世界死亡时，显示静态信息
            static_text = self.small_font.render("First world vehicles inactive", True, (255, 100, 100))
            self.screen.blit(static_text, (10, 200))
        else:
            # 游戏进行中，显示动态信息
            player_idx = self._get_player_agent_index()
            if player_idx >= 0:
                env0_states = self.simulator.agents_state[0]
                speed_val = float(env0_states[player_idx, 3].item())
                
                # 显示离散动作信息（来自上一步采样的动作）
                a_idx = 0  # 默认值
                if hasattr(self, 'last_actions'):
                    actions_cpu = self.last_actions
                    a_idx = int(actions_cpu[0, player_idx].item())
            else:
                speed_val = 0.0
                a_idx = 0
                
            speed_text = self.small_font.render(f"speed: {speed_val:.2f} m/s", True, self.colors['text'])
            heading_text = self.small_font.render(f"heading: {math.degrees(pyaw):.1f}°", True, self.colors['text'])
            action_text = self.small_font.render(f"current action idx: {a_idx}", True, self.colors['text'])
            
            self.screen.blit(speed_text, (10, 200))
            self.screen.blit(heading_text, (10, 220))
            self.screen.blit(action_text, (10, 260))
        
        # 显示汽车世界坐标
        car_world_pos = f"car world pos: ({px:.1f}, {py:.1f})"
        car_pos_text = self.small_font.render(car_world_pos, True, self.colors['text'])
        self.screen.blit(car_pos_text, (10, 300))
        
        # 显示目标世界坐标
        if hasattr(self, 'goal') and self.goal:
            goal_world_pos = f"goal world pos: ({self.goal['x']:.1f}, {self.goal['y']:.1f})"
            goal_pos_text = self.small_font.render(goal_world_pos, True, self.colors['text'])
            self.screen.blit(goal_pos_text, (10, 320))
        
        # 显示视野信息
        vision_info = f"vision radius: 100m, visible points: {len(self.get_visible_boundary_points())}"
        vision_text = self.small_font.render(vision_info, True, self.colors['text'])
        self.screen.blit(vision_text, (10, 340))
        
        # 显示车辆信息（env0 存活/死亡数量）
        env0 = self.simulator.agents_state[0]
        active_mask = (env0[:, 6] > 0.5).detach().to('cpu')  # active状态，转换为CPU
        
        # 获取done状态，如果没有则默认为False
        if hasattr(self, 'last_done_env0') and self.last_done_env0 is not None:
            done_mask = self.last_done_env0
        else:
            done_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        
        # 真正存活的车辆：active=True 且 done=False
        alive_mask = active_mask & (~done_mask)
        alive_num = int(alive_mask.sum().item())
        
        # 死亡车辆：active=True 且 done=True（只计算原本存在但已死亡的车辆）
        dead_mask = active_mask & done_mask
        dead_num = int(dead_mask.sum().item())
        
        # 总车辆数：active=True的车辆总数
        total_active = int(active_mask.sum().item())
        
        car_info = f"alive: {alive_num}, dead: {dead_num}, total: {total_active}"
        car_text = self.small_font.render(car_info, True, self.colors['text'])
        self.screen.blit(car_text, (10, 360))
        
        # 显示多世界状态信息
        try:
            all_agents_state = self.simulator.agents_state  # (B, M, S)
            B, M, S = all_agents_state.shape
            
            # 显示当前视角世界信息
            if hasattr(self, '_current_view_world'):
                current_view_world = self._current_view_world
            else:
                current_view_world = 0
            
            world_info = f"Worlds: {B}, Agents per world: {M}, View: World {current_view_world}"
            world_text = self.small_font.render(world_info, True, self.colors['text'])
            self.screen.blit(world_text, (10, 380))
            
            # 显示每个世界的存活状态
            alive_worlds = 0
            for b in range(B):
                world_agents = all_agents_state[b]
                active_mask = world_agents[:, 6] > 0.5
                if active_mask.any():
                    if hasattr(self.simulator, 'done') and self.simulator.done is not None:
                        world_done = self.simulator.done[b]
                        alive_mask = active_mask & (~world_done)
                        if alive_mask.any():
                            alive_worlds += 1
                    else:
                        alive_worlds += 1
            
            alive_worlds_info = f"Alive worlds: {alive_worlds}/{B}"
            alive_worlds_text = self.small_font.render(alive_worlds_info, True, self.colors['text'])
            self.screen.blit(alive_worlds_text, (10, 400))
        except Exception:
            pass
        
    def run(self):
        """运行游戏主循环（可视化）。"""
        while self.running:
            # 处理事件，避免窗口无响应 别删
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
            # 推进模拟器并绘制
            self.update_game_state()
            # 绘制画面
            self.draw()
            # 帧率限制
            if self.clock is not None:
                self.clock.tick(60)
    
if __name__ == "__main__":
    game = CarGame()
    game.run()

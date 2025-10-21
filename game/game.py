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
import matplotlib.pyplot as plt
import swanlab

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
        self.stop_lines = self.simulator.stop_lines

        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.simulator.agents_path_plans_local,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )
        # 缓存用于可视化的邻居观测
        self.last_neighbors_local = neighbors_local
        # 缓存用于可视化的边界观测（oob参考）
        self.last_w_boundaries_local = w_boundaries_local

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
        self.entropy_coef = float(self.config.get('training').get('entropy_coef'))
        self.value_loss_coef = float(self.config.get('training').get('value_loss_coef', 0.5))
        self.max_grad_norm = float(self.config.get('training').get('max_grad_norm', 1.0))
        self.learning_rate = float(self.config.get('training').get('learning_rate'))
        
        # 优势过滤参数（模仿ddppo.py）
        self.beta = float(self.config.get('training').get('advantage_filter_beta', 0.25))
        self.advantage_filter_threshold = float(self.config.get('training').get('advantage_filter_threshold', 0.01))
        self.A_max_ewma = None
        self.batch_size_per_gpu = int(self.config.get('training').get('batch_size_per_gpu', 32000))
        
        # 初始化优化器
        self.policy_optimizer = torch.optim.Adam(self.model.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.model.value_network.parameters(), lr=self.learning_rate)

        # 余弦退火学习率调度器（模仿ddppo.py）
        total_iterations = int(self.config.get('training').get('iteration', 1000))
    
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=total_iterations, eta_min=0.0)
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.value_optimizer, T_max=total_iterations, eta_min=0.0)
        
        # 初始化buffer
        self.reset_buffers()

        # 训练度量与可视化
        self.update_index = 0
        self.avg_rewards_per_update = []
        self.policy_losses_per_update = []
        self.value_losses_per_update = []
        self._init_plots()
        
        # 路径观察长度动态调整相关
        self.path_observation_length = 2  # 初始路径观察长度
        self.reward_positive_count = 0  # 连续正奖励计数
        self.reward_positive_threshold = 1  # 需要连续3次正奖励才增加路径长度

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

        # 控制模式：默认自动模式（相当于启动时自动按F1）
        self.auto_mode = True
        # 待执行的手动动作索引（按键触发后生效一次）
        self.pending_manual_action_idx = None
        # 键位映射到动作索引（qwertyuiop[] -> 12个离散动作）
        self.key_to_action = {
            pygame.K_q: 0,  pygame.K_w: 1,  pygame.K_e: 2,  pygame.K_r: 3,
            pygame.K_t: 4,  pygame.K_y: 5,  pygame.K_u: 6,  pygame.K_i: 7,
            pygame.K_o: 8,  pygame.K_p: 9,  pygame.K_LEFTBRACKET: 10,  pygame.K_RIGHTBRACKET: 11,
        }

        # 创建一个SwanLab项目
        swanlab.init(
            # 设置项目名
            project="selfrace",
            # 设置超参数
            config={
                "learning_rate": self.learning_rate,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": total_iterations
            }
        )
    
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
        """GAE优势计算，修复done状态的处理"""
        T = rewards.shape[0]
        done_mask = dones.to(rewards.dtype)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        # 从最后一个时间步开始向前计算
        for t in range(T - 1, -1, -1):
            # 计算TD误差：delta = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            # 注意：done_t表示在时间步t之后是否结束，所以done_t=1时，V(s_{t+1})应该为0
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - done_mask[t]) - values[t]
            # 计算GAE：gae_t = delta_t + γ * λ * (1 - done_t) * gae_{t+1}
            gae = delta + gamma * gae_lambda * (1.0 - done_mask[t]) * gae
            advantages[t] = gae
        
        # 计算returns = advantages + values[:-1]
        returns = advantages + values[:-1]
        return advantages, returns

    def _init_plots(self):
        """初始化Matplotlib交互式窗口与轴"""
        try:
            # 使用交互模式，避免阻塞pygame主循环
            plt.ion()
            # 单独起一个窗口，三个子图：上-损失；中-平均回报；下-动作概率分布
            self.fig, (self.ax_loss, self.ax_reward, self.ax_action_prob) = plt.subplots(3, 1, figsize=(8, 10))
            self.fig.canvas.manager.set_window_title("PPO Metrics & Action Probabilities")
            # 初始空图（左右双坐标轴）：左轴-Value Loss；右轴-Policy Loss
            self.ax_loss_right = self.ax_loss.twinx()
            self.loss_value_line, = self.ax_loss.plot([], [], label="Value Loss", color="tab:blue")
            self.loss_policy_line, = self.ax_loss_right.plot([], [], label="Policy Loss", color="tab:red")
            self.ax_loss.set_xlabel("Update")
            self.ax_loss.set_ylabel("Value Loss")
            self.ax_loss_right.set_ylabel("Policy Loss")
            # 合并图例
            lines = [self.loss_value_line, self.loss_policy_line]
            labels = [l.get_label() for l in lines]
            self.ax_loss.legend(lines, labels, loc="best")
            self.ax_loss.grid(True, linestyle=":", alpha=0.4)

            self.reward_line, = self.ax_reward.plot([], [], color="tab:green", label="Avg Reward")
            self.ax_reward.set_xlabel("Update")
            self.ax_reward.set_ylabel("Avg Reward")
            self.ax_reward.legend(loc="best")
            self.ax_reward.grid(True, linestyle=":", alpha=0.4)

            # 动作概率分布子图
            self.ax_action_prob.set_xlabel("Action Index")
            self.ax_action_prob.set_ylabel("Probability")
            self.ax_action_prob.set_title("Current Vehicle Action Probability Distribution")
            self.ax_action_prob.grid(True, linestyle=":", alpha=0.4)
            # 初始化12个动作的条形图
            self.action_bars = self.ax_action_prob.bar(range(12), [0]*12, color='skyblue', alpha=0.7)
            self.ax_action_prob.set_xlim(-0.5, 11.5)
            self.ax_action_prob.set_ylim(0, 1.0)
            # 设置x轴标签
            self.ax_action_prob.set_xticks(range(12))
            self.ax_action_prob.set_xticklabels([f'A{i}' for i in range(12)])
            # 添加文本显示当前观测车辆信息
            self.vehicle_info_text = self.ax_action_prob.text(0.02, 0.95, '', transform=self.ax_action_prob.transAxes, 
                                                             verticalalignment='top', fontsize=10, 
                                                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            # 可视化失败不影响训练
            self.fig = None
            self.ax_loss = None
            self.ax_loss_right = None
            self.ax_reward = None
            self.ax_action_prob = None

    def _update_plots(self):
        """刷新曲线数据（非阻塞）"""
        if self.ax_loss is None or self.ax_reward is None:
            return
        try:
            x = list(range(1, self.update_index + 1))
            # 更新损失
            self.loss_policy_line.set_data(x, self.policy_losses_per_update)
            self.loss_value_line.set_data(x, self.value_losses_per_update)
            # 自适应坐标范围（左右轴分别自适应）
            if len(x) > 0:
                self.ax_loss.set_xlim(1, max(5, self.update_index))
                # 左轴：Value Loss
                if len(self.value_losses_per_update) > 0:
                    vmin = min(self.value_losses_per_update)
                    vmax = max(self.value_losses_per_update)
                else:
                    vmin, vmax = 0.0, 1.0
                if vmin == vmax:
                    vmax = vmin + 1.0
                pad_v = 0.05 * abs(vmax)
                self.ax_loss.set_ylim(vmin - pad_v, vmax + pad_v)
                # 右轴：Policy Loss
                if hasattr(self, 'ax_loss_right') and self.ax_loss_right is not None:
                    if len(self.policy_losses_per_update) > 0:
                        pmin = min(self.policy_losses_per_update)
                        pmax = max(self.policy_losses_per_update)
                    else:
                        pmin, pmax = 0.0, 1.0
                    if pmin == pmax:
                        pmax = pmin + 1.0
                    pad_p = 0.05 * abs(pmax)
                    self.ax_loss_right.set_ylim(pmin - pad_p, pmax + pad_p)

            # 更新回报
            self.reward_line.set_data(x, self.avg_rewards_per_update)
            if len(x) > 0:
                self.ax_reward.set_xlim(1, max(5, self.update_index))
                rmin = min(self.avg_rewards_per_update) if len(self.avg_rewards_per_update) > 0 else 0.0
                rmax = max(self.avg_rewards_per_update) if len(self.avg_rewards_per_update) > 0 else 1.0
                if rmin == rmax:
                    rmax = rmin + 1.0
                self.ax_reward.set_ylim(rmin - 0.05 * abs(rmax), rmax + 0.05 * abs(rmax))

            # 更新动作概率分布
            if hasattr(self, 'ax_action_prob') and self.ax_action_prob is not None:
                self._update_action_probabilities()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def _update_action_probabilities(self):
        """更新当前观测车辆的动作概率分布显示"""
        try:
            # 获取当前观测的车辆索引
            b_obs, m_obs = self._get_player_agent_bm()
            if b_obs == -1 or m_obs == -1:
                # 没有可观测的车辆，显示零概率
                for i, bar in enumerate(self.action_bars):
                    bar.set_height(0.0)
                if hasattr(self, 'vehicle_info_text'):
                    self.vehicle_info_text.set_text('No active vehicle')
                return

            # 更新车辆信息文本
            if hasattr(self, 'vehicle_info_text'):
                mode_text = 'Auto' if self.auto_mode else 'Manual'
                self.vehicle_info_text.set_text(f'Vehicle B={b_obs}, M={m_obs} ({mode_text})')

            # 获取当前车辆的动作概率
            action_probs = self._get_current_action_probabilities(b_obs, m_obs)
            
            # 更新条形图高度
            for i, bar in enumerate(self.action_bars):
                if i < len(action_probs):
                    bar.set_height(action_probs[i])
                else:
                    bar.set_height(0.0)
            
            # 高亮当前选择的动作
            if hasattr(self, 'last_actions') and self.last_actions is not None:
                if b_obs < self.last_actions.shape[0] and m_obs < self.last_actions.shape[1]:
                    current_action = self.last_actions[b_obs, m_obs].item()
                    # 重置所有条形颜色
                    for bar in self.action_bars:
                        bar.set_color('skyblue')
                    # 高亮当前动作
                    if 0 <= current_action < len(self.action_bars):
                        self.action_bars[current_action].set_color('red')
                        
        except Exception as e:
            # 如果出错，显示零概率
            for bar in self.action_bars:
                bar.set_height(0.0)
            if hasattr(self, 'vehicle_info_text'):
                self.vehicle_info_text.set_text('Error loading probabilities')

    def _get_current_action_probabilities(self, b_idx, m_idx):
        """获取指定车辆的动作概率分布"""
        try:
            if not self.auto_mode:
                # 手动模式：返回one-hot分布
                if hasattr(self, 'last_actions') and self.last_actions is not None:
                    if b_idx < self.last_actions.shape[0] and m_idx < self.last_actions.shape[1]:
                        current_action = self.last_actions[b_idx, m_idx].item()
                        probs = [0.0] * 12
                        if 0 <= current_action < 12:
                            probs[current_action] = 1.0
                        return probs
                return [0.0] * 12
            
            # 自动模式：通过网络获取概率分布
            if hasattr(self, 'features_tensor') and self.features_tensor is not None:
                with torch.no_grad():
                    # 获取当前车辆的特征
                    if b_idx < self.features_tensor.shape[0] and m_idx < self.features_tensor.shape[1]:
                        vehicle_features = self.features_tensor[b_idx:b_idx+1, m_idx:m_idx+1]
                        action_logits = self.model.forward(vehicle_features, mode="policy")
                        action_probs = torch.softmax(action_logits, dim=-1)
                        return action_probs[0, 0].cpu().numpy().tolist()
            
            return [0.0] * 12
        except Exception:
            return [0.0] * 12

    def _check_all_worlds_no_alive_agents(self) -> bool:
        """检查是否所有世界都没有存活agents（用于决定是否开启新iteration）。"""
        if self.simulator is None:
            return True
        try:
            # 检查所有世界的agents状态
            all_agents_state = self.simulator.agents_state  # (B, M, S)
            B, M, S = all_agents_state.shape
            # 如果没有done状态记录，检查是否有active的agents
            if not hasattr(self, 'cumulative_done_all') or self.cumulative_done_all is None:
                # 初始化时，只要有active的agents就认为有存活的
                for b in range(B):
                    world_agents = all_agents_state[b]  # (M, S)
                    active_mask = world_agents[:, 6] > 0.5  # active状态
                    if active_mask.any():
                        return False  # 有active的agents，认为有存活的
                return True  # 没有active的agents
            # 快速检查，无多余日志
            # 检查每个世界是否有存活的agents
            for b in range(B):
                world_agents = all_agents_state[b]  # (M, S)
                active_mask = world_agents[:, 6] > 0.5  # active状态
                # 如果有active的agents，检查是否有存活的
                if active_mask.any():
                    world_done = self.cumulative_done_all[b].to(active_mask.device)  # 使用累积done状态
                    alive_mask = active_mask & (~world_done)
                    if alive_mask.any():
                        return False  # 这个世界还有存活的agents
            return True  # 所有世界都没有存活的agents
        except Exception:
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
        
        # 如果没有done状态记录，返回第一个active的agent
        if not hasattr(self, 'cumulative_done_all') or self.cumulative_done_all is None:
            for b in range(B):
                world_agents = all_agents_state[b]  # (M, S)
                active_mask = world_agents[:, 6] > 0.5  # active状态
                if active_mask.any():
                    idx = torch.nonzero(active_mask, as_tuple=False)
                    if idx.numel() > 0:
                        return b, idx[0, 0].item()
            return -1, -1
        
        for b in range(B):
            world_agents = all_agents_state[b]  # (M, S)
            active_mask = world_agents[:, 6] > 0.5  # active状态
            
            if active_mask.any():
                # 检查是否有存活的agents
                world_done = self.cumulative_done_all[b].to(active_mask.device)
                alive_mask = active_mask & (~world_done)
                if alive_mask.any():
                    idx = torch.nonzero(alive_mask, as_tuple=False)
                    if idx.numel() > 0:
                        return b, idx[0, 0].item()
        
        return -1, -1  # 没有找到存活的智能体

    def _get_player_agent_bm(self) -> Tuple[int, int]:
        """返回当前用于可视化的玩家智能体(B, M)索引。
        优先在当前视角world内选择还活着的agent；若当前视角world无活体，则选择任意world内的活体并切换视角。
        若均无，则返回(-1, -1)。"""
        if self.simulator is None:
            return -1, -1
        all_agents_state = self.simulator.agents_state  # (B, M, S)
        B, M, _ = all_agents_state.shape
        
        # 如果没有done状态记录，返回第一个active的agent
        if not hasattr(self, 'cumulative_done_all') or self.cumulative_done_all is None:
            # 优先当前视角world
            view_world_idx = getattr(self, '_current_view_world', 0)
            try_order = list(range(B))
            if view_world_idx in try_order:
                try_order.remove(view_world_idx)
                try_order.insert(0, view_world_idx)
            for b in try_order:
                world_agents = all_agents_state[b]
                active_mask = world_agents[:, 6] > 0.5
                if active_mask.any():
                    idx = torch.nonzero(active_mask, as_tuple=False)
                    if idx.numel() > 0:
                        if getattr(self, '_current_view_world', None) != b:
                            self._current_view_world = b
                        return b, idx[0, 0].item()
            return -1, -1
            
        # 优先当前视角world
        view_world_idx = getattr(self, '_current_view_world', 0)
        try_order = list(range(B))
        if view_world_idx in try_order:
            try_order.remove(view_world_idx)
            try_order.insert(0, view_world_idx)
        for b in try_order:
            world_agents = all_agents_state[b]
            active_mask = world_agents[:, 6] > 0.5
            if not active_mask.any():
                continue
            world_done = self.cumulative_done_all[b].to(active_mask.device)
            alive_mask = active_mask & (~world_done)
            idx = torch.nonzero(alive_mask, as_tuple=False)
            if idx.numel() > 0:
                if getattr(self, '_current_view_world', None) != b:
                    self._current_view_world = b
                return b, idx[0, 0].item()
        return -1, -1

    def _get_player_pose(self) -> Tuple[float, float, float]:
        """选择第一个仍然 alive 的智能体（active 且未 dead）的位姿；若无则沿用上一次视角或返回(0,0,0)。"""
        b, m = self._get_player_agent_bm()
        if m >= 0 and b >= 0:
            world_agents = self.simulator.agents_state[b]
            x = world_agents[m, 0].item()
            y = world_agents[m, 1].item()
            yaw = world_agents[m, 2].item()
            self._last_camera_pose = (x, y, yaw)
            return x, y, yaw
        if hasattr(self, '_last_camera_pose'):
            return self._last_camera_pose
        return 0.0, 0.0, 0.0

    def update_game_state(self):
        """推进一步：默认人工模式仅在按下动作键时推进；F1切换自动模式后按原逻辑连续推进。"""
        if self.simulator is None:
            return
        # 若所有world均无存活agent
        if self._check_all_worlds_no_alive_agents():
            if self.auto_mode and self.buffer_step_count > 0:
                self.perform_ppo_update()
            self.reset_simulator()
            return

        # 选择动作：自动模式 -> 网络；人工模式 -> 等待按键
        if self.auto_mode:
            with torch.no_grad():
                action_logits = self.model.forward(self.features_tensor, mode="policy")
                value_pred = self.model.forward(self.features_tensor, mode="value")
            action_dist = torch.distributions.Categorical(logits=action_logits)
            actions = action_dist.sample()
            self.last_actions = actions.detach().to('cpu')
        else:
            # 未按键则不推进
            if self.pending_manual_action_idx is None:
                return
            # 构造整批动作：所有激活智能体同一个动作（简单手动控制）
            B, M, _ = self.simulator.agents_state.shape
            act = int(self.pending_manual_action_idx)
            actions = torch.full((B, M), act, dtype=torch.long, device=self.device)
            self.last_actions = actions.detach().to('cpu')
            value_pred = torch.zeros((B, M), device=self.device)
            # 清除一次性按键
            self.pending_manual_action_idx = None

        # 在推进环境前缓存当前状态
        pre_state = self.simulator.agents_state.clone()

        # 推进一步
        obs, reward, done = self.simulator.step(actions)

        # 写入训练buffer（手动/自动均写入；手动模式用0占位的value，并存动作概率为one-hot）
        self.states_buffer.append(pre_state)
        self.rewards_buffer.append(reward.clone())
        self.dones_buffer.append(done.clone())
        self.values_buffer.append(value_pred.clone())
        if self.auto_mode and 'action_dist' in locals():
            # 存储整分布概率 (B,M,A)
            self.old_log_probs_buffer.append(action_dist.probs.detach().clone())
        else:
            # 手动模式：构造one-hot概率分布 (B,M,A)
            B, M = actions.shape
            num_actions = int(self.simulator.dynamics_model.discrete_action_space.num_actions)
            probs = torch.zeros((B, M, num_actions), device=self.device)
            probs.scatter_(-1, actions.long().unsqueeze(-1), 1.0)
            self.old_log_probs_buffer.append(probs)
        self.actions_buffer.append(actions.clone())
        self.buffer_step_count += 1
        
        # 更新特征以供下一个step使用
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(obs, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.simulator.agents_path_plans_local,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )
        # 更新可视化用邻居观测
        self.last_neighbors_local = neighbors_local
        # 更新可视化用边界观测
        self.last_w_boundaries_local = w_boundaries_local

        # 保存本step用于UI显示的reward（当前视角玩家）
        b_ui, m_ui = self._get_player_agent_bm()
        if b_ui >= 0 and m_ui >= 0:
            self.current_step_reward = float(reward[b_ui, m_ui].item())
        else:
            self.current_step_reward = 0.0
        # 直接使用simulator传递的实时done状态
        self.current_done_all = done.detach().to('cpu').bool()  # (B, M)
        
        # 累积done状态，记录这一轮iteration中done过的车辆
        if not hasattr(self, 'cumulative_done_all') or self.cumulative_done_all is None:
            self.cumulative_done_all = self.current_done_all.clone()
        else:
            self.cumulative_done_all = self.cumulative_done_all | self.current_done_all

        # 死亡（done）处理：标记并保存死亡时姿态（所有世界）
        all_done = self.current_done_all  # 使用实时done状态
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
        
        # current_step_reward 已在上方设置
        self.step_count += 1
        
        # 仅在自动模式下根据rollout进行更新
        if self.auto_mode and (self.buffer_step_count >= self.rollout_length or self.step_count >= self.max_steps):
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
                    # 仅清空采样buffer，保留累计的dones用于可视化与死亡着色
                    self.reset_buffers()
        
        # 每步都更新动作概率分布显示
        if hasattr(self, 'ax_action_prob') and self.ax_action_prob is not None:
            self._update_action_probabilities()
    
    def reset_simulator(self):
        """重置simulator，模仿ddppo.py - 相当于进入新的iteration"""
        self.iteration_count += 1
        print(f"🔄 开始第 {self.iteration_count} 个iteration，重置simulator...")
        
        initial_observation = self.simulator.reset()
        self.stop_lines = self.simulator.stop_lines
        
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.simulator.agents_path_plans_local,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )
        # 重置可视化用邻居观测
        self.last_neighbors_local = neighbors_local
        # 重置可视化用边界观测
        self.last_w_boundaries_local = w_boundaries_local
        
        # 重置死亡/完成状态（保留跨rollout的一致性从新iteration开始）
        self.dead_mask_all_worlds = None
        self.dead_pose_all_worlds = {}
        self.current_done_all = None
        self.cumulative_done_all = None
        
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
        old_log_probs_tensor = torch.stack(self.old_log_probs_buffer, dim=0)  # (T, B, M) or (T, B, M, A)
        actions_tensor = torch.stack(self.actions_buffer, dim=0)  # (T, B, M)
        
        # 计算最后一个状态的价值（bootstrap）
        with torch.no_grad():
            last_value_pred = self.model.forward(self.features_tensor, mode="value")
        
        # 构建values_tp1用于GAE计算
        # 确保维度匹配：values_tensor (T, B, M), last_value_pred (B, M, 1)
        if last_value_pred.dim() == 3 and last_value_pred.shape[-1] == 1:
            last_value_pred = last_value_pred.squeeze(-1)  # (B, M)
        values_tp1 = torch.cat([values_tensor, last_value_pred.unsqueeze(0)], dim=0)
        # 计算GAE优势（使用段内前缀 OR 的累计 done 掩码，符合PPO定义）
        dones_accum = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
        advantages, returns = self.gae_advantages(rewards_tensor, values_tp1, dones_accum, self.gamma, self.gae_lambda)
        
        # 优势过滤（模仿ddppo.py）
        A_max = torch.max(torch.abs(advantages)).item()
        if self.A_max_ewma is None:
            self.A_max_ewma = A_max
        else:
            self.A_max_ewma = self.beta * A_max + (1 - self.beta) * self.A_max_ewma
        eta = self.advantage_filter_threshold * self.A_max_ewma
        keep_mask = (torch.abs(advantages) >= eta)

        # 额外剔除：每个(B,M)智能体在第一次 done 之后的所有时间步（但保留第一次 done 发生的那个时间步）
        # seen_done_inclusive[t] 表示到 t 为止是否出现过 done
        seen_done_inclusive = (torch.cumsum(dones_tensor.to(torch.int32), dim=0) > 0)
        # 第一次出现 done 的时间步
        seen_done_prev = torch.roll(seen_done_inclusive, shifts=1, dims=0)
        seen_done_prev[0] = False
        first_done_step = dones_tensor & (~seen_done_prev)
        # 第一次之后的所有时间步（严格在第一次 done 之后）
        post_done_mask = seen_done_inclusive & (~first_done_step)
        # 从候选中移除这些 post-done 样本
        keep_mask = keep_mask & (~post_done_mask)
        cand_idx = keep_mask.nonzero(as_tuple=False)
        
        print(f"🎯 第 {self.iteration_count} 个iteration - 最大|A|: {A_max:.4f}, EWMA: {self.A_max_ewma:.4f}, 阈值: {eta:.4f}")
        print(f"📊 过滤前: {keep_mask.numel()}, 过滤后: {keep_mask.sum().item()}")
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
        # old_log_probs 兼容两种格式：
        # 1) 标量对数概率 (T,B,M)
        # 2) 概率分布 (T,B,M,A) —— 需按所选动作提取并取log
        if old_log_probs_tensor.dim() == 4:
            action_idx_flat = actions_tensor[selected_t, selected_b, selected_m]
            probs_selected = old_log_probs_tensor[selected_t, selected_b, selected_m, action_idx_flat]
            old_log_probs_batch = torch.log(torch.clamp(probs_selected, min=1e-8))
        else:
            # (T,B,M) 标量logp
            old_log_probs_batch = old_log_probs_tensor[selected_t, selected_b, selected_m]
        old_log_probs_batch = old_log_probs_batch.view(-1)
        advantages_batch = advantages[selected_t, selected_b, selected_m].view(-1)
        returns_batch = returns[selected_t, selected_b, selected_m].view(-1)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        actions_batch = actions_tensor[selected_t, selected_b, selected_m].view(-1)
        batch_N = old_log_probs_batch.shape[0]
        print(f"🎯 开始PPO更新，样本数量: {batch_N}")
        
        # 训练模式
        self.model.train()
        # PPO更新循环
        epoch_policy_losses = []
        epoch_value_losses = []
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
            
            # 获取唯一的状态组合（包含agent索引）
            uniq_tbm, inverse_mb = torch.unique(torch.stack([mb_t, mb_b, mb_m], dim=1), dim=0, return_inverse=True)
            t_u_mb = uniq_tbm[:, 0]
            b_u_mb = uniq_tbm[:, 1]
            m_u_mb = uniq_tbm[:, 2]
            
            # 为每个唯一的(t,b,m)组合生成观测
            agents_states_mb = states_tensor[t_u_mb, b_u_mb]  # (unique_samples, M, S)
            
            # 生成观测
            obs_mb = self.simulator.observation_generator.generate(agents_states_mb)
            agents_state_dec_mb, neighbors_local_mb, w_lanes_local_mb, w_boundaries_local_mb = decompose_observation(obs_mb, self.config_ns)
            
            # 构建特征 - 确保batch size匹配
            batch_size_mb = agents_state_dec_mb.shape[0]
            path_plan_mb = self.simulator.agents_path_plans_local[b_u_mb]  # 形状应该是 (unique_batches, M, L, 2)
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
            
            # 策略更新（逐样本-对应agent构造分布，避免跨agent维度混淆）
            action_logits = self.model.forward(mb_features, mode="policy")
            row_idx = torch.arange(mb_actions.shape[0], device=self.device)
            logits_selected = action_logits[row_idx, mb_agent_idx]  # [N, num_actions]
            dist_selected = torch.distributions.Categorical(logits=logits_selected)
            new_log_probs = dist_selected.log_prob(mb_actions)  # [N]
            
            # 计算比率和损失
            ratio = torch.exp(new_log_probs - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失（基于该agent分布）
            entropy = dist_selected.entropy().mean()
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
            epoch_policy_losses.append(float(policy_loss.item()))
            epoch_value_losses.append(float(value_loss.item()))
            #print(f"   Epoch {epoch+1}/{self.ppo_epochs}: Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}, Entropy: {entropy.item():.6f}")

            swanlab.log({"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()})

        # 切回评估模式
        self.model.eval()
        print(f"✅ 第 {self.iteration_count} 个iteration - 经验采样训练完成")
        # 学习率退火步进
        try:
            if hasattr(self, 'policy_scheduler') and self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if hasattr(self, 'value_scheduler') and self.value_scheduler is not None:
                self.value_scheduler.step()
        except Exception:
            pass

        # 统计并记录本次更新的均值指标
        try:
            # 平均reward（按本次buffer的全部样本）
            #avg_reward_this_update = float(torch.stack(self.rewards_buffer, dim=0).mean().item())
            rewards_tensor = torch.stack(self.rewards_buffer, dim=0)
            avg_reward_this_update = float(rewards_tensor[selected_t, selected_b, selected_m].mean().item())
        except Exception:
            avg_reward_this_update = 0.0
        try:
            mean_policy_loss = float(sum(epoch_policy_losses) / max(1, len(epoch_policy_losses)))
            mean_value_loss = float(sum(epoch_value_losses) / max(1, len(epoch_value_losses)))
        except Exception:
            mean_policy_loss = 0.0
            mean_value_loss = 0.0

        # 记录到 SwanLab
        try:
            swanlab.log({
                "avg_reward": avg_reward_this_update,
                "mean_policy_loss": mean_policy_loss,
                "mean_value_loss": mean_value_loss,
                "update_index": self.update_index + 1,
            })
        except Exception:
            pass

        self.update_index += 1
        self.avg_rewards_per_update.append(avg_reward_this_update)
        self.policy_losses_per_update.append(mean_policy_loss)
        self.value_losses_per_update.append(mean_value_loss)
        
        # 检查平均奖励并调整路径观察长度
        self._check_and_adjust_path_length(avg_reward_this_update)
        
        self._update_plots()
    
    def _check_and_adjust_path_length(self, avg_reward: float):
        """
        检查平均奖励并动态调整路径观察长度
        
        Args:
            avg_reward (float): 当前更新的平均奖励
        """
        # 如果平均奖励大于0，增加连续正奖励计数
        if avg_reward > 0:
            self.reward_positive_count += 1
            print(f"🎯 平均奖励为正: {avg_reward:.4f}, 连续正奖励计数: {self.reward_positive_count}")
        else:
            # 如果平均奖励不为正，重置计数
            self.reward_positive_count = 0
        
        # 当连续正奖励达到阈值且当前路径长度小于最大值时，增加路径长度
        if (self.reward_positive_count >= self.reward_positive_threshold and 
            self.path_observation_length < 128):
            
            # 渐进式增加路径长度，每次增加1
            new_length = min(128, self.path_observation_length + 1)
            self.path_observation_length = new_length
            
            # 更新simulator中的路径观察长度
            if hasattr(self.simulator, 'set_path_observation_length'):
                self.simulator.set_path_observation_length(new_length)
            
            print(f"🚀 路径观察长度已增加到: {self.path_observation_length}")
            
            # 重置计数，避免连续增加
            self.reward_positive_count = 0

# ========================= 绘图 =================================

    def draw(self):
        """绘制游戏画面（从 TeraflowSimulator 读取状态与道路）。"""
        # 清屏
        self.screen.fill(self.colors['grass'])
        # 绘制道路
        self.draw_road_from_network()
        # 绘制视野虚线圆
        self.draw_vision_circle()
        # 已移除导航目标与路径绘制
        # 绘制来自模拟器的智能体（当前视角世界内激活体）；死亡的车辆变色并停在死亡时姿态
        # 确定当前视角来自哪个世界
        if hasattr(self, '_current_view_world'):
            view_world_idx = self._current_view_world
        else:
            view_world_idx = 0  # 默认世界0
        # 仅可视化：玩家车辆 + 通过观测重建的邻居车辆 + oob观测点（边界）
        b_p, m_p = self._get_player_agent_bm()
        if b_p >= 0 and m_p >= 0 and b_p == view_world_idx:
            # 绘制玩家车辆
            self.draw_car(m_p, is_player=True, color_override=None, world_idx=view_world_idx)
            # 绘制玩家的导航信息：绘制路径规划中的所有有效点（局部->世界，散点显示，不连线）
            try:
                if hasattr(self.simulator, 'agents_path_plans_local') and self.simulator.agents_path_plans_local is not None:
                    path_local = self.simulator.agents_path_plans_local[b_p, m_p]  # (L, 2)
                    # 有效点：排除占位的(0,0)
                    if path_local.ndim >= 2 and path_local.shape[-1] >= 2:
                        valid_mask = (path_local[..., 0] != 0) | (path_local[..., 1] != 0)
                        valid_pts_local = path_local[valid_mask]
                        if valid_pts_local.shape[0] > 0:
                            pts_local = valid_pts_local[:, :2]
                            # 局部->世界
                            player_state = self.simulator.agents_state[b_p, m_p]
                            px = float(player_state[0].item())
                            py = float(player_state[1].item())
                            pyaw = float(player_state[2].item())
                            cos_y = math.cos(pyaw)
                            sin_y = math.sin(pyaw)
                            rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
                            world_pts = pts_local @ rot.T + torch.tensor([px, py], dtype=torch.float32, device=self.device)
                            screen_pts = self.convert_world_to_screen(world_pts)
                            # 散点显示所有路径点
                            for sp in screen_pts:
                                pygame.draw.circle(self.screen, (0, 180, 0), sp, 3)
            except Exception:
                pass

            # 绘制观测到的邻居（基于 neighbors_local 的局部相对量重建世界坐标）
            try:
                if hasattr(self, 'last_neighbors_local') and self.last_neighbors_local is not None:
                    neigh_local = self.last_neighbors_local[b_p, m_p]  # (K, 7)
                    player_state = self.simulator.agents_state[b_p, m_p]
                    px = float(player_state[0].item())
                    py = float(player_state[1].item())
                    pyaw = float(player_state[2].item())
                    cos_y = math.cos(pyaw)
                    sin_y = math.sin(pyaw)
                    rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
                    for k in range(neigh_local.shape[0]):
                        active_k = float(neigh_local[k, 6].item()) > 0.5
                        if not active_k:
                            continue
                        # 局部相对位置与相对速度
                        dx = float(neigh_local[k, 0].item())
                        dy = float(neigh_local[k, 1].item())
                        dvx_local = float(neigh_local[k, 2].item())
                        dvy_local = float(neigh_local[k, 3].item())
                        length = float(neigh_local[k, 4].item())
                        width = float(neigh_local[k, 5].item())
                        # 局部中心 -> 世界中心
                        local_pt = torch.tensor([[dx, dy]], dtype=torch.float32, device=self.device)
                        world_pt = (local_pt @ rot.T).squeeze(0) + torch.tensor([px, py], dtype=torch.float32, device=self.device)
                        # 自车绝对速度（世界）
                        ego_speed = float(player_state[3].item())
                        vx_ego = ego_speed * cos_y
                        vy_ego = ego_speed * sin_y
                        # 相对速度(局部) -> 世界相对速度
                        rv = torch.tensor([[dvx_local, dvy_local]], dtype=torch.float32, device=self.device)
                        rv_world = (rv @ rot.T).squeeze(0)
                        rvx_world = float(rv_world[0].item())
                        rvy_world = float(rv_world[1].item())
                        # 近似邻居绝对速度 = 自车绝对速度 + 相对世界速度
                        nvx_world = vx_ego + rvx_world
                        nvy_world = vy_ego + rvy_world
                        speed_mag = math.hypot(nvx_world, nvy_world)
                        if speed_mag > 1e-2:
                            yaw_world = math.atan2(nvy_world, nvx_world)
                        else:
                            yaw_world = pyaw
                        half_l = max(0.1, length * 0.5)
                        half_w = max(0.1, width * 0.5)
                        local_box = torch.tensor([
                            [-half_l, -half_w],
                            [ half_l, -half_w],
                            [ half_l,  half_w],
                            [-half_l,  half_w],
                        ], dtype=torch.float32, device=self.device)
                        cos_n = math.cos(yaw_world)
                        sin_n = math.sin(yaw_world)
                        rot_n = torch.tensor([[cos_n, -sin_n],[sin_n, cos_n]], dtype=torch.float32, device=self.device)
                        world_box = local_box @ rot_n.T + world_pt
                        pts = self.convert_world_to_screen(world_box)
                        if len(pts) >= 4:
                            pygame.draw.polygon(self.screen, self.colors['other_car'], pts, width=0)
            except Exception:
                pass
            
            # 绘制边界观测点（oob观测）：将 w_boundaries_local 从局部坐标变换到世界坐标并绘制散点
            try:
                if hasattr(self, 'last_w_boundaries_local') and self.last_w_boundaries_local is not None:
                    bounds_local = self.last_w_boundaries_local[b_p, m_p]  # (Nb, 2)
                    player_state = self.simulator.agents_state[b_p, m_p]
                    px = float(player_state[0].item())
                    py = float(player_state[1].item())
                    pyaw = float(player_state[2].item())
                    cos_y = math.cos(pyaw)
                    sin_y = math.sin(pyaw)
                    rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
                    # 有效边界点：排除 (0,0) 占位
                    if bounds_local.ndim >= 2 and bounds_local.shape[-1] >= 2:
                        valid = (bounds_local[..., 0] != 0) | (bounds_local[..., 1] != 0)
                        pts_local = bounds_local[valid][..., :2]
                        if pts_local.numel() > 0:
                            # (N,2) -> 旋转平移到世界坐标
                            world_pts = pts_local @ rot.T + torch.tensor([px, py], dtype=torch.float32, device=self.device)
                            screen_pts = self.convert_world_to_screen(world_pts)
                            for sp in screen_pts:
                                pygame.draw.circle(self.screen, (0, 0, 0), sp, 2)
            except Exception:
                pass

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

    def draw_dashed_circle(self, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], width: int = 1, dash_deg: float = 8.0, gap_deg: float = 6.0):
        """绘制虚线圆（通过短线段近似）。"""
        try:
            cx, cy = center
            angle = 0.0
            total_deg = 360.0
            while angle < total_deg:
                start_deg = angle
                end_deg = min(angle + dash_deg, total_deg)
                # 将弧段用两端点连线近似
                start_rad = math.radians(start_deg)
                end_rad = math.radians(end_deg)
                x1 = int(cx + radius * math.cos(start_rad))
                y1 = int(cy + radius * math.sin(start_rad))
                x2 = int(cx + radius * math.cos(end_rad))
                y2 = int(cy + radius * math.sin(end_rad))
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)
                angle += (dash_deg + gap_deg)
        except Exception:
            pass

    def draw_vision_circle(self):
        """在屏幕中心绘制与观测半径相同的虚线圆，用于可视化视野大小。"""
        # 从缓存参数读取缩放与中心（convert_world_to_screen 中已缓存）
        if not hasattr(self, '_cached_scale_params') or self._cached_scale_params is None:
            # 触发一次坐标转换以填充缓存
            visible_boundary_points = self.get_visible_boundary_points()
            _ = self.convert_world_to_screen(visible_boundary_points[:1])
        if not hasattr(self, '_cached_scale_params') or self._cached_scale_params is None:
            return
        scale_x, scale_y, car_x, car_y = self._cached_scale_params
        # 与 convert_world_to_screen 中一致的视野半径
        vision_radius_m = 100.0
        # 使用较小的缩放保持宽高比，convert_world_to_screen 已做 min(scale_x, scale_y)
        scale = min(scale_x, scale_y)
        radius_px = int(vision_radius_m * scale)
        center = (self.width // 2, self.height // 2)
        self.draw_dashed_circle(center, radius_px, (0, 0, 0), width=1, dash_deg=8.0, gap_deg=6.0)
  
    def draw_ui(self):
        """绘制用户界面"""
        # 统一显示UI：step信息 + 当前观察玩家的(B,M)
        step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
        iteration_text = self.font.render(f"Iteration: {self.iteration_count}", True, self.colors['text'])
        buffer_text = self.font.render(f"Buffer steps: {self.buffer_step_count}/{self.rollout_length}", True, self.colors['text'])
        b_obs, m_obs = self._get_player_agent_bm()
        bm_text = self.font.render(f"View Player: B={max(b_obs,0)}, M={max(m_obs,0)}", True, (255, 100, 0))
        mode_text = self.font.render(f"Mode: {'Auto(F1)' if self.auto_mode else 'Manual(F1)'}", True, (0, 128, 255))
        step_reward_text = self.small_font.render(f"step reward: {getattr(self, 'current_step_reward', 0.0):.4f}", True, self.colors['text'])
        # 调整左上角位置：去掉 step reward 后上移
        self.screen.blit(step_text, (10, 10))
        self.screen.blit(iteration_text, (10, 50))
        self.screen.blit(buffer_text, (10, 90))
        self.screen.blit(bm_text, (10, 130))
        self.screen.blit(step_reward_text, (10, 270))
        self.screen.blit(mode_text, (10, 290))

        # 显示玩家车辆状态（来自当前视角world）
        px, py, pyaw = self._get_player_pose()

        # 游戏进行中，显示动态信息
        b_cur, m_cur = self._get_player_agent_bm()
        if b_cur >= 0 and m_cur >= 0:
            states_cur = self.simulator.agents_state[b_cur]
            speed_val = float(states_cur[m_cur, 3].item())
            a_idx = 0
            if hasattr(self, 'last_actions'):
                actions_cpu = self.last_actions.detach().to('cpu')
                a_idx = int(actions_cpu[b_cur, m_cur].item())
        else:
            speed_val = 0.0
            a_idx = 0

        # 当前观测车辆 active/done 状态
        if b_cur >= 0 and m_cur >= 0:
            active_bool = bool((self.simulator.agents_state[b_cur, m_cur, 6] > 0.5).item())
            if hasattr(self, 'cumulative_done_all') and self.cumulative_done_all is not None and b_cur < self.cumulative_done_all.shape[0] and m_cur < self.cumulative_done_all.shape[1]:
                done_bool = bool(self.cumulative_done_all[b_cur, m_cur].item())
            else:
                done_bool = False
        else:
            active_bool = False
            done_bool = False

        # 车世界坐标（放到 speed 上方，字体与 speed 相同）
        car_world_pos = f"car world pos: ({px:.1f}, {py:.1f})"
        car_pos_text = self.small_font.render(car_world_pos, True, self.colors['text'])
        speed_text = self.small_font.render(f"speed: {speed_val:.2f} m/s", True, self.colors['text'])
        heading_text = self.small_font.render(f"heading: {math.degrees(pyaw):.1f}°", True, self.colors['text'])
        active_text = self.small_font.render(f"active: {active_bool}", True, self.colors['text'])
        done_text = self.small_font.render(f"done: {done_bool}", True, self.colors['text'])
        # 先绘制车世界坐标，再绘制速度等
        self.screen.blit(car_pos_text, (10, 180))
        self.screen.blit(speed_text, (10, 200))
        self.screen.blit(heading_text, (10, 220))
        self.screen.blit(active_text, (10, 240))
        self.screen.blit(done_text, (10, 260))
        # 显示目标世界坐标
        if hasattr(self, 'goal') and self.goal:
            goal_world_pos = f"goal world pos: ({self.goal['x']:.1f}, {self.goal['y']:.1f})"
            goal_pos_text = self.small_font.render(goal_world_pos, True, self.colors['text'])
            self.screen.blit(goal_pos_text, (10, 320))
        else:
            # 手动控制提示
            help1 = self.small_font.render("Manual: q w e r t y u i o p [ ] -> actions", True, self.colors['text'])
            help2 = self.small_font.render("F1 toggle Auto; ESC quit", True, self.colors['text'])
            self.screen.blit(help1, (10, 320))
            self.screen.blit(help2, (10, 340))
    
        # 显示当前视角world的车辆信息
        view_b = getattr(self, '_current_view_world', 0)
        world_states = self.simulator.agents_state[view_b]
        active_mask = (world_states[:, 6] > 0.5).detach().to('cpu')
        if hasattr(self, 'cumulative_done_all') and self.cumulative_done_all is not None:
            done_mask = self.cumulative_done_all[view_b]
        else:
            done_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        alive_mask = active_mask & (~done_mask)
        alive_num = int(alive_mask.sum().item())
        dead_num = int((active_mask & done_mask).sum().item())
        total_active = int(active_mask.sum().item())
        car_info = f"World {view_b} - alive: {alive_num}, dead: {dead_num}, total: {total_active}"
        car_text = self.small_font.render(car_info, True, self.colors['text'])
        # 显示在 car world pos 之上，字体与其一致
        self.screen.blit(car_text, (10, 160))
        
    def run(self):
        """运行游戏主循环（可视化）。"""
        while self.running:
            # 处理事件，避免窗口无响应 别删
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_F1:
                        # 切换自动/人工模式
                        self.auto_mode = not self.auto_mode
                        mode = 'Auto' if self.auto_mode else 'Manual'
                        print(f"切换模式: {mode}")
                    else:
                        # 人工模式下记录一次性动作
                        if not self.auto_mode and event.key in self.key_to_action:
                            self.pending_manual_action_idx = self.key_to_action[event.key]
            # 推进模拟器并绘制
            self.update_game_state()
            # 绘制画面
            self.draw()
            # # 帧率限制
            # if self.clock is not None:
            #     self.clock.tick(60)
    
if __name__ == "__main__":
    game = CarGame()
    game.run()

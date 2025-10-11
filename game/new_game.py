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
import matplotlib
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

class Game:
    def __init__(self):
        # 加载默认配置
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
        self.states_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.values_buffer = []
        self.old_log_probs_buffer = []
        self.actions_buffer = []
        self.buffer_step_count = 0

    def _init_plots(self):
        """初始化Matplotlib交互式窗口与轴"""
        try:
            # 使用交互模式，避免阻塞pygame主循环
            plt.ion()
            # 单独起一个窗口，两个子图：上-损失；下-平均回报
            self.fig, (self.ax_loss, self.ax_reward) = plt.subplots(2, 1, figsize=(7, 6))
            self.fig.canvas.manager.set_window_title("PPO Metrics")
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
            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            # 可视化失败不影响训练
            self.fig = None
            self.ax_loss = None
            self.ax_loss_right = None
            self.ax_reward = None

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
            # # 帧率限制
            # if self.clock is not None:
            #     self.clock.tick(60)
if __name__ == "__main__":
    game = Game()
    game.run()  
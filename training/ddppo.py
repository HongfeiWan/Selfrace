from networkx import to_dict_of_dicts
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import Adam

from collections import deque

from network import SharedNetwork, create_network

import numpy as np
import yaml
import os
import time

#Todo: 优势归一化

class ExperienceBuffer:
    """经验回放缓冲区 - 支持8张显卡的大规模训练"""
    def __init__(self, capacity=int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, experience):
        """添加经验到缓冲区"""
        self.buffer.append(experience)
    def sample(self, batch_size):
        """采样经验批次"""
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class AdvantageFilter:
    """优势过滤器 - 实现 Algorithm 1"""
    def __init__(self, beta=0.25, eta_multiplier=0.01):
        self.beta = beta  # EWMA衰减参数
        self.eta_multiplier = eta_multiplier  # 过滤阈值乘数
        self.amax_ewma = None  # 指数加权移动平均的最大优势值
        
    def compute_gae(self, rewards, values, dones, gamma=0.999, gae_lambda=0.95):
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update_amax_ewma(self, advantages, iteration):
        """更新最大优势值的指数加权移动平均"""
        current_amax = torch.abs(advantages).max().item()
        
        if iteration == 0:
            self.amax_ewma = current_amax
        else:
            self.amax_ewma = self.beta * self.amax_ewma + (1 - self.beta) * current_amax
    
    def filter_experiences(self, experiences, advantages, iteration):
        """根据 Algorithm 1 过滤经验"""
        # 更新 EWMA
        self.update_amax_ewma(advantages, iteration)
        
        # 计算过滤阈值
        eta = self.eta_multiplier * self.amax_ewma
        
        # 过滤经验
        filtered_indices = torch.abs(advantages) < eta
        filtered_experiences = [exp for i, exp in enumerate(experiences) if filtered_indices[i]]
        
        return filtered_experiences, eta

class DistributedPPOTrainer:
    """分布式PPO训练器 - 支持8张显卡并行训练"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.rank = int(config.device.split(':')[1]) if ':' in config.device else 0

        # 从配置文件加载分布式设置
        self.world_size = config.distributed.get('world_size', 8)  # 8张显卡
        self.envs_per_gpu = config.distributed.get('envs_per_gpu', 4800)  # 每个GPU 4800个环境
        self.backend = config.distributed.get('backend', 'nccl')
        self.init_method = config.distributed.get('init_method', 'tcp://localhost:12355')
        
        # 特征维度配置 - 根据GIGAFLOW论文
        # 注意：vehicle_state维度将通过ObservationGenerator动态计算

        self.feature_dims = {
            'road_boundary': 20,    # 道路边界特征
            'lane_points': 30,       # 车道点特征
            'stop_lines': 10,        # 停止线特征
            'vehicle_state': None,   # 车辆状态特征 (将动态计算)
            'other_agents': 25,      # 其他智能体特征
            'conditioning': 12       # 条件和目标特征
        }
        
        # 网络配置
        self.num_actions = 12  # 动作数量
        self.network_dim = 128  # 网络隐藏层维度
        
        # 经验缓冲区 - 针对8张显卡优化
        buffer_capacity = config.training.get('experience_buffer_capacity', 50000)
        self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)
        
        # 训练统计
        self.training_stats = {
            'episodes_completed': 0,
            'total_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'start_time': time.time()
        }
        
        # 添加优势过滤器
        self.advantage_filter = AdvantageFilter(
            beta=0.25,  # EWMA衰减参数
            eta_multiplier=0.01  # 过滤阈值乘数
        )
        
        # 训练迭代计数器
        self.training_iteration = 0
        
    def setup_distributed(self):
        """初始化分布式环境 - 8张显卡配置"""
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 初始化进程组
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank
        )

    def create_models(self):
        """创建使用SharedNetwork的分布式模型 - 8张显卡优化"""
        # 创建共享网络
        self.shared_network = create_network(
            network_type="shared",
            feature_dims=self.feature_dims,
            num_actions=self.num_actions,
            network_dim=self.network_dim
        ).to(self.device)
        
        # 包装为分布式模型 - 针对8张显卡优化
        self.shared_network = DDP(
            self.shared_network, 
            device_ids=[self.rank],
            find_unused_parameters=True,
            broadcast_buffers=False,  # 减少通信开销
            bucket_cap_mb=25  # 优化梯度桶大小
        )
        
        # 创建优化器 - 针对大规模训练优化
        self.optimizer = Adam(
            self.shared_network.parameters(),
            lr=self.config.training.get('learning_rate', 3e-4),
            eps=1e-5,
            weight_decay=1e-4  # 添加权重衰减防止过拟合
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100, 
            gamma=0.9
        )
        
    def create_env(self, env_id):
        """创建环境实例 - 每个GPU管理4800个环境"""
        # 这里应该创建您的Carla环境
        # 示例：return CarlaEnv(env_id, self.config)
        # 每个环境对应一个车辆实例
        pass
        
    def collect_single_env_experience(self, env, num_steps=1000):
        """从单个环境收集经验 - 针对8张显卡优化"""
        experiences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'action_logits': [],
            'dones': [],
            'advantages': [],
            'returns': []
        }
        state = env.reset()
        
        for step in range(num_steps):
            # 准备特征输入
            features = self.prepare_features(state)
            
            # 获取动作和值
            with torch.no_grad():
                action_logits, value = self.shared_network(features)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action.item())
            
            # 存储经验
            experiences['states'].append(state)
            experiences['actions'].append(action.item())
            experiences['rewards'].append(reward)
            experiences['values'].append(value.item())
            experiences['action_logits'].append(action_logits)
            experiences['dones'].append(done)
            
            state = next_state
            if done:
                state = env.reset()
                
        return experiences
    
    def prepare_features(self, state):
        """准备网络输入特征 - 8张显卡批处理优化"""
        # 这里需要根据您的状态格式来准备特征
        # 示例实现 - 实际使用时需要根据真实状态数据调整
        batch_size = 1  # 可以根据需要调整批次大小
        
        features = {
            'road_boundary': torch.randn(batch_size, self.feature_dims['road_boundary']).to(self.device),
            'lane_points': torch.randn(batch_size, self.feature_dims['lane_points']).to(self.device),
            'stop_lines': torch.randn(batch_size, self.feature_dims['stop_lines']).to(self.device),
            'vehicle_state': torch.randn(batch_size, self.feature_dims['vehicle_state']).to(self.device),
            'other_agents': torch.randn(batch_size, self.feature_dims['other_agents']).to(self.device),
            'conditioning': torch.randn(batch_size, self.feature_dims['conditioning']).to(self.device)
        }
        return features
        
    def collect_experience(self):
        """并行收集经验 - 8张显卡并行处理"""
        experiences = []
        for env_id in range(self.envs_per_gpu):
            env = self.create_env(env_id)
            exp = self.collect_single_env_experience(env)
            experiences.append(exp)
        return experiences
    
    def compute_policy_loss(self, experiences):
        """计算PPO策略损失 - 8张显卡优化版本"""
        # 这里需要实现完整的PPO策略损失计算
        # 包括重要性采样比率、裁剪等
        # 简化实现 - 实际使用时需要完整的PPO算法
        policy_loss = torch.tensor(0.0, device=self.device)
        # 计算重要性采样比率
        # ratio = new_policy_prob / old_policy_prob
        
        # 计算裁剪后的目标
        # clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        
        # 计算策略损失
        # policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        return policy_loss
    
    def compute_value_loss(self, experiences):
        """计算价值损失 - 8张显卡优化版本"""
        # 这里需要实现完整的价值损失计算
        # 包括TD误差等
        
        # 简化实现
        value_loss = torch.tensor(0.0, device=self.device)
        
        # 计算价值损失
        # value_loss = F.mse_loss(predicted_values, target_values)
        
        return value_loss
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """计算广义优势估计 (GAE) - 8张显卡优化"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, device=self.device)
        
    def update_policy(self, experiences):
        """更新策略 - 8张显卡同步更新"""
        # 计算损失
        policy_loss = self.compute_policy_loss(experiences)
        value_loss = self.compute_value_loss(experiences)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(
            self.shared_network.parameters(), 
            max_norm=0.5
        )
        
        # 优化器步骤
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
    def save_checkpoint(self, episode):
        """保存检查点 - 只在主进程保存"""
        if self.rank == 0:  # 只在主进程保存
            checkpoint = {
                'episode': episode,
                'model_state_dict': self.shared_network.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'training_stats': self.training_stats
            }
            
            # 创建检查点目录
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/checkpoint_episode_{episode}.pt'
            torch.save(checkpoint, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点 - 支持8张显卡同步加载"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.shared_network.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练统计
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
            
        return checkpoint['episode']
        
    def train(self):
        """主训练循环 - 8张显卡分布式训练"""
        num_episodes = self.config.training.get('num_episodes', 1000)
        checkpoint_interval = self.config.training.get('checkpoint_interval', 100)
        
        for episode in range(num_episodes):
            # 收集经验
            experiences = self.collect_experience()
            
            # 将经验添加到缓冲区
            for exp in experiences:
                self.experience_buffer.push(exp)
            
            # 如果缓冲区有足够的数据，进行训练
            min_buffer_size = self.config.training.get('min_buffer_size_for_training', 1000)
            if len(self.experience_buffer) >= min_buffer_size:
                # 采样经验批次
                batch_size = self.config.training.get('batch_size', 256)
                batch = self.experience_buffer.sample(batch_size)
                if batch is not None:
                    # 更新策略
                    losses = self.update_policy(batch)
                    
                    # 更新训练统计
                    self.training_stats['episodes_completed'] += 1
                    self.training_stats['policy_loss'] = losses['policy_loss']
                    self.training_stats['value_loss'] = losses['value_loss']
            
            # 保存检查点
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode)

class TrainingConfig:
    """训练配置类 - 8张显卡优化配置"""
    def __init__(self, config_path="configs/default_config.yaml"):
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 分布式配置
        self.distributed = config_data['training']['distributed']
        # 设置设备配置
        self.device = config_data['training']['distributed']['device']
    
        # 训练配置
        self.training = config_data['training']
        
        # 其他配置
        self.simulator = config_data['simulator']
        self.reward = config_data['reward']
        self.policy = config_data.get('policy', {})
        
        # 设置默认值
        self.world_size = self.distributed.get('world_size', 8)
        self.envs_per_gpu = self.distributed.get('envs_per_gpu', 4800)
        self.num_episodes = self.training.get('num_episodes', 1000)
        self.learning_rate = self.training.get('learning_rate', 3e-4)
        self.batch_size = self.training.get('batch_size', 256)
        self.gamma = self.training.get('gamma', 0.99)
        self.gae_lambda = self.training.get('gae_lambda', 0.95)
        self.clip_epsilon = self.training.get('clip_range', 0.2)
        self.value_loss_coef = self.training.get('vf_coef', 0.5)
        self.entropy_coef = self.training.get('ent_coef', 0.01)

def main_worker(rank, world_size, config):
    """工作进程主函数 - 8张显卡分布式训练"""
    # 设置设备
    config.device = f"cuda:{rank}"
    config.world_size = world_size
    
    # 创建训练器
    trainer = DistributedPPOTrainer(config)
    
    try:
        # 设置分布式环境
        trainer.setup_distributed()
        
        # 创建模型
        trainer.create_models()
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """主函数 - 8张显卡分布式训练启动"""
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        exit(1)
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        exit(1)
    
    # 加载配置
    config = TrainingConfig()
    
    # 启动多进程
    mp.spawn(
        main_worker,
        args=(config.world_size, config),
        nprocs=config.world_size,
        join=True
    )

if __name__ == "__main__":
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        exit(1)
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        exit(1)
    # 启动分布式训练
    main()
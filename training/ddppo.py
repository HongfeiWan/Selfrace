import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import deque
from network import SharedNetwork, create_network

class ExperienceBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
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

class DistributedPPOTrainer:
    def __init__(self, config):
        self.config = config
        self.world_size = config.world_size
        self.rank = config.rank
        self.device = torch.device(f'cuda:{self.rank}')

        # 特征维度配置
        self.feature_dims = {
            'road_boundary': 20,    # 道路边界特征
            'lane_points': 30,       # 车道点特征
            'stop_lines': 10,        # 停止线特征
            'vehicle_state': 15,     # 车辆状态特征
            'other_agents': 25,      # 其他智能体特征
            'conditioning': 12       # 条件和目标特征
        }
        
        # 网络配置
        self.num_actions = 4  # 动作数量
        self.network_dim = 128  # 网络隐藏层维度
        
        # 经验缓冲区
        self.experience_buffer = ExperienceBuffer(capacity=50000)
        
    def setup_distributed(self):
        """初始化分布式环境"""
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:12355',
            world_size=self.world_size,
            rank=self.rank
        )

    def create_models(self):
        """创建使用SharedNetwork的分布式模型"""
        # 创建共享网络
        self.shared_network = create_network(
            network_type="shared",
            feature_dims=self.feature_dims,
            num_actions=self.num_actions,
            network_dim=self.network_dim
        ).to(self.device)
        
        # 包装为分布式模型
        self.shared_network = DDP(
            self.shared_network, 
            device_ids=[self.rank],
            find_unused_parameters=True
        )
        
        # 创建优化器
        self.optimizer = Adam(
            self.shared_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
    def create_env(self, env_id):
        """创建环境实例"""
        # 这里应该创建您的Carla环境
        # 示例：return CarlaEnv(env_id)
        pass
        
    def collect_single_env_experience(self, env, num_steps=1000):
        """从单个环境收集经验"""
        experiences = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'action_logits': [],
            'dones': []
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
        """准备网络输入特征"""
        # 这里需要根据您的状态格式来准备特征
        # 示例实现
        features = {
            'road_boundary': torch.randn(1, self.feature_dims['road_boundary']).to(self.device),
            'lane_points': torch.randn(1, self.feature_dims['lane_points']).to(self.device),
            'stop_lines': torch.randn(1, self.feature_dims['stop_lines']).to(self.device),
            'vehicle_state': torch.randn(1, self.feature_dims['vehicle_state']).to(self.device),
            'other_agents': torch.randn(1, self.feature_dims['other_agents']).to(self.device),
            'conditioning': torch.randn(1, self.feature_dims['conditioning']).to(self.device)
        }
        return features
        
    def collect_experience(self):
        """并行收集经验"""
        experiences = []
        for env_id in range(self.config.envs_per_gpu):
            env = self.create_env(env_id)
            exp = self.collect_single_env_experience(env)
            experiences.append(exp)
        return experiences
    
    def compute_policy_loss(self, experiences):
        """计算PPO策略损失"""
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
        """计算价值损失"""
        # 这里需要实现完整的价值损失计算
        # 包括TD误差等
        
        # 简化实现
        value_loss = torch.tensor(0.0, device=self.device)
        
        # 计算价值损失
        # value_loss = F.mse_loss(predicted_values, target_values)
        
        return value_loss
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
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
        
        return torch.tensor(advantages, device=self.device)
        
    def update_policy(self, experiences):
        """更新策略"""
        # 计算损失
        policy_loss = self.compute_policy_loss(experiences)
        value_loss = self.compute_value_loss(experiences)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.shared_network.parameters(), 
            max_norm=0.5
        )
        
        # 优化器步骤
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def save_checkpoint(self, episode):
        """保存检查点"""
        if self.rank == 0:  # 只在主进程保存
            checkpoint = {
                'episode': episode,
                'model_state_dict': self.shared_network.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, f'checkpoint_episode_{episode}.pt')
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.shared_network.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode']
        
    def train(self):
        """主训练循环"""
        print(f"🚀 开始分布式PPO训练 (Rank {self.rank})")
        
        for episode in range(self.config.num_episodes):
            # 收集经验
            experiences = self.collect_experience()
            
            # 将经验添加到缓冲区
            for exp in experiences:
                self.experience_buffer.push(exp)
            
            # 如果缓冲区有足够的数据，进行训练
            if len(self.experience_buffer) >= self.config.batch_size:
                # 采样经验批次
                batch = self.experience_buffer.sample(self.config.batch_size)
                if batch is not None:
                    # 更新策略
                    losses = self.update_policy(batch)
                    
                    # 记录日志
                    if self.rank == 0 and episode % 10 == 0:
                        print(f"Episode {episode}: Policy Loss: {losses['policy_loss']:.4f}, "
                              f"Value Loss: {losses['value_loss']:.4f}, "
                              f"Total Loss: {losses['total_loss']:.4f}")
            
            # 保存检查点
            if episode % 100 == 0:
                self.save_checkpoint(episode)
                
        print(f"✅ 训练完成 (Rank {self.rank})")

class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        self.world_size = torch.cuda.device_count()
        self.rank = 0  # 将在spawn中设置
        self.envs_per_gpu = 4
        self.num_episodes = 1000
        self.learning_rate = 3e-4
        self.batch_size = 256
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

def main_worker(rank, world_size, config):
    """工作进程主函数"""
    config.rank = rank
    config.world_size = world_size
    
    trainer = DistributedPPOTrainer(config)
    
    # 设置分布式环境
    trainer.setup_distributed()
    
    # 创建模型
    trainer.create_models()
    
    # 开始训练
    trainer.train()

def main():
    """主函数"""
    config = TrainingConfig()
    world_size = torch.cuda.device_count()
    print(f"🎯 启动分布式PPO训练，使用 {world_size} 个GPU")
    # 启动多进程
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查GPU设置")
        exit(1)
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("❌ 未检测到GPU")
        exit(1)
    
    print(f"🎯 检测到 {gpu_count} 个GPU")
    
    # 启动分布式训练
    # main()
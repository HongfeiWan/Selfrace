# 神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SimpleFeatureEncoder(nn.Module):
    """
    简单特征编码器 - 用于简单特征向量 (S(t), G(t), reward系数,车辆风格系数等)
    完全向量化，支持批量处理
    """
    def __init__(self, input_dim, output_dim=64):
        super(SimpleFeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # 应用Orthogonal初始化
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    def forward(self, x):
        """
        完全向量化的前向传播
        Args:
            x: [B, M, input_dim] - B是batch_size，M是环境数量
        Returns:
            encoded: [B, M, output_dim]
        """
        # 输入一定是 [B, M, input_dim] 格式
        B, M, vector_dim = x.shape              # 输入是 [B, M, vector_dim]
        x_reshaped = x.view(-1, vector_dim)     # 重塑为 [B*M, vector_dim]
        encoded = self.mlp(x_reshaped)          # 编码
        return encoded.view(B, M, -1)           # 重塑回 [B, M, output_dim]

class PermutationInvariantEncoder(nn.Module):
    """
    排列不变编码器 - 完全向量化，支持批量处理
    用于多特征集合 (W(t)_lane, W(t)_boundary, W(t)_stop, A(t))
    """
    def __init__(self, feature_dim, output_dim=64):
        super(PermutationInvariantEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        # 编码每个元素的MLP
        self.element_encoder = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # 应用Orthogonal初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        完全向量化的前向传播，实现排列不变性
        Args:
            x: [B, M, feature_dim] - B是batch_size，M是环境数量，feature_dim是每个元素的特征维度
        Returns:
            encoded: [B, M, output_dim]
        """
        # 输入是 [B, M, feature_dim] 格式，需要重塑为 [B, M, 1, feature_dim] 来实现排列不变性
        B, M, feature_dim = x.shape
        # 重塑为 [B, M, 1, feature_dim] 以保持与原始设计的兼容性
        x_reshaped = x.unsqueeze(2)  # 添加 num_elements 维度
        # 编码每个元素
        encoded_elements = self.element_encoder(x_reshaped.view(-1, feature_dim))
        # 重塑回 [B, M, 1, output_dim]
        encoded_elements = encoded_elements.view(B, M, 1, self.output_dim)
        # 通过maxpooling实现排列不变性，聚合多个元素的信息
        encoded = torch.max(encoded_elements, dim=2)[0]  # [B, M, output_dim]
        return encoded

class FeatureEncoder(nn.Module):
    """
    完全向量化的特征编码器 - 通过配置文件指导参数
    输入为单个大张量，按固定位置切片提取特征
    """
    def __init__(self, config):
        super(FeatureEncoder, self).__init__()
        # 从配置文件读取所有参数
        network_config = config.training.network
        self.encoder_dim = network_config.encoder_dim
        self.simple_feature_dims = network_config.simple_feature_dims
        self.permutation_feature_dims = network_config.permutation_feature_dims
        # 计算总输入维度
        self.total_input_dim = sum(self.simple_feature_dims) + sum(self.permutation_feature_dims)
        # 创建简单特征编码器 - 直接创建4个
        self.simple_encoders = nn.ModuleList([
            SimpleFeatureEncoder(self.simple_feature_dims[0], self.encoder_dim),  # S(t): 10维
            SimpleFeatureEncoder(self.simple_feature_dims[1], self.encoder_dim),  # G(t): 1024维
            SimpleFeatureEncoder(self.simple_feature_dims[2], self.encoder_dim),  # reward系数: 10维
            SimpleFeatureEncoder(self.simple_feature_dims[3], self.encoder_dim)   # 车辆风格参数: 4维
        ])
        # 创建排列不变特征编码器 - 直接创建4个
        self.permutation_encoders = nn.ModuleList([
            PermutationInvariantEncoder(self.permutation_feature_dims[0], self.encoder_dim),  # road_boundary: 52维
            PermutationInvariantEncoder(self.permutation_feature_dims[1], self.encoder_dim),  # lane_points: 50维
            PermutationInvariantEncoder(self.permutation_feature_dims[2], self.encoder_dim),  # stop_lines: 20维
            PermutationInvariantEncoder(self.permutation_feature_dims[3], self.encoder_dim)   # other_agents: 140维
        ])
        # 计算总输出维度 - 固定8个编码器
        self.total_output_dim = (len(self.simple_encoders) + len(self.permutation_encoders)) * self.encoder_dim

    def forward(self, features_tensor):
        """
        完全向量化的特征编码 - 直接使用两个编码器列表
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            output: [B, M, total_output_dim] 编码后的特征张量
        """
        B, M, _ = features_tensor.shape
        # 批量NaN处理
        if torch.isnan(features_tensor).any():
            features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 预分配输出张量 [B, M, total_output_dim]
        output = torch.zeros(B, M, self.total_output_dim, device=features_tensor.device)
        
        # 编码简单特征 - 直接使用固定索引
        # S(t): 10维
        s_t = features_tensor[:, :, 0:self.simple_feature_dims[0]]
        output[:, :, 0:self.encoder_dim] = self.simple_encoders[0](s_t)
        
        # G(t): 1024维
        g_t = features_tensor[:, :, self.simple_feature_dims[0]:self.simple_feature_dims[0] + self.simple_feature_dims[1]]
        output[:, :, self.encoder_dim:2*self.encoder_dim] = self.simple_encoders[1](g_t)
        
        # reward系数: 10维
        reward_coef = features_tensor[:, :, self.simple_feature_dims[0] + self.simple_feature_dims[1]:self.simple_feature_dims[0] + self.simple_feature_dims[1] + self.simple_feature_dims[2]]
        output[:, :, 2*self.encoder_dim:3*self.encoder_dim] = self.simple_encoders[2](reward_coef)
        
        # 车辆风格参数: 4维
        vehicle_style = features_tensor[:, :, self.simple_feature_dims[0] + self.simple_feature_dims[1] + self.simple_feature_dims[2]:sum(self.simple_feature_dims)]
        output[:, :, 3*self.encoder_dim:4*self.encoder_dim] = self.simple_encoders[3](vehicle_style)
        
        # 编码排列不变特征 - 直接使用固定索引
        simple_end = sum(self.simple_feature_dims)
        
        # road_boundary: 52维
        road_boundary = features_tensor[:, :, simple_end:simple_end + self.permutation_feature_dims[0]]
        output[:, :, 4*self.encoder_dim:5*self.encoder_dim] = self.permutation_encoders[0](road_boundary)
        
        # lane_points: 50维
        lane_points = features_tensor[:, :, simple_end + self.permutation_feature_dims[0]:simple_end + self.permutation_feature_dims[0] + self.permutation_feature_dims[1]]
        output[:, :, 5*self.encoder_dim:6*self.encoder_dim] = self.permutation_encoders[1](lane_points)
        
        # stop_lines: 20维
        stop_lines = features_tensor[:, :, simple_end + self.permutation_feature_dims[0] + self.permutation_feature_dims[1]:simple_end + self.permutation_feature_dims[0] + self.permutation_feature_dims[1] + self.permutation_feature_dims[2]]
        output[:, :, 6*self.encoder_dim:7*self.encoder_dim] = self.permutation_encoders[2](stop_lines)
        
        # other_agents: 140维
        other_agents = features_tensor[:, :, simple_end + self.permutation_feature_dims[0] + self.permutation_feature_dims[1] + self.permutation_feature_dims[2]:self.total_input_dim]
        output[:, :, 7*self.encoder_dim:8*self.encoder_dim] = self.permutation_encoders[3](other_agents)
        
        return output

class SharedNetwork(nn.Module):
    """
    共享网络（同时输出策略和值函数）
    完全向量化，支持批量处理和多GPU分布式训练
    符合论文描述的MLP架构：[1024 × 1024 × 1024]
    """
    def __init__(self, config):
        super(SharedNetwork, self).__init__()
        # 从配置文件读取所有参数
        network_config = config.training.network
        self.num_actions = network_config.num_actions
        self.network_dim = network_config.network_dim
        # 特征编码器 - 完全依赖配置文件
        self.feature_encoder = FeatureEncoder(config=config)
        # 从特征编码器获取总输出维度
        total_encoded_dim = self.feature_encoder.total_output_dim
        # 符合论文描述的MLP骨干网络：[1024 × 1024 × 1024]
        self.fc1 = nn.Linear(total_encoded_dim, self.network_dim)
        self.fc2 = nn.Linear(self.network_dim, self.network_dim)
        self.fc3 = nn.Linear(self.network_dim, self.network_dim)
        # 策略头（输出动作logits）
        self.action_head = nn.Linear(self.network_dim, self.num_actions)
        # 值函数头（输出状态值）
        self.value_head = nn.Linear(self.network_dim, 1)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, features_tensor):
        """
        完全向量化的前向传播，支持批量处理
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            action_logits: 动作logits [B, M, num_actions]
            value: 状态值 [B, M]
        """
        # 编码各种特征
        encoded_features = self.feature_encoder(features_tensor)
        # 向量化NaN处理
        if torch.isnan(encoded_features).any():
            encoded_features = torch.nan_to_num(encoded_features, nan=0.0, posinf=1.0, neginf=-1.0)
        # 符合论文描述的MLP骨干网络：[1024 × 1024 × 1024]
        x = F.relu(self.fc1(encoded_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 分别输出策略和值函数
        action_logits = self.action_head(x)
        value = self.value_head(x).squeeze(-1)
        return action_logits, value

def create_network(config, network_type="shared"):
    """
    创建网络实例的工厂函数
    Args:
        config: 配置文件对象（必需）
        network_type: 网络类型 ("shared")
    Returns:
        网络实例
    """
    if network_type == "shared":
        return SharedNetwork(config=config)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试网络 - 使用配置文件
    print("🧪 测试神经网络（使用配置文件）...")
    try:
        import yaml
        from types import SimpleNamespace
        import json
        # 设置设备为 cuda:0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {device}")
        # 读取配置文件
        with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        # 转换为对象
        config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
        model = create_network(config=config, network_type="shared")
        model = model.to(device)  # 将模型移动到GPU
        print(f"🔍 模型参数数量: {count_parameters(model)}")
        # 创建示例输入用于测试
        B = 4800
        M = 150
        # 创建固定长度的特征张量 [B, M, total_input_dim] 并移动到GPU
        features_tensor = torch.randn(B, M, model.feature_encoder.total_input_dim, device=device)
        # 测试前向传播
        action_logits, value = model(features_tensor)
        print(f"✅ 前向传播成功")
        print(f"Action logits shape: {action_logits.shape}")
        print(f"Value shape: {value.shape}")
        # 测试反向传播
        loss = action_logits.sum() + value.sum()
        loss.backward()
        print(f"✅ 反向传播成功")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


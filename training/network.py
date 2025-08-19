# 神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeatureEncoder(nn.Module):
    """
    简单特征编码器 - 用于简单特征向量 (S(t), G(t), C_reward等)
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
            x: [batch_size, input_dim] 或 [batch_size, num_envs, input_dim] 或 [batch_size, num_envs, num_elements, input_dim]
        Returns:
            encoded: [batch_size, output_dim] 或 [batch_size, num_envs, output_dim]
        """
        if x.dim() == 2:
            return self.mlp(x)
        elif x.dim() == 3:
            # [batch_size, num_envs, input_dim]
            batch_size, num_envs, input_dim = x.shape
            x_reshaped = x.view(-1, input_dim)
            encoded = self.mlp(x_reshaped)
            return encoded.view(batch_size, num_envs, -1)
        elif x.dim() == 4:
            # [batch_size, num_envs, num_elements, input_dim]
            batch_size, num_envs, num_elements, input_dim = x.shape
            x_reshaped = x.view(-1, input_dim)
            encoded = self.mlp(x_reshaped)
            # 取平均
            encoded = encoded.view(batch_size, num_envs, num_elements, -1)
            return encoded.mean(dim=2)  # [batch_size, num_envs, output_dim]
        else:
            raise ValueError(f"Expected 2D, 3D or 4D input, got {x.dim()}D")

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
        完全向量化的前向传播，确保与SimpleFeatureEncoder输出一致
        Args:
            x: [batch_size, num_elements, feature_dim] 或 [batch_size, num_envs, num_elements, feature_dim]
        Returns:
            encoded: [batch_size, output_dim] 或 [batch_size, num_envs, output_dim]
        """
        if x.dim() == 2:
            # 单个特征，直接编码
            return self.element_encoder(x)
        elif x.dim() == 3:
            # 多个特征，编码后maxpooling
            batch_size, num_elements, feature_dim = x.shape
            # 重塑为 [batch_size * num_elements, feature_dim]
            x_reshaped = x.view(-1, feature_dim)
            # 编码每个元素
            encoded_elements = self.element_encoder(x_reshaped)
            # 重塑回 [batch_size, num_elements, output_dim]
            encoded_elements = encoded_elements.view(batch_size, num_elements, self.output_dim)
            # 通道维度的maxpooling
            encoded = torch.max(encoded_elements, dim=1)[0]  # [batch_size, output_dim]
            return encoded
        elif x.dim() == 4:
            # 多环境多特征，完全向量化处理
            batch_size, num_envs, num_elements, feature_dim = x.shape
            # 重塑为 [batch_size * num_envs * num_elements, feature_dim]
            x_reshaped = x.view(-1, feature_dim)
            # 编码每个元素
            encoded_elements = self.element_encoder(x_reshaped)
            # 重塑回 [batch_size, num_envs, num_elements, output_dim]
            encoded_elements = encoded_elements.view(batch_size, num_envs, num_elements, self.output_dim)
            # 通道维度的maxpooling
            encoded = torch.max(encoded_elements, dim=2)[0]  # [batch_size, num_envs, output_dim]
            return encoded
        else:
            raise ValueError(f"Expected 2D, 3D or 4D input, got {x.dim()}D")

class FeatureEncoder(nn.Module):
    """
    特征编码器 - 完全向量化，支持批量处理
    符合论文描述的架构
    """
    def __init__(self, feature_dims, encoder_dim=64):
        super(FeatureEncoder, self).__init__()
        self.encoder_dim = encoder_dim
        
        # 简单特征编码器 (S(t), G(t), C_reward等)
        self.simple_encoders = nn.ModuleDict()
        simple_features = ['vehicle_state', 'conditioning']  # 简单特征向量
        for feature_name in simple_features:
            if feature_name in feature_dims and feature_dims[feature_name] > 0:
                self.simple_encoders[feature_name] = SimpleFeatureEncoder(
                    feature_dims[feature_name], encoder_dim
                )
        
        # 排列不变编码器 (W(t)_lane, W(t)_boundary, W(t)_stop, A(t))
        self.permutation_encoders = nn.ModuleDict()
        permutation_features = ['road_boundary', 'lane_points', 'stop_lines', 'other_agents']
        for feature_name in permutation_features:
            if feature_name in feature_dims and feature_dims[feature_name] > 0:
                self.permutation_encoders[feature_name] = PermutationInvariantEncoder(
                    feature_dims[feature_name], encoder_dim
                )

    def forward(self, features):
        """
        完全向量化的特征编码
        Args:
            features: 字典，包含各种特征张量
        Returns:
            concatenated_features: 拼接后的特征张量
        """
        encoded_features = []
        # 向量化处理简单特征
        for feature_name, encoder in self.simple_encoders.items():
            if feature_name in features and features[feature_name] is not None:
                feature = features[feature_name]
                # 向量化NaN处理
                if torch.isnan(feature).any():
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                encoded = encoder(feature)
                encoded_features.append(encoded)
        
        # 向量化处理排列不变特征
        for feature_name, encoder in self.permutation_encoders.items():
            if feature_name in features and features[feature_name] is not None:
                feature = features[feature_name]
                # 向量化NaN处理
                if torch.isnan(feature).any():
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                encoded = encoder(feature)
                encoded_features.append(encoded)
        
        # 向量化拼接所有编码后的特征
        if encoded_features:
            concatenated_features = torch.cat(encoded_features, dim=-1)
        else:
            # 如果没有特征，创建零张量
            batch_size = next(iter(features.values())).shape[0] if features else 1
            total_encoders = len(self.simple_encoders) + len(self.permutation_encoders)
            concatenated_features = torch.zeros(batch_size, total_encoders * self.encoder_dim, 
                                              device=next(iter(features.values())).device if features else 'cpu')
        return concatenated_features

class SharedNetwork(nn.Module):
    """
    共享网络（同时输出策略和值函数）
    完全向量化，支持批量处理和多GPU分布式训练
    符合论文描述的MLP架构：[1024 × 1024 × 1024]
    """
    def __init__(self, feature_dims, num_actions, network_dim=1024):
        super(SharedNetwork, self).__init__()
        # 特征编码器
        self.feature_encoder = FeatureEncoder(feature_dims, encoder_dim=64)
        # 计算编码后的特征总维度
        total_encoders = len([k for k, v in feature_dims.items() if v > 0])
        total_encoded_dim = total_encoders * 64  # 每个编码器输出64维
        
        # 符合论文描述的MLP骨干网络：[1024 × 1024 × 1024]
        self.fc1 = nn.Linear(total_encoded_dim, network_dim)
        self.fc2 = nn.Linear(network_dim, network_dim)
        self.fc3 = nn.Linear(network_dim, network_dim)
        # 策略头（输出动作logits）
        self.action_head = nn.Linear(network_dim, num_actions)
        # 值函数头（输出状态值）
        self.value_head = nn.Linear(network_dim, 1)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        完全向量化的前向传播，支持批量处理
        Args:
            features: 字典，包含各种输入特征张量
        Returns:
            action_logits: 动作logits [batch_size, num_actions] 或 [batch_size, num_envs, num_actions]
            value: 状态值 [batch_size] 或 [batch_size, num_envs]
        """
        # 编码各种特征
        encoded_features = self.feature_encoder(features)
        
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

def create_network(network_type="shared", feature_dims=None, num_actions=12, network_dim=1024):
    """
    创建网络实例的工厂函数
    Args:
        network_type: 网络类型 ("shared")
        feature_dims: 特征维度字典
        num_actions: 动作数量
        network_dim: 网络隐藏层维度
    Returns:
        网络实例
    """
    if feature_dims is None:
        # 默认特征维度
        feature_dims = {
            'road_boundary': 52,        # 道路边界特征
            'lane_points': 50,          # 车道点特征
            'stop_lines': 20,           # 停止线特征
            'vehicle_state': 7,         # 车辆状态特征
            'other_agents': 140,        # 其他智能体特征
            'conditioning': 512         # 条件和目标特征
        }

    if network_type == "shared":
        return SharedNetwork(feature_dims, num_actions, network_dim)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试网络
    model = create_network(network_type="shared", feature_dims=None, num_actions=12, network_dim=1024)
    print(f"🔍 模型参数数量: {count_parameters(model)}")
    print("🧪 测试神经网络...")
    
    # 创建示例输入用于测试
    batch_size = 32
    num_envs = 4800  # 每个GPU的环境数量
    feature_dims = {
        'road_boundary': 52,
        'lane_points': 50,
        'stop_lines': 20,
        'vehicle_state': 7,
        'other_agents': 140,
        'conditioning': 512
    }

    # 创建示例特征 - 支持多环境
    features = {
        'road_boundary': torch.randn(batch_size, num_envs, feature_dims['road_boundary']),
        'lane_points': torch.randn(batch_size, num_envs, feature_dims['lane_points']),
        'stop_lines': torch.randn(batch_size, num_envs, feature_dims['stop_lines']),
        'vehicle_state': torch.randn(batch_size, num_envs, feature_dims['vehicle_state']),
        'other_agents': torch.randn(batch_size, num_envs, feature_dims['other_agents']),
        'conditioning': torch.randn(batch_size, num_envs, feature_dims['conditioning'])
    }
    
    # 测试前向传播
    try:
        action_logits, value = model(features)
        print(f"✅ 前向传播成功")
        print(f"Action logits shape: {action_logits.shape}")
        print(f"Value shape: {value.shape}")
        
        # 测试反向传播
        loss = action_logits.sum() + value.sum()
        loss.backward()
        print(f"✅ 反向传播成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


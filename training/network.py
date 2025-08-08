# 神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeatureEncoder(nn.Module):
    """
    简单特征编码器 - 用于简单特征向量 (S(t), G(t), C_reward等)
    S(t) = [x, y, yaw, speed, length, width, active]    # 车辆状态
    G(t) = [x, y]                                       # 目标点
    C_reward = [reward, goal_reached]                   # 奖励和目标是否达到
    """
    def __init__(self, input_dim, output_dim=64):
        super(SimpleFeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class PermutationInvariantEncoder(nn.Module):
    """
    排列不变编码器 - 用于多特征集合 (W(t)_lane, W(t)_boundary, W(t)_stop, A(t))
    W(t)_lane = [dx, dy] for each lane point
    W(t)_boundary = [dx, dy] for each boundary point
    W(t)_stop = [dx, dy] for each stop line
    A(t) = [dx, dy, dvx, dvy, length, width, active] for each agent
    论文描述: 对每个特征类型，使用小型全连接网络编码每个元素，然后进行通道维度的maxpooling
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
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_elements, feature_dim] 或 [batch_size, feature_dim]
        Returns:
            encoded: [batch_size, output_dim]
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
        else:
            raise ValueError(f"输入维度错误: {x.shape}")

class FeatureEncoder(nn.Module):
    """
    特征编码器 - 符合论文描述的架构
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
        编码各种特征
        Args:
            features: 字典，包含各种特征
        Returns:
            concatenated_features: 拼接后的特征张量
        """
        encoded_features = []
        
        # 处理简单特征
        for feature_name, encoder in self.simple_encoders.items():
            if feature_name in features and features[feature_name] is not None:
                feature = features[feature_name]
                if torch.isnan(feature).any():
                    print(f"⚠️ {feature_name} 包含NaN，正在处理...")
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                encoded = encoder(feature)
                encoded_features.append(encoded)
        
        # 处理排列不变特征
        for feature_name, encoder in self.permutation_encoders.items():
            if feature_name in features and features[feature_name] is not None:
                feature = features[feature_name]
                if torch.isnan(feature).any():
                    print(f"⚠️ {feature_name} 包含NaN，正在处理...")
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                encoded = encoder(feature)
                encoded_features.append(encoded)
        
        # 拼接所有编码后的特征
        if encoded_features:
            concatenated_features = torch.cat(encoded_features, dim=-1)
        else:
            # 如果没有特征，创建零张量
            batch_size = next(iter(features.values())).shape[0] if features else 1
            total_encoders = len(self.simple_encoders) + len(self.permutation_encoders)
            concatenated_features = torch.zeros(batch_size, total_encoders * self.encoder_dim)
        
        return concatenated_features

class SharedNetwork(nn.Module):
    """
    共享网络（同时输出策略和值函数）
    符合论文描述的MLP架构：[1024 × 1024 × 1024]
    目标：每个网络约3百万参数
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
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化，提高训练稳定性
            torch.nn.init.xavier_normal_(module.weight, gain=0.1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 字典，包含各种输入特征
        Returns:
            action_logits: 动作logits [batch_size, num_actions]
            value: 状态值 [batch_size]
        """
        
        # 编码各种特征
        encoded_features = self.feature_encoder(features)
        
        # 检查编码后的特征
        if torch.isnan(encoded_features).any():
            print(f"⚠️ 编码特征包含NaN: {encoded_features}")
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

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = create_network(network_type="shared", feature_dims=None, num_actions=12, network_dim=1024)
    print(f"🔍 模型参数数量: {count_parameters(model)}")
    print("🧪 测试神经网络...")
    
    # 创建示例输入用于可视化
    batch_size = 1
    feature_dims = {
        'road_boundary': 52,
        'lane_points': 50,
        'stop_lines': 20,
        'vehicle_state': 7,
        'other_agents': 140,
        'conditioning': 512
    }

    # 创建示例特征
    features = {
        'road_boundary': torch.randn(batch_size, feature_dims['road_boundary']),
        'lane_points': torch.randn(batch_size, feature_dims['lane_points']),
        'stop_lines': torch.randn(batch_size, feature_dims['stop_lines']),
        'vehicle_state': torch.randn(batch_size, feature_dims['vehicle_state']),
        'other_agents': torch.randn(batch_size, feature_dims['other_agents']),
        'conditioning': torch.randn(batch_size, feature_dims['conditioning'])
    }
    
    # 使用torchviz可视化网络结构
    try:
        from torchviz import make_dot
        print("📊 使用torchviz可视化网络结构...")
        # 前向传播
        action_logits, value = model(features)
        print(action_logits)
        print(value)
        # 创建计算图
        dot = make_dot(action_logits, params=dict(model.named_parameters()))
        # 尝试保存为PNG格式
        try:
            dot.render("network_architecture", format="png", cleanup=True)
            print("✅ 网络结构图已保存为 network_architecture.png")
        except Exception as e:
            print(f"⚠️ 无法生成PNG图片: {e}")
            # 尝试保存为DOT文件
            dot.save("network_architecture.dot")
    except ImportError:
        print("⚠️ torchviz未安装，跳过可视化")


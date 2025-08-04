# 神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    """
    特征编码器 - 处理不同类型的输入特征
    对于road_boundary、lane_points、stop_lines、other_agents使用MLP+MaxPool
    对于vehicle_state和conditioning直接使用MLP
    """
    def __init__(self, feature_dims, hidden_dim):
        super(FeatureEncoder, self).__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # 需要MaxPool的特征类型
        self.maxpool_features = ['road_boundary', 'lane_points', 'stop_lines', 'other_agents']
        # 直接MLP的特征类型
        self.direct_features = ['vehicle_state', 'conditioning']
        
        # 为每种特征类型创建独立的编码器
        self.encoders = nn.ModuleDict()
        for feature_name, feature_dim in feature_dims.items():
            if feature_dim > 0:
                if feature_name in self.maxpool_features:
                    # 需要MaxPool的特征：MLP + MaxPool
                    self.encoders[feature_name] = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                    )
                else:
                    # 直接MLP的特征
                    self.encoders[feature_name] = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
    
    def forward(self, features):
        """
        编码各种特征
        
        Args:
            features: 字典，包含各种特征
                - road_boundary: 道路边界特征 [batch_size, road_boundary_dim]
                - lane_points: 车道点特征 [batch_size, lane_points_dim]  
                - stop_lines: 停止线特征 [batch_size, stop_lines_dim]
                - vehicle_state: 车辆状态特征 [batch_size, vehicle_state_dim]
                - other_agents: 其他智能体特征 [batch_size, other_agents_dim]
                - conditioning: 条件和目标特征 [batch_size, conditioning_dim]
                
        Returns:
            encoded_features: 编码后的特征张量
        """
        maxpool_features = []
        direct_features = []
        
        for feature_name, encoder in self.encoders.items():
            if feature_name in features and features[feature_name] is not None:
                feature = features[feature_name]
                # 检查NaN并处理
                if torch.isnan(feature).any():
                    print(f"⚠️ {feature_name} 包含NaN，正在处理...")
                    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
                
                encoded = encoder(feature)
                
                if feature_name in self.maxpool_features:
                    # 对于需要MaxPool的特征，先收集起来
                    maxpool_features.append(encoded)
                else:
                    # 对于直接MLP的特征，直接收集
                    direct_features.append(encoded)
        
        # 处理需要MaxPool的特征
        if maxpool_features:
            # 将所有需要MaxPool的特征拼接
            maxpool_input = torch.cat(maxpool_features, dim=-1)
            # 重塑为 [batch_size, num_features, hidden_dim] 以便进行MaxPool
            batch_size = maxpool_input.shape[0]
            num_features = len(maxpool_features)
            maxpool_input = maxpool_input.view(batch_size, num_features, self.hidden_dim)
            # 在特征维度上进行MaxPool
            maxpool_output = torch.max(maxpool_input, dim=1)[0]  # [batch_size, hidden_dim]
        else:
            maxpool_output = torch.zeros(features.get('vehicle_state', torch.zeros(1)).shape[0], self.hidden_dim)
        
        # 处理直接MLP的特征
        if direct_features:
            direct_output = torch.cat(direct_features, dim=-1)
        else:
            direct_output = torch.zeros(features.get('vehicle_state', torch.zeros(1)).shape[0], 
                                      self.hidden_dim * len(self.direct_features))
        
        # 拼接MaxPool输出和直接MLP输出
        final_features = torch.cat([maxpool_output, direct_output], dim=-1)
        return final_features

class SharedNetwork(nn.Module):
    """
    共享网络（同时输出策略和值函数）
    支持多种输入特征：道路边界、车道点、停止线、车辆状态、其他智能体、条件和目标
    """
    def __init__(self, feature_dims, num_actions, network_dim=64):
        super(SharedNetwork, self).__init__()
        
        # 特征编码器
        self.feature_encoder = FeatureEncoder(feature_dims, network_dim)
        
        # 计算编码后的特征总维度
        maxpool_features_count = len([k for k, v in feature_dims.items() if v > 0 and k in ['road_boundary', 'lane_points', 'stop_lines', 'other_agents']])
        direct_features_count = len([k for k, v in feature_dims.items() if v > 0 and k in ['vehicle_state', 'conditioning']])
        
        # MaxPool输出维度 + 直接MLP输出维度
        total_encoded_dim = network_dim + (network_dim * direct_features_count)
        
        # 共享的特征提取层
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
                - road_boundary: 道路边界特征 [batch_size, road_boundary_dim]
                - lane_points: 车道点特征 [batch_size, lane_points_dim]
                - stop_lines: 停止线特征 [batch_size, stop_lines_dim]
                - vehicle_state: 车辆状态特征 [batch_size, vehicle_state_dim]
                - other_agents: 其他智能体特征 [batch_size, other_agents_dim]
                - conditioning: 条件和目标特征 [batch_size, conditioning_dim]
            
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
            
        # 共享特征提取
        x = F.relu(self.fc1(encoded_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))    
        
        # 分别输出策略和值函数
        action_logits = self.action_head(x)
        value = self.value_head(x).squeeze(-1)
        
        return action_logits, value

def create_network(network_type="shared", feature_dims=None, num_actions=4, network_dim=64):
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
            'road_boundary': 20,    # 道路边界特征
            'lane_points': 30,       # 车道点特征
            'stop_lines': 10,        # 停止线特征
            'vehicle_state': 15,     # 车辆状态特征
            'other_agents': 25,      # 其他智能体特征
            'conditioning': 12       # 条件和目标特征
        }

    if network_type == "shared":
        return SharedNetwork(feature_dims, num_actions, network_dim)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = create_network(network_type="shared", feature_dims=None, num_actions=4, network_dim=128)
    print(f"🔍 模型参数数量: {count_parameters(model)}")
    print("🧪 测试神经网络...")

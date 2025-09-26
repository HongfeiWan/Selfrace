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
    排列不变编码器 - 支持集合输入并进行对称聚合（默认 max 池化）
    用于多特征集合 (W(t)_lane, W(t)_boundary, W(t)_stop, A(t))

    使用方式：
    - 输入 x 可为 [B, M, K, d]（K 个元素、每元素维度 d）；
    - 也可为 [B, M, N] 的扁平向量，但需在初始化时指定 element_dim（单元素维度 d），
      以便自动重塑为 [B, M, K=N//d, d] 并沿 K 维进行聚合（置换不变）。
    """
    def __init__(self, feature_dim, output_dim=64, element_dim=None):
        super(PermutationInvariantEncoder, self).__init__()
        self.flat_total_dim = feature_dim
        self.element_dim = element_dim  # 若提供，则 K = feature_dim // element_dim
        self.output_dim = output_dim

        element_input_dim = self.element_dim if self.element_dim is not None else feature_dim
        self.element_encoder = nn.Sequential(
            nn.Linear(element_input_dim, output_dim),
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
    
    def forward(self, x, mask: torch.Tensor = None):
        """
        排列不变前向传播（对 K 维做 max 池化）
        Args:
            x: [B, M, K, d] 或 [B, M, N]
            mask: 可选，[B, M, K]，True 表示该元素有效
        Returns:
            encoded: [B, M, output_dim]
        """
        if x.dim() == 3:
            B, M, N = x.shape
            if self.element_dim is not None:
                assert N % self.element_dim == 0, \
                    f"total_dim={N} 不能被 element_dim={self.element_dim} 整除"
                K = N // self.element_dim
                x = x.view(B, M, K, self.element_dim)
            else:
                x = x.unsqueeze(2)  # [B, M, 1, N]
        elif x.dim() != 4:
            raise ValueError(f"x 期望为 3D 或 4D 张量，得到 {x.dim()}D")

        B, M, K, d = x.shape
        encoded_elements = self.element_encoder(x.reshape(-1, d))  # [(B*M*K), output_dim]
        encoded_elements = encoded_elements.reshape(B, M, K, self.output_dim)

        if mask is not None:
            neg_inf = torch.finfo(encoded_elements.dtype).min
            encoded_elements = encoded_elements.masked_fill(~mask.unsqueeze(-1), neg_inf)

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
            SimpleFeatureEncoder(self.simple_feature_dims[0], self.encoder_dim),  # S(t): 7维
            SimpleFeatureEncoder(self.simple_feature_dims[1], self.encoder_dim),  # G(t): 256维
            SimpleFeatureEncoder(self.simple_feature_dims[2], self.encoder_dim),  # reward系数: 10维
            SimpleFeatureEncoder(self.simple_feature_dims[3], self.encoder_dim)   # 车辆风格参数: 4维
        ])
        # 创建排列不变特征编码器 - 直接创建4个
        self.permutation_encoders = nn.ModuleList([
            PermutationInvariantEncoder(self.permutation_feature_dims[0], self.encoder_dim, element_dim=2),  # road_boundary: 26x2=52
            PermutationInvariantEncoder(self.permutation_feature_dims[1], self.encoder_dim, element_dim=2),  # lane_points: 25x2=50
            PermutationInvariantEncoder(self.permutation_feature_dims[2], self.encoder_dim, element_dim=2),  # stop_lines: 5x2=20
            PermutationInvariantEncoder(self.permutation_feature_dims[3], self.encoder_dim, element_dim=7)   # other_agents: 20x7=140
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

class IndependentNetwork(nn.Module):
    """
    独立网络类 - 包含两个完全独立的网络，参数不共享
    分别负责输出动作和值函数，可以选择单独使用
    """
    def __init__(self, config):
        super(IndependentNetwork, self).__init__()
        
        # 从配置文件读取参数
        network_config = config.training.network
        self.network_dim = network_config.network_dim
        self.num_actions = network_config.num_actions
        
        # ============================== 策略网络（动作网络） ==============================
        # 策略网络的特征编码器
        self.policy_feature_encoder = FeatureEncoder(config)
        policy_encoded_dim = self.policy_feature_encoder.total_output_dim
        
        # 策略网络的MLP
        self.policy_network = nn.Sequential(
            nn.Linear(policy_encoded_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, self.num_actions)
        )
        
        # ============================== 值函数网络 ==============================
        # 值函数网络的特征编码器
        self.value_feature_encoder = FeatureEncoder(config)
        value_encoded_dim = self.value_feature_encoder.total_output_dim
        
        # 值函数网络的MLP
        self.value_network = nn.Sequential(
            nn.Linear(value_encoded_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.ReLU(),
            nn.Linear(self.network_dim, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, features_tensor, mode="both"):
        """
        前向传播 - 可选择使用策略网络、值函数网络或两者
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
            mode: "policy", "value", "both" - 选择使用哪个网络
        Returns:
            根据mode返回不同的输出
        """
        if mode == "policy":
            return self.forward_policy(features_tensor)
        elif mode == "value":
            return self.forward_value(features_tensor)
        elif mode == "both":
            return self.forward_both(features_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'policy', 'value', or 'both'")
    
    def forward_policy(self, features_tensor):
        """
        仅策略网络前向传播
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            action_logits: 动作logits [B, M, num_actions]
        """
        # 策略网络特征编码
        policy_encoded_features = self.policy_feature_encoder(features_tensor)
        # 向量化NaN处理
        if torch.isnan(policy_encoded_features).any():
            policy_encoded_features = torch.nan_to_num(policy_encoded_features, nan=0.0, posinf=1.0, neginf=-1.0)
        # 策略网络前向传播
        action_logits = self.policy_network(policy_encoded_features)
        return action_logits
    
    def forward_value(self, features_tensor):
        """
        仅值函数网络前向传播
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            value: 状态值 [B, M]
        """
        # 值函数网络特征编码
        value_encoded_features = self.value_feature_encoder(features_tensor)
        # 向量化NaN处理
        if torch.isnan(value_encoded_features).any():
            value_encoded_features = torch.nan_to_num(value_encoded_features, nan=0.0, posinf=1.0, neginf=-1.0)
        # 值函数网络前向传播
        value = self.value_network(value_encoded_features).squeeze(-1)
        return value
    
    def forward_both(self, features_tensor):
        """
        两个网络同时前向传播
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            action_logits: 动作logits [B, M, num_actions]
            value: 状态值 [B, M]
        """
        action_logits = self.forward_policy(features_tensor)
        value = self.forward_value(features_tensor)
        return action_logits, value
    
def create_network(config, network_type="shared"):
    """
    创建网络实例的工厂函数
    Args:
        config: 配置文件对象（必需）
        network_type: 网络类型 ("shared" 或 "independent")
    Returns:
        网络实例
    """
    if network_type == "shared":
        return SharedNetwork(config=config)
    elif network_type == "independent":
        return IndependentNetwork(config=config)
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
        
        # 测试共享网络
        print("\n🔍 测试共享网络 (SharedNetwork)...")
        shared_model = create_network(config=config, network_type="shared")
        shared_model = shared_model.to(device)
        print(f"🔍 共享网络参数数量: {count_parameters(shared_model)}")
        
        # 测试独立网络
        print("\n🔍 测试独立网络 (IndependentNetwork)...")
        independent_model = create_network(config=config, network_type="independent")
        independent_model = independent_model.to(device)
        print(f"🔍 独立网络参数数量: {count_parameters(independent_model)}")
        
        # 创建示例输入用于测试
        B = 2000
        M = 150
        features_tensor = torch.randn(B, M, shared_model.feature_encoder.total_input_dim, device=device)
        
        # 测试共享网络前向传播
        print("\n🧪 测试共享网络前向传播...")
        action_logits_shared, value_shared = shared_model(features_tensor)
        print(f"✅ 共享网络前向传播成功")
        print(f"Action logits shape: {action_logits_shared.shape}")
        print(f"Value shape: {value_shared.shape}")
        
        # 测试独立网络前向传播
        print("\n🧪 测试独立网络前向传播...")
        
        # 测试同时使用两个网络
        action_logits_indep, value_indep = independent_model(features_tensor, mode="both")
        print(f"✅ 独立网络双网络前向传播成功")
        print(f"Action logits shape: {action_logits_indep.shape}")
        print(f"Value shape: {value_indep.shape}")
        
        # 测试仅策略网络
        action_logits_policy = independent_model(features_tensor, mode="policy")
        print(f"✅ 独立网络仅策略网络前向传播成功")
        print(f"Policy Action logits shape: {action_logits_policy.shape}")
        
        # 测试仅值函数网络
        value_only = independent_model(features_tensor, mode="value")
        print(f"✅ 独立网络仅值函数网络前向传播成功")
        print(f"Value only shape: {value_only.shape}")
        
        # 测试反向传播
        print("\n🧪 测试反向传播...")
        loss_shared = action_logits_shared.sum() + value_shared.sum()
        loss_shared.backward()
        print(f"✅ 共享网络反向传播成功")
        
        # 测试独立网络的反向传播
        loss_indep_both = action_logits_indep.sum() + value_indep.sum()
        loss_indep_both.backward()
        print(f"✅ 独立网络双网络反向传播成功")
        
        # 测试仅策略网络的反向传播
        loss_policy = action_logits_policy.sum()
        loss_policy.backward()
        print(f"✅ 独立网络仅策略网络反向传播成功")
        
        # 测试仅值函数网络的反向传播
        loss_value = value_only.sum()
        loss_value.backward()
        print(f"✅ 独立网络仅值函数网络反向传播成功")
        
        print(f"共享网络参数数量: {count_parameters(shared_model)}, 独立网络参数数量: {count_parameters(independent_model)}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


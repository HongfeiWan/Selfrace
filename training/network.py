# 神经网络模块
import torch
import torch.nn as nn

from feature_schema import FEATURE_PAD_VALUE, FeatureSchema


DEFAULT_WEIGHT_INIT = {"type": "orthogonal", "gain": 1.0, "bias_zero": True}


def _config_value(container, name, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def resolve_weight_init(config):
    training_config = _config_value(config, "training")
    configured = _config_value(training_config, "weight_init", {})
    options = {
        "type": str(_config_value(configured, "type", DEFAULT_WEIGHT_INIT["type"])).lower(),
        "gain": float(_config_value(configured, "gain", DEFAULT_WEIGHT_INIT["gain"])),
        "bias_zero": bool(_config_value(configured, "bias_zero", DEFAULT_WEIGHT_INIT["bias_zero"])),
    }
    if options["type"] != "orthogonal":
        raise ValueError(f"unsupported weight initialization: {options['type']!r}")
    return options


def initialize_linear(module, options):
    if not isinstance(module, nn.Linear):
        return
    torch.nn.init.orthogonal_(module.weight, gain=options["gain"])
    if options["bias_zero"] and module.bias is not None:
        torch.nn.init.constant_(module.bias, 0)

class SimpleFeatureEncoder(nn.Module):
    """
    简单特征编码器 - 用于简单特征向量 (S(t), reward系数,车辆风格系数等)
    完全向量化，支持批量处理
    """
    def __init__(self, input_dim, output_dim=64, weight_init=None):
        super(SimpleFeatureEncoder, self).__init__()
        self._weight_init = weight_init or DEFAULT_WEIGHT_INIT
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # 应用Orthogonal初始化
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """初始化网络权重 - 使用Orthogonal初始化且bias为0"""
        initialize_linear(module, self._weight_init)
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
    def __init__(self, feature_dim, output_dim=64, element_dim=None, weight_init=None):
        super(PermutationInvariantEncoder, self).__init__()
        self.element_dim = element_dim  # 若提供，则 K = feature_dim // element_dim
        self.output_dim = output_dim
        self._weight_init = weight_init or DEFAULT_WEIGHT_INIT

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
        initialize_linear(module, self._weight_init)
    
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
        if mask is not None:
            flat_mask = mask.reshape(-1)
            flat_x = x.reshape(-1, d)
            valid_x = flat_x[flat_mask]
            encoded_valid = self.element_encoder(valid_x)
            neg_inf = torch.finfo(encoded_valid.dtype).min
            encoded_flat = torch.full(
                (B * M, self.output_dim),
                neg_inf,
                device=x.device,
                dtype=encoded_valid.dtype,
            )
            group_ids = torch.arange(
                B * M, device=x.device, dtype=torch.long
            ).repeat_interleave(K)[flat_mask]
            scatter_index = group_ids.unsqueeze(-1).expand(-1, self.output_dim)
            encoded_flat.scatter_reduce_(
                0, scatter_index, encoded_valid, reduce='amax', include_self=True
            )
            encoded = encoded_flat.view(B, M, self.output_dim)
            all_invalid = ~mask.any(dim=2)
            return torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(encoded), encoded)

        encoded_elements = self.element_encoder(x.reshape(-1, d))  # [(B*M*K), output_dim]
        encoded_elements = encoded_elements.reshape(B, M, K, self.output_dim)
        return torch.max(encoded_elements, dim=2)[0]  # [B, M, output_dim]

class FeatureEncoder(nn.Module):
    """
    完全向量化的特征编码器 - 通过配置文件指导参数
    输入为单个大张量，按固定位置切片提取特征
    """
    def __init__(self, config):
        super(FeatureEncoder, self).__init__()
        network_config = config.training.network
        self.encoder_dim = network_config.encoder_dim
        self._weight_init = resolve_weight_init(config)
        self.feature_schema = FeatureSchema.from_config(config)
        self.total_input_dim = self.feature_schema.total_input_dim
        self.simple_encoders = nn.ModuleList([
            SimpleFeatureEncoder(group.flat_dim, self.encoder_dim, weight_init=self._weight_init)
            for group in self.feature_schema.simple_groups
        ])
        self.permutation_encoders = nn.ModuleList([
            PermutationInvariantEncoder(
                group.flat_dim,
                self.encoder_dim,
                element_dim=group.element_dim,
                weight_init=self._weight_init,
            )
            for group in self.feature_schema.set_groups
        ])
        self.total_output_dim = (len(self.simple_encoders) + len(self.permutation_encoders)) * self.encoder_dim

    @staticmethod
    def _flat_set_mask(x, element_dim, active_channel=None):
        """从扁平特征中恢复 padding mask，避免空元素参与 max pooling。"""
        B, M, N = x.shape
        K = N // element_dim
        elements = x.view(B, M, K, element_dim)
        if active_channel is not None:
            return elements[..., active_channel] > 0.5
        return torch.isfinite(elements).all(dim=-1) & (elements > FEATURE_PAD_VALUE + 0.5).all(dim=-1)

    def forward(self, features_tensor):
        """
        完全向量化的特征编码 - 直接使用两个编码器列表
        Args:
            features_tensor: [B, M, total_input_dim] 所有特征拼接的大张量
        Returns:
            output: [B, M, total_output_dim] 编码后的特征张量
        """
        B, M, width = features_tensor.shape
        self.feature_schema.validate_tensor_width(width)
        features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        # 预分配输出张量 [B, M, total_output_dim]
        output = torch.zeros(B, M, self.total_output_dim, device=features_tensor.device, dtype=features_tensor.dtype)
        
        output_offset = 0
        for group, encoder in zip(self.feature_schema.simple_groups, self.simple_encoders):
            simple_feature = features_tensor[:, :, self.feature_schema.flat_slice(group.name)]
            output[:, :, output_offset:output_offset + self.encoder_dim] = encoder(simple_feature)
            output_offset += self.encoder_dim
        
        for group, encoder in zip(self.feature_schema.set_groups, self.permutation_encoders):
            set_features = features_tensor[:, :, self.feature_schema.flat_slice(group.name)]
            set_mask = self._flat_set_mask(
                set_features,
                element_dim=group.element_dim,
                active_channel=group.resolved_active_channel,
            )
            output[:, :, output_offset:output_offset + self.encoder_dim] = encoder(set_features, mask=set_mask)
            output_offset += self.encoder_dim
        
        return output

class IndependentNetwork(nn.Module):
    """
    独立网络类 - 包含两个完全独立的网络，参数不共享
    分别负责输出动作和值函数，可以选择单独使用
    """
    def __init__(self, config):
        super(IndependentNetwork, self).__init__()
        self._weight_init = resolve_weight_init(config)
        
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
        initialize_linear(module, self._weight_init)

    def policy_parameters(self):
        yield from self.policy_feature_encoder.parameters()
        yield from self.policy_network.parameters()

    def value_parameters(self):
        yield from self.value_feature_encoder.parameters()
        yield from self.value_network.parameters()
    
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
    
def create_network(config, network_type="independent"):
    """
    创建网络实例的工厂函数
    Args:
        config: 配置文件对象（必需）
        network_type: 仅支持原文采用的独立 actor/critic 网络
    Returns:
        网络实例
    """
    if network_type == "independent":
        return IndependentNetwork(config=config)
    raise ValueError(f"Unknown network type: {network_type}; expected 'independent'")

# GIGAFLOW

## RewardCalculator 模块文档

RewardCalculator 是 GIGAFLOW 项目中的奖励计算模块，负责为智能体计算各种奖励和惩罚。该模块遵循 GIGAFLOW 论文中描述的奖励结构。

### 模块概述

RewardCalculator 类提供了完整的奖励计算功能，包括目标奖励、碰撞惩罚、离路惩罚、舒适度惩罚、车道对齐奖励、车道中心对齐奖励、速度奖励、倒车惩罚、停止线违规惩罚和时间步惩罚。

### 主要函数

#### 1. `__init__(config: Dict, device: torch.device)`
**功能**: 初始化奖励计算器
**输入**:
- `config`: 包含奖励参数的配置字典
- `device`: 计算设备 (CPU/GPU)

**说明**: 初始化所有随机参数和固定参数，包括各种 alpha 值、目标速度、目标奖励等。

---

#### 2. `calculate_goal_reward(agent_positions, goal_positions, speeds, waypoint_reached)`
**功能**: 计算目标奖励
**输入**:

- `agent_positions` (torch.Tensor): 智能体位置 (B, M, 2) - (x, y)
- `goal_positions` (torch.Tensor): 目标位置 (B, M, 2)
- `speeds` (torch.Tensor): 智能体速度 (B, M)
- `waypoint_reached` (torch.Tensor): 是否到达路点 (B, M) - 布尔值

**输出**: `torch.Tensor` - 目标奖励 (B, M)

**公式**: `Rgoal = 1 if (||x-g|| < δgoal ∧ (waypoint_reached ∨ |v| < vgoal)) else 0`

---

#### 3. `calculate_collision_penalty(agents_state, all_collisions)`
**功能**: 计算碰撞惩罚
**输入**:
- `agents_state` (torch.Tensor): 智能体状态张量 (B, M, N)
- `all_collisions` (torch.Tensor): 碰撞掩码 (B, M)，布尔值张量

**输出**: `torch.Tensor` - 碰撞惩罚 (B, M)

**公式**: `Rcollision = -(αcollision + 0.1|v|)1collision`

---

#### 4. `calculate_offroad_penalty(offroad_mask)`
**功能**: 计算离路惩罚
**输入**:
- `offroad_mask` (torch.Tensor): 离路掩码 (B, M)，布尔值张量

**输出**: `torch.Tensor` - 离路惩罚 (B, M)

**公式**: `Roff-road = -αboundary1boundary`

---

#### 5. `calculate_comfort_penalty(along, alat, along_jerk, alat_jerk)`
**功能**: 计算舒适度惩罚
**输入**:
- `along` (torch.Tensor): 纵向加速度 (B, M)
- `alat` (torch.Tensor): 横向加速度 (B, M)
- `along_jerk` (torch.Tensor): 纵向加加速度 (B, M)
- `alat_jerk` (torch.Tensor): 横向加加速度 (B, M)

**输出**: `torch.Tensor` - 舒适度惩罚 (B, M)

**公式**: `Rcomfort = -αcomfort * (1|along|>3 + 1|alat|>3 + 1|along_jerk|>5 ∨ |alat_jerk|>5)`

---

#### 6. `calculate_lane_alignment_reward(theta_f, speeds, dt=0.3)`
**功能**: 计算车道对齐奖励
**输入**:
- `theta_f` (torch.Tensor): 车道角度误差 (B, M)，弧度
- `speeds` (torch.Tensor): 智能体速度 (B, M)
- `dt` (float): 时间步长，默认0.3

**输出**: `torch.Tensor` - 车道对齐奖励 (B, M)

**公式**: `R_{l-align} = α_{l-align} * Δt * (min(cos(θ_f), 0) + α_{vel-align} * min(cos(θ_f) * v, 0) + 0.0025 * (1 - |θ_f|/(π/2)))`

---

#### 7. `calculate_lane_center_reward(theta_f, d, dt=0.3)`
**功能**: 计算车道中心对齐奖励
**输入**:
- `theta_f` (torch.Tensor): 车道角度误差 (B, M)
- `d` (torch.Tensor): Frenet坐标系中的横向距离 (B, M)
- `dt` (float): 时间步长，默认0.3

**输出**: `torch.Tensor` - 车道中心对齐奖励 (B, M)

**公式**: `R_{l-center} = -α_{l-center} * Δt * (1_{cos(θ_f) > 0.5} * |d - α_{center-bias}| - 0.05/exp(|d - α_{center-bias}| - 0.5))`

---

#### 8. `calculate_velocity_reward(theta_f, speeds, dt=0.3)`
**功能**: 计算速度奖励
**输入**:
- `theta_f` (torch.Tensor): 车道角度误差 (B, M)
- `speeds` (torch.Tensor): 智能体速度 (B, M)
- `dt` (float): 时间步长，默认0.3

**输出**: `torch.Tensor` - 速度奖励 (B, M)

**公式**: `Rvelocity = αvelocity * Δt * max(cos(θ_f), 0.0) * 1_{|v| > 2.5}`

---

#### 9. `calculate_reverse_penalty(speeds, dt=0.3)`
**功能**: 计算倒车惩罚
**输入**:
- `speeds` (torch.Tensor): 智能体速度 (B, M)
- `dt` (float): 时间步长，默认0.3

**输出**: `torch.Tensor` - 倒车惩罚 (B, M)

**公式**: `Rreverse = -αreverse * Δt * 1_{v < 0}`

---

#### 10. `calculate_stop_line_penalty(stop_line_violation)`
**功能**: 计算停止线违规惩罚
**输入**:
- `stop_line_violation` (torch.Tensor): 停止线违规掩码 (B, M) - 布尔值

**输出**: `torch.Tensor` - 停止线违规惩罚 (B, M)

**公式**: `Rstop-line = -αstop-line * 1stop-line-violation`

---

#### 11. `calculate_timestep_penalty(speeds, along, alat, dt=0.3)`
**功能**: 计算时间步惩罚
**输入**:
- `speeds` (torch.Tensor): 智能体速度 (B, M)
- `along` (torch.Tensor): 纵向加速度 (B, M)
- `alat` (torch.Tensor): 横向加速度 (B, M)
- `dt` (float): 时间步长，默认0.3

**输出**: `torch.Tensor` - 时间步惩罚 (B, M)

**公式**: `Rtimestep = -(αtimestep * Δt) * 1_{|v| > 0 ∨ |a| > 0}`

---

#### 12. `calculate(agents_state, all_collisions, offroad_mask, dt=0.3, goal_positions=None, waypoint_reached=None, stop_line_violation=None)`
**功能**: 为所有智能体计算总奖励
**输入**:
- `agents_state` (torch.Tensor): 智能体状态张量，形状为 (B, M, N)
  - B: 批次大小 (batch size)
  - M: 智能体数量
  - N: 状态维度
  - 论文中的设置是[38400,150,10]
  - 其中 N = 10，包含以下字段:
    - [0]: x - 全局 x 坐标
    - [1]: y - 全局 y 坐标  
    - [2]: heading - 朝向角度 (弧度)
    - [3]: speed - 速度大小
    - [4]: along - 纵向加速度
    - [5]: alat - 横向加速度
    - [6]: along_jerk - 纵向加加速度
    - [7]: alat_jerk - 横向加加速度
    - [8]: theta_f - Frenet 角度误差
    - [9]: d - Frenet 横向距离
- `all_collisions` (torch.Tensor): 碰撞掩码 (B, M)，布尔值张量
- `offroad_mask` (torch.Tensor): 离路掩码 (B, M)，布尔值张量
- `goal_positions` (torch.Tensor): 目标位置 (B, M, 2)
- `waypoint_reached` (torch.Tensor): 路点到达掩码 (B, M)，布尔值张量
- `stop_line_violation` (torch.Tensor): 停止线违规掩码 (B, M)，布尔值张量

**输出**: `torch.Tensor` - 每个智能体的总奖励 (B, M)

**说明**: 这是主要的计算函数，整合了所有奖励和惩罚组件，返回每个智能体的总奖励。

---

### 辅助函数

#### `reset_episode()`
**功能**: 重置episode相关的随机参数
**输入**: 无
**输出**: 无
**说明**: 在每个新episode开始时调用，重新采样所有随机参数

#### `_initialize_random_parameters()`
**功能**: 初始化所有随机参数
**输入**: 无
**输出**: 无
**说明**: 内部函数，用于初始化所有随机参数

---

### 测试函数

模块还包含多个测试函数，用于验证各种奖励组件的正确性：

- `test_goal_reward()`: 测试目标奖励
- `test_collision_penalty()`: 测试碰撞惩罚
- `test_offroad_penalty()`: 测试离路惩罚
- `test_comfort_penalty()`: 测试舒适度惩罚
- `test_lane_alignment_reward()`: 测试车道对齐奖励
- `test_lane_center_reward()`: 测试车道中心对齐奖励
- `test_velocity_reward()`: 测试速度奖励
- `test_reverse_penalty()`: 测试倒车惩罚
- `test_stop_line_penalty()`: 测试停止线违规惩罚
- `test_timestep_penalty()`: 测试时间步惩罚

每个测试函数都会生成相应的可视化图表，帮助分析奖励组件的分布和特性。






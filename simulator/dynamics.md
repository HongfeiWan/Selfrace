# dynamics.py 变量与结构说明

本文档概述 `simulator/dynamics.py` 中 `DiscreteActionSpace` 和 `KinematicBicycleModel` 的核心数据结构、输入输出与使用方式，便于阅读、调试与上层调用。

## 目录
- 类与初始化
- DiscreteActionSpace
- KinematicBicycleModel
- 状态变量与控制流程
- 方法说明
- 使用建议

---

## 类与初始化

### DiscreteActionSpace(device: torch.device, config: Dict)

- 作用：定义离散动作空间，支持jerk（加加速度）控制。将连续的jerk值离散化为有限的动作集合，便于强化学习算法使用。
- 输入：
  - `device`: 张量所处设备（`cpu` 或 `cuda`）
  - `config`: 配置字典，包含jerk范围参数（可选）
- 处理流程：
  1) 从配置读取jerk范围参数（默认为：纵向[-15, 4]，横向[-4, 4]）
  2) 生成纵向和横向jerk值列表
  3) 创建所有可能的动作组合（4×3=12个动作）
  4) 转换为PyTorch张量存储在指定设备上

### KinematicBicycleModel(config: Dict, device: torch.device, vehicle_params: Dict[str, torch.Tensor])

- 作用：实现精确且批量化的运动学自行车模型。使用运动学方程的解析解确保离散时间步长内的物理准确性，特别是在转弯场景中。所有操作都已完全向量化，支持批量处理。
- 输入：
  - `config`: 包含车辆物理参数的配置字典（支持从`simulator.dynamics`子配置读取）
  - `device`: 计算设备（`cpu` 或 `cuda`）
  - `vehicle_params`: 批量车辆参数字典，必需包含：
    - `'length'`: (N,) 车辆长度（米）
    - `'width'`: (N,) 车辆宽度（米）
    - `'wheelbase'`: (N,) 车辆轴距（米）
- 处理流程：
  1) 从配置读取动力学参数（转向角限制、加速度约束、速度约束等）
  2) 采样驾驶风格参数（`Cthrottle`, `Csteer`, `Cacc`, `Cvel`）
  3) 初始化离散动作空间
  4) 初始化控制状态变量（延迟初始化，在首次`step`时根据batch_size创建）

---

## DiscreteActionSpace

### 成员变量

- `along_values: List[float]`
  - 纵向jerk的离散值列表，默认：`[min_long_jerk, -max_long_jerk, 0, max_long_jerk]`
  - 典型值：`[-15.0, -4.0, 0.0, 4.0]` m/s³
- `alat_values: List[float]`
  - 横向jerk的离散值列表，默认：`[min_lat_jerk, 0, max_lat_jerk]`
  - 典型值：`[-4.0, 0.0, 4.0]` m/s³
- `actions: List[List[float]]`
  - 所有可能的动作组合列表，长度为12，每个元素为`[along_jerk, alat_jerk]`
- `num_actions: int`
  - 动作总数，固定为12
- `actions_tensor: FloatTensor (num_actions, 2)`
  - 所有动作的张量表示，形状为`(12, 2)`，设备由`device`参数指定

### 方法

- `get_action(action_idx: torch.Tensor) -> torch.Tensor`
  - 根据动作索引获取实际动作值
  - 输入：动作索引张量，形状为`(N,)`或标量
  - 输出：实际动作值，形状为`(N, 2)`或`(2,)`
- `get_all_actions() -> torch.Tensor`
  - 获取所有动作的张量表示

---

## KinematicBicycleModel

### 配置参数

以下参数从配置字典`config`中读取（支持从`simulator.dynamics`子配置读取）：

- `vehicle_max_steer_angle`: 最大转向角（度），默认35.0°，转换为弧度存储在`max_steer_rad`
- `max_steering_rate`: 最大转向角变化率（rad/s），默认0.6
- `max_longitudinal_accel`: 最大纵向加速度（m/s²），默认2.5
- `min_longitudinal_accel`: 最小纵向加速度（m/s²），默认-5.0
- `max_lateral_accel`: 最大横向加速度（m/s²），默认4.0
- `min_lateral_accel`: 最小横向加速度（m/s²），默认-4.0
- `max_velocity`: 最大速度（m/s），默认20.0
- `min_velocity`: 最小速度（m/s），默认-2.0
- `curvature_epsilon`: 曲率计算的数值稳定性参数，默认1e-5

### 车辆参数

- `vehicle_params: Dict[str, torch.Tensor]`
  - `'length'`: (N,) 车辆长度（米）
  - `'width'`: (N,) 车辆宽度（米）
  - `'wheelbase'`: (N,) 车辆轴距（米）
  - 批量参数应与`world_init`采样顺序一致

### 驾驶风格参数

通过`DrivingStyleSampler`采样生成（批量）：

- `Cthrottle: FloatTensor (N,)`
  - 油门控制系数，影响纵向jerk的响应速度
- `Csteer: FloatTensor (N,)`
  - 转向控制系数，影响横向jerk的响应速度
- `Cacc: FloatTensor (N,)`
  - 加速度系数，用于缩放最大纵向加速度
- `Cvel: FloatTensor (N,)`
  - 速度系数，用于缩放最大速度

### 离散动作空间

- `discrete_action_space: DiscreteActionSpace`
  - 离散动作空间实例，用于将动作索引转换为jerk值

### 控制状态变量

以下变量在首次调用`step`时根据`batch_size`延迟初始化：

- `current_along: FloatTensor (N,)`
  - 当前纵向加速度（m/s²），从jerk积分得到，应用约束后存储
- `current_alat: FloatTensor (N,)`
  - 当前横向加速度（m/s²），根据实际转向角重新计算以确保物理一致性
- `current_steering_angle: FloatTensor (N,)`
  - 当前有效转向角（弧度），受转向角变化率限制
- `prev_along: FloatTensor (N,)`
  - 前一步的纵向加速度，用于梯形积分计算速度

---

## 状态变量与控制流程

### 状态表示

- `states: FloatTensor (N, 4)`
  - `[x, y, yaw, speed]`
  - `x, y`: 位置坐标（米）
  - `yaw`: 偏航角（弧度），范围[-π, π]
  - `speed`: 速度（m/s），可正可负（正为前进，负为倒车）

### step 方法控制流程

1. **Jerk到加速度更新**
   - 从动作索引获取jerk值：`[along_jerk, alat_jerk]`
   - 更新加速度：`new_along = current_along + along_jerk * dt * Cthrottle`
   - 更新横向加速度：`new_alat = current_alat + alat_jerk * dt * Csteer`

2. **符号变化检测**
   - 检测纵向加速度符号变化：如果`a(t-1)_long * a(t)_long < 0`，将纵向和横向加速度都置零
   - 检测速度符号变化：如果速度从正变负或从负变正，将速度置零

3. **约束应用**
   - 纵向加速度：`a(t)_long ← clip(a(t)_long, min_long_accel, max_long_accel*Cacc)`
   - 横向加速度：`a(t)_lat ← clip(a(t)_lat, min_lat_accel, max_lat_accel)`
   - 速度：`v(t) ← clip(v(t), min_velocity, max_velocity*Cvel)`

4. **速度更新（梯形积分）**
   - `v(t) = v(t-1) + 0.5 * (a(t)_long + a(t-1)_long) * dt`

5. **转向角计算与限制**
   - 从横向加速度反推目标转向角：`φ_target = arctan((alat / v²) * l_wb)`
   - 限制转向角变化率：`δφ = clip(φ_target - φ(t-1), -max_steering_rate*dt, max_steering_rate*dt)`
   - 更新转向角：`φ(t) = clip(φ(t-1) + δφ, -φmax, φmax)`

6. **物理一致性修正**
   - 从实际转向角重新计算曲率：`ρ^(-1) = tan(φ(t)) / l_wb`
   - 重新计算横向加速度：`a_lat(t) = v² * ρ^(-1)`
   - 再次应用横向加速度约束，更新`current_alat`

7. **位置更新**
   - 使用平均速度计算位移：`d = 0.5(v(t) + v(t-1)) * dt`
   - 计算角位移：`θ = d * ρ^(-1)`
   - 位置变化：`Δx = d * cos(yaw)`, `Δy = d * sin(yaw)`
   - 偏航角变化：`Δyaw = θ`
   - 归一化偏航角到[-π, π]范围

---

## 方法说明

### step(states: torch.Tensor, actions: torch.Tensor, dt: float) -> torch.Tensor

对一批车辆状态进行一步精确更新。

- 输入：
  - `states`: 形状为`(N, 4)`的当前状态张量`[x, y, yaw, speed]`
  - `actions`: 形状为`(N,)`的动作索引张量
  - `dt`: 模拟时间步长（秒）
- 输出：
  - `torch.Tensor`: 形状为`(N, 4)`的下一时刻状态张量
- 注意事项：
  - `batch_size`必须与初始化时的`num_vehicles`匹配
  - 控制状态变量会在首次调用或`batch_size`变化时自动初始化

### reset_control_state()

重置控制状态（加速度和转向角），清除所有控制状态变量。

- 作用：让`step`方法在下次调用时重新初始化正确的`batch_size`
- 清除的变量：
  - `current_along`
  - `current_alat`
  - `current_steering_angle`
  - `prev_along`（如果存在）

### calculate_steering_angle(alat: torch.Tensor, speed: torch.Tensor, wheelbases: torch.Tensor, epsilon: float = None) -> torch.Tensor

根据横向加速度和速度计算转向角。

- 输入：
  - `alat`: 横向加速度（m/s²）
  - `speed`: 速度（m/s）
  - `wheelbases`: 批量轴距参数，形状为`(N,)`
  - `epsilon`: 数值稳定性参数（可选，默认使用配置值）
- 输出：
  - `torch.Tensor`: 转向角（弧度），形状为`(N,)`
- 计算公式：
  - 曲率：`ρ^(-1) = alat / max(v², ε)`
  - 转向角：`φ = arctan(ρ^(-1) * l_wb)`

### get_discrete_action_space() -> DiscreteActionSpace

获取离散动作空间实例。

---

## 使用建议

### 初始化流程

1. 准备配置字典：包含`simulator.dynamics`子配置或根级别的`dynamics`配置
2. 采样车辆参数：使用`VehicleParameterSampler`生成批量车辆参数
3. 创建模型实例：`dynamics_model = KinematicBicycleModel(config, device, vehicle_params)`

### 状态更新流程

1. 准备初始状态：形状为`(N, 4)`的张量`[x, y, yaw, speed]`
2. 选择动作：从离散动作空间中选择动作索引（0-11）
3. 调用step：`new_states = dynamics_model.step(states, actions, dt)`
4. 访问控制状态：通过`dynamics_model.current_along`、`current_alat`、`current_steering_angle`获取当前控制状态

### 批量处理注意事项

- 确保`batch_size`与初始化时的`num_vehicles`一致
- 控制状态变量会根据`batch_size`自动调整，但建议在同一episode内保持`batch_size`不变
- 如需重置控制状态（如新episode开始），调用`reset_control_state()`

### 数值稳定性

- 模型内置数值稳定性处理，包括：
  - 曲率计算的epsilon保护
  - 速度接近零时的特殊处理
  - 角度归一化到合理范围
- 如遇到数值问题，可调整配置中的`curvature_epsilon`参数

### 与强化学习集成

- 动作空间：使用`dynamics_model.get_discrete_action_space()`获取动作空间信息
- 状态空间：`(N, 4)`的状态张量，可直接用于状态表示
- 控制状态访问：可通过模型成员变量访问加速度、转向角等控制状态，用于奖励计算或观察空间扩展


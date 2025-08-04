# Gigaflow

## RoadNetwork 模块
RoadNetwork 是 GIGAFLOW 项目中的道路网络管理模块，负责加载和管理从预处理的 CARLA 地图数据中提取的道路网络。该模块将地图数据（主要是四边形路块 'quads'）加载到 PyTorch 张量中，以便于在 GPU 上进行高效的批量化计算。

### 模块概述
RoadNetwork 类提供了查询地图几何信息（如车道中心线、边界线）的核心功能，并实现了 Frenet 坐标系的计算，包括横向距离 d 和角度误差 θ_f。该模块是自动驾驶系统中路径规划和车辆定位的重要组成部分。

### 主要函数

#### 1. `__init__(map_path: str, device: torch.device)`
**功能**: 初始化道路网络
**输入**:
- `map_path` (str): 指向预处理后的地图 JSON 文件的路径
- `device` (torch.device): 用于存储地图数据的计算设备 ('cpu' 或 'cuda')

**说明**: 加载和处理地图数据，计算道路几何信息，存储元数据和全局航点。

---

#### 2. `_load_map_data(map_path: str) -> Dict`
**功能**: 从 JSON 文件加载地图数据
**输入**:
- `map_path` (str): 地图文件路径

**输出**: `Dict` - 地图数据字典

**说明**: 内部函数，负责读取和解析地图 JSON 文件。

---

#### 3. `_process_map_data(map_data: Dict)`
**功能**: 处理地图数据并转换为 PyTorch 张量
**输入**:
- `map_data` (Dict): 从 JSON 加载的地图数据

**说明**: 提取顶点坐标，计算道路几何信息，存储元数据，加载全局航点。

---

#### 4. `_extract_vertices(quads_data)`
**功能**: 提取顶点坐标
**输入**:
- `quads_data`: quads 数据列表

**输出**: 四个顶点张量 (p0, p1, p2, p3)

**说明**: 顶点顺序映射:
- p0 (left_start) -> vertices[2]
- p1 (left_end) -> vertices[1] 
- p2 (right_end) -> vertices[0]
- p3 (right_start) -> vertices[3]

---

#### 5. `_compute_road_geometry(vertices)`
**功能**: 计算道路几何信息
**输入**:
- `vertices`: 四个顶点张量 (p0, p1, p2, p3)

**说明**: 计算道路中心线、边界线、方向向量等几何信息。

---

#### 6. `_store_metadata(quads_data)`
**功能**: 存储元数据
**输入**:
- `quads_data`: quads 数据列表

**说明**: 存储 quad IDs、lane IDs 和关联的航点信息。

---

#### 7. `_load_global_waypoints(map_data)`
**功能**: 加载全局航点
**输入**:
- `map_data` (Dict): 地图数据字典

**说明**: 加载车道航点和边界航点数据。

---

#### 8. `get_all_lanes_left_boundaries() -> torch.Tensor`
**功能**: 返回所有车道左边界线段
**输出**: `torch.Tensor` - 形状为 (num_quads, 2, 2) 的张量

---

#### 9. `get_all_lanes_right_boundaries() -> torch.Tensor`
**功能**: 返回所有车道右边界线段
**输出**: `torch.Tensor` - 形状为 (num_quads, 2, 2) 的张量

---

#### 10. `get_all_lanes_centerlines() -> torch.Tensor`
**功能**: 返回地图上所有 quad 的中心线段
**输出**: `torch.Tensor` - 形状为 (num_quads, 2, 2) 的张量，代表所有中心线的起点和终点

---

#### 11. `find_nearest_lanes(points: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]`
**功能**: 为一批输入点找到最近的 k 个车道 (quads)
**输入**:
- `points` (torch.Tensor): 形状为 (N, 2) 的点坐标张量
- `k` (int): 需要为每个点找到的最近车道的数量，默认为1

**输出**: `Tuple[torch.Tensor, torch.Tensor]`
- `distances`: 形状为 (N, k) 的距离张量
- `indices`: 形状为 (N, k) 的最近车道 (quads) 的索引张量

**说明**: 使用欧氏距离计算点到车道中心点的距离，返回最近的 k 个车道。

---

#### 12. `get_global_waypoints_by_ids(ids: torch.Tensor, point_type: str) -> torch.Tensor`
**功能**: 根据ID列表从全局航点库中获取航点坐标
**输入**:
- `ids` (torch.Tensor): 航点ID列表
- `point_type` (str): 点类型 ('w_lane' 或 'w_boundary')

**输出**: `torch.Tensor` - 航点坐标张量

**说明**: 支持车道航点和边界航点的查询。

---

#### 13. `calculate_frenet_coordinates(vehicle_positions: torch.Tensor, vehicle_headings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
**功能**: 计算车辆在Frenet坐标系中的横向距离d和角度误差theta_f
**输入**:
- `vehicle_positions` (torch.Tensor): 车辆位置，形状为 (B, M, 2) 或 (N, 2)
- `vehicle_headings` (torch.Tensor): 车辆朝向角度（弧度），形状为 (B, M) 或 (N,)

**输出**: `Tuple[torch.Tensor, torch.Tensor]`
- `d`: 横向距离，正值表示在道路右侧，负值表示在道路左侧
- `theta_f`: 角度误差（弧度），正值表示车辆朝向偏左，负值表示偏右

**算法说明**:
1. **找到最近道路段**: 使用 `find_nearest_lanes` 找到距离车辆最近的道路段
2. **计算车辆朝向向量**: 将角度转换为单位向量 `[cos(heading), sin(heading)]`
3. **计算横向距离**: 使用二维叉积计算车辆到道路的垂直距离
   ```
   AP = vehicle_position - road_start
   d = AP × road_direction
   ```
4. **计算角度误差**: 使用 atan2 计算车辆朝向与道路方向的夹角
   ```
   theta_f = atan2(vehicle_direction × road_direction, vehicle_direction · road_direction)
   ```

**符号约定**:
- **d > 0**: 车辆在道路右侧
- **d < 0**: 车辆在道路左侧
- **d = 0**: 车辆在道路中心线上
- **theta_f > 0**: 车辆朝向偏左（需要向右转向）
- **theta_f < 0**: 车辆朝向偏右（需要向左转向）
- **theta_f = 0**: 车辆朝向与道路方向一致

---

### 数据结构

#### 主要属性
- `quads_vertices` (torch.Tensor): 所有 quads 的顶点坐标 (num_quads, 4, 2)
- `quad_centerlines` (torch.Tensor): 所有 quads 的中心线 (num_quads, 2, 2)
- `left_boundaries` (torch.Tensor): 所有 quads 的左边界 (num_quads, 2, 2)
- `right_boundaries` (torch.Tensor): 所有 quads 的右边界 (num_quads, 2, 2)
- `quad_directions` (torch.Tensor): 所有 quads 的方向向量 (num_quads, 2)
- `quad_ids` (torch.Tensor): 所有 quads 的ID (num_quads,)
- `lane_ids` (torch.Tensor): 所有 quads 的车道ID (num_quads,)
- `global_w_lane_waypoints` (torch.Tensor): 全局车道航点
- `global_w_boundary_points` (torch.Tensor): 全局边界航点

---

### 测试功能

模块包含完整的测试功能，可以独立运行：

```python
if __name__ == '__main__':
    # 测试地图加载和Frenet坐标计算
    # 包括地图可视化、车辆位置生成、Frenet坐标计算等
```

**测试功能包括**:
- 地图数据加载和验证
- 道路几何信息计算
- 最近道路段查找
- Frenet坐标计算
- 地图可视化（包括quads、中心线、车辆位置等）

---

## RewardCalculator 模块
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






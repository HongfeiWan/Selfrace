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

RoadNetwork 模块存储了以下主要变量及其内容：

#### 主要属性

##### 几何信息
- `self.quads_vertices` (torch.Tensor): 所有 quads 的顶点坐标 (num_quads, 4, 2)
  - 存储每个quad的四个顶点坐标
  - 顶点顺序: p0 (left_start) -> vertices[2], p1 (left_end) -> vertices[1], p2 (right_end) -> vertices[0], p3 (right_start) -> vertices[3]

- `self.quad_centerlines` (torch.Tensor): 所有 quads 的中心线 (num_quads, 2, 2)
  - 存储每个quad的中心线起点和终点坐标
  - 用于计算道路方向和Frenet坐标

- `self.left_boundaries` (torch.Tensor): 所有 quads 的左边界 (num_quads, 2, 2)
  - 存储每个quad的左边界线段起点和终点

- `self.right_boundaries` (torch.Tensor): 所有 quads 的右边界 (num_quads, 2, 2)
  - 存储每个quad的右边界线段起点和终点

- `self.quad_directions` (torch.Tensor): 所有 quads 的方向向量 (num_quads, 2)
  - 存储每个quad的归一化方向向量
  - 用于计算Frenet坐标系中的角度误差

##### 标识信息
- `self.quad_ids` (torch.Tensor): 所有 quads 的ID (num_quads,)
  - 存储每个quad的唯一标识符

- `self.lane_ids` (torch.Tensor): 所有 quads 的车道ID (num_quads,)
  - 存储每个quad所属的车道标识

##### 航点信息
- `self.global_w_lane_waypoints` (torch.Tensor): 全局车道航点
  - 存储地图中所有车道航点的坐标
  - 用于路径规划和导航

- `self.global_w_boundary_points` (torch.Tensor): 全局边界航点
  - 存储地图中所有边界航点的坐标
  - 用于边界检测和离路判断

##### 关联信息
- `self.quad_w_lane_ids_assoc` (List): 每个quad关联的车道航点ID列表
  - 存储每个quad与车道航点的关联关系
- `self.quad_w_boundary_ids_assoc` (List): 每个quad关联的边界航点ID列表
  - 存储每个quad与边界航点的关联关系

##### 统计信息
- `self.num_quads` (int): quads的总数量
  - 记录地图中quad的总数
- `self.device` (torch.device): 计算设备
  - 存储张量所在的设备（CPU或GPU）

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

---

## OffroadChecker 模块
OffroadChecker 是 GIGAFLOW 项目中的离路检测模块，负责高效地检测车辆是否在道路上。该模块使用基于 GPU 加速的批量化算法，结合空间哈希索引来快速判断车辆位置是否在道路边界内。
### 模块概述
OffroadChecker 类提供了高性能的离路检测功能，通过将车辆边界框离散化为多个采样点，并使用射线投射法（Ray Casting）来判断这些点是否在道路多边形内。该模块是自动驾驶系统中安全性和合规性检查的重要组成部分。
### 主要函数
#### 1. `__init__(map_data: RoadNetwork, spatial_hash: SpatialHash, points_per_vehicle_edge: int = 3)`
**功能**: 初始化离路检测器
**输入**:
- `map_data` (RoadNetwork): 包含路面几何信息的 RoadNetwork 对象
- `spatial_hash` (SpatialHash): 预初始化的空间哈希对象
- `points_per_vehicle_edge` (int): 沿着车辆边界框每条边采样的点数，默认为3

**说明**: 初始化检测器，构建静态路面索引，创建本地边界框点模板。使用共享的 spatial_hash 来提高查询效率。

---

#### 2. `_create_local_bbox_points() -> Tensor`
**功能**: 为单位尺寸的边界框创建点模板
**输入**: 无
**输出**: `torch.Tensor` - 形状为 (num_points, 2) 的点模板张量

**说明**: 创建单位尺寸边界框（范围从-0.5到0.5）的点模板，包括边界上的采样点和中心点。这些点将在后续计算中被缩放和旋转到实际的车辆位置。

---

#### 3. `_get_discretized_bounding_boxes(states: Tensor) -> Tensor`
**功能**: 将本地边界框点集根据车辆状态转换到世界坐标系
**输入**:
- `states` (torch.Tensor): 车辆状态张量，形状为 (N, 5)
  - [0]: x - 车辆x坐标
  - [1]: y - 车辆y坐标
  - [2]: heading - 车辆朝向角度（弧度）
  - [3]: length - 车辆长度
  - [4]: width - 车辆宽度

**输出**: `torch.Tensor` - 形状为 (N, num_points, 2) 的世界坐标点张量

**算法说明**:
1. **缩放**: 根据车辆尺寸缩放点模板
2. **旋转**: 根据车辆朝向旋转点集
3. **平移**: 将点集平移到车辆实际位置

---

#### 4. `_batch_point_in_polygon_test(points: Tensor) -> Tensor`
**功能**: 使用射线投射法执行并行的"点在多边形内"测试
**输入**:
- `points` (torch.Tensor): 待测试的点坐标，形状为 (M, 2)

**输出**: `torch.Tensor` - 形状为 (M,) 的布尔张量，True表示点在道路内

**算法说明**:
1. **空间查询**: 使用空间哈希快速找到候选道路多边形
2. **射线投射**: 对每个点向任意方向发射射线，统计与多边形边界的交点数
3. **奇偶判断**: 如果交点数为奇数，则点在多边形内；否则在多边形外

**数学原理**: 基于射线投射定理，从点向任意方向发射射线，统计与多边形边界的交点数。奇数个交点表示点在多边形内部。

---

#### 5. `check_on_road(states: Tensor) -> Tensor`
**功能**: 批量检测车辆是否在道路上
**输入**:
- `states` (torch.Tensor): 车辆状态张量，形状为 (N, 5)
  - [0]: x - 车辆x坐标
  - [1]: y - 车辆y坐标
  - [2]: heading - 车辆朝向角度（弧度）
  - [3]: length - 车辆长度
  - [4]: width - 车辆宽度

**输出**: `torch.Tensor` - 形状为 (N,) 的布尔张量，True表示车辆在道路上

**算法流程**:
1. **边界框离散化**: 将车辆边界框转换为世界坐标系中的采样点
2. **批量点测试**: 对所有采样点执行"点在多边形内"测试
3. **整体判断**: 只有当车辆边界框的所有采样点都在道路内时，才认为车辆在道路上

**说明**: 这是主要的检测函数，整合了所有子步骤，返回每个车辆的离路状态。

---

### 数据结构

OffroadChecker 模块存储了以下主要变量及其内容：

#### 主要属性

##### 几何信息
- `self.road_polygons` (torch.Tensor): 道路多边形顶点，形状为 (num_polygons, num_vertices, 2)
  - 存储所有道路多边形的顶点坐标
  - 用于射线投射算法的几何计算

- `self.local_bbox_points` (torch.Tensor): 本地边界框点模板，形状为 (num_points, 2)
  - 存储单位尺寸边界框的采样点
  - 包括边界上的采样点和中心点

##### 配置参数
- `self.points_per_vehicle_edge` (int): 每条边的采样点数
  - 控制边界框离散化的精度
  - 默认值为3，可根据需要调整

- `self.device` (torch.device): 计算设备
  - 存储张量所在的设备（CPU或GPU）

##### 空间索引
- `self.spatial_hash` (SpatialHash): 空间哈希对象
  - 用于快速查询候选道路多边形
  - 提高大规模场景下的查询效率

---

### 测试函数

**测试功能包括**:
- 地图数据加载和验证
- 随机车辆位置生成
- 多朝向车辆状态测试
- 离路检测结果验证
- Frenet坐标计算
- 地图可视化（包括quads、车辆位置、检测结果等）

**可视化输出**:
- 道路网络地图显示
- 车辆位置和朝向的可视化
- 在道路/离路状态的视觉区分
- Frenet坐标计算结果


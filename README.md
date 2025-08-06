# Gigaflow自动驾驶

## RandomizeComponents 模块
RandomizeComponents 是 GIGAFLOW 项目中的随机参数采样模块，负责为仿真环境中的车辆和智能体生成各种随机参数。该模块包含三个主要的采样器类：DrivingStyleSampler、RewardParameterSampler 和 VehicleParameterSampler，分别用于采样车辆行驶风格、奖励计算参数和车辆物理参数。
### 模块概述
RandomizeComponents 模块提供了完整的随机参数生成功能，支持批量采样和约束条件应用。所有采样器都支持 GPU 加速，能够高效地处理大规模仿真场景。该模块是自动驾驶系统中参数多样化和真实性的重要组成部分。
### 主要类
#### DrivingStyleSampler 类
**功能**: 车辆行驶风格抽样器，从混合均匀分布中采样车辆控制参数
**说明**: 从混合均匀分布 X(a) = 0.5U(a-1,1) + 0.5U(1,a) 中采样 Cthrottle、Csteer 和 Cacc，其中 a > 1，用于生成不同的车辆行驶风格。

##### 主要函数

###### 1. `__init__(device: torch.device = None)`
**功能**: 初始化行驶风格抽样器
**输入**:

- `device` (torch.device): 计算设备，默认为 'cuda'

**说明**: 设置计算设备，用于后续的张量操作。

---

###### 2. `sample_mixed_uniform(a: float, size: int = 1) -> torch.Tensor`
**功能**: 从混合均匀分布 X(a) = 0.5U(a-1,1) + 0.5U(1,a) 中采样
**输入**:
- `a` (float): 混合均匀分布参数，必须大于1
- `size` (int): 采样数量，默认为1

**输出**: `torch.Tensor` - 采样的值，形状为 (size,)

**算法说明**:
- 50% 的概率从第一个均匀分布 U(a-1,1) 采样
- 50% 的概率从第二个均匀分布 U(1,a) 采样
- 返回混合后的采样结果

---

###### 3. `sample_driving_style(size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]`
**功能**: 采样车辆行驶风格参数 Cthrottle 和 Csteer
**输入**:
- `size` (int): 采样数量，默认为1

**输出**: `Tuple[torch.Tensor, torch.Tensor]` - (Cthrottle, Csteer) 参数对

**说明**: 从 X(1.25) 分布采样 Cthrottle 和 Csteer 参数。

---

###### 4. `sample_driving_Cacc(size: int = 1) -> torch.Tensor`
**功能**: 采样车辆行驶风格参数 Cacc
**输入**:
- `size` (int): 采样数量，默认为1

**输出**: `torch.Tensor` - Cacc 参数

**说明**: 从 X(1.5) 分布采样 Cacc 参数。

---

###### 5. `sample_driving_Cvel(size: int = 1) -> torch.Tensor`
**功能**: 采样车辆行驶风格参数 Cvel
**输入**:
- `size` (int): 采样数量，默认为1

**输出**: `torch.Tensor` - Cvel 参数

**说明**: 从 X(1.5) 分布采样 Cvel 参数。

---

###### 6. `get_distribution_info(a: float) -> Dict`
**功能**: 获取分布信息
**输入**:
- `a` (float): 混合均匀分布参数

**输出**: `Dict` - 包含分布参数的字典

**返回内容**:
- `a`: 分布参数
- `distribution`: 分布公式描述
- `support`: 支撑区间
- `expected_value`: 期望值

---

#### RewardParameterSampler 类
**功能**: 奖励参数采样器，用于从各种分布中采样奖励计算所需的参数
**说明**: 该类负责管理所有与奖励计算相关的随机参数采样，支持从配置文件中加载参数范围。

##### 主要函数

###### 1. `__init__(config: Dict, device: torch.device)`
**功能**: 初始化奖励参数采样器
**输入**:
- `config` (Dict): 包含奖励参数的配置字典
- `device` (torch.device): 计算设备

**说明**: 从配置中加载所有参数范围，初始化采样器。

---

###### 2. `_load_parameter_ranges()`
**功能**: 加载所有参数的范围配置
**输入**: 无
**输出**: 无

**说明**: 内部函数，从配置中提取各种奖励参数的最小值和最大值。

---

###### 3. `sample_delta_goal() -> torch.Tensor`
**功能**: 从均匀分布采样delta_goal值
**输入**: 无
**输出**: `torch.Tensor` - 采样的delta_goal值

**说明**: 用于目标奖励计算的距离阈值参数。

---

###### 4. `sample_collision_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样碰撞alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于碰撞惩罚计算的权重参数。

---

###### 5. `sample_boundary_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样边界alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于离路惩罚计算的权重参数。

---

###### 6. `sample_comfort_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样舒适度alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于舒适度惩罚计算的权重参数。

---

###### 7. `sample_l_align_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样车道对齐alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于车道对齐奖励计算的权重参数。

---

###### 8. `sample_vel_align_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样速度对齐alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于速度对齐奖励计算的权重参数。

---

###### 9. `sample_l_center_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样车道中心对齐alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于车道中心对齐奖励计算的权重参数。

---

###### 10. `sample_center_bias_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样中心偏置alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于车道中心对齐的偏置参数。

---

###### 11. `sample_reverse_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样倒车alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于倒车惩罚计算的权重参数。

---

###### 12. `sample_stop_line_alpha() -> torch.Tensor`
**功能**: 从均匀分布采样停止线alpha值
**输入**: 无
**输出**: `torch.Tensor` - 采样的alpha值

**说明**: 用于停止线违规惩罚计算的权重参数。

---

###### 13. `sample_all_parameters() -> Dict[str, torch.Tensor]`
**功能**: 采样所有参数并返回字典
**输入**: 无
**输出**: `Dict[str, torch.Tensor]` - 包含所有采样参数的字典

**返回内容**:
- `delta_goal`: 目标距离阈值
- `collision_alpha`: 碰撞惩罚权重
- `boundary_alpha`: 边界惩罚权重
- `comfort_alpha`: 舒适度惩罚权重
- `l_align_alpha`: 车道对齐权重
- `vel_align_alpha`: 速度对齐权重
- `l_center_alpha`: 车道中心对齐权重
- `center_bias_alpha`: 中心偏置参数
- `reverse_alpha`: 倒车惩罚权重
- `stop_line_alpha`: 停止线惩罚权重

---

#### VehicleParameterSampler 类
**功能**: 批量车辆参数采样器，用于world_init中多辆车的批量采样
**说明**: 支持批量采样车辆长度、宽度和轴距，并应用约束条件，确保车辆参数的物理合理性。

##### 主要函数

###### 1. `__init__(config: Dict, device: torch.device)`
**功能**: 初始化车辆参数采样器
**输入**:
- `config` (Dict): 包含车辆参数的配置字典
- `device` (torch.device): 计算设备

**说明**: 从配置中加载车辆参数的边界值，设置轴距比例。

---

###### 2. `sample_batch_vehicle_parameters(batch_size: int) -> Dict[str, torch.Tensor]`
**功能**: 批量采样车辆参数
**输入**:
- `batch_size` (int): 批量大小，即要采样的车辆数量

**输出**: `Dict[str, torch.Tensor]` - 包含车辆参数的字典
- `length`: 车辆长度 [batch_size]
- `width`: 车辆宽度 [batch_size] (已应用约束)
- `wheelbase`: 轴距 [batch_size]

**算法说明**:
1. **长度采样**: 从配置的范围内均匀采样车辆长度
2. **宽度采样**: 从配置的范围内均匀采样车辆宽度
3. **约束应用**: 应用宽度约束 `width = min(width, length)`
4. **轴距计算**: 计算轴距 `wheelbase = length * 0.6`

---

###### 3. `sample_single_vehicle_parameters() -> Dict[str, torch.Tensor]`
**功能**: 采样单个车辆参数
**输入**: 无
**输出**: `Dict[str, torch.Tensor]` - 包含单个车辆参数的字典

**说明**: 调用批量采样函数，返回单个车辆的参数。

---

###### 4. `get_vehicle_bounds() -> Dict[str, torch.Tensor]`
**功能**: 获取车辆参数的边界值
**输入**: 无
**输出**: `Dict[str, torch.Tensor]` - 包含边界值的字典

**返回内容**:
- `length_min`: 长度最小值
- `length_max`: 长度最大值
- `width_min`: 宽度最小值
- `width_max`: 宽度最大值
- `wheelbase_ratio`: 轴距比例

---

### 辅助函数

#### 1. `load_config_from_yaml(config_path: str) -> dict`
**功能**: 从YAML文件加载配置
**输入**:
- `config_path` (str): 配置文件路径

**输出**: `dict` - 配置字典

**说明**: 支持错误处理，如果文件不存在或解析失败，返回空字典。

---

### 测试函数

模块包含完整的测试功能，可以独立运行：

#### 1. `test_reward_parameter_sampler()`
**功能**: 测试 RewardParameterSampler 类的参数采样功能
**测试内容**:
- 批量参数采样
- 参数范围验证
- 分布统计信息
- 可视化分布图
- 均匀分布验证

**输出**: 
- 控制台统计信息
- 分布图保存为 'reward_parameter_distributions.png'

---

#### 2. `test_driving_style_sampler()`
**功能**: 测试 DrivingStyleSampler 类的参数采样功能
**测试内容**:
- 混合均匀分布采样
- 行驶风格参数采样
- 分布信息获取
- 参数范围验证

**输出**: 控制台统计信息

---

#### 3. `test_vehicle_parameter_sampler()`
**功能**: 测试 VehicleParameterSampler 类的参数采样功能
**测试内容**:
- 批量车辆参数采样
- 约束条件验证
- 参数统计信息
- 分布可视化
- 边界值测试

**输出**:
- 控制台统计信息
- 分布图保存为 'vehicle_parameter_distributions.png'
- 约束条件验证结果

---

### 数据结构

RandomizeComponents 模块包含以下主要数据结构：

#### DrivingStyleSampler 属性
- `self.device` (torch.device): 计算设备

#### RewardParameterSampler 属性
- `self.device` (torch.device): 计算设备
- `self.reward_config` (Dict): 奖励配置字典
- 各种参数范围属性（如 `delta_goal_min`, `collision_alpha_max` 等）

#### VehicleParameterSampler 属性
- `self.device` (torch.device): 计算设备
- `self.vehicle_length_min/max` (float): 车辆长度范围
- `self.vehicle_width_min/max` (float): 车辆宽度范围
- `self.wheelbase_ratio` (float): 轴距比例（0.6）

---

### 配置参数

模块支持从YAML配置文件加载参数，主要配置项包括：

#### 车辆参数配置 (dynamics)
- `vehicle_length_min`: 车辆长度最小值
- `vehicle_length_max`: 车辆长度最大值
- `vehicle_width_min`: 车辆宽度最小值
- `vehicle_width_max`: 车辆宽度最大值

#### 奖励参数配置 (reward)
- `delta_goal_min/max`: 目标距离阈值范围
- `collision_alpha_min/max`: 碰撞惩罚权重范围
- `boundary_alpha_min/max`: 边界惩罚权重范围
- `comfort_alpha_min/max`: 舒适度惩罚权重范围
- `l_align_alpha_min/max`: 车道对齐权重范围
- `vel_align_alpha_min/max`: 速度对齐权重范围
- `l_center_alpha_min/max`: 车道中心对齐权重范围
- `center_bias_alpha_min/max`: 中心偏置参数范围
- `reverse_alpha_min/max`: 倒车惩罚权重范围
- `stop_line_alpha_min/max`: 停止线惩罚权重范围

---

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

## Observation模块
ObservationGenerator 是 GIGAFLOW 项目中的观测生成模块，负责为批次中的所有智能体生成局部观测。该模块通过完全向量化的操作，一次性为所有环境中的所有智能体高效地计算观测，避免了Python循环，最大限度地利用GPU并行能力。
### 模块概述
ObservationGenerator 类提供了完整的观测生成功能，包括邻居检测、坐标转换、地图特征提取等。该模块将全局世界坐标系下的智能体状态和地图信息转换为每个智能体的局部坐标系，生成包含局部状态、邻居特征、车道航点和边界点的观测向量。该模块已经通过测试，确保观测生成的正确性和效率。
### 主要函数
#### 1. `__init__(road_network: RoadNetwork, config: Dict, device: torch.device)`
**功能**: 初始化观测生成器
**输入**:
- `road_network` (RoadNetwork): 道路网络对象，包含地图几何信息
- `config` (Dict): 包含观测参数的配置字典
- `device` (torch.device): 计算设备 (CPU/GPU)

**说明**: 从配置中加载观测参数，包括邻居数量、车道数量、边界数量、视野范围等。

---

#### 2. `get_observation_dim() -> int`
**功能**: 计算观测向量的总维度
**输入**: 无
**输出**: `int` - 观测向量的总维度

**计算方式**:
```python
total_dim = local_state_dim + 
            (num_neighbors * neighbor_feature_dim) + 
            (num_w_lanes * waypoint_feature_dim) + 
            (num_w_boundaries * boundary_feature_dim)
```

**说明**: 动态计算观测维度，支持不同配置下的维度变化。

---

#### 3. `generate(agents_state: torch.Tensor) -> torch.Tensor`
**功能**: 为所有环境中的所有智能体生成一批观测
**输入**:
- `agents_state` (torch.Tensor): 全局状态张量 (B, M, 7)
  - B: 批次大小 (batch size)
  - M: 智能体数量
  - 7个特征: [x, y, yaw, speed, vehicle_length, vehicle_width, active]

**输出**: `torch.Tensor` - 展平后的观测向量张量 (B, M, feature_dim)

**算法流程**:

1. **获取世界坐标系特征**: 调用 `_get_nearest_neighbors` 和 `_get_nearby_global_points` 获取邻居状态和地图特征
2. **坐标转换**: 调用 `_world_to_ego_centric` 将所有特征转换到每个智能体的局部坐标系
3. **展平拼接**: 将局部状态、邻居特征、车道航点和边界点拼接成最终的观测向量

**返回的观测组成**:

- `agents_state`: (B, M, 7) - 智能体自身的全局状态(x, y, yaw, speed, length, width, active)
- `neighbors_local`: (B, M, K, 7) - 邻居特征 (dx, dy, vx, vy, length, width, active)
- `w_lanes_local`: (B, M, N_lanes, 2) - 车道航点相对位置
- `w_boundaries_local`: (B, M, N_boundaries, 2) - 边界点相对位置

---

#### 4. `_get_nearest_neighbors(agents_state: torch.Tensor) -> torch.Tensor`
**功能**: 为每个智能体找到最近的K个邻居，完全向量化版本
**输入**:

- `agents_state` (torch.Tensor): 智能体状态张量 (B, M, 7)

**输出**: `torch.Tensor` - 邻居状态张量 (B, M, K, 7)

**算法说明**:
1. **距离计算**: 使用 `torch.cdist` 计算所有智能体之间的配对距离
2. **邻居筛选**: 应用多层筛选条件
   - 智能体不能是其自身的邻居 (对角线掩码)
   - 不活跃的智能体不能作为邻居
   - 距离超过视野范围的邻居不考虑
3. **最近邻居选择**: 使用 `torch.topk` 选择最近的K个邻居
4. **状态收集**: 使用高级索引高效地收集邻居状态

---

#### 5. `_get_nearby_global_points(agents_state: torch.Tensor, source_points: torch.Tensor, num_points: int) -> torch.Tensor`
**功能**: 为所有智能体从全局点集中找到k个最近的点
**输入**:
- `agents_state` (torch.Tensor): 智能体状态张量 (B, M, 7)
- `source_points` (torch.Tensor): 源点集 (如车道航点或边界点)
- `num_points` (int): 需要找到的最近点数量

**输出**: `torch.Tensor` - 最近点坐标张量 (B, M, num_points, 2)

**算法说明**:
1. **距离计算**: 使用 `torch.cdist` 计算智能体位置到所有源点的距离
2. **最近点选择**: 使用 `torch.topk` 选择最近的num_points个点
3. **坐标收集**: 收集选中点的坐标

---

#### 6. `_world_to_ego_centric(ego_states, neighbor_states, w_lanes_world, w_boundaries_world)`
**功能**: 将世界坐标系下的特征转换到每个智能体的局部坐标系
**输入**:
- `ego_states` (torch.Tensor): 智能体状态 (B, M, 7)
- `neighbor_states` (torch.Tensor): 邻居状态 (B, M, K, 7)
- `w_lanes_world` (torch.Tensor): 世界坐标系下的车道航点 (B, M, N_lanes, 2)
- `w_boundaries_world` (torch.Tensor): 世界坐标系下的边界点 (B, M, N_boundaries, 2)

**输出**: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
- `local_state`: 局部状态 (B, M, 7) - 包含全局坐标 (x, y, yaw, speed, length, width, active)
- `neighbors_local`: 局部邻居特征 (B, M, K, 7)
- `w_lanes_local`: 局部车道航点 (B, M, N_lanes, 2)
- `w_boundaries_local`: 局部边界点 (B, M, N_boundaries, 2)

**坐标转换算法**:

1. **旋转矩阵构建**: 使用标准2D旋转矩阵 `[[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]`
2. **批量坐标转换**: 使用 `batch_rotate` 函数进行批量转换
   
   ```python
   def batch_rotate(points_world, ego_pos, rot_matrix):
       rel_pos = points_world - ego_pos.unsqueeze(2)
       B, M, N, D = rel_pos.shape
       return torch.bmm(rel_pos.view(B*M, N, D), rot_matrix.view(B*M, D, D)).view(B, M, N, D)
   ```
3. **邻居特征转换**: 包括位置、速度、长度、宽度和活跃状态的转换

---

### 数据结构

ObservationGenerator 模块包含以下主要变量及其内容：

#### 配置参数
- `self.num_neighbors` (int): 邻居数量，默认为1
- `self.num_w_lanes` (int): 车道航点数量，默认为25
- `self.num_w_boundaries` (int): 边界点数量，默认为26
- `self.horizon` (float): 视野范围，默认为100.0米
- `self.local_state_dim` (int): 局部状态维度，默认为7 (x, y, yaw, speed, length, width, active)
- `self.neighbor_feature_dim` (int): 邻居特征维度，默认为7 (dx, dy, vx, vy, length, width, active)
- `self.waypoint_feature_dim` (int): 航点特征维度，默认为2 (dx, dy)
- `self.boundary_feature_dim` (int): 边界特征维度，默认为2 (dx, dy)

#### 对象引用
- `self.road_network` (RoadNetwork): 道路网络对象
- `self.config` (Dict): 配置字典
- `self.device` (torch.device): 计算设备

---

### 观测向量组成

观测向量包含以下四个主要部分：

#### 1. 智能体状态 (agents_state)
**维度**: (B, M, 7)
**内容**: 智能体自身的全局状态信息
- `[0]`: x - 全局x坐标
- `[1]`: y - 全局y坐标
- `[2]`: yaw - 朝向角度（弧度）
- `[3]`: speed - 速度大小
- `[4]`: length - 车辆长度
- `[5]`: width - 车辆宽度
- `[6]`: active - 活跃状态 (0或1)

#### 2. 邻居特征 (neighbors)
**维度**: (B, M, K, 7)
**内容**: 附近智能体的相对位置、速度、长度、宽度、活跃状态
- `[0]`: dx - 相对于ego的x方向位置
- `[1]`: dy - 相对于ego的y方向位置
- `[2]`: vx - 相对于ego的x方向速度
- `[3]`: vy - 相对于ego的y方向速度
- `[4]`: length - 邻居车辆长度
- `[5]`: width - 邻居车辆宽度
- `[6]`: active - 邻居活跃状态 (0或1)

#### 3. 车道航点 (w_lanes)
**维度**: (B, M, N_lanes, 2)
**内容**: 附近车道航点的相对位置
- `[0]`: dx - 相对于ego的x方向位置
- `[1]`: dy - 相对于ego的y方向位置

#### 4. 边界点 (w_boundaries)
**维度**: (B, M, N_boundaries, 2)
**内容**: 附近道路边界点的相对位置
- `[0]`: dx - 相对于ego的x方向位置
- `[1]`: dy - 相对于ego的y方向位置

---

### 动态维度计算

ObservationGenerator 支持动态计算观测维度，通过 `get_observation_dim()` 方法根据当前配置计算总维度：

```python
def get_observation_dim(self) -> int:
    local_state_size = self.local_state_dim  # 局部状态维度
    neighbors_size = self.num_neighbors * self.neighbor_feature_dim  # 邻居特征维度
    w_lanes_size = self.num_w_lanes * self.waypoint_feature_dim  # 车道航点维度
    w_boundaries_size = self.num_w_boundaries * self.boundary_feature_dim  # 边界点维度
    
    total_dim = local_state_size + neighbors_size + w_lanes_size + w_boundaries_size
    return total_dim
```

**优势**:
- **灵活性**: 可以根据配置自动调整网络输入维度
- **一致性**: 确保训练代码和观测生成器使用相同的维度
- **可维护性**: 修改配置时不需要手动更新硬编码的维度值
- **错误预防**: 避免维度不匹配的问题

---

### 坐标转换系统

#### 坐标系定义
- **世界坐标系 (World Coordinate System)**: 全局坐标系，所有智能体和地图元素的绝对位置
- **Ego坐标系 (Ego-centric Coordinate System)**: 以每个智能体为中心的局部坐标系
  - X轴: 智能体前进方向
  - Y轴: 智能体右侧方向

#### 坐标转换过程

##### 旋转矩阵构建
```python
# 标准2D旋转矩阵，用于行向量乘法
cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
rot_matrix = torch.stack([
    torch.stack([cos_yaw, -sin_yaw], dim=-1), 
    torch.stack([sin_yaw, cos_yaw], dim=-1)
], dim=-2)
```

##### 批量坐标转换
```python
def batch_rotate(points_world, ego_pos, rot_matrix):
    # points_world: (B, M, N, 2), ego_pos: (B, M, 2), rot_matrix: (B, M, 2, 2)
    rel_pos = points_world - ego_pos.unsqueeze(2)
    B, M, N, D = rel_pos.shape
    return torch.bmm(rel_pos.view(B*M, N, D), rot_matrix.view(B*M, D, D)).view(B, M, N, D)
```

#### 邻居状态转换
对于每个邻居智能体，需要转换：
1. **位置转换**: 相对位置从世界坐标转换到ego坐标
2. **速度转换**: 世界坐标系下的速度分量转换到ego坐标系
3. **尺寸转换**: 长度和宽度直接保持原值
4. **活跃状态**: 直接保持布尔值

```python
# 位置转换
rel_pos_neighbors = neighbor_states[..., :2] - ego_pos.unsqueeze(2)
local_pos_neighbors = torch.bmm(
    rel_pos_neighbors.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)
).view(B, M, K_neighbors, 2)

# 速度转换
neighbor_speed = neighbor_states[..., 3]
neighbor_yaw = neighbor_states[..., 2]
vx_world = neighbor_speed * torch.cos(neighbor_yaw)
vy_world = neighbor_speed * torch.sin(neighbor_yaw)
v_world = torch.stack([vx_world, vy_world], dim=-1)
v_local = torch.bmm(
    v_world.view(B*M, K_neighbors, 2), rot_matrix.view(B*M, 2, 2)
).view(B, M, K_neighbors, 2)

# 尺寸和状态
length = neighbor_states[..., 4].unsqueeze(-1)
width = neighbor_states[..., 5].unsqueeze(-1)
active_flag = neighbor_states[..., 6].unsqueeze(-1)
neighbors_local = torch.cat([local_pos_neighbors, v_local, length, width, active_flag], dim=-1)
```

---

### 测试功能

模块包含完整的测试功能，可以独立运行：

```python
if __name__ == '__main__':
    # 测试RoadNetwork和ObservationGenerator
    # 包括地图加载、车辆生成、观测生成、可视化等
```

**测试功能包括**:
- 地图数据加载和验证
- 随机车辆位置生成
- 观测生成和验证
- 坐标转换正确性检查
- 邻居检测算法验证
- 地图特征提取测试
- 动态维度计算验证
- 可视化功能（包括车辆绘制、观测可视化等）

**可视化输出**:
- 道路网络地图显示
- 车辆位置和朝向的可视化
- 观测结果的可视化（车道航点、边界点、邻居位置等）
- 邻居车辆的位置、速度和尺寸可视化
- Frenet坐标计算结果
- 观测维度信息输出

---
```
/teraflow_replication
|
├── configs/
|   └── default_config.yaml    # 用于所有超参数的集中式YAML文件。
├── simulator/
|   ├── __init__.py
|   ├── simulator.py         # 主要的TeraFlowSimulator协调器类。
|   ├── world_init.py        # 用于初始化世界和智能体的逻辑。
|   ├── dynamics.py          # 批处理的车辆动力学模型。
|   ├── road.py              # 道路表示和定位。
|   ├── collision.py         # 批处理的碰撞检测逻辑。
|   └── observation.py       # 按需构建观测。
├── policy/
|   ├── __init__.py
|   ├── networks.py          # Actor (策略) 和 Critic (价值) 网络架构。
|   └── encoders.py          # 用于处理集合输入的置换不变编码器。
├── training/
|   ├── __init__.py
|   ├── ppo.py               # 带有优势筛选的PPO算法。
|   ├── advantage_filter.py  # 特定的优势筛选逻辑。
|   └── train_loop.py        # 运行训练的主脚本。
├── maps/
|   └── carla_maps/
|       └── carla_map_data_Town01_stitched.json         #初始地图.json文件
|   └── carla_connect_final.py      # 对原始carla地图进行处理，生成初始地图.json文件
|   └── preprocessor.py             # 对初始.json文件二次处理，生成oob_points，连通路块，识别交叉路口。处理航点信息
|   └── carla_connect_final.py      # 对原始carla地图进行处理，生成初始地图.json文件
|
└── utils/
    ├── __init__.py
    └── spatial_hash.py      # 关键的基于GPU的空间哈希工具。
```
### A. 模拟器核心 (simulator/)

* **`GigaFlowSimulator` (simulator.py)**：中央协调器。其 `step()` 方法是模拟的核心，它按照附录 A 中描述的精确顺序调用子组件。它持有形状为 `(N, A, state_dim)` 的主 `state_tensor`（状态张量）。
* **`SpatialHash` (utils/spatial_hash.py)**：这是最关键的性能组件。它将所有地图几何形状（四边形、越界点）预先分桶到一个 2D 网格中。在运行时，它支持极快的查找（`query_points`、`query_boxes`）以找到附近的实体，避免了 `O(N*A*P)` 或 `O(N*A^2)` 的复杂度，并使得大规模定位和碰撞检测成为可能。
* **`RoadModel` 和 `RoadLocalizer` (simulator/road.py)**：`RoadModel` 加载地图，该地图被预处理成一个由凸四边形组成的张量。`RoadLocalizer` 使用 `SpatialHash` 来高效地为每个智能体找到候选的道路多边形，并执行到 Frenet 坐标的批量转换（`to_frenet`）。
* **`CollisionDetector` (simulator/collision.py)**：实现附录 A.5 中的碰撞检测逻辑。它首先使用 `SpatialHash` 和智能体移动的轴对齐边界框（AABBs）来找到潜在的碰撞对，然后仅对这些碰撞对执行更精确的检查。
* **`DynamicsModel` (simulator/dynamics.py)**：一个纯粹的、向量化的实现，源自附录 B.2 中的加加速度驱动的自行车模型 (jerk-actuated bicycle model)。它接收当前状态和智能体动作，并完全通过张量运算计算下一个状态。
* **`WorldInitializer` (simulator/world_init.py)**：处理将多达 150 个智能体放置在无碰撞配置中的复杂任务。如论文中所述，这使用了顺序拒绝采样 (sequential rejection sampling)，该方法难以完美地进行批处理，但为了性能可以进行近似处理。

---

### B. 强化学习 (RL) 与策略 (policy/ 和 training/)

* **`GigaFlowPolicy` (policy/networks.py)**：实现统一的 Actor-Critic 网络（附录 D）。其 `forward` 方法接受一个包含观测张量的字典。
* **`PermutationInvariantEncoder` (policy/encoders.py)**：关键的是，它使用这些编码器（MLP 后接最大池化）来处理基于集合的观测（`W_lane`、`W_boundary`、`A_t`），使得网络对智能体或地图特征的顺序不敏感。
* **`ObservationBuilder` (simulator/observation.py)**：一个至关重要的模块，它根据紧凑的 `world_state` 张量按需重构完整的观测字典，如论文中为节省内存而提到的。它以批处理方式查询 `SpatialHash`，为每个智能体找到附近的智能体和地图特征。
* **`AdvantageFilteringPPO` (training/ppo.py)**：这不是一个标准的 PPO。它用“优势筛选” (Advantage Filtering) 技术（附录 C，算法 1）包装了核心的 PPO 逻辑。
* **`AdvantageFilter` (training/advantage_filter.py)**：在更新步骤之前，此组件计算所有收集到的经验的 GAE 优势 (GAE advantage)，然后丢弃多达 80% 的低绝对优势的转换。这使得梯度更新集中在信息最丰富、最关键的经验上。

---

### 4. 关键设计考量

#### 数据流与张量形状：

* **模拟步骤 (`simulator.step`)**:
    * **输入**: `actions` 张量，形状为 `(N, A)`。
    * `DynamicsModel` 接收 `state_tensor`（形状 `(N, A, state_dim)`）和 `actions` -> 输出 `next_state_tensor`（形状 `(N, A, state_dim)`）。
    * `CollisionDetector` 接收 `state_tensor` 和 `next_state_tensor` -> 输出 `collisions` 布尔张量（形状 `(N, A)`）。
    * `ObservationBuilder` 接收 `next_state_tensor` -> 输出 `obs_dict`，其中每个值都是一个张量，例如 `obs['nearby_agents']` 的形状为 `(N*A, num_observed_agents, agent_feature_dim)`。
* **训练步骤 (`ppo.update`)**:
    * `RolloutBuffer` 收集转换数据。`rewards`, `dones`, `log_probs` 的形状均为 `(num_steps, N, A)`。
    * `AdvantageFilter` 计算 `advantages`（形状 `(num_steps, N, A)`）并根据其大小筛选缓冲区。
    * PPO 循环从未经筛选的、更小的缓冲区中提取的小批量数据上进行迭代。

#### **配置管理**
所有超参数、随机化范围和模型参数都在 `configs/default_config.yaml` 中管理。这对可复现性至关重要。您应该使用此文件来控制一切，从学习率到附录 B.4 中描述的交通信号灯时间的随机化。

#### **随机化** (附录 B.4)
广泛的随机化（车辆尺寸、目标、动力学、奖励、交通信号灯）是模型泛化能力的关键。这应该通过批处理的 GPU 操作来实现。在每个回合开始时（即当一个智能体完成任务时），您应该从配置文件中指定的范围为其采样新的参数。例如，创建一个形状为 `(N, A, num_params)` 的随机系数张量，并将其传递给相关模块（`DynamicsModel`、`RewardCalculator`）。

---
### 潜在陷阱与应对策略

* **性能**：`SpatialHash` 是最大的性能瓶颈。一个低效的实现将使项目瘫痪。重点是使其查询操作高度优化并完全批处理化。要不懈地对此组件进行性能分析。
* **数值稳定性**：动力学模型涉及除法运算（例如，除以速度的平方）。这可能导致 `NaNs`（非数值）。如附录 B.2 所述，在分母上添加一个小的 epsilon (1e-6) 以确保稳定性。
* **调试**：调试数千个并行模拟是出了名的困难。一个错误可能仅在 38,400 个世界中的一个发生。
    * **策略**：实现一个“调试模式”。当检测到罕见错误（如状态中出现 `NaN`）时，保存整个 `state_tensor` 以及导致该错误的动作序列。编写一个可以加载此状态并仅重播单个失败世界的独立脚本，从而实现有针对性的调试。
* **可视化**：尽早构建简单的可视化工具。一个能够接收来自单个世界的已保存轨迹并将其渲染为俯瞰图像或动画的脚本是无价的。

这个架构蓝图提供了一个坚实的起点。接下来的步骤将涉及填写已创建文件中的 `TODO` 部分，从 `SpatialHash` 和 `DynamicsModel` 开始，因为它们构成了模拟器的核心。

---

## 脚本与函数总览

### simulator/
- **simulator.py**
  - `class TeraflowSimulator`：主模拟器协调器，负责环境批量管理、步进、重置等。
  - `def __init__`：初始化模拟器，加载地图、初始化各组件（动力学、碰撞检测、观测生成等）。
  - `def reset`：重置所有环境，生成初始智能体状态，返回初始观测。
  - `def step`：执行一步模拟，更新智能体状态，检测碰撞和离路，计算奖励和终止条件。
  - `def _get_observation`：为所有智能体生成观测，调用观测生成器。
  - `def _calculate_reward`：计算所有智能体的奖励，基于状态、碰撞、离路等信息。
  - `def _check_done`：检查所有智能体是否应该结束（碰撞或离路）。
  - `def render`：可视化当前所有环境状态（接口方法）。

- **dynamics.py**
  - `class DrivingStyleSampler`：采样驾驶风格参数。
    - `def __init__`：初始化采样器，设置设备。
    - `def sample_mixed_uniform`：采样混合均匀分布参数。
    - `def sample_driving_style`：采样驾驶风格（油门和转向系数）。
    - `def sample_driving_Cacc`：采样加速度控制系数。
    - `def sample_driving_Cvel`：采样速度控制系数。
    - `def get_distribution_info`：获取分布信息。
  - `class DiscreteActionSpace`：定义离散动作空间。
    - `def __init__`：初始化离散动作空间，设置jerk值。
    - `def get_action`：根据动作索引获取对应的jerk动作。
    - `def get_all_actions`：获取所有可能的动作。
  - `class KinematicBicycleModel`：批量化自行车动力学模型，支持jerk控制。
    - `def __init__`：初始化动力学模型，设置车辆参数和控制系数。
    - `def step`：执行一步动力学更新，基于jerk控制更新车辆状态。
    - `def reset_control_state`：重置控制状态（加速度和转向角）。
    - `def calculate_steering_angle`：根据横向加速度和速度计算转向角。
    - `def get_discrete_action_space`：获取离散动作空间。

- **collision.py**
  - `class CollisionChecker`：批量化碰撞检测。
    - `def __init__`：初始化碰撞检测器，设置参数。
    - `def check`：执行碰撞检测，返回碰撞信息。
    - `def _broad_phase_vectorized`：宽阶段碰撞检测，使用空间哈希快速筛选。
    - `def _narrow_phase_vectorized`：窄阶段精确碰撞检测。
    - `def _continuous_collision_detection`：连续碰撞检测。

- **road.py**
  - `class RoadNetwork`：道路网络加载与定位。
    - `def __init__`：初始化道路网络，加载地图数据。
    - `def calculate_frenet_coordinates`：计算Frenet坐标系坐标。
    - `def find_nearest_quad`：找到最近的四边形。
    - `def get_lane_waypoints`：获取车道航点。

- **reward.py**
  - `class RewardCalculator`：奖励计算。
    - `def __init__`：初始化奖励计算器，设置奖励参数。
    - `def calculate`：计算奖励，基于多个奖励组件。
    - `def _calculate_progress_reward`：计算进度奖励。
    - `def _calculate_safety_reward`：计算安全奖励。
    - `def _calculate_efficiency_reward`：计算效率奖励。

- **offroad.py**
  - `class OffroadChecker`：离路检测。
    - `def __init__`：初始化离路检测器。
    - `def check_offroad`：检查智能体是否离路。

- **observation.py**
  - `class ObservationGenerator`：观测生成器，负责为批次中的所有Agent生成局部观测。
    - `def __init__`：初始化观测生成器，设置观测参数（邻居数量、车道数量、边界数量、视野范围等）。
    - `def generate`：为所有环境中的所有agent生成一批观测，返回展平后的观测向量张量 (B, M, feature_dim)。
    - `def _get_nearest_neighbors`：为每个agent找到最近的K个邻居，完全向量化版本，使用torch.cdist计算距离。
    - `def _get_nearby_global_points`：为所有agent从全局点集中找到k个最近的点（车道航点、边界点）。
    - `def _world_to_ego_centric`：将世界坐标系下的状态转换为以每个agent为中心的局部坐标系。

- **world_init.py**
  - `class WorldInitializer`：世界初始化与智能体布置。
    - `def __init__`：初始化世界初始化器。
    - `def initialize_world`：初始化世界，布置智能体。
    - `def _place_agents`：放置智能体到无碰撞位置。

### policy/
- **networks.py**
  - `class TeraFlowPolicy`：统一Actor-Critic策略网络。
    - `def __init__`：初始化策略网络，设置网络架构。
    - `def forward`：前向传播，生成动作、对数概率和价值。
    - `def evaluate_actions`：评估给定动作的对数概率和价值。
    - `def _encode_observations`：编码观测输入并融合特征。

- **encoders.py**
  - `class PermutationInvariantEncoder`：集合输入编码器。
    - `def __init__`：初始化编码器，设置网络参数。
    - `def forward`：前向传播，处理集合输入。

### training/
- **ppo.py**
  - `class AdvantageFilteringPPO`：带优势筛选的PPO算法。
    - `def __init__`：初始化PPO算法，设置超参数。
    - `def update`：执行PPO更新步骤。
    - `def compute_loss`：计算PPO损失。

- **advantage_filter.py**
  - `class AdvantageFilter`：GAE优势筛选。
    - `def __init__`：初始化优势筛选器。
    - `def filter`：筛选高优势的经验。
    - `def compute_gae`：计算GAE优势。

- **train_loop.py**
  - `def main`：训练主循环，协调整个训练过程。

- **test_ppo.py**
  - `class SharedNetwork`：PPO用共享网络。
    - `def __init__`：初始化共享网络。
    - `def forward`：前向传播。
  - `def compute_gae`：计算广义优势估计。
  - `def ppo`：PPO主训练循环。
  - `class RewardNormalizer`：奖励归一化。
    - `def __init__`：初始化归一化器。
    - `def normalize`：归一化奖励。
  - `class CosineAnnealingLR`：余弦退火学习率调度

    - `def __init__`：初始化学习率调度器。
        - `def step`：更新学习率。
        - `def get_lr`：获取当前学习率。

- **training_monitor.py**
  - `class TrainingMonitor`：训练过程监控与可视化。
    - `def __init__`：初始化监控器。
    - `def log_episode`：记录回合信息。
    - `def plot_training_curves`：绘制训练曲线。
    - `def save_checkpoint`：保存检查点。

- **gym_visualization.py**
  - `class SimpleEnvGym`：兼容Gym的环境。
    - `def __init__`：初始化Gym环境。
    - `def reset`：重置环境。
    - `def step`：执行环境步进。
    - `def render`：渲染环境。
    - `def close`：关闭环境。
  - `def load_config`：加载配置文件。
  - `def create_gym_env`：创建Gym环境。

### utils/
- **spatial_hash.py**
  - `class SpatialHash`：GPU空间哈希。
    - `def __init__`：初始化空间哈希，设置网格参数。
    - `def query_points`：查询点附近的实体。
    - `def query_boxes`：查询边界框内的实体。
    - `def update`：更新哈希表。

- **spatial_hash_visualization.py**
  - `def load_map_data`：加载地图数据。
  - `def visualize_spatial_hash_grid`：可视化空间哈希网格。
  - `def test_spatial_hash_query`：测试空间哈希查询。
  - `def visualize_occupied_cells`：可视化被占用的网格单元。
  - `def main`：主函数，执行可视化测试。

- **keyboard_vehicle_control.py**
  - `class KeyboardVehicleController`：键盘控制器。
    - `def __init__`：初始化键盘控制器。
    - `def start_control`：开始控制循环。
    - `def handle_key_press`：处理按键事件。
    - `def update_vehicle_state`：更新车辆状态。
  - `def main`：主函数，启动键盘控制。

### maps/
- **preprocessor.py**
  - `class NumpyEncoder`：JSON编码器，处理numpy数组。
  - `def get_quad_center_3d`：计算四边形中心点。
  - `def is_point_in_quad_2d`：判断点是否在四边形内。
  - `class SpatialGrid3D`：3D空间网格。
  - `class PointIndexGrid`：点索引网格。
  - `def run_dijkstra`：运行Dijkstra算法。
  - `def generate_oob_points`：生成越界点。
  - `def build_connectivity_graph_3d`：构建3D连通性图。
  - `def find_junction_nodes`：找到交叉口节点。
  - `def calculate_all_routing_data_parallel`：并行计算所有路由数据。
  - `def enhance_w_lanes_from_carla`：从Carla增强车道航点。
  - `def find_closest_poi_downstream`：找到下游最近的兴趣点。
  - `def preprocess_map`：预处理地图。

- **carla_connect_final.py**
  - 多个函数处理原始carla地图数据。
  - `def main`：主函数，处理Carla地图。

- **dijkstra.py**
  - `def build_graph`：构建图结构。
  - `def find_shortest_path`：找到最短路径。
  - `def plan_path_and_visualize`：规划路径并可视化。
  - `def main`：主函数，测试路径规划。

- **visualize_*.py**
  - 各类地图/航点/路口/四边形可视化函数。
  - 每个文件都有对应的`main`函数作为入口。

### 根目录脚本
- **single_agent_ppo.py**
  - `class SimpleRolloutBuffer`：简单经验缓冲区。
    - `def __init__`：初始化缓冲区。
    - `def add`：添加经验。
    - `def get_batches`：获取批次数据。
    - `def reset`：重置缓冲区。
  - `class SingleAgentTrainer`：单智能体PPO训练器。
    - `def __init__`：初始化训练器。
    - `def _create_simulator`：创建模拟器。
    - `def _create_policy`：创建策略网络。
    - `def collect_experience`：收集经验。
    - `def update_policy`：更新策略。
    - `def train`：执行训练。
    - `def save_model`：保存模型。
    - `def load_model`：加载模型。
    - `def plot_training_curves`：绘制训练曲线。
  - `def main`：主函数，启动单智能体训练。

- **single_simulator.py**
  - `class SingleSimulatorVisualizer`：单环境可视化器。
    - `def __init__`：初始化可视化器。
    - `def run_animation`：运行动画。
    - `def _init_animation`：初始化动画。
    - `def _update_frame`：更新帧。
  - `def main`：主函数，启动单环境可视化。

- **test_and_visualize_simulator.py**
  - `class SimulatorVisualizer`：批量环境可视化器。
    - `def __init__`：初始化可视化器。
    - `def run_animation`：运行动画。
    - `def _init_animation`：初始化动画。
    - `def _update_frame`：更新帧。
    - `def _draw_static_map`：绘制静态地图。
    - `def _add_agent_artists`：添加智能体图形。
    - `def _update_collision_debug_viz`：更新碰撞调试可视化。
    - `def _update_highlighted_info`：更新高亮信息。
  - `def main`：主函数，启动批量环境可视化。

---

## Observation系统与坐标转换详解

### 1. Observation系统概述

`ObservationGenerator`是TeraFlow中的核心观测生成模块，负责为批次中的所有智能体生成局部观测。该模块通过完全向量化的操作，一次性为所有环境中的所有智能体高效地计算观测，避免了Python循环，最大限度地利用GPU并行能力。

#### 1.1 观测组成
观测向量包含以下四个主要部分：
- **局部状态 (local_state)**：智能体自身的状态信息 (x, y, yaw, speed, length, width, active)
- **邻居特征 (neighbors)**：附近智能体的相对位置、速度、活跃状态 (dx, dy, vx, vy, length, width, active)
- **车道航点 (w_lanes)**：附近车道航点的相对位置 (dx, dy)
- **边界点 (w_boundaries)**：附近道路边界点的相对位置 (dx, dy)

#### 1.2 配置参数
```yaml
observation:
  num_neighbors: 1          # 邻居数量
  num_w_lanes: 25          # 车道航点数量
  num_w_boundaries: 26     # 边界点数量
  horizon: 100.0           # 视野范围
  local_state_dim: 7       # 局部状态维度 (x, y, yaw, speed, length, width, active)
  neighbor_feature_dim: 7  # 邻居特征维度
  waypoint_feature_dim: 2  # 航点特征维度
```

### 2. 坐标转换系统

#### 2.1 坐标系定义
- **世界坐标系 (World Coordinate System)**：全局坐标系，所有智能体和地图元素的绝对位置
- **Ego坐标系 (Ego-centric Coordinate System)**：以每个智能体为中心的局部坐标系
  - X轴：智能体前进方向
  - Y轴：智能体右侧方向（在matplotlib可视化中向下为正）

#### 2.2 坐标转换过程

##### 2.2.1 旋转矩阵构建
```python
# 为行向量构建正确的旋转矩阵
# 要将世界坐标点按 -ego_yaw 旋转，对于行向量 v' = v @ R
cos_yaw, sin_yaw = torch.cos(ego_yaw), torch.sin(ego_yaw)
rot_matrix = torch.stack([
    torch.stack([cos_yaw, -sin_yaw], dim=-1), 
    torch.stack([sin_yaw, cos_yaw], dim=-1)
], dim=-2)
```

##### 2.2.2 批量坐标转换
```python
def batch_rotate(points_world, ego_pos, rot_matrix):
    # points_world: (B, M, N, 2), ego_pos: (B, M, 2), rot_matrix: (B, M, 2, 2)
    rel_pos = points_world - ego_pos.unsqueeze(2)
    B, M, N, D = rel_pos.shape
    return torch.bmm(rel_pos.view(B*M, N, D), rot_matrix.view(B*M, D, D)).view(B, M, N, D)
```

#### 2.3 邻居状态转换
对于每个邻居智能体，需要转换：
1. **位置转换**：相对位置从世界坐标转换到ego坐标
2. **速度转换**：世界坐标系下的速度分量转换到ego坐标系
3. **活跃状态**：直接保持布尔值

```python
# 位置转换
rel_pos_neighbors = neighbor_states[..., :2] - ego_pos.unsqueeze(2)
local_pos_neighbors = torch.bmm(rel_pos_neighbors.view(B*M, K, 2), rot_matrix.view(B*M, 2, 2))

# 速度转换
vx_world = neighbor_speed * torch.cos(neighbor_yaw)
vy_world = neighbor_speed * torch.sin(neighbor_yaw)
v_world = torch.stack([vx_world, vy_world], dim=-1)
v_local = torch.bmm(v_world.view(B*M, K, 2), rot_matrix.view(B*M, 2, 2))
```

### 3. 邻居检测算法

#### 3.1 距离计算
使用`torch.cdist`计算所有智能体之间的配对距离：
```python
dist_sq = torch.cdist(query_pos, query_pos, p=2).pow(2)  # (B, M, M)
```

#### 3.2 邻居筛选
应用多层筛选条件：
1. **自身排除**：智能体不能是其自身的邻居
2. **活跃状态**：不活跃的智能体不能作为邻居
3. **视野范围**：距离超过视野范围的邻居不考虑

```python
# 创建筛选掩码
self_mask = torch.eye(max_agents, device=device, dtype=torch.bool).expand(batch_size, -1, -1)
inactive_mask = (agents_state[..., 6] < 0.5).unsqueeze(1).expand(-1, max_agents, -1)
dist_sq[self_mask | inactive_mask] = float('inf')
dist_sq[dist_sq > self.horizon**2] = float('inf')
```

#### 3.3 最近邻居选择
使用`torch.topk`选择最近的K个邻居：
```python
_, topk_indices = torch.topk(dist_sq, k=self.num_neighbors, dim=-1, largest=False)
```

### 4. 地图特征提取

#### 4.1 车道航点和边界点选择
为每个智能体从全局点集中找到最近的k个点：
```python
def _get_nearby_global_points(self, agents_state, source_points, num_points):
    query_pos = agents_state[..., :2].view(-1, 2)  # (B*M, 2)
    dist_sq = torch.cdist(query_pos, source_points, p=2).pow(2)
    _, topk_indices = torch.topk(dist_sq, k=num_points, dim=1, largest=False)
    selected_points = source_points[topk_indices]  # (B*M, k, 2)
    return selected_points.view(batch_size, max_agents, num_points, 2)
```

### 5. 观测向量构建

#### 5.1 特征拼接
将所有观测特征展平并拼接成最终的观测向量：
```python
observation = torch.cat([
    local_state,                                    # (B, M, local_state_dim)
    neighbors_local.flatten(start_dim=2),          # (B, M, K * neighbor_feature_dim)
    w_lanes_local.flatten(start_dim=2),            # (B, M, N_lanes * waypoint_feature_dim)
    w_boundaries_local.flatten(start_dim=2)        # (B, M, N_bounds * waypoint_feature_dim)
], dim=2)
```

#### 5.2 维度验证
确保生成的观测向量维度与配置一致：
```python
expected_dim = (local_state_dim + 
                num_neighbors * neighbor_feature_dim +
                num_w_lanes * waypoint_feature_dim +
                num_w_boundaries * waypoint_feature_dim)
assert observation.shape[2] == expected_dim
```

### 6. 可视化与调试

#### 6.1 双视图可视化
系统提供两种可视化视图：
1. **全局视图**：显示所有智能体在世界坐标系中的位置和观测
2. **Ego视图**：以ego为中心的局部坐标系视图，显示相对位置和方向

#### 6.2 坐标轴显示
在ego视图中，明确显示ego坐标系：
- **X轴（红色）**：ego前进方向
- **Y轴（绿色）**：ego右侧方向（向下为正）
- **原点（黄色X）**：ego当前位置

#### 6.3 观测点可视化
- **车道航点**：绿色"+"标记
- **边界点**：青色圆圈标记
- **邻居车辆**：红色矩形，显示相对位置和速度信息

### 7. 性能优化

#### 7.1 向量化操作
- 使用`torch.cdist`进行批量距离计算
- 使用`torch.topk`进行批量最近邻选择
- 使用`torch.bmm`进行批量矩阵乘法


### 8. 使用示例

```python
# 初始化观测生成器
obs_generator = ObservationGenerator(road_network, config, device)

# 生成观测
agents_state = torch.randn(B, M, 7, device=device)  # 智能体状态
observation = obs_generator.generate(agents_state)   # 生成观测

# 观测形状: (B, M, feature_dim)
print(f"Observation shape: {observation.shape}")
```



## 坐标转换系统详解

### 1. ObservationGenerator中的坐标转换过程

#### 1.1 世界坐标系到ego坐标系的转换（`_world_to_ego_centric`方法）

**核心转换公式**：

```
ego_coord = R * (world_coord - ego_world_pos)
ego_vel = R * world_vel
```

其中R是旋转矩阵：

```
R = [[cos(ego_yaw), -sin(ego_yaw)],
     [sin(ego_yaw),  cos(ego_yaw)]]
```

**转换步骤**：

1. **提取ego状态**：`ego_pos = ego_states[..., :2]`，`ego_yaw = ego_states[..., 2]`
2. **构建旋转矩阵**：基于ego的yaw角度构建2x2旋转矩阵
3. **批量旋转函数**：`batch_rotate(points_world, ego_pos, rot_matrix)`
4. **转换各种特征**：
   - 车道航点：`w_lanes_local = batch_rotate(w_lanes_world, ego_pos, rot_matrix)`
   - 边界点：`w_boundaries_local = batch_rotate(w_boundaries_world, ego_pos, rot_matrix)`
   - 邻居车辆位置和速度

#### 1.2 邻居车辆状态转换

```python
# 位置转换
rel_pos_neighbors = neighbor_states[..., :2] - ego_pos.unsqueeze(2)
local_pos_neighbors = torch.bmm(rel_pos_neighbors.view(B*M, K_neighbors, 2), 
                               rot_matrix.view(B*M, 2, 2)).view(B, M, K_neighbors, 2)

# 速度转换
vx_world = neighbor_speed * torch.cos(neighbor_yaw)
vy_world = neighbor_speed * torch.sin(neighbor_yaw)
v_world = torch.stack([vx_world, vy_world], dim=-1)
v_local = torch.bmm(v_world.view(B*M, K_neighbors, 2), 
                   rot_matrix.view(B*M, 2, 2)).view(B, M, K_neighbors, 2)
```

### 2. Figure显示坐标系的转换流程

#### 2.1 车辆位置转换到figure2坐标系

```python
# 1. 计算相对位置（世界坐标系）
rel_x = x - ego_x
rel_y = y - ego_y

# 2. 旋转到ego坐标系
cos_yaw = np.cos(-ego_yaw)  # 注意是负角度
sin_yaw = np.sin(-ego_yaw)
ego_coord_x = rel_x * cos_yaw - rel_y * sin_yaw
ego_coord_y = rel_x * sin_yaw + rel_y * cos_yaw

# 3. 翻转Y轴以匹配matplotlib坐标系（向下为正）
ego_coord_y = -ego_coord_y
```

#### 2.2 车辆朝向转换到figure2坐标系

```python
# 1. 计算相对朝向
ego_rel_yaw = agent_yaw - ego_yaw

# 2. 由于figure2中Y轴向下为正，需要翻转yaw角度
ego_rel_yaw = -ego_rel_yaw
```

#### 2.3 速度向量转换到figure2坐标系

```python
# 使用已经翻转过的ego_rel_yaw计算速度向量
neighbor_vx = agent_speed * np.cos(ego_rel_yaw)
neighbor_vy = agent_speed * np.sin(ego_rel_yaw)
# 注意：ego_rel_yaw已经考虑了Y轴翻转，所以不需要再次翻转Y分量
```

### 3. 坐标转换总结

#### 3.1 转换层次结构

1. **世界坐标系** → **ego坐标系**（ObservationGenerator内部）
   - 位置：`ego_coord = R * (world_coord - ego_pos)`
   - 速度：`ego_vel = R * world_vel`
   - 朝向：`ego_rel_yaw = agent_yaw - ego_yaw`

2. **ego坐标系** → **figure2显示坐标系**
   - 位置：`display_y = -ego_coord_y`（Y轴翻转）
   - 朝向：`display_yaw = -ego_rel_yaw`（角度翻转）
   - 速度：使用翻转后的朝向计算，Y分量不再翻转

#### 3.2 关键特点

1. **ego坐标系**：以ego车辆为中心，ego前进方向为X轴正方向
2. **figure2坐标系**：Y轴向下为正，与标准数学坐标系相反
3. **速度向量**：始终指向车辆自己的前进方向
4. **一致性**：所有转换都考虑了Y轴翻转，确保显示正确

#### 3.3 转换矩阵

**世界到ego**：

```
R = [[cos(ego_yaw), -sin(ego_yaw)],
     [sin(ego_yaw),  cos(ego_yaw)]]
```

**ego到figure2**：

```
T = [[1,  0],
     [0, -1]]  # Y轴翻转
```

#### 3.4 应用场景

- **观测生成**：将全局状态转换为每个agent的局部观测
- **可视化**：在figure2中正确显示车辆位置、朝向和速度向量
- **策略网络**：为神经网络提供标准化的局部坐标系输入
- **碰撞检测**：在统一的坐标系中进行精确的碰撞计算

这套坐标转换系统确保了在整个TeraFlow系统中，所有组件都能在正确的坐标系下工作，为强化学习算法提供准确的观测信息。

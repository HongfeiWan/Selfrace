# world_init.py 模块说明

本文档概述 `simulator/world_init.py` 中 `WorldInitializer` 的核心职责、主要成员与方法，以及初始化流程与可视化入口，便于阅读、调试与上层调用。格式参考 `road.md` 与 `offroad.md`。

## 目录
- 类与初始化
- 车辆状态生成
- 初始化流程（批量、拒绝采样）
- 依赖与配置读取
- 使用建议

---

## 类与初始化

### WorldInitializer(road_network: RoadNetwork, offroad_checker: OffroadChecker, collision_checker: CollisionChecker, config: Dict)
- 作用：根据地图与配置，批量生成无碰撞、在道路上的初始交通流（B 个环境，每环境 M 辆车）。
- 输入：
  - `road_network`：`RoadNetwork` 实例（提供 `quad_centerlines` 等几何信息）。
  - `offroad_checker`：用于“离路”检测（几何点在多边形内判定）。
  - `collision_checker`：用于车辆间连续碰撞检测。
  - `config`：整体配置（读取自 `configs/default_config.json`）。
- 成员（关键）：
  - `self.max_agents: int`：每环境最大保留槽位（来自 `simulator.M`）。
  - `self.num_agents_per_env: int`：需要放置的车辆数（来自 `simulator.num_npc_vehicles`）。
  - `self.speed_range: Tuple[float, float]`：速度采样范围（来自 `simulator.dynamics.[min_velocity,max_velocity]`）。
  - `self.local_state_dim: int`：车辆状态维度，默认 7；格式 `[x,y,yaw,v,length,width,active]`。

---

## 车辆状态生成

### _generate_states_on_quads(quad_indices: Tensor, lengths: Tensor, widths: Tensor) -> Tensor
- 作用：在指定 `quad` 上生成车辆状态。
- 输入：
  - `quad_indices (N,)`：目标四边形索引。
  - `lengths/widths (N,)`：由采样器生成的车辆几何尺寸。
- 输出：`(N,7)` 的状态张量。
- 规则：
  - 位置：在 `quad_centerlines` 两端点间按随机系数 `t∈[0,1]` 线性插值。
  - 朝向：使用中心线方向向量 `atan2(dy,dx)`。
  - 速度：在 `speed_range` 内均匀采样。
  - 尺寸：使用传入的 `lengths/widths`。
  - `active=1.0` 标记为激活。

---

## 初始化流程（批量、拒绝采样）

### initialize_world(num_envs: int) -> Tuple[Tensor, Tensor, Tensor]
- 作用：生成一批新的、无碰撞的世界状态。
- 返回：
  - `agents_state: (B, max_agents, 7)` 初始车辆状态（含未用槽位，`active=0`）。
  - `ego_agents_idx: (B,)` 主车索引（默认 0）。
  - `agents_start_quad_ids: (B, max_agents)` 记录每辆启用车辆的起始 `quad_id`。
- 流程要点：
  1) 批量采样所有环境、所有槽位的候选：
     - 随机四边形索引；
     - 由 `VehicleParameterSampler` 采样每辆车的 `length/width`；
     - 调用 `_generate_states_on_quads` 生成 `(B*M,7)` 候选，重塑为 `(B,M,7)`。
  2) 离路检测：
     - 提取 `[x,y,yaw,length,width]`，展平为 `(B*M,5)`，调用 `OffroadChecker.check_on_road`。
  3) 碰撞检测：
     - 调用 `CollisionChecker.check(agents_state, agents_state)`，得到 `(B,max_agents)` 布尔矩阵。
  4) 无效筛除：
     - `invalid = (~is_on_road) | collisions`，无效的置 `active=0`。
  5) 记录有效 `quad_id`：
     - 对通过筛选的条目写入 `agents_start_quad_ids`。

说明：若需严格遵循“顺序拒绝采样”（逐车放置、确保不与已放置车辆碰撞），可采用按槽位的跨环境并行方案：每个槽位为所有环境生成 K 个候选，批量离路+碰撞筛选，选择首个有效候选落位；该实现可在需要时替换当前批量方案。

---

## 依赖与配置读取

- 依赖：
  - `RoadNetwork`：提供 `quad_centerlines`、`quads_vertices` 等几何张量。
  - `OffroadChecker`：批量点-多边形测试（使用 `SpatialHash` 作为索引）。
  - `CollisionChecker`：宽阶段（哈希候选）+窄阶段（连续碰撞）完全并行化计算。
- 配置关键项（`configs/default_config.json`）：
  - `device`：张量设备（cpu/cuda）。
  - `simulator.M`：每环境最大槽位。
  - `simulator.num_npc_vehicles`：目标车辆数。
  - `simulator.dynamics.min_velocity/max_velocity`：速度采样范围。
  - `simulator.observation.local_state_dim`：状态维度（默认 7）。

---

## 使用建议

- 性能：
  - 大规模初始化时，优先采用“按槽位跨环境并行+分块离路/碰撞”的顺序拒绝采样版本，降低峰值内存；
  - 合理设置候选批量 K 与分块大小，避免过大张量（可与 `SpatialHash.query_points` 的分块策略一致）。
- 稳定性：
  - `VehicleParameterSampler` 的取值范围需与车辆几何与道路尺度匹配；
  - 在小地图放置大量车辆时应限制最大重试次数或加入启发式采样提升一次命中率。



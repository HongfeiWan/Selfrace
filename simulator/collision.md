# collision.py 变量与结构说明

本文档概述 `simulator/collision.py` 中 `CollisionChecker` 的核心数据结构、输入输出与交互测试入口，便于阅读、调试与上层调用。

## 目录
- 类与初始化
- 核心流程与方法
- 配置读取
- 交互测试入口（__main__）
- 使用建议

---

## 类与初始化

### CollisionChecker(config: Dict, spatial_hash: SpatialHash)
- 作用：基于 GPU 的批量化车辆-车辆/车辆-障碍物连续碰撞检测器，宽阶段使用共享的 `SpatialHash` 做候选过滤，窄阶段做精确检测。
- 输入：
  - `config`：全局配置（通常来自 `configs/default_config.json`）。
  - `spatial_hash`：`utils/spatial_hash.py::SpatialHash` 实例（共享索引，供宽阶段检索相邻候选）。
- 初始化成员：
  - `self.device: torch.device`
  - `self.spatial_hash: SpatialHash`
  - `self.cell_size: float`（来自 `simulator.hash.cell_size`）
  - `self.max_neighbors: int`（优先 `simulator.observation.num_neighbors`，回退到 `simulator.max_neighbors` 或顶层）

---

## 核心流程与方法

### check(self, states_t0: Tensor, states_t1: Tensor, static_obstacles: Optional[Tensor] = None, debug: bool = False, debug_env_idx: int = 0) -> Tensor
- 作用：批量进行完整的连续碰撞检测，返回形状 (B, M) 的布尔张量。
- 输入：
  - `states_t0/states_t1`：形状 (B, M, 7) 的车辆状态，格式 `[x, y, yaw, v, length, width, active]`。
  - `static_obstacles`：可选，形状 (O, 4, 2) 的静态AABB多边形顶点或等价表示（在当前实现中作为“静止车辆”处理）。
  - `debug`：是否返回调试信息（宽阶段候选等）。
- 逻辑：
  1. 计算 t0/t1 的车辆包络四顶点：`_get_world_vertices`。
  2. 宽阶段：调用 `self.spatial_hash.query_dynamic_pairs`（B×M并行）筛选候选对。
  3. 窄阶段：`_narrow_phase_vectorized`，采用 Gather-Compute-Scatter 模式；
     - Gather：一次性收集所有需要检测的 (j, k) 状态与顶点；
     - Compute：两次单向检测 `_check_one_way_collision` 组成双向检测；
     - Scatter：将对撞结果写回 (B,M) 阵列。
  4. 合并静态障碍检测（如提供）。
  5. 与 `active_mask` 相与，得到最终结果。

### _broad_phase_vectorized(self, active_mask, verts_t0, verts_t1, debug=False, debug_env_idx=0)
- 作用：使用 `SpatialHash` 对 (B,M) 的车辆批量计算动态候选对，返回候选索引与可选调试信息。

### _narrow_phase_vectorized(self, candidate_pairs, active_mask, states_t0, states_t1, verts_t0, verts_t1)
- 作用：对候选对进行连续碰撞检测；内部执行两次 `_check_one_way_collision` 完成双向判定。

### _check_one_way_collision(self, ref_states_t0, ref_states_t1, mov_verts_t0, mov_verts_t1)
- 作用：单向检测，将移动物体顶点轨迹转换到参考物体局部坐标系中，用 Slab 法与参考方 AABB 相交测试。

### _get_world_vertices(self, states)
- 作用：从车辆状态 `[x,y,yaw,length,width]` 推导四顶点世界坐标（并行计算）。

### _line_segment_aabb_intersection(self, p0, p1, aabb_min, aabb_max)
- 作用：Slab 方法批量检测线段与轴对齐包围盒的相交（向量化实现）。

### _check_static_collisions(self, states_t0, states_t1, static_obstacles)
- 作用：将静态障碍视为特殊“参考物体”，批量执行单向检测并合并结果。

---

## 配置读取

- 设备：`device = torch.device(config['device'] or 'cuda')`
- 栅格：`cell_size = config['simulator']['hash']['cell_size']`（存在则取值，否则使用安全默认）
- 邻居数：优先 `config['simulator']['observation']['num_neighbors']`，回退到 `config['simulator'].get('max_neighbors')` 或顶层 `config.get('max_neighbors')`。

---

## 交互测试入口（__main__）

- 功能（当前示例代码可能被注释关闭，以下为预期与示例实现要点）：
  1. 读取 `configs/default_config.json`。
  2. 构建 `SpatialHash` 并初始化 `CollisionChecker`。
  3. 加载 `RoadNetwork`，绘制路面四边形轮廓作为底图。
  4. 随机在路面上放置 100 辆静态车辆（速度 0、朝向取自对应 quad 的方向）。
  5. 初始化 1 辆可控车辆，使用键盘交互：
     - 上/下：沿朝向前进/后退；
     - 左/右：按固定步长旋转；
     - +/-：缩放视野；
     - 仅显示可控车周围的局部视野。
  6. 每次交互后运行碰撞检测：若发生碰撞，可控车边框渲染为红色，否则为蓝色。

---

## 使用建议

- 性能：
  - 宽阶段选择合适的 `cell_size` 与 `max_neighbors`，提升大规模仿真效率；
  - 采用批量/向量化的输入，减少 Python 循环与 Host-Device 往返。
- 数据一致性：
  - `states_t0` 与 `states_t1` 应保证形状一致且在同一 `device`；
  - 车辆尺寸应与仿真参数一致（`vehicle_length/width`）。
- 可视化：
  - 建议仅绘制必要图层（底图与车辆），并维持小范围视野以稳定帧率。



# offroad.py 变量与结构说明

本文档概述 `simulator/offroad.py` 中 `OffroadChecker` 的数据结构、核心方法与交互测试入口，便于阅读、调试与上层调用。

## 目录
- 类与初始化
- 函数与实现原理
  - `_create_local_bbox_points`
  - `_precompute_convex_quad_edges`
  - `_get_discretized_bounding_boxes`
  - `_batch_point_in_polygon_test`
  - `check_on_road`
- 可视化/交互测试入口（__main__）
- 使用建议

---

## 类与初始化

### OffroadChecker(map_data: RoadNetwork, spatial_hash: SpatialHash, points_per_vehicle_edge: int = 3)
- 作用：构建一个基于 GPU 的批量化离路检测器，复用共享的 `SpatialHash` 做候选检索，并对候选多边形执行向量化半平面测试。
- 输入：
  - `map_data`：`RoadNetwork` 实例，提供路面四边形顶点 `quads_vertices`（形状 Q×4×2）。
  - `spatial_hash`：`utils/spatial_hash.py::SpatialHash` 实例，用于快速查询点所处网格的候选多边形索引。
  - `points_per_vehicle_edge`：沿车体包络每条边离散的采样点数（≥2）。
- 初始化成员：
  - `self.device: torch.device`
  - `self.points_per_vehicle_edge: int`
  - `self.road_polygons: FloatTensor (Q,4,2)` 路面四边形顶点（在 device 上）
  - `self.spatial_hash: SpatialHash` 共享空间哈希索引
  - `self.local_bbox_points: FloatTensor (K,2)` 单位车辆边界离散点模板
  - `self.poly_verts: FloatTensor (Q,4,2)` 预计算顶点
  - `self.poly_edges: FloatTensor (Q,4,2)` 预计算顺序边向量
  - `self.poly_sign: FloatTensor (Q,)` 绕序符号（+1/-1）

---

## 函数与实现原理

### `_create_local_bbox_points(self) -> Tensor`
- 作用：在车辆局部坐标系构建单位边界框的离散点模板，范围 [-0.5, 0.5]×[-0.5, 0.5]。
- 输出：形状 (K,2) 的点集（沿四条边各取 n 个点，首尾不重复拼接，另加中心点）。
- 用途：后续按车辆长宽缩放、按朝向旋转、按位置平移，得到世界坐标下的车辆边界采样点，用于离路判定。

### `_precompute_convex_quad_edges(self)`
- 作用：为每个四边形预计算半平面测试所需的数据，避免重复计算。
- 输出缓存：
  - `self.poly_verts: (Q,4,2)` 顶点坐标；
  - `self.poly_edges: (Q,4,2)` 顺序边向量（v[i+1]-v[i]）；
  - `self.poly_sign: (Q,)` 顶点绕序符号（鞋带公式 2×有向面积≥0 记为 +1：逆时针，<0 记为 -1：顺时针）。
- 判定规则统一为 `(poly_sign * cross(e, p - v)) >= -eps` 四条边均成立则点在多边形内。

### `_get_discretized_bounding_boxes(self, states: Tensor) -> Tensor`
- 作用：将单位边界框点模板按车辆状态批量变换到世界坐标系。
- 输入：形状 (N,5) 的车辆状态 `[x, y, heading, length, width]`。
- 输出：形状 (N,K,2) 的世界坐标采样点。
- 原理：先按长宽缩放，再按朝向旋转，最后平移到 `(x,y)`。

### `_batch_point_in_polygon_test(self, points: Tensor) -> Tensor`
- 作用：对一批世界坐标点执行“点在道路多边形内”的批量检测。
- 步骤：
  1. 使用 `self.spatial_hash.query_points(points)` 获取候选 `(point_idx, poly_idx)`；
  2. 取对应 `poly_verts/poly_edges/poly_sign` 做四边半平面测试；
  3. 用 `scatter_add_` 将每个点的多个候选结果归并（任一点命中任意一个多边形即视为在路上）。
- 返回：形状 (M,) 的 `bool` 向量，表示每个点是否在道路上。

### `check_on_road(self, states: Tensor) -> Tensor`
- 作用：批量判断车辆是否在道路上。
- 流程：
  1. `world_points = _get_discretized_bounding_boxes(states)`；
  2. 扁平化为 (N×K,2) 做 `_batch_point_in_polygon_test`；
  3. 重塑回 (N,K) 并沿点维度取 `all`，得到 (N,) 的布尔结果。

---

## 可视化/交互测试入口（__main__）

- 读取 `configs/default_config.json` 的 `map_path` 并指定 `town2.json`，加载 `RoadNetwork`。
- 随机选择一个 quad 并在其中随机点初始化车辆位置；朝向随机。
- 计算车辆附近的 quads 并绘制：最近一个 quad 显示中心线与方向箭头（方向取自 `quad_directions`）。
- 交互：按左右方向键以 5° 步长调整车辆朝向，实时重绘车体与重新计算在/离路状态与 Frenet 坐标（`d, theta_f`）。

---

## 使用建议

- 空间查询：建议预先用 `SpatialHash.build_static_index` 为路面多边形构建静态索引，再多次查询点集合。
- Frenet 坐标：可调用 `utils/geometry_utils.calculate_frenet_coordinates(device, quad_directions, quad_centerlines, positions, headings, k, spatial_hash)` 获得 `d, theta_f`，与离路检测互补使用。

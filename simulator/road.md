# road.py 变量与结构说明

本文档概述 `simulator/road.py` 中 `RoadNetwork` 的核心数据结构、输入输出与可视化测试入口，便于阅读、调试与上层调用。

## 目录
- 类与初始化
- 几何与元数据
- 全局航点
- 可视化测试入口（__main__）
- 使用建议

---

## 类与初始化

### RoadNetwork(map_path: str, device: torch.device)
- 作用：从导出的地图 JSON 加载并构建道路网络张量，用于后续快速几何/邻域计算与可视化。
- 输入：
  - `map_path`：地图 JSON 路径（如 `../maps/town2.json`）
  - `device`：张量所处设备（`cpu` 或 `cuda`）
- 处理流程：
  1) `_load_map_data` 读取 JSON。
  2) `_store_metadata` 统一构建几何与元数据张量（顶点、中心线、方向、ID、全局航点等）。

---

## 几何与元数据

以下成员均在 `__init__` 中预初始化为空，再由 `_store_metadata` 填充。

- `quads_vertices: FloatTensor (Q, 4, 2)`
  - 每个 quad 的四个顶点（按 [TL, BL, BR, TR] 存放，单位：米）。
- `quad_centerlines: FloatTensor (Q, 2, 2)`
  - 每个 quad 的中心线段两端点坐标（[后中心点, 前中心点]）。
- `left_boundaries: FloatTensor (Q, 2, 2)`
  - 每个 quad 的左边界线段（按道路方向由 `direction_angle` 推断）。
- `right_boundaries: FloatTensor (Q, 2, 2)`
  - 每个 quad 的右边界线段（按道路方向由 `direction_angle` 推断）。
- `quad_directions: FloatTensor (Q, 2)`
  - 每个 quad 的单位方向向量，由 JSON 的 `direction_angle` 直接转换 `(cos, sin)`。
- `quad_ids: LongTensor (Q,)`
  - 每个 quad 的 `poly_id`。
- `lane_ids: IntTensor (Q,)`
  - 每个 quad 所属的 `lane_id`（重排/分组后写入 JSON 再读取）。
- `road_ids: IntTensor (Q,)`
  - 每个 quad 所属的 `road_id`。
- `w_lane_ids: List[List[int]]`
  - quad 关联的 `w_lane` ID 列表（如有）。
- `w_boundary_ids: List[List[int]]`
  - quad 关联的越界边界点 ID 列表（如有）。

构建规则（来自 `_store_metadata`）：
- 顶点顺序按导出 JSON：[TL, TR, BR, BL]，内部存为 `[TL, BL, BR, TR]` 便于两侧边直接读取。
- 中心线：后中心点 = (TL+TR)/2，前中心点 = (BL+BR)/2。
- 方向：由 `direction_angle` 直接生成 `(cos, sin)`，不再从几何估计。
- 左/右边界：以左法向与两条侧边中点向量的点积大小决定左右分配，保证与 `direction_angle` 一致。

---

## 全局航点

- `global_w_lane: FloatTensor (N_wl, 2)`
  - 来自 JSON 中 `w_lanes.center` 的全局车道航点（仅 XY）。
- `global_w_boundary: FloatTensor (N_oob, 2)`
  - 来自 JSON 中 `oob_points` 的全局越界边界点（仅 XY）。

---

## 可视化测试入口（__main__）

- 运行 `python simulator/road.py`：
  - 默认读取 `../maps/town2.json`，绘制：
    - 底层：仅 quad 轮廓（无填充）。
    - 复选框图层：
      - `quad_centerlines`（橙色）
      - `left_boundaries`（蓝色）
      - `right_boundaries`（绿色）
      - `quad_directions`（红色箭头）
      - `global_w_lane`（红色点）
      - `global_w_boundary`（紫色点）

---

## 使用建议

- 最近车道查询：`utils/geometry_utils.find_nearest_lanes(device, quad_centerlines, points, k, spatial_hash)`。
  - 推荐用 `utils/spatial_hash.py::SpatialHash` 对 `quad_centerlines` 的 AABB 构建静态索引，并传入 `spatial_hash` 加速候选过滤。
- Frenet 坐标计算：`utils/geometry_utils.calculate_frenet_coordinates(device, quad_directions, quad_centerlines, vehicle_positions, vehicle_headings, k, spatial_hash)`，返回 `(d, theta_f)`。

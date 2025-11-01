# preprocessor.py 函数说明与实现原理

本文档介绍 `maps/preprocessor.py` 中主要函数/流程的作用、输入、输出与核心实现思想，便于阅读、调试与扩展。

## 目录
- 初始化与配置
- 基础几何/采样相关
  - `sample_line_points`
  - `sample_circle_points`
  - `sample_arc_points`
  - `generate_line_points_gpu / generate_circle_points_gpu / generate_arc_points_gpu`
  - `compute_directions_gpu`
  - `compute_trapezoid_vertices_gpu`
  - `create_seamless_trapezoid`
- 去重与网格索引
  - `check_duplicate_geometry`
  - `SpatialGrid3D`（引自 utils/geometry_utils）
- OOB 生成
  - `generate_oob_points_cpu`
  - `generate_oob_points_gpu`
  - `generate_oob_points`
- 路网方向校正
  - `adjust_road_directions_gpu`
  - 反转后的 `poly_id` 顺序校正
- 道路重叠分组与 ID/Lane 重排
  - `group_roads_by_overlap`
  - `reassign_road_lane_ids`
- 车道级统计量
  - `sample_w_lanes`
  - `compute_lane_s_for_quads`
  - `compute_lane_curvature`
- 导出与可视化调用

---

## 初始化与配置
从 `configs/default_config.json` 读取预处理参数，如：
- `tolerance`（几何/命中容差）
- `sample_distance`（采样间距）
- `rectangle_length / rectangle_width`（梯形尺寸）
- `oob_nudge_distance`（OOB 外推距离）
- `cell_size`（空间网格大小）
- `w_lane_sample_distance`（W_lane 采样间距）

同时检测 GPU 可用性，设置 `DEVICE = 'cuda' / 'cpu'`。

---

## 基础几何/采样相关

### sample_line_points(start, end, sample_distance)
- 作用：按等距在直线段上生成点序列（含首尾）。
- 输入：起点/终点二维坐标、采样间距。
- 输出：点列表、总长度。
- 原理：线性插值，末点对齐。

### sample_circle_points(center, radius, sample_distance)
- 作用：在圆上等距生成点（含起点闭合）。
- 输出：点列表、周长。

### sample_arc_points(center, radius, start_angle, end_angle, sample_distance)
- 作用：在圆弧上等距生成点（含首尾）。角度为弧度。
- 输出：点列表、弧长。

### generate_*_points_gpu(...)
- 作用：对应线/圆/圆弧，使用 GPU 批量生成采样点。
- 输出：`numpy.ndarray` 点数组，总长度/弧长。
- 原理：向量化计算+`torch.linspace` 参数化。

### compute_directions_gpu(points)
- 作用：对点列计算切向方向角（弧度）。末点复用前一段方向。
- 输入：(N,2) 点数组。
- 输出：(N,) 方向角数组。

### compute_trapezoid_vertices_gpu(points, angles, rectangle_length, rectangle_width)
- 作用：按中心点+方向角批量构造四边形（梯形）顶点。
- 输出：(N,4,2) 顶点数组。
- 原理：切向/法向单位向量线性组合得到四个角点，批量张量运算提速。

### create_seamless_trapezoid(...)
- 作用：逐点生成与前一梯形无缝衔接的四边形，避免缝隙。
- 输入：中心、切向/法向角、上/下底长度与高度、前一梯形顶点（可空）。
- 输出：四个顶点（顺序：上左、上右、下右、下左）。
- 原理：首个梯形直接按参数构造，其后共享前一梯形下底作为当前上底实现无缝。

---

## 去重与网格索引

### check_duplicate_geometry(new_item, existing_items)
- 作用：读取 DXF 时剔除几何重复（直线/圆/弧对比关键要素距离或角度差）。

### SpatialGrid3D（utils/geometry_utils）
- 作用：将四边形投影到二维网格加速邻域查询。
- 用途：OOB 命中/相交候选检索/方向校正候选检索等。

---

## OOB 生成

### generate_oob_points_cpu(quads, grid, quads_by_id, nudge_distance)
### generate_oob_points_gpu(quads, grid, quads_by_id, device, batch_size, nudge_distance)
### generate_oob_points(...)
- 作用：在四边形边外侧按法向外推小距离采样“越界点”，并剔除仍在任意四边形内部的点。
- 输入：四边形列表、空间网格、ID 索引、外推距离，GPU 版本支持批量验证点在多边形内。
- 输出：OOB 点列表（x,y,z）。
- 原理：中点+边法线方向微探测确定外/内侧，候选点统一用点-四边形内测函数筛除。

---

## 路网方向校正

### adjust_road_directions_gpu(lines_data, arcs_data, tolerance, device)
- 作用：基于四边形连通关系统一道路行驶方向（按 road 级别），同时维护几何与四边形角度。
- 输入：`lines_data`/`arcs_data`（几何），容差，设备。
- 输出：就地修改 `lines_data`/`arcs_data` 的起止定义；就地调整相关道路所有四边形的 `direction_angle`。
- 核心原理：
  1. 以“道路终点”落入相邻道路的哪一段（start/end/middle）来推断两路期望相对方向（同向0/反向1）。
  2. 构图后以并查集带“奇偶约束”求全局一致方向。
  3. 对需要翻转的道路：
     - 交换几何的 `start/end`（线）或反转 `direction`（弧）；
     - 将该路所有四边形 `direction_angle += π`；
     - 反转该路四边形列表顺序，并把该路原来的 `poly_id` 集合（升序）重新依次赋给新序列，保证每条路最终 `poly_id` 随新方向从 start→end 递增。

> 注：该函数在“生成四边形”和“OOB 生成”之后调用，确保使用最新的四边形拓扑关系。

---

## 道路重叠分组与 ID/Lane 重排

### group_roads_by_overlap(polygons_data, threshold, device)
- 作用：按道路间四边形重叠比例（相交矩阵）构建道路组。
- 输入：所有四边形、相交阈值（默认0.8）、设备。
- 输出：道路组列表（每组为若干旧 `road_id`）。
- 原理：
  1. 先按道路聚合四边形，AABB 粗筛；
  2. 使用 `quads_intersection_matrix_gpu` 计算两路相交矩阵；
  3. 相交比例取 A 命中 B 或 B 命中 A 的较大者，与阈值比较；
  4. 构建邻接后以 BFS 得到连通分量即道路组。

### reassign_road_lane_ids(polygons_data, groups)
- 作用：对每个分组分配统一的 `road_id`，并在组内按旧 `road_id` 升序分配 `lane_id = 1..k`。
- 输出：就地更新 `polygons_data` 的 `road_id` 和新增 `lane_id`。

---

## 车道级统计量

### sample_w_lanes(polygons_data, sample_distance_m)
- 作用：对每条车道 `(road_id, lane_id)` 按中心曲线等距采样 W_lane 点（包含首尾）。
- 输入：`polygons_data`、采样间距（默认 40m）。
- 输出：W_lane 点列表：`w_lane_id / road_id / lane_id / center / direction_angle / width / poly_id`。
- 原理：
  1. 按车道聚合四边形并按 `poly_id` 递增（即行驶方向）；
  2. 对中心点累计弧长，并在 0..L 按固定步长选择最近的离散点；
  3. 宽度由上/下底长度平均得到。

### compute_lane_s_for_quads(polygons_data)
- 作用：为每条车道按 `poly_id` 顺序计算每个 quad 的弧长参数 `s`（Frenet s）。
- 输出：就地写入 `q['s']`。
- 原理：`s[i] = s[i-1] + ||center_i - center_{i-1}||`，`s[0] = 0`。

### compute_lane_curvature(polygons_data)
- 作用：为每条车道计算每个 quad 的曲率 `curvature`（带符号，单位 1/m）。
- 输出：就地写入 `q['curvature']`。
- 原理：三点法曲率：
  - 中间点 i 用三点 `(i-1, i, i+1)`；
  - 两端点用单边三点：起点 `(0,1,2)`，终点 `(n-3,n-2,n-1)`；
  - 曲率由三角形有向面积与边长乘积得到，符号由叉积符号决定。

---

## 导出与可视化调用

脚本末尾：
1) 生成并显示可视化（matplotlib），支持多图层（OOB、方向箭头、按道路着色、W_lane、s 标签、曲率）。
2) 导出 JSON：
   - `quads`: `poly_id / road_id / lane_id / center / vertices / direction_angle / s / curvature`
   - `oob_points`: `x/y/z`
   - `w_lanes`: `w_lane_id / road_id / lane_id / poly_id / center / direction_angle / width`
   - `geometry`: `lines/circles/arcs` 原始几何（便于还原/调试）

---

## 典型流程（高层概览）
1. 读取 DXF → 直线/圆/弧去重入库
2. 采样点与方向角（CPU/GPU）→ 构造无缝梯形 `polygons_data`
3. 生成 OOB 点（CPU 或 GPU 内测）
4. 路网方向校正（road 级）：并查集+奇偶 → 需要翻转的路：几何翻转+角度翻转+`poly_id` 顺序重排
5. 采样 W_lane（按车道）
6. 可视化预览（可选）
7. 基于四边形重叠进行道路分组 → `road_id` 归并、`lane_id` 递增
8. 计算每条车道的 `s` 与 `curvature`
9. 导出 JSON（含所有字段）并支持从 JSON 直接可视化

---

## 注意事项与扩展建议
- 若需要在“分组重排后”再次保证 poly_id 随新方向递增，可调用 `ensure_lane_poly_id_monotonic`（若在项目中保留）。
- 相交矩阵 GPU 版本依赖 `utils/geometry_utils.quads_intersection_matrix_gpu`，随数据量可适当调整批大小避免显存峰值。
- 曲率与 s 的准确性取决于中心线平滑程度与采样间距，可按需要调整 `sample_distance` 与平滑策略。



# goal.py 变量与结构说明

本文档概述 `simulator/goal.py` 中 `PathPlanner` 的核心职责、关键数据结构与主要接口，便于阅读、调试与上层调用。格式参考 `simulator/road.md`。

## 目录
- 类与初始化
- 与 RoadNetwork 的集成
- 预计算（路径矩阵与 w_lane_ids 相关索引）
- 路径规划接口 `path_plan`
- 收集路径的 w_lane_ids `collect_path_w_lane_ids`
- 可视化入口（__main__）
- 使用建议

---

## 类与初始化

### PathPlanner(device: str = 'cuda', road_network: RoadNetwork)
- 作用：在 GPU 上完成批量最短路径规划与 w_lane_ids 序列收集。
- 依赖：必须传入已经构建完成的 `RoadNetwork` 实例。
- 配置来源（`configs/default_config.json`）
  - `simulator.observation.INVALID_MARKER`（统一无效标记源值）
  - `simulator.observation.num_navigation_chains` → `max_path_length`（内部也作为 `w_lane_ids_length` 使用）
  - `simulator.observation.num_start&end_chains` → `max_chain_len`（起/终段长度）

初始化时会：
- 从 `RoadNetwork` 复用图结构与索引：
  - `lane_keys`、`n_lanes`、`lane_to_idx`、`lane_start_end`
  - `start_positions`、`end_positions`、`adjacency_matrix`、`edge_weights`
  - `poly_id_to_lane_idx`、`poly_id_lookup`
- 预计算 w_lane_ids 相关结构（详见下文）。
- 预计算所有起点→终点的路径矩阵（Floyd-Warshall + 全量路径矩阵）。

---

## 与 RoadNetwork 的集成
- `PathPlanner` 不再读取地图 JSON，全部几何/索引数据来自 `RoadNetwork`：
  - `quads_by_id`、`quads_vertices`、`quad_centerlines` 等仅在可视化中使用
  - 规划与采样阶段使用 `poly_id_lookup`、`adjacency_matrix`、`edge_weights`

---

## 预计算（路径矩阵与 w_lane_ids 相关索引）

### 路径矩阵（Floyd-Warshall + 全量矩阵构建）
- 使用 `edge_weights` 在 GPU 上执行 Floyd-Warshall：
  - 产出 `dist_matrix` 与 `prev_matrix`
- 基于 `prev_matrix` 向量化构建三维 `path_matrix (n, n, max_len)`：
  - `path_matrix[i, j, :]` 表示从 `lane_i` 到 `lane_j` 的路径（以 lane 索引存储），不足部分用 `INVALID_PATH_MARKER` 填充

### w_lane_ids 预计算
- `w_lane_features (N_w, 3)`：每个 w_lane 的 `(x, y, direction_angle)` 特征
- `lane_w_lane_ids (n_lanes, W_max, 3)` 与 `lane_w_lane_ids_count (n_lanes,)`：
  - 将同一 `(road_id, lane_id)` 下的所有 w_lane（按 quad 的 `s` 排序）拼接为定长表
- `poly` 的 next/prev 序列（CSR）：
  - `poly_next_seq_flat_idx/offsets/lengths`
  - `poly_prev_seq_flat_idx/offsets/lengths`
  - 供起点/终点段的链式取样使用（长度 `max_chain_len`）

---

## 路径规划接口 `path_plan`

签名：
- `path_plan(start_poly_ids: LongTensor[B, M], end_poly_ids: LongTensor[B, M]) -> LongTensor[B, M, max_path_len]`

流程：
- 使用 `poly_id_lookup` 将 `(start_poly_id, end_poly_id)` 转为 `(start_lane_idx, end_lane_idx)`
- 批量从 `path_matrix` 取出路径并返回；无效位置为 `INVALID_PATH_MARKER`

---

## 收集路径的 w_lane_ids `collect_path_w_lane_ids`

签名：
- `collect_path_w_lane_ids(paths: LongTensor[B, M, max_path_len], start_poly_ids, end_poly_ids) -> FloatTensor[B, M, w_lane_ids_length, 3]`

核心要点：
- 起点段/终点段：
  - 基于 `poly` 的 CSR 序列，分别按 `direction='next'|'prev'` 取 `max_chain_len` 个 w_lane 特征，不采样
- 中间段：
  - 对路径中间所有 lane（剔除首尾）批量查表 `lane_w_lane_ids` 写入，完全向量化；
  - 通过 per-row cumsum 计算写入 offset，使用 `nonzero + repeat_interleave + gather` 避免显存爆炸
- 采样策略：
  - 仅对中段做长度约束，最大为 `w_lane_ids_length - 2*max_chain_len`
  - 若超长：按 `interval = (count // max_middle_len) + 1` 进行“间隔采样”（等间隔选点），完全向量化
- 返回：三段直接 `cat`，左对齐，无效用 `INVALID_w_lane_id_MARKER` 填充

---

## 可视化入口（__main__）

- 后台生成一批随机 `(B, M)` 起终点，并基于 `path_plan + collect_path_w_lane_ids` 取得路径与 w_lane_ids。
- 使用 Pygame + OpenGL 渲染：
  - 背景地图：`quads_vertices`
  - 当前路径：以紫色线段绘制每个 w_lane 的方向箭头
  - 起点/终点：直接使用当前路径的第一个/最后一个有效 w_lane 点绘制彩色点（绿色/红色）
  - 交互：空格切换下一个 `(B, M)` 路径，ESC 退出

---

## 使用建议
- 若仅需路径长度或可达性，直接读取 `dist_matrix`/`prev_matrix` 更轻量。
- 大批量收集 w_lane_ids 时，可按 `tile` 分块（代码已内置 `tile=2048`），结合显存情况调整。
- 如需改变起/终段长度，改 `num_start&end_chains`（配置）；改变总长度改 `num_navigation_chains`。
- 可视化如需更贴近车道中心，可改用 `quad_centerlines` 的中点绘制起终点参考。

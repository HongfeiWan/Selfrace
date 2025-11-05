# collect_path_w_lane_ids 优化方案可行性分析

## 当前实现分析

### 内存占用
- **输出tensor**: `(B, M, w_lane_ids_length, 3)`
  - `w_lane_ids_length = max_path_length // 2 = 128`
  - 对于 B=4800, M=150: `4800 × 150 × 128 × 3 × 4 bytes ≈ 1.1 GB`

- **分块处理中的临时缓冲** (tile=4096):
  - `compact_buf_all`: `(4096, P*W, 3)` 其中 P=256, W≈50 → `4096 × 12800 × 12 ≈ 630 MB`
  - `buffer3`: `(4096, 3K, 3)` 其中 K=128 → `4096 × 384 × 12 ≈ 19 MB`
  - **总计每个tile**: ≈ 650 MB

- **分块原因**: 避免一次性处理 B×M=720,000 条路径导致显存爆炸

### 当前流程
1. 分块处理 (tile=4096)
2. 对每个 tile:
   - 获取起点链 (K个点)
   - 获取终点链 (K个点)
   - 收集中间段所有 w_lane_ids → 压紧到 compact_buf_all
   - 采样中间段到 K 个点
   - 拼接三段 (3K) → buffer3
   - 最终采样到 K 个点

## TODO 优化方案分析

### 方案概述
1. **不分块**: 直接处理整个 B×M
2. **三个空tensor**: `start_segment`, `middle_segment`, `end_segment`，每个 `(B, M, max_path_len, 3)`
3. **直接写入**: 通过 `road_id, lane_id → w_lane_ids` 映射直接获取中间段
4. **智能采样**: 如果总长度超过 max_path_len，降采样

### 可行性评估

#### ✅ **优点**

1. **消除中间缓冲**
   - 不再需要 `compact_buf_all` (P*W 维度)
   - 不再需要 `buffer3` (3K 维度)
   - 直接写入最终tensor，减少内存碎片

2. **利用已有数据结构**
   - ✅ `lane_w_lane_ids`: `(n_lanes, max_w_lanes_per_lane, 3)` 已存在
   - ✅ `lane_w_lane_ids_count`: `(n_lanes,)` 已存在
   - ✅ `paths` 包含 lane_idx，可直接索引

3. **向量化更彻底**
   - 可以一次性处理所有 B×M 条路径
   - 减少循环和条件判断

#### ⚠️ **挑战和风险**

1. **内存占用大幅增加**
   ```
   三个tensor内存 = 3 × (B × M × max_path_len × 3 × 4 bytes)
                  = 3 × (4800 × 150 × 256 × 12)
                  ≈ 6.3 GB
   ```
   - **当前方案**: 输出 1.1 GB + 临时 650 MB (分块) ≈ 1.75 GB 峰值
   - **优化方案**: 6.3 GB (三个tensor同时存在)
   - **增加**: 3.6倍内存占用

2. **输出维度变化**
   - 当前: `(B, M, 128, 3)` - w_lane_ids_length
   - TODO: `(B, M, 256, 3)` - max_path_length
   - **注意**: 如果下游代码期望 128 维度，需要修改

3. **实现复杂度**
   - 需要精确计算每段的写入位置和长度
   - 拼接三段需要处理边界情况
   - 采样逻辑需要处理三段长度不一致的情况

#### 🔧 **技术实现要点**

1. **起点链和终点链**
   ```python
   # 当前已有函数，可直接使用
   start_chain = _get_w_lane_chain_w_lane_ids_from_poly_vectorized(
       start_poly_ids, direction='next', max_chain_len=max_path_len)
   # 返回 (B*M, max_path_len, 3)，但实际有效长度可能 < max_path_len
   ```

2. **中间段获取**
   ```python
   # paths: (B, M, max_path_len) - lane_idx
   # 需要提取中间段 (pos > 0 and pos < path_len-1)
   middle_lane_indices = paths[:, :, 1:-1]  # 去掉首尾
   
   # 直接索引 lane_w_lane_ids
   # 但每个 lane 的 w_lane_ids 数量不同，需要：
   # - 收集所有 w_lane_ids
   # - 按路径顺序拼接
   # - 处理变长序列
   ```

3. **拼接和采样**
   ```python
   # 总长度 = start_len + middle_len + end_len
   # 如果 > max_path_len，需要降采样
   # 策略：按比例采样三段，保持相对比例
   ```

## 建议的优化方案（改进版）

### 方案A: 完全按照TODO实现（高风险）
- **适用**: 显存充足 (≥10GB)
- **优点**: 代码最简洁，无中间缓冲
- **缺点**: 内存占用大，需要下游代码适配

### 方案B: 混合方案（推荐）
1. **保持分块**: 但减小 tile 大小，或使用更智能的分块策略
2. **消除 compact_buf_all**: 
   - 直接写入 middle_segment，不先压紧到 P*W
   - 使用 `lane_w_lane_ids` 直接写入，按需采样
3. **优化 buffer3**: 
   - 只在需要时创建（当总长度>K时）
   - 或使用原地操作

### 方案C: 渐进式优化
1. **第一步**: 消除 compact_buf_all，直接写入三段
2. **第二步**: 优化采样逻辑，减少中间缓冲
3. **第三步**: 如果显存允许，考虑完全不分块

## 具体实现建议

### 关键数据结构
```python
# 需要构建的映射（如果还没有）
lane_idx_to_w_lane_ids: (n_lanes, max_w_lanes_per_lane, 3)  # 已有
lane_idx_to_w_lane_ids_count: (n_lanes,)  # 已有

# 新方案需要：
# 1. 起点段: (B, M, max_path_len, 3) - 直接从poly获取
# 2. 中间段: (B, M, max_path_len, 3) - 从lane_idx获取
# 3. 终点段: (B, M, max_path_len, 3) - 直接从poly获取
```

### 实现难点
1. **中间段收集**: 
   - 每个路径的中间段包含多个 lanes
   - 每个 lane 的 w_lane_ids 数量不同
   - 需要向量化拼接变长序列

2. **三段拼接**:
   - 需要知道每段的实际长度
   - 处理边界情况（路径长度为0, 1, 2的情况）
   - 采样时保持三段的相对比例

## 结论

### ✅ 可行性：**中等**

**优点**:
- 理论上可行，消除中间缓冲
- 代码逻辑更清晰
- 可以利用已有数据结构

**风险**:
- 内存占用增加 3.6倍
- 实现复杂度较高（特别是中间段的向量化拼接）
- 需要下游代码适配（输出维度变化）

### 推荐方案
**采用方案B（混合方案）**:
1. 保留分块，但优化中间缓冲的使用
2. 直接写入三段，避免 compact_buf_all
3. 只在必要时创建 buffer3
4. 逐步优化，降低风险

这样可以：
- 减少内存占用（相比完全不分块）
- 消除最大的中间缓冲 compact_buf_all
- 保持代码的可维护性
- 降低实现风险


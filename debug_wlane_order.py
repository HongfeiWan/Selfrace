"""
调试 w_lane 序列的存储顺序
检查 next_w_lane_id 和 prev_w_lane_id 的实际顺序
"""
import json
import numpy as np

data = json.load(open(r'maps/town2.json', 'r'))
quads = {q['poly_id']: q for q in data['quads']}
w_lanes = {w['w_lane_id']: w for w in data.get('w_lanes', [])}

print("=" * 80)
print("检查 next_w_lane_id 和 prev_w_lane_id 的存储顺序")
print("=" * 80)

# 找一个有足够多 next/prev w_lane 的 quad 作为示例
example_quad = None
for poly_id, quad in quads.items():
    next_list = quad.get('next_w_lane_id', [])
    prev_list = quad.get('prev_w_lane_id', [])
    if len(next_list) >= 5 and len(prev_list) >= 5:
        example_quad = quad
        break

if not example_quad:
    print("找不到合适的示例 quad")
    exit(1)

poly_id = example_quad['poly_id']
center = example_quad['center']
s_value = example_quad.get('s', 0)

print(f"\n示例 quad:")
print(f"  poly_id: {poly_id}")
print(f"  center: {center}")
print(f"  s: {s_value:.2f}")
print(f"  road_id: {example_quad.get('road_id')}, lane_id: {example_quad.get('lane_id')}")

# 检查 next_w_lane_id
next_list = example_quad.get('next_w_lane_id', [])
print(f"\n{'='*80}")
print(f"next_w_lane_id ({len(next_list)} 个):")
print(f"  完整列表: {next_list}")
print(f"\n  详细信息（前10个）:")

for i, w_id in enumerate(next_list[:10]):
    if w_id in w_lanes:
        w = w_lanes[w_id]
        w_center = w['center']
        w_poly_id = w.get('poly_id')
        # 获取对应 quad 的 s 值
        w_s = quads[w_poly_id].get('s', 0) if w_poly_id in quads else 0
        print(f"    [{i}] w_lane_id={w_id}, poly_id={w_poly_id}, center={w_center}, s={w_s:.2f}")

# 检查 prev_w_lane_id
prev_list = example_quad.get('prev_w_lane_id', [])
print(f"\n{'='*80}")
print(f"prev_w_lane_id ({len(prev_list)} 个):")
print(f"  完整列表: {prev_list}")
print(f"\n  详细信息（前10个）:")

for i, w_id in enumerate(prev_list[:10]):
    if w_id in w_lanes:
        w = w_lanes[w_id]
        w_center = w['center']
        w_poly_id = w.get('poly_id')
        # 获取对应 quad 的 s 值
        w_s = quads[w_poly_id].get('s', 0) if w_poly_id in quads else 0
        print(f"    [{i}] w_lane_id={w_id}, poly_id={w_poly_id}, center={w_center}, s={w_s:.2f}")

# 分析顺序
print(f"\n{'='*80}")
print("顺序分析:")

# next 的 s 值变化
next_s_values = []
for w_id in next_list[:10]:
    if w_id in w_lanes:
        w_poly_id = w_lanes[w_id].get('poly_id')
        if w_poly_id in quads:
            next_s_values.append(quads[w_poly_id].get('s', 0))

if len(next_s_values) > 1:
    is_increasing = all(next_s_values[i] <= next_s_values[i+1] for i in range(len(next_s_values)-1))
    is_decreasing = all(next_s_values[i] >= next_s_values[i+1] for i in range(len(next_s_values)-1))
    print(f"\nnext_w_lane_id 的 s 值: {[f'{s:.2f}' for s in next_s_values]}")
    if is_increasing:
        print("  ✓ s 值递增 → next_w_lane_id 从近到远（离当前quad越来越远）")
    elif is_decreasing:
        print("  ✗ s 值递减 → next_w_lane_id 从远到近（异常！）")
    else:
        print("  ? s 值无序")

# prev 的 s 值变化
prev_s_values = []
for w_id in prev_list[:10]:
    if w_id in w_lanes:
        w_poly_id = w_lanes[w_id].get('poly_id')
        if w_poly_id in quads:
            prev_s_values.append(quads[w_poly_id].get('s', 0))

if len(prev_s_values) > 1:
    is_increasing = all(prev_s_values[i] <= prev_s_values[i+1] for i in range(len(prev_s_values)-1))
    is_decreasing = all(prev_s_values[i] >= prev_s_values[i+1] for i in range(len(prev_s_values)-1))
    print(f"\nprev_w_lane_id 的 s 值: {[f'{s:.2f}' for s in prev_s_values]}")
    if is_increasing:
        print("  ✓ s 值递增 → prev_w_lane_id 从远到近（从起点到当前quad）")
    elif is_decreasing:
        print("  ✗ s 值递减 → prev_w_lane_id 从近到远（从当前quad到起点）")
    else:
        print("  ? s 值无序")

print(f"\n{'='*80}\n")


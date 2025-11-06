import json
import math

data = json.load(open(r'maps/town2.json', 'r'))
quads_1_2 = [q for q in data['quads'] if q.get('road_id')==1 and q.get('lane_id')==2]
quads_1_2.sort(key=lambda q: q['poly_id'])

print(f"=== (road_id=1, lane_id=2) 详细分析 ===")
print(f"Quad 数量: {len(quads_1_2)}")
print(f"poly_id 范围: {quads_1_2[0]['poly_id']} - {quads_1_2[-1]['poly_id']}")

# 计算总长度（按 center 距离）
centers = [tuple(q['center']) for q in quads_1_2]
cum = [0.0]
for i in range(1, len(centers)):
    prev = centers[i-1]
    cur = centers[i]
    cum.append(cum[-1] + math.hypot(cur[0]-prev[0], cur[1]-prev[1]))
total_len = cum[-1]

print(f"\n沿 quad centers 的总长度: {total_len:.2f} 米")
print(f"平均 quad 间距: {total_len/(len(quads_1_2)-1) if len(quads_1_2)>1 else 0:.2f} 米")

# 按 40 米采样应该有多少点
W_LANE_SAMPLE_DISTANCE = 40.0
expected_points = int(total_len / W_LANE_SAMPLE_DISTANCE) + 2  # +首尾
print(f"\n按 {W_LANE_SAMPLE_DISTANCE}m 采样，预期 w_lane 数: {expected_points}")

# 检查前几个 quad 的具体位置
print(f"\n前 5 个 quad 的位置:")
for i in range(min(5, len(quads_1_2))):
    q = quads_1_2[i]
    print(f"  poly_id={q['poly_id']}, center={q['center'][:2]}, s={cum[i]:.2f}m")

# 检查是否所有 quads 都有 next_w_lane_id 和 prev_w_lane_id
has_next = sum(1 for q in quads_1_2 if q.get('next_w_lane_id'))
has_prev = sum(1 for q in quads_1_2 if q.get('prev_w_lane_id'))
print(f"\n有 next_w_lane_id 的 quad 数: {has_next}")
print(f"有 prev_w_lane_id 的 quad 数: {has_prev}")

# 检查示例 quad 的 w_lane_ids 字段
print(f"\n示例 quad (poly_id=75) 的 w_lane 字段:")
q75 = next((q for q in quads_1_2 if q['poly_id']==75), None)
if q75:
    print(f"  next_w_lane_id: {q75.get('next_w_lane_id', 'N/A')}")
    print(f"  prev_w_lane_id: {q75.get('prev_w_lane_id', 'N/A')}")
    print(f"  w_lane_ids: {q75.get('w_lane_ids', 'N/A')}")


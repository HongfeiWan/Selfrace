"""
验证 w_lanes 与 quads 的 (road_id, lane_id) 一致性
用于检查预处理器修复是否有效
"""
import json
from collections import defaultdict

print("=" * 80)
print("验证 w_lanes 与 quads 的一致性")
print("=" * 80)

# 加载数据
data = json.load(open(r'maps/town2.json', 'r'))
quads_list = data['quads']
wlanes_list = data.get('w_lanes', [])

print(f"\n基本统计:")
print(f"  总 quads: {len(quads_list)}")
print(f"  总 w_lanes: {len(wlanes_list)}")

# 1. 检查 w_lane 与其 poly_id 对应的 quad 的一致性
print(f"\n{'='*80}")
print("检查 1: w_lane 的 (road_id, lane_id) 是否与其 poly_id 对应的 quad 一致")
print(f"{'='*80}")

poly_to_quad = {q['poly_id']: q for q in quads_list}
inconsistent_count = 0
inconsistent_examples = []

for w in wlanes_list:
    poly_id = w.get('poly_id')
    if poly_id in poly_to_quad:
        quad = poly_to_quad[poly_id]
        w_rid, w_lid = w.get('road_id'), w.get('lane_id')
        q_rid, q_lid = quad.get('road_id'), quad.get('lane_id')
        
        if w_rid != q_rid or w_lid != q_lid:
            inconsistent_count += 1
            if len(inconsistent_examples) < 5:
                inconsistent_examples.append({
                    'w_lane_id': w.get('w_lane_id'),
                    'poly_id': poly_id,
                    'w_lane': (w_rid, w_lid),
                    'quad': (q_rid, q_lid)
                })

if inconsistent_count == 0:
    print("✓ 所有 w_lane 的 (road_id, lane_id) 与其对应 quad 一致")
else:
    print(f"✗ 发现 {inconsistent_count} 个不一致的 w_lane")
    print(f"  示例（前5个）:")
    for ex in inconsistent_examples:
        print(f"    w_lane_id={ex['w_lane_id']}, poly_id={ex['poly_id']}")
        print(f"      w_lane: (road_id={ex['w_lane'][0]}, lane_id={ex['w_lane'][1]})")
        print(f"      quad:   (road_id={ex['quad'][0]}, lane_id={ex['quad'][1]})")

# 2. 统计 quads 和 w_lanes 中的 (road_id, lane_id) 组合
print(f"\n{'='*80}")
print("检查 2: 每个 (road_id, lane_id) 组合是否都有 w_lane")
print(f"{'='*80}")

# quads 中的组合
quad_road_lane = defaultdict(list)
for q in quads_list:
    key = (q.get('road_id'), q.get('lane_id'))
    quad_road_lane[key].append(q['poly_id'])

# w_lanes 中的组合
wlane_road_lane = defaultdict(list)
for w in wlanes_list:
    key = (w.get('road_id'), w.get('lane_id'))
    wlane_road_lane[key].append(w.get('w_lane_id'))

print(f"Quads 中的 (road_id, lane_id) 组合数: {len(quad_road_lane)}")
print(f"W_lanes 中的 (road_id, lane_id) 组合数: {len(wlane_road_lane)}")

# 找出只在 quads 中而不在 w_lanes 中的组合
missing_in_wlanes = set(quad_road_lane.keys()) - set(wlane_road_lane.keys())
if len(missing_in_wlanes) == 0:
    print("✓ 所有 quad 的 (road_id, lane_id) 组合都有对应的 w_lane")
else:
    print(f"✗ 有 {len(missing_in_wlanes)} 个组合只在 quads 中存在，但没有 w_lane")
    print(f"  示例（前10个）:")
    for i, key in enumerate(sorted(missing_in_wlanes)[:10]):
        quad_count = len(quad_road_lane[key])
        poly_ids = quad_road_lane[key][:3]  # 显示前3个poly_id
        print(f"    (road_id={key[0]}, lane_id={key[1]}): {quad_count} 个 quads, poly_ids={poly_ids}...")

# 3. 详细分析每个 (road_id, lane_id) 组合
print(f"\n{'='*80}")
print("检查 3: 详细统计")
print(f"{'='*80}")

print(f"\n按 road_id 分组统计:")
road_stats = defaultdict(lambda: {'total_quads': 0, 'total_wlanes': 0, 'lanes': defaultdict(lambda: {'quads': 0, 'wlanes': 0})})

for q in quads_list:
    rid, lid = q.get('road_id'), q.get('lane_id')
    road_stats[rid]['total_quads'] += 1
    road_stats[rid]['lanes'][lid]['quads'] += 1

for w in wlanes_list:
    rid, lid = w.get('road_id'), w.get('lane_id')
    road_stats[rid]['total_wlanes'] += 1
    road_stats[rid]['lanes'][lid]['wlanes'] += 1

# 显示前10个 road_id 的统计
print(f"Road_id | Total Quads | Total W_lanes | Lanes | 缺失 W_lane 的 Lanes")
print("-" * 80)
for rid in sorted(road_stats.keys())[:10]:
    stats = road_stats[rid]
    total_lanes = len(stats['lanes'])
    missing_lanes = [lid for lid, counts in stats['lanes'].items() if counts['wlanes'] == 0]
    missing_count = len(missing_lanes)
    missing_str = f"{missing_lanes[:3]}..." if len(missing_lanes) > 3 else str(missing_lanes)
    
    print(f"{rid:7d} | {stats['total_quads']:11d} | {stats['total_wlanes']:13d} | "
          f"{total_lanes:5d} | {missing_count} {missing_str if missing_count > 0 else ''}")

# 4. 总结
print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")

total_issues = inconsistent_count + len(missing_in_wlanes)
if total_issues == 0:
    print("✓✓✓ 完美！所有检查都通过")
    print("  - w_lane 与 quad 的 (road_id, lane_id) 完全一致")
    print("  - 所有 (road_id, lane_id) 组合都有对应的 w_lane")
else:
    print(f"✗✗✗ 发现 {total_issues} 个问题:")
    if inconsistent_count > 0:
        print(f"  - {inconsistent_count} 个 w_lane 与其 quad 的 (road_id, lane_id) 不一致")
    if len(missing_in_wlanes) > 0:
        print(f"  - {len(missing_in_wlanes)} 个 (road_id, lane_id) 组合缺少 w_lane")
    print("\n建议: 需要重新运行预处理器 (preprocessor.py) 以应用修复")

print(f"{'='*80}\n")


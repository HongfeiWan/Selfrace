import json

data = json.load(open(r'maps/town2.json', 'r'))
wlanes_list = data.get('w_lanes', [])

print("=== 检查 w_lane_id=4,5,6 的属性 ===")
for wid in [4, 5, 6]:
    w = next((w for w in wlanes_list if w.get('w_lane_id')==wid), None)
    if w:
        print(f"w_lane_id={wid}: road_id={w.get('road_id')}, lane_id={w.get('lane_id')}, poly_id={w.get('poly_id')}")
    else:
        print(f"w_lane_id={wid}: NOT FOUND")

# 检查 road_id=1 的所有 w_lanes
print(f"\n=== road_id=1 的所有 w_lanes ===")
wlanes_rid1 = [w for w in wlanes_list if w.get('road_id')==1]
print(f"总数: {len(wlanes_rid1)}")

# 按 lane_id 分组
from collections import defaultdict
by_lane = defaultdict(list)
for w in wlanes_rid1:
    by_lane[w.get('lane_id')].append(w)

print(f"按 lane_id 分组:")
for lid in sorted(by_lane.keys()):
    print(f"  lane_id={lid}: {len(by_lane[lid])} 个 w_lanes")

# 检查 poly_id=75,76,77 对应的 w_lane
print(f"\n=== 检查 poly_id=75-77 关联的 w_lane ===")
for pid in [75, 76, 77]:
    w = next((w for w in wlanes_list if w.get('poly_id')==pid), None)
    if w:
        print(f"poly_id={pid} → w_lane_id={w.get('w_lane_id')}, road_id={w.get('road_id')}, lane_id={w.get('lane_id')}")
    else:
        print(f"poly_id={pid} → 无对应 w_lane")


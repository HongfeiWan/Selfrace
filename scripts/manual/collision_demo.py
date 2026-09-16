"""Manual diagnostics moved from simulator/collision.py."""

from _bootstrap import add_project_paths

add_project_paths()

import torch

from collision import CollisionChecker
from spatial_hash import SpatialHash

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

config = {
    'device': device,
    'map_extent_m': 10000.0, # 10km map
    'grid_width': 1000,      # Fixed grid resolution
    'grid_height': 1000,
    'vehicle_length': 4.5,
    'max_speed': 30.0,      # Max speed in m/s for testing
    'sim_dt': 0.3,          # Sim time step for testing
}

# 修正：正确实例化 SpatialHash
map_extent = config['map_extent_m']
cell_size = map_extent / config['grid_width']
# 假设地图中心为原点
min_bounds = torch.tensor([-map_extent/2, -map_extent/2], device=device)
max_bounds = torch.tensor([map_extent/2, map_extent/2], device=device)
spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, device)

checker = CollisionChecker(config=config, spatial_hash=spatial_hash)
print("\n--- Testing Optimized Full Batch Collision Checker (Large Map Support) ---")

def run_test(name, s0, s1, expected_collision, static_obs=None):
    result = checker.check(s0, s1, static_obstacles=static_obs)
    # 根据返回值类型进行解包
    if isinstance(result, tuple):
        coll, _ = result
    else:
        coll = result
    test_result = "PASSED" if torch.equal(coll.cpu(), expected_collision.cpu()) else "FAILED"
    print(f"\nTest: {name} -> {test_result}")

    # 对于大型测试，显示碰撞统计信息
    if s0.shape[0] >= 5 or s0.shape[1] >= 10:
        actual_collisions = coll.sum().item()
        expected_collisions = expected_collision.sum().item()
        print(f"  - Actual collisions: {actual_collisions}")
        print(f"  - Expected collisions: {expected_collisions}")

        # 如果失败，显示具体的差异
        if test_result == "FAILED":
            diff = coll.cpu() != expected_collision.cpu()
            diff_count = diff.sum().item()
            print(f"  - Differences: {diff_count} positions")
            if diff_count < 20:  # 只显示少量差异
                diff_indices = torch.where(diff)
                for i in range(min(10, len(diff_indices[0]))):
                    b, m = diff_indices[0][i], diff_indices[1][i]
                    print(f"    Batch {b}, Agent {m}: got {coll[b,m]}, expected {expected_collision[b,m]}")
    else:
        print(f"  - Got:      {coll}")
        print(f"  - Expected: {expected_collision}")

# --- 场景 1: 静态碰撞 ---
s0_1 = torch.zeros(1, 2, 7, device=device)
# 将智能体放置在地图中心附近，以测试坐标转换
map_center = config['map_extent_m'] / 2.0
s0_1[..., :2] = map_center
s0_1[..., 4:7] = torch.tensor([4.0, 2.0, 1.0])
s0_1[0, 1, 0] = map_center + 1.0
s1_1 = s0_1.clone()
expected_1 = torch.tensor([[True, True]], dtype=torch.bool)
run_test("Static Collision", s0_1, s1_1, expected_1)

# --- 场景 2: 大规模测试（150个智能体） ---
B, M = 200, 150  # 使用150个智能体测试

# 使用确定性种子确保测试一致性
torch.manual_seed(42)

# 创建大规模测试场景
s0_large = torch.zeros(B, M, 7, device=device)

# 设置基础位置
base_x = config['map_extent_m'] / 2.0
base_y = config['map_extent_m'] / 2.0

# 为所有智能体分配分散的位置，避免网格冲突
# 使用15x10的网格布局，每个网格间隔50米
for b in range(B):
    for m in range(M):
        # 计算网格位置
        grid_x = (m % 15) * 50  # 15列，每列间隔50米
        grid_y = (m // 15) * 50  # 每行间隔50米

        # 在地图中心附近分配位置
        s0_large[b, m, 0] = base_x - 2000 + grid_x
        s0_large[b, m, 1] = base_y - 2000 + grid_y
        s0_large[b, m, 2] = 0.0  # 偏航角
        s0_large[b, m, 3] = 0.0  # 速度
        s0_large[b, m, 4:6] = torch.tensor([4.5, 2.0], device=device)
        s0_large[b, m, 6] = 1.0  # 激活状态

# 制造一些特定的碰撞（只有少数几对）
# 让智能体0和1在同一网格内碰撞
s0_large[:, 1, :2] = s0_large[:, 0, :2] + torch.tensor([1.0, 0.5], device=device)
# 让智能体10和11在同一网格内碰撞
s0_large[:, 11, :2] = s0_large[:, 10, :2] + torch.tensor([0.5, 1.0], device=device)

s1_large = s0_large.clone()
# 让碰撞的智能体稍微移动
s1_large[:, 0, 0] += 0.5
s1_large[:, 10, 1] += 0.5

expected_large = torch.zeros(B, M, dtype=torch.bool)
expected_large[:, 0] = True   # Agent 0 与 Agent 1 碰撞
expected_large[:, 1] = True   # Agent 1 与 Agent 0 碰撞
expected_large[:, 10] = True  # Agent 10 与 Agent 11 碰撞
expected_large[:, 11] = True  # Agent 11 与 Agent 10 碰撞

import time
print("\n--- Running large scale test (150 agents) ---")
start_time = time.time()
for i in range(5):
    run_test(f"Large Test {i+1} ({B}x{M})", s0_large, s1_large, expected_large)
end_time = time.time()
print(f"Large scale test (5 runs) finished in {(end_time - start_time):.4f} seconds.")

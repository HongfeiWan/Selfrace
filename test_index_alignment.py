"""
测试累积计算和可视化读取的索引对齐
"""
import torch
import numpy as np

def test_cumulative_calculation():
    """测试累积距离计算的索引对齐"""
    print("=" * 60)
    print("测试1: 累积距离计算的索引对齐")
    print("=" * 60)
    
    # 模拟5个路径点（全部有效）
    L = 5
    segment_lengths = torch.tensor([10.0, 15.0, 12.0, 8.0])  # 4个段
    zeros_tail = torch.zeros(1)
    segment_with_tail = torch.cat([segment_lengths, zeros_tail])
    
    print(f"段长度: {segment_lengths.tolist()}")
    print(f"段长度+尾部: {segment_with_tail.tolist()}")
    print(f"  索引对应关系:")
    for i in range(L):
        if i < L - 1:
            print(f"    segment_with_tail[{i}] = 从点{i}到点{i+1}的距离 = {segment_with_tail[i]:.1f}")
        else:
            print(f"    segment_with_tail[{i}] = 0 (最后一个点到终点的距离)")
    
    # 累积计算
    flipped = torch.flip(segment_with_tail, dims=[0])
    print(f"\n翻转后: {flipped.tolist()}")
    
    cumsum = torch.cumsum(flipped, dim=0)
    print(f"累积和: {cumsum.tolist()}")
    
    cumulative = torch.flip(cumsum, dims=[0])
    print(f"再翻转: {cumulative.tolist()}")
    
    print(f"\n累积结果索引对应关系:")
    for i in range(L):
        print(f"    cumulative[{i}] = 从点{i}到终点的距离 = {cumulative[i]:.1f}")
    
    # 验证
    expected = [45.0, 35.0, 20.0, 8.0, 0.0]
    assert torch.allclose(cumulative, torch.tensor(expected)), "累积计算错误！"
    print("\n✓ 累积计算正确：最终点距离为0，起始点距离为总和")
    
    # 测试有无效点的情况
    print("\n" + "-" * 60)
    print("测试2: 有无效点的情况")
    print("-" * 60)
    
    # 假设点2是无效的
    valid_waypoints = torch.tensor([True, True, False, True, True])
    coords = torch.tensor([
        [0.0, 0.0],   # P0
        [10.0, 0.0],  # P1
        [0.0, 0.0],   # P2 (无效，设为0)
        [37.0, 0.0],  # P3
        [45.0, 0.0],  # P4
    ])
    
    segment_vecs = coords[1:] - coords[:-1]
    segment_lengths = torch.norm(segment_vecs, dim=-1)
    segment_valid = valid_waypoints[:-1] & valid_waypoints[1:]
    segment_lengths = torch.where(segment_valid, segment_lengths, torch.zeros_like(segment_lengths))
    
    print(f"有效点掩码: {valid_waypoints.tolist()}")
    print(f"段长度: {segment_lengths.tolist()}")
    print(f"段有效性: {segment_valid.tolist()}")
    
    zeros_tail = torch.zeros(1)
    segment_with_tail = torch.cat([segment_lengths, zeros_tail])
    cumulative = torch.flip(torch.cumsum(torch.flip(segment_with_tail, dims=[0]), dim=0), dims=[0])
    
    print(f"累积结果: {cumulative.tolist()}")
    print(f"\n问题：点2是无效的，但 cumulative[2] 仍然有值！")
    print(f"应该只在有效点处赋值距离")


def test_visualization_indexing():
    """测试可视化读取的索引对齐"""
    print("\n" + "=" * 60)
    print("测试3: 可视化读取的索引对齐")
    print("=" * 60)
    
    # 模拟 w_lanes_local_with_goal_distances
    K = 5  # 5个观察到的 w_lane
    w_lanes_local = torch.tensor([
        [1.0, 2.0],   # w_lane 0: dx=1, dy=2
        [3.0, 4.0],   # w_lane 1: dx=3, dy=4
        [5.0, 6.0],   # w_lane 2: dx=5, dy=6
        [7.0, 8.0],   # w_lane 3: dx=7, dy=8
        [9.0, 10.0],  # w_lane 4: dx=9, dy=10
    ])
    
    # 模拟从 w_lane_goal_distances_full 获取的 Δs 值
    delta_s = torch.tensor([100.0, 80.0, 60.0, 40.0, 20.0])
    
    w_lanes_with_goal = torch.cat([w_lanes_local, delta_s.unsqueeze(-1)], dim=-1)
    
    print(f"w_lanes_local_with_goal_distances 形状: {w_lanes_with_goal.shape}")
    print(f"  每行格式: [dx, dy, Δs]")
    print(f"\n索引对应关系:")
    for i in range(K):
        dx, dy, ds = w_lanes_with_goal[i]
        print(f"    w_lanes_with_goal[{i}] = [dx={dx:.1f}, dy={dy:.1f}, Δs={ds:.1f}]")
    
    # 模拟可视化代码读取
    print(f"\n可视化代码读取:")
    for i in range(K):
        dx = w_lanes_with_goal[i, 0]
        dy = w_lanes_with_goal[i, 1]
        delta_val = w_lanes_with_goal[i, 2]
        print(f"    索引{i}: dx={dx:.1f}, dy={dy:.1f}, Δs={delta_val:.1f}")
    
    print("\n✓ 可视化读取索引正确：索引i对应第i个观察到的w_lane")


if __name__ == "__main__":
    test_cumulative_calculation()
    test_visualization_indexing()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. 累积计算：逻辑正确，但需要确保只在有效点处赋值")
    print("2. 可视化读取：索引对齐正确")
    print("3. 潜在问题：当路径中有无效点时，累积计算可能受影响")


"""Manual diagnostics moved from simulator/goals.py."""

from _bootstrap import add_project_paths

add_project_paths()

import matplotlib.pyplot as plt
import numpy as np
import torch

from goals import PathPlanner

# 初始化规划器
path_planner = PathPlanner(
    map_path='maps/processed_map_Town01_stitched.json',
    device=torch.device('cuda'),
    enable_dense_paths=True,
)

# 测试指定的起点和终点对
# 测试1: [10, 67, -2] 到 [17, 65, -2]
# 测试2: [15, 19, -1] 到 [8, 67, -2]

start_ids = torch.tensor([
    [10, 67, -2],  # 测试1起点
    [19, 23,  1]
], dtype=torch.int32, device=path_planner.device)

end_ids = torch.tensor([
    [17, 65, -2],  # 测试1终点
    [10, 24, -1]
], dtype=torch.int32, device=path_planner.device)

print("=== 测试两种路径规划方法的差异 ===")
print(f"起点: {start_ids}")
print(f"终点: {end_ids}")

# 方法1：直接使用batch_shortest_paths_fixed_len
print("\n--- 方法1：直接使用batch_shortest_paths_fixed_len ---")
if path_planner.waypoint_graph_gpu is not None:
    triplets, mask, node_indices = path_planner.waypoint_graph_gpu.batch_shortest_paths_fixed_len(
        start_ids, end_ids, fixed_len=100
    )
    # 获取方法1的路径坐标
    method1_paths = []
    for i in range(len(start_ids)):
        valid_node_idx = node_indices[i][mask[i]]
        if len(valid_node_idx) > 0:
            coords = path_planner.waypoint_graph_gpu.node_xy[valid_node_idx].cpu().numpy()
            method1_paths.append(coords)
            print(f"方法1路径 {i+1}: {len(coords)}个点")
            print(f"  起点: {coords[0]}")
            print(f"  终点: {coords[-1]}")
        else:
            method1_paths.append(None)
            print(f"方法1路径 {i+1}: 无有效路径")
else:
    print("waypoint_graph_gpu未初始化")
    method1_paths = [None, None]

# 方法2：通过start_ids和end_ids找到对应的quad_id，使用plan_path
print("\n--- 方法2：通过quad_id使用plan_path ---")

# 找到start_ids和end_ids对应的quad_id
method2_quad_ids = []
for i in range(len(start_ids)):
    start_triplet = start_ids[i]
    end_triplet = end_ids[i]

    # 在waypoint_graph中找到对应的节点索引
    start_match = torch.all(path_planner.waypoint_graph_gpu.node_triplets == start_triplet, dim=1)
    end_match = torch.all(path_planner.waypoint_graph_gpu.node_triplets == end_triplet, dim=1)

    start_idx = torch.where(start_match)[0]
    end_idx = torch.where(end_match)[0]

    if start_idx.numel() > 0 and end_idx.numel() > 0:
        # 获取三元组对应的坐标
        start_coord = path_planner.waypoint_graph_gpu.node_xy[start_idx[0]]
        end_coord = path_planner.waypoint_graph_gpu.node_xy[end_idx[0]]

        # 通过坐标找到对应的quad_id
        # 在quads_info中查找最近的quad
        start_quad_found = False
        end_quad_found = False
        start_quad_id = -1
        end_quad_id = -1

        # 查找start_quad_id - 通过坐标匹配
        if hasattr(path_planner, 'quads_info'):
            # 计算所有quad中心点到start_coord的距离
            quad_centers = torch.stack([path_planner.quads_info['center_x'], 
                                      path_planner.quads_info['center_y']], dim=1)
            start_distances = torch.norm(quad_centers - start_coord, dim=1)
            start_nearest_idx = torch.argmin(start_distances)
            start_min_distance = start_distances[start_nearest_idx]

            # 如果距离足够近（比如小于5米），认为找到了对应的quad
            if start_min_distance < 5.0:
                start_quad_id = path_planner.quads_info['polyId'][start_nearest_idx].item()
                start_quad_found = True

        # 查找end_quad_id - 通过坐标匹配
        if hasattr(path_planner, 'quads_info'):
            end_distances = torch.norm(quad_centers - end_coord, dim=1)
            end_nearest_idx = torch.argmin(end_distances)
            end_min_distance = end_distances[end_nearest_idx]

            # 如果距离足够近（比如小于5米），认为找到了对应的quad
            if end_min_distance < 5.0:
                end_quad_id = path_planner.quads_info['polyId'][end_nearest_idx].item()
                end_quad_found = True

        if start_quad_found and end_quad_found:
            method2_quad_ids.append((start_quad_id, end_quad_id))

            print(f"路径 {i+1}:")
            print(f"  三元组: {start_triplet.tolist()} -> {end_triplet.tolist()}")
            print(f"  节点索引: {start_idx[0].item()} -> {end_idx[0].item()}")
            print(f"  坐标: {start_coord.tolist()} -> {end_coord.tolist()}")
            print(f"  真正的quad_id: {start_quad_id} -> {end_quad_id}")
        else:
            method2_quad_ids.append(None)
            print(f"路径 {i+1}: 未找到对应的quad_id")
            if not start_quad_found:
                print(f"    未找到坐标 {start_coord.tolist()} 对应的quad_id (最近距离: {start_min_distance.item():.2f})")
            if not end_quad_found:
                print(f"    未找到坐标 {end_coord.tolist()} 对应的quad_id (最近距离: {end_min_distance.item():.2f})")
    else:
        method2_quad_ids.append(None)
        print(f"路径 {i+1}: 未找到对应的节点索引")

# 使用plan_path规划路径
if all(quad_ids is not None for quad_ids in method2_quad_ids):
    # 构建quad_id张量
    start_quad_tensor = torch.tensor([[quad_ids[0] for quad_ids in method2_quad_ids]], 
                                    dtype=torch.int32, device=path_planner.device)
    end_quad_tensor = torch.tensor([[quad_ids[1] for quad_ids in method2_quad_ids]], 
                                  dtype=torch.int32, device=path_planner.device)
    # 强制修改第二个元素
    if start_quad_tensor.numel() >= 2:
        start_quad_tensor[0, 1] = torch.tensor(10894, dtype=torch.int32, device=path_planner.device)
    if end_quad_tensor.numel() >= 2:
        end_quad_tensor[0, 1] = torch.tensor(5679, dtype=torch.int32, device=path_planner.device)

    print("\n调用plan_path:")
    print(f"  输入start_quad_tensor: {start_quad_tensor}")
    print(f"  输入end_quad_tensor: {end_quad_tensor}")

    # 调用plan_path
    method2_path = path_planner.plan_path(start_quad_tensor, end_quad_tensor)

    # 提取方法2的路径坐标
    method2_paths = []
    for i in range(method2_path.shape[1]):
        path_i = method2_path[0, i].cpu().numpy()  # [512, 2]
        valid_mask = (path_i[:, 0] != -1) & (path_i[:, 1] != -1)
        valid_coords = path_i[valid_mask]

        if len(valid_coords) > 0:
            method2_paths.append(valid_coords)
            print(f"方法2路径 {i+1}: {len(valid_coords)}个点")
            print(f"  起点: {valid_coords[0]}")
            print(f"  终点: {valid_coords[-1]}")
        else:
            method2_paths.append(None)
            print(f"方法2路径 {i+1}: 无有效路径")
else:
    print("无法构建quad_id张量，跳过plan_path测试")
    method2_paths = [None, None]

# 比较两种方法的路径
print("\n--- 路径比较 ---")
for i in range(len(method1_paths)):
    print(f"\n路径 {i+1} 比较:")

    if method1_paths[i] is not None and method2_paths[i] is not None:
        path1 = method1_paths[i]
        path2 = method2_paths[i]

        print(f"  方法1点数: {len(path1)}")
        print(f"  方法2点数: {len(path2)}")

        # 比较起点和终点
        start_diff = np.linalg.norm(path1[0] - path2[0])
        end_diff = np.linalg.norm(path1[-1] - path2[-1])
        print(f"  起点差异: {start_diff:.6f}")
        print(f"  终点差异: {end_diff:.6f}")

        # 比较路径长度
        path1_length = np.sum(np.linalg.norm(np.diff(path1, axis=0), axis=1))
        path2_length = np.sum(np.linalg.norm(np.diff(path2, axis=0), axis=1))
        print(f"  方法1路径长度: {path1_length:.6f}")
        print(f"  方法2路径长度: {path2_length:.6f}")
        print(f"  路径长度差异: {abs(path1_length - path2_length):.6f}")

        # 检查路径是否一致
        if start_diff < 1e-6 and end_diff < 1e-6:
            print("  ✅ 起点和终点一致")
        else:
            print("  ❌ 起点或终点不一致")

        if abs(path1_length - path2_length) < 1e-6:
            print("  ✅ 路径长度一致")
        else:
            print("  ❌ 路径长度不一致")
    else:
        print("  无法比较：至少一种方法返回了无效路径")

# 可视化比较
print("\n--- 可视化比较 ---")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i in range(len(method1_paths)):
    ax = axes[i]

    # 绘制方法1的路径
    if method1_paths[i] is not None:
        path1 = method1_paths[i]
        ax.plot(path1[:, 0], path1[:, 1], 'b-', linewidth=2, label='Method1: batch_shortest_paths')
        ax.scatter(path1[0, 0], path1[0, 1], c='blue', marker='o', s=100, label='start')
        ax.scatter(path1[-1, 0], path1[-1, 1], c='blue', marker='x', s=100, label='goal')

    # 绘制方法2的路径
    if method2_paths[i] is not None:
        path2 = method2_paths[i]
        ax.plot(path2[:, 0], path2[:, 1], 'r--', linewidth=2, label='Method2: plan_path')
        ax.scatter(path2[0, 0], path2[0, 1], c='red', marker='o', s=100, label='start')
        ax.scatter(path2[-1, 0], path2[-1, 1], c='red', marker='x', s=100, label='goal')

    ax.set_title('road graph and agent positions, path plans')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

# 额外绘制：道路quads叠加到现有子图ax上
try:
    import json
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    with open('maps/processed_map_Town01_stitched.json', 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    quads_data = map_data.get('quads', [])

    # 将quads叠加到两个子图上
    if quads_data:
        patches = []
        for q in quads_data:
            verts = q.get('vertices', [])
            if len(verts) == 4:
                patches.append(Polygon([[verts[0]['x'], verts[0]['y']],
                                        [verts[1]['x'], verts[1]['y']],
                                        [verts[2]['x'], verts[2]['y']],
                                        [verts[3]['x'], verts[3]['y']]], closed=True))
        for ax in axes:
            if patches:
                p = PatchCollection(patches, alpha=0.12, facecolor='lightblue', edgecolor='black', linewidth=0.1)
                ax.add_collection(p)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"绘制道路网络可视化时出错: {e}")

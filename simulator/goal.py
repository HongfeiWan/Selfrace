"""
基于 GPU 加速的 Dijkstra 路径规划系统
用于计算地图中任意两个 w_lane_id 之间的最短路径
"""
import torch
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

class PathPlanner:
    """
    基于 GPU 的路径规划器
    使用单向导通图计算任意两个(road_id, lane_id)之间的路径
    
    图结构说明：
    - 节点：每个(road_id, lane_id)表示一个lane
    - 单向边：lane_i的end可以导向lane_j的start（如果两者距离<5）
    - 自然流动：每个lane内部从start到end是自然导通的，不需要显式表示
    通过这个图结构，可以计算出从任意lane的end到任意lane的end的所有路径
    """
    def __init__(self, map_path: str, device: str = 'cuda'):
        """
        初始化路径规划器（全部数据存储在GPU tensor上）
        Args:
            map_path: 地图 JSON 文件路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        # 检查设备可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            device = 'cpu'
        self.device = torch.device(device)
        self.map_path = map_path
        # 加载地图数据（临时，仅用于读取，不存储）
        with open(map_path, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
        w_lanes = map_data['w_lanes']
        quads = map_data['quads']
        # 创建quad的查找表，用于快速访问
        quads_by_id = {q['poly_id']: q for q in quads}
        
        # 存储quads数据以便后续使用
        self.quads_by_id = quads_by_id
        # 按 road_id, lane_id 分组 w_lanes，并找出每个分组的 start 和 end
        lane_groups = defaultdict(list)
        for w_lane in w_lanes:
            key = (w_lane['road_id'], w_lane['lane_id'])
            lane_groups[key].append(w_lane)
        # 为每个 road_id, lane_id 找出 start 和 end 的 w_lane
        lane_start_end_dict = {}
        w_lane_id_to_pos = {}  # 映射 w_lane_id 到 position
        for (road_id, lane_id), w_lanes_in_lane in lane_groups.items():
            # 根据 poly_id 获取对应的 quad 的 s 值
            w_lanes_with_s = []
            for w_lane in w_lanes_in_lane:
                poly_id = w_lane['poly_id']
                quad = quads_by_id[poly_id]
                s = quad.get('s', 0.0)
                w_lanes_with_s.append((w_lane, s))
            # 按 s 值排序
            w_lanes_with_s.sort(key=lambda x: x[1])
            # start 是 s 值最小的，end 是 s 值最大的
            start_w_lane = w_lanes_with_s[0][0]
            end_w_lane = w_lanes_with_s[-1][0]
            lane_start_end_dict[(road_id, lane_id)] = {
                'start': start_w_lane['w_lane_id'],
                'end': end_w_lane['w_lane_id']
            }
        # 创建 (road_id, lane_id) 到索引的映射
        self.lane_keys = sorted(lane_start_end_dict.keys())
        self.n_lanes = len(self.lane_keys)
        self.lane_to_idx = {key: idx for idx, key in enumerate(self.lane_keys)}
        # 批量构建所有w_lanes的坐标矩阵，一次性传输到GPU
        # 创建w_lane_id到索引的映射
        w_lane_id_to_idx = {w_lane['w_lane_id']: i for i, w_lane in enumerate(w_lanes)}
        # 批量创建坐标数组（先CPU numpy）
        coords_array = np.array([[w['center'][0], w['center'][1]] for w in w_lanes], dtype=np.float32)
        # 一次性转换为GPU tensor
        all_w_lane_positions = torch.from_numpy(coords_array).to(device=self.device, dtype=torch.float32)  # (n_w_lanes, 2)

        # 构建start和end位置tensor
        start_indices = [w_lane_id_to_idx[lane_start_end_dict[key]['start']] for key in self.lane_keys]
        end_indices = [w_lane_id_to_idx[lane_start_end_dict[key]['end']] for key in self.lane_keys]
        start_indices_t = torch.tensor(start_indices, dtype=torch.long, device=self.device)
        end_indices_t = torch.tensor(end_indices, dtype=torch.long, device=self.device)
        
        # 使用索引批量提取坐标
        self.start_positions = all_w_lane_positions[start_indices_t]  # (n_lanes, 2)
        self.end_positions = all_w_lane_positions[end_indices_t]  # (n_lanes, 2)
        # 3. 构建联通图：计算每个 lane 的 end 到其他 lane 的 start 的距离
        # 形成单向导通图：lane_i的end -> lane_j的start（如果距离够近）
        # self.end_positions: (n_lanes, 2)
        # self.start_positions: (n_lanes, 2)
        # 计算距离矩阵：dist[i,j] = ||end[i] - start[j]||
        end_positions_expanded = self.end_positions.unsqueeze(1)  # (n_lanes, 1, 2)
        start_positions_expanded = self.start_positions.unsqueeze(0)  # (1, n_lanes, 2)
        distances = torch.norm(end_positions_expanded - start_positions_expanded, dim=2)  # (n_lanes, n_lanes)
        
        # 4. 建立邻接矩阵和边权重矩阵
        # adjacency_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
        CONNECTION_THRESHOLD = 5.0
        self.end_to_start_matrix = (distances < CONNECTION_THRESHOLD).float()  # (n_lanes, n_lanes)
        # 预计算边权重矩阵：连接存在的边权重为距离，不存在的为INF
        # 这样在Dijkstra中就不需要重复计算了
        INF = 1e10
        self.edge_weights = torch.where(self.end_to_start_matrix > 0, distances, 
                                        torch.full_like(distances, INF))
        # 5. 图结构已构建完成
        # end_to_start_matrix[i][j] = 1 表示 lane_i 的 end 可以导向 lane_j 的 start
        # 由于每个lane内部 start->end 是天然导通的，因此：
        # 从 lane_i 的 end 可以到达 lane_j 的 start，
        # 那么就可以继续到达 lane_j 的 end
        # 将数据保留在 CPU 上的字典中（用于调试和索引）
        self.lane_start_end = lane_start_end_dict
        # 为了兼容BFS接口，将邻接矩阵统一命名为adjacency_matrix
        self.adjacency_matrix = self.end_to_start_matrix
        
        # 创建poly_id到lane_idx的映射
        # 每个poly_id对应一个quad，每个quad有road_id和lane_id
        self.poly_id_to_lane_idx = {}
        max_poly_id = max(quads_by_id.keys()) if quads_by_id else 0
        
        # 创建GPU tensor用于快速查找
        # poly_id_lookup[i] 存储poly_id=i对应的lane_idx，如果不存在则为-1
        poly_id_lookup_cpu = np.full(max_poly_id + 1, -1, dtype=np.int64)
        for poly_id, quad in quads_by_id.items():
            road_id = quad['road_id']
            lane_id = quad['lane_id']
            key = (road_id, lane_id)
            if key in self.lane_to_idx:
                lane_idx = self.lane_to_idx[key]
                self.poly_id_to_lane_idx[poly_id] = lane_idx
                poly_id_lookup_cpu[poly_id] = lane_idx
        # 转换为GPU tensor
        self.poly_id_lookup = torch.from_numpy(poly_id_lookup_cpu).to(device=self.device, dtype=torch.long)
        
        # 6. 预计算所有起点到终点的最短路径并存储在显存中
        print(f"开始预计算所有 {self.n_lanes} x {self.n_lanes} = {self.n_lanes * self.n_lanes} 条路径...")
        self._precompute_all_paths()
        print("路径预计算完成")

    def _precompute_all_paths(self):
        """
        预计算所有起点到终点的最短路径前驱矩阵
        优化方案：使用Floyd-Warshall算法的前驱矩阵
        存储优化：只存储前驱矩阵而不是完整路径，大大减少显存占用
        原方案：存储所有完整路径 O(n^2 * L) - 大量冗余
        优化方案：存储前驱矩阵 O(n^2) - 无冗余，快速重建路径
        时间复杂度：O(n^3)，空间复杂度：O(n^2)
        """
        print("  使用Floyd-Warshall算法优化计算...")
        INF = 1e10
        # 初始化距离矩阵和前驱矩阵
        # dist[i,j] 存储从i到j的最短距离
        # prev[i,j] 存储从i到j的最短路径上，j的前驱节点
        dist = self.edge_weights.clone()  # (n_lanes, n_lanes)
        prev = torch.full((self.n_lanes, self.n_lanes), -1, dtype=torch.long, device=self.device)
        # 初始化：自己到自己是0，直接连通的边设置前驱
        for i in range(self.n_lanes):
            dist[i, i] = 0  # 自己到自己是0
            prev[i, i] = i  # 自己到自己的前驱是自己
            for j in range(self.n_lanes):
                if i != j and self.edge_weights[i, j] < INF:
                    prev[i, j] = i  # 直接连通的前驱是起点
        # Floyd-Warshall主循环：动态规划
        # 考虑通过节点k中转，更新所有(i,j)对的最短路径
        for k in range(self.n_lanes):
            # 向量化更新：对于所有(i,j)，尝试通过k中转
            dist_ik = dist[:, k:k+1].expand(-1, self.n_lanes)  # (n_lanes, n_lanes)
            dist_kj = dist[k:k+1, :].expand(self.n_lanes, -1)  # (n_lanes, n_lanes)
            dist_ikj = dist_ik + dist_kj  # (n_lanes, n_lanes)
            # 找到通过k中转更短的路径
            better_mask = dist_ikj < dist
            # 更新距离和前驱
            dist = torch.where(better_mask, dist_ikj, dist)
            # 如果通过k中转更短，则j的前驱变为k的前驱
            prev_kj = prev[k:k+1, :].expand(self.n_lanes, -1)
            prev = torch.where(better_mask, prev_kj, prev)
        # 存储前驱矩阵和距离矩阵（用于快速重建路径）
        self.prev_matrix = prev  # (n_lanes, n_lanes)
        self.dist_matrix = dist  # (n_lanes, n_lanes)
        # 统计连通性
        valid_paths_count = (prev >= 0).sum().item()
        max_dist = dist[dist < INF].max().item() if (dist < INF).any() else 0
        print(f"  有效路径数: {valid_paths_count}")
        print(f"  最大路径距离: {max_dist:.2f}")
    
    def find_shortest_path(self, start_lane_idx, target_lane_idx):
        """
        从前驱矩阵重建完整路径
        
        Args:
            start_lane_idx: 起始lane索引
            target_lane_idx: 目标lane索引
        Returns:
            torch.Tensor: 路径tensor，如果不可达返回None
        """
        if self.prev_matrix[start_lane_idx, target_lane_idx] < 0:
            return None
        # 反向追踪前驱节点构建路径
        path_list = []
        current = target_lane_idx
        # 从前驱矩阵中构建路径（反向追踪）
        # 注意：前驱矩阵中的值表示从start到current的路径上，current的前驱节点
        path_list.append(current)
        while current != start_lane_idx:
            current = self.prev_matrix[start_lane_idx, current]
            if current < 0:
                return None
            path_list.append(current)
        # 反转路径（从start到target）
        path_list.reverse()
        return torch.tensor(path_list, device=self.device, dtype=torch.long)
    
    def path_plan(self, start_poly_ids, end_poly_ids):
        """
        批量路径规划：从poly_id到poly_id的最短路径
        
        Args:
            start_poly_ids: (B, M) 起点poly_id的tensor
            end_poly_ids: (B, M) 终点poly_id的tensor
            
        Returns:
            List[List[torch.Tensor]]: (B, M) 每条路径的结果
                每个元素是一个torch.Tensor，包含该路径经过的所有lane_idx
                如果路径不存在则返回None
        """
        B, M = start_poly_ids.shape
        
        # 确保输入tensor在正确设备上
        start_poly_ids = start_poly_ids.to(self.device)
        end_poly_ids = end_poly_ids.to(self.device)
        
        # 使用GPU加速批量转换poly_id到lane_idx
        # poly_id_lookup[i] 存储poly_id=i对应的lane_idx，如果不存在则为-1
        start_lane_indices = self.poly_id_lookup[start_poly_ids]  # (B, M)
        end_lane_indices = self.poly_id_lookup[end_poly_ids]  # (B, M)
        
        # 批量查询最短路径
        results = []
        for b in range(B):
            batch_results = []
            for m in range(M):
                start_lane_idx = start_lane_indices[b, m].item()
                end_lane_idx = end_lane_indices[b, m].item()
                
                # 检查有效性
                if start_lane_idx < 0 or end_lane_idx < 0:
                    batch_results.append(None)
                else:
                    # 调用find_shortest_path查询路径
                    path = self.find_shortest_path(start_lane_idx, end_lane_idx)
                    batch_results.append(path)
            results.append(batch_results)
        return results
    
if __name__ == '__main__':
    import random
    planner = PathPlanner(map_path='maps/town2.json', device='cuda')
    print(f"总共 {len(planner.lane_start_end)} 个 (road_id, lane_id) 组合")
    print(f"邻接矩阵形状: {planner.adjacency_matrix.shape}")
    print(f"联通数量: {planner.adjacency_matrix.sum().item()}")
    # 加载地图数据以获取所有poly_id
    with open('maps/town2.json', 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    all_quads = map_data['quads']
    all_poly_ids = [quad['poly_id'] for quad in all_quads if quad['poly_id'] in planner.poly_id_to_lane_idx]
    # 随机选择两个不同的poly_id
    start_poly_id = random.choice(all_poly_ids)
    end_poly_id = random.choice(all_poly_ids)
    while end_poly_id == start_poly_id:
        end_poly_id = random.choice(all_poly_ids)
    print(f"\n随机选择路径:")
    print(f"  起点poly_id: {start_poly_id}")
    print(f"  终点poly_id: {end_poly_id}")
    # 使用path_plan查询路径
    start_poly_tensor = torch.tensor([[start_poly_id]], dtype=torch.long)
    end_poly_tensor = torch.tensor([[end_poly_id]], dtype=torch.long)
    paths = planner.path_plan(start_poly_tensor, end_poly_tensor)
    path = paths[0][0]
    if path is None:
        print("未找到路径，尝试其他随机起点和终点...")
        # 尝试多次找到有路径的点
        for _ in range(10):
            start_poly_id = random.choice(all_poly_ids)
            end_poly_id = random.choice(all_poly_ids)
            while end_poly_id == start_poly_id:
                end_poly_id = random.choice(all_poly_ids)
        
            start_poly_tensor = torch.tensor([[start_poly_id]], dtype=torch.long)
            end_poly_tensor = torch.tensor([[end_poly_id]], dtype=torch.long)
            paths = planner.path_plan(start_poly_tensor, end_poly_tensor)
            path = paths[0][0]
        
            if path is not None:
                print(f"\n重新选择路径:")
                print(f"  起点poly_id: {start_poly_id}")
                print(f"  终点poly_id: {end_poly_id}")
                break
    if path is None:
        print("未找到任何路径，退出")
        exit(0)
    print(f"\n找到路径 (共 {len(path)} 个lane):")
    for i, lane_idx in enumerate(path.cpu().tolist()):
        print(f"  {i+1}. {planner.lane_keys[lane_idx]}")
    # 获取起点和终点的lane_idx用于可视化
    start_lane_idx = planner.poly_id_to_lane_idx[start_poly_id]
    target_lane_idx = planner.poly_id_to_lane_idx[end_poly_id]
    # 可视化路径
    print("\n开始可视化路径...")
    fig, ax = plt.subplots(figsize=(12, 8))
    # 地图数据已在前面加载
    quads = all_quads
    # 绘制 quads 作为背景
    for quad in quads:
        vertices = quad['vertices']
        vertices_2d = [(v[0], v[1]) for v in vertices]
        polygon = Polygon(vertices_2d, closed=True,
                         facecolor='lightgray', edgecolor='gray',
                         alpha=0.1, linewidth=0.1)
        ax.add_patch(polygon)
    # 转换 tensor 到 numpy 用于绘制
    start_pos = planner.start_positions.cpu().numpy()
    end_pos = planner.end_positions.cpu().numpy()
    # 提取中间路径上的所有waypoints（从第二个到倒数第二个lane）
    all_w_lanes = map_data['w_lanes']
    quads_by_id = {q['poly_id']: q for q in all_quads}
    # 按road_id和lane_id分组w_lanes
    lane_groups = defaultdict(list)
    for w_lane in all_w_lanes:
        key = (w_lane['road_id'], w_lane['lane_id'])
        lane_groups[key].append(w_lane)
    # 绘制中间路径上的waypoints（从第二个到倒数第二个lane）
    for i, lane_idx in enumerate(path):
        if i == 0 or i == len(path) - 1:
            # 跳过第一个和最后一个lane
            continue
        
        # 获取当前lane的(road_id, lane_id)
        road_id, lane_id = planner.lane_keys[lane_idx]
        
        # 获取该lane的所有waypoints
        w_lanes_in_lane = lane_groups[(road_id, lane_id)]
        
        # 按poly_id的s值排序
        w_lanes_with_s = []
        for w_lane in w_lanes_in_lane:
            poly_id = w_lane['poly_id']
            quad = quads_by_id[poly_id]
            s = quad.get('s', 0.0)
            w_lanes_with_s.append((w_lane, s))
        w_lanes_with_s.sort(key=lambda x: x[1])
        
        # 提取坐标并绘制
        waypoint_positions = []
        for w_lane, s in w_lanes_with_s:
            center = w_lane['center']
            waypoint_positions.append([center[0], center[1]])
        
        if len(waypoint_positions) > 1:
            waypoint_positions = np.array(waypoint_positions)
            # 绘制waypoints（小圆点）
            ax.scatter(waypoint_positions[:, 0], waypoint_positions[:, 1], 
                      c='purple', s=20, alpha=0.7, zorder=5, marker='o')
            
            # 用箭头连接waypoints
            for j in range(len(waypoint_positions) - 1):
                ax.annotate('', xy=(waypoint_positions[j+1, 0], waypoint_positions[j+1, 1]), 
                           xytext=(waypoint_positions[j, 0], waypoint_positions[j, 1]),
                           arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, alpha=0.6, zorder=4))
    
    # 绘制起点（绿色大圆圈）- 起点在起始lane的end位置
    ax.scatter(end_pos[start_lane_idx, 0], end_pos[start_lane_idx, 1], 
              c='green', s=100, alpha=0.9, label='Start (end)', zorder=6, edgecolors='black', linewidth=2)
    # 绘制终点（红色大圆圈）- 终点在目标lane的start位置
    ax.scatter(start_pos[target_lane_idx, 0], start_pos[target_lane_idx, 1], 
              c='red', s=100, alpha=0.9, label='Target (start)', zorder=6, edgecolors='black', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Shortest Path: from poly_id {start_poly_id} to poly_id {end_poly_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    print("可视化完成，显示图片...")
    plt.tight_layout()
    plt.show()
    
    
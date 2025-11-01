import torch
from typing import Dict, Tuple, Optional
import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.spatial_hash import SpatialHash

class CollisionChecker:
    """
    一个完全批处理化的碰撞检测器，专为大规模地图和高性能GPU计算优化。
    核心优化:
    1.  宽阶段 (_broad_phase_vectorized):
        - 针对大地图，使用稀疏张量 (Sparse Tensors) 表示网格占用，
          极大地减少了内存消耗和计算量。
        - 通过批处理的稀疏矩阵乘法并行查找候选对，解决了密集矩阵的扩展性问题。
    2.  窄阶段 (_narrow_phase_vectorized):
        - 继续应用 "Gather-Compute-Scatter" 模式，保持高效的全并行计算。
    """
    def __init__(self, config: Dict, spatial_hash: SpatialHash):
        """
        目的: 初始化碰撞检测器的配置参数。
        逻辑:
        - 设置计算设备。
        - 从配置中获取地图的物理范围 (map_extent_m) 和网格的分辨率 (grid_width/height)。
        - 基于以上参数，动态计算出每个网格单元的物理尺寸 (cell_size)。
        - 预估并存储每个智能体可能覆盖的最大网格数，这是一个用于内存预分配的优化参数。
        - 设置在宽阶段为每个智能体筛选的最大邻居数。
        """
        # 显式初始化所有成员变量，保证其在使用前已定义
        self.device = torch.device(config.get('device', 'cuda'))
        self.spatial_hash = spatial_hash
        self.cell_size = 0.0
        self.max_neighbors = 0
        
        # 从配置文件读取空间哈希参数
        simulator_config = config.get('simulator')
        hash_config = simulator_config.get('hash')
        # 读取cell_size（新键名为主，兼容旧键名），并提供安全默认值
        self.cell_size = hash_config.get('cell_size')
        # --- 优化的 max_cells_per_agent 计算 ---
        # 目的: 估算单个智能体在一个时间步内可能覆盖的最大网格单元数。
        # 逻辑:
        # 1. 计算最大位移：max_speed * sim_dt。
        # 2. 计算扫掠体的最大维度：vehicle_length + displacement。
        # 3. 将此物理维度转换为网格单元数 (span)，并添加安全余量。
        # 4. 计算最终的方形区域内的单元格总数。
        # 获取simulator配置，支持嵌套配置结构
        simulator_config = config.get('simulator')
        observation_cfg = simulator_config.get('observation')
        self.max_neighbors = observation_cfg.get('num_neighbors')
        #print(f"Collision checker initialized.")
        print(f"Calculated cell size: {self.cell_size:.2f}m")

    def check(self,
              states_t0: torch.Tensor,
              states_t1: torch.Tensor,
              static_obstacles: Optional[torch.Tensor] = None,
              debug: bool = False,
              debug_env_idx: int = 0) -> torch.Tensor:
        """
        目的: 作为主入口函数，对一批智能体的状态进行完整的碰撞检测。

        逻辑:
        1.  从 t1 时刻的状态中提取出当前处于激活状态的智能体。
        2.  计算所有智能体在 t0 和 t1 时刻的边界框顶点。
        3.  调用宽阶段方法 (`_broad_phase_vectorized`)，利用空间哈希快速筛选出可能碰撞的智能体对。
        4.  调用窄阶段方法 (`_narrow_phase_vectorized`)，对候选对进行精确的、连续的碰撞检测。
        5.  如果提供了静态障碍物，则调用静态碰撞检测方法。
        6.  合并动态和静态碰撞结果，并用激活掩码过滤，返回最终结果。
        7.  (Debug) 如果开启debug模式，额外返回用于可视化的调试信息。
        """
        # 确保输入张量在正确的设备上
        states_t0 = states_t0.to(self.device)
        states_t1 = states_t1.to(self.device)
        
        active_mask = states_t1[..., 6] > 0.5
        verts_t0 = self._get_world_vertices(states_t0)
        verts_t1 = self._get_world_vertices(states_t1)

        candidate_pairs, broad_phase_debug_info = self._broad_phase_vectorized(
            active_mask, verts_t0, verts_t1, debug=debug, debug_env_idx=debug_env_idx
        )
        
        dynamic_collisions = self._narrow_phase_vectorized(
            candidate_pairs, active_mask, states_t0, states_t1, verts_t0, verts_t1
        )

        final_collisions = dynamic_collisions
        
        if static_obstacles is not None:
            static_collisions = self._check_static_collisions(states_t0, states_t1, static_obstacles)
            final_collisions = torch.logical_or(final_collisions, static_collisions)

        final_collisions = final_collisions & active_mask

        if debug:
            debug_data = {
                'broad_phase': broad_phase_debug_info
            }
            return final_collisions, debug_data
        
        return final_collisions

    def _broad_phase_vectorized(self, active_mask: torch.Tensor,
                                verts_t0: torch.Tensor, verts_t1: torch.Tensor,
                                debug: bool = False, debug_env_idx: int = 0) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        目的: 调用共享的空间哈希对象来高效地找出所有可能发生碰撞的智能体对。
        """
        B, M, _, _ = verts_t0.shape
        return self.spatial_hash.query_dynamic_pairs(
            B, M, active_mask, verts_t0, verts_t1, self.max_neighbors, debug, debug_env_idx
        )

    def _narrow_phase_vectorized(self, candidate_pairs: torch.Tensor,
                                 active_mask: torch.Tensor, states_t0: torch.Tensor, states_t1: torch.Tensor,
                                 verts_t0: torch.Tensor, verts_t1: torch.Tensor) -> torch.Tensor:
        """
        目的: 对宽阶段筛选出的候选对进行精确的连续碰撞检测。

        逻辑: (Gather-Compute-Scatter)
        1.  Gather (收集): 
            - 过滤掉无效和重复的候选对。
            - 将所有有效的、需要检查的智能体对的索引展平。
            - 使用高级索引 (advanced indexing) 一次性从原始状态张量中提取出所有这些对的数据（t0/t1的状态和顶点）。
        2.  Compute (计算):
            - 将收集到的数据作为一个大的批次，传递给 `_check_one_way_collision` 函数。
            - 执行两次单向检测（(j, k) 和 (k, j)）以完成双向检测，所有计算完全并行。
        3.  Scatter (散播):
            - 创建一个全零的最终结果张量。
            - 将计算出的碰撞结果（一个布尔列表）并行地写回到结果张量中对应的智能体位置。
        """
        B, M, K = candidate_pairs.shape
        if K == 0:
            return torch.zeros((B, M), dtype=torch.bool, device=self.device)

        # 确保所有输入张量都在正确的设备上
        candidate_pairs = candidate_pairs.to(self.device)
        active_mask = active_mask.to(self.device)
        
        agent_j_indices = torch.arange(M, device=self.device).view(1, M, 1).expand(B, M, K)
        agent_k_indices = candidate_pairs

        valid_mask = (agent_k_indices != -1) & active_mask.unsqueeze(-1)
        unique_mask = agent_j_indices < agent_k_indices
        final_mask = valid_mask & unique_mask

        batch_idx_flat = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, M, K)[final_mask]
        j_idx_flat = agent_j_indices[final_mask]
        k_idx_flat = agent_k_indices[final_mask]

        if j_idx_flat.numel() == 0:
            return torch.zeros((B, M), dtype=torch.bool, device=self.device)

        j_states_t0, j_states_t1 = states_t0[batch_idx_flat, j_idx_flat], states_t1[batch_idx_flat, j_idx_flat]
        j_verts_t0, j_verts_t1 = verts_t0[batch_idx_flat, j_idx_flat], verts_t1[batch_idx_flat, j_idx_flat]
        k_states_t0, k_states_t1 = states_t0[batch_idx_flat, k_idx_flat], states_t1[batch_idx_flat, k_idx_flat]
        k_verts_t0, k_verts_t1 = verts_t0[batch_idx_flat, k_idx_flat], verts_t1[batch_idx_flat, k_idx_flat]

        coll_1 = self._check_one_way_collision(j_states_t0, j_states_t1, k_verts_t0, k_verts_t1)
        coll_2 = self._check_one_way_collision(k_states_t0, k_states_t1, j_verts_t0, j_verts_t1)
        pair_collisions = torch.logical_or(coll_1, coll_2)

        collisions = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        colliding_batch_idx = batch_idx_flat[pair_collisions]
        colliding_j_idx = j_idx_flat[pair_collisions]
        colliding_k_idx = k_idx_flat[pair_collisions]

        collisions.index_put_((colliding_batch_idx, colliding_j_idx), torch.tensor(True, device=self.device))
        collisions.index_put_((colliding_batch_idx, colliding_k_idx), torch.tensor(True, device=self.device))
        
        return collisions
    
    def _get_world_vertices(self, states: torch.Tensor) -> torch.Tensor:
        """
        目的: 从智能体的状态向量中计算其矩形边界框的四个顶点在世界坐标系下的坐标。

        逻辑:
        - 解包状态向量获取中心点、偏航角和尺寸。
        - 根据偏航角计算出车身坐标系的基向量（x轴和y轴方向）。
        - 从中心点出发，通过基向量和半长/半宽，计算出四个顶点的坐标。
        - 所有操作都是并行的张量运算。
        """
        center_x, center_y, yaw, _, length, width, _ = states.unbind(-1)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        vec_x = torch.stack([cos_yaw, sin_yaw], dim=-1)
        vec_y = torch.stack([-sin_yaw, cos_yaw], dim=-1)
        half_l, half_w = length.unsqueeze(-1) / 2, width.unsqueeze(-1) / 2
        center = states[..., :2]
        v1 = center + half_l * vec_x + half_w * vec_y
        v2 = center - half_l * vec_x + half_w * vec_y
        v3 = center - half_l * vec_x - half_w * vec_y
        v4 = center + half_l * vec_x - half_w * vec_y
        return torch.stack([v1, v2, v3, v4], dim=-2)

    def _transform_points_to_local_frame(self, points: torch.Tensor, ref_state: torch.Tensor) -> torch.Tensor:
        """
        目的: 将一批世界坐标系下的点，转换到一批参考智能体的局部坐标系中。

        逻辑:
        1.  将所有点相对于参考智能体的中心点进行平移。
        2.  构建一个基于参考智能体偏航角的逆旋转矩阵。
        3.  将平移后的点应用逆旋转，得到局部坐标。
        """
        center = ref_state[..., :2].unsqueeze(-2)
        yaw = ref_state[..., 2]
        cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)
        rot_matrix = torch.stack([torch.stack([cos_y, -sin_y], dim=-1), torch.stack([sin_y, cos_y], dim=-1)], dim=-2)
        translated_points = points - center
        return torch.matmul(translated_points, rot_matrix.transpose(-2, -1))

    def _check_one_way_collision(self, ref_states_t0, ref_states_t1, mov_verts_t0, mov_verts_t1) -> torch.Tensor:
        """
        目的: 执行单向的连续碰撞检测。将移动物体(mov)视为在参考物体(ref)的局部坐标系中运动。

        逻辑:
        1.  将移动物体的 t0 和 t1 时刻的顶点，全部转换到参考物体的 t0 和 t1 时刻的局部坐标系中。
        2.  在参考物体的局部坐标系中，其自身是一个固定的、以原点为中心的AABB。
        3.  移动物体顶点的运动轨迹在参考系中变成了一系列的线段。
        4.  调用线段与AABB的相交测试函数，检查是否有任何一条运动轨迹线段与参考物体的AABB相交。
        """
        mov_verts_local_t0 = self._transform_points_to_local_frame(mov_verts_t0, ref_states_t0)
        mov_verts_local_t1 = self._transform_points_to_local_frame(mov_verts_t1, ref_states_t1)
        ref_dims = ref_states_t0[..., 4:6] / 2.0
        intersections = self._line_segment_aabb_intersection(mov_verts_local_t0, mov_verts_local_t1, -ref_dims.unsqueeze(-2), ref_dims.unsqueeze(-2))
        return intersections.any(dim=-1)

    def _line_segment_aabb_intersection(self, p0, p1, aabb_min, aabb_max) -> torch.Tensor:
        """
        目的: 使用Slab方法，批量检测一批线段是否与一个轴对齐包围盒(AABB)相交。

        逻辑:
        - 该方法将AABB看作是几个"厚板"（slabs）的交集（例如，x方向一个，y方向一个）。
        - 计算线段与每个厚板的两个边界平面的相交时间参数 t。
        - 找到所有厚板进入时间的最大值 (t_enter) 和所有厚板离开时间的最小值 (t_exit)。
        - 如果 t_enter < t_exit，并且相交区间 [t_enter, t_exit] 与线段自身的参数区间 [0, 1] 有重叠，则发生碰撞。
        """
        eps = 1e-8
        direction = p1 - p0
        # 使用 torch.copysign 避免除以零，同时保持正确的方向性，比简单的加eps更鲁棒
        inv_direction = 1.0 / (direction + torch.copysign(torch.full_like(direction, eps), direction))

        # 计算与两条边界平面的相交时间
        t_plane1 = (aabb_min - p0) * inv_direction
        t_plane2 = (aabb_max - p0) * inv_direction

        # 修正: 必须对每个轴的两个相交时间进行排序，以正确处理负方向的射线
        # t_near_per_axis 是进入每个轴向厚板的时间
        # t_far_per_axis 是离开每个轴向厚板的时间
        t_near_per_axis = torch.min(t_plane1, t_plane2)
        t_far_per_axis = torch.max(t_plane1, t_plane2)

        # 最终的进入时间是所有轴进入时间中的最晚的那个
        t_enter, _ = torch.max(t_near_per_axis, dim=-1)
        # 最终的离开时间是所有轴离开时间中的最早的那个
        t_exit, _ = torch.min(t_far_per_axis, dim=-1)
        
        # 如果进入时间晚于离开时间，或者整个相交区间都在线段之外，则没有碰撞
        no_collision = (t_enter >= t_exit) | (t_exit <= 0) | (t_enter >= 1)
        return ~no_collision

    def _check_static_collisions(self, states_t0: torch.Tensor, states_t1: torch.Tensor, static_obstacles: torch.Tensor) -> torch.Tensor:
        """
        目的: 批量检测所有动态智能体与所有静态障碍物之间的碰撞。

        逻辑:
        - 将静态障碍物视为一种特殊的智能体，其在 t0 和 t1 时刻的状态完全相同。
        - 利用广播机制，将动态智能体（作为移动方）与所有静态障碍物（作为参考方）进行配对。
        - 调用已经向量化的 `_check_one_way_collision` 函数，一次性完成所有动态-静态对的碰撞检测。
        - 如果一个智能体与任何一个障碍物发生碰撞，则将其标记为碰撞。
        """
        B, M, _ = states_t0.shape
        O, _, _ = static_obstacles.shape
        all_verts_t0 = self._get_world_vertices(states_t0)
        all_verts_t1 = self._get_world_vertices(states_t1)
        obs_centers = torch.mean(static_obstacles, dim=1)
        obs_states = torch.zeros(O, 7, device=self.device)
        obs_states[:, :2] = obs_centers
        obs_states[:, 6] = 1.0
        mov_verts_t0 = all_verts_t0.unsqueeze(2)
        mov_verts_t1 = all_verts_t1.unsqueeze(2)
        ref_states = obs_states.view(1, 1, O, 7)
        collisions = self._check_one_way_collision(ref_states, ref_states, mov_verts_t0, mov_verts_t1)
        return collisions.any(dim=-1)

# if __name__ == '__main__':
#     import random
#     import math
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Polygon as MplPolygon

#     # 读取默认配置
#     from road import RoadNetwork
#     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     cfg_path = os.path.join(repo_root, 'configs', 'default_config.json')
#     with open(cfg_path, 'r', encoding='utf-8') as f:
#         full_cfg = json.load(f)

#     # 设备
#     device_str = full_cfg.get('device')
#     device = torch.device(device_str)
#     print(f"Using device: {device}")

#     # 从配置中取需要的字段（带回退）
#     sim_cfg = full_cfg.get('simulator', {})
#     hash_cfg = sim_cfg.get('hash', {})
#     cell_size = hash_cfg.get('cell_size', 20.0)

#     # 为演示与SpatialHash初始化设置合理边界（如无专门字段则使用10km）
#     map_extent = sim_cfg.get('map_extent_m', 10000.0)
#     grid_width = int(max(1, map_extent / max(cell_size, 1e-6)))

#     # 假设地图中心为原点
#     min_bounds = torch.tensor([-map_extent/2, -map_extent/2], device=device)
#     max_bounds = torch.tensor([map_extent/2, map_extent/2], device=device)
#     spatial_hash = SpatialHash(cell_size, min_bounds, max_bounds, device)
#     checker = CollisionChecker(config=full_cfg, spatial_hash=spatial_hash)
    
#     # 加载路网
#     map_path = os.path.join(repo_root, 'maps', full_cfg.get('default_map'))
#     rn = RoadNetwork(map_path, device)
#     quads_t = rn.quads_vertices
#     quads = quads_t.detach().cpu().numpy()
#     quad_dirs_t = rn.quad_directions
#     quad_dirs = quad_dirs_t.detach().cpu().numpy()
#     centerlines_t = rn.quad_centerlines
#     quad_centers = centerlines_t[:, 0, :].detach().cpu().numpy()

#     # 配置车辆尺寸
#     dyn_cfg = sim_cfg.get('dynamics', {})
#     vehicle_length = float(dyn_cfg.get('vehicle_length', 4.5))
#     vehicle_width = float(dyn_cfg.get('vehicle_width', 2.0))

#     # 随机放置100辆静态车
#     random.seed(42)
#     num_static = 100
#     total_quads = quads.shape[0]
#     choose_n = min(num_static, total_quads)
#     chosen_indices = random.sample(range(total_quads), choose_n) if total_quads >= choose_n else list(range(total_quads))

#     def make_state(center, yaw, active=1.0):
#         return [float(center[0]), float(center[1]), float(yaw), 0.0, vehicle_length, vehicle_width, float(active)]

#     static_states = [
#         make_state(quad_centers[i], math.atan2(quad_dirs[i, 1], quad_dirs[i, 0]), 1.0)
#         for i in chosen_indices
#     ]

#     # 可控车辆
#     ctrl_idx = random.randrange(total_quads)
#     ctrl_state = make_state(quad_centers[ctrl_idx], math.atan2(quad_dirs[ctrl_idx, 1], quad_dirs[ctrl_idx, 0]), 1.0)

#     def pack_states(ctrl, stat_list):
#         all_states = [ctrl] + stat_list
#         return torch.tensor([all_states], dtype=torch.float32, device=device)

#     states_t0 = pack_states(ctrl_state, static_states)
#     states_t1 = states_t0.clone()
#     # 盒装为可变容器，便于闭包内更新
#     states_t0_box = [states_t0]
#     states_t1_box = [states_t1]

#     # 绘制
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title('Collision view (Blue: safe, Red: collision). Arrow keys to move. +/- to zoom')

#     # 画路面四边形边界
#     for q in quads:
#         poly = MplPolygon(q, closed=True, edgecolor=(0.8, 0.8, 0.8, 0.6), facecolor='none', linewidth=0.5)
#         ax.add_patch(poly)

#     vehicle_patches = []

#     def draw_vehicles(states, collided=False):
#         for p in vehicle_patches:
#             p.remove()
#         vehicle_patches.clear()
#         verts = checker._get_world_vertices(states).squeeze(0).detach().cpu().numpy()
#         color_ctrl = 'red' if collided else 'blue'
#         p0 = MplPolygon(verts[0], closed=True, edgecolor=color_ctrl, facecolor='none', linewidth=2.0)
#         ax.add_patch(p0); vehicle_patches.append(p0)
#         for i in range(1, verts.shape[0]):
#             p = MplPolygon(verts[i], closed=True, edgecolor='black', facecolor='none', linewidth=1.0, alpha=0.6)
#             ax.add_patch(p); vehicle_patches.append(p)

#     def set_view(center, zoom=80.0):
#         cx, cy = center
#         ax.set_xlim(cx - zoom, cx + zoom)
#         ax.set_ylim(cy - zoom, cy + zoom)

#     current_zoom_box = [80.0]
#     set_view((ctrl_state[0], ctrl_state[1]), zoom=current_zoom_box[0])

#     def update_and_render():
#         result = checker.check(states_t0_box[0], states_t1_box[0])
#         collided = bool(result.squeeze(0)[0].item())
#         draw_vehicles(states_t1_box[0], collided=collided)
#         fig.canvas.draw_idle()

#     update_and_render()

#     move_step = 2.0
#     rot_step = math.radians(5.0)
#     zoom_step = 10.0

#     def on_key(event):
#         ctrl = states_t1_box[0][0, 0].clone()
#         x, y, yaw = float(ctrl[0].item()), float(ctrl[1].item()), float(ctrl[2].item())
#         if event.key == 'up':
#             x += move_step * math.cos(yaw)
#             y += move_step * math.sin(yaw)
#         elif event.key == 'down':
#             x -= move_step * math.cos(yaw)
#             y -= move_step * math.sin(yaw)
#         elif event.key == 'left':
#             yaw += rot_step
#         elif event.key == 'right':
#             yaw -= rot_step
#         elif event.key in ['+', '=']:
#             current_zoom_box[0] = max(20.0, current_zoom_box[0] - zoom_step)
#         elif event.key in ['-', '_']:
#             current_zoom_box[0] = min(400.0, current_zoom_box[0] + zoom_step)
#         else:
#             return
#         ctrl[0] = x; ctrl[1] = y; ctrl[2] = yaw
#         states_t0_box[0] = states_t1_box[0].clone()
#         states_t1_box[0][0, 0] = ctrl
#         set_view((x, y), zoom=current_zoom_box[0])
#         update_and_render()

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     plt.show()
    
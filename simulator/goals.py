import torch
import json
import numpy as np


class PathPlanner:
    def __init__(self,map_path: str, device: torch.device):
        print(f"==========PathPlanner init==========")
        self.device = device
        # 如果提供了map，自动加载数据
        if map_path is not None:
            cross_data_path = map_path.replace('processed_map_', 'cross_data_processed_map_')
            # 加载cross数据
            with open(cross_data_path, 'r', encoding='utf-8') as f:
                cross_data = json.load(f)
            with open(map_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
        else:
            raise ValueError("map_path is required")
        # ===================存直路的quad_id==============================
        self.filtered_quad_indices = torch.tensor(cross_data.get('filtered_quad_indices', []), dtype=torch.int32, device=self.device)

        # ===================存cross的信息================================
        cross_data_dict = {}
        for key, value in cross_data.items():
            if not key.startswith('cross_'):
                continue
            cross_id = int(value['cross_id'])
            # 处理start_waypoints
            start_wps = value.get('start_waypoints', [])
            if start_wps:
                start_waypoint_ids = torch.tensor([wp['waypoint_id'] for wp in start_wps], dtype=torch.int32, device=self.device)
                start_x = torch.tensor([wp['x'] for wp in start_wps], dtype=torch.float32, device=self.device)
                start_y = torch.tensor([wp['y'] for wp in start_wps], dtype=torch.float32, device=self.device)
                start_road_id = torch.tensor([wp['road_id'] for wp in start_wps], dtype=torch.int32, device=self.device)
                start_lane_id = torch.tensor([wp['lane_id'] for wp in start_wps], dtype=torch.int32, device=self.device)
            else:
                start_x = start_y = start_road_id = start_lane_id = torch.empty(0, device=self.device)

            # 处理end_waypoints
            end_wps = value.get('end_waypoints', [])
            if end_wps:
                end_waypoint_ids = torch.tensor([wp['waypoint_id'] for wp in end_wps], dtype=torch.int32, device=self.device)
                end_x = torch.tensor([wp['x'] for wp in end_wps], dtype=torch.float32, device=self.device)
                end_y = torch.tensor([wp['y'] for wp in end_wps], dtype=torch.float32, device=self.device)
                end_road_id = torch.tensor([wp['road_id'] for wp in end_wps], dtype=torch.int32, device=self.device)
                end_lane_id = torch.tensor([wp['lane_id'] for wp in end_wps], dtype=torch.int32, device=self.device)
            else:
                end_x = end_y = end_road_id = end_lane_id = torch.empty(0, device=self.device)

            # 处理paths
            paths = value.get('paths', [])
            paths_list = []
            for path in paths:
                from_end = path.get('from_end_waypoint', {})
                to_start = path.get('to_start_waypoint', {})
                path_quad_ids = path.get('path_quad_ids', [])
                path_dict = {
                    'from_end_waypoint_id': torch.tensor([from_end.get('from_end_waypoint_id', 0)], dtype=torch.int32, device=self.device),
                    'from_end_x': torch.tensor([from_end.get('x', 0.0)], dtype=torch.float32, device=self.device),
                    'from_end_y': torch.tensor([from_end.get('y', 0.0)], dtype=torch.float32, device=self.device),
                    'from_end_road_id': torch.tensor([from_end.get('road_id', 0)], dtype=torch.int32, device=self.device),
                    'from_end_lane_id': torch.tensor([from_end.get('lane_id', 0)], dtype=torch.int32, device=self.device),
                    'to_start_waypoint_id': torch.tensor([to_start.get('to_start_waypoint_id', 0)], dtype=torch.int32, device=self.device),
                    'to_start_x': torch.tensor([to_start.get('x', 0.0)], dtype=torch.float32, device=self.device),
                    'to_start_y': torch.tensor([to_start.get('y', 0.0)], dtype=torch.float32, device=self.device),
                    'to_start_road_id': torch.tensor([to_start.get('road_id', 0)], dtype=torch.int32, device=self.device),
                    'to_start_lane_id': torch.tensor([to_start.get('lane_id', 0)], dtype=torch.int32, device=self.device),
                    'path_quad_ids': torch.tensor(path_quad_ids, dtype=torch.int32, device=self.device)
                }
                paths_list.append(path_dict)

            cross_data_dict[cross_id] = {
                'start_waypoint_ids': start_waypoint_ids,
                'start_x': start_x,
                'start_y': start_y,
                'start_road_id': start_road_id,
                'start_lane_id': start_lane_id,
                'end_waypoint_ids': end_waypoint_ids,
                'end_x': end_x,
                'end_y': end_y,
                'end_road_id': end_road_id,
                'end_lane_id': end_lane_id,
                'paths': paths_list
            }
        self.cross_data = cross_data_dict

        # ===================存global_w_lane_waypoints===================
        global_w_lane_waypoints_list = map_data.get('global_w_lane_waypoints', [])
        if global_w_lane_waypoints_list:
            # 提取所有字段并创建结构化张量
            num_waypoints = len(global_w_lane_waypoints_list)
            # 创建结构化张量来存储waypoint数据
            waypoint_data = {
                'x': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device),
                'y': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device),
                'z': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device),
                'direction_x': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device),
                'direction_y': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device),
                'routing_poly_id': torch.zeros(num_waypoints, dtype=torch.int32, device=self.device),
                    'carla_waypoint_info': {
                    'road_id': torch.zeros(num_waypoints, dtype=torch.int32, device=self.device),
                    'lane_id': torch.zeros(num_waypoints, dtype=torch.int32, device=self.device),
                    's': torch.zeros(num_waypoints, dtype=torch.float32, device=self.device)
                }
            }
            # 填充数据
            for i, wp in enumerate(global_w_lane_waypoints_list):
                waypoint_data['x'][i] = wp['x']
                waypoint_data['y'][i] = wp['y']
                waypoint_data['z'][i] = wp['z']
                direction = wp['direction']
                waypoint_data['direction_x'][i] = direction[0]
                waypoint_data['direction_y'][i] = direction[1]
                waypoint_data['routing_poly_id'][i] = wp.get('routing_poly_id', 0)
                # 处理carla_waypoint_info字段
                carla_info = wp.get('carla_waypoint_info', {})
                waypoint_data['carla_waypoint_info']['road_id'][i] = carla_info['road_id']
                waypoint_data['carla_waypoint_info']['lane_id'][i] = carla_info['lane_id']
                waypoint_data['carla_waypoint_info']['s'][i] = carla_info['s']
            self.global_w_lane_waypoints = waypoint_data
            self.lanes = self._group_waypoints_by_lane_gpu()
            
        # ===================创建waypoint_id到lane的映射===================
            # 收集所有waypoint_id并创建映射
            all_waypoint_ids = []
            all_road_ids = []
            all_lane_ids = []
            all_lane_indices = []
            for (road_id, lane_id), lane_indices in self.lanes.items():
                for wp_id in lane_indices:
                    all_waypoint_ids.append(wp_id.item())
                    all_road_ids.append(road_id)
                    all_lane_ids.append(lane_id)
                    all_lane_indices.append(lane_indices)
            # 计算所有lane_indices的最大长度
            max_lane_length = max(len(lane_indices) for lane_indices in all_lane_indices)
            # 创建定长张量存储所有lane_indices
            lane_indices_tensor = torch.full((len(all_lane_indices), max_lane_length), -1, dtype=torch.int32, device=self.device)
            lane_lengths = torch.zeros((len(all_lane_indices),), dtype=torch.int32, device=self.device)
            for i_fill, lane_indices in enumerate(all_lane_indices):
                lane_indices_tensor[i_fill, :len(lane_indices)] = lane_indices
                lane_lengths[i_fill] = len(lane_indices)
            self.waypoint_to_lane = {
                'waypoint_ids': torch.tensor(all_waypoint_ids, dtype=torch.int32, device=self.device),
                'road_ids': torch.tensor(all_road_ids, dtype=torch.int32, device=self.device),
                'lane_ids': torch.tensor(all_lane_ids, dtype=torch.int32, device=self.device),
                'lane_indices': all_lane_indices,  # 保持为列表，因为每个lane的waypoint数量不同
                'lane_indices_tensor': lane_indices_tensor,
                'lane_lengths': lane_lengths
            }
        # ===================存quads的信息===================
        quads_info = map_data.get('quads', [])
        if quads_info:
            num_quads = len(quads_info)
            quad_data = {
                'polyId': torch.zeros(num_quads, dtype=torch.int32, device=self.device),
                'road_id': torch.zeros(num_quads, dtype=torch.int32, device=self.device),
                'lane_id': torch.zeros(num_quads, dtype=torch.int32, device=self.device),
                'center_x': torch.zeros(num_quads, dtype=torch.float32, device=self.device),
                'center_y': torch.zeros(num_quads, dtype=torch.float32, device=self.device),
                'direction_x': torch.zeros(num_quads, dtype=torch.float32, device=self.device),
                'direction_y': torch.zeros(num_quads, dtype=torch.float32, device=self.device),
            }
            # 先批量收集所有顶点
            vertices_tensor = torch.zeros(num_quads, 4, 2, dtype=torch.float32, device=self.device)
            for i, quad in enumerate(quads_info):
                quad_data['polyId'][i] = quad['polyId']
                quad_data['road_id'][i] = quad['road_id']
                quad_data['lane_id'][i] = quad['lane_id']
                for v in range(4):
                    vertices_tensor[i, v, 0] = quad['vertices'][v]['x']
                    vertices_tensor[i, v, 1] = quad['vertices'][v]['y']
            # 并行计算center
            centers = vertices_tensor.mean(dim=1)  # [num_quads, 2]
            quad_data['center_x'] = centers[:, 0]
            quad_data['center_y'] = centers[:, 1]
            # 并行计算direct
            v0 = vertices_tensor[:, 0, :]
            v2 = vertices_tensor[:, 2, :]
            directions = v2 - v0
            norms = torch.norm(directions, dim=1, keepdim=True)
            # 避免除零
            safe_norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
            directions_normalized = directions / safe_norms
            # 对于零向量，替换为[1,0]
            zero_mask = (norms <= 1e-8).squeeze(1)
            directions_normalized[zero_mask] = torch.tensor([1.0, 0.0], device=self.device)
            quad_data['direction_x'] = directions_normalized[:, 0]
            quad_data['direction_y'] = directions_normalized[:, 1]
            self.quads_info = quad_data

        # ===================存quad_to_next_waypoint的信息===================
        quad_to_next_waypoint = map_data.get('quad_to_next_waypoint', {})
        # 将quad_to_next_waypoint字典转换为tensor映射关系
        if quad_to_next_waypoint:
            # 获取所有quad_id并排序
            quad_ids = sorted([int(k) for k in quad_to_next_waypoint.keys()])
            next_waypoint_values = [quad_to_next_waypoint[str(quad_id)] for quad_id in quad_ids]
            # 创建tensor映射关系
            self.quad_to_next_waypoint_quad_ids = torch.tensor(quad_ids, dtype=torch.int32, device=self.device)
            self.quad_to_next_waypoint_values = torch.tensor(next_waypoint_values, dtype=torch.int32, device=self.device)

        # ===================存quad_to_prev_waypoint的信息===================
        quad_to_prev_waypoint = map_data.get('quad_to_prev_waypoint', {})
        if quad_to_prev_waypoint:
            quad_ids = sorted([int(k) for k in quad_to_prev_waypoint.keys()])
            prev_waypoint_values = [quad_to_prev_waypoint[str(quad_id)] for quad_id in quad_ids]
            self.quad_to_prev_waypoint_quad_ids = torch.tensor(quad_ids, dtype=torch.int32, device=self.device)
            self.quad_to_prev_waypoint_values = torch.tensor(prev_waypoint_values, dtype=torch.int32, device=self.device)

        # ===================收集所有cross_data中的path_quad_ids===================
        all_path_quad_ids = []
        all_path_cross_ids = []
        all_path_indices = []
        for cross_id, cross_info in self.cross_data.items():
            paths = cross_info['paths']
            for path_idx, path_dict in enumerate(paths):
                path_quad_ids = path_dict['path_quad_ids']
                if path_quad_ids.numel() > 0:
                    all_path_quad_ids.append(path_quad_ids)
                    all_path_cross_ids.append(torch.full((path_quad_ids.shape[0],), int(cross_id), dtype=torch.int32, device=self.device))
                    all_path_indices.append(torch.full((path_quad_ids.shape[0],), path_idx, dtype=torch.int32, device=self.device))
        if all_path_quad_ids:
            # 合并所有path_quad_ids
            self.all_quad_ids_flat = torch.cat(all_path_quad_ids, dim=0)
            self.all_cross_ids_flat = torch.cat(all_path_cross_ids, dim=0)
            self.all_path_indices_flat = torch.cat(all_path_indices, dim=0)
            # 获取这些quad的中心点和方向
            self.cross_quad_centers_x = self.quads_info['center_x'][self.all_quad_ids_flat]
            self.cross_quad_centers_y = self.quads_info['center_y'][self.all_quad_ids_flat]
            self.cross_quad_directions_x = self.quads_info['direction_x'][self.all_quad_ids_flat]
            self.cross_quad_directions_y = self.quads_info['direction_y'][self.all_quad_ids_flat]

        # ===================预计算不在filtered_quad_indices内的quad的最近邻===================
        # 获取所有不在filtered_quad_indices内的quad_id
        all_quad_ids = torch.arange(self.quads_info['center_x'].shape[0], device=self.device)
        non_filtered_mask = ~torch.isin(all_quad_ids, self.filtered_quad_indices)
        non_filtered_quad_ids = all_quad_ids[non_filtered_mask]
        if non_filtered_quad_ids.numel() > 0 and hasattr(self, 'all_quad_ids_flat') and self.all_quad_ids_flat.numel() > 0:
            # 获取这些quad的中心点和方向
            non_filtered_centers_x = self.quads_info['center_x'][non_filtered_quad_ids]
            non_filtered_centers_y = self.quads_info['center_y'][non_filtered_quad_ids]
            non_filtered_directions_x = self.quads_info['direction_x'][non_filtered_quad_ids]
            non_filtered_directions_y = self.quads_info['direction_y'][non_filtered_quad_ids]
            # 预计算最近邻信息
            self.nearest_neighbor_info = {}
            for i, quad_id in enumerate(non_filtered_quad_ids):
                start_center_x = non_filtered_centers_x[i]
                start_center_y = non_filtered_centers_y[i]
                start_dir_x = non_filtered_directions_x[i]
                start_dir_y = non_filtered_directions_y[i]
                # 计算距离
                distances = torch.sqrt((self.cross_quad_centers_x - start_center_x)**2 + (self.cross_quad_centers_y - start_center_y)**2)
                # 计算方向向量夹角（使用点积）
                start_dir_norm = torch.sqrt(start_dir_x**2 + start_dir_y**2)
                quad_dir_norms = torch.sqrt(self.cross_quad_directions_x**2 + self.cross_quad_directions_y**2)
                # 避免除零
                safe_start_norm = torch.where(start_dir_norm > 1e-8, start_dir_norm, torch.ones_like(start_dir_norm))
                safe_quad_norms = torch.where(quad_dir_norms > 1e-8, quad_dir_norms, torch.ones_like(quad_dir_norms))
                # 归一化方向向量
                start_dir_x_norm = start_dir_x / safe_start_norm
                start_dir_y_norm = start_dir_y / safe_start_norm
                quad_dir_x_norm = self.cross_quad_directions_x / safe_quad_norms
                quad_dir_y_norm = self.cross_quad_directions_y / safe_quad_norms
                # 计算点积
                dot_products = start_dir_x_norm * quad_dir_x_norm + start_dir_y_norm * quad_dir_y_norm
                # 计算夹角（弧度）
                angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
                angles_deg = angles * 180 / torch.pi
                # 筛选方向夹角在90度以内的quad
                angle_mask = angles_deg <= 90.0
                if angle_mask.any():
                    # 在符合条件的quad中找到距离最近的
                    valid_distances = torch.where(angle_mask, distances, torch.full_like(distances, float('inf')))
                    nearest_idx = torch.argmin(valid_distances)
                    if valid_distances[nearest_idx] < float('inf'):
                        # 找到最近的quad_neighbor
                        nearest_quad_id = self.all_quad_ids_flat[nearest_idx]
                        nearest_cross_id = self.all_cross_ids_flat[nearest_idx]
                        nearest_path_idx = self.all_path_indices_flat[nearest_idx]
                        # 获取对应的path_dict
                        cross_info = self.cross_data[nearest_cross_id.item()]
                        path_dict = cross_info['paths'][nearest_path_idx.item()]
                        # 找到nearest_quad_id在path_quad_ids中的位置
                        path_quad_ids = path_dict['path_quad_ids']
                        quad_positions = torch.where(path_quad_ids == nearest_quad_id)[0]
                        if quad_positions.numel() > 0:
                            quad_position = quad_positions[0]
                            # 保留从nearest_quad_id到path_quad_ids末尾的所有quad_ids（用于情况2）
                            remaining_quad_ids = path_quad_ids[quad_position:]
                            # 保留从path_quad_ids开头到nearest_quad_id的所有quad_ids（用于情况4）
                            path_start_to_nearest_quad_ids = path_quad_ids[:quad_position+1]
                            # 存储预计算的信息
                            self.nearest_neighbor_info[quad_id.item()] = {
                                'nearest_quad_id': nearest_quad_id.item(),
                                'nearest_cross_id': nearest_cross_id.item(),
                                'nearest_path_idx': nearest_path_idx.item(),
                                'to_start_waypoint_id': path_dict['to_start_waypoint_id'].item(),
                                'to_start_road_id': path_dict['to_start_road_id'].item(),
                                'to_start_lane_id': path_dict['to_start_lane_id'].item(),
                                'from_end_waypoint_id': path_dict['from_end_waypoint_id'].item(),
                                'from_end_road_id': path_dict['from_end_road_id'].item(),
                                'from_end_lane_id': path_dict['from_end_lane_id'].item(),
                                'remaining_quad_ids': remaining_quad_ids.tolist(),
                                'path_start_to_nearest_quad_ids': path_start_to_nearest_quad_ids.tolist(),
                                'path_quad_ids': path_dict['path_quad_ids'].tolist(),
                                'distance': valid_distances[nearest_idx].item(),
                                'angle': angles_deg[nearest_idx].item()
                            }           
        else:
            # 如果没有cross数据或没有非filtered的quad，创建空的nearest_neighbor_info
            self.nearest_neighbor_info = {}

        # ===================预处理cross匹配数据===================
        # 批量匹配cross的end_waypoint_ids（合并所有cross）- 在初始化时预处理
        all_end_ids_list = []
        all_end_cids_list = []

        for cid, info in self.cross_data.items():
            ends = info['end_waypoint_ids']
            if isinstance(ends, torch.Tensor) and ends.numel() > 0:
                all_end_ids_list.append(ends)
                all_end_cids_list.append(torch.full((ends.shape[0],), int(cid), dtype=torch.int32, device=self.device))
        
        if len(all_end_ids_list) > 0:
            self.all_end_ids = torch.cat(all_end_ids_list, dim=0)
            self.all_end_cids = torch.cat(all_end_cids_list, dim=0)
        else:
            self.all_end_ids = torch.empty(0, dtype=torch.int32, device=self.device)
            self.all_end_cids = torch.empty(0, dtype=torch.int32, device=self.device)

        # 批量匹配cross的start_waypoint_ids（合并所有cross）- 在初始化时预处理
        all_start_ids_list = []
        all_start_cids_list = []

        for cid, info in self.cross_data.items():
            starts = info['start_waypoint_ids']
            if isinstance(starts, torch.Tensor) and starts.numel() > 0:
                all_start_ids_list.append(starts)
                all_start_cids_list.append(torch.full((starts.shape[0],), int(cid), dtype=torch.int32, device=self.device))
        if len(all_start_ids_list) > 0:
            self.all_start_ids = torch.cat(all_start_ids_list, dim=0)
            self.all_start_cids = torch.cat(all_start_cids_list, dim=0)
        else:
            self.all_start_ids = torch.empty(0, dtype=torch.int32, device=self.device)
            self.all_start_cids = torch.empty(0, dtype=torch.int32, device=self.device)

    def plan_path(self, start_quad_id: torch.Tensor, goal_quad_id: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            start_quad_id: 起点quad_id(B,M,1),为tensor
            goal_quad_id: 终点quad_id(B,M,1),为tensor
        Returns:
            path: 路径(B,M,max_path_length,2)
        '''
        # 初始化规划的路径：
        # path = (B,M,512,2)
        path = torch.full((start_quad_id.shape[0], start_quad_id.shape[1], 512, 2), -1, dtype=torch.float32, device=self.device)
        # 确定起始点和目标点的类型（是否在filtered_quad_indices内）
        # 使用GPU张量进行快速查询
        # 重塑张量以便进行广播比较
        start_quad_flat = start_quad_id.view(-1)  # (B*M,)
        goal_quad_flat = goal_quad_id.view(-1)    # (B*M,)
        # 查询哪些quad_id在filtered_quad_indices内
        start_in_filtered_mask = torch.isin(start_quad_flat, self.filtered_quad_indices)
        goal_in_filtered_mask = torch.isin(goal_quad_flat, self.filtered_quad_indices)
        # 获取在filtered_quad_indices内的具体值

        # 初始化结果张量 - 与start_in_filtered_mask长度一致
        batch_size = start_quad_flat.shape[0]
        start_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        end_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        
        # 使用列表存储有效的lane_waypoints，避免-1填充
        # lane_waypoints_list用于存储起点的lane waypoints（情况1和情况2）
        lane_waypoints_list = [[] for _ in range(batch_size)]
        # goal_lane_waypoints_list用于存储终点的lane waypoints（情况3和情况4）
        goal_lane_waypoints_list = [[] for _ in range(batch_size)]
        
        # 情况1：处理起点在filtered_quad_indices内的情况
        if start_in_filtered_mask.any():
            # 获取满足条件的start_quad索引
            valid_start_indices = torch.where(start_in_filtered_mask)[0]
            valid_start_quads = start_quad_flat[valid_start_indices]
            # 使用广播操作进行批量查找
            # 1. 广播查找quad_to_next_waypoint映射
            quad_expanded = valid_start_quads.unsqueeze(1)  # (num_valid, 1)
            quad_ids_expanded = self.quad_to_next_waypoint_quad_ids.unsqueeze(0)  # (1, num_quad_ids)
            quad_matches = (quad_expanded == quad_ids_expanded)  # (num_valid, num_quad_ids)
            
            # 获取匹配的索引和对应的waypoint_ids
            matched_indices = torch.where(quad_matches)
            if len(matched_indices[0]) > 0:
                valid_quad_indices = matched_indices[0]  # 在valid_start_quads中的索引
                quad_to_next_indices = matched_indices[1]  # 在quad_to_next_waypoint_quad_ids中的索引
                waypoint_ids = self.quad_to_next_waypoint_values[quad_to_next_indices]#获取了所有起始quad的下一个waypoint

                # 2. 使用预创建的waypoint_to_lane映射进行批量查找（GPU向量化）
                waypoint_ids_expanded = waypoint_ids.unsqueeze(1)  # (N, 1)
                map_ids_expanded = self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)  # (1, K)
                waypoint_matches = (waypoint_ids_expanded == map_ids_expanded)  # (N, K)

                # 3. 批量处理匹配（GPU向量化）
                waypoint_matches_int = waypoint_matches.int()        # 将布尔张量转换为整数张量
                match_indices = waypoint_matches_int.argmax(dim=1)   # 获取每个waypoint在waypoint_to_lane中匹配的索引
                valid_matches = waypoint_matches.any(dim=1)          # 标记哪些waypoint在waypoint_to_lane中找到了有效值
                valid_wp_idx = torch.where(valid_matches)[0]         # 获取所有有效匹配的waypoint的索引

                if valid_wp_idx.numel() > 0:
                    sel_match_idx = match_indices[valid_wp_idx]
                    roads_found = self.waypoint_to_lane['road_ids'][sel_match_idx]
                    lanes_found = self.waypoint_to_lane['lane_ids'][sel_match_idx]
                    # 使用预处理的lane索引张量
                    lane_indices_tensor = self.waypoint_to_lane['lane_indices_tensor'][sel_match_idx]
                    lane_lengths = self.waypoint_to_lane['lane_lengths'][sel_match_idx]

                    # waypoint在对应lane中的位置（GPU并行查找）
                    wp_to_find = waypoint_ids[valid_wp_idx]
                    eq = (wp_to_find.unsqueeze(1) == lane_indices_tensor)  # (Nv, L)
                    eq_int = eq.int()
                    pos = eq_int.argmax(dim=1)           # 第一个匹配位置
                    has_pos = eq.any(dim=1)
                    pos = torch.where(has_pos, pos, torch.full_like(pos, -1))

                    # 末端waypoint（每条lane最后一个有效点）
                    last_idx = torch.clamp(lane_lengths - 1, min=0)
                    final_wp_ids = lane_indices_tensor[torch.arange(lane_indices_tensor.shape[0], device=self.device), last_idx]
                    final_wp_ids = torch.where((lane_lengths > 0) & (pos >= 0), final_wp_ids, torch.full_like(final_wp_ids, -1))
                    
                    # 使用预处理的cross匹配数据
                    if self.all_end_ids.numel() > 0:
                        m = (final_wp_ids.unsqueeze(1) == self.all_end_ids.unsqueeze(0))  # (Nv, T)
                        any_m = m.any(dim=1)
                        m_int = m.int()
                        idx = m_int.argmax(dim=1)
                        cross_ids_found = torch.where(any_m, self.all_end_cids[idx], torch.full_like(final_wp_ids, -1))
                    else:
                        cross_ids_found = torch.full_like(final_wp_ids, -1)
                    
                    # 回写 start_ids（仅对有效项）
                    orig_idx = valid_start_indices[valid_quad_indices[valid_wp_idx]]
                    write_mask = (cross_ids_found != -1) & (pos >= 0)
                    if write_mask.any():
                        idx_w = orig_idx[write_mask]
                        start_ids[idx_w, 0] = cross_ids_found[write_mask]
                        start_ids[idx_w, 1] = roads_found[write_mask]
                        start_ids[idx_w, 2] = lanes_found[write_mask]
                        # 填充lane_waypoints（只保存有效值）
                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        for j in range(sel_indices.shape[0]):
                            # 从预处理的张量中获取lane数据
                            lane_data = lane_indices_tensor[sel_indices[j]]
                            sp = sel_pos[j]
                            # 获取从位置sp到末尾的有效waypoint
                            seq_from = lane_data[sp:]
                            # 只保存有效的waypoint值，不填充-1
                            valid_waypoints = seq_from[seq_from >= 0]  # 过滤掉-1值
                            if valid_waypoints.numel() > 0:
                                # 将waypoint索引转换为坐标序列 (x,y)
                                waypoint_coords = self.get_waypoint_coords(valid_waypoints)
                                lane_waypoints_list[idx_w[j]] = waypoint_coords  

        # 情况2：处理起点不在filtered_quad_indices内的情况
        if (~start_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的start_quad
            invalid_start_indices = torch.where(~start_in_filtered_mask)[0]
            invalid_start_quads = start_quad_flat[invalid_start_indices]
            if invalid_start_quads.numel() > 0:
                # 使用预计算的最近邻信息
                for i, start_idx in enumerate(invalid_start_indices):
                    quad_id = invalid_start_quads[i].item()
                    # 从预计算的信息中获取最近邻数据
                    if quad_id in self.nearest_neighbor_info:
                        neighbor_info = self.nearest_neighbor_info[quad_id]
                        if neighbor_info['nearest_quad_id'] != -1:
                            # 更新start_ids
                            start_ids[start_idx, 0] = neighbor_info['nearest_cross_id']
                            start_ids[start_idx, 1] = neighbor_info['to_start_road_id']
                            start_ids[start_idx, 2] = neighbor_info['to_start_lane_id']
                            # 填充lane_waypoints（使用预计算的path_start_to_nearest_quad_ids）
                            remaining_quad_ids = neighbor_info['remaining_quad_ids']
                            if remaining_quad_ids:
                                # 将quad_ids转换为坐标序列 (x,y)
                                remaining_quad_tensor = torch.tensor(remaining_quad_ids, dtype=torch.int32, device=self.device)
                                quad_coords = self.get_quad_centers(remaining_quad_tensor)
                                lane_waypoints_list[start_idx] = quad_coords
                        else:
                            print(f'警告: quad_id={quad_id} 没有找到有效的最近邻')
                    else:
                        print(f'警告: quad_id={quad_id} 不在预计算的最近邻信息中')

        # 情况3：处理终点在filtered_quad_indices内的情况（类似逻辑）
        if goal_in_filtered_mask.any():
            # 获取满足条件的goal_quad索引
            valid_goal_indices = torch.where(goal_in_filtered_mask)[0]
            valid_goal_quads = goal_quad_flat[valid_goal_indices]
            # 使用广播操作进行批量查找
            # 1. 广播查找quad_to_prev_waypoint映射
            quad_expanded = valid_goal_quads.unsqueeze(1)  # (num_valid, 1)
            quad_ids_expanded = self.quad_to_prev_waypoint_quad_ids.unsqueeze(0)  # (1, num_quad_ids)
            quad_matches = (quad_expanded == quad_ids_expanded)  # (num_valid, num_quad_ids)
            
            # 获取匹配的索引和对应的waypoint_ids
            matched_indices = torch.where(quad_matches)
            if len(matched_indices[0]) > 0:
                valid_quad_indices = matched_indices[0]  # 在valid_goal_quads中的索引
                quad_to_prev_indices = matched_indices[1]  # 在quad_to_prev_waypoint_quad_ids中的索引
                waypoint_ids = self.quad_to_prev_waypoint_values[quad_to_prev_indices]  # 获取了所有终点quad的上一个waypoint

                # 2. 使用预创建的waypoint_to_lane映射进行批量查找（GPU向量化）
                waypoint_ids_expanded = waypoint_ids.unsqueeze(1)  # (N, 1)
                map_ids_expanded = self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)  # (1, K)
                waypoint_matches = (waypoint_ids_expanded == map_ids_expanded)  # (N, K)

                # 3. 批量处理匹配（GPU向量化）
                waypoint_matches_int = waypoint_matches.int()        # 将布尔张量转换为整数张量
                match_indices = waypoint_matches_int.argmax(dim=1)   # 获取每个waypoint在waypoint_to_lane中匹配的索引
                valid_matches = waypoint_matches.any(dim=1)          # 标记哪些waypoint在waypoint_to_lane中找到了有效值
                valid_wp_idx = torch.where(valid_matches)[0]         # 获取所有有效匹配的waypoint的索引

                if valid_wp_idx.numel() > 0:
                    sel_match_idx = match_indices[valid_wp_idx]
                    roads_found = self.waypoint_to_lane['road_ids'][sel_match_idx]
                    lanes_found = self.waypoint_to_lane['lane_ids'][sel_match_idx]

                    # 使用预处理的lane索引张量
                    lane_indices_tensor = self.waypoint_to_lane['lane_indices_tensor'][sel_match_idx]
                    lane_lengths = self.waypoint_to_lane['lane_lengths'][sel_match_idx]
                    
                    # waypoint在对应lane中的位置（GPU并行查找）
                    wp_to_find = waypoint_ids[valid_wp_idx]
                    eq = (wp_to_find.unsqueeze(1) == lane_indices_tensor)  # (Nv, L)
                    eq_int = eq.int()
                    pos = eq_int.argmax(dim=1)           # 第一个匹配位置
                    has_pos = eq.any(dim=1)
                    pos = torch.where(has_pos, pos, torch.full_like(pos, -1))

                    # 首端waypoint（每条lane第一个有效点）
                    first_wp_ids = lane_indices_tensor[torch.arange(lane_indices_tensor.shape[0], device=self.device), 0]
                    first_wp_ids = torch.where((lane_lengths > 0) & (pos >= 0), first_wp_ids, torch.full_like(first_wp_ids, -1))
                    
                    # 使用预处理的cross匹配数据
                    if self.all_start_ids.numel() > 0:
                        m = (first_wp_ids.unsqueeze(1) == self.all_start_ids.unsqueeze(0))  # (Nv, T)
                        any_m = m.any(dim=1)
                        m_int = m.int()
                        idx = m_int.argmax(dim=1)
                        cross_ids_found = torch.where(any_m, self.all_start_cids[idx], torch.full_like(first_wp_ids, -1))
                    else:
                        cross_ids_found = torch.full_like(first_wp_ids, -1)
                    
                    # 回写 end_ids（仅对有效项）
                    orig_idx = valid_goal_indices[valid_quad_indices[valid_wp_idx]]
                    write_mask = (cross_ids_found != -1) & (pos >= 0)
                    if write_mask.any():
                        idx_w = orig_idx[write_mask]
                        end_ids[idx_w, 0] = cross_ids_found[write_mask]
                        end_ids[idx_w, 1] = roads_found[write_mask]
                        end_ids[idx_w, 2] = lanes_found[write_mask]
                        # 填充goal_lane_waypoints（只保存有效值）
                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        for j in range(sel_indices.shape[0]):
                            # 从预处理的张量中获取lane数据
                            lane_data = lane_indices_tensor[sel_indices[j]]
                            sp = sel_pos[j]
                            seq_from = lane_data[:sp+1]  # 从开始到当前位置（包含当前位置）
                            # 只保存有效的waypoint值，不填充-1
                            valid_waypoints = seq_from[seq_from >= 0]  # 过滤掉-1值
                            if valid_waypoints.numel() > 0:
                                # 将waypoint索引转换为坐标序列 (x,y)
                                waypoint_coords = self.get_waypoint_coords(valid_waypoints)
                                goal_lane_waypoints_list[idx_w[j]] = waypoint_coords

        # 情况4：处理终点不在filtered_quad_indices内的情况
        if (~goal_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的goal_quad
            invalid_goal_indices = torch.where(~goal_in_filtered_mask)[0]
            invalid_goal_quads = goal_quad_flat[invalid_goal_indices]
            if invalid_goal_quads.numel() > 0:
                # 使用预计算的最近邻信息
                for i, goal_idx in enumerate(invalid_goal_indices):
                    quad_id = invalid_goal_quads[i].item()
                    # 从预计算的信息中获取最近邻数据
                    if quad_id in self.nearest_neighbor_info:
                        neighbor_info = self.nearest_neighbor_info[quad_id]
                        if neighbor_info['nearest_quad_id'] != -1:
                            # 更新goal_ids
                            end_ids[goal_idx, 0] = neighbor_info['nearest_cross_id']
                            end_ids[goal_idx, 1] = neighbor_info['from_end_road_id']
                            end_ids[goal_idx, 2] = neighbor_info['from_end_lane_id']
                            # 填充goal_lane_waypoints（使用预计算的path_start_to_nearest_quad_ids）
                            path_start_to_nearest_quad_ids = neighbor_info['path_start_to_nearest_quad_ids']
                            if path_start_to_nearest_quad_ids:
                                # 将quad_ids转换为坐标序列 (x,y)
                                path_quad_tensor = torch.tensor(path_start_to_nearest_quad_ids, dtype=torch.int32, device=self.device)
                                quad_coords = self.get_quad_centers(path_quad_tensor)
                                goal_lane_waypoints_list[goal_idx] = quad_coords
                        else:
                            print(f'警告: quad_id={quad_id} 没有找到有效的最近邻')
                    else:
                        print(f'警告: quad_id={quad_id} 不在预计算的最近邻信息中')
        




#=======================查找工具函数=======================

    def get_waypoint_coords(self, indices: torch.Tensor) -> torch.Tensor:
        """获取指定索引的waypoint坐标"""
        if self.global_w_lane_waypoints is None:
            return torch.zeros(len(indices), 2, device=self.device)
        x = self.global_w_lane_waypoints['x'][indices]
        y = self.global_w_lane_waypoints['y'][indices]
        return torch.stack([x, y], dim=1)
    
    def get_waypoint_direction(self, indices: torch.Tensor) -> torch.Tensor:
        """获取指定索引的waypoint方向"""
        if self.global_w_lane_waypoints is None:
            return torch.zeros(len(indices), 2, device=self.device)
        dx = self.global_w_lane_waypoints['direction_x'][indices]
        dy = self.global_w_lane_waypoints['direction_y'][indices]
        return torch.stack([dx, dy], dim=1)
    
    def get_waypoint_carla_info(self, indices: torch.Tensor) -> dict:
        """获取指定索引的waypoint的carla信息"""
        if self.global_w_lane_waypoints is None:
            return {
                'road_id': torch.zeros(len(indices), dtype=torch.int32, device=self.device),
                'lane_id': torch.zeros(len(indices), dtype=torch.int32, device=self.device),
                's': torch.zeros(len(indices), dtype=torch.float32, device=self.device)
            }
        return {
            'road_id': self.global_w_lane_waypoints['carla_waypoint_info']['road_id'][indices],
            'lane_id': self.global_w_lane_waypoints['carla_waypoint_info']['lane_id'][indices],
            's': self.global_w_lane_waypoints['carla_waypoint_info']['s'][indices]
        }
    
    def find_waypoints_by_road_lane(self, road_id: int, lane_id: int) -> torch.Tensor:
        """根据road_id和lane_id查找waypoints"""
        if self.global_w_lane_waypoints is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        road_mask = self.global_w_lane_waypoints['carla_waypoint_info']['road_id'] == road_id
        lane_mask = self.global_w_lane_waypoints['carla_waypoint_info']['lane_id'] == lane_id
        combined_mask = road_mask & lane_mask
        return torch.where(combined_mask)[0]
    
    def get_quad_centers(self, quad_ids: torch.Tensor) -> torch.Tensor:
        """获取指定quad_id的中心点坐标"""
        if not hasattr(self, 'quads_info') or self.quads_info is None:
            return torch.zeros(len(quad_ids), 2, device=self.device)
        # 获取quad的中心点坐标
        center_x = self.quads_info['center_x'][quad_ids]
        center_y = self.quads_info['center_y'][quad_ids]
        return torch.stack([center_x, center_y], dim=1)

#=======================预处理函数=======================
    def _group_waypoints_by_lane_gpu(self):
        """
        按车道分组并排序航点，基于GPU张量self.global_w_lane_waypoints。
        返回: lanes字典，key为(road_id, lane_id)，value为waypoint索引的Tensor（已按s排序，且过滤掉距离太短的车道）
        """
        if self.global_w_lane_waypoints is None:
            return {}
        road_ids = self.global_w_lane_waypoints['carla_waypoint_info']['road_id']
        lane_ids = self.global_w_lane_waypoints['carla_waypoint_info']['lane_id']
        s_vals = self.global_w_lane_waypoints['carla_waypoint_info']['s']
        xs = self.global_w_lane_waypoints['x']
        ys = self.global_w_lane_waypoints['y']
        lane_keys = road_ids * 10000 + lane_ids #制造一个同road_id和lane_id的标识符，保证唯一
        unique_keys = torch.unique(lane_keys)
        lanes = {}
        for key in unique_keys:
            mask = (lane_keys == key)
            indices = torch.where(mask)[0]
            s_in_lane = s_vals[indices]
            is_reverse = lane_ids[indices[0]] < 0
            sorted_idx = torch.argsort(s_in_lane, descending=bool(is_reverse))
            sorted_indices = indices[sorted_idx]
            # 过滤距离
            if len(sorted_indices) >= 2:
                x0, y0 = xs[sorted_indices[0]], ys[sorted_indices[0]]
                x1, y1 = xs[sorted_indices[-1]], ys[sorted_indices[-1]]
                dist = torch.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if dist >= 10:
                    road_id = road_ids[sorted_indices[0]].item()
                    lane_id = lane_ids[sorted_indices[0]].item()
                    lanes[(road_id, lane_id)] = sorted_indices
        return lanes

if __name__ == "__main__":
    path_planner = PathPlanner(map_path='maps/processed_map_Town01_stitched.json', device=torch.device('cuda'))
    # 测试参数
    B, M = 1, 1
    # 生成随机的quad_id张量 (B, M, 1)
    # 从path_planner的quads_info中获取有效的quad_id范围
    if hasattr(path_planner, 'quads_info') and path_planner.quads_info is not None:
        # 使用实际的quad_id范围
        valid_quad_ids = path_planner.quads_info['polyId'].cpu().numpy()
        min_quad_id = valid_quad_ids.min()
        max_quad_id = valid_quad_ids.max()
        # 随机选择quad_id
        import random
        start_quad_ids = []
        goal_quad_ids = []
        for b in range(B):
            batch_start = []
            batch_goal = []
            for m in range(M):
                # 随机选择quad_id
                start_quad = random.choice(valid_quad_ids)
                goal_quad = random.choice(valid_quad_ids)
                batch_start.append([start_quad])
                batch_goal.append([goal_quad])
            start_quad_ids.append(batch_start)
            goal_quad_ids.append(batch_goal)
        # 转换为tensor
        start_quad_tensor = torch.tensor(start_quad_ids, dtype=torch.int32, device=path_planner.device)
        goal_quad_tensor = torch.tensor(goal_quad_ids, dtype=torch.int32, device=path_planner.device)
    # 测试plan_path方法
    try:
        result = path_planner.plan_path(start_quad_tensor, goal_quad_tensor)
        #print(start_quad_tensor.cpu().numpy(), '→' , goal_quad_tensor.cpu().numpy())
        print(f"返回结果: {result.shape if hasattr(result, 'shape') else '无返回值'}")
    except Exception as e:
        print(f"plan_path执行出错: {e}")
        import traceback
        traceback.print_exc()
    

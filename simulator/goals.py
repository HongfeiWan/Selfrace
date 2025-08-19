import torch
import json
import numpy as np
import time

class PathPlanner:
    def __init__(self,map_path: str, device: torch.device):
        print(f"==========PathPlanner init==========")
        start_time = time.time()
        self.device = device
        # 如果提供了map，自动加载数据
        if map_path is not None:
            cross_data_path = map_path.replace('processed_map_', 'cross_data_processed_map_')
            # 加载cross数据
            load_start = time.time()
            with open(cross_data_path, 'r', encoding='utf-8') as f:
                cross_data = json.load(f)
            with open(map_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            load_time = time.time() - load_start
            print(f"地图数据加载耗时: {load_time:.4f}秒")
        else:
            raise ValueError("map_path is required")
        # ===================存直路的quad_id==============================
        filtered_start = time.time()
        self.filtered_quad_indices = torch.tensor(cross_data.get('filtered_quad_indices', []), dtype=torch.int32, device=self.device)
        filtered_time = time.time() - filtered_start
        print(f"直路quad_id处理耗时: {filtered_time:.4f}秒")
        # ===================存cross的信息================================
        cross_start = time.time()
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
        cross_time = time.time() - cross_start
        print(f"cross信息处理耗时: {cross_time:.4f}秒")

        # ===================存global_w_lane_waypoints===================
        waypoints_start = time.time()
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
            waypoints_time = time.time() - waypoints_start
            print(f"waypoints处理耗时: {waypoints_time:.4f}秒")
            
        # ===================创建waypoint_id到lane的映射===================
        waypoint_mapping_start = time.time()
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
        waypoint_mapping_time = time.time() - waypoint_mapping_start
        print(f"waypoint映射处理耗时: {waypoint_mapping_time:.4f}秒")
        # ===================存quads的信息===================
        quads_start = time.time()
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
        quads_time = time.time() - quads_start
        print(f"quads信息处理耗时: {quads_time:.4f}秒")

        # ===================存quad_to_next_waypoint的信息===================
        quad_next_start = time.time()
        quad_to_next_waypoint = map_data.get('quad_to_next_waypoint', {})
        # 将quad_to_next_waypoint字典转换为tensor映射关系
        if quad_to_next_waypoint:
            # 获取所有quad_id并排序
            quad_ids = sorted([int(k) for k in quad_to_next_waypoint.keys()])
            next_waypoint_values = [quad_to_next_waypoint[str(quad_id)] for quad_id in quad_ids]
            # 创建tensor映射关系
            self.quad_to_next_waypoint_quad_ids = torch.tensor(quad_ids, dtype=torch.int32, device=self.device)
            self.quad_to_next_waypoint_values = torch.tensor(next_waypoint_values, dtype=torch.int32, device=self.device)
        quad_next_time = time.time() - quad_next_start
        print(f"quad_to_next_waypoint处理耗时: {quad_next_time:.4f}秒")

        # ===================存quad_to_prev_waypoint的信息===================
        quad_prev_start = time.time()
        quad_to_prev_waypoint = map_data.get('quad_to_prev_waypoint', {})
        if quad_to_prev_waypoint:
            quad_ids = sorted([int(k) for k in quad_to_prev_waypoint.keys()])
            prev_waypoint_values = [quad_to_prev_waypoint[str(quad_id)] for quad_id in quad_ids]
            self.quad_to_prev_waypoint_quad_ids = torch.tensor(quad_ids, dtype=torch.int32, device=self.device)
            self.quad_to_prev_waypoint_values = torch.tensor(prev_waypoint_values, dtype=torch.int32, device=self.device)
        quad_prev_time = time.time() - quad_prev_start
        print(f"quad_to_prev_waypoint处理耗时: {quad_prev_time:.4f}秒")

        # ===================收集所有cross_data中的path_quad_ids===================
        path_collect_start = time.time()
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
        path_collect_time = time.time() - path_collect_start
        print(f"path_quad_ids收集处理耗时: {path_collect_time:.4f}秒")

        # ===================预计算不在filtered_quad_indices内的quad的最近邻===================
        nearest_neighbor_start = time.time()
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
            # 批量计算距离矩阵 (num_non_filtered, num_cross_quads)
            # 使用广播计算所有距离
            centers_diff_x = non_filtered_centers_x.unsqueeze(1) - self.cross_quad_centers_x.unsqueeze(0)  # (num_non_filtered, num_cross_quads)
            centers_diff_y = non_filtered_centers_y.unsqueeze(1) - self.cross_quad_centers_y.unsqueeze(0)  # (num_non_filtered, num_cross_quads)
            distances = torch.sqrt(centers_diff_x**2 + centers_diff_y**2)  # (num_non_filtered, num_cross_quads)
            # 批量计算方向向量夹角
            # 归一化方向向量
            start_dir_norms = torch.sqrt(non_filtered_directions_x**2 + non_filtered_directions_y**2)  # (num_non_filtered,)
            quad_dir_norms = torch.sqrt(self.cross_quad_directions_x**2 + self.cross_quad_directions_y**2)  # (num_cross_quads,)
            # 避免除零
            safe_start_norms = torch.where(start_dir_norms > 1e-8, start_dir_norms, torch.ones_like(start_dir_norms))
            safe_quad_norms = torch.where(quad_dir_norms > 1e-8, quad_dir_norms, torch.ones_like(quad_dir_norms))
            # 归一化方向向量
            start_dir_x_norm = non_filtered_directions_x / safe_start_norms  # (num_non_filtered,)
            start_dir_y_norm = non_filtered_directions_y / safe_start_norms  # (num_non_filtered,)
            quad_dir_x_norm = self.cross_quad_directions_x / safe_quad_norms  # (num_cross_quads,)
            quad_dir_y_norm = self.cross_quad_directions_y / safe_quad_norms  # (num_cross_quads,)
            # 计算点积矩阵
            dot_products = (start_dir_x_norm.unsqueeze(1) * quad_dir_x_norm.unsqueeze(0) + 
                           start_dir_y_norm.unsqueeze(1) * quad_dir_y_norm.unsqueeze(0))  # (num_non_filtered, num_cross_quads)
            # 计算夹角（弧度）
            angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
            angles_deg = angles * 180 / torch.pi
            # 筛选方向夹角在90度以内的quad
            angle_mask = angles_deg <= 90.0  # (num_non_filtered, num_cross_quads)
            # 在符合条件的quad中找到距离最近的
            valid_distances = torch.where(angle_mask, distances, torch.full_like(distances, float('inf')))
            nearest_indices = torch.argmin(valid_distances, dim=1)  # (num_non_filtered,)
            min_distances = torch.gather(valid_distances, 1, nearest_indices.unsqueeze(1)).squeeze(1)  # (num_non_filtered,)
            # 筛选有效的最近邻（距离不是无穷大）
            valid_mask = min_distances < float('inf')
            valid_non_filtered_indices = torch.where(valid_mask)[0]
            if valid_non_filtered_indices.numel() > 0:
                # 获取有效的最近邻信息
                valid_nearest_indices = nearest_indices[valid_non_filtered_indices]
                valid_quad_ids = non_filtered_quad_ids[valid_non_filtered_indices]
                valid_min_distances = min_distances[valid_non_filtered_indices]
                valid_angles = torch.gather(angles_deg[valid_non_filtered_indices], 1, valid_nearest_indices.unsqueeze(1)).squeeze(1)
                # 获取对应的cross信息
                valid_nearest_quad_ids = self.all_quad_ids_flat[valid_nearest_indices]
                valid_nearest_cross_ids = self.all_cross_ids_flat[valid_nearest_indices]
                valid_nearest_path_indices = self.all_path_indices_flat[valid_nearest_indices]
                # 预分配张量存储所有最近邻信息
                num_valid = valid_non_filtered_indices.shape[0]# 有效quad的数量
                max_path_length = max(len(self.cross_data[cross_id.item()]['paths'][path_idx.item()]['path_quad_ids']) 
                                    for cross_id, path_idx in zip(valid_nearest_cross_ids, valid_nearest_path_indices))# 最大路径长度
                # 创建张量存储结构
                self.nearest_neighbor_tensors = {
                    'quad_ids': valid_quad_ids,  # (num_valid,)
                    'nearest_quad_ids': valid_nearest_quad_ids,  # (num_valid,)
                    'nearest_cross_ids': valid_nearest_cross_ids,  # (num_valid,)
                    'nearest_path_indices': valid_nearest_path_indices,  # (num_valid,)
                    'distances': valid_min_distances,  # (num_valid,)
                    'angles': valid_angles,  # (num_valid,)
                    'to_start_waypoint_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'to_start_road_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'to_start_lane_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'from_end_waypoint_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'from_end_road_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'from_end_lane_ids': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'remaining_quad_ids': torch.full((num_valid, max_path_length), -1, dtype=torch.int32, device=self.device),  # (num_valid, max_path_length)
                    'remaining_quad_lengths': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                    'path_start_to_nearest_quad_ids': torch.full((num_valid, max_path_length), -1, dtype=torch.int32, device=self.device),  # (num_valid, max_path_length)
                    'path_start_to_nearest_lengths': torch.zeros(num_valid, dtype=torch.int32, device=self.device),  # (num_valid,)
                }
                
                # 批量填充张量数据
                for i, (quad_id, cross_id, path_idx) in enumerate(zip(valid_quad_ids, valid_nearest_cross_ids, valid_nearest_path_indices)):
                    cross_info = self.cross_data[cross_id.item()]
                    path_dict = cross_info['paths'][path_idx.item()]
                    # 填充waypoint信息
                    self.nearest_neighbor_tensors['to_start_waypoint_ids'][i] = path_dict['to_start_waypoint_id']
                    self.nearest_neighbor_tensors['to_start_road_ids'][i] = path_dict['to_start_road_id']
                    self.nearest_neighbor_tensors['to_start_lane_ids'][i] = path_dict['to_start_lane_id']
                    self.nearest_neighbor_tensors['from_end_waypoint_ids'][i] = path_dict['from_end_waypoint_id']
                    self.nearest_neighbor_tensors['from_end_road_ids'][i] = path_dict['from_end_road_id']
                    self.nearest_neighbor_tensors['from_end_lane_ids'][i] = path_dict['from_end_lane_id']
                    
                    # 找到nearest_quad_id在path_quad_ids中的位置
                    path_quad_ids = path_dict['path_quad_ids']
                    nearest_quad_id = valid_nearest_quad_ids[i]
                    quad_positions = torch.where(path_quad_ids == nearest_quad_id)[0]
                    
                    if quad_positions.numel() > 0:
                        quad_position = quad_positions[0]
                        
                        # 保留从nearest_quad_id到path_quad_ids末尾的所有quad_ids（用于情况2）
                        remaining_quad_ids = path_quad_ids[quad_position:]
                        remaining_length = remaining_quad_ids.shape[0]
                        self.nearest_neighbor_tensors['remaining_quad_ids'][i, :remaining_length] = remaining_quad_ids
                        self.nearest_neighbor_tensors['remaining_quad_lengths'][i] = remaining_length
                        
                        # 保留从path_quad_ids开头到nearest_quad_id的所有quad_ids（用于情况4）
                        path_start_to_nearest_quad_ids = path_quad_ids[:quad_position+1]
                        path_start_length = path_start_to_nearest_quad_ids.shape[0]
                        self.nearest_neighbor_tensors['path_start_to_nearest_quad_ids'][i, :path_start_length] = path_start_to_nearest_quad_ids
                        self.nearest_neighbor_tensors['path_start_to_nearest_lengths'][i] = path_start_length
                
                # 创建quad_id到张量索引的映射，用于快速查找
                self.quad_id_to_tensor_index = {}
                for i, quad_id in enumerate(valid_quad_ids):
                    self.quad_id_to_tensor_index[quad_id.item()] = i

                # 保持向后兼容的字典格式（用于plan_path中的情况2和情况4）
                self.nearest_neighbor_info = {}
                for i, quad_id in enumerate(valid_quad_ids):
                    self.nearest_neighbor_info[quad_id.item()] = {
                        'nearest_quad_id': valid_nearest_quad_ids[i],
                        'nearest_cross_id': valid_nearest_cross_ids[i],
                        'nearest_path_idx': valid_nearest_path_indices[i],
                        'to_start_waypoint_id': self.nearest_neighbor_tensors['to_start_waypoint_ids'][i],
                        'to_start_road_id': self.nearest_neighbor_tensors['to_start_road_ids'][i],
                        'to_start_lane_id': self.nearest_neighbor_tensors['to_start_lane_ids'][i],
                        'from_end_waypoint_id': self.nearest_neighbor_tensors['from_end_waypoint_ids'][i],
                        'from_end_road_id': self.nearest_neighbor_tensors['from_end_road_ids'][i],
                        'from_end_lane_id': self.nearest_neighbor_tensors['from_end_lane_ids'][i],
                        'remaining_quad_ids': self.nearest_neighbor_tensors['remaining_quad_ids'][i, :self.nearest_neighbor_tensors['remaining_quad_lengths'][i]],
                        'path_start_to_nearest_quad_ids': self.nearest_neighbor_tensors['path_start_to_nearest_quad_ids'][i, :self.nearest_neighbor_tensors['path_start_to_nearest_lengths'][i]],
                        'path_quad_ids': path_dict['path_quad_ids'],
                        'distance': valid_min_distances[i],
                        'angle': valid_angles[i]
                    }
        nearest_neighbor_time = time.time() - nearest_neighbor_start
        print(f"最近邻预计算处理耗时: {nearest_neighbor_time:.4f}秒")

        # ===================预处理cross匹配数据===================
        cross_match_start = time.time()
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
        cross_match_time = time.time() - cross_match_start
        print(f"cross匹配数据预处理耗时: {cross_match_time:.4f}秒")
        
        total_init_time = time.time() - start_time
        print(f"PathPlanner初始化总耗时: {total_init_time:.4f}秒")

    def plan_path(self, start_quad_id: torch.Tensor, goal_quad_id: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            start_quad_id: 起点quad_id(B,M,1),为tensor
            goal_quad_id: 终点quad_id(B,M,1),为tensor
        Returns:
            path: 路径(B,M,max_path_length,2)
        '''
        plan_start_time = time.time()
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

        # 初始化结果张量 - 与start_in_filtered_mask长度一致
        batch_size = start_quad_flat.shape[0]
        start_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        end_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        
        # 使用张量存储有效的lane_waypoints，支持向量化操作
        # 预分配固定大小的张量，最大waypoint数量设为100
        max_waypoints_per_lane = 100  # 增加到100，确保足够空间
        lane_waypoints_tensor = torch.full((batch_size, max_waypoints_per_lane, 2), -1, dtype=torch.float32, device=self.device)
        goal_lane_waypoints_tensor = torch.full((batch_size, max_waypoints_per_lane, 2), -1, dtype=torch.float32, device=self.device)
        # 记录每个lane的实际waypoint数量
        lane_waypoints_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        goal_lane_waypoints_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        # 情况1：处理起点在filtered_quad_indices内的情况
        case1_start = time.time()
        if start_in_filtered_mask.any():
            # 获取满足条件的start_quad索引
            case1_step1_start = time.time()
            valid_start_indices = torch.where(start_in_filtered_mask)[0]
            valid_start_quads = start_quad_flat[valid_start_indices]
            case1_step1_time = time.time() - case1_step1_start
            print(f"  情况1-步骤1(获取有效索引): {case1_step1_time:.4f}秒")
            
            # 使用广播操作进行批量查找
            # 1. 广播查找quad_to_next_waypoint映射
            case1_step2_start = time.time()
            quad_expanded = valid_start_quads.unsqueeze(1)  # (num_valid, 1)
            quad_ids_expanded = self.quad_to_next_waypoint_quad_ids.unsqueeze(0)  # (1, num_quad_ids)
            quad_matches = (quad_expanded == quad_ids_expanded)  # (num_valid, num_quad_ids)
            case1_step2_time = time.time() - case1_step2_start
            print(f"  情况1-步骤2(广播查找): {case1_step2_time:.4f}秒")
            
            # 获取匹配的索引和对应的waypoint_ids
            case1_step3_start = time.time()
            matched_indices = torch.where(quad_matches)
            if len(matched_indices[0]) > 0:
                valid_quad_indices = matched_indices[0]  # 在valid_start_quads中的索引
                quad_to_next_indices = matched_indices[1]  # 在quad_to_next_waypoint_quad_ids中的索引
                waypoint_ids = self.quad_to_next_waypoint_values[quad_to_next_indices]#获取了所有起始quad的下一个waypoint
            case1_step3_time = time.time() - case1_step3_start
            print(f"  情况1-步骤3(获取匹配索引): {case1_step3_time:.4f}秒")

            if len(matched_indices[0]) > 0:
                # 2. 使用预创建的waypoint_to_lane映射进行批量查找（GPU向量化）
                case1_step4a_start = time.time()
                waypoint_ids_expanded = waypoint_ids.unsqueeze(1)  # (N, 1)
                map_ids_expanded = self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)  # (1, K)
                waypoint_matches = (waypoint_ids_expanded == map_ids_expanded)  # (N, K)
                case1_step4a_time = time.time() - case1_step4a_start
                print(f"  情况1-步骤4a(waypoint广播查找): {case1_step4a_time:.4f}秒")

                # 3. 批量处理匹配（GPU向量化）
                case1_step4b_start = time.time()
                waypoint_matches_int = waypoint_matches.int()        # 将布尔张量转换为整数张量
                match_indices = waypoint_matches_int.argmax(dim=1)   # 获取每个waypoint在waypoint_to_lane中匹配的索引
                valid_matches = waypoint_matches.any(dim=1)          # 标记哪些waypoint在waypoint_to_lane中找到了有效值
                valid_wp_idx = torch.where(valid_matches)[0]         # 获取所有有效匹配的waypoint的索引
                case1_step4b_time = time.time() - case1_step4b_start
                print(f"  情况1-步骤4b(waypoint匹配处理): {case1_step4b_time:.4f}秒")

                if valid_wp_idx.numel() > 0:
                    case1_step5_start = time.time()
                    sel_match_idx = match_indices[valid_wp_idx]
                    roads_found = self.waypoint_to_lane['road_ids'][sel_match_idx]
                    lanes_found = self.waypoint_to_lane['lane_ids'][sel_match_idx]
                    # 使用预处理的lane索引张量
                    lane_indices_tensor = self.waypoint_to_lane['lane_indices_tensor'][sel_match_idx]
                    lane_lengths = self.waypoint_to_lane['lane_lengths'][sel_match_idx]
                    case1_step5_time = time.time() - case1_step5_start
                    print(f"  情况1-步骤5(获取lane信息): {case1_step5_time:.4f}秒")

                    # waypoint在对应lane中的位置（GPU并行查找）
                    case1_step6_start = time.time()
                    wp_to_find = waypoint_ids[valid_wp_idx]
                    eq = (wp_to_find.unsqueeze(1) == lane_indices_tensor)  # (Nv, L)
                    eq_int = eq.int()
                    pos = eq_int.argmax(dim=1)           # 第一个匹配位置
                    has_pos = eq.any(dim=1)
                    pos = torch.where(has_pos, pos, torch.full_like(pos, -1))
                    case1_step6_time = time.time() - case1_step6_start
                    print(f"  情况1-步骤6(waypoint位置查找): {case1_step6_time:.4f}秒")

                    # 末端waypoint（每条lane最后一个有效点）
                    case1_step7_start = time.time()
                    last_idx = torch.clamp(lane_lengths - 1, min=0)
                    final_wp_ids = lane_indices_tensor[torch.arange(lane_indices_tensor.shape[0], device=self.device), last_idx]
                    final_wp_ids = torch.where((lane_lengths > 0) & (pos >= 0), final_wp_ids, torch.full_like(final_wp_ids, -1))
                    case1_step7_time = time.time() - case1_step7_start
                    print(f"  情况1-步骤7(末端waypoint计算): {case1_step7_time:.4f}秒")
                    
                    # 使用预处理的cross匹配数据
                    case1_step8_start = time.time()
                    if self.all_end_ids.numel() > 0:
                        m = (final_wp_ids.unsqueeze(1) == self.all_end_ids.unsqueeze(0))  # (Nv, T)
                        any_m = m.any(dim=1)
                        m_int = m.int()
                        idx = m_int.argmax(dim=1)
                        cross_ids_found = torch.where(any_m, self.all_end_cids[idx], torch.full_like(final_wp_ids, -1))
                    else:
                        cross_ids_found = torch.full_like(final_wp_ids, -1)
                    case1_step8_time = time.time() - case1_step8_start
                    print(f"  情况1-步骤8(cross匹配): {case1_step8_time:.4f}秒")
                    
                    # 回写 start_ids（仅对有效项）
                    case1_step9_start = time.time()
                    orig_idx = valid_start_indices[valid_quad_indices[valid_wp_idx]]
                    write_mask = (cross_ids_found != -1) & (pos >= 0)
                    if write_mask.any():
                        idx_w = orig_idx[write_mask]
                        start_ids[idx_w, 0] = cross_ids_found[write_mask]
                        start_ids[idx_w, 1] = roads_found[write_mask]
                        start_ids[idx_w, 2] = lanes_found[write_mask]
                    case1_step9_time = time.time() - case1_step9_start
                    print(f"  情况1-步骤9(回写start_ids): {case1_step9_time:.4f}秒")
                    
                    if write_mask.any():
                        # 填充lane_waypoints（只保存有效值）- 优化版本
                        case1_step10_start = time.time()
                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        case1_step10_time = time.time() - case1_step10_start
                        print(f"  情况1-步骤10(回写start_ids): {case1_step10_time:.4f}秒")
            
                        # 批量处理，使用向量化操作
                        if sel_indices.numel() > 0:
                            # 获取所有需要处理的lane数据
                            case1_step11_start = time.time()
                            batch_lane_data = lane_indices_tensor[sel_indices]
                            batch_positions = sel_pos
                            case1_step11_time = time.time() - case1_step11_start
                            print(f"  情况1-步骤11(获取lane数据): {case1_step11_time:.4f}秒")
                            
                            # 向量化处理：为每个位置创建mask
                            case1_step12_start = time.time()
                            max_lane_length = batch_lane_data.shape[1]
                            position_indices = torch.arange(max_lane_length, device=self.device).unsqueeze(0).expand(batch_positions.shape[0], -1)
                            start_positions = batch_positions.unsqueeze(1)
                            case1_step12_time = time.time() - case1_step12_start
                            print(f"  情况1-步骤12(创建mask): {case1_step12_time:.4f}秒")
                            
                            # 创建有效mask：从start_position到末尾，且值>=0
                            valid_mask = (position_indices >= start_positions) & (batch_lane_data >= 0)
                            # 对每个lane，获取有效的waypoint - 完全GPU向量化版本
                            case1_step13_start = time.time()
                            # 使用向量化操作直接处理所有waypoint
                            # 创建waypoint计数和偏移量
                            waypoint_counts = valid_mask.sum(dim=1)  # 每个lane的有效waypoint数量
                            total_waypoints = waypoint_counts.sum()
                
                            if total_waypoints > 0:
                                # 创建扁平化的waypoint索引
                                waypoint_offsets = torch.cumsum(waypoint_counts, dim=0) - waypoint_counts
                                # 获取所有有效的waypoint
                                valid_waypoints_flat = batch_lane_data[valid_mask]
                                # 批量获取所有坐标
                                all_coords = self.get_waypoint_coords(valid_waypoints_flat)
                                
                                # 使用向量化操作分配坐标到对应的lane - 完全消除for循环
                                # 创建累积偏移量
                                cumsum_counts = torch.cumsum(waypoint_counts, dim=0)
                                start_indices = cumsum_counts - waypoint_counts
                                end_indices = cumsum_counts
            
                                # 使用向量化操作直接分配坐标
                                valid_mask_counts = waypoint_counts > 0
                                valid_indices = torch.where(valid_mask_counts)[0]
        
                                if valid_indices.numel() > 0:
                                    # 批量获取所有有效的坐标范围
                                    valid_starts = start_indices[valid_indices]
                                    valid_ends = end_indices[valid_indices]
                                    # 使用向量化操作分配坐标 - 完全消除for循环
                                    # 创建索引映射
                                    target_indices = idx_w[valid_indices]
                                    # 完全向量化批量分配坐标 - 使用高级张量操作
                                    # 使用scatter操作批量填充
                                    # 创建scatter操作的索引
                                    batch_indices_for_scatter = torch.repeat_interleave(target_indices, valid_ends - valid_starts)
                                    coord_indices_for_scatter = torch.arange((valid_ends - valid_starts).sum(), device=self.device)
                                    
                                    # 使用scatter操作批量填充
                                    # 限制坐标数量不超过预分配空间
                                    safe_coord_indices = coord_indices_for_scatter[:max_waypoints_per_lane * len(target_indices)]
                                    safe_batch_indices = batch_indices_for_scatter[:len(safe_coord_indices)]
                                    safe_coords = all_coords[:len(safe_coord_indices)]
                                    
                                    # 计算每个batch在tensor中的位置
                                    waypoint_indices = torch.arange(len(safe_coord_indices), device=self.device) % max_waypoints_per_lane
                                    
                                    # 批量填充
                                    lane_waypoints_tensor[safe_batch_indices, waypoint_indices] = safe_coords
                                    
                                    # 更新长度信息
                                    safe_lengths = torch.clamp(valid_ends - valid_starts, max=max_waypoints_per_lane).to(torch.int32)
                                    lane_waypoints_lengths[target_indices] = safe_lengths

                            case1_step13_time = time.time() - case1_step13_start
                            print(f"  情况1-步骤13(获取有效waypoint): {case1_step13_time:.4f}秒") 
        case1_time = time.time() - case1_start
        print(f"情况1处理耗时: {case1_time:.4f}秒")
        
        # 情况2：处理起点不在filtered_quad_indices内的情况 - 使用张量批量查找
        case2_start = time.time()
        if (~start_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的start_quad
            invalid_start_indices = torch.where(~start_in_filtered_mask)[0]
            invalid_start_quads = start_quad_flat[invalid_start_indices]
            if invalid_start_quads.numel() > 0:
                # 使用新的张量批量查找方法
                case2_step1_start = time.time()
                neighbor_info_batch = self.get_nearest_neighbor_info_batch(invalid_start_quads)
                case2_step1_time = time.time() - case2_step1_start
                print(f"  情况2-步骤1(批量获取最近邻): {case2_step1_time:.4f}秒")
                # 筛选有效的最近邻信息
                valid_mask = neighbor_info_batch['valid_mask']
                if valid_mask.any():
                    # 获取有效的索引
                    valid_indices = torch.where(valid_mask)[0]
                    valid_start_indices = invalid_start_indices[valid_indices]
                    # 批量更新start_ids
                    case2_step2_start = time.time()
                    start_ids[valid_start_indices, 0] = neighbor_info_batch['nearest_cross_ids'][valid_indices]
                    start_ids[valid_start_indices, 1] = neighbor_info_batch['to_start_road_ids'][valid_indices]
                    start_ids[valid_start_indices, 2] = neighbor_info_batch['to_start_lane_ids'][valid_indices]
                    case2_step2_time = time.time() - case2_step2_start
                    print(f"  情况2-步骤2(批量更新start_ids): {case2_step2_time:.4f}秒")
                    # 批量处理waypoints - 使用张量操作
                    case2_step3_start = time.time()
                    # 获取有效的remaining_quad_ids
                    valid_remaining_quad_ids = neighbor_info_batch['remaining_quad_ids'][valid_indices]
                    valid_remaining_lengths = neighbor_info_batch['remaining_quad_lengths'][valid_indices]
                    # 完全向量化批量获取坐标 - 使用高级张量操作
                    if len(valid_start_indices) > 0:
                        # 批量获取所有有效的quad_ids
                        max_remaining_length = valid_remaining_lengths.max() if valid_remaining_lengths.numel() > 0 else 0
                        
                        if max_remaining_length > 0:
                            # 创建批量索引：为每个start_idx创建对应的行索引
                            batch_indices = valid_start_indices.unsqueeze(1).expand(-1, max_remaining_length)
                            # 创建列索引：0到max_remaining_length-1
                            col_indices = torch.arange(max_remaining_length, device=self.device).unsqueeze(0).expand(len(valid_start_indices), -1)
                            # 创建长度mask
                            length_mask = col_indices < valid_remaining_lengths.unsqueeze(1)
                            
                            # 批量获取所有有效的quad_ids
                            all_remaining_quads = valid_remaining_quad_ids[:, :max_remaining_length]
                            # 过滤掉-1值，创建有效mask
                            valid_quad_mask = (all_remaining_quads >= 0) & length_mask
                            
                            # 获取所有有效的quad_ids
                            all_valid_quads = all_remaining_quads[valid_quad_mask]
                            
                            if all_valid_quads.numel() > 0:
                                # 批量获取所有坐标
                                all_coords = self.get_quad_centers(all_valid_quads)
                                
                                # 计算每个batch的实际坐标数量
                                valid_counts = valid_quad_mask.sum(dim=1)  # (num_batches,)
                                
                                # 创建scatter操作的索引
                                # 为每个有效坐标创建对应的batch索引
                                batch_indices_for_scatter = torch.repeat_interleave(valid_start_indices, valid_counts)
                                coord_indices_for_scatter = torch.arange(all_valid_quads.numel(), device=self.device)
                                
                                # 使用scatter操作批量填充
                                # 限制坐标数量不超过预分配空间
                                safe_coord_indices = coord_indices_for_scatter[:max_waypoints_per_lane * len(valid_start_indices)]
                                safe_batch_indices = batch_indices_for_scatter[:len(safe_coord_indices)]
                                safe_coords = all_coords[:len(safe_coord_indices)]
                                
                                # 计算每个batch在tensor中的位置
                                waypoint_indices = torch.arange(len(safe_coord_indices), device=self.device) % max_waypoints_per_lane
                                
                                # 批量填充
                                lane_waypoints_tensor[safe_batch_indices, waypoint_indices] = safe_coords

                                # 更新长度信息
                                safe_lengths = torch.clamp(valid_counts, max=max_waypoints_per_lane).to(torch.int32)
                                lane_waypoints_lengths[valid_start_indices] = safe_lengths   

                    case2_step3_time = time.time() - case2_step3_start
                    print(f"  情况2-步骤3(批量处理waypoints): {case2_step3_time:.4f}秒")            
        case2_time = time.time() - case2_start
        print(f"情况2处理耗时: {case2_time:.4f}秒")

        case3_start = time.time()
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
                        # 填充goal_lane_waypoints（只保存有效值）- 优化版本
                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        
                        # 批量处理，使用向量化操作
                        if sel_indices.numel() > 0:
                            # 获取所有需要处理的lane数据
                            batch_lane_data = lane_indices_tensor[sel_indices]
                            batch_positions = sel_pos
                            
                            # 向量化处理：为每个位置创建mask
                            max_lane_length = batch_lane_data.shape[1]
                            position_indices = torch.arange(max_lane_length, device=self.device).unsqueeze(0).expand(batch_positions.shape[0], -1)
                            end_positions = batch_positions.unsqueeze(1)
                            
                            # 创建有效mask：从开始到end_position，且值>=0
                            valid_mask = (position_indices <= end_positions) & (batch_lane_data >= 0)
                            
                            # 对每个lane，获取有效的waypoints - 完全GPU向量化版本
                            # 使用向量化操作直接处理所有waypoint
                            # 创建waypoint计数和偏移量
                            waypoint_counts = valid_mask.sum(dim=1)  # 每个lane的有效waypoint数量
                            total_waypoints = waypoint_counts.sum()
                            
                            if total_waypoints > 0:
                                # 创建扁平化的waypoint索引
                                waypoint_offsets = torch.cumsum(waypoint_counts, dim=0) - waypoint_counts
                                
                                # 获取所有有效的waypoint
                                valid_waypoints_flat = batch_lane_data[valid_mask]
                                
                                # 批量获取所有坐标
                                all_coords = self.get_waypoint_coords(valid_waypoints_flat)
                                
                                # 使用向量化操作分配坐标到对应的lane - 张量版本
                                # 创建累积偏移量
                                cumsum_counts = torch.cumsum(waypoint_counts, dim=0)
                                start_indices = cumsum_counts - waypoint_counts
                                end_indices = cumsum_counts
                                
                                # 使用向量化操作直接分配坐标
                                valid_mask_counts = waypoint_counts > 0
                                valid_indices = torch.where(valid_mask_counts)[0]
                                
                                if valid_indices.numel() > 0:
                                    # 批量获取所有有效的坐标范围
                                    valid_starts = start_indices[valid_indices]
                                    valid_ends = end_indices[valid_indices]
                                    
                                    # 完全向量化批量分配坐标 - 使用高级张量操作
                                    target_indices = idx_w[valid_indices]
                                    
                                    # 使用scatter操作批量填充
                                    # 创建scatter操作的索引
                                    batch_indices_for_scatter = torch.repeat_interleave(target_indices, valid_ends - valid_starts)
                                    coord_indices_for_scatter = torch.arange((valid_ends - valid_starts).sum(), device=self.device)
                                    
                                    # 使用scatter操作批量填充
                                    # 限制坐标数量不超过预分配空间
                                    safe_coord_indices = coord_indices_for_scatter[:max_waypoints_per_lane * len(target_indices)]
                                    safe_batch_indices = batch_indices_for_scatter[:len(safe_coord_indices)]
                                    safe_coords = all_coords[:len(safe_coord_indices)]
                                    
                                    # 计算每个batch在tensor中的位置
                                    waypoint_indices = torch.arange(len(safe_coord_indices), device=self.device) % max_waypoints_per_lane
                                    
                                    # 批量填充
                                    goal_lane_waypoints_tensor[safe_batch_indices, waypoint_indices] = safe_coords
                                    
                                    # 更新长度信息
                                    safe_lengths = torch.clamp(valid_ends - valid_starts, max=max_waypoints_per_lane).to(torch.int32)
                                    goal_lane_waypoints_lengths[target_indices] = safe_lengths                                
        case3_time = time.time() - case3_start
        print(f"情况3处理耗时: {case3_time:.4f}秒")

        case4_start = time.time()
        # 情况4：处理终点不在filtered_quad_indices内的情况 - 使用张量批量查找
        if (~goal_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的goal_quad
            invalid_goal_indices = torch.where(~goal_in_filtered_mask)[0]
            invalid_goal_quads = goal_quad_flat[invalid_goal_indices]
            if invalid_goal_quads.numel() > 0:
                # 使用新的张量批量查找方法
                case4_step1_start = time.time()
                neighbor_info_batch = self.get_nearest_neighbor_info_batch(invalid_goal_quads)
                case4_step1_time = time.time() - case4_step1_start
                print(f"  情况4-步骤1(批量获取最近邻): {case4_step1_time:.4f}秒")
                # 筛选有效的最近邻信息
                valid_mask = neighbor_info_batch['valid_mask']
                if valid_mask.any():
                    # 获取有效的索引
                    valid_indices = torch.where(valid_mask)[0]
                    valid_goal_indices = invalid_goal_indices[valid_indices]
                    # 批量更新end_ids
                    case4_step2_start = time.time()
                    end_ids[valid_goal_indices, 0] = neighbor_info_batch['nearest_cross_ids'][valid_indices]
                    end_ids[valid_goal_indices, 1] = neighbor_info_batch['from_end_road_ids'][valid_indices]
                    end_ids[valid_goal_indices, 2] = neighbor_info_batch['from_end_lane_ids'][valid_indices]
                    case4_step2_time = time.time() - case4_step2_start
                    print(f"  情况4-步骤2(批量更新end_ids): {case4_step2_time:.4f}秒")
                    # 批量处理waypoints - 使用张量操作
                    case4_step3_start = time.time()
                    # 获取有效的path_start_to_nearest_quad_ids
                    valid_path_start_quad_ids = neighbor_info_batch['path_start_to_nearest_quad_ids'][valid_indices]
                    valid_path_start_lengths = neighbor_info_batch['path_start_to_nearest_lengths'][valid_indices]
                    # 完全向量化批量获取坐标 - 使用高级张量操作
                    if len(valid_goal_indices) > 0:
                        # 批量获取所有有效的quad_ids
                        max_path_length = valid_path_start_lengths.max() if valid_path_start_lengths.numel() > 0 else 0
                        if max_path_length > 0:
                            # 创建批量索引：为每个goal_idx创建对应的行索引
                            batch_indices = valid_goal_indices.unsqueeze(1).expand(-1, max_path_length)
                            # 创建列索引：0到max_path_length-1
                            col_indices = torch.arange(max_path_length, device=self.device).unsqueeze(0).expand(len(valid_goal_indices), -1)
                            # 创建长度mask
                            length_mask = col_indices < valid_path_start_lengths.unsqueeze(1)
                            
                            # 批量获取所有有效的quad_ids
                            all_path_start_quads = valid_path_start_quad_ids[:, :max_path_length]
                            # 过滤掉-1值，创建有效mask
                            valid_quad_mask = (all_path_start_quads >= 0) & length_mask
                            
                            # 获取所有有效的quad_ids
                            all_valid_quads = all_path_start_quads[valid_quad_mask]
                            
                            if all_valid_quads.numel() > 0:
                                # 批量获取所有坐标
                                all_coords = self.get_quad_centers(all_valid_quads)
                                
                                # 计算每个batch的实际坐标数量
                                valid_counts = valid_quad_mask.sum(dim=1)  # (num_batches,)
                                
                                # 创建scatter操作的索引
                                # 为每个有效坐标创建对应的batch索引
                                batch_indices_for_scatter = torch.repeat_interleave(valid_goal_indices, valid_counts)
                                coord_indices_for_scatter = torch.arange(all_valid_quads.numel(), device=self.device)
                                
                                # 使用scatter操作批量填充
                                # 限制坐标数量不超过预分配空间
                                safe_coord_indices = coord_indices_for_scatter[:max_waypoints_per_lane * len(valid_goal_indices)]
                                safe_batch_indices = batch_indices_for_scatter[:len(safe_coord_indices)]
                                safe_coords = all_coords[:len(safe_coord_indices)]
                                
                                # 计算每个batch在tensor中的位置
                                waypoint_indices = torch.arange(len(safe_coord_indices), device=self.device) % max_waypoints_per_lane
                                
                                # 批量填充
                                goal_lane_waypoints_tensor[safe_batch_indices, waypoint_indices] = safe_coords
                                
                                # 更新长度信息
                                safe_lengths = torch.clamp(valid_counts, max=max_waypoints_per_lane).to(torch.int32)
                                goal_lane_waypoints_lengths[valid_goal_indices] = safe_lengths
                
                    case4_step3_time = time.time() - case4_step3_start
                    print(f"  情况4-步骤3(批量处理waypoints): {case4_step3_time:.4f}秒")
        case4_time = time.time() - case4_start
        print(f"情况4处理耗时: {case4_time:.4f}秒")      

        # 在这里已经得到了start_ids，end_ids。通过cross_data内部的"waypoint_graph"得到两个节点之间的其它节点。
        plan_total_time = time.time() - plan_start_time
        print(f"plan_path总耗时: {plan_total_time:.4f}秒")
        print()
        
        # 返回张量结构的结果，便于后续处理
        # lane_waypoints_tensor: (batch_size, max_waypoints_per_lane, 2) - 起点lane坐标
        # goal_lane_waypoints_tensor: (batch_size, max_waypoints_per_lane, 2) - 终点lane坐标
        # lane_waypoints_lengths: (batch_size,) - 每个起点lane的实际waypoint数量
        # goal_lane_waypoints_lengths: (batch_size,) - 每个终点lane的实际waypoint数量


        return start_ids
        

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
    
    def get_nearest_neighbor_info_batch(self, quad_ids: torch.Tensor) -> dict:
        """
        批量获取最近邻信息，使用张量操作
        Args:
            quad_ids: 要查询的quad_id张量 (N,) 
        Returns:
            dict: 包含批量最近邻信息的字典
        """
        # 使用广播查找匹配的quad_ids
        get_nearest_neighbor_info_batch_start = time.time()
        quad_expanded = quad_ids.unsqueeze(1)  # (N, 1)
        stored_quad_ids_expanded = self.nearest_neighbor_tensors['quad_ids'].unsqueeze(0)  # (1, num_stored)
        quad_matches = (quad_expanded == stored_quad_ids_expanded)  # (N, num_stored)
        # 获取匹配的索引
        matched_indices = torch.where(quad_matches)
        # 获取匹配的索引
        query_indices = matched_indices[0]  # 在quad_ids中的索引
        stored_indices = matched_indices[1]  # 在nearest_neighbor_tensors中的索引
        # 创建结果张量
        num_quads = quad_ids.shape[0]
        max_path_length = self.nearest_neighbor_tensors['remaining_quad_ids'].shape[1]
        # 初始化结果张量
        result = {
            'valid_mask': torch.zeros(num_quads, dtype=torch.bool, device=self.device),
            'nearest_quad_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'nearest_cross_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'nearest_path_indices': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'to_start_road_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'to_start_lane_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'from_end_road_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'from_end_lane_ids': torch.full((num_quads,), -1, dtype=torch.int32, device=self.device),
            'remaining_quad_ids': torch.full((num_quads, max_path_length), -1, dtype=torch.int32, device=self.device),
            'remaining_quad_lengths': torch.zeros(num_quads, dtype=torch.int32, device=self.device),
            'path_start_to_nearest_quad_ids': torch.full((num_quads, max_path_length), -1, dtype=torch.int32, device=self.device),
            'path_start_to_nearest_lengths': torch.zeros(num_quads, dtype=torch.int32, device=self.device),
        }
        # 批量填充匹配的数据
        result['valid_mask'][query_indices] = True
        result['nearest_quad_ids'][query_indices] = self.nearest_neighbor_tensors['nearest_quad_ids'][stored_indices]
        result['nearest_cross_ids'][query_indices] = self.nearest_neighbor_tensors['nearest_cross_ids'][stored_indices]
        result['nearest_path_indices'][query_indices] = self.nearest_neighbor_tensors['nearest_path_indices'][stored_indices]
        result['to_start_road_ids'][query_indices] = self.nearest_neighbor_tensors['to_start_road_ids'][stored_indices]
        result['to_start_lane_ids'][query_indices] = self.nearest_neighbor_tensors['to_start_lane_ids'][stored_indices]
        result['from_end_road_ids'][query_indices] = self.nearest_neighbor_tensors['from_end_road_ids'][stored_indices]
        result['from_end_lane_ids'][query_indices] = self.nearest_neighbor_tensors['from_end_lane_ids'][stored_indices]
        get_nearest_neighbor_info_batch_time = time.time() - get_nearest_neighbor_info_batch_start

        # 完全向量化填充路径信息 - 使用高级张量操作
        if len(query_indices) > 0:
            # 批量获取长度信息
            remaining_lengths = self.nearest_neighbor_tensors['remaining_quad_lengths'][stored_indices]
            path_start_lengths = self.nearest_neighbor_tensors['path_start_to_nearest_lengths'][stored_indices]
            
            # 方法1：使用高级索引操作 - 完全向量化
            # 创建索引张量用于批量填充
            max_remaining_length = remaining_lengths.max() if remaining_lengths.numel() > 0 else 0
            max_path_length = path_start_lengths.max() if path_start_lengths.numel() > 0 else 0
            
            if max_remaining_length > 0:
                # 创建批量索引：为每个query_idx创建对应的行索引
                batch_indices = query_indices.unsqueeze(1).expand(-1, max_remaining_length)
                # 创建列索引：0到max_remaining_length-1
                col_indices = torch.arange(max_remaining_length, device=self.device).unsqueeze(0).expand(len(query_indices), -1)
                # 创建长度mask
                length_mask = col_indices < remaining_lengths.unsqueeze(1)
                # 批量填充remaining_quad_ids
                source_data = self.nearest_neighbor_tensors['remaining_quad_ids'][stored_indices, :max_remaining_length]
                result['remaining_quad_ids'][batch_indices[length_mask], col_indices[length_mask]] = source_data[length_mask]
                result['remaining_quad_lengths'][query_indices] = remaining_lengths
            if max_path_length > 0:
                # 创建批量索引：为每个query_idx创建对应的行索引
                batch_indices = query_indices.unsqueeze(1).expand(-1, max_path_length)
                # 创建列索引：0到max_path_length-1
                col_indices = torch.arange(max_path_length, device=self.device).unsqueeze(0).expand(len(query_indices), -1)
                # 创建长度mask
                length_mask = col_indices < path_start_lengths.unsqueeze(1)
                # 批量填充path_start_to_nearest_quad_ids
                source_data = self.nearest_neighbor_tensors['path_start_to_nearest_quad_ids'][stored_indices, :max_path_length]
                result['path_start_to_nearest_quad_ids'][batch_indices[length_mask], col_indices[length_mask]] = source_data[length_mask]
                result['path_start_to_nearest_lengths'][query_indices] = path_start_lengths
        return result

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
    B, M = 2400, 150
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
    

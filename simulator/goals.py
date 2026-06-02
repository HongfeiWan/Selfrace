import torch
import json
import numpy as np
import time
from typing import Dict, Tuple, List, Optional, Any

class PathPlanner:
    LANE_START_NODE = -100
    LANE_END_NODE = -101
    LANE_CONNECTOR_THRESHOLD = 3.0
    LANE_UTURN_THRESHOLD = 5.0
    LANE_UTURN_PENALTY = 10000.0

    def __init__(self, map_path: str, device: torch.device, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print(f"==========PathPlanner init==========")
        start_time = time.time()
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

        # === 方案A：将 filtered_quad_indices(存的是polyId) 转换为索引空间 ===
        self.filtered_quad_indices_idx = torch.empty(0, dtype=torch.int32, device=self.device)
        try:
            if hasattr(self, 'quads_info') and 'polyId' in self.quads_info and isinstance(self.filtered_quad_indices, torch.Tensor) and self.filtered_quad_indices.numel() > 0:
                poly_ids_tensor = self.quads_info['polyId']  # (num_quads,) int32，存储每个索引对应的polyId
                indices_list = []
                for pid in self.filtered_quad_indices.tolist():
                    match = torch.where(poly_ids_tensor == pid)[0]
                    if match.numel() > 0:
                        indices_list.append(int(match[0].item()))
                if len(indices_list) > 0:
                    self.filtered_quad_indices_idx = torch.tensor(indices_list, dtype=torch.int32, device=self.device)
        except Exception as _e:
            # 若构建失败，保持为空，不影响后续运行
            pass

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
        non_filtered_mask = ~torch.isin(all_quad_ids, self.filtered_quad_indices_idx)
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
        
        # ===================初始化WaypointGraphGPU===================
        self._route_distance_ready = False
        try:
            lane_route_graph, lane_route_edge_coords = self._build_lane_route_graph(cross_data)
            self.waypoint_graph_gpu = WaypointGraphGPU(
                cross_data_path,
                device=str(self.device),
                waypoint_graph=lane_route_graph,
            )
            self._install_prebuilt_waypoint_graph_edge_expansions(lane_route_edge_coords)
            self._precompute_route_distance_tables()
        except Exception as e:
            print(f"WaypointGraphGPU初始化失败: {e}")
            self.waypoint_graph_gpu = None
        
        total_init_time = time.time() - start_time
        if self.verbose:
            print(f"PathPlanner模块初始化(预存数据)总耗时: {total_init_time:.4f}秒")

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
        # path = torch.full((start_quad_id.shape[0], start_quad_id.shape[1], 512, 2), -1, dtype=torch.float32, device=self.device)
        # 确定起始点和目标点的类型（是否在filtered_quad_indices内）
        # 使用GPU张量进行快速查询
        # 重塑张量以便进行广播比较
        start_quad_flat = start_quad_id.view(-1)  # (B*M,)
        goal_quad_flat = goal_quad_id.view(-1)    # (B*M,)

        # 查询哪些quad_id在filtered_quad_indices内
        start_in_filtered_mask = torch.isin(start_quad_flat, self.filtered_quad_indices_idx)
        goal_in_filtered_mask = torch.isin(goal_quad_flat, self.filtered_quad_indices_idx)

        # 初始化结果张量 - 与start_in_filtered_mask长度一致
        batch_size = start_quad_flat.shape[0]
        start_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        end_ids = torch.full((batch_size, 3), -1, dtype=torch.int32, device=self.device)  # [cross_id, road_id, lane_id]
        
        # 使用张量存储有效的lane_waypoints，支持向量化操作
        # 预分配固定大小的张量，最大waypoint数量设为100
        max_waypoints_per_lane = 50  # 增加到100，确保足够空间
        lane_waypoints_tensor = torch.full((batch_size, max_waypoints_per_lane, 2), -1, dtype=torch.float32, device=self.device)
        goal_lane_waypoints_tensor = torch.full((batch_size, max_waypoints_per_lane, 2), -1, dtype=torch.float32, device=self.device)
        # 记录每个lane的实际waypoint数量
        lane_waypoints_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        goal_lane_waypoints_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        # 情况1：处理起点在filtered_quad_indices内的情况
        if start_in_filtered_mask.any():
            # 获取满足条件的start_quad索引
            valid_start_indices = torch.where(start_in_filtered_mask)[0]
            valid_start_quads = start_quad_flat[valid_start_indices]
            
            # 广播查找quad_to_next_waypoint映射
            quad_expanded = valid_start_quads.unsqueeze(1)  # (num_valid, 1)
            quad_ids_expanded = self.quad_to_next_waypoint_quad_ids.unsqueeze(0)  # (1, num_quad_ids)
            quad_matches = (quad_expanded == quad_ids_expanded)  # (num_valid, num_quad_ids)
            matched_indices = torch.where(quad_matches)
            if len(matched_indices[0]) > 0:
                valid_quad_indices = matched_indices[0]  # 在valid_start_quads中的索引
                quad_to_next_indices = matched_indices[1]  # 在quad_to_next_waypoint_quad_ids中的索引
                waypoint_ids = self.quad_to_next_waypoint_values[quad_to_next_indices]#获取了所有起始quad的下一个waypoint

            if len(matched_indices[0]) > 0:
                waypoint_ids_expanded = waypoint_ids.unsqueeze(1)  # (N, 1)
                map_ids_expanded = self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)  # (1, K)
                waypoint_matches = (waypoint_ids_expanded == map_ids_expanded)  # (N, K)
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

                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        if sel_indices.numel() > 0:
                            batch_lane_data = lane_indices_tensor[sel_indices]
                            max_lane_length = batch_lane_data.shape[1]
                            position_indices = torch.arange(max_lane_length, device=self.device).unsqueeze(0).expand(sel_pos.shape[0], -1)
                            valid_mask = (position_indices >= sel_pos.unsqueeze(1)) & (batch_lane_data >= 0)
                            source_ids = torch.where(valid_mask, batch_lane_data, torch.full_like(batch_lane_data, -1))
                            source_lengths = valid_mask.sum(dim=1)
                            self._write_indexed_coords(
                                idx_w,
                                source_ids,
                                source_lengths,
                                lane_waypoints_tensor,
                                lane_waypoints_lengths,
                                self.get_waypoint_coords,
                            )

        # 情况2：处理起点不在filtered_quad_indices内的情况 - 使用张量批量查找
        if (~start_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的start_quad
            invalid_start_indices = torch.where(~start_in_filtered_mask)[0]
            invalid_start_quads = start_quad_flat[invalid_start_indices]
            if invalid_start_quads.numel() > 0:
                # 使用新的张量批量查找方法
                neighbor_info_batch = self.get_nearest_neighbor_info_batch(invalid_start_quads)

                # 筛选有效的最近邻信息
                valid_mask = neighbor_info_batch['valid_mask']
                if valid_mask.any():
                    # 获取有效的索引
                    valid_indices = torch.where(valid_mask)[0]
                    valid_start_indices = invalid_start_indices[valid_indices]
                    # 批量更新start_ids

                    start_ids[valid_start_indices, 0] = neighbor_info_batch['nearest_cross_ids'][valid_indices]
                    start_ids[valid_start_indices, 1] = neighbor_info_batch['to_start_road_ids'][valid_indices]
                    start_ids[valid_start_indices, 2] = neighbor_info_batch['to_start_lane_ids'][valid_indices]

                    valid_remaining_quad_ids = neighbor_info_batch['remaining_quad_ids'][valid_indices]
                    valid_remaining_lengths = neighbor_info_batch['remaining_quad_lengths'][valid_indices]
                    self._write_indexed_coords(
                        valid_start_indices,
                        valid_remaining_quad_ids,
                        valid_remaining_lengths,
                        lane_waypoints_tensor,
                        lane_waypoints_lengths,
                        self.get_quad_centers,
                    )

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
                        sel_pos = pos[write_mask]
                        sel_indices = torch.where(write_mask)[0]
                        if sel_indices.numel() > 0:
                            batch_lane_data = lane_indices_tensor[sel_indices]
                            max_lane_length = batch_lane_data.shape[1]
                            position_indices = torch.arange(max_lane_length, device=self.device).unsqueeze(0).expand(sel_pos.shape[0], -1)
                            valid_mask = (position_indices <= sel_pos.unsqueeze(1)) & (batch_lane_data >= 0)
                            source_ids = torch.where(valid_mask, batch_lane_data, torch.full_like(batch_lane_data, -1))
                            source_lengths = valid_mask.sum(dim=1)
                            self._write_indexed_coords(
                                idx_w,
                                source_ids,
                                source_lengths,
                                goal_lane_waypoints_tensor,
                                goal_lane_waypoints_lengths,
                                self.get_waypoint_coords,
                            )

        # 情况4：处理终点不在filtered_quad_indices内的情况 - 使用张量批量查找
        if (~goal_in_filtered_mask).any():
            # 获取不在filtered_quad_indices内的goal_quad
            invalid_goal_indices = torch.where(~goal_in_filtered_mask)[0]
            invalid_goal_quads = goal_quad_flat[invalid_goal_indices]
            if invalid_goal_quads.numel() > 0:
                # 使用新的张量批量查找方法
                neighbor_info_batch = self.get_nearest_neighbor_info_batch(invalid_goal_quads)

                # 筛选有效的最近邻信息
                valid_mask = neighbor_info_batch['valid_mask']
                if valid_mask.any():
                    # 获取有效的索引
                    valid_indices = torch.where(valid_mask)[0]
                    valid_goal_indices = invalid_goal_indices[valid_indices]
                    # 批量更新end_ids

                    end_ids[valid_goal_indices, 0] = neighbor_info_batch['nearest_cross_ids'][valid_indices]
                    end_ids[valid_goal_indices, 1] = neighbor_info_batch['from_end_road_ids'][valid_indices]
                    end_ids[valid_goal_indices, 2] = neighbor_info_batch['from_end_lane_ids'][valid_indices]
                    valid_path_start_quad_ids = neighbor_info_batch['path_start_to_nearest_quad_ids'][valid_indices]
                    valid_path_start_lengths = neighbor_info_batch['path_start_to_nearest_lengths'][valid_indices]
                    self._write_indexed_coords(
                        valid_goal_indices,
                        valid_path_start_quad_ids,
                        valid_path_start_lengths,
                        goal_lane_waypoints_tensor,
                        goal_lane_waypoints_lengths,
                        self.get_quad_centers,
                    )

        self._overwrite_quad_lane_endpoint_segments(
            start_quad_flat,
            lane_waypoints_tensor,
            lane_waypoints_lengths,
            use_next_waypoint=True,
        )
        self._overwrite_quad_lane_endpoint_segments(
            goal_quad_flat,
            goal_lane_waypoints_tensor,
            goal_lane_waypoints_lengths,
            use_next_waypoint=False,
        )
        self._fill_endpoint_ids_from_quad_waypoints(
            start_quad_flat,
            start_ids,
            use_next_waypoint=True,
            endpoint_code=self.LANE_END_NODE,
            row_mask=start_in_filtered_mask,
        )
        self._fill_endpoint_ids_from_quad_waypoints(
            goal_quad_flat,
            end_ids,
            use_next_waypoint=False,
            endpoint_code=self.LANE_START_NODE,
            row_mask=goal_in_filtered_mask,
        )
        non_filtered_start_valid = (~start_in_filtered_mask) & (start_ids[:, 1] != -1)
        non_filtered_goal_valid = (~goal_in_filtered_mask) & (end_ids[:, 1] != -1)
        start_ids[non_filtered_start_valid, 0] = self.LANE_START_NODE
        end_ids[non_filtered_goal_valid, 0] = self.LANE_END_NODE

        # 返回张量结构的结果，便于后续处理
        # lane_waypoints_tensor: (B*M, max_waypoints_per_lane, 2) - 起点lane坐标
        # goal_lane_waypoints_tensor: (B*M, max_waypoints_per_lane, 2) - 终点lane坐标
        # lane_waypoints_lengths: (B*M,) - 每个起点lane的实际waypoint数量
        # goal_lane_waypoints_lengths: (B*M,) - 每个终点lane的实际waypoint数量
        # start_ids：（B*M,3）
        # end_ids: (B*M,3)
        # 注意这里的值都没有左对齐，因此可能第一个有效值是从中间某个地方开始的。

        start_centers = self.get_quad_centers(start_quad_flat.to(torch.long))
        goal_centers = self.get_quad_centers(goal_quad_flat.to(torch.long))
        lane_endpoint_valid = (lane_waypoints_tensor[..., 0] != -1) & (lane_waypoints_tensor[..., 1] != -1)
        lane_endpoint_counts = lane_endpoint_valid.sum(dim=1)
        lane_last_idx = torch.clamp(lane_endpoint_counts - 1, min=0).to(torch.long)
        row_idx = torch.arange(batch_size, device=self.device)
        start_graph_xy = lane_waypoints_tensor[row_idx, lane_last_idx]
        start_graph_xy = torch.where(lane_endpoint_counts.unsqueeze(1) > 0, start_graph_xy, start_centers)

        goal_endpoint_valid = (goal_lane_waypoints_tensor[..., 0] != -1) & (goal_lane_waypoints_tensor[..., 1] != -1)
        goal_endpoint_counts = goal_endpoint_valid.sum(dim=1)
        goal_first_xy = goal_lane_waypoints_tensor[:, 0]
        end_graph_xy = torch.where(goal_endpoint_counts.unsqueeze(1) > 0, goal_first_xy, goal_centers)
        waypoint_graph, waypoint_mask, waypoint_node_idx = self.waypoint_graph_gpu.batch_shortest_paths_fixed_len(
            start_ids,
            end_ids,
            start_xy=start_graph_xy,
            end_xy=end_graph_xy,
        )

        # WaypointGraphGPU 现在直接返回完整 batch 的 node indices，避免压缩 batch 回填错行。
        Bm = start_ids.shape[0]
        L_wg = waypoint_graph.shape[1]
        if waypoint_node_idx.shape == (Bm, L_wg):
            full_node_idx = waypoint_node_idx
        else:
            full_node_idx = torch.full((Bm, L_wg), -1, dtype=torch.long, device=self.device)
        
        if self.waypoint_graph_gpu.edge_expansion_points is not None:
            graph_coord_len = 1 + max(0, L_wg - 1) * int(self.waypoint_graph_gpu.edge_expansion_points.shape[1])
        else:
            graph_coord_len = L_wg
        waypoint_graph_coords, waypoint_graph_coord_mask = self.waypoint_graph_gpu.expand_node_indices_to_coords(
            full_node_idx,
            fixed_len=graph_coord_len,
        )

        # =================== 将拼接为路径 ===================
        device = self.device
        # 2) 起点 lane 坐标左对齐（使用其自身的 N，避免与全局 N 混淆）
        max_lane_len = lane_waypoints_tensor.shape[1]
        lane_valid_mask = (lane_waypoints_tensor[..., 0] != -1) & (lane_waypoints_tensor[..., 1] != -1)  # (N_lane, W)
        N_lane = lane_valid_mask.shape[0]
        lane_indices = torch.arange(max_lane_len, device=device).unsqueeze(0).expand(N_lane, -1)
        lane_sort_scores = (~lane_valid_mask).long() * max_lane_len + lane_indices
        lane_order = torch.argsort(lane_sort_scores, dim=1, stable=True)

        # 3) waypoint_graph 坐标左对齐（根据其自身 mask）
        wg_W = waypoint_graph_coords.shape[1]
        wg_valid_mask = waypoint_graph_coord_mask  # (N_wg, wg_W)
        N_wg = wg_valid_mask.shape[0]
        wg_indices = torch.arange(wg_W, device=device).unsqueeze(0).expand(N_wg, -1)
        wg_sort_scores = (~wg_valid_mask).long() * wg_W + wg_indices
        wg_order = torch.argsort(wg_sort_scores, dim=1, stable=True)
        
        # 4) 终点 lane 坐标左对齐
        gl_max = goal_lane_waypoints_tensor.shape[1]
        gl_valid_mask = (goal_lane_waypoints_tensor[..., 0] != -1) & (goal_lane_waypoints_tensor[..., 1] != -1)  # (N_gl, gl_max)
        N_gl = gl_valid_mask.shape[0]
        gl_indices = torch.arange(gl_max, device=device).unsqueeze(0).expand(N_gl, -1)
        gl_sort_scores = (~gl_valid_mask).long() * gl_max + gl_indices
        gl_order = torch.argsort(gl_sort_scores, dim=1, stable=True)

        # ========= GPU批量拼接并左对齐（无for循环） =========
        # 基本参数
        N = start_quad_flat.shape[0]
        Lmax = 128
        # 对三段进行左对齐重排
        lane_left = lane_waypoints_tensor.gather(1, lane_order.unsqueeze(-1).expand(-1, -1, 2))  # (N_lane,W,2)
        wg_left = waypoint_graph_coords.gather(1, wg_order.unsqueeze(-1).expand(-1, -1, 2))      # (N_wg,wg_W,2)
        gl_left = goal_lane_waypoints_tensor.gather(1, gl_order.unsqueeze(-1).expand(-1, -1, 2)) # (N_gl,gl_W,2)
        
        # 组装大矩阵: [起点(1)] + lane(W1) + wg(W2) + gl(W3) + [终点(1)]
        segs = [
            start_centers.unsqueeze(1),
            lane_left,
            wg_left,
            gl_left,
            goal_centers.unsqueeze(1)
        ]
        # 统一各段批大小为N（已有均为N维度）
        concat_all = torch.cat(segs, dim=1)  # (N, T, 2)
        # 有效掩码
        valid_all = (concat_all[..., 0] != -1) & (concat_all[..., 1] != -1)  # (N, T)
        T = concat_all.shape[1]
        # 行内左对齐（保持行顺序稳定）
        col_idx = torch.arange(T, device=device).unsqueeze(0).expand(N, -1)
        order_score = (~valid_all).long() * T + col_idx
        order = torch.argsort(order_score, dim=1, stable=True)
        concat_left = concat_all.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))  # (N, T, 2)
        valid_left = valid_all.gather(1, order)
        valid_counts = valid_left.sum(dim=1)

        # 压缩到固定Lmax：短路径保持原序，长路径按累计距离采样，保留终点而不是硬截断。
        use_len = min(Lmax, T)
        path_flat = torch.full((N, Lmax, 2), -1, dtype=torch.float32, device=device)
        path_flat[:, :use_len, :] = concat_left[:, :use_len, :]
        long_rows = valid_counts > Lmax
        if torch.any(long_rows):
            long_coords = concat_left[long_rows]
            long_valid = valid_left[long_rows]
            long_counts = valid_counts[long_rows].to(torch.long)
            pair_valid = long_valid[:, 1:] & long_valid[:, :-1]
            step_dist = torch.norm(long_coords[:, 1:] - long_coords[:, :-1], dim=-1)
            step_dist = torch.where(pair_valid, step_dist, torch.zeros_like(step_dist))
            cum_dist = torch.cat(
                [torch.zeros((long_coords.shape[0], 1), dtype=torch.float32, device=device), torch.cumsum(step_dist, dim=1)],
                dim=1,
            )
            last_idx = torch.clamp(long_counts - 1, min=0)
            total_dist = cum_dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            sample_t = torch.linspace(0.0, 1.0, Lmax, dtype=torch.float32, device=device).unsqueeze(0)
            targets = total_dist.unsqueeze(1) * sample_t
            hi = torch.searchsorted(cum_dist.contiguous(), targets.contiguous(), right=False)
            hi = torch.clamp(hi, 0, T - 1)
            lo = torch.clamp(hi - 1, 0, T - 1)
            hi_dist = torch.abs(cum_dist.gather(1, hi) - targets)
            lo_dist = torch.abs(targets - cum_dist.gather(1, lo))
            sample_idx = torch.where(hi_dist < lo_dist, hi, lo)
            sample_idx[:, -1] = last_idx
            path_flat[long_rows] = long_coords.gather(1, sample_idx.unsqueeze(-1).expand(-1, -1, 2))
        # 形状还原为 (B,M,Lmax,2)
        path = path_flat.view(start_quad_id.shape[0], start_quad_id.shape[1], Lmax, 2)
        # 在这里已经得到了start_ids，end_ids。通过cross_data内部的"waypoint_graph"得到两个节点之间的其它节点。
        plan_total_time = time.time() - plan_start_time
        if self.verbose:
            print(f"plan_path总耗时: {plan_total_time:.4f}秒")
        return path
        
#=======================查找工具函数=======================

    def get_waypoint_coords(self, indices: torch.Tensor) -> torch.Tensor:
        """获取指定索引的waypoint坐标"""
        if self.global_w_lane_waypoints is None:
            return torch.zeros(len(indices), 2, device=self.device)
        x = self.global_w_lane_waypoints['x'][indices]
        y = self.global_w_lane_waypoints['y'][indices]
        return torch.stack([x, y], dim=1)
    
    def get_quad_centers(self, quad_ids: torch.Tensor) -> torch.Tensor:
        """获取指定quad_id的中心点坐标"""
        # 目前兼容(B, M)和(N=B*M,)两种形状
        # 保存原始形状
        original_shape = quad_ids.shape
        
        # 如果输入是2D张量 (B, M)，展平为1D
        if quad_ids.dim() == 2:
            quad_ids_flat = quad_ids.flatten()
        else:
            quad_ids_flat = quad_ids

        # 处理无效的quad_id（-1）
        valid_mask = quad_ids_flat >= 0
        centers_flat = torch.full((quad_ids_flat.shape[0], 2), -1.0, dtype=torch.float32, device=self.device)
        
        if valid_mask.any():
            valid_quad_ids = quad_ids_flat[valid_mask]
            # 获取quad的中心点坐标
            center_x = self.quads_info['center_x'][valid_quad_ids]
            center_y = self.quads_info['center_y'][valid_quad_ids]
            centers_flat[valid_mask] = torch.stack([center_x, center_y], dim=1)
        
        # 如果原始输入是2D，重塑为 (B, M, 2)
        if len(original_shape) == 2:
            return centers_flat.view(original_shape[0], original_shape[1], 2)
        else:
            return centers_flat

    def _map_graph_triplets_to_groups(self, triplets: torch.Tensor) -> torch.Tensor:
        """Map [cross, road, lane] triplets to WaypointGraphGPU group ids."""
        if self.waypoint_graph_gpu is None or triplets.numel() == 0:
            return torch.full(triplets.shape[:-1], -1, dtype=torch.long, device=self.device)
        wg = self.waypoint_graph_gpu
        if wg.triplet_unique_keys.numel() == 0:
            return torch.full(triplets.shape[:-1], -1, dtype=torch.long, device=self.device)

        original_shape = triplets.shape[:-1]
        flat = triplets.to(device=self.device, dtype=torch.long).reshape(-1, 3)
        off = flat - wg._triplet_mins.view(1, 3)
        keys = off[:, 0] * wg._triplet_mul_cross + off[:, 1] * wg._triplet_mul_road + off[:, 2]
        pos = torch.searchsorted(wg.triplet_unique_keys, keys)
        pos = torch.clamp(pos, 0, wg.triplet_unique_keys.numel() - 1)
        matched = wg.triplet_unique_keys[pos] == keys
        groups = torch.full((flat.shape[0],), -1, dtype=torch.long, device=self.device)
        groups[matched] = pos[matched]
        return groups.view(original_shape)

    def _graph_node_for_groups(self, groups: torch.Tensor) -> torch.Tensor:
        """Return the first concrete graph node for each triplet group."""
        if self.waypoint_graph_gpu is None or groups.numel() == 0:
            return torch.full_like(groups, -1, dtype=torch.long)
        wg = self.waypoint_graph_gpu
        out = torch.full(groups.shape, -1, dtype=torch.long, device=self.device)
        valid = (groups >= 0) & (groups < wg.triplet_group_starts.numel())
        if valid.any():
            safe_groups = groups[valid].to(torch.long)
            starts = wg.triplet_group_starts[safe_groups]
            counts = wg.triplet_group_counts[safe_groups]
            has_node = counts > 0
            valid_positions = torch.where(valid)[0]
            if has_node.any():
                out[valid_positions[has_node]] = wg.nodes_sorted_by_triplet[starts[has_node]]
        return out

    def _build_quad_to_waypoint_lookup(self, map_quad_ids: torch.Tensor, waypoint_values: torch.Tensor) -> torch.Tensor:
        """Build a dense quad-index -> W_lane waypoint lookup from map polyId keys."""
        num_quads = int(self.quads_info['center_x'].shape[0])
        lookup = torch.full((num_quads,), -1, dtype=torch.long, device=self.device)
        if map_quad_ids.numel() == 0 or waypoint_values.numel() == 0:
            return lookup

        poly_ids = self.quads_info.get('polyId', torch.arange(num_quads, device=self.device, dtype=torch.int32))
        for qid, wp_id in zip(map_quad_ids.detach().cpu().tolist(), waypoint_values.detach().cpu().tolist()):
            qid_int = int(qid)
            wp_int = int(wp_id)
            match = torch.where(poly_ids == qid_int)[0]
            if match.numel() > 0:
                lookup[match[0].long()] = wp_int
            elif 0 <= qid_int < num_quads:
                lookup[qid_int] = wp_int
        return lookup

    def _precompute_route_distance_tables(self):
        """Precompute lane waypoint metadata used for W_lane graph-route distances."""
        self._route_distance_ready = False
        if self.waypoint_graph_gpu is None or not hasattr(self, 'global_w_lane_waypoints'):
            return
        num_waypoints = int(self.global_w_lane_waypoints['x'].shape[0])
        if num_waypoints == 0:
            return

        road_ids = self.global_w_lane_waypoints['carla_waypoint_info']['road_id'].to(device=self.device, dtype=torch.long)
        lane_ids = self.global_w_lane_waypoints['carla_waypoint_info']['lane_id'].to(device=self.device, dtype=torch.long)
        start_triplets = torch.stack([
            torch.full_like(road_ids, self.LANE_START_NODE),
            road_ids,
            lane_ids,
        ], dim=1)
        end_triplets = torch.stack([
            torch.full_like(road_ids, self.LANE_END_NODE),
            road_ids,
            lane_ids,
        ], dim=1)
        self.w_lane_start_group = self._map_graph_triplets_to_groups(start_triplets)
        self.w_lane_end_group = self._map_graph_triplets_to_groups(end_triplets)
        self.w_lane_end_node_idx = self._graph_node_for_groups(self.w_lane_end_group)
        self.w_lane_road_ids = road_ids
        self.w_lane_lane_ids = lane_ids
        self.w_lane_progress = torch.zeros(num_waypoints, dtype=torch.float32, device=self.device)
        self.w_lane_remaining_to_end = torch.zeros(num_waypoints, dtype=torch.float32, device=self.device)

        for lane_indices in self.lanes.values():
            if lane_indices.numel() == 0:
                continue
            lane_indices = lane_indices.to(device=self.device, dtype=torch.long)
            coords = self.get_waypoint_coords(lane_indices)
            if coords.shape[0] <= 1:
                progress = torch.zeros((coords.shape[0],), dtype=torch.float32, device=self.device)
            else:
                step = torch.norm(coords[1:] - coords[:-1], dim=1)
                progress = torch.cat([
                    torch.zeros((1,), dtype=torch.float32, device=self.device),
                    torch.cumsum(step, dim=0),
                ], dim=0)
            total = progress[-1] if progress.numel() > 0 else torch.tensor(0.0, device=self.device)
            self.w_lane_progress[lane_indices] = progress
            self.w_lane_remaining_to_end[lane_indices] = total - progress

        prev_lookup = self._build_quad_to_waypoint_lookup(
            getattr(self, 'quad_to_prev_waypoint_quad_ids', torch.empty(0, device=self.device, dtype=torch.int32)),
            getattr(self, 'quad_to_prev_waypoint_values', torch.empty(0, device=self.device, dtype=torch.int32)),
        )
        next_lookup = self._build_quad_to_waypoint_lookup(
            getattr(self, 'quad_to_next_waypoint_quad_ids', torch.empty(0, device=self.device, dtype=torch.int32)),
            getattr(self, 'quad_to_next_waypoint_values', torch.empty(0, device=self.device, dtype=torch.int32)),
        )

        quad_centers = torch.stack([self.quads_info['center_x'], self.quads_info['center_y']], dim=1)
        prev_valid = (prev_lookup >= 0) & (prev_lookup < num_waypoints)
        next_valid = (next_lookup >= 0) & (next_lookup < num_waypoints)
        safe_prev = torch.clamp(prev_lookup, 0, max(num_waypoints - 1, 0))
        safe_next = torch.clamp(next_lookup, 0, max(num_waypoints - 1, 0))
        waypoint_xy = self.get_waypoint_coords(torch.arange(num_waypoints, device=self.device, dtype=torch.long))
        prev_dist = torch.norm(waypoint_xy[safe_prev] - quad_centers, dim=1)
        next_dist = torch.norm(waypoint_xy[safe_next] - quad_centers, dim=1)
        prev_dist = prev_dist.masked_fill(~prev_valid, float('inf'))
        next_dist = next_dist.masked_fill(~next_valid, float('inf'))
        use_next = next_dist < prev_dist
        self.quad_goal_waypoint = torch.where(use_next, next_lookup, prev_lookup).to(torch.long)
        self.quad_goal_waypoint = torch.where(
            torch.isfinite(torch.minimum(prev_dist, next_dist)),
            self.quad_goal_waypoint,
            torch.full_like(self.quad_goal_waypoint, -1),
        )
        self._route_distance_ready = True

    def route_distances_from_w_lanes_to_goal_quads(self, w_lane_ids: torch.Tensor, goal_quad_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute graph-route distance from each observed W_lane waypoint to the current goal quad.

        The distance follows lane direction, uses the waypoint graph between lane
        endpoints, and falls back to inf when either endpoint is not routable.
        """
        route_dist = torch.full(w_lane_ids.shape, float('inf'), dtype=torch.float32, device=self.device)
        if not self._route_distance_ready or self.waypoint_graph_gpu is None:
            return route_dist

        num_waypoints = int(self.w_lane_progress.shape[0])
        num_quads = int(self.quad_goal_waypoint.shape[0])
        if num_waypoints == 0 or num_quads == 0:
            return route_dist

        w_lane_ids = w_lane_ids.to(device=self.device, dtype=torch.long)
        goal_quad_ids = goal_quad_ids.to(device=self.device, dtype=torch.long)
        source_valid = (w_lane_ids >= 0) & (w_lane_ids < num_waypoints)
        goal_valid = (goal_quad_ids >= 0) & (goal_quad_ids < num_quads)
        safe_w = torch.clamp(w_lane_ids, 0, num_waypoints - 1)
        safe_goal_quad = torch.clamp(goal_quad_ids, 0, num_quads - 1)

        goal_wp = self.quad_goal_waypoint[safe_goal_quad]
        goal_valid = goal_valid & (goal_wp >= 0) & (goal_wp < num_waypoints)
        safe_goal_wp = torch.clamp(goal_wp, 0, num_waypoints - 1)

        target_start_group = self.w_lane_start_group[safe_goal_wp].unsqueeze(-1)
        source_end_node = self.w_lane_end_node_idx[safe_w]
        graph_valid = (
            source_valid
            & goal_valid.unsqueeze(-1)
            & (target_start_group >= 0)
            & (source_end_node >= 0)
        )
        if graph_valid.any():
            expanded_target_group = target_start_group.expand_as(source_end_node)
            graph_values = self.waypoint_graph_gpu.end_dist_tensor[
                expanded_target_group[graph_valid].to(torch.long),
                source_end_node[graph_valid].to(torch.long),
            ]
            route_dist[graph_valid] = graph_values

        source_progress = self.w_lane_progress[safe_w]
        source_remaining = self.w_lane_remaining_to_end[safe_w]
        goal_progress = self.w_lane_progress[safe_goal_wp].unsqueeze(-1)
        route_dist = source_remaining + route_dist + goal_progress

        same_lane = (
            source_valid
            & goal_valid.unsqueeze(-1)
            & (self.w_lane_road_ids[safe_w] == self.w_lane_road_ids[safe_goal_wp].unsqueeze(-1))
            & (self.w_lane_lane_ids[safe_w] == self.w_lane_lane_ids[safe_goal_wp].unsqueeze(-1))
        )
        direct_forward = goal_progress - source_progress
        use_direct = same_lane & (direct_forward >= 0.0)
        route_dist = torch.where(use_direct, direct_forward, route_dist)
        return torch.where(source_valid & goal_valid.unsqueeze(-1), route_dist, torch.full_like(route_dist, float('inf')))
    
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

    def _write_indexed_coords(
        self,
        target_indices: torch.Tensor,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        output_tensor: torch.Tensor,
        output_lengths: torch.Tensor,
        coord_getter,
    ):
        if target_indices.numel() == 0 or source_ids.numel() == 0:
            return

        target_indices = target_indices.to(device=self.device, dtype=torch.long)
        source_ids = source_ids.to(device=self.device, dtype=torch.long)
        if source_ids.dim() == 1:
            source_ids = source_ids.unsqueeze(1)
        R, S = source_ids.shape
        max_out = output_tensor.shape[1]
        valid = source_ids >= 0
        ranks = valid.long().cumsum(dim=1) - 1
        write_mask = valid & (ranks < max_out)
        if not bool(write_mask.any().item()):
            output_lengths[target_indices] = 0
            return

        rows = target_indices.unsqueeze(1).expand(R, S)[write_mask]
        out_cols = ranks[write_mask]
        coords = coord_getter(source_ids[write_mask])
        output_tensor[rows, out_cols] = coords
        output_lengths[target_indices] = torch.clamp(valid.sum(dim=1), max=max_out).to(output_lengths.dtype)

    def _overwrite_quad_lane_endpoint_segments(
        self,
        quad_ids: torch.Tensor,
        output_tensor: torch.Tensor,
        output_lengths: torch.Tensor,
        use_next_waypoint: bool,
    ):
        if quad_ids.numel() == 0:
            return
        map_quad_ids = self.quad_to_next_waypoint_quad_ids if use_next_waypoint else self.quad_to_prev_waypoint_quad_ids
        map_waypoint_values = self.quad_to_next_waypoint_values if use_next_waypoint else self.quad_to_prev_waypoint_values
        if map_quad_ids.numel() == 0:
            return

        quad_flat = quad_ids.to(device=self.device, dtype=torch.int32).view(-1)
        matches = quad_flat.unsqueeze(1) == map_quad_ids.unsqueeze(0)
        matched = torch.where(matches)
        if len(matched[0]) == 0:
            return

        target_rows = matched[0]
        waypoint_ids = map_waypoint_values[matched[1]]
        waypoint_matches = waypoint_ids.unsqueeze(1) == self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)
        valid_waypoints = waypoint_matches.any(dim=1)
        if not bool(valid_waypoints.any().item()):
            return

        valid_idx = torch.where(valid_waypoints)[0]
        target_rows = target_rows[valid_idx]
        lane_row_idx = waypoint_matches.int().argmax(dim=1)[valid_idx]
        lane_indices_tensor = self.waypoint_to_lane['lane_indices_tensor'][lane_row_idx]
        lane_lengths = self.waypoint_to_lane['lane_lengths'][lane_row_idx]
        keep = torch.where(lane_lengths > 0)[0]
        if keep.numel() == 0:
            return

        target_rows = target_rows[keep]
        lane_indices_tensor = lane_indices_tensor[keep]
        lane_lengths = lane_lengths[keep].to(torch.long)

        max_lane_length = lane_indices_tensor.shape[1]
        lane_valid = lane_indices_tensor >= 0
        safe_lane_ids = torch.clamp(lane_indices_tensor.to(torch.long), min=0)
        lane_coords = self.get_waypoint_coords(safe_lane_ids.reshape(-1)).view(lane_indices_tensor.shape[0], max_lane_length, 2)
        quad_centers = self.get_quad_centers(quad_flat[target_rows].to(torch.long))
        lane_dist = torch.norm(lane_coords - quad_centers.unsqueeze(1), dim=2)
        lane_dist = lane_dist.masked_fill(~lane_valid, float('inf'))
        nearest_pos = torch.argmin(lane_dist, dim=1).to(torch.long)

        prev_pos = torch.clamp(nearest_pos - 1, min=0)
        next_pos = torch.minimum(nearest_pos + 1, lane_lengths - 1)
        rows = torch.arange(nearest_pos.shape[0], device=self.device)
        nearest_xy = lane_coords[rows, nearest_pos]
        prev_xy = lane_coords[rows, prev_pos]
        next_xy = lane_coords[rows, next_pos]
        tangent = next_xy - prev_xy
        along = ((quad_centers - nearest_xy) * tangent).sum(dim=1)
        if use_next_waypoint:
            positions = torch.where(along > 0, next_pos, nearest_pos)
        else:
            positions = torch.where(along < 0, prev_pos, nearest_pos)

        position_indices = torch.arange(max_lane_length, device=self.device).unsqueeze(0).expand(positions.shape[0], -1)
        if use_next_waypoint:
            valid_mask = (position_indices >= positions.unsqueeze(1)) & (lane_indices_tensor >= 0)
        else:
            valid_mask = (position_indices <= positions.unsqueeze(1)) & (lane_indices_tensor >= 0)
        source_ids = torch.where(valid_mask, lane_indices_tensor, torch.full_like(lane_indices_tensor, -1))
        source_lengths = valid_mask.sum(dim=1)

        output_tensor[target_rows.long()] = -1.0
        output_lengths[target_rows.long()] = 0
        self._write_indexed_coords(
            target_rows,
            source_ids,
            source_lengths,
            output_tensor,
            output_lengths,
            self.get_waypoint_coords,
        )

    def _fill_endpoint_ids_from_quad_waypoints(
        self,
        quad_ids: torch.Tensor,
        ids_tensor: torch.Tensor,
        use_next_waypoint: bool,
        endpoint_code: int,
        row_mask: Optional[torch.Tensor] = None,
    ):
        if quad_ids.numel() == 0:
            return
        map_quad_ids = self.quad_to_next_waypoint_quad_ids if use_next_waypoint else self.quad_to_prev_waypoint_quad_ids
        map_waypoint_values = self.quad_to_next_waypoint_values if use_next_waypoint else self.quad_to_prev_waypoint_values
        if map_quad_ids.numel() == 0:
            return

        quad_flat = quad_ids.to(device=self.device, dtype=torch.int32).view(-1)
        rows = torch.arange(quad_flat.numel(), device=self.device, dtype=torch.long)
        if row_mask is not None:
            rows = rows[row_mask.to(device=self.device, dtype=torch.bool).view(-1)]
        if rows.numel() == 0:
            return

        matches = quad_flat[rows].unsqueeze(1) == map_quad_ids.unsqueeze(0)
        matched = torch.where(matches)
        if len(matched[0]) == 0:
            return

        target_rows = rows[matched[0]]
        waypoint_ids = map_waypoint_values[matched[1]]
        waypoint_matches = waypoint_ids.unsqueeze(1) == self.waypoint_to_lane['waypoint_ids'].unsqueeze(0)
        valid_waypoints = waypoint_matches.any(dim=1)
        if not bool(valid_waypoints.any().item()):
            return

        valid_idx = torch.where(valid_waypoints)[0]
        lane_row_idx = waypoint_matches.int().argmax(dim=1)[valid_idx]
        target_rows = target_rows[valid_idx]
        ids_tensor[target_rows, 0] = int(endpoint_code)
        ids_tensor[target_rows, 1] = self.waypoint_to_lane['road_ids'][lane_row_idx].to(ids_tensor.dtype)
        ids_tensor[target_rows, 2] = self.waypoint_to_lane['lane_ids'][lane_row_idx].to(ids_tensor.dtype)

    def _dedupe_coords(self, coords: List[Tuple[float, float]], eps: float = 1e-4) -> List[Tuple[float, float]]:
        if not coords:
            return coords
        deduped = [coords[0]]
        eps2 = eps * eps
        for x, y in coords[1:]:
            lx, ly = deduped[-1]
            if (x - lx) * (x - lx) + (y - ly) * (y - ly) > eps2:
                deduped.append((x, y))
        return deduped

    def _coords_path_length(self, coords: List[Tuple[float, float]]) -> float:
        if len(coords) < 2:
            return 0.0
        total = 0.0
        last_x, last_y = coords[0]
        for x, y in coords[1:]:
            dx = x - last_x
            dy = y - last_y
            total += float((dx * dx + dy * dy) ** 0.5)
            last_x, last_y = x, y
        return total

    def _build_lane_route_graph(self, raw_cross_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[List[Tuple[float, float]]]]:
        if not self.lanes:
            return raw_cross_data.get('waypoint_graph', {'nodes': [], 'edges': []}), []

        nodes: List[List[Any]] = []
        node_indices: Dict[Tuple[int, int, str], int] = {}
        lane_coords: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

        def add_node(road_id: int, lane_id: int, endpoint: str, xy: Tuple[float, float]) -> Tuple[int, int, str]:
            key = (int(road_id), int(lane_id), endpoint)
            if key not in node_indices:
                cross_code = self.LANE_START_NODE if endpoint == 'start' else self.LANE_END_NODE
                node_indices[key] = len(nodes)
                nodes.append([cross_code, int(road_id), int(lane_id), float(xy[0]), float(xy[1]), endpoint])
            return key

        for road_id, lane_id in sorted(self.lanes.keys()):
            indices = self.lanes[(road_id, lane_id)]
            coords_t = self.get_waypoint_coords(indices)
            coords = [(float(x), float(y)) for x, y in coords_t.detach().cpu().tolist()]
            coords = self._dedupe_coords(coords)
            if not coords:
                continue
            lane_key = (int(road_id), int(lane_id))
            lane_coords[lane_key] = coords
            add_node(road_id, lane_id, 'start', coords[0])
            add_node(road_id, lane_id, 'end', coords[-1])

        edges: List[List[Any]] = []
        edge_coords: List[List[Tuple[float, float]]] = []
        edge_pairs = set()

        def node_xy(key: Tuple[int, int, str]) -> Tuple[float, float]:
            node = nodes[node_indices[key]]
            return (float(node[3]), float(node[4]))

        def add_edge(
            u_key: Tuple[int, int, str],
            v_key: Tuple[int, int, str],
            coords_after_u: List[Tuple[float, float]],
            extra_weight: float = 0.0,
        ):
            if u_key not in node_indices or v_key not in node_indices:
                return
            pair = (u_key, v_key)
            if pair in edge_pairs:
                return
            edge_pairs.add(pair)
            u_node = nodes[node_indices[u_key]]
            v_node = nodes[node_indices[v_key]]
            v_xy = node_xy(v_key)
            coords = self._dedupe_coords(coords_after_u or [v_xy])
            if not coords:
                coords = [v_xy]
            if (coords[-1][0] - v_xy[0]) ** 2 + (coords[-1][1] - v_xy[1]) ** 2 > 1e-6:
                coords.append(v_xy)
            weight = self._coords_path_length([node_xy(u_key)] + coords) + float(extra_weight)
            edges.append([u_node, v_node, float(weight)])
            edge_coords.append(coords)

        for lane_key, coords in lane_coords.items():
            expansion = coords[1:] if len(coords) > 1 else [coords[-1]]
            add_edge((lane_key[0], lane_key[1], 'start'), (lane_key[0], lane_key[1], 'end'), expansion)

        for key, cross_info in raw_cross_data.items():
            if not key.startswith('cross_'):
                continue
            for path in cross_info.get('paths', []):
                from_end = path.get('from_end_waypoint', {})
                to_start = path.get('to_start_waypoint', {})
                u_key = (int(from_end.get('road_id', 0)), int(from_end.get('lane_id', 0)), 'end')
                v_key = (int(to_start.get('road_id', 0)), int(to_start.get('lane_id', 0)), 'start')
                if u_key not in node_indices or v_key not in node_indices:
                    continue
                coords: List[Tuple[float, float]] = []
                quad_ids = torch.tensor(path.get('path_quad_ids', []), dtype=torch.long, device=self.device)
                if quad_ids.numel() > 0:
                    centers = self.get_quad_centers(quad_ids).detach().cpu().tolist()
                    coords.extend((float(x), float(y)) for x, y in centers)
                add_edge(u_key, v_key, coords)

        start_nodes = [
            (lane_key, (lane_key[0], lane_key[1], 'start'), coords[0])
            for lane_key, coords in lane_coords.items()
        ]
        for lane_key, coords in lane_coords.items():
            u_key = (lane_key[0], lane_key[1], 'end')
            ux, uy = coords[-1]
            for start_lane_key, v_key, (vx, vy) in start_nodes:
                if start_lane_key == lane_key:
                    continue
                dx = ux - vx
                dy = uy - vy
                dist = float((dx * dx + dy * dy) ** 0.5)
                if dist <= self.LANE_CONNECTOR_THRESHOLD:
                    add_edge(u_key, v_key, [(vx, vy)])
                elif dist <= self.LANE_UTURN_THRESHOLD:
                    add_edge(u_key, v_key, [(vx, vy)], extra_weight=self.LANE_UTURN_PENALTY)

        return {'nodes': nodes, 'edges': edges}, edge_coords

    def _install_prebuilt_waypoint_graph_edge_expansions(self, edge_coords: List[List[Tuple[float, float]]]):
        if self.waypoint_graph_gpu is None or not edge_coords:
            return
        max_len = max(len(coords) for coords in edge_coords)
        points = torch.full((len(edge_coords), max_len, 2), -1.0, dtype=torch.float32, device=self.device)
        lengths = torch.zeros(len(edge_coords), dtype=torch.int32, device=self.device)
        for edge_idx, coords in enumerate(edge_coords):
            coord_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
            points[edge_idx, :coord_tensor.shape[0]] = coord_tensor
            lengths[edge_idx] = coord_tensor.shape[0]
        self.waypoint_graph_gpu.set_edge_expansions(points, lengths)

    def _straight_edge_coords(self, u_node: List[Any], v_node: List[Any]) -> List[Tuple[float, float]]:
        road_id = int(u_node[1])
        lane_id = int(u_node[2])
        lane_indices = self.lanes.get((road_id, lane_id))
        v_xy = (float(v_node[3]), float(v_node[4]))
        if lane_indices is None or lane_indices.numel() == 0:
            return [v_xy]

        lane_x = self.global_w_lane_waypoints['x'][lane_indices]
        lane_y = self.global_w_lane_waypoints['y'][lane_indices]
        lane_xy = torch.stack([lane_x, lane_y], dim=1)
        u_xy_t = torch.tensor([float(u_node[3]), float(u_node[4])], dtype=lane_xy.dtype, device=self.device)
        v_xy_t = torch.tensor([v_xy[0], v_xy[1]], dtype=lane_xy.dtype, device=self.device)
        u_pos = int(torch.argmin(torch.norm(lane_xy - u_xy_t.unsqueeze(0), dim=1)).item())
        v_pos = int(torch.argmin(torch.norm(lane_xy - v_xy_t.unsqueeze(0), dim=1)).item())

        if u_pos < v_pos:
            selected = lane_xy[u_pos + 1:v_pos + 1]
        elif u_pos > v_pos:
            selected = torch.flip(lane_xy[v_pos:u_pos], dims=[0])
        else:
            selected = lane_xy[u_pos:u_pos + 1]

        coords = [(float(x), float(y)) for x, y in selected.detach().cpu().tolist()]
        if not coords or (coords[-1][0] - v_xy[0]) ** 2 + (coords[-1][1] - v_xy[1]) ** 2 > 1e-6:
            coords.append(v_xy)
        return self._dedupe_coords(coords)

    def _cross_edge_coords(self, u_node: List[Any], v_node: List[Any], raw_cross_data: Dict[str, Any]) -> List[Tuple[float, float]]:
        v_xy = (float(v_node[3]), float(v_node[4]))
        if int(u_node[0]) != int(v_node[0]):
            return [v_xy]
        cross_info = raw_cross_data.get(f"cross_{int(u_node[0])}", {})
        for path in cross_info.get('paths', []):
            from_end = path.get('from_end_waypoint', {})
            to_start = path.get('to_start_waypoint', {})
            if (
                int(from_end.get('road_id', 0)) == int(u_node[1])
                and int(from_end.get('lane_id', 0)) == int(u_node[2])
                and int(to_start.get('road_id', 0)) == int(v_node[1])
                and int(to_start.get('lane_id', 0)) == int(v_node[2])
            ):
                quad_ids = torch.tensor(path.get('path_quad_ids', []), dtype=torch.long, device=self.device)
                coords: List[Tuple[float, float]] = []
                if quad_ids.numel() > 0:
                    centers = self.get_quad_centers(quad_ids).detach().cpu().tolist()
                    coords.extend((float(x), float(y)) for x, y in centers)
                coords.append(v_xy)
                return self._dedupe_coords(coords)
        return [v_xy]

    def _install_waypoint_graph_edge_expansions(self, raw_cross_data: Dict[str, Any]):
        raw_edges = raw_cross_data.get('waypoint_graph', {}).get('edges', [])
        if self.waypoint_graph_gpu is None or not raw_edges:
            return

        edge_coords: List[List[Tuple[float, float]]] = []
        for u_node, v_node, _weight in raw_edges:
            if int(u_node[1]) == int(v_node[1]) and int(u_node[2]) == int(v_node[2]):
                coords = self._straight_edge_coords(u_node, v_node)
            else:
                coords = self._cross_edge_coords(u_node, v_node, raw_cross_data)
            edge_coords.append(coords or [(float(v_node[3]), float(v_node[4]))])

        max_len = max(len(coords) for coords in edge_coords)
        points = torch.full((len(edge_coords), max_len, 2), -1.0, dtype=torch.float32, device=self.device)
        lengths = torch.zeros(len(edge_coords), dtype=torch.int32, device=self.device)
        for edge_idx, coords in enumerate(edge_coords):
            coord_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
            points[edge_idx, :coord_tensor.shape[0]] = coord_tensor
            lengths[edge_idx] = coord_tensor.shape[0]
        self.waypoint_graph_gpu.set_edge_expansions(points, lengths)

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

class WaypointGraphGPU:
    """
    将 waypoint_graph 预处理为 GPU 常驻的稀疏入边表，并提供固定长度的批量最短路径生成功能。
    """
    def __init__(self, json_path: Optional[str] = None, device: Optional[str] = None, waypoint_graph: Optional[Dict[str, Any]] = None):
        # 1) 读取 JSON 并解析 waypoint_graph
        if waypoint_graph is not None:
            wg = waypoint_graph
        else:
            if json_path is None:
                raise ValueError("json_path or waypoint_graph is required")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'waypoint_graph' not in data:
                raise ValueError(f"文件中未找到 'waypoint_graph': {json_path}")
            wg = data['waypoint_graph']
        raw_nodes: List[List[Any]] = wg['nodes']
        raw_edges: List[List[Any]] = wg['edges']

        # 2) 向量化解析 nodes
        nodes_arr = np.array(raw_nodes, dtype=object)  # [N,6]
        N = int(nodes_arr.shape[0])
        cross_col = nodes_arr[:, 0].astype(np.int64)
        road_col = nodes_arr[:, 1].astype(np.int64)
        lane_col = nodes_arr[:, 2].astype(np.int64)
        x_col = nodes_arr[:, 3].astype(np.float64)
        y_col = nodes_arr[:, 4].astype(np.float64)
        type_col = nodes_arr[:, 5].astype(str)

        # 三元组张量 [N,3]
        node_triplets_np = np.stack([cross_col, road_col, lane_col], axis=1)
        # 构造节点唯一键（字符串），用于与边端点对齐
        # 使用固定小数格式保证稳定性
        x_str = np.char.mod('%.6f', x_col)
        y_str = np.char.mod('%.6f', y_col)
        key_nodes = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        cross_col.astype(str), '|'), road_col.astype(str)), '|'), lane_col.astype(str)), '|')
        key_nodes = np.char.add(np.char.add(np.char.add(np.char.add(key_nodes, x_str), '|'), y_str), '|')
        key_nodes = np.char.add(key_nodes, type_col)

        # 3) 向量化解析 edges
        edges_arr = np.array(raw_edges, dtype=object)  # [E,3]
        # 提取 u/v 节点矩阵 [E,6]
        u_arr = np.stack(edges_arr[:, 0])
        v_arr = np.stack(edges_arr[:, 1])
        w_arr = edges_arr[:, 2].astype(np.float32)
        # 为 u/v 节点生成键
        u_key = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        u_arr[:, 0].astype(str), '|'), u_arr[:, 1].astype(str)), '|'), u_arr[:, 2].astype(str)), '|')
        u_x_str = np.char.mod('%.6f', u_arr[:, 3].astype(np.float64))
        u_y_str = np.char.mod('%.6f', u_arr[:, 4].astype(np.float64))
        u_key = np.char.add(np.char.add(np.char.add(np.char.add(u_key, u_x_str), '|'), u_y_str), '|')
        u_key = np.char.add(u_key, u_arr[:, 5].astype(str))
        v_key = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        v_arr[:, 0].astype(str), '|'), v_arr[:, 1].astype(str)), '|'), v_arr[:, 2].astype(str)), '|')
        v_x_str = np.char.mod('%.6f', v_arr[:, 3].astype(np.float64))
        v_y_str = np.char.mod('%.6f', v_arr[:, 4].astype(np.float64))
        v_key = np.char.add(np.char.add(np.char.add(np.char.add(v_key, v_x_str), '|'), v_y_str), '|')
        v_key = np.char.add(v_key, v_arr[:, 5].astype(str))

        # 4) 用排序 + searchsorted 将 u/v 键映射到节点索引（避免 Python 字典循环）
        order_nodes = np.argsort(key_nodes)
        key_nodes_sorted = key_nodes[order_nodes]
        pos_u = np.searchsorted(key_nodes_sorted, u_key)
        pos_v = np.searchsorted(key_nodes_sorted, v_key)
        u_idx = order_nodes[pos_u].astype(np.int64)
        v_idx = order_nodes[pos_v].astype(np.int64)

        # 5) 基于 v_idx 分组构建稠密入边表 [N, max_in_deg]
        deg = np.bincount(v_idx, minlength=N)
        max_in_deg = int(deg.max()) if N > 0 else 1
        if max_in_deg == 0:
            max_in_deg = 1
        # 无效位置用0占位，配合 inf 权重在松弛时不会被选中
        incoming_src_idx_np = np.full((N, max_in_deg), 0, dtype=np.int64)
        incoming_w_np = np.full((N, max_in_deg), np.inf, dtype=np.float32)
        if max_in_deg > 0 and v_idx.size > 0:
            order_e = np.argsort(v_idx, kind='stable')
            v_sorted = v_idx[order_e]
            u_sorted = u_idx[order_e]
            w_sorted = w_arr[order_e]
            # 每个节点的起始偏移（按排序后）
            starts = np.cumsum(np.concatenate(([0], deg[:-1])))
            # 逐节点切片填充（这个环节循环的是节点数而非边数，且纯切片赋值，已较轻）
            nz_nodes = np.nonzero(deg)[0]
            for v in nz_nodes:
                s = starts[v]
                d = deg[v]
                incoming_src_idx_np[v, :d] = u_sorted[s:s + d]
                incoming_w_np[v, :d] = w_sorted[s:s + d]

        # 3) 迁移到设备并保存
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.incoming_src_idx = torch.from_numpy(incoming_src_idx_np).to(device)  # [N, M]
        self.incoming_w = torch.from_numpy(incoming_w_np).to(device)              # [N, M]
        self.node_triplets = torch.from_numpy(node_triplets_np.astype(np.int64)).to(device)
        # 保存每个节点的坐标 (x,y)，用于直接索引生成坐标路径
        self.node_xy = torch.from_numpy(np.stack([x_col, y_col], axis=1).astype(np.float32)).to(device)
        self.edge_u_idx = torch.from_numpy(u_idx.astype(np.int64)).to(device)
        self.edge_v_idx = torch.from_numpy(v_idx.astype(np.int64)).to(device)
        self.edge_id_lookup = torch.full((N, N), -1, dtype=torch.long, device=device)
        if u_idx.size > 0:
            edge_ids = torch.arange(u_idx.size, dtype=torch.long, device=device)
            self.edge_id_lookup[self.edge_u_idx, self.edge_v_idx] = edge_ids
        self.edge_expansion_points = None
        self.edge_expansion_lengths = None

        # 同步构建出边表 [N, max_out_deg]（用于终点反向DP+前向next跳转）
        deg_out = np.bincount(u_idx, minlength=N)
        max_out_deg = int(deg_out.max()) if N > 0 else 1
        if max_out_deg == 0:
            max_out_deg = 1
        out_tgt_idx_np = np.full((N, max_out_deg), 0, dtype=np.int64)
        out_w_np = np.full((N, max_out_deg), np.inf, dtype=np.float32)
        if max_out_deg > 0 and u_idx.size > 0:
            order_e2 = np.argsort(u_idx, kind='stable')
            u_sorted2 = u_idx[order_e2]
            v_sorted2 = v_idx[order_e2]
            w_sorted2 = w_arr[order_e2]
            starts_out = np.cumsum(np.concatenate(([0], deg_out[:-1])))
            nz_u = np.nonzero(deg_out)[0]
            for u in nz_u:
                s = starts_out[u]
                d = deg_out[u]
                out_tgt_idx_np[u, :d] = v_sorted2[s:s + d]
                out_w_np[u, :d] = w_sorted2[s:s + d]
        self.outgoing_tgt_idx = torch.from_numpy(out_tgt_idx_np).to(device)  # [N, M_out]
        self.outgoing_w = torch.from_numpy(out_w_np).to(device)              # [N, M_out]


        # 4) 纯GPU三元组分组结构（唯一键、分组起始/大小、排序后的节点索引）
        mins = self.node_triplets.min(dim=0).values.to(torch.int64)
        maxs = self.node_triplets.max(dim=0).values.to(torch.int64)
        ranges = (maxs - mins + 1).to(torch.int64)
        mul_cross = (ranges[1] * ranges[2]).to(torch.int64)
        mul_road = ranges[2].to(torch.int64)
        off = (self.node_triplets - mins)
        keys_nodes = off[:, 0] * mul_cross + off[:, 1] * mul_road + off[:, 2]
        keys_sorted, order_nodes2 = torch.sort(keys_nodes)
        uniq_keys_t, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.int64), counts[:-1]]), dim=0)
        self._triplet_mins = mins
        self._triplet_mul_cross = mul_cross
        self._triplet_mul_road = mul_road
        self.triplet_unique_keys = uniq_keys_t
        self.triplet_group_starts = starts
        self.triplet_group_counts = counts
        self.nodes_sorted_by_triplet = order_nodes2
        group_centers = torch.empty((uniq_keys_t.numel(), 2), dtype=self.node_xy.dtype, device=device)
        for g in range(uniq_keys_t.numel()):
            start_g = int(starts[g].item())
            count_g = int(counts[g].item())
            nodes_g = order_nodes2[start_g:start_g + count_g]
            group_centers[g] = self.node_xy[nodes_g].mean(dim=0)
        self.triplet_group_centers = group_centers
        
        # 缓存（offset位掩码、终点树）
        self._offset_masks_cache = {}
        # 预计算所有终点组的最短路径树，使用张量存储
        self._precompute_all_end_trees_tensor()

    def _precompute_all_end_trees_tensor(self):
        """为每个终点组预计算最短路径树，使用张量存储"""
        device = self.device
        U = self.triplet_unique_keys.numel()
        N = self.outgoing_tgt_idx.size(0)

        # 预分配张量存储所有终点组的结果
        # dist_tensor: [U, N] - 每个终点组到所有节点的最短距离
        # next_tensor: [U, N] - 每个终点组的最短路径下一跳
        dist_tensor = torch.full((U, N), float('inf'), dtype=torch.float32, device=device)
        next_tensor = torch.full((U, N), -1, dtype=torch.long, device=device)
        # 批量构建所有终点组的最短路径树
        for g in range(U):
            dist_g, next_g = self._build_end_tree(g)
            dist_tensor[g] = dist_g
            next_tensor[g] = next_g
        self.end_dist_tensor = dist_tensor  # [U, N]
        self.end_next_tensor = next_tensor  # [U, N]

    def _build_end_tree(self, end_group_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """为给定终点组构建最短路径树"""
        device = self.device
        N = self.outgoing_tgt_idx.size(0)
        
        # 终点组内所有节点
        start = self.triplet_group_starts[end_group_id]
        count = self.triplet_group_counts[end_group_id]
        nodes = self.nodes_sorted_by_triplet[start:start + count]
        
        dist = torch.full((N,), float('inf'), dtype=torch.float32, device=device)
        next_idx = torch.full((N,), -1, dtype=torch.long, device=device)
        if count > 0:
            dist.index_fill_(0, nodes, 0.0)
        
        # Bellman-Ford 松弛
        for _ in range(max(1, N - 1)):
            tgt_d = dist[self.outgoing_tgt_idx]           # [N, M_out]
            cand = tgt_d + self.outgoing_w                # [N, M_out]
            new_d, min_pos = torch.min(cand, dim=1)       # [N]
            improved = new_d < dist
            if not torch.any(improved):
                break
            dist = torch.minimum(dist, new_d)
            best_v = self.outgoing_tgt_idx.gather(1, min_pos.view(-1, 1)).squeeze(1)  # [N]
            next_idx[improved] = best_v[improved]
        
        return dist, next_idx

    def _get_offset_masks(self, L: int) -> torch.Tensor:
        """
        返回形状 [K, L] 的 bool 掩码，第 k 行表示 offset 中第 k 个比特是否为 1。
        缓存到 (L, device) 键。
        """
        device = self.device
        L = int(L)
        K = int(max(1, (L - 1).bit_length()))
        key = (L, device)
        cached = self._offset_masks_cache.get(key, None)
        if cached is not None and cached.shape == (K, L):
            return cached
        offsets = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)  # [1,L]
        bits = torch.arange(K, device=device, dtype=torch.long).unsqueeze(1)     # [K,1]
        use = ((offsets >> bits) & 1).to(torch.bool)                              # [K,L]
        self._offset_masks_cache[key] = use
        return use

    def set_edge_expansions(self, points: torch.Tensor, lengths: torch.Tensor):
        if points.dim() != 3 or points.size(-1) != 2:
            raise ValueError(f"edge expansion points must be [E, K, 2], got {points.shape}")
        if points.size(0) != self.edge_u_idx.numel():
            raise ValueError(f"edge expansion count mismatch: {points.size(0)} vs {self.edge_u_idx.numel()}")
        if lengths.shape[0] != points.shape[0]:
            raise ValueError(f"edge expansion lengths mismatch: {lengths.shape} vs {points.shape}")
        self.edge_expansion_points = points.to(device=self.device, dtype=torch.float32)
        self.edge_expansion_lengths = lengths.to(device=self.device, dtype=torch.long)

    def expand_node_indices_to_coords(self, node_indices: torch.Tensor, fixed_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_indices.dim() != 2:
            raise ValueError(f"node_indices must be [B, L], got {node_indices.shape}")
        device = self.device
        B, L_nodes = node_indices.shape
        L_out = int(fixed_len)
        coords = torch.full((B, L_out, 2), -1.0, dtype=torch.float32, device=device)
        mask = torch.zeros((B, L_out), dtype=torch.bool, device=device)
        if B == 0 or L_out <= 0:
            return coords, mask

        node_indices = node_indices.to(device=device, dtype=torch.long)
        valid_nodes = node_indices >= 0
        first_valid = valid_nodes.any(dim=1)
        safe_first = torch.clamp(node_indices[:, 0], 0, self.node_xy.shape[0] - 1)
        start_coords = self.node_xy[safe_first].unsqueeze(1)
        start_coords = torch.where(first_valid.view(B, 1, 1), start_coords, torch.full_like(start_coords, -1.0))

        if (
            self.edge_expansion_points is None
            or self.edge_expansion_lengths is None
            or L_nodes <= 1
        ):
            safe_nodes = torch.clamp(node_indices, 0, self.node_xy.shape[0] - 1)
            node_coords = self.node_xy[safe_nodes]
            node_coords = torch.where(valid_nodes.unsqueeze(-1), node_coords, torch.full_like(node_coords, -1.0))
            use_len = min(L_out, L_nodes)
            coords[:, :use_len] = node_coords[:, :use_len]
            mask[:, :use_len] = valid_nodes[:, :use_len]
            return coords, mask

        u = node_indices[:, :-1]
        v = node_indices[:, 1:]
        pair_valid = (u >= 0) & (v >= 0)
        safe_u = torch.clamp(u, 0, self.edge_id_lookup.shape[0] - 1)
        safe_v = torch.clamp(v, 0, self.edge_id_lookup.shape[1] - 1)
        edge_ids = self.edge_id_lookup[safe_u, safe_v]
        edge_valid = pair_valid & (edge_ids >= 0)

        safe_edge_ids = torch.clamp(edge_ids, 0, self.edge_expansion_points.shape[0] - 1)
        edge_points = self.edge_expansion_points[safe_edge_ids]
        edge_lengths = self.edge_expansion_lengths[safe_edge_ids]
        edge_lengths = torch.where(edge_valid, edge_lengths, torch.zeros_like(edge_lengths))
        K = self.edge_expansion_points.shape[1]
        point_idx = torch.arange(K, device=device).view(1, 1, K)
        edge_point_valid = point_idx < edge_lengths.unsqueeze(-1)
        edge_points = torch.where(edge_point_valid.unsqueeze(-1), edge_points, torch.full_like(edge_points, -1.0))

        expanded = torch.cat([start_coords, edge_points.reshape(B, -1, 2)], dim=1)
        expanded_valid = (expanded[..., 0] != -1) & (expanded[..., 1] != -1)
        T = expanded.shape[1]
        col_idx = torch.arange(T, device=device).view(1, T).expand(B, -1)
        order_score = (~expanded_valid).long() * T + col_idx
        order = torch.argsort(order_score, dim=1, stable=True)
        expanded_left = expanded.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
        expanded_valid_left = expanded_valid.gather(1, order)

        use_len = min(L_out, T)
        coords[:, :use_len] = expanded_left[:, :use_len]
        mask[:, :use_len] = expanded_valid_left[:, :use_len]
        return coords, mask

    def batch_shortest_paths_fixed_len(self,
                                   start_ids: torch.Tensor,
                                   end_ids: torch.Tensor,
                                   fixed_len: int = 100,
                                   pad_value: int = -1,
                                   start_xy: Optional[torch.Tensor] = None,
                                   end_xy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回: (paths, mask, node_indices)
            paths: [B, L, 3] long，三元组 (cross, road, lane)，用 pad_value 填充
            mask:  [B, L] bool，True 表示该步有效
            node_indices: [B, L] long，节点索引，用于直接获取坐标
        Args:
            start_ids: [B, 3] long tensor，起点三元组 (cross, road, lane)
            end_ids: [B, 3] long tensor，终点三元组 (cross, road, lane)
        """
        if start_ids.shape != end_ids.shape:
            raise ValueError(f"start_ids 与 end_ids 形状必须一致，got {start_ids.shape} vs {end_ids.shape}")
        if start_ids.dim() != 2 or start_ids.size(1) != 3:
            raise ValueError(f"start_ids 必须是 [B, 3] 形状，got {start_ids.shape}")
        
        device = self.device
        N = self.outgoing_tgt_idx.size(0)
        U = self.end_dist_tensor.size(0)
        B = start_ids.size(0)
        L = int(fixed_len)
        
        # 确保张量在正确设备上
        start_tensor = start_ids.to(device=device, dtype=torch.long)
        end_tensor = end_ids.to(device=device, dtype=torch.long)
        
        # 输出张量
        triplets = torch.full((B, L, 3), pad_value, dtype=torch.long, device=device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        node_indices = torch.full((B, L), -1, dtype=torch.long, device=device)
        
        # 将三元组映射为组id
        s_off = (start_tensor - self._triplet_mins)
        s_keys = s_off[:, 0] * self._triplet_mul_cross + s_off[:, 1] * self._triplet_mul_road + s_off[:, 2]
        pos_s = torch.searchsorted(self.triplet_unique_keys, s_keys)
        pos_s = torch.clamp(pos_s, 0, U - 1)
        matched_s = self.triplet_unique_keys[pos_s] == s_keys  # [B]

        e_off = (end_tensor - self._triplet_mins)
        e_keys = e_off[:, 0] * self._triplet_mul_cross + e_off[:, 1] * self._triplet_mul_road + e_off[:, 2]
        pos_e = torch.searchsorted(self.triplet_unique_keys, e_keys)
        pos_e = torch.clamp(pos_e, 0, U - 1)
        matched_e = self.triplet_unique_keys[pos_e] == e_keys  # [B]

        # 预取 offsets 掩码
        use_masks = self._get_offset_masks(L)
        K = use_masks.shape[0]

        start_groups = torch.full((B,), -1, dtype=torch.long, device=device)
        end_groups = torch.full((B,), -1, dtype=torch.long, device=device)
        start_groups[matched_s] = pos_s[matched_s]
        end_groups[matched_e] = pos_e[matched_e]

        if start_xy is not None and self.triplet_group_centers.numel() > 0:
            start_xy_t = start_xy.to(device=device, dtype=self.triplet_group_centers.dtype).reshape(B, 2)
            missing_start = start_groups < 0
            if torch.any(missing_start):
                start_dist = torch.cdist(start_xy_t[missing_start], self.triplet_group_centers)
                start_groups[missing_start] = torch.argmin(start_dist, dim=1)

        if end_xy is not None and self.triplet_group_centers.numel() > 0:
            end_xy_t = end_xy.to(device=device, dtype=self.triplet_group_centers.dtype).reshape(B, 2)
            missing_end = end_groups < 0
            if torch.any(missing_end):
                end_dist = torch.cdist(end_xy_t[missing_end], self.triplet_group_centers)
                end_groups[missing_end] = torch.argmin(end_dist, dim=1)

        valid_b = torch.nonzero((start_groups >= 0) & (end_groups >= 0), as_tuple=False).squeeze(1)
        if valid_b.numel() > 0:
            effective_end_groups = end_groups[valid_b].clone()
            dist_groups = self.end_dist_tensor[effective_end_groups]
            start_pos_sel = start_groups[valid_b]
            start_node_idx = torch.full((valid_b.numel(),), -1, dtype=torch.long, device=device)

            g_starts = self.triplet_group_starts[start_pos_sel]
            g_counts = self.triplet_group_counts[start_pos_sel]
            max_c = int(g_counts.max().item()) if g_counts.numel() > 0 else 0

            if max_c > 0:
                ar = torch.arange(max_c, device=device, dtype=torch.int64)
                grid = g_starts.unsqueeze(1) + ar.unsqueeze(0)
                valid = ar.unsqueeze(0) < g_counts.unsqueeze(1)
                grid_clamped = torch.clamp(grid, 0, self.nodes_sorted_by_triplet.numel() - 1)
                nodes_mat = self.nodes_sorted_by_triplet[grid_clamped]
                dist_mat = dist_groups[torch.arange(valid_b.numel(), device=device).unsqueeze(1), nodes_mat]
                dist_mat = dist_mat.masked_fill(~valid, float('inf'))
                min_vals, min_pos = torch.min(dist_mat, dim=1)
                chosen = nodes_mat.gather(1, min_pos.unsqueeze(1)).squeeze(1)
                ok = torch.isfinite(min_vals)
                start_node_idx[ok] = chosen[ok]

            fallback_rows = torch.nonzero(start_node_idx < 0, as_tuple=False).squeeze(1)
            if fallback_rows.numel() > 0:
                for row in fallback_rows.tolist():
                    start_group = int(start_pos_sel[row].item())
                    group_start = int(self.triplet_group_starts[start_group].item())
                    group_count = int(self.triplet_group_counts[start_group].item())
                    if group_count <= 0:
                        continue
                    nodes = self.nodes_sorted_by_triplet[group_start:group_start + group_count]
                    route_dist = self.end_dist_tensor[:, nodes]
                    min_route_dist, min_node_pos = torch.min(route_dist, dim=1)
                    reachable = torch.isfinite(min_route_dist)
                    if not torch.any(reachable):
                        continue
                    desired_center = self.triplet_group_centers[effective_end_groups[row]]
                    snap_dist = torch.norm(self.triplet_group_centers - desired_center.unsqueeze(0), dim=1)
                    score = snap_dist + 0.01 * min_route_dist
                    score = score.masked_fill(~reachable, float('inf'))
                    chosen_group = torch.argmin(score)
                    chosen_node = nodes[min_node_pos[chosen_group]]
                    effective_end_groups[row] = chosen_group
                    start_node_idx[row] = chosen_node

            routed = start_node_idx >= 0
            if torch.any(routed):
                routed_rows = torch.nonzero(routed, as_tuple=False).squeeze(1)
                out_rows = valid_b[routed_rows]
                next_groups = self.end_next_tensor[effective_end_groups[routed_rows]]
                valid_count = routed_rows.numel()
                v = start_node_idx[routed_rows].view(valid_count, 1).expand(-1, L)
                cur = next_groups

                for bit in range(K):
                    use = use_masks[bit].view(1, L).expand(valid_count, -1)
                    if torch.any(use):
                        safe_v = torch.clamp(v, 0, N - 1)
                        jumped = cur.gather(1, safe_v)
                        jumped = torch.where(v >= 0, jumped, torch.full_like(v, -1))
                        v = torch.where(use, jumped, v)

                    last = cur
                    safe_idx = torch.clamp(last, 0, N - 1)
                    nxt = last.gather(1, safe_idx)
                    cur = torch.where(last >= 0, nxt, torch.full_like(last, -1))

                local_path_idx = v
                local_mask = local_path_idx >= 0

                if torch.any(local_mask):
                    col_idx = torch.arange(L, device=device).view(1, L).expand(valid_count, -1)
                    order_score = (~local_mask).to(torch.long) * L + col_idx
                    order = torch.argsort(order_score, dim=1, stable=True)
                    local_path_idx = local_path_idx.gather(1, order)
                    local_mask = local_mask.gather(1, order)

                tris = torch.full((valid_count, L, 3), pad_value, dtype=torch.long, device=device)
                if torch.any(local_mask):
                    flat_idx = local_path_idx[local_mask]
                    tris_flat = self.node_triplets.index_select(0, flat_idx)
                    tris[local_mask] = tris_flat
                triplets[out_rows] = tris
                mask[out_rows] = local_mask
                node_indices[out_rows] = local_path_idx

        return triplets, mask, node_indices

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化规划器
    path_planner = PathPlanner(map_path='maps/processed_map_Town01_stitched.json', device=torch.device('cuda'))

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
        
        print(f"\n调用plan_path:")
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
        
        ax.set_title(f'road graph and agent positions, path plans')
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

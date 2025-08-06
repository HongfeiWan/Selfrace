import numpy as np
import torch
import networkx as nx
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union
import torch.nn.functional as F

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据类型定义
Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

class PathPlanner:
    """
    GPU加速的路径规划器类
    输入cross信息和map信息，输出导航坐标序列
    支持批量处理和GPU张量计算
    """
    def __init__(self, cross_data: Dict = None, map_data: Dict = None, device: Optional[torch.device] = None, 
                 map_path: str = None):
        """
        初始化GPU加速的路径规划器
        
        Args:
            cross_data: 交叉路口数据（可选，如果提供map_path则自动加载）
            map_data: 地图数据（可选，如果提供map_path则自动加载）
            device: GPU设备，默认为全局device
            map_path: 地图文件路径，如果提供则自动加载cross_data和map_data
        """
        self.device = device if device is not None else globals()['device']
        
        # 如果提供了map_path，自动加载数据
        if map_path is not None:
            cross_data_path = map_path.replace('processed_map_', 'cross_data_processed_map_')
            print(f"自动加载cross数据: {cross_data_path}")
            # 加载cross数据
            if cross_data is None:
                cross_data = load_cross_data(cross_data_path)
                if cross_data is None:
                    print(f"警告: 无法加载cross数据文件: {cross_data_path}")
                    cross_data = {}
            # 加载地图数据
            if map_data is None:
                map_data = load_map_data(map_path)
                if map_data is None:
                    print(f"警告: 无法加载地图数据文件: {map_path}")
                    map_data = {}

        self.cross_data = cross_data or {}
        # 从map_data中提取必要信息
        self.quads_data = map_data.get('quads', []) if map_data else []
        self.global_w_lane_waypoints = map_data.get('global_w_lane_waypoints', []) if map_data else []
        self.quad_to_next_waypoint = {int(k): v for k, v in map_data.get('quad_to_next_waypoint', {}).items()} if map_data else {}
        self.quad_to_prev_waypoint = {int(k): v for k, v in map_data.get('quad_to_prev_waypoint', {}).items()} if map_data else {}
        # 构建索引映射
        self.quads_by_id = {q['polyId']: q for q in self.quads_data if 'polyId' in q}
        self.polyid_to_index = {q['polyId']: i for i, q in enumerate(self.quads_data) if 'polyId' in q}
        # GPU缓存
        self._lanes_cache = None
        self._lanes_cache_waypoints = None
        self._quad_centers_gpu = None
        self._quad_directions_gpu = None
        self._waypoints_gpu = None
        # CPU缓存（用于兼容性）
        self.quad_centers = {}
        self._quad_directions_cache = {}
        
        # 预计算GPU张量数据
        self._precompute_gpu_tensors()
        
        # 构建waypoint图
        self.G = None
        self.cross_waypoint_records = None
        if cross_data:
            self.G, self.cross_waypoint_records = self._build_waypoint_graph(cross_data)
    
    def _precompute_gpu_tensors(self):
        """预计算GPU张量数据，提高后续计算效率"""
        if not self.quads_data:
            return
        # 预计算所有quad的中心点和方向向量
        num_quads = len(self.quads_data)
        quad_centers = torch.zeros(num_quads, 2, dtype=torch.float32, device=self.device)
        quad_directions = torch.zeros(num_quads, 2, dtype=torch.float32, device=self.device)
        
        for i, quad in enumerate(self.quads_data):
            # 计算中心点
            vertices = torch.tensor([[point['x'], point['y']] for point in quad['vertices']], 
                                  dtype=torch.float32, device=self.device)
            center = torch.mean(vertices, dim=0)
            quad_centers[i] = center
            
            # 计算方向向量
            v0 = vertices[0]
            v2 = vertices[2]
            direction = v2 - v0
            norm = torch.norm(direction)
            if norm > 1e-8:  # 使用更小的阈值避免数值误差
                direction = direction / norm
            else:
                # 如果v0和v2相同，尝试使用其他顶点对
                if len(vertices) > 3:
                    v1 = vertices[1]
                    v3 = vertices[3]
                    direction = v3 - v1
                    norm = torch.norm(direction)
                    if norm > 1e-8:
                        direction = direction / norm
                    else:
                        # 如果还是零向量，使用默认方向
                        direction = torch.tensor([1.0, 0.0], device=self.device)
                else:
                    # 使用默认方向
                    direction = torch.tensor([1.0, 0.0], device=self.device)
            quad_directions[i] = direction
            
            # 同时更新CPU缓存
            self.quad_centers[i] = center.cpu().numpy()
            self._quad_directions_cache[i] = direction.cpu().numpy()
        
        self._quad_centers_gpu = quad_centers
        self._quad_directions_gpu = quad_directions
        
        # 预计算waypoints张量
        if self.global_w_lane_waypoints:
            num_waypoints = len(self.global_w_lane_waypoints)
            waypoints_tensor = torch.zeros(num_waypoints, 6, dtype=torch.float32, device=self.device)
            
            for i, wp in enumerate(self.global_w_lane_waypoints):
                waypoints_tensor[i, 0] = wp['x']
                waypoints_tensor[i, 1] = wp['y']  # 移除Y轴翻转
                waypoints_tensor[i, 2] = wp['carla_waypoint_info']['road_id']
                waypoints_tensor[i, 3] = wp['carla_waypoint_info']['lane_id']
                waypoints_tensor[i, 4] = wp['carla_waypoint_info']['s']
                waypoints_tensor[i, 5] = i  # 原始索引
            
            self._waypoints_gpu = waypoints_tensor
    
    def _build_waypoint_graph(self, cross_data: Dict) -> Tuple[nx.DiGraph, Dict]:
        """
        构建cross的waypoint级别的有向图
        1. 同road_id和lane_id的start_waypoints指向end_waypoints
        2. 同一个cross_id下，不同road_id或lane_id的连线仅根据paths内有记录的from_end_waypoint和to_start_waypoint建立
        3. 在每个cross_id内记录from_end_waypoint和to_start_waypoint
        """
        G = nx.DiGraph()
        cross_waypoint_records = defaultdict(lambda: {"from_end_waypoint": [], "to_start_waypoint": []})
        
        # 收集所有waypoint
        all_start = []
        all_end = []
        cross_start = defaultdict(list)
        cross_end = defaultdict(list)
        
        for key, value in cross_data.items():
            if key.startswith('cross_'):
                cross_id = value['cross_id']
                for wp in value.get('start_waypoints', []):
                    wp_info = dict(wp)
                    wp_info['cross_id'] = cross_id
                    wp_info['type'] = 'start'
                    all_start.append(wp_info)
                    cross_start[cross_id].append(wp_info)
                for wp in value.get('end_waypoints', []):
                    wp_info = dict(wp)
                    wp_info['cross_id'] = cross_id
                    wp_info['type'] = 'end'
                    all_end.append(wp_info)
                    cross_end[cross_id].append(wp_info)
        
        # 1. 同road_id和lane_id的start_waypoints指向end_waypoints
        for s_wp in all_start:
            for e_wp in all_end:
                if (s_wp['road_id'] == e_wp['road_id'] and s_wp['lane_id'] == e_wp['lane_id']):
                    s_id = (s_wp['cross_id'], s_wp['road_id'], s_wp['lane_id'], s_wp['x'], s_wp['y'], s_wp['s'], 'start')
                    e_id = (e_wp['cross_id'], e_wp['road_id'], e_wp['lane_id'], e_wp['x'], e_wp['y'], e_wp['s'], 'end')
                    distance = abs(e_wp['s'] - s_wp['s'])
                    G.add_edge(s_id, e_id, distance=distance)
        
        # 2. 只根据paths建立cross内部连线
        for key, value in cross_data.items():
            if key.startswith('cross_') and 'paths' in value:
                cross_id = value['cross_id']
                for path in value['paths']:
                    s_wp = path['to_start_waypoint']
                    e_wp = path['from_end_waypoint']
                    s_id = (cross_id, s_wp['road_id'], s_wp['lane_id'], s_wp['x'], s_wp['y'], s_wp['s'], 'start')
                    e_id = (cross_id, e_wp['road_id'], e_wp['lane_id'], e_wp['x'], e_wp['y'], e_wp['s'], 'end')
                    distance = path['distance']
                    G.add_edge(e_id, s_id, distance=distance)
                    cross_waypoint_records[cross_id]['from_end_waypoint'].append(e_id)
                    cross_waypoint_records[cross_id]['to_start_waypoint'].append(s_id)
        
        return G, cross_waypoint_records
    
    def _group_waypoints_by_lane(self, waypoints: List[Dict]) -> Dict:
        """按车道分组并排序航点，使用缓存避免重复计算"""
        # 检查缓存是否有效
        if (self._lanes_cache is not None and 
            self._lanes_cache_waypoints is not None and 
            self._lanes_cache_waypoints is waypoints):
            return self._lanes_cache
        
        # 计算并缓存结果
        lanes = defaultdict(list)
        for wp in waypoints:
            lanes[(wp['carla_waypoint_info']['road_id'], wp['carla_waypoint_info']['lane_id'])].append(wp)
        
        # 对每个车道内的航点进行排序
        for lane_id_tuple, wps_in_lane in lanes.items():
            if wps_in_lane:
                is_reverse_lane = wps_in_lane[0]['carla_waypoint_info']['lane_id'] < 0
                wps_in_lane.sort(key=lambda w: w['carla_waypoint_info']['s'], reverse=is_reverse_lane)
        
        # 过滤掉首尾waypoint距离太短的车道
        min_lane_length = 10  # 最小车道长度阈值（米）
        lanes_to_remove = []
        for lane_id, wps_in_lane in lanes.items():
            if len(wps_in_lane) >= 2:
                # 计算首尾waypoint之间的距离
                start_wp = wps_in_lane[0]
                end_wp = wps_in_lane[-1]
                distance = np.sqrt((end_wp['x'] - start_wp['x'])**2 + (end_wp['y'] - start_wp['y'])**2)
                
                if distance < min_lane_length:
                    lanes_to_remove.append(lane_id)
        
        # 从lanes字典中删除太短的车道
        for lane_id in lanes_to_remove:
            del lanes[lane_id]
        
        # 更新缓存
        self._lanes_cache = lanes
        self._lanes_cache_waypoints = waypoints
        
        return lanes
    
    def _group_waypoints_by_lane_gpu(self) -> Dict:
        """使用GPU加速的waypoint分组方法"""
        if self._waypoints_gpu is None:
            return self._group_waypoints_by_lane(self.global_w_lane_waypoints)
        
        # 使用GPU张量进行分组
        waypoints_tensor = self._waypoints_gpu  # [num_waypoints, 6]
        
        # 提取road_id和lane_id
        road_ids = waypoints_tensor[:, 2].long()
        lane_ids = waypoints_tensor[:, 3].long()
        
        # 创建唯一的车道标识
        lane_keys = road_ids * 10000 + lane_ids  # 假设road_id < 10000
        
        # 找到唯一车道
        unique_lanes = torch.unique(lane_keys)
        
        lanes = {}
        for lane_key in unique_lanes:
            # 找到属于该车道的所有waypoint索引
            mask = lane_keys == lane_key
            indices = torch.where(mask)[0]
            
            # 按s坐标排序
            s_coords = waypoints_tensor[indices, 4]
            sorted_indices = torch.argsort(s_coords, descending=(lane_ids[indices[0]] < 0))
            sorted_indices = indices[sorted_indices]
            
            # 转换为字典格式
            wps_in_lane = []
            for idx in sorted_indices:
                wp_data = waypoints_tensor[idx]
                wp_dict = {
                    'x': wp_data[0].item(),
                    'y': -wp_data[1].item(),  # 翻转回原始坐标系
                    'carla_waypoint_info': {
                        'road_id': wp_data[2].item(),
                        'lane_id': wp_data[3].item(),
                        's': wp_data[4].item()
                    }
                }
                wps_in_lane.append(wp_dict)
            
            # 过滤短车道
            if len(wps_in_lane) >= 2:
                start_wp = wps_in_lane[0]
                end_wp = wps_in_lane[-1]
                distance = np.sqrt((end_wp['x'] - start_wp['x'])**2 + (end_wp['y'] - start_wp['y'])**2)
                
                if distance >= 10:  # 最小车道长度阈值
                    road_id = wps_in_lane[0]['carla_waypoint_info']['road_id']
                    lane_id = wps_in_lane[0]['carla_waypoint_info']['lane_id']
                    lanes[(road_id, lane_id)] = wps_in_lane
        
        return lanes
    
    def _find_shortest_path(self, start_id: List, end_id: List) -> List[List]:
        """
        在waypoint图中找到最短路径
        
        Args:
            start_id: [cross_id, road_id, lane_id]
            end_id: [cross_id, road_id, lane_id]
            
        Returns:
            路径上所有[cross_id, road_id, lane_id]的列表（不含重复）
        """
        if not self.G:
            return []
        
        # 找到所有起点和终点的节点（type可以是start或end）
        start_nodes = [n for n in self.G.nodes if list(n[:3]) == start_id]
        end_nodes = [n for n in self.G.nodes if list(n[:3]) == end_id]
        
        if not start_nodes or not end_nodes:
            return []
        
        # 搜索所有组合，找最短路径
        min_path = None
        min_length = float('inf')
        
        for s in start_nodes:
            for e in end_nodes:
                try:
                    path = nx.shortest_path(self.G, source=s, target=e, weight='distance')
                    length = nx.shortest_path_length(self.G, source=s, target=e, weight='distance')
                    if length < min_length:
                        min_length = length
                        min_path = path
                except nx.NetworkXNoPath:
                    continue
        
        if min_path is None:
            return []
        
        # 提取[cross_id, road_id, lane_id]，去重
        result = []
        seen = set()
        for n in min_path:
            key = tuple(n[:3])
            if key not in seen:
                result.append(list(key))
                seen.add(key)
        
        return result
    
    def _get_quad_center(self, quad_id: int) -> Union[np.ndarray, Tensor]:
        """获取quad中心点，使用GPU张量"""
        # 如果quad_id是polyId，需要转换为数组索引
        if quad_id in self.polyid_to_index:
            array_index = self.polyid_to_index[quad_id]
        else:
            # 如果不在polyid_to_index中，直接使用quad_id作为索引
            array_index = quad_id
        
        if array_index >= len(self.quads_data) or self._quad_centers_gpu is None:
            return np.array([0, 0]) if isinstance(quad_id, int) else torch.zeros(2, device=self.device)
        
        if isinstance(quad_id, int):
            result = self._quad_centers_gpu[array_index].cpu().numpy()
            return result
        else:
            result = self._quad_centers_gpu[array_index]
            return result
    
    def get_quad_center(self, quad_id: int) -> np.ndarray:
        """获取quad中心点，兼容CPU版本接口"""
        if quad_id in self.quad_centers:
            return self.quad_centers[quad_id]
        # 如果缓存中没有，则计算并缓存
        quad = self.quads_data[quad_id]
        vertices = np.array([[point['x'], point['y']] for point in quad['vertices']])
        center = np.mean(vertices, axis=0)
        self.quad_centers[quad_id] = center
        return center
    
    def _get_quad_direction(self, quad_id: int) -> Union[np.ndarray, Tensor]:
        """获取quad方向，使用GPU张量"""
        # 如果quad_id是polyId，需要转换为数组索引
        if quad_id in self.polyid_to_index:
            array_index = self.polyid_to_index[quad_id]
        else:
            # 如果不在polyid_to_index中，直接使用quad_id作为索引
            array_index = quad_id
        
        if array_index >= len(self.quads_data) or self._quad_directions_gpu is None:
            return np.array([0, 0]) if isinstance(quad_id, int) else torch.zeros(2, device=self.device)
        
        if isinstance(quad_id, int):
            result = self._quad_directions_gpu[array_index].cpu().numpy()
            return result
        else:
            result = self._quad_directions_gpu[array_index]
            return result
    
    def get_quad_direction(self, quad_id: int) -> np.ndarray:
        """获取quad方向，兼容CPU版本接口"""
        if quad_id in self._quad_directions_cache:
            return self._quad_directions_cache[quad_id]
        # 如果缓存中没有，则计算并缓存
        quad = self.quads_data[quad_id]
        vertices = np.array([[point['x'], point['y']] for point in quad['vertices']])
        v0 = vertices[0]
        v2 = vertices[2]
        direction = v2 - v0
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        else:
            # 如果v0和v2相同，尝试使用其他顶点对
            if len(vertices) > 3:
                v1 = vertices[1]
                v3 = vertices[3]
                direction = v3 - v1
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                else:
                    # 如果还是零向量，使用默认方向
                    direction = np.array([1.0, 0.0])
            else:
                # 使用默认方向
                direction = np.array([1.0, 0.0])
        self._quad_directions_cache[quad_id] = direction
        return direction
    
    def _get_quad_centers_batch(self, quad_ids: Tensor) -> Tensor:
        """批量获取quad中心点"""
        if self._quad_centers_gpu is None:
            return torch.zeros(len(quad_ids), 2, device=self.device)
        
        # 将polyId转换为数组索引
        array_indices = torch.zeros_like(quad_ids, dtype=torch.long)
        for i, quad_id in enumerate(quad_ids):
            if quad_id.item() in self.polyid_to_index:
                array_indices[i] = self.polyid_to_index[quad_id.item()]
            else:
                array_indices[i] = quad_id.item()
        
        valid_mask = (array_indices >= 0) & (array_indices < len(self.quads_data))
        centers = torch.zeros(len(quad_ids), 2, device=self.device)
        centers[valid_mask] = self._quad_centers_gpu[array_indices[valid_mask]]
        return centers
    
    def _get_quad_directions_batch(self, quad_ids: Tensor) -> Tensor:
        """批量获取quad方向"""
        if self._quad_directions_gpu is None:
            return torch.zeros(len(quad_ids), 2, device=self.device)
        
        # 将polyId转换为数组索引
        array_indices = torch.zeros_like(quad_ids, dtype=torch.long)
        for i, quad_id in enumerate(quad_ids):
            if quad_id.item() in self.polyid_to_index:
                array_indices[i] = self.polyid_to_index[quad_id.item()]
            else:
                array_indices[i] = quad_id.item()
        
        valid_mask = (array_indices >= 0) & (array_indices < len(self.quads_data))
        directions = torch.zeros(len(quad_ids), 2, device=self.device)
        directions[valid_mask] = self._quad_directions_gpu[array_indices[valid_mask]]
        return directions
    
    def _is_direction_similar(self, dir1: Union[np.ndarray, Tensor], dir2: Union[np.ndarray, Tensor], 
                             angle_threshold_deg: float = 90) -> Union[bool, Tensor]:
        """判断两个方向向量是否相似，支持GPU张量"""
        if isinstance(dir1, np.ndarray) and isinstance(dir2, np.ndarray):
            # CPU版本 - 修复除零错误
            import math
            norm1 = np.linalg.norm(dir1)
            norm2 = np.linalg.norm(dir2)
            
            # 检查是否为零向量
            if norm1 < 1e-8 or norm2 < 1e-8:
                return False  # 零向量不相似
            
            cos_angle = np.dot(dir1, dir2) / (norm1 * norm2)
            angle = math.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            return angle < angle_threshold_deg
        else:
            # GPU张量版本
            dir1_tensor = dir1 if isinstance(dir1, Tensor) else torch.tensor(dir1, device=self.device)
            dir2_tensor = dir2 if isinstance(dir2, Tensor) else torch.tensor(dir2, device=self.device)
            
            # 计算余弦相似度，避免除零
            norm1 = torch.norm(dir1_tensor, dim=-1)
            norm2 = torch.norm(dir2_tensor, dim=-1)
            
            # 检查是否为零向量
            zero_mask = (norm1 < 1e-8) | (norm2 < 1e-8)
            
            cos_angle = torch.sum(dir1_tensor * dir2_tensor, dim=-1) / (norm1 * norm2 + 1e-8)
            
            # 计算角度
            angle = torch.acos(torch.clamp(cos_angle, -1, 1)) * 180 / torch.pi
            
            # 零向量返回False
            result = angle < angle_threshold_deg
            if isinstance(result, Tensor):
                result = torch.where(zero_mask, torch.tensor(False, device=self.device), result)
            return result
    
    def _is_direction_similar_batch(self, dir1_batch: Tensor, dir2_batch: Tensor, 
                                   angle_threshold_deg: float = 90) -> Tensor:
        """批量判断方向向量相似性"""
        cos_angle = torch.sum(dir1_batch * dir2_batch, dim=-1) / (
            torch.norm(dir1_batch, dim=-1) * torch.norm(dir2_batch, dim=-1) + 1e-8)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1)) * 180 / torch.pi
        return angle < angle_threshold_deg
    
    def _find_cross_id_by_waypoint(self, waypoint: Dict) -> Optional[int]:
        """根据 waypoint 坐标查找所属的 cross_id"""
        if not self.cross_data:
            return None
        
        wp_x, wp_y = waypoint['x'], waypoint['y']
        
        # 遍历所有 cross，查找包含该 waypoint 的 cross
        for cross_key, cross_info in self.cross_data.items():
            if not cross_key.startswith('cross_'):
                continue
            
            cross_id = cross_info['cross_id']
            
            # 检查 start_waypoints
            for wp in cross_info.get('start_waypoints', []):
                if (abs(wp['x'] - wp_x) < 0.1 and abs(wp['y'] - wp_y) < 0.1 and
                    wp['road_id'] == waypoint['carla_waypoint_info']['road_id'] and
                    wp['lane_id'] == waypoint['carla_waypoint_info']['lane_id']):
                    return cross_id
            
            # 检查 end_waypoints
            for wp in cross_info.get('end_waypoints', []):
                if (abs(wp['x'] - wp_x) < 0.1 and abs(wp['y'] - wp_y) < 0.1 and
                    wp['road_id'] == waypoint['carla_waypoint_info']['road_id'] and
                    wp['lane_id'] == waypoint['carla_waypoint_info']['lane_id']):
                    return cross_id
        
        return None
    
    def _insert_path_quads_between_nodes(self, cross_id: int, current_nodes: List, next_nodes: List) -> List[Dict]:
        """在同一cross内的两个节点之间插入路径quads"""
        if not self.cross_data:
            return []
        
        # 找到对应的cross信息
        cross_info = None
        for cross_key, cross_data in self.cross_data.items():
            if cross_key.startswith('cross_') and cross_data.get('cross_id') == cross_id:
                cross_info = cross_data
                break
        
        if not cross_info:
            return []
        
        # 找到from_end_waypoint和to_start_waypoint
        from_end_waypoint = None
        to_start_waypoint = None
        
        # 从当前节点中找到end类型的waypoint
        for node in current_nodes:
            if node[6] == 'end':
                from_end_waypoint = {
                    'road_id': node[1],
                    'lane_id': node[2],
                    'x': node[3],
                    'y': node[4],
                    's': node[5]
                }
                break
        
        # 从下一个节点中找到start类型的waypoint
        for node in next_nodes:
            if node[6] == 'start':
                to_start_waypoint = {
                    'road_id': node[1],
                    'lane_id': node[2],
                    'x': node[3],
                    'y': node[4],
                    's': node[5]
                }
                break
        
        if not from_end_waypoint or not to_start_waypoint:
            return []
        
        # 在cross的paths中查找匹配的路径
        paths = cross_info.get('paths', [])
        for path in paths:
            path_from_end = path.get('from_end_waypoint', {})
            path_to_start = path.get('to_start_waypoint', {})
            
            # 检查是否匹配
            if (path_from_end.get('road_id') == from_end_waypoint['road_id'] and
                path_from_end.get('lane_id') == from_end_waypoint['lane_id'] and
                path_to_start.get('road_id') == to_start_waypoint['road_id'] and
                path_to_start.get('lane_id') == to_start_waypoint['lane_id']):
                
                # 找到匹配的路径，返回path_quad_ids
                path_quad_ids = path.get('path_quad_ids', [])
                result = []
                for poly_id in path_quad_ids:
                    quad_idx = self.polyid_to_index.get(poly_id)
                    if quad_idx is not None:
                        quad_center = self._get_quad_center(quad_idx)
                        result.append({
                            'type': 'path_quad',
                            'quad_id': quad_idx,
                            'poly_id': poly_id,
                            'coords': {'x': quad_center[0], 'y': quad_center[1]},
                            'cross_id': cross_id
                        })
                return result
        
        return []
    
    def _insert_lane_waypoints_between_nodes(self, current_nodes: List, next_nodes: List) -> List[Dict]:
        """在不同cross的两个节点之间插入车道waypoints"""
        # 找到当前节点的start waypoint和下一个节点的end waypoint
        current_start_waypoint = None
        next_end_waypoint = None
        
        # 从当前节点中找到start类型的waypoint
        for node in current_nodes:
            if node[6] == 'start':
                current_start_waypoint = {
                    'road_id': node[1],
                    'lane_id': node[2],
                    'x': node[3],
                    'y': node[4],
                    's': node[5]
                }
                break
        
        # 从下一个节点中找到end类型的waypoint
        for node in next_nodes:
            if node[6] == 'end':
                next_end_waypoint = {
                    'road_id': node[1],
                    'lane_id': node[2],
                    'x': node[3],
                    'y': node[4],
                    's': node[5]
                }
                break
        
        if not current_start_waypoint or not next_end_waypoint:
            return []
        
        # 检查两个waypoint是否在同一车道内
        if (current_start_waypoint['road_id'] == next_end_waypoint['road_id'] and 
            current_start_waypoint['lane_id'] == next_end_waypoint['lane_id']):
            
            # 使用_group_waypoints_by_lane方法按车道分组并排序航点
            lanes = self._group_waypoints_by_lane(self.global_w_lane_waypoints)
            lane_key = (current_start_waypoint['road_id'], current_start_waypoint['lane_id'])
            
            if lane_key in lanes:
                wps_in_lane = lanes[lane_key]
                
                # 找到当前start waypoint和下一个end waypoint在车道中的位置
                current_start_idx = None
                next_end_idx = None
                
                for i, wp in enumerate(wps_in_lane):
                    wp_info = wp['carla_waypoint_info']
                    
                    # 检查是否匹配当前start waypoint
                    x_diff = abs(wp['x'] - current_start_waypoint['x'])
                    y_diff = abs(wp['y'] - current_start_waypoint['y'])
                    s_diff = abs(wp_info['s'] - current_start_waypoint['s'])
                    if (x_diff < 0.1 and y_diff < 0.1 and s_diff < 0.1):
                        current_start_idx = i
                    
                    # 检查是否匹配下一个end waypoint
                    x_diff = abs(wp['x'] - next_end_waypoint['x'])
                    y_diff = abs(wp['y'] - next_end_waypoint['y'])
                    s_diff = abs(wp_info['s'] - next_end_waypoint['s'])
                    if (x_diff < 0.1 and y_diff < 0.1 and s_diff < 0.1):
                        next_end_idx = i
                
                # 如果找到了两个waypoint的位置，插入它们之间的waypoints
                if current_start_idx is not None and next_end_idx is not None:
                    
                    # 确定起始和结束索引（确保start_idx < end_idx）
                    if current_start_idx < next_end_idx:
                        start_idx = current_start_idx
                        end_idx = next_end_idx
                        # 从start_idx+1到end_idx-1，正序
                        waypoint_indices = list(range(start_idx + 1, end_idx))
                    else:
                        start_idx = next_end_idx
                        end_idx = current_start_idx
                        # 从start_idx+1到end_idx-1，正序
                        waypoint_indices = list(range(start_idx + 1, end_idx))
                    
                    # 插入两个节点之间的waypoints（不含这两个节点本身）
                    result = []
                    for i in waypoint_indices:
                        wp = wps_in_lane[i]
                        wp_info = wp['carla_waypoint_info']
                        
                        result.append({
                            'type': 'lane_waypoint',
                            'coords': {'x': wp['x'], 'y': wp['y']},
                            'road_id': wp_info['road_id'],
                            'lane_id': wp_info['lane_id'],
                            's': wp_info['s']
                        })
                    return result
        
        return []
    
    def plan_path(self, start_quad_id: int, goal_quad_id: int) -> List[Dict]:
        """
        规划从起始quad到目标quad的路径
        Args:
            start_quad_id: 起始quad的ID
            goal_quad_id: 目标quad的ID
            
        Returns:
            导航坐标序列，包含路径上的所有关键点
        """
        if not self.G or not self.cross_data:
            return []
        
        # 获取起始点和目标点的信息
        start_quad = self.quads_data[start_quad_id] if start_quad_id < len(self.quads_data) else None
        goal_quad = self.quads_data[goal_quad_id] if goal_quad_id < len(self.quads_data) else None
        
        if not start_quad or not goal_quad:
            return []
        
        # 确定起始点和目标点的类型（是否在filtered_quad_indices内）
        filtered_quad_indices = set(self.cross_data.get('filtered_quad_indices', []))
        start_in_filtered = start_quad_id in filtered_quad_indices
        goal_in_filtered = goal_quad_id in filtered_quad_indices
        
        # 构建起始点和目标点的waypoint ID
        start_id = None
        end_id = None
        
        # 处理起始点
        if start_in_filtered:
            # 在filtered_quad_indices内：使用车道终点作为start_id
            target_wp_idx = self.quad_to_next_waypoint.get(start_quad.get('polyId'))
            if target_wp_idx is not None and target_wp_idx < len(self.global_w_lane_waypoints):
                target_wp = self.global_w_lane_waypoints[target_wp_idx]
                lanes = self._group_waypoints_by_lane(self.global_w_lane_waypoints)
                lane_key = (target_wp['carla_waypoint_info']['road_id'], target_wp['carla_waypoint_info']['lane_id'])
                if lane_key in lanes:
                    wps_in_lane = lanes[lane_key]
                    if wps_in_lane:
                        end_wp = wps_in_lane[-1]  # 车道终点
                        cross_id = self._find_cross_id_by_waypoint(end_wp)
                        if cross_id is not None:
                            start_id = [cross_id, end_wp['carla_waypoint_info']['road_id'], end_wp['carla_waypoint_info']['lane_id']]
        else:
            # 在filtered_quad_indices外：使用cross路径的to_start_waypoint作为start_id
            start_polyid = start_quad.get('polyId')
            start_center = self._get_quad_center(start_quad_id)
            start_dir = self._get_quad_direction(start_quad_id)
            
            # 找到最近的cross和路径
            min_dist = float('inf')
            best_cross = None
            best_path = None
            best_quad_idx = None
            
            for cross_key, cross_info in self.cross_data.items():
                if not cross_key.startswith('cross_'):
                    continue
                paths = cross_info.get('paths', [])
                for path in paths:
                    path_quad_ids = path.get('path_quad_ids', [])
                    for idx, poly_id in enumerate(path_quad_ids):
                        quad_idx = self.polyid_to_index.get(poly_id)
                        if quad_idx is None:
                            continue
                        center = self._get_quad_center(quad_idx)
                        dist = np.linalg.norm(center - start_center)
                        if dist < 3.0:
                            dir2 = self._get_quad_direction(quad_idx)
                            if self._is_direction_similar(start_dir, dir2):
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cross = cross_info
                                    best_path = path
                                    best_quad_idx = idx
            
            if best_path is not None:
                to_start_wp = best_path.get('to_start_waypoint', None)
                if to_start_wp and isinstance(to_start_wp, dict):
                    start_id = [best_cross['cross_id'], to_start_wp['road_id'], to_start_wp['lane_id']]
        
        # 处理目标点
        if goal_in_filtered:
            # 在filtered_quad_indices内：使用车道起点作为end_id
            target_wp_idx = self.quad_to_prev_waypoint.get(goal_quad.get('polyId'))
            if target_wp_idx is not None and target_wp_idx < len(self.global_w_lane_waypoints):
                target_wp = self.global_w_lane_waypoints[target_wp_idx]
                lanes = self._group_waypoints_by_lane(self.global_w_lane_waypoints)
                lane_key = (target_wp['carla_waypoint_info']['road_id'], target_wp['carla_waypoint_info']['lane_id'])
                if lane_key in lanes:
                    wps_in_lane = lanes[lane_key]
                    if wps_in_lane:
                        start_wp = wps_in_lane[0]  # 车道起点
                        cross_id = self._find_cross_id_by_waypoint(start_wp)
                        if cross_id is not None:
                            end_id = [cross_id, start_wp['carla_waypoint_info']['road_id'], start_wp['carla_waypoint_info']['lane_id']]
        else:
            # 在filtered_quad_indices外：使用cross路径的from_end_waypoint作为end_id
            goal_polyid = goal_quad.get('polyId')
            goal_center = self._get_quad_center(goal_quad_id)
            goal_dir = self._get_quad_direction(goal_quad_id)
            
            # 找到最近的cross和路径
            min_dist = float('inf')
            best_cross = None
            best_path = None
            best_quad_idx = None
            
            for cross_key, cross_info in self.cross_data.items():
                if not cross_key.startswith('cross_'):
                    continue
                paths = cross_info.get('paths', [])
                for path in paths:
                    path_quad_ids = path.get('path_quad_ids', [])
                    for idx, poly_id in enumerate(path_quad_ids):
                        quad_idx = self.polyid_to_index.get(poly_id)
                        if quad_idx is None:
                            continue
                        center = self._get_quad_center(quad_idx)
                        dist = np.linalg.norm(center - goal_center)
                        if dist < 3.0:
                            dir2 = self._get_quad_direction(quad_idx)
                            if self._is_direction_similar(goal_dir, dir2):
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cross = cross_info
                                    best_path = path
                                    best_quad_idx = idx
            
            if best_path is not None:
                from_end_wp = best_path.get('from_end_waypoint', None)
                if from_end_wp and isinstance(from_end_wp, dict):
                    end_id = [best_cross['cross_id'], from_end_wp['road_id'], from_end_wp['lane_id']]
        
        # 如果无法确定起始点或目标点，返回空路径
        if not start_id or not end_id:
            return []
        
        # 计算最短路径
        path_result = self._find_shortest_path(start_id, end_id)
        if not path_result:
            return []
        
        # 构建完整的导航坐标序列
        navigation_sequence = []
        
        # 添加起始点信息
        start_center = self._get_quad_center(start_quad_id)
        navigation_sequence.append({
            'type': 'start',
            'quad_id': start_quad_id,
            'coords': {'x': start_center[0], 'y': start_center[1]}
        })
        
        # ========== 新增：补全起始点到车道终点/起点的waypoint或quad序列 ==========
        # 如果起始点在filtered_quad_indices内，补全起始点到车道终点的waypoint
        if start_in_filtered:
            target_wp_idx = self.quad_to_next_waypoint.get(start_quad.get('polyId'))
            if target_wp_idx is not None and target_wp_idx < len(self.global_w_lane_waypoints):
                target_wp = self.global_w_lane_waypoints[target_wp_idx]
                lanes = self._group_waypoints_by_lane(self.global_w_lane_waypoints)
                lane_key = (target_wp['carla_waypoint_info']['road_id'], target_wp['carla_waypoint_info']['lane_id'])
                if lane_key in lanes:
                    wps_in_lane = lanes[lane_key]
                    # 找到起始点在车道中的索引
                    idx = None
                    for i, wp in enumerate(wps_in_lane):
                        if (wp['x'] == target_wp['x'] and wp['y'] == target_wp['y'] and
                            wp['carla_waypoint_info']['s'] == target_wp['carla_waypoint_info']['s']):
                            idx = i
                            break
                    if idx is not None:
                        # 补全从起始点到车道终点的所有waypoint（包含target_wp本身）
                        for i in range(idx, len(wps_in_lane)):
                            wp = wps_in_lane[i]
                            navigation_sequence.append({
                                'type': 'lane_waypoint',
                                'coords': {'x': wp['x'], 'y': wp['y']},
                                'road_id': wp['carla_waypoint_info']['road_id'],
                                'lane_id': wp['carla_waypoint_info']['lane_id'],
                                's': wp['carla_waypoint_info']['s']
                            })
        # 如果起始点在filtered_quad_indices外，补全起始点到车道起点的quad序列
        if not start_in_filtered:
            start_center = self._get_quad_center(start_quad_id)
            start_dir = self._get_quad_direction(start_quad_id)
            min_dist = float('inf')
            best_cross = None
            best_path = None
            best_quad_idx = None
            for cross_key, cross_info in self.cross_data.items():
                if not cross_key.startswith('cross_'):
                    continue
                paths = cross_info.get('paths', [])
                for path in paths:
                    path_quad_ids = path.get('path_quad_ids', [])
                    for idx, poly_id in enumerate(path_quad_ids):
                        quad_idx = self.polyid_to_index.get(poly_id)
                        if quad_idx is None:
                            continue
                        center = self._get_quad_center(quad_idx)
                        dist = np.linalg.norm(center - start_center)
                        if dist < 3.0:
                            dir2 = self._get_quad_direction(quad_idx)
                            if self._is_direction_similar(start_dir, dir2):
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cross = cross_info
                                    best_path = path
                                    best_quad_idx = idx
            if best_path is not None:
                # 正确方向：补全从当前quad到cross path终点（to_start_waypoint）
                path_quad_ids = best_path.get('path_quad_ids', [])
                for i in range(best_quad_idx + 1, len(path_quad_ids)):
                    poly_id = path_quad_ids[i]
                    quad_idx = self.polyid_to_index.get(poly_id)
                    if quad_idx is not None:
                        quad_center = self._get_quad_center(quad_idx)
                        navigation_sequence.append({
                            'type': 'path_quad',
                            'quad_id': quad_idx,
                            'poly_id': poly_id,
                            'coords': {'x': quad_center[0], 'y': quad_center[1]},
                            'cross_id': best_cross['cross_id']
                        })
        
        
        # 添加路径节点和中间路径
        for i, ids in enumerate(path_result):
            # 找到所有对应的节点（可能有多个type）
            nodes = [n for n in self.G.nodes if list(n[:3]) == ids]
            for node in nodes:
                cross_id, road_id, lane_id, x, y, s, typ = node
                navigation_sequence.append({
                    'type': 'shortest_path_node',
                    'cross_id': cross_id,
                    'road_id': road_id,
                    'lane_id': lane_id,
                    'coords': {'x': x, 'y': y},
                    's': s,
                    'node_type': typ
                })
            # 如果不是最后一个节点，检查下一个节点是否在同一cross内
            if i < len(path_result) - 1:
                next_ids = path_result[i + 1]
                next_nodes = [n for n in self.G.nodes if list(n[:3]) == next_ids]
                
                # 检查当前节点和下一个节点是否在同一cross内
                current_cross_id = None
                next_cross_id = None
                
                if nodes:
                    current_cross_id = nodes[0][0]  # 第一个节点的cross_id
                
                if next_nodes:
                    next_cross_id = next_nodes[0][0]  # 第一个节点的cross_id
                
                # 如果两个节点在同一cross内，插入路径quads
                if current_cross_id is not None and next_cross_id is not None and current_cross_id == next_cross_id:
                    path_quads = self._insert_path_quads_between_nodes(current_cross_id, nodes, next_nodes)
                    navigation_sequence.extend(path_quads)
                # 如果两个节点在不同cross内，插入车道waypoints
                elif current_cross_id != next_cross_id:
                    lane_waypoints = self._insert_lane_waypoints_between_nodes(nodes, next_nodes)
                    navigation_sequence.extend(lane_waypoints)
        
        # 如果目标点在filtered_quad_indices内，补全目标点到车道起点的waypoint
        if goal_in_filtered:
            target_wp_idx = self.quad_to_prev_waypoint.get(goal_quad.get('polyId'))
            if target_wp_idx is not None and target_wp_idx < len(self.global_w_lane_waypoints):
                target_wp = self.global_w_lane_waypoints[target_wp_idx]
                lanes = self._group_waypoints_by_lane(self.global_w_lane_waypoints)
                lane_key = (target_wp['carla_waypoint_info']['road_id'], target_wp['carla_waypoint_info']['lane_id'])
                if lane_key in lanes:
                    wps_in_lane = lanes[lane_key]
                    # 找到目标点在车道中的索引
                    idx = None
                    for i, wp in enumerate(wps_in_lane):
                        if (wp['x'] == target_wp['x'] and wp['y'] == target_wp['y'] and
                            wp['carla_waypoint_info']['s'] == target_wp['carla_waypoint_info']['s']):
                            idx = i
                            break
                    if idx is not None:
                        # 补全从车道起点到目标点的所有waypoint（包含target_wp本身，正序）
                        for i in range(0, idx+1):
                            wp = wps_in_lane[i]
                            navigation_sequence.append({
                                'type': 'lane_waypoint',
                                'coords': {'x': wp['x'], 'y': wp['y']},
                                'road_id': wp['carla_waypoint_info']['road_id'],
                                'lane_id': wp['carla_waypoint_info']['lane_id'],
                                's': wp['carla_waypoint_info']['s']
                            })

        # 如果目标点在filtered_quad_indices外，补全车道终点到目标点的quad序列
        if not goal_in_filtered:
            # 找到目标点所在的cross和路径
            goal_center = self._get_quad_center(goal_quad_id)
            goal_dir = self._get_quad_direction(goal_quad_id)
            
            # 找到最近的cross和路径
            min_dist = float('inf')
            best_cross = None
            best_path = None
            best_quad_idx = None
            
            for cross_key, cross_info in self.cross_data.items():
                if not cross_key.startswith('cross_'):
                    continue
                paths = cross_info.get('paths', [])
                for path in paths:
                    path_quad_ids = path.get('path_quad_ids', [])
                    for idx, poly_id in enumerate(path_quad_ids):
                        quad_idx = self.polyid_to_index.get(poly_id)
                        if quad_idx is None:
                            continue
                        center = self._get_quad_center(quad_idx)
                        dist = np.linalg.norm(center - goal_center)
                        if dist < 3.0:
                            dir2 = self._get_quad_direction(quad_idx)
                            if self._is_direction_similar(goal_dir, dir2):
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cross = cross_info
                                    best_path = path
                                    best_quad_idx = idx
            
            if best_path is not None:
                # 简化逻辑：直接使用path_quad_ids从0到best_quad_idx的所有quads
                path_quad_ids = best_path.get('path_quad_ids', [])
                
                # 添加从路径起点到目标quad的所有quads
                for i in range(best_quad_idx + 1):
                    poly_id = path_quad_ids[i]
                    quad_idx = self.polyid_to_index.get(poly_id)
                    if quad_idx is not None:
                        quad_center = self._get_quad_center(quad_idx)
                        navigation_sequence.append({
                            'type': 'path_quad',
                            'quad_id': quad_idx,
                            'poly_id': poly_id,
                            'coords': {'x': quad_center[0], 'y': quad_center[1]},
                            'cross_id': best_cross['cross_id']
                        })
        
        # 添加目标点信息
        goal_center = self._get_quad_center(goal_quad_id)
        navigation_sequence.append({
            'type': 'goal',
            'quad_id': goal_quad_id,
            'coords': {'x': goal_center[0], 'y': goal_center[1]}
        })
        return navigation_sequence
    
    def get_path_distance(self, path_result: List[List]) -> float:
        """计算路径的总距离"""
        if not path_result or not self.G:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path_result) - 1):
            src_ids = path_result[i]
            dst_ids = path_result[i+1]
            # 找到对应的节点对
            src_nodes = [n for n in self.G.nodes if list(n[:3]) == src_ids]
            dst_nodes = [n for n in self.G.nodes if list(n[:3]) == dst_ids]
            for s in src_nodes:
                for d in dst_nodes:
                    if self.G.has_edge(s, d):
                        total_distance += self.G[s][d]['distance']
                        break
                else:
                    continue
                break
        
        return total_distance
    
    def get_path_distance_from_navigation_sequence(self, navigation_sequence: List[Dict]) -> float:
        """从导航序列计算路径距离"""
        if not navigation_sequence:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(navigation_sequence) - 1):
            current = navigation_sequence[i]
            next_item = navigation_sequence[i + 1]
            
            # 获取坐标
            current_coords = current.get('coords', {})
            next_coords = next_item.get('coords', {})
            
            current_x = current_coords.get('x', 0)
            current_y = current_coords.get('y', 0)
            next_x = next_coords.get('x', 0)
            next_y = next_coords.get('y', 0)
            
            # 计算欧几里得距离
            distance = np.sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
            total_distance += distance
        
        return total_distance
    
    def plan_path_batch(self, start_quad_ids: Tensor, goal_quad_ids: Tensor) -> Tensor:
        """
        批量路径规划
        
        Args:
            start_quad_ids: 起始quad ID张量 [batch_size]
            goal_quad_ids: 目标quad ID张量 [batch_size]
            
        Returns:
            批量路径坐标张量 [batch_size, max_path_length, 2]，其中2表示(x, y)坐标
            如果某个路径为空，对应位置填充为0
        """
        if not isinstance(start_quad_ids, Tensor):
            start_quad_ids = torch.tensor(start_quad_ids, device=self.device)
        if not isinstance(goal_quad_ids, Tensor):
            goal_quad_ids = torch.tensor(goal_quad_ids, device=self.device)
        
        batch_size = len(start_quad_ids)
        
        # 批量获取quad中心点和方向
        start_centers = self._get_quad_centers_batch(start_quad_ids)
        goal_centers = self._get_quad_centers_batch(goal_quad_ids)
        start_directions = self._get_quad_directions_batch(start_quad_ids)
        goal_directions = self._get_quad_directions_batch(goal_quad_ids)
        
        # 先获取所有路径以确定最大长度
        all_paths = []
        max_path_length = 0
        
        for i in range(batch_size):
            start_id = start_quad_ids[i].item()
            goal_id = goal_quad_ids[i].item()
            
            # 使用单路径规划方法
            path_result = self.plan_path(start_id, goal_id)
            all_paths.append(path_result)
            
            # 提取坐标点
            coords = []
            for item in path_result:
                if 'coords' in item:
                    coords.append([item['coords']['x'], item['coords']['y']])
            
            max_path_length = max(max_path_length, len(coords))
        
        # 创建结果张量 [batch_size, max_path_length, 2]
        if max_path_length == 0:
            # 如果所有路径都为空，返回空张量
            return torch.zeros(batch_size, 0, 2, device=self.device)
        
        result_tensor = torch.zeros(batch_size, max_path_length, 2, device=self.device)
        
        # 填充坐标数据
        for i, path_result in enumerate(all_paths):
            coords = []
            for item in path_result:
                if 'coords' in item:
                    coords.append([item['coords']['x'], item['coords']['y']])
            
            if coords:
                coords_tensor = torch.tensor(coords, device=self.device, dtype=torch.float32)
                result_tensor[i, :len(coords), :] = coords_tensor
        
        return result_tensor
    
    def plan_path_batch_with_lengths(self, start_quad_ids: Tensor, goal_quad_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        批量路径规划，同时返回路径长度信息
        
        Args:
            start_quad_ids: 起始quad ID张量 [batch_size]
            goal_quad_ids: 目标quad ID张量 [batch_size]
            
        Returns:
            paths_tensor: 批量路径坐标张量 [batch_size, max_path_length, 2]
            path_lengths: 每个路径的实际长度张量 [batch_size]
        """
        if not isinstance(start_quad_ids, Tensor):
            start_quad_ids = torch.tensor(start_quad_ids, device=self.device)
        if not isinstance(goal_quad_ids, Tensor):
            goal_quad_ids = torch.tensor(goal_quad_ids, device=self.device)
        
        batch_size = len(start_quad_ids)
        
        # 批量获取quad中心点和方向
        start_centers = self._get_quad_centers_batch(start_quad_ids)
        goal_centers = self._get_quad_centers_batch(goal_quad_ids)
        start_directions = self._get_quad_directions_batch(start_quad_ids)
        goal_directions = self._get_quad_directions_batch(goal_quad_ids)
        
        # 先获取所有路径以确定最大长度
        all_paths = []
        path_lengths = []
        max_path_length = 0
        
        for i in range(batch_size):
            start_id = start_quad_ids[i].item()
            goal_id = goal_quad_ids[i].item()
            
            # 使用单路径规划方法
            path_result = self.plan_path(start_id, goal_id)
            all_paths.append(path_result)
            
            # 提取坐标点
            coords = []
            for item in path_result:
                if 'coords' in item:
                    coords.append([item['coords']['x'], item['coords']['y']])
            
            path_lengths.append(len(coords))
            max_path_length = max(max_path_length, len(coords))
        
        # 创建结果张量 [batch_size, max_path_length, 2]
        if max_path_length == 0:
            # 如果所有路径都为空，返回空张量
            return torch.zeros(batch_size, 0, 2, device=self.device), torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        result_tensor = torch.zeros(batch_size, max_path_length, 2, device=self.device)
        path_lengths_tensor = torch.tensor(path_lengths, device=self.device, dtype=torch.long)
        
        # 填充坐标数据
        for i, path_result in enumerate(all_paths):
            coords = []
            for item in path_result:
                if 'coords' in item:
                    coords.append([item['coords']['x'], item['coords']['y']])
            
            if coords:
                coords_tensor = torch.tensor(coords, device=self.device, dtype=torch.float32)
                result_tensor[i, :len(coords), :] = coords_tensor
        
        return result_tensor, path_lengths_tensor
    
    def find_nearest_quads_gpu(self, query_points: Tensor, max_distance: float = 3.0) -> Tuple[Tensor, Tensor]:
        """
        使用GPU加速查找最近quads
        
        Args:
            query_points: 查询点张量 [num_points, 2]
            max_distance: 最大距离阈值
            
        Returns:
            nearest_quad_ids: 最近quad ID张量 [num_points]
            distances: 距离张量 [num_points]
        """
        if self._quad_centers_gpu is None:
            return torch.zeros(len(query_points), dtype=torch.long, device=self.device), \
                   torch.zeros(len(query_points), dtype=torch.float32, device=self.device)
        
        # 计算所有查询点到所有quad中心的距离
        # [num_points, 1, 2] - [1, num_quads, 2] = [num_points, num_quads, 2]
        diff = query_points.unsqueeze(1) - self._quad_centers_gpu.unsqueeze(0)
        distances = torch.norm(diff, dim=-1)  # [num_points, num_quads]
        
        # 找到每个查询点的最近quad
        nearest_distances, nearest_indices = torch.min(distances, dim=1)
        
        # 应用距离阈值
        valid_mask = nearest_distances < max_distance
        nearest_quad_ids = torch.where(valid_mask, nearest_indices, 
                                     torch.full_like(nearest_indices, -1))
        
        return nearest_quad_ids, nearest_distances
    
    def compute_path_distances_batch(self, path_results: List[List[List]]) -> Tensor:
        """
        批量计算路径距离
        
        Args:
            path_results: 路径结果列表 [batch_size, path_length, 3]
            
        Returns:
            距离张量 [batch_size]
        """
        if not self.G:
            return torch.zeros(len(path_results), dtype=torch.float32, device=self.device)
        
        distances = torch.zeros(len(path_results), dtype=torch.float32, device=self.device)
        
        for i, path_result in enumerate(path_results):
            if not path_result:
                continue
            
            total_distance = 0.0
            for j in range(len(path_result) - 1):
                src_ids = path_result[j]
                dst_ids = path_result[j + 1]
                
                # 找到对应的节点对
                src_nodes = [n for n in self.G.nodes if list(n[:3]) == src_ids]
                dst_nodes = [n for n in self.G.nodes if list(n[:3]) == dst_ids]
                
                for s in src_nodes:
                    for d in dst_nodes:
                        if self.G.has_edge(s, d):
                            total_distance += self.G[s][d]['distance']
                            break
                    else:
                        continue
                    break
            
            distances[i] = total_distance
        
        return distances
    
    def clear_gpu_cache(self):
        """清理GPU缓存，释放内存"""
        if self._quad_centers_gpu is not None:
            del self._quad_centers_gpu
            self._quad_centers_gpu = None
        
        if self._quad_directions_gpu is not None:
            del self._quad_directions_gpu
            self._quad_directions_gpu = None
        
        if self._waypoints_gpu is not None:
            del self._waypoints_gpu
            self._waypoints_gpu = None
        
        # 清理CPU缓存
        self._lanes_cache = None
        self._lanes_cache_waypoints = None
        
        # 强制GPU内存回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def to_device(self, target_device: torch.device):
        """将模型移动到指定设备"""
        self.device = target_device
        
        # 重新计算GPU张量
        if self._quad_centers_gpu is not None:
            self._quad_centers_gpu = self._quad_centers_gpu.to(target_device)
        if self._quad_directions_gpu is not None:
            self._quad_directions_gpu = self._quad_directions_gpu.to(target_device)
        if self._waypoints_gpu is not None:
            self._waypoints_gpu = self._waypoints_gpu.to(target_device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # 计算张量内存使用
        tensor_memory = 0
        for attr_name in ['_quad_centers_gpu', '_quad_directions_gpu', '_waypoints_gpu']:
            tensor = getattr(self, attr_name, None)
            if tensor is not None:
                tensor_memory += tensor.numel() * tensor.element_size()
        
        memory_info['tensor_memory_mb'] = tensor_memory / 1024**2  # MB
        
        return memory_info

def load_cross_data(cross_data_path: str) -> Optional[Dict]:
    """加载cross数据文件"""
    if not os.path.exists(cross_data_path):
        print(f"错误: cross数据文件不存在: {cross_data_path}")
        return None
    
    with open(cross_data_path, 'r', encoding='utf-8') as f:
        cross_data = json.load(f)
    return cross_data

def load_map_data(map_data_path: str) -> Optional[Dict]:
    """加载地图数据文件"""
    if not os.path.exists(map_data_path):
        print(f"错误: 地图数据文件不存在: {map_data_path}")
        return None
    
    with open(map_data_path, 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    return map_data 
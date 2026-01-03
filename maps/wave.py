"""
波函数探索（Wave Function Collapse）算法生成道路网络
生成与 preprocessor.py 相同格式的地图数据结构

使用方法：
1. 直接运行此文件生成基础几何数据：
   python maps/wave.py

2. 在代码中使用：
   from maps.wave import generate_map_with_wfc, get_preprocessor_compatible_data
   
   # 生成道路网络
   map_data = generate_map_with_wfc(grid_size=15, seed=42)
   
   # 获取preprocessor兼容格式
   preprocessor_data = get_preprocessor_compatible_data(
       map_data['lines_data'],
       map_data['circles_data'],
       map_data['arcs_data']
   )
   # 然后可以将这些数据传递给preprocessor的处理流程

3. 与preprocessor集成：
   修改preprocessor.py，将DXF读取部分替换为：
   from maps.wave import generate_map_with_wfc, get_preprocessor_compatible_data
   map_data = generate_map_with_wfc()
   preprocessor_data = get_preprocessor_compatible_data(...)
   lines_data = preprocessor_data['lines_data']
   circles_data = preprocessor_data['circles_data']
   arcs_data = preprocessor_data['arcs_data']
   然后继续使用preprocessor的后续处理函数
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')
import math
import numpy as np
import random
import json
import os
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional

# 导入工具模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry_utils import (
    calculate_distance, normalize_angle, normalize_angle_degrees, angle_difference
)

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
config = load_config()

# 从配置文件读取参数
TOLERANCE = config['preprocessor']['tolerance']
SAMPLE_DISTANCE = config['preprocessor']['sample_distance']
RECTANGLE_LENGTH = config['preprocessor']['rectangle_length']
RECTANGLE_WIDTH = config['preprocessor']['rectangle_width']
OOB_NUDGE_DISTANCE = config['preprocessor']['oob_nudge_distance']
CELL_SIZE = config['preprocessor']['cell_size']
W_LANE_SAMPLE_DISTANCE = config['preprocessor']['w_lane_sample_distance']

# ==================== 波函数探索核心算法 ====================
class RoadSegment:
    """道路段基类"""
    def __init__(self, road_id: int, start: Tuple[float, float, float], 
                 end: Tuple[float, float, float], direction: int = 1):
        self.road_id = road_id
        self.start = start
        self.end = end
        self.direction = direction  # 1: 正向, -1: 反向
        self.start_angle = None
        self.end_angle = None
        self._compute_angles()
    
    def _compute_angles(self):
        """计算起点和终点的角度"""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        angle = math.atan2(dy, dx)
        self.start_angle = normalize_angle_degrees(math.degrees(angle))
        self.end_angle = normalize_angle_degrees(math.degrees(angle))
    
    def get_connection_point(self, side: str) -> Tuple[float, float, float]:
        """获取连接点（start或end）"""
        return self.start if side == 'start' else self.end
    
    def get_connection_angle(self, side: str) -> float:
        """获取连接点的角度"""
        if side == 'start':
            return self.start_angle if self.direction == 1 else (self.end_angle + 180) % 360
        else:
            return self.end_angle if self.direction == 1 else (self.start_angle + 180) % 360


class LineSegment(RoadSegment):
    """直线段"""
    def __init__(self, road_id: int, start: Tuple[float, float, float], 
                 end: Tuple[float, float, float], direction: int = 1):
        super().__init__(road_id, start, end, direction)
        self.type = 'line'
        self.length = calculate_distance(start[:2], end[:2])
        self.center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.0)
    
    def to_dict(self):
        """转换为字典格式（与preprocessor兼容）"""
        return {
            'road_id': self.road_id,
            'layer': 'WFC',
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'center': self.center
        }


class ArcSegment(RoadSegment):
    """圆弧段"""
    def __init__(self, road_id: int, center: Tuple[float, float, float], 
                 radius: float, start_angle: float, end_angle: float, direction: int = 1):
        self.type = 'arc'
        self.road_id = road_id
        self.center = center
        self.radius = radius
        self.start_angle = normalize_angle_degrees(start_angle)
        self.end_angle = normalize_angle_degrees(end_angle)
        self.direction = direction
        
        # 计算起点和终点（保证起点与传入的几何约束一致）
        start_rad = math.radians(self.start_angle)
        end_rad = math.radians(self.end_angle)
        start = (center[0] + radius * math.cos(start_rad),
                 center[1] + radius * math.sin(start_rad), center[2])
        end = (center[0] + radius * math.cos(end_rad),
               center[1] + radius * math.sin(end_rad), center[2])
        super().__init__(road_id, start, end, direction)
    
    def to_dict(self):
        """转换为字典格式（与preprocessor兼容）"""
        return {
            'road_id': self.road_id,
            'center': self.center,
            'radius': self.radius,
            'start_angle': self.start_angle,
            'end_angle': self.end_angle,
            'direction': self.direction
        }


class CircleSegment(RoadSegment):
    """圆形段（环岛）"""
    def __init__(self, road_id: int, center: Tuple[float, float, float], radius: float):
        self.type = 'circle'
        self.road_id = road_id
        self.center = center
        self.radius = radius
        # 圆形是闭合的，起点和终点相同；为了更好地与上游几何连接，直接使用圆心作为起点
        start = (center[0], center[1], center[2])
        super().__init__(road_id, start, start, 1)
        self.start_angle = 0.0
        self.end_angle = 360.0
    
    def to_dict(self):
        """转换为字典格式（与preprocessor兼容）"""
        return {
            'road_id': self.road_id,
            'center': self.center,
            'radius': self.radius
        }


class WaveFunctionCollapse:
    """波函数探索算法实现"""
    
    def __init__(self, grid_size: int = 20, cell_size: float = 50.0, 
                 min_road_length: float = 30.0, max_road_length: float = 100.0,
                 min_arc_radius: float = 20.0, max_arc_radius: float = 50.0):
        """
        初始化WFC算法
        
        参数:
        grid_size: 网格大小（grid_size x grid_size）
        cell_size: 每个网格单元的实际尺寸（米）
        min_road_length: 最小道路长度
        max_road_length: 最大道路长度
        min_arc_radius: 最小圆弧半径
        max_arc_radius: 最大圆弧半径
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.min_road_length = min_road_length
        self.max_road_length = max_road_length
        self.min_arc_radius = min_arc_radius
        self.max_arc_radius = max_arc_radius
        
        # 定义道路段类型
        self.segment_types = ['line', 'arc_left', 'arc_right', 'circle']
        
        # 定义连接规则：每个方向（上、右、下、左）允许的段类型
        # 格式: {segment_type: {direction: [allowed_types]}}
        self.connection_rules = {
            'line': {
                'up': ['line', 'arc_left', 'arc_right'],
                'right': ['line', 'arc_left', 'arc_right'],
                'down': ['line', 'arc_left', 'arc_right'],
                'left': ['line', 'arc_left', 'arc_right']
            },
            'arc_left': {
                'up': ['line', 'arc_left'],
                'right': ['line', 'arc_left'],
                'down': ['line', 'arc_left'],
                'left': ['line', 'arc_left']
            },
            'arc_right': {
                'up': ['line', 'arc_right'],
                'right': ['line', 'arc_right'],
                'down': ['line', 'arc_right'],
                'left': ['line', 'arc_right']
            },
            'circle': {
                'up': ['line'],
                'right': ['line'],
                'down': ['line'],
                'left': ['line']
            }
        }
        
        # 网格状态：每个单元格的可能状态集合
        self.grid = {}
        # 已坍缩的单元格
        self.collapsed = {}
        # 生成的道路段
        self.segments = []
        # 单元格到道路段的映射（用于连接）
        self.cell_to_segment = {}
        # 道路段连接点映射（用于确保连接）
        self.connection_points = {}  # {(x, y, direction): (point, angle)}
        self.road_id_counter = 1
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int, str]]:
        """获取邻居单元格（上、右、下、左）"""
        neighbors = []
        if y > 0:
            neighbors.append((x, y - 1, 'up'))
        if x < self.grid_size - 1:
            neighbors.append((x + 1, y, 'right'))
        if y < self.grid_size - 1:
            neighbors.append((x, y + 1, 'down'))
        if x > 0:
            neighbors.append((x - 1, y, 'left'))
        return neighbors
    
    def _initialize_grid(self):
        """初始化网格，所有单元格都包含所有可能的状态"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.grid[(x, y)] = set(self.segment_types)
                self.collapsed[(x, y)] = None
    
    def _calculate_entropy(self, x: int, y: int) -> float:
        """计算单元格的熵（不确定性）"""
        if (x, y) in self.collapsed and self.collapsed[(x, y)] is not None:
            return float('inf')  # 已坍缩
        
        possible_states = self.grid.get((x, y), set())
        if len(possible_states) == 0:
            return float('inf')  # 无解
        
        # 熵 = -log(状态数)
        return -math.log(len(possible_states))
    
    def _has_connected_neighbor(self, x: int, y: int) -> bool:
        """检查单元格是否有已连接的邻居"""
        neighbors = self._get_neighbors(x, y)
        for nx, ny, _ in neighbors:
            if (nx, ny) in self.cell_to_segment:
                return True
        return False
    
    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """找到熵最小的单元格，优先选择有已连接邻居的单元格。
        若当前已无与现有道路相邻的单元格，则停止生成，避免产生零散“孤岛”几何。
        """
        min_entropy = float('inf')
        min_cell = None
        candidates_with_neighbors = []
        candidates_without_neighbors = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                entropy = self._calculate_entropy(x, y)
                if entropy == float('inf'):
                    continue
                
                has_neighbor = self._has_connected_neighbor(x, y)
                if has_neighbor:
                    candidates_with_neighbors.append(((x, y), entropy))
                else:
                    candidates_without_neighbors.append(((x, y), entropy))
        
        # 优先选择有邻居的单元格；如果没有，则直接停止（不再生成新的“种子”道路）
        if candidates_with_neighbors:
            candidates_with_neighbors.sort(key=lambda c: c[1])
            return candidates_with_neighbors[0][0]
        # 没有任何与现有网络相邻的格子 → 返回 None 终止生成
        return None
    
    def _propagate_constraints(self, x: int, y: int, segment_type: str):
        """传播约束到邻居单元格"""
        queue = deque([(x, y)])
        visited = set([(x, y)])
        
        while queue:
            cx, cy = queue.popleft()
            neighbors = self._get_neighbors(cx, cy)
            
            for nx, ny, direction in neighbors:
                if (nx, ny) in visited:
                    continue
                
                # 获取当前单元格的状态
                current_states = self.grid.get((cx, cy), set())
                if len(current_states) == 0:
                    continue
                
                # 确定当前单元格的状态（如果已坍缩）
                if (cx, cy) in self.collapsed and self.collapsed[(cx, cy)] is not None:
                    current_type = self.collapsed[(cx, cy)]
                elif len(current_states) == 1:
                    current_type = list(current_states)[0]
                else:
                    continue
                
                # 根据连接规则更新邻居的可能状态
                allowed_types = self.connection_rules.get(current_type, {}).get(direction, [])
                neighbor_states = self.grid.get((nx, ny), set())
                
                # 计算新的可能状态（交集）
                new_states = neighbor_states & set(allowed_types)
                
                # 如果状态发生变化，需要继续传播
                if new_states != neighbor_states:
                    self.grid[(nx, ny)] = new_states
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
    
    def _get_connection_from_neighbor(self, x: int, y: int) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """从相邻单元格获取连接点和角度"""
        neighbors = self._get_neighbors(x, y)
        
        # 检查每个邻居是否有已生成的道路段
        for nx, ny, direction in neighbors:
            if (nx, ny) in self.cell_to_segment:
                segment = self.cell_to_segment[(nx, ny)]
                
                # 确定连接点（根据方向）
                if direction == 'up':  # 邻居在上方，我们从邻居的底部连接
                    connection_point = segment.end
                    connection_angle = segment.end_angle
                elif direction == 'right':  # 邻居在右侧，我们从邻居的左侧连接
                    connection_point = segment.start
                    connection_angle = (segment.start_angle + 180) % 360
                elif direction == 'down':  # 邻居在下方，我们从邻居的顶部连接
                    connection_point = segment.start
                    connection_angle = (segment.start_angle + 180) % 360
                else:  # direction == 'left'，邻居在左侧，我们从邻居的右侧连接
                    connection_point = segment.end
                    connection_angle = segment.end_angle
                
                return (connection_point, connection_angle)
        
        return None
    
    def _create_segment(self, x: int, y: int, segment_type: str) -> Optional[RoadSegment]:
        """在指定位置创建道路段，确保与相邻道路段连接"""
        road_id = self.road_id_counter
        self.road_id_counter += 1
        
        # 尝试从相邻单元格获取连接点（若存在则一定与已有道路相连）
        connection_info = self._get_connection_from_neighbor(x, y)
        
        if connection_info:
            # 有连接点，从连接点开始生成
            start_point, start_angle = connection_info
            start_angle_rad = math.radians(start_angle)
        else:
            # 没有连接点，使用网格中心作为起点
            world_x = (x - self.grid_size / 2) * self.cell_size
            world_y = (y - self.grid_size / 2) * self.cell_size
            start_point = (world_x, world_y, 0.0)
            # 初始种子方向任意
            start_angle_rad = random.uniform(0, 2 * math.pi)
            start_angle = math.degrees(start_angle_rad)
        
        if segment_type == 'line':
            # 生成直线段：若来自已有道路，则在原方向附近小范围偏转，增强连贯性
            if connection_info:
                delta = math.radians(random.uniform(-30.0, 30.0))
                line_angle_rad = start_angle_rad + delta
            else:
                line_angle_rad = start_angle_rad
            length = random.uniform(self.min_road_length, self.max_road_length)
            end = (
                start_point[0] + length * math.cos(line_angle_rad),
                start_point[1] + length * math.sin(line_angle_rad),
                0.0
            )
            
            return LineSegment(road_id, start_point, end)
        
        elif segment_type == 'arc_left' or segment_type == 'arc_right':
            # 创建左转或右转圆弧（保证起点与已有几何严格相连）
            radius = random.uniform(self.min_arc_radius, self.max_arc_radius)
            start_angle_deg = normalize_angle_degrees(start_angle)
            
            # 左转90度或右转90度
            angle_diff = 90 if segment_type == 'arc_left' else -90
            end_angle_deg = normalize_angle_degrees(start_angle_deg + angle_diff)
            
            # 计算圆心位置：确保 start_point 落在圆上
            # x_s = cx + r cos(theta_s)  =>  cx = x_s - r cos(theta_s)
            # y_s = cy + r sin(theta_s)  =>  cy = y_s - r sin(theta_s)
            start_rad = math.radians(start_angle_deg)
            center = (
                start_point[0] - radius * math.cos(start_rad),
                start_point[1] - radius * math.sin(start_rad),
                0.0,
            )
            
            return ArcSegment(road_id, center, radius, start_angle_deg, end_angle_deg)
        
        elif segment_type == 'circle':
            # 创建环岛（以起点为圆心）
            radius = random.uniform(self.min_arc_radius, self.max_arc_radius)
            center = start_point
            return CircleSegment(road_id, center, radius)
        
        return None
    
    def generate(self, seed: Optional[int] = None) -> List[RoadSegment]:
        """生成道路网络"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 初始化网格
        self._initialize_grid()
        
        # 从中心开始
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        
        # 强制中心单元格为直线，并设置初始方向
        self.collapsed[(start_x, start_y)] = 'line'
        self.grid[(start_x, start_y)] = {'line'}
        
        # 创建初始道路段（从中心开始，随机方向）
        world_x = (start_x - self.grid_size / 2) * self.cell_size
        world_y = (start_y - self.grid_size / 2) * self.cell_size
        initial_angle = random.uniform(0, 2 * math.pi)
        length = random.uniform(self.min_road_length, self.max_road_length)
        start = (world_x, world_y, 0.0)
        end = (world_x + length * math.cos(initial_angle),
               world_y + length * math.sin(initial_angle), 0.0)
        
        initial_segment = LineSegment(self.road_id_counter, start, end)
        self.road_id_counter += 1
        self.segments.append(initial_segment)
        self.cell_to_segment[(start_x, start_y)] = initial_segment
        
        # 传播初始约束
        self._propagate_constraints(start_x, start_y, 'line')
        
        # 迭代坍缩
        max_iterations = self.grid_size * self.grid_size
        iteration = 0
        
        while iteration < max_iterations:
            # 找到熵最小的单元格
            cell = self._find_min_entropy_cell()
            if cell is None:
                break
            
            x, y = cell
            possible_states = self.grid.get((x, y), set())
            
            if len(possible_states) == 0:
                # 无解，跳过
                self.collapsed[(x, y)] = None
                iteration += 1
                continue
            
            # 随机选择一个可能的状态
            segment_type = random.choice(list(possible_states))
            
            # 坍缩单元格
            self.collapsed[(x, y)] = segment_type
            self.grid[(x, y)] = {segment_type}
            
            # 创建道路段
            segment = self._create_segment(x, y, segment_type)
            if segment:
                self.segments.append(segment)
                # 记录单元格到道路段的映射
                self.cell_to_segment[(x, y)] = segment
            
            # 传播约束
            self._propagate_constraints(x, y, segment_type)
            
            iteration += 1
        
        return self.segments


# ==================== 数据转换函数 ====================

def segments_to_preprocessor_format(segments: List[RoadSegment]) -> Tuple[List, List, List]:
    """
    将WFC生成的道路段转换为preprocessor格式
    
    返回:
    lines_data, circles_data, arcs_data
    """
    lines_data = []
    circles_data = []
    arcs_data = []
    
    for segment in segments:
        if isinstance(segment, LineSegment):
            lines_data.append(segment.to_dict())
        elif isinstance(segment, CircleSegment):
            circles_data.append(segment.to_dict())
        elif isinstance(segment, ArcSegment):
            arcs_data.append(segment.to_dict())
    
    return lines_data, circles_data, arcs_data


def generate_map_with_wfc(grid_size: int = 20, cell_size: float = 50.0,
                          min_road_length: float = 30.0, max_road_length: float = 100.0,
                          min_arc_radius: float = 20.0, max_arc_radius: float = 50.0,
                          seed: Optional[int] = None) -> Dict:
    """
    使用WFC生成完整的地图数据（与preprocessor格式兼容）
    
    返回包含以下键的字典:
    - lines_data: 直线数据列表
    - circles_data: 圆形数据列表
    - arcs_data: 圆弧数据列表
    """
    wfc = WaveFunctionCollapse(
        grid_size=grid_size,
        cell_size=cell_size,
        min_road_length=min_road_length,
        max_road_length=max_road_length,
        min_arc_radius=min_arc_radius,
        max_arc_radius=max_arc_radius
    )
    
    segments = wfc.generate(seed=seed)
    lines_data, circles_data, arcs_data = segments_to_preprocessor_format(segments)
    
    return {
        'lines_data': lines_data,
        'circles_data': circles_data,
        'arcs_data': arcs_data,
        'segments': segments
    }


# ==================== 完整地图生成（集成preprocessor处理流程） ====================

def process_with_preprocessor(lines_data: List, circles_data: List, arcs_data: List) -> Dict:
    """
    将WFC生成的数据传递给preprocessor的处理流程
    
    这个方法通过修改preprocessor的全局变量来复用其处理逻辑
    注意：这需要preprocessor.py支持作为模块导入
    
    返回:
    包含完整处理结果的字典
    """
    # 由于preprocessor.py是脚本式执行，我们需要采用不同的策略
    # 方案：直接复制preprocessor的关键处理函数到这里
    
    # 这里返回基础数据，实际使用时需要完整集成
    return {
        'lines_data': lines_data,
        'circles_data': circles_data,
        'arcs_data': arcs_data,
        'note': '需要调用preprocessor的处理函数生成polygons_data, oob_points, w_lanes'
    }


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("=== 波函数探索道路网络生成 ===")
    
    # 生成地图
    map_data = generate_map_with_wfc(
        grid_size=15,
        cell_size=40.0,
        min_road_length=25.0,
        max_road_length=80.0,
        min_arc_radius=15.0,
        max_arc_radius=40.0,
        seed=42
    )
    
    print(f"生成了 {len(map_data['lines_data'])} 条直线")
    print(f"生成了 {len(map_data['circles_data'])} 个圆形")
    print(f"生成了 {len(map_data['arcs_data'])} 条圆弧")
    print(f"总共 {len(map_data['segments'])} 个道路段")
    
    # 导出为JSON格式（基础几何数据）
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "wfc_generated.json")
    
    export_data = {
        "map_name": "wfc_generated.json",
        "quads": [],  # 需要后续处理生成
        "oob_points": [],  # 需要后续处理生成
        "w_lanes": [],  # 需要后续处理生成
        "geometry": {
            "lines": [
                {
                    "road_id": item["road_id"],
                    "lane_id": int(item.get("lane_id", 1)),
                    "start_poly_id": None,
                    "end_poly_id": None,
                    "layer": str(item.get("layer", "WFC")),
                    "start": [float(item["start"][0]), float(item["start"][1]), float(item["start"][2])],
                    "end": [float(item["end"][0]), float(item["end"][1]), float(item["end"][2])],
                    "length": float(item.get("length", 0.0))
                }
                for item in map_data['lines_data']
            ],
            "circles": [
                {
                    "road_id": item["road_id"],
                    "lane_id": int(item.get("lane_id", 1)),
                    "start_poly_id": None,
                    "end_poly_id": None,
                    "center": [float(item["center"][0]), float(item["center"][1]), float(item["center"][2])],
                    "radius": float(item["radius"])
                }
                for item in map_data['circles_data']
            ],
            "arcs": [
                {
                    "road_id": item["road_id"],
                    "lane_id": int(item.get("lane_id", 1)),
                    "start_poly_id": None,
                    "end_poly_id": None,
                    "center": [float(item["center"][0]), float(item["center"][1]), float(item["center"][2])],
                    "radius": float(item["radius"]),
                    "start_angle": float(item["start_angle"]),
                    "end_angle": float(item["end_angle"])
                }
                for item in map_data['arcs_data']
            ]
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已导出基础几何数据到: {output_path}")
    print("\n提示：要生成完整的polygons_data, oob_points, w_lanes等，")
    print("      可以修改preprocessor.py，将DXF读取部分替换为WFC生成的数据")


def get_preprocessor_compatible_data(lines_data: List, circles_data: List, arcs_data: List) -> Dict:
    """
    将WFC生成的数据转换为preprocessor可以直接使用的格式
    这个方法返回的数据可以直接赋值给preprocessor.py中的全局变量：
    - lines_data
    - circles_data  
    - arcs_data
    
    然后可以调用preprocessor的其他处理函数生成完整地图
    
    返回:
    包含格式化数据的字典
    """
    # 确保所有数据都有lane_id字段
    for item in lines_data:
        if 'lane_id' not in item:
            item['lane_id'] = 1
    for item in circles_data:
        if 'lane_id' not in item:
            item['lane_id'] = 1
    for item in arcs_data:
        if 'lane_id' not in item:
            item['lane_id'] = 1
    
    return {
        'lines_data': lines_data,
        'circles_data': circles_data,
        'arcs_data': arcs_data
    }


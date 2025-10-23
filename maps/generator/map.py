import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
import ezdxf, math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc, Polygon

# 读取DXF文件
doc = ezdxf.readfile("./maps/generator/map.dxf")
msp = doc.modelspace()
# 创建matplotlib图形
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')

# 存储所有几何数据
lines_data = []
circles_data = []
arcs_data = []

# 存储所有四边形数据
polygons_data = []

# 重合检测的容差
TOLERANCE = 0.01  # 1mm容差
# 道路ID计数器
road_id_counter = 1
# 四边形ID计数器
poly_id_counter = 1

# 采样参数
SAMPLE_DISTANCE = 1.0  # 1米采样距离
RECTANGLE_LENGTH = 1.0  # 四边形长度1米
RECTANGLE_WIDTH = 5.0   # 四边形宽度5米

def normalize_angle(angle):
    """将角度标准化到[0, 2π]范围"""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle

def angle_difference(angle1, angle2):
    """计算两个角度之间的最小差值"""
    diff = angle1 - angle2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

def calculate_distance(point1, point2):
    """计算两点之间的距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def normalize_angle_degrees(angle):
    """将角度标准化到[0, 360)范围（度数）"""
    return angle % 360

def increment_road_id():
    """递增道路ID并返回新ID"""
    global road_id_counter
    road_id_counter += 1
    return road_id_counter - 1

def check_duplicate_geometry(new_item, existing_items, tolerance=TOLERANCE):
    """检查几何体是否重复"""
    for existing_item in existing_items:
        if calculate_distance(new_item['center'], existing_item['center']) < tolerance:
            # 检查其他参数
            if 'radius' in new_item and 'radius' in existing_item:
                if abs(new_item['radius'] - existing_item['radius']) < tolerance:
                    # 对于圆弧，还需要检查角度
                    if 'start_angle' in new_item and 'start_angle' in existing_item:
                        start_angle_diff = abs(new_item['start_angle'] - existing_item['start_angle'])
                        end_angle_diff = abs(new_item['end_angle'] - existing_item['end_angle'])
                        
                        # 处理角度跨越0度的情况
                        if start_angle_diff > 180:
                            start_angle_diff = 360 - start_angle_diff
                        if end_angle_diff > 180:
                            end_angle_diff = 360 - end_angle_diff
                            
                        if start_angle_diff < 1.0 and end_angle_diff < 1.0:
                            return True
                    else:
                        # 圆形或线条
                        return True
            elif 'start' in new_item and 'start' in existing_item:
                # 检查起点和终点
                start_dist = calculate_distance(new_item['start'], existing_item['start'])
                end_dist = calculate_distance(new_item['end'], existing_item['end'])
                start_end_dist = calculate_distance(new_item['start'], existing_item['end'])
                end_start_dist = calculate_distance(new_item['end'], existing_item['start'])
                if ((start_dist < tolerance and end_dist < tolerance) or 
                    (start_end_dist < tolerance and end_start_dist < tolerance)):
                    return True
    return False

def create_seamless_trapezoid(center_x, center_y, tangent_angle, normal_angle, 
                             bottom_width, top_width, height, prev_vertices=None):
    """
    创建无缝连接的梯形，确保与相邻梯形紧密连接
    
    参数:
    center_x, center_y: 梯形中心点
    tangent_angle: 切线方向角度
    normal_angle: 法线方向角度
    bottom_width: 下底宽度
    top_width: 上底宽度
    height: 梯形高度
    prev_vertices: 前一个梯形的顶点（用于无缝连接）
    
    返回:
    vertices: 四个顶点坐标
    """
    if prev_vertices is None:
        # 第一个梯形，创建起始梯形
        # 标准化角度
        tangent_angle = normalize_angle(tangent_angle)
        normal_angle = normalize_angle(normal_angle)
        
        # 计算切线方向单位向量
        t_x = math.cos(tangent_angle)
        t_y = math.sin(tangent_angle)
        
        n_x = math.cos(normal_angle)
        n_y = math.sin(normal_angle)
        
        # 计算四个顶点
        bottom_half = bottom_width / 2
        top_half = top_width / 2
        height_half = height / 2
        
        # 下底两个顶点：p_i - (bottom_i / 2) * t_i ± (h_i / 2) * n_i
        bottom_left_x = center_x - bottom_half * t_x - height_half * n_x
        bottom_left_y = center_y - bottom_half * t_y - height_half * n_y
        
        bottom_right_x = center_x - bottom_half * t_x + height_half * n_x
        bottom_right_y = center_y - bottom_half * t_y + height_half * n_y
        
        # 上底两个顶点：p_i + (top_i / 2) * t_i ± (h_i / 2) * n_i
        top_left_x = center_x + top_half * t_x - height_half * n_x
        top_left_y = center_y + top_half * t_y - height_half * n_y
        
        top_right_x = center_x + top_half * t_x + height_half * n_x
        top_right_y = center_y + top_half * t_y + height_half * n_y
        
        # 四个顶点（按顺序排列）
        vertices = [
            (top_left_x, top_left_y),      # 上底左顶点
            (top_right_x, top_right_y),     # 上底右顶点
            (bottom_right_x, bottom_right_y),  # 下底右顶点
            (bottom_left_x, bottom_left_y)   # 下底左顶点
        ]
    else:
        # 后续梯形，与前一个梯形无缝连接
        prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
        prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
        
        # 计算当前梯形的下底顶点
        t_x = math.cos(tangent_angle)
        t_y = math.sin(tangent_angle)
        
        n_x = math.cos(normal_angle)
        n_y = math.sin(normal_angle)
        
        # 计算当前梯形的下底顶点
        bottom_half = bottom_width / 2
        height_half = height / 2
        
        # 下底左顶点
        bottom_left_x = center_x - bottom_half * t_x - height_half * n_x
        bottom_left_y = center_y - bottom_half * t_y - height_half * n_y
        
        # 下底右顶点
        bottom_right_x = center_x - bottom_half * t_x + height_half * n_x
        bottom_right_y = center_y - bottom_half * t_y + height_half * n_y
        
        # 四个顶点
        vertices = [
            prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
            prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
            (bottom_right_x, bottom_right_y),  # 下底右顶点
            (bottom_left_x, bottom_left_y)     # 下底左顶点
        ]
    
    return vertices

def sample_line_points(start, end, sample_distance):
    """为直线生成等间距采样点"""
    start_x, start_y = start[0], start[1]
    end_x, end_y = end[0], end[1]
    
    # 计算直线长度
    length = calculate_distance(start, end)
    
    # 计算采样点数量
    n_points = max(1, int(length / sample_distance) + 1)
    
    # 生成采样点
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            x, y = start_x, start_y
        elif i == n_points - 1:
            # 最后一个点，精确使用终点
            x, y = end_x, end_y
        else:
            # 中间点，使用参数化方法
            t = i / (n_points - 1)
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
        points.append((x, y))
    
    
    return points, length

def sample_circle_points(center, radius, sample_distance):
    """为圆形生成等间距采样点"""
    circumference = 2 * math.pi * radius
    n_points = max(1, int(circumference / sample_distance) + 1)
    
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            angle = 0
        elif i == n_points - 1:
            # 最后一个点，精确使用起点（圆形闭合）
            angle = 0
        else:
            # 中间点，使用参数化方法
            angle = 2 * math.pi * i / n_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    return points, circumference

def sample_arc_points(center, radius, start_angle, end_angle, sample_distance):
    """为圆弧生成等间距采样点"""
    # 计算圆弧长度
    angle_diff = end_angle - start_angle
    if angle_diff < 0:
        angle_diff += 2 * math.pi
    
    arc_length = radius * angle_diff
    n_points = max(1, int(arc_length / sample_distance) + 1)
    
    points = []
    for i in range(n_points):
        if n_points == 1:
            # 只有一个点，使用起点
            angle = start_angle
        elif i == n_points - 1:
            # 最后一个点，精确使用终点
            angle = end_angle
        else:
            # 中间点，使用参数化方法
            t = i / (n_points - 1)
            angle = start_angle + t * angle_diff
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    
    return points, arc_length

# 处理直路
for line in msp.query("LINE"):
    start = line.dxf.start.xyz
    end   = line.dxf.end.xyz
    dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 检查是否与已有线条重合
    line_item = {
        'center': ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
        'start': start,
        'end': end
    }
    
    if not check_duplicate_geometry(line_item, lines_data):
        # 存储线条数据
        lines_data.append({
            'road_id': road_id_counter,
            'layer': line.dxf.layer,
            'start': start,
            'end': end,
            'length': length,
            'center': ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        })
        # 绘制线条，处理精度问题
        start_x, start_y = round(start[0], 6), round(start[1], 6)
        end_x, end_y = round(end[0], 6), round(end[1], 6)
        ax.plot([start_x, end_x], [start_y, end_y],
                color='blue', linewidth=2, label=f"line-road{road_id_counter}")
        increment_road_id()
# 处理环岛
for c in msp.query("CIRCLE"):
    center = c.dxf.center.xyz
    radius = c.dxf.radius

    # 圆心坐标精度规整：统一使用6位小数精度
    center_x = round(center[0], 6)
    center_y = round(center[1], 6)
    center_z = round(center[2], 6)
    center = (center_x, center_y, center_z)

    # 检查是否与已有圆形重合
    circle_item = {
        'center': (center_x, center_y),
        'radius': radius
    }
    
    if not check_duplicate_geometry(circle_item, circles_data):
        # 存储圆形数据
        circles_data.append({
            'road_id': road_id_counter,
            'center': center,
            'radius': radius
        })

        # 绘制圆形
        circle = Circle((center_x, center_y), radius,
                       fill=False, edgecolor='red', linewidth=2,
                       label=f"circle-road{road_id_counter}")
        ax.add_patch(circle)

        increment_road_id()
# 处理弯道
for a in msp.query("ARC"):
    center = a.dxf.center.xyz
    radius = a.dxf.radius
    # 圆心坐标精度规整：统一使用6位小数精度
    center_x = round(center[0], 6)
    center_y = round(center[1], 6)
    center_z = round(center[2], 6)
    center = (center_x, center_y, center_z)
    # DXF 的 ARC.start_angle / end_angle 单位为度
    start_angle_deg_raw = a.dxf.start_angle
    end_angle_deg_raw = a.dxf.end_angle
    # 归一化到 [0, 360)
    start_angle = normalize_angle_degrees(start_angle_deg_raw)
    end_angle = normalize_angle_degrees(end_angle_deg_raw)
    # 检查是否与已有圆弧重合
    arc_item = {
        'center': (center_x, center_y),
        'radius': radius,
        'start_angle': start_angle,
        'end_angle': end_angle
    }
    
    if not check_duplicate_geometry(arc_item, arcs_data):
        # 转为弧度用于三角函数计算
        start_angle_rad = math.radians(start_angle)
        end_angle_rad = math.radians(end_angle)
        
        # 存储圆弧数据
        arcs_data.append({
            'road_id': road_id_counter,
            'center': center,
            'radius': radius,
            'start_angle': start_angle,
            'end_angle': end_angle
        })
        
        # 计算圆弧的起始和结束点，处理数值精度
        # 使用更高精度计算，然后四舍五入到合理精度
        start_x = round(center_x + radius * math.cos(start_angle_rad), 6)
        start_y = round(center_y + radius * math.sin(start_angle_rad), 6)
        end_x = round(center_x + radius * math.cos(end_angle_rad), 6)
        end_y = round(center_y + radius * math.sin(end_angle_rad), 6)
        
        # 计算角度差，正确处理跨越0度的情况
        start_angle_deg = math.degrees(start_angle_rad)
        end_angle_deg = math.degrees(end_angle_rad)
        # 标准化到0-360度范围
        start_angle_deg = start_angle_deg % 360
        end_angle_deg = end_angle_deg % 360
        # 检查是否跨越0度
        if start_angle_deg > end_angle_deg:
            # 跨越0度的情况，如270°到0°
            # 从起始角度到360°
            angles1 = np.linspace(start_angle_rad, 2*math.pi, 20)
            # 从0°到结束角度
            angles2 = np.linspace(0, end_angle_rad, 20)
            angles = np.concatenate([angles1, angles2])
        else:
            # 正常情况
            angle_diff = end_angle_rad - start_angle_rad
            num_points = max(20, int(abs(math.degrees(angle_diff))))
            angles = np.linspace(start_angle_rad, end_angle_rad, num_points)
        # 计算圆弧坐标，处理数值精度
        # 使用更高精度计算，然后四舍五入到合理精度
        arc_x = np.round(center_x + radius * np.cos(angles), 6)
        arc_y = np.round(center_y + radius * np.sin(angles), 6)
        # 绘制圆弧，确保与线条端点精确匹配
        # 如果圆弧的起始点与某条线的端点接近，强制匹配
        for line_data in lines_data:
            line_start = (round(line_data['start'][0], 6), round(line_data['start'][1], 6))
            line_end = (round(line_data['end'][0], 6), round(line_data['end'][1], 6))
            
            # 检查圆弧起始点是否与线条端点匹配
            if abs(start_x - line_start[0]) < 0.001 and abs(start_y - line_start[1]) < 0.001:
                start_x, start_y = line_start
            elif abs(start_x - line_end[0]) < 0.001 and abs(start_y - line_end[1]) < 0.001:
                start_x, start_y = line_end
                
            # 检查圆弧结束点是否与线条端点匹配
            if abs(end_x - line_start[0]) < 0.001 and abs(end_y - line_start[1]) < 0.001:
                end_x, end_y = line_start
            elif abs(end_x - line_end[0]) < 0.001 and abs(end_y - line_end[1]) < 0.001:
                end_x, end_y = line_end
        
        # 重新计算圆弧路径，确保端点匹配
        if start_angle_deg > end_angle_deg:
            # 跨越0度的情况
            angles1 = np.linspace(start_angle_rad, 2*math.pi, 20)
            angles2 = np.linspace(0, end_angle_rad, 20)
            angles = np.concatenate([angles1, angles2])
        else:
            angle_diff = end_angle_rad - start_angle_rad
            num_points = max(20, int(abs(math.degrees(angle_diff))))
            angles = np.linspace(start_angle_rad, end_angle_rad, num_points)
        
        # 重新计算圆弧坐标
        arc_x = np.round(center_x + radius * np.cos(angles), 6)
        arc_y = np.round(center_y + radius * np.sin(angles), 6)
        
        # 绘制圆弧
        ax.plot(arc_x, arc_y, color='green', linewidth=2,
                label=f"arc-road{road_id_counter}")

        increment_road_id()

# 为所有道路生成采样点和四边形

# 处理直线道路的采样
for line_data in lines_data:
    road_id = line_data['road_id']
    start = line_data['start']
    end = line_data['end']
    
    # 生成采样点
    points, total_length = sample_line_points(start, end, SAMPLE_DISTANCE)
    
    
    # 计算所有点的方向角度
    angles = []
    for i, (x, y) in enumerate(points):
        if i < len(points) - 1:
            # 使用下一个点计算方向
            next_x, next_y = points[i + 1]
            direction_angle = math.atan2(next_y - y, next_x - x)
        else:
            # 最后一个点使用前一个点计算方向
            prev_x, prev_y = points[i - 1]
            direction_angle = math.atan2(y - prev_y, x - prev_x)
        angles.append(direction_angle)
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else None
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        # 计算梯形参数
        if next_angle is not None:
            # 根据下一个点的方向计算上底宽度
            angle_diff = abs(angle_difference(current_angle, next_angle))
            top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        else:
            # 最后一个点：使用与前一个点相同的上底宽度，确保无缝连接
            if i > 0:
                # 计算与前一个点的角度差
                prev_angle = angles[i - 1]
                angle_diff = abs(angle_difference(prev_angle, current_angle))
                top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)
            else:
                top_width = RECTANGLE_LENGTH
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': vertices,
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1
# 处理圆形道路的采样
for circle_data in circles_data:
    road_id = circle_data['road_id']
    center = circle_data['center']
    radius = circle_data['radius']
    
    # 生成采样点
    points, total_length = sample_circle_points(center, radius, SAMPLE_DISTANCE)
    
    
    # 计算所有点的方向角度
    angles = []
    for i, (x, y) in enumerate(points):
        # 计算切线方向（圆形切线方向）
        # 从圆心到采样点的向量，切线方向垂直于此向量
        direction_angle = math.atan2(y - center[1], x - center[0]) + math.pi / 2
        angles.append(direction_angle)
    
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else angles[0]  # 圆形道路，第一个点连接最后一个点
        
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        
        # 计算梯形参数
        angle_diff = abs(angle_difference(current_angle, next_angle))
        top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': vertices,
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1
# 处理圆弧道路的采样
for arc_data in arcs_data:
    road_id = arc_data['road_id']
    center = arc_data['center']
    radius = arc_data['radius']
    start_angle = arc_data['start_angle']
    end_angle = arc_data['end_angle']
    # 转换为弧度
    start_angle_rad = math.radians(start_angle)
    end_angle_rad = math.radians(end_angle)
    
    # 生成采样点
    points, total_length = sample_arc_points(center, radius, start_angle_rad, end_angle_rad, SAMPLE_DISTANCE)
    
    
    # 计算所有点的方向角度
    angles = []
    for i, (x, y) in enumerate(points):
        # 计算切线方向（圆弧切线方向）
        # 从圆心到采样点的向量，切线方向垂直于此向量
        direction_angle = math.atan2(y - center[1], x - center[0]) + math.pi / 2
        angles.append(direction_angle)
    
    # 为每个采样点创建无缝连接的梯形
    prev_vertices = None
    for i, (x, y) in enumerate(points):
        current_angle = angles[i]
        next_angle = angles[i + 1] if i < len(points) - 1 else None
        
        # 计算切线方向和法线方向
        if i == 0 and next_angle is not None:
            # 第一个点：使用从第一个点到第二个点的方向
            tangent_angle = next_angle
            normal_angle = next_angle + math.pi / 2
        elif i == len(points) - 1 and next_angle is None:
            # 最后一个点：使用从倒数第二个点到最后一个点的方向
            if i > 0:
                # 计算从倒数第二个点到最后一个点的方向
                prev_x, prev_y = points[i - 1]
                direction_angle = math.atan2(y - prev_y, x - prev_x)
                tangent_angle = direction_angle
                normal_angle = direction_angle + math.pi / 2
            else:
                # 如果只有一个点，使用当前点的角度
                tangent_angle = current_angle
                normal_angle = current_angle + math.pi / 2
        else:
            # 其他点：使用当前点的角度
            tangent_angle = current_angle
            normal_angle = current_angle + math.pi / 2
        
        # 计算梯形参数
        if next_angle is not None:
            # 根据下一个点的方向计算上底宽度
            angle_diff = abs(angle_difference(current_angle, next_angle))
            top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)  # 动态调整上底宽度
        else:
            # 最后一个点：使用与前一个点相同的上底宽度，确保无缝连接
            if i > 0:
                # 计算与前一个点的角度差
                prev_angle = angles[i - 1]
                angle_diff = abs(angle_difference(prev_angle, current_angle))
                top_width = RECTANGLE_LENGTH * (1 + angle_diff / math.pi)
            else:
                top_width = RECTANGLE_LENGTH
        
        bottom_width = RECTANGLE_LENGTH
        height = RECTANGLE_WIDTH
        
        
        # 创建无缝连接的梯形
        # 最后一个点使用特殊的生成方式
        if i == len(points) - 1:
            # 最后一个点：使用曲线真实终点作为中心，确保与上一个矩形连接
            if prev_vertices is not None:
                # 使用前一个梯形的下底顶点作为当前梯形的上底顶点（无缝连接）
                prev_bottom_left = prev_vertices[3]  # 前一个梯形的下底左顶点
                prev_bottom_right = prev_vertices[2]  # 前一个梯形的下底右顶点
                
                # 计算当前梯形的下底顶点（使用更大的尺寸确保覆盖）
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用正常尺寸，确保与正常处理一样大
                extended_bottom_width = bottom_width  # 使用正常下底宽度
                extended_top_width = top_width        # 使用正常上底宽度
                
                # 计算当前梯形的下底顶点
                bottom_half = extended_bottom_width / 2
                height_half = height / 2
                
                # 下底左顶点（确保覆盖到道路终点，向前延伸）
                bottom_left_x = x + bottom_half * t_x - height_half * n_x
                bottom_left_y = y + bottom_half * t_y - height_half * n_y
                
                # 下底右顶点（确保覆盖到道路终点，向前延伸）
                bottom_right_x = x + bottom_half * t_x + height_half * n_x
                bottom_right_y = y + bottom_half * t_y + height_half * n_y
                
                # 四个顶点
                vertices = [
                    prev_bottom_left,           # 上底左顶点（与前一个梯形共享）
                    prev_bottom_right,          # 上底右顶点（与前一个梯形共享）
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)     # 下底左顶点
                ]
            else:
                # 如果没有前一个梯形，直接计算
                # 标准化角度
                tangent_angle = normalize_angle(tangent_angle)
                normal_angle = normalize_angle(normal_angle)
                
                # 计算切线方向单位向量
                t_x = math.cos(tangent_angle)
                t_y = math.sin(tangent_angle)
                
                n_x = math.cos(normal_angle)
                n_y = math.sin(normal_angle)
                
                # 使用更大的尺寸确保完全覆盖
                extended_bottom_width = bottom_width * 2.0
                extended_top_width = top_width * 2.0
                
                # 计算四个顶点
                bottom_half = extended_bottom_width / 2
                top_half = extended_top_width / 2
                height_half = height / 2
                
                # 下底两个顶点
                bottom_left_x = x - bottom_half * t_x - height_half * n_x
                bottom_left_y = y - bottom_half * t_y - height_half * n_y
                
                bottom_right_x = x - bottom_half * t_x + height_half * n_x
                bottom_right_y = y - bottom_half * t_y + height_half * n_y
                
                # 上底两个顶点
                top_left_x = x + top_half * t_x - height_half * n_x
                top_left_y = y + top_half * t_y - height_half * n_y
                
                top_right_x = x + top_half * t_x + height_half * n_x
                top_right_y = y + top_half * t_y + height_half * n_y
                
                # 四个顶点（按顺序排列）
                vertices = [
                    (top_left_x, top_left_y),      # 上底左顶点
                    (top_right_x, top_right_y),     # 上底右顶点
                    (bottom_right_x, bottom_right_y),  # 下底右顶点
                    (bottom_left_x, bottom_left_y)   # 下底左顶点
                ]
        else:
            # 其他点：使用无缝连接方式
            vertices = create_seamless_trapezoid(
                x, y, tangent_angle, normal_angle,
                bottom_width, top_width, height, prev_vertices
            )
        
        
        # 存储四边形数据
        polygons_data.append({
            'poly_id': poly_id_counter,
            'road_id': road_id,
            'center': (x, y),
            'vertices': vertices,
            'direction_angle': current_angle
        })
        
        # 保存当前梯形的顶点，供下一个梯形使用
        prev_vertices = vertices
        poly_id_counter += 1



# 可视化所有四边形
for poly_data in polygons_data:
    poly_id = poly_data['poly_id']
    road_id = poly_data['road_id']
    vertices = poly_data['vertices']
    
    # 统一使用黄色
    facecolor = 'yellow'
    edgecolor = 'orange'
    alpha = 0.6
    linewidth = 1
    
    # 创建多边形补丁
    polygon = Polygon(vertices, closed=True, 
                     facecolor=facecolor, edgecolor=edgecolor, 
                     alpha=alpha, linewidth=linewidth)
    ax.add_patch(polygon)
    
    # 可选：在四边形中心添加ID标签（如果四边形不太多的话）
    if len(polygons_data) < 100:  # 只在四边形数量较少时显示标签
        center_x, center_y = poly_data['center']
        ax.text(center_x, center_y, f'P{poly_id}', 
                fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))


# 设置图形属性
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('MAP - Trapezoid Visualization')
# 自动调整坐标轴范围
ax.autoscale()
# 显示图形
# plt.tight_layout()
plt.show()

# 可选：保存图形
# plt.savefig('dxf_visualization.png', dpi=300, bbox_inches='tight')
import math
import json
import os
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt


def visualize_map(lines_data, circles_data, arcs_data, polygons_data, oob_points, ax):
    """
    统一的可视化函数，将所有可视化逻辑集中管理
    参数:
    - lines_data: 直线数据列表
    - circles_data: 圆形数据列表
    - arcs_data: 圆弧数据列表
    - polygons_data: 四边形数据列表
    - oob_points: OOB点数据列表
    - ax: matplotlib轴对象
    """
    LANE_WIDTH = 0.5

    # 仅默认绘制四边形（quads）
    for poly_data in polygons_data:
        poly_id = poly_data['poly_id']
        vertices = poly_data['vertices']
        # 支持三维顶点：仅投影到 (x, y)
        vertices_2d = [(v[0], v[1]) for v in vertices]

        facecolor = 'yellow'
        edgecolor = 'black'
        alpha = 0.2
        linewidth = 0.2

        polygon = Polygon(vertices_2d, closed=True,
                          facecolor=facecolor, edgecolor=edgecolor,
                          alpha=alpha, linewidth=linewidth)
        ax.add_patch(polygon)

        # 可选：在四边形中心添加ID标签
        if len(polygons_data) < 100:
            center_x, center_y = poly_data['center']
            ax.text(center_x, center_y, f'P{poly_id}',
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.2))

    # 交互式复选框：按需绘制 line / circle / arc / oob
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MAP - Trapezoid Visualization')
    ax.autoscale()

    # 存放各层的绘图元素与是否已生成
    rendered = {"line": False, "circle": False, "arc": False, "oob": False}
    artists = {"line": [], "circle": [], "arc": [], "oob": []}

    def render_lines():
        if rendered["line"]:
            return
        for line_data in lines_data:
            start = line_data['start']
            end = line_data['end']
            start_x, start_y = round(start[0], 6), round(start[1], 6)
            end_x, end_y = round(end[0], 6), round(end[1], 6)
            line_obj, = ax.plot([start_x, end_x], [start_y, end_y],
                                color='blue', linewidth=LANE_WIDTH)
            artists["line"].append(line_obj)
        rendered["line"] = True

    def render_circles():
        if rendered["circle"]:
            return
        for circle_data in circles_data:
            center = circle_data['center']
            radius = circle_data['radius']
            center_x = round(center[0], 6)
            center_y = round(center[1], 6)
            circle = Circle((center_x, center_y), radius,
                            fill=False, edgecolor='pink', linewidth=LANE_WIDTH)
            ax.add_patch(circle)
            artists["circle"].append(circle)
        rendered["circle"] = True

    def render_arcs():
        if rendered["arc"]:
            return
        for arc_data in arcs_data:
            center = arc_data['center']
            radius = arc_data['radius']
            start_angle = arc_data['start_angle']
            end_angle = arc_data['end_angle']
            center_x = round(center[0], 6)
            center_y = round(center[1], 6)

            start_angle_rad = math.radians(start_angle)
            end_angle_rad = math.radians(end_angle)

            start_angle_deg = math.degrees(start_angle_rad) % 360
            end_angle_deg = math.degrees(end_angle_rad) % 360

            if start_angle_deg > end_angle_deg:
                angles1 = np.linspace(start_angle_rad, 2 * math.pi, 20)
                angles2 = np.linspace(0, end_angle_rad, 20)
                angles = np.concatenate([angles1, angles2])
            else:
                angle_diff = end_angle_rad - start_angle_rad
                num_points = max(20, int(abs(math.degrees(angle_diff))))
                angles = np.linspace(start_angle_rad, end_angle_rad, num_points)

            arc_x = np.round(center_x + radius * np.cos(angles), 6)
            arc_y = np.round(center_y + radius * np.sin(angles), 6)

            arc_line, = ax.plot(arc_x, arc_y, color='green', linewidth=LANE_WIDTH)
            artists["arc"].append(arc_line)
        rendered["arc"] = True

    def render_oob():
        if rendered["oob"]:
            return
        if len(oob_points) == 0:
            rendered["oob"] = True
            return
        xs = [p['x'] for p in oob_points]
        ys = [p['y'] for p in oob_points]
        scatter = ax.scatter(xs, ys, s=3, c='red', alpha=0.3)
        artists["oob"].append(scatter)
        rendered["oob"] = True

    # 复选框放置在右侧
    fig = ax.figure
    cb_ax = fig.add_axes([0.86, 0.6, 0.12, 0.2])  # [left, bottom, width, height]
    labels = ['line', 'circle', 'arc', 'oob']
    visibility = [False, False, False, False]
    check = CheckButtons(cb_ax, labels, visibility)
    cb_ax.set_title('Layers')

    def on_clicked(label):
        if label == 'line':
            if not rendered['line']:
                render_lines()
            else:
                for a in artists['line']:
                    a.set_visible(not a.get_visible())
        elif label == 'circle':
            if not rendered['circle']:
                render_circles()
            else:
                for a in artists['circle']:
                    a.set_visible(not a.get_visible())
        elif label == 'arc':
            if not rendered['arc']:
                render_arcs()
            else:
                for a in artists['arc']:
                    a.set_visible(not a.get_visible())
        elif label == 'oob':
            if not rendered['oob']:
                render_oob()
            else:
                for a in artists['oob']:
                    vis = a.get_visible()
                    a.set_visible(not vis)
        ax.figure.canvas.draw_idle()

    check.on_clicked(on_clicked)

    # 防止被垃圾回收：将引用挂到 figure 上
    fig._layer_checkbuttons = check
    fig._layer_artists = artists
    fig._layer_rendered_flags = rendered
    fig._layer_callback = on_clicked

    # 初次重绘
    fig.canvas.draw_idle()

def visualize_map_from_json(json_path: str):
    """从导出的地图 JSON 加载并可视化。"""
    if not os.path.isabs(json_path):
        json_path = os.path.abspath(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    # 解析 quads
    quads = payload.get('quads', [])
    polygons_data = []
    for q in quads:
        polygons_data.append({
            'poly_id': q.get('poly_Id'),
            'road_id': q.get('road_id'),
            'center': tuple(q.get('center', [0.0, 0.0, 0.0])[:2]),
            'vertices': [tuple(v) for v in q.get('vertices', [])]
        })

    # 解析 OOB 点
    oob_points = payload.get('oob_points', [])

    # 解析几何（可选）
    geometry = payload.get('geometry', {})
    lines = geometry.get('lines', [])
    circles = geometry.get('circles', [])
    arcs = geometry.get('arcs', [])

    # 转换成 visualize_map 期望的数据结构
    lines_data = [
        {
            'road_id': item.get('road_id'),
            'layer': item.get('layer', ''),
            'start': tuple(item.get('start', [0.0, 0.0, 0.0])),
            'end': tuple(item.get('end', [0.0, 0.0, 0.0])),
            'length': item.get('length', 0.0)
        }
        for item in lines
    ]

    circles_data = [
        {
            'road_id': item.get('road_id'),
            'center': tuple(item.get('center', [0.0, 0.0, 0.0])),
            'radius': item.get('radius', 0.0)
        }
        for item in circles
    ]

    arcs_data = [
        {
            'road_id': item.get('road_id'),
            'center': tuple(item.get('center', [0.0, 0.0, 0.0])),
            'radius': item.get('radius', 0.0),
            'start_angle': item.get('start_angle', 0.0),
            'end_angle': item.get('end_angle', 0.0)
        }
        for item in arcs
    ]

    # 创建画布并调用统一可视化
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    visualize_map(lines_data, circles_data, arcs_data, polygons_data, oob_points, ax)
    plt.show()


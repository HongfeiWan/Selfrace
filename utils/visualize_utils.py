import math
import json
import os
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt

def visualize_map(lines_data, circles_data, arcs_data, polygons_data, oob_points, ax, w_lanes=None):
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
    if w_lanes is None:
        w_lanes = []
    rendered = {"line": False, "circle": False, "arc": False, "oob": False, "direction": False, "road_color": False, "w_lane": False, "s_label": False, "curvature": False}
    artists = {"line": [], "circle": [], "arc": [], "oob": [], "direction": [], "road_color": [], "w_lane": [], "s_label": [], "curvature": []}

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

    def render_directions():
        if rendered["direction"]:
            return
        for poly_data in polygons_data:
            center = poly_data['center']
            direction_angle = poly_data.get('direction_angle', 0.0)
            
            # 计算箭头端点
            arrow_length = 0.1  # 箭头长度
            end_x = center[0] + arrow_length * math.cos(direction_angle)
            end_y = center[1] + arrow_length * math.sin(direction_angle)
            
            # 绘制方向箭头
            arrow = ax.annotate('', xy=(end_x, end_y), xytext=(center[0], center[1]),
                              arrowprops=dict(arrowstyle='->', color='red', lw=0.2, alpha=0.3))
            artists["direction"].append(arrow)
        rendered["direction"] = True

    def render_road_color():
        if rendered["road_color"]:
            return
        # 使用离散色图为不同road_id赋色
        cmap = plt.get_cmap('tab20')
        # 收集所有road_id并建立索引
        road_ids = sorted(list({p.get('road_id') for p in polygons_data}))
        rid_to_idx = {rid: i for i, rid in enumerate(road_ids)}
        n_colors = max(1, len(road_ids))
        for poly_data in polygons_data:
            vertices = poly_data['vertices']
            vertices_2d = [(v[0], v[1]) for v in vertices]
            rid = poly_data.get('road_id')
            idx = rid_to_idx.get(rid, 0)
            color = cmap(float(idx % 20) / 20.0)  # tab20循环使用
            polygon = Polygon(vertices_2d, closed=True,
                              facecolor=color, edgecolor='none',
                              alpha=0.35, linewidth=0.0)
            ax.add_patch(polygon)
            artists["road_color"].append(polygon)
        rendered["road_color"] = True

    def render_w_lanes():
        if rendered["w_lane"]:
            return
        if len(w_lanes) == 0:
            rendered["w_lane"] = True
            return
        xs = [float(item.get('center', (0.0, 0.0))[0]) for item in w_lanes]
        ys = [float(item.get('center', (0.0, 0.0))[1]) for item in w_lanes]
        scatter = ax.scatter(xs, ys, s=8, c='red', alpha=0.9)
        artists["w_lane"].append(scatter)
        # 为每个w_lane绘制一个小方向箭头
        arrow_len = 0.15
        for item in w_lanes:
            cx, cy = float(item.get('center', (0.0, 0.0))[0]), float(item.get('center', (0.0, 0.0))[1])
            ang = float(item.get('direction_angle', 0.0))
            ex = cx + arrow_len * math.cos(ang)
            ey = cy + arrow_len * math.sin(ang)
            arr = ax.annotate('', xy=(ex, ey), xytext=(cx, cy),
                              arrowprops=dict(arrowstyle='->', color='red', lw=0.4, alpha=0.9))
            artists["w_lane"].append(arr)
        rendered["w_lane"] = True

    def render_s_labels():
        if rendered["s_label"]:
            return
        if len(polygons_data) == 0:
            rendered["s_label"] = True
            return
        # 先按 (road_id, lane_id) 分组
        road_lane_to_quads = {}
        for q in polygons_data:
            rid = q.get('road_id')
            lid = q.get('lane_id', 1)
            road_lane_to_quads.setdefault((rid, lid), []).append(q)

        rid_to_lanes = {}
        for (rid, lid), quads in road_lane_to_quads.items():
            rid_to_lanes.setdefault(rid, []).append((lid, quads))

        for rid, lanes in rid_to_lanes.items():
            if not lanes:
                continue
            # 选择 lane_id == 1 的那条车道，若不存在则选择 lane_id 最小者
            lanes.sort(key=lambda x: x[0])
            selected = None
            for lid_i, qs in lanes:
                if lid_i == 1:
                    selected = (lid_i, qs)
                    break
            if selected is None:
                selected = lanes[0]
            lid, quads = selected
            # 要求该lane具备 s 字段
            if not any('s' in q for q in quads):
                continue
            # 按 poly_id 排序，稳定采样
            quads.sort(key=lambda it: it.get('poly_id', 0) if isinstance(it.get('poly_id', 0), (int, float)) else 0)
            n = len(quads)
            if n == 0:
                continue
            # 只在中间段取5个（不含首末两端）。若中段不足5个，尽量均匀取。
            middle_count = max(0, n - 2)
            if middle_count <= 0:
                continue
            k = min(5, middle_count)
            if k == 1:
                idxs = [1]
            else:
                idxs = [1 + int(round(i*(middle_count-1)/(k-1))) for i in range(k)]
            for n_i, i in enumerate(idxs):
                q = quads[i]
                cx, cy = q['center']
                sval = float(q.get('s', 0.0))
                # 1) 在中心画一个小点
                dot = ax.scatter([cx], [cy], s=10, c='blue', alpha=0.9, zorder=3)
                artists["s_label"].append(dot)
                # 2) 文本框偏移位置 + 连接线
                # 交替选择偏移方向，避免重叠
                sign = 1 if (n_i % 2 == 0) else -1
                dx = 0.8 * sign
                dy = 0.6
                ann = ax.annotate(
                    f"s={sval:.1f}",
                    xy=(cx, cy),
                    xytext=(cx + dx, cy + dy),
                    textcoords='data',
                    fontsize=6,
                    ha='left' if sign > 0 else 'right',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='blue', linewidth=0.4),
                    arrowprops=dict(arrowstyle='-', color='blue', lw=0.6)
                )
                artists["s_label"].append(ann)
        rendered["s_label"] = True

    def render_curvature():
        if rendered["curvature"]:
            return
        if len(polygons_data) == 0:
            rendered["curvature"] = True
            return
        # 收集曲率绝对值的最大值用于归一化
        curvs = [abs(float(p.get('curvature', 0.0))) for p in polygons_data]
        kmax = max(curvs) if len(curvs) > 0 else 0.0
        if kmax <= 0.0:
            rendered["curvature"] = True
            return
        for p in polygons_data:
            k = float(p.get('curvature', 0.0))
            if k == 0.0:
                continue  # 保持原色
            inten = min(1.0, abs(k) / kmax)
            if k > 0:
                color = (0.0, inten, 0.0, 0.45)  # 绿色，强度随曲率
            else:
                color = (0.0, 0.0, inten, 0.45)  # 蓝色
            vertices_2d = [(v[0], v[1]) for v in p['vertices']]
            poly = Polygon(vertices_2d, closed=True, facecolor=color, edgecolor='none', alpha=color[3], linewidth=0.0)
            ax.add_patch(poly)
            artists["curvature"].append(poly)
        rendered["curvature"] = True

    # 复选框放置在右侧
    fig = ax.figure
    cb_ax = fig.add_axes([0.86, 0.6, 0.12, 0.2])  # [left, bottom, width, height]
    labels = ['line', 'circle', 'arc', 'oob', 'direction', 'road_color', 'w_lane', 's_label', 'curvature']
    visibility = [False, False, False, False, False, False, False, False, False]
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
        elif label == 'direction':
            if not rendered['direction']:
                render_directions()
            else:
                for a in artists['direction']:
                    vis = a.get_visible()
                    a.set_visible(not vis)
        elif label == 'road_color':
            if not rendered['road_color']:
                render_road_color()
            else:
                for a in artists['road_color']:
                    vis = a.get_visible()
                    a.set_visible(not vis)
        elif label == 'w_lane':
            if not rendered['w_lane']:
                render_w_lanes()
            else:
                for a in artists['w_lane']:
                    vis = a.get_visible()
                    a.set_visible(not vis)
        elif label == 's_label':
            if not rendered['s_label']:
                render_s_labels()
            else:
                for a in artists['s_label']:
                    a.set_visible(not a.get_visible())
        elif label == 'curvature':
            if not rendered['curvature']:
                render_curvature()
            else:
                for a in artists['curvature']:
                    a.set_visible(not a.get_visible())
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
            'poly_id': q.get('poly_id', q.get('poly_Id')),
            'road_id': q.get('road_id'),
            'lane_id': q.get('lane_id', 1),
            'center': tuple(q.get('center', [0.0, 0.0, 0.0])[:2]),
            'vertices': [tuple(v) for v in q.get('vertices', [])],
            'direction_angle': q.get('direction_angle', 0.0),
            's': q.get('s', 0.0),
            'curvature': q.get('curvature', 0.0)
        })

    # 解析 OOB 点
    oob_points = payload.get('oob_points', [])

    # 解析 W_lane
    w_lanes_raw = payload.get('w_lanes', [])
    w_lanes = []
    for item in w_lanes_raw:
        w_lanes.append({
            'w_lane_id': item.get('w_lane_id'),
            'road_id': item.get('road_id'),
            'lane_id': item.get('lane_id'),
            'center': tuple(item.get('center', [0.0, 0.0, 0.0])[:2]),
            'direction_angle': item.get('direction_angle', 0.0),
            'width': item.get('width', 0.0),
            'poly_id': item.get('poly_id')
        })

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
    visualize_map(lines_data, circles_data, arcs_data, polygons_data, oob_points, ax, w_lanes=w_lanes)
    plt.show()

if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(__file__), "..", "maps", "town2.json")
    visualize_map_from_json(json_path)
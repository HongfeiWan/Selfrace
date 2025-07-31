import json
import os
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def build_waypoint_graph(cross_data):
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

def visualize_waypoint_graph(G, highlight_path_ids=None):
    plt.figure(figsize=(16, 12))
    pos = {node: (node[3], -node[4]) for node in G.nodes}
    # 默认节点和边
    nx.draw(G, pos, with_labels=False, node_size=30, edge_color='gray', arrowsize=10)
    for node, (x, y) in pos.items():
        cross_id, road_id, lane_id = node[0], node[1], node[2]
        plt.text(x, y, f'{cross_id}\n{road_id},{lane_id}', fontsize=7, ha='right', va='bottom')
    # 边distance标签
    edge_labels = {(u, v): f"{G[u][v]['distance']:.2f}" for u, v in G.edges}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.5,
        font_color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    # 高亮路径
    if highlight_path_ids is not None and len(highlight_path_ids) > 1:
        # 找到路径上的所有节点
        highlight_nodes = [n for n in G.nodes if list(n[:3]) in highlight_path_ids]
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='red', node_size=80)
        # 找到路径上的所有边
        highlight_edges = []
        for i in range(len(highlight_path_ids) - 1):
            src_ids = highlight_path_ids[i]
            dst_ids = highlight_path_ids[i+1]
            # 找到所有对应的节点对
            src_nodes = [n for n in G.nodes if list(n[:3]) == src_ids]
            dst_nodes = [n for n in G.nodes if list(n[:3]) == dst_ids]
            for s in src_nodes:
                for d in dst_nodes:
                    if G.has_edge(s, d):
                        highlight_edges.append((s, d))
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='red', width=2, arrowsize=15)
    plt.title("Waypoints 级别连接关系可视化（红色为最短路径）")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def find_shortest_path(G, start_id, end_id):
    """
    输入:
        G: networkx.DiGraph
        start_id: [cross_id, road_id, lane_id]
        end_id: [cross_id, road_id, lane_id]
    返回:
        路径上所有[cross_id, road_id, lane_id]的列表（不含重复）
    """
    # 找到所有起点和终点的节点（type可以是start或end）
    start_nodes = [n for n in G.nodes if list(n[:3]) == start_id]
    end_nodes = [n for n in G.nodes if list(n[:3]) == end_id]
    if not start_nodes or not end_nodes:
        print("起点或终点不存在")
        return []
    # 搜索所有组合，找最短路径
    min_path = None
    min_length = float('inf')
    for s in start_nodes:
        for e in end_nodes:
            try:
                path = nx.shortest_path(G, source=s, target=e, weight='distance')
                length = nx.shortest_path_length(G, source=s, target=e, weight='distance')
                if length < min_length:
                    min_length = length
                    min_path = path
            except nx.NetworkXNoPath:
                continue
    if min_path is None:
        print("没有可达路径")
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

def main():
    # 修改为读取同目录下的processed_map_Town01_stitched.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_map_path = os.path.join(current_dir, 'processed_map_Town01_stitched.json')
    
    if not os.path.exists(processed_map_path):
        print(f"错误: processed_map_Town01_stitched.json 文件不存在: {processed_map_path}")
        return
    
    # 读取processed_map文件
    with open(processed_map_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    # 从processed_data中提取cross_data
    cross_data = processed_data.get('cross_data', {})
    if not cross_data:
        print("错误: processed_map文件中没有找到cross_data字段")
        return
    
    print(f"加载processed_map文件: {processed_map_path}")
    print("正在构建waypoint级别的图...")
    G, cross_waypoint_records = build_waypoint_graph(cross_data)
    print(f"节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")
    
    # 将waypoint图结构写入processed_map文件
    def node_no_s(node):
        return [node[0], node[1], node[2], node[3], node[4], node[6]]
    
    # 如果已存在则先删除
    if 'waypoint_graph' in cross_data:
        del cross_data['waypoint_graph']
    
    waypoint_graph = {
        "nodes": [node_no_s(node) for node in G.nodes],
        "edges": [
            [node_no_s(u), node_no_s(v), G[u][v]['distance']]
            for u, v in G.edges
        ]
    }
    cross_data['waypoint_graph'] = waypoint_graph
    
    # 更新processed_data中的cross_data
    processed_data['cross_data'] = cross_data
    
    # 保存更新后的文件
    with open(processed_map_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("waypoint_graph 已写入 processed_map_Town01_stitched.json")

    # 测试 find_shortest_path
    print("\n测试 find_shortest_path ...")
    # 随机选取两个不同的节点[cross_id, road_id, lane_id]作为起点和终点
    node_list = list(G.nodes)
    if len(node_list) >= 2:
        start_node = node_list[0]
        end_node = node_list[-1]
        start_id = list(start_node[:3])
        end_id = list(end_node[:3])
        print(f"起点: {start_id}, 终点: {end_id}")
        path_result = find_shortest_path(G, start_id, end_id)
        print("最短路径经过的[cross_id, road_id, lane_id]:")
        print(path_result)
        # 可视化高亮最短路径
        visualize_waypoint_graph(G, highlight_path_ids=path_result)
    else:
        print("节点数不足，无法测试最短路径功能。")

if __name__ == "__main__":
    main()
    
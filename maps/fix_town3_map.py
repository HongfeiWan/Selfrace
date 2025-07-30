import json
import numpy as np
import os
import yaml
from scipy.spatial import KDTree

def calculate_quad_direction(quad):
    """
    计算quad的方向向量（从后到前）
    """
    verts_3d = np.array([[v['x'], v['y'], v['z']] for v in quad['vertices']])
    verts_2d = verts_3d[:, :2]
    front_center = (verts_2d[0] + verts_2d[1]) / 2.0  # 前中心
    back_center = (verts_2d[2] + verts_2d[3]) / 2.0   # 后中心
    direction = front_center - back_center
    return direction

def find_nearest_quad_with_direction(quad_id, quad_centers_3d, quad_directions, kdtree, poly_ids_sorted):
    """
    找到距离指定quad最近的、有非零方向向量的quad
    """
    if quad_id not in quad_centers_3d:
        return None, None
    
    center = quad_centers_3d[quad_id][:2]
    
    # 查找最近的几个quads
    distances, indices = kdtree.query(center, k=min(10, len(poly_ids_sorted)))
    
    for i, idx in enumerate(indices):
        neighbor_id = poly_ids_sorted[idx]
        if neighbor_id == quad_id:
            continue
            
        if neighbor_id in quad_directions:
            direction = quad_directions[neighbor_id]
            if np.linalg.norm(direction[:2]) > 1e-6:  # 方向向量非零
                return neighbor_id, direction[:2]
    
    return None, None

def fix_short_direction_quad(quad, reference_direction, target_length=0.5):
    """
    修复方向向量过短的quad，通过调整vertices的坐标
    Args:
        quad: 要修复的quad数据
        reference_direction: 参考方向向量（从最近的有方向的quad获得）
        target_length: 目标方向向量长度，默认为0.5m
    """
    # 归一化参考方向向量
    ref_dir_norm = np.linalg.norm(reference_direction)
    if ref_dir_norm < 1e-6:
        return False
    
    normalized_ref_dir = reference_direction / ref_dir_norm
    
    # 计算需要延长的距离
    # 目标是将方向向量长度延长到target_length
    current_direction = calculate_quad_direction(quad)
    current_length = np.linalg.norm(current_direction[:2])
    
    if current_length >= target_length:
        return True  # 已经达到目标长度，无需修复
    
    # 计算需要延长的距离
    extension_distance = (target_length - current_length) / 2.0  # 前后各延长一半
    
    # 调整vertices的坐标
    # 第1、2个顶点（前部）沿着方向增加extension_distance
    # 第3、4个顶点（后部）沿着逆方向减少extension_distance
    
    for i, vertex in enumerate(quad['vertices']):
        if i < 2:  # 前部顶点 (0, 1)
            # 沿着方向增加extension_distance
            vertex['x'] += normalized_ref_dir[0] * extension_distance
            vertex['y'] += normalized_ref_dir[1] * extension_distance
        else:  # 后部顶点 (2, 3)
            # 沿着逆方向减少extension_distance
            vertex['x'] -= normalized_ref_dir[0] * extension_distance
            vertex['y'] -= normalized_ref_dir[1] * extension_distance
    
    return True

def fix_zero_direction_quad(quad, reference_direction):
    """
    修复方向向量为0的quad，通过调整vertices的坐标
    Args:
        quad: 要修复的quad数据
        reference_direction: 参考方向向量（从最近的有方向的quad获得）
    """
    # 归一化参考方向向量
    ref_dir_norm = np.linalg.norm(reference_direction)
    if ref_dir_norm < 1e-6:
        return False
    
    normalized_ref_dir = reference_direction / ref_dir_norm
    
    # 调整vertices的坐标
    # 第1、2个顶点（前部）沿着方向增加0.5m
    # 第3、4个顶点（后部）沿着逆方向减少0.5m
    
    for i, vertex in enumerate(quad['vertices']):
        if i < 2:  # 前部顶点 (0, 1)
            # 沿着方向增加0.5m
            vertex['x'] += normalized_ref_dir[0] * 0.5
            vertex['y'] += normalized_ref_dir[1] * 0.5
        else:  # 后部顶点 (2, 3)
            # 沿着逆方向减少0.5m
            vertex['x'] -= normalized_ref_dir[0] * 0.5
            vertex['y'] -= normalized_ref_dir[1] * 0.5
    
    return True

def fix_oob_points_x(data):
    """
    修复oob_points的x坐标
    """
    print("=== 修复OOB Points的X坐标 ===")
    
    oob_points = data["oob_points"]
    
    # 4621和4626的x改为4625的x
    x_4625 = oob_points[4625]["x"]
    oob_points[4621]["x"] = x_4625
    oob_points[4626]["x"] = x_4625
    oob_points[12417]["x"] = x_4625
    
    # 4623和4637的x改为4638的x
    x_4638 = oob_points[4638]["x"]
    oob_points[4623]["x"] = x_4638
    oob_points[4637]["x"] = x_4638
    
    print("✅ OOB Points的X坐标修复完成")

def fix_quad_vertices(data):
    """
    修复特定quad的顶点坐标
    """
    print("=== 修复Quad顶点坐标 ===")
    
    quads = data["quads"]
    
    # 找到指定的quads
    quad13942 = None
    quad5273 = None
    quad5272 = None
    quad5284 = None
    
    for quad in quads:
        if quad["polyId"] == 13942:
            quad13942 = quad
        elif quad["polyId"] == 5273:
            quad5273 = quad
        elif quad["polyId"] == 5272:
            quad5272 = quad
        elif quad["polyId"] == 5284:
            quad5284 = quad
    
    if not all([quad13942, quad5273, quad5272, quad5284]):
        print("❌ 错误：未找到所有指定的quads")
        return False
    
    print("找到所有指定的quads:")
    print(f"quad13942: {quad13942['polyId']}")
    print(f"quad5273: {quad5273['polyId']}")
    print(f"quad5272: {quad5272['polyId']}")
    print(f"quad5284: {quad5284['polyId']}")
    
    # 修改quad13942的vertices
    # 将quad13942的vertices中的第1个坐标等于quad5273的vertices中的第4个坐标
    quad13942["vertices"][0]["x"] = quad5273["vertices"][3]["x"]
    quad13942["vertices"][0]["y"] = quad5273["vertices"][3]["y"]
    
    # 将quad13942的vertices中的第2个坐标等于quad5273的vertices中的第3个坐标
    quad13942["vertices"][1]["x"] = quad5273["vertices"][2]["x"]
    quad13942["vertices"][1]["y"] = quad5273["vertices"][2]["y"]
    
    # 修改quad5272的vertices
    # 将quad5272的vertices中的第1个坐标等于quad5284的vertices中的第4个坐标
    quad5272["vertices"][0]["x"] = quad5284["vertices"][3]["x"]
    quad5272["vertices"][0]["y"] = quad5284["vertices"][3]["y"]
    
    # 将quad5272的vertices中的第2个坐标等于quad5284的vertices中的第3个坐标
    quad5272["vertices"][1]["x"] = quad5284["vertices"][2]["x"]
    quad5272["vertices"][1]["y"] = quad5284["vertices"][2]["y"]
    
    print("✅ Quad顶点坐标修复完成")
    print(f"quad13942 vertices[0] 现在等于 quad5273 vertices[3]: ({quad13942['vertices'][0]['x']}, {quad13942['vertices'][0]['y']})")
    print(f"quad13942 vertices[1] 现在等于 quad5273 vertices[2]: ({quad13942['vertices'][1]['x']}, {quad13942['vertices'][1]['y']})")
    print(f"quad5272 vertices[0] 现在等于 quad5284 vertices[3]: ({quad5272['vertices'][0]['x']}, {quad5272['vertices'][0]['y']})")
    print(f"quad5272 vertices[1] 现在等于 quad5284 vertices[2]: ({quad5272['vertices'][1]['x']}, {quad5272['vertices'][1]['y']})")
    
    return True

def fix_zero_direction_quads(data, min_length=0.5):
    """
    修复地图文件中所有方向向量为0或长度小于指定值的quads
    Args:
        data: 地图数据
        min_length: 最小方向向量长度，默认为0.5m
    """
    print("=== 修复方向向量异常的Quads ===")
    
    quads_data = data.get('quads', [])
    print(f"总共有 {len(quads_data)} 个quads")
    
    # 翻转Y轴（如果需要）
    for q in quads_data:
        for v in q['vertices']:
            v['y'] = -v['y']
    
    # 计算所有quads的中心点和方向向量
    quad_centers_3d = {}
    quad_directions = {}
    
    for q in quads_data:
        poly_id = q['polyId']
        verts_3d = np.array([[v['x'], v['y'], v['z']] for v in q['vertices']])
        quad_centers_3d[poly_id] = np.mean(verts_3d, axis=0)
        quad_directions[poly_id] = calculate_quad_direction(q)
    
    # 构建KDTree用于快速最近邻搜索
    poly_ids_sorted = sorted(quad_centers_3d.keys())
    quad_centers_array = np.array([quad_centers_3d[pid][:2] for pid in poly_ids_sorted])
    centers_kdtree = KDTree(quad_centers_array)
    
    # 找到所有需要修复的quads（方向向量为0或长度小于min_length）
    zero_direction_quads = []
    short_direction_quads = []
    
    for q in quads_data:
        poly_id = q['polyId']
        direction = quad_directions[poly_id]
        direction_length = np.linalg.norm(direction[:2])
        
        if direction_length < 1e-6:
            zero_direction_quads.append(q)
        elif direction_length < min_length:
            short_direction_quads.append(q)
    
    print(f"发现 {len(zero_direction_quads)} 个方向向量为0的quads")
    print(f"发现 {len(short_direction_quads)} 个方向向量长度小于{min_length}m的quads")
    
    total_quads_to_fix = len(zero_direction_quads) + len(short_direction_quads)
    if total_quads_to_fix == 0:
        print("✅ 没有发现需要修复的quads")
        return True
    
    # 修复每个方向向量为0的quad
    fixed_zero_count = 0
    for quad in zero_direction_quads:
        poly_id = quad['polyId']
        print(f"修复方向向量为0的quad {poly_id}...")
        
        # 找到最近的、有非零方向向量的quad
        nearest_id, reference_direction = find_nearest_quad_with_direction(
            poly_id, quad_centers_3d, quad_directions, centers_kdtree, poly_ids_sorted
        )
        
        if nearest_id is None:
            print(f"  ❌ 无法为quad {poly_id} 找到参考方向")
            continue
        
        print(f"  使用quad {nearest_id} 的方向作为参考: {reference_direction}")
        
        # 修复quad
        if fix_zero_direction_quad(quad, reference_direction):
            fixed_zero_count += 1
            print(f"  ✅ quad {poly_id} 修复成功")
        else:
            print(f"  ❌ quad {poly_id} 修复失败")
    
    # 修复每个方向向量过短的quad
    fixed_short_count = 0
    for quad in short_direction_quads:
        poly_id = quad['polyId']
        current_direction = quad_directions[poly_id]
        current_length = np.linalg.norm(current_direction[:2])
        print(f"修复方向向量过短的quad {poly_id} (当前长度: {current_length:.3f}m)...")
        
        # 找到最近的、有非零方向向量的quad
        nearest_id, reference_direction = find_nearest_quad_with_direction(
            poly_id, quad_centers_3d, quad_directions, centers_kdtree, poly_ids_sorted
        )
        
        if nearest_id is None:
            print(f"  ❌ 无法为quad {poly_id} 找到参考方向")
            continue
        
        print(f"  使用quad {nearest_id} 的方向作为参考: {reference_direction}")
        
        # 修复quad
        if fix_short_direction_quad(quad, reference_direction, min_length):
            fixed_short_count += 1
            print(f"  ✅ quad {poly_id} 修复成功")
        else:
            print(f"  ❌ quad {poly_id} 修复失败")
    
    total_fixed = fixed_zero_count + fixed_short_count
    print(f"\n修复完成: {total_fixed}/{total_quads_to_fix} 个quads被修复")
    print(f"  - 方向向量为0的quads: {fixed_zero_count}/{len(zero_direction_quads)}")
    print(f"  - 方向向量过短的quads: {fixed_short_count}/{len(short_direction_quads)}")
    
    # 验证修复结果
    print("\n=== 验证修复结果 ===")
    zero_direction_count_after = 0
    short_direction_count_after = 0
    
    for q in quads_data:
        direction = calculate_quad_direction(q)
        direction_length = np.linalg.norm(direction[:2])
        
        if direction_length < 1e-6:
            zero_direction_count_after += 1
        elif direction_length < min_length:
            short_direction_count_after += 1
    
    print(f"修复后仍有 {zero_direction_count_after} 个方向向量为0的quads")
    print(f"修复后仍有 {short_direction_count_after} 个方向向量长度小于{min_length}m的quads")
    
    # 翻转Y轴回原来的方向
    for q in quads_data:
        for v in q['vertices']:
            v['y'] = -v['y']
    
    return True

def fix_town3_map(map_data_path, output_path=None, min_length=0.5, compact_json=True):
    """
    修复Town3地图的所有问题
    Args:
        map_data_path: 输入地图文件路径
        output_path: 输出地图文件路径，如果为None则覆盖原文件
        min_length: 最小方向向量长度，默认为0.5m
        compact_json: 是否使用紧凑的JSON格式，默认为True
    """
    print("🚀 开始修复Town3地图...")
    
    # 加载地图数据
    with open(map_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 执行所有修复操作
    fix_oob_points_x(data)
    fix_quad_vertices(data)
    fix_zero_direction_quads(data, min_length)
    
    # 保存修复后的地图文件
    if output_path is None:
        output_path = map_data_path
    
    # 根据compact_json参数决定JSON格式
    if compact_json:
        # 使用紧凑格式，不添加缩进，减少文件大小
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
    else:
        # 使用格式化输出，便于阅读
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 所有修复完成！修复后的地图文件已保存到: {output_path}")
    if compact_json:
        print("📝 使用紧凑JSON格式，文件大小已优化")

def main():
    """主函数"""
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml')
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    map_path = config.get('simulator', {}).get('map_path')
    if not map_path:
        print("错误: 配置文件中未找到simulator.map_path字段")
        return
    
    # 构建完整路径
    if os.path.isabs(map_path):
        map_path_full = map_path
    else:
        # 相对于项目根目录的路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        map_path_full = os.path.normpath(os.path.join(project_root, map_path.lstrip('./')))
    
    if not os.path.exists(map_path_full):
        print(f"错误: 地图文件不存在: {map_path_full}")
        return
    
    # 执行修复，使用紧凑JSON格式
    fix_town3_map(map_path_full, min_length=0.5, compact_json=True)

if __name__ == "__main__":
    main() 
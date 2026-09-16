"""Manual diagnostics moved from simulator/reward.py."""

from _bootstrap import add_project_paths

add_project_paths()

import numpy as np
import torch

from reward import RewardCalculator

def test_goal_reward():
    print("=== 测试目标到达奖励 ===")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建测试配置
    test_config = {
        'reward': {
            'v_goal': 3.0,
            'goal_reward': 1.0
        }
    }
    # 初始化奖励计算器
    reward_calculator = RewardCalculator(test_config, device)
    # 定义8种情况的测试数据
    # 条件组合: [距离<阈值, 路点到达, 速度<阈值] -> 预期奖励
    # 情况0: [False, False, False] -> 0
    # 情况1: [False, False, True]  -> 0  
    # 情况2: [False, True, False]  -> 0
    # 情况3: [False, True, True]   -> 0
    # 情况4: [True, False, False]  -> 0
    # 情况5: [True, False, True]   -> 1 (距离<阈值 且 速度<阈值)
    # 情况6: [True, True, False]   -> 1 (距离<阈值 且 路点到达)
    # 情况7: [True, True, True]    -> 1 (距离<阈值 且 (路点到达 或 速度<阈值))
    # 目标位置（所有智能体共享同一个目标）
    case_idx = 0
    goal_positions = torch.tensor([[[100.0, 200.0]]], device=device)
    for distance_condition in [False, True]:  # 距离条件
        for waypoint_condition in [False, True]:  # 路点条件
            for speed_condition in [False, True]:  # 速度条件
                # 根据条件设置测试数据
                if distance_condition:
                    # 距离 < delta_goal (到达目标区域)
                    agent_pos = torch.tensor([[[100.0, 200.0]]], device=device)  # (1, 1, 2)
                else:
                    # 距离 >= delta_goal (没到达目标区域)
                    agent_pos = torch.tensor([[[120.0, 220.0]]], device=device)  # (1, 1, 2)
                if speed_condition:
                    # 速度 < v_goal
                    speed = torch.tensor([[2.0]], device=device)
                else:
                    # 速度 >= v_goal
                    speed = torch.tensor([[5.0]], device=device)

                if waypoint_condition:
                    # 路点已到达
                    waypoint = torch.tensor([[True]], device=device)
                else:
                    # 路点未到达
                    waypoint = torch.tensor([[False]], device=device)
                # 计算奖励
                goal_reward = reward_calculator.calculate_goal_reward(
                    agent_pos, goal_positions, speed, waypoint
                )

                # 计算距离用于显示
                distance = torch.norm(agent_pos - goal_positions, dim=-1)

                # 打印结果
                print(f"\n情况{case_idx}:")
                print(f"  距离条件: {distance_condition} (距离: {distance[0, 0].item():.2f} {'<' if distance_condition else '>='} {reward_calculator.delta_goal})")
                print(f"  路点条件: {waypoint_condition} (路点到达: {waypoint[0, 0].item()})")
                print(f"  速度条件: {speed_condition} (速度: {speed[0, 0].item()} {'<' if speed_condition else '>='} {reward_calculator.v_goal})")
                print(f"  综合条件: {distance_condition} AND ({waypoint_condition} OR {speed_condition})")
                print(f"  奖励分数: {goal_reward[0, 0].item()}")
                case_idx += 1

def test_collision_penalty():
    print("=== 测试碰撞惩罚 ===")
    import matplotlib.pyplot as plt
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建测试配置
    test_config = {
        'reward': {
            'collision_speed_mult': 0.1
        }
    }
    # 收集多次初始化的collision_alpha值
    num_iterations = 1000
    collision_alphas = []
    # 生成速度范围
    speeds_range = np.linspace(0, 20, 21)  # 0, 1, 2, ..., 20 m/s
    # 存储每次初始化的结果
    collision_alphas = []
    all_penalties = []  # 存储所有速度对应的惩罚
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        collision_alpha_value = reward_calculator.collision_alpha.cpu().item()
        collision_alphas.append(collision_alpha_value)
        # 计算当前alpha下所有速度对应的惩罚
        penalties_for_current_alpha = []
        for speed in speeds_range:
            # 创建单个速度的测试数据
            speed_tensor = torch.tensor([[speed]], device=device)
            collision_tensor = torch.tensor([[True]], device=device)
            agents_state = torch.zeros(1, 1, 7, device=device)
            agents_state[0, 0, 3] = speed_tensor
            # 计算惩罚
            penalty = reward_calculator.calculate_collision_penalty(
                agents_state, collision_tensor
            )
            penalties_for_current_alpha.append(penalty[0, 0].cpu().item())
        all_penalties.append(penalties_for_current_alpha)
        if i % 10 == 0:
            print(f"初始化 {i+1}/{num_iterations}, collision_alpha: {collision_alpha_value:.6f}")
    # 转换为numpy数组
    collision_alphas = np.array(collision_alphas)
    all_penalties = np.array(all_penalties)  # shape: (num_iterations, num_speeds)
    # 绘制结果
    plt.figure(figsize=(15, 5))
    # 第一个子图：collision_alpha的分布
    plt.subplot(1, 3, 1)
    plt.hist(collision_alphas, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Collision Alpha")
    plt.ylabel("Frequency")
    plt.title("Collision Alpha Distribution")
    plt.grid(True, alpha=0.3)
    # 第二个子图：不同alpha下速度vs惩罚的关系
    plt.subplot(1, 3, 2)
    for i in range(0, num_iterations, 10):  # 每10次显示一条线
        plt.plot(speeds_range, all_penalties[i], alpha=0.3, linewidth=1)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Collision Penalty")
    plt.title("Speed vs Penalty (Multiple α)")
    plt.grid(True, alpha=0.3)
    # 第三个子图：惩罚的统计分布
    plt.subplot(1, 3, 3)
    mean_penalties = np.mean(all_penalties, axis=0)
    std_penalties = np.std(all_penalties, axis=0)
    plt.errorbar(speeds_range, mean_penalties, yerr=std_penalties, capsize=3)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Mean Collision Penalty")
    plt.title("Mean Penalty vs Speed")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def test_offroad_penalty():
    print("=== 测试离路惩罚 ===")
    import matplotlib.pyplot as plt
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建测试配置
    test_config = {
        'reward': {
            'boundary_alpha': 1.0
        }
    }

    # 测试参数
    num_iterations = 1000
    boundary_alphas = []
    all_penalties = []

    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        boundary_alpha_value = reward_calculator.boundary_alpha.cpu().item()
        boundary_alphas.append(boundary_alpha_value)

        # 创建离路掩码（所有车辆都离路）
        offroad_mask = torch.ones(1, 100, dtype=torch.bool, device=device)  # 100个车辆都离路

        # 计算离路惩罚
        offroad_penalties = reward_calculator.calculate_offroad_penalty(offroad_mask)
        penalties = offroad_penalties.cpu().numpy().flatten()
        all_penalties.append(penalties)

        if i % 100 == 0:
            print(f"初始化 {i+1}/{num_iterations}, boundary_alpha: {boundary_alpha_value:.6f}")

    # 转换为numpy数组
    boundary_alphas = np.array(boundary_alphas)
    all_penalties = np.array(all_penalties)  # shape: (num_iterations, num_vehicles)

    # 绘制结果
    plt.figure(figsize=(15, 5))

    # 第一个子图：boundary_alpha的分布
    plt.subplot(1, 3, 1)
    plt.hist(boundary_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Boundary Alpha")
    plt.ylabel("Frequency")
    plt.title("Boundary Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：惩罚的分布
    plt.subplot(1, 3, 2)
    penalties_flat = all_penalties.flatten()
    plt.hist(penalties_flat, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Offroad Penalty")
    plt.ylabel("Frequency")
    plt.title("Offroad Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：alpha vs 惩罚的关系
    plt.subplot(1, 3, 3)
    mean_penalties_per_alpha = np.mean(all_penalties, axis=1)
    plt.scatter(boundary_alphas, mean_penalties_per_alpha, alpha=0.6, s=10)
    plt.xlabel("Boundary Alpha")
    plt.ylabel("Mean Offroad Penalty")
    plt.title("Alpha vs Mean Penalty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def test_comfort_penalty():
    print("=== 测试舒适度惩罚 ===")
    import matplotlib.pyplot as plt
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建测试配置
    test_config = {
        'reward': {
            'comfort_alpha_min': 0.0,
            'comfort_alpha_max': 0.1
        }
    }
    # 定义16种情况的测试数据
    # 条件组合: [along违规, alat违规, along_jerk违规, alat_jerk违规] -> 预期惩罚
    # 情况0: [False, False, False, False] -> 0
    # 情况1: [True, False, False, False] -> -0.1 (1次违规)
    # 情况2: [False, True, False, False] -> -0.1 (1次违规)
    # 情况3: [True, True, False, False] -> -0.2 (2次违规)
    # 情况4: [False, False, True, False] -> -0.1 (1次违规)
    # 情况5: [True, False, True, False] -> -0.2 (2次违规)
    # 情况6: [False, True, True, False] -> -0.2 (2次违规)
    # 情况7: [True, True, True, False] -> -0.3 (3次违规)
    # 情况8: [False, False, False, True] -> -0.1 (1次违规)
    # 情况9: [True, False, False, True] -> -0.2 (2次违规)
    # 情况10: [False, True, False, True] -> -0.2 (2次违规)
    # 情况11: [True, True, False, True] -> -0.3 (3次违规)
    # 情况12: [False, False, True, True] -> -0.1 (1次违规，jerk是OR关系)
    # 情况13: [True, False, True, True] -> -0.2 (2次违规)
    # 情况14: [False, True, True, True] -> -0.2 (2次违规)
    # 情况15: [True, True, True, True] -> -0.3 (最大惩罚)
    # 测试随机化效果 - 16种情况反复运行
    print("\n=== 测试舒适度惩罚的随机化效果 ===")
    num_iterations = 1000
    all_results = []  # 存储所有结果
    # 定义16种情况的违规组合
    violation_combinations = []
    for along_violation in [False, True]:
        for alat_violation in [False, True]:
            for along_jerk_violation in [False, True]:
                for alat_jerk_violation in [False, True]:
                    violation_combinations.append((along_violation, alat_violation, along_jerk_violation, alat_jerk_violation))
    # 为每种情况创建测试数据
    test_cases = []
    for along_violation, alat_violation, along_jerk_violation, alat_jerk_violation in violation_combinations:
        if along_violation:
            along = torch.tensor([[4.0]], device=device)  # > 3.0
        else:
            along = torch.tensor([[2.0]], device=device)  # <= 3.0

        if alat_violation:
            alat = torch.tensor([[4.0]], device=device)  # > 3.0
        else:
            alat = torch.tensor([[2.0]], device=device)  # <= 3.0

        if along_jerk_violation:
            along_jerk = torch.tensor([[6.0]], device=device)  # > 5.0
        else:
            along_jerk = torch.tensor([[3.0]], device=device)  # <= 5.0

        if alat_jerk_violation:
            alat_jerk = torch.tensor([[6.0]], device=device)  # > 5.0
        else:
            alat_jerk = torch.tensor([[3.0]], device=device)  # <= 5.0

        test_cases.append((along, alat, along_jerk, alat_jerk))
    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        comfort_alpha_value = reward_calculator.comfort_alpha.cpu().item()

        # 测试所有16种情况
        iteration_results = []
        for case_idx, (along, alat, along_jerk, alat_jerk) in enumerate(test_cases):
            # 计算舒适度惩罚
            comfort_penalty = reward_calculator.calculate_comfort_penalty(
                along, alat, along_jerk, alat_jerk
            )

            # 计算预期违规次数
            violations = 0
            if torch.abs(along) > 3.0:
                violations += 1
            if torch.abs(alat) > 3.0:
                violations += 1
            if torch.abs(along_jerk) > 5.0 or torch.abs(alat_jerk) > 5.0:
                violations += 1

            iteration_results.append({
                'case_idx': case_idx,
                'comfort_alpha': comfort_alpha_value,
                'violations': violations,
                'penalty': comfort_penalty[0, 0].cpu().item(),
                'expected_penalty': -comfort_alpha_value * violations
            })

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    import numpy as np

    # 提取数据
    comfort_alphas = []
    penalties_by_case = [[] for _ in range(16)]
    violations_by_case = [[] for _ in range(16)]

    for iteration_results in all_results:
        for result in iteration_results:
            case_idx = result['case_idx']
            comfort_alphas.append(result['comfort_alpha'])
            penalties_by_case[case_idx].append(result['penalty'])
            violations_by_case[case_idx].append(result['violations'])

    comfort_alphas = np.array(comfort_alphas)
    penalties_by_case = np.array(penalties_by_case)
    violations_by_case = np.array(violations_by_case)

    # 绘制结果
    plt.figure(figsize=(20, 12))

    # 第一个子图：comfort_alpha的分布
    plt.subplot(2, 3, 1)
    plt.hist(comfort_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Comfort Alpha")
    plt.ylabel("Frequency")
    plt.title("Comfort Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：每种情况的平均惩罚
    plt.subplot(2, 3, 2)
    mean_penalties = np.mean(penalties_by_case, axis=1)
    std_penalties = np.std(penalties_by_case, axis=1)
    case_labels = [f"Case {i}" for i in range(16)]
    plt.bar(case_labels, mean_penalties, yerr=std_penalties, capsize=5, alpha=0.7)
    plt.xlabel("Test Case")
    plt.ylabel("Mean Penalty")
    plt.title("Mean Penalty by Case")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 第三个子图：违规次数分布
    plt.subplot(2, 3, 3)
    unique_violations, counts = np.unique(violations_by_case.flatten(), return_counts=True)
    plt.bar(unique_violations, counts, alpha=0.7)
    plt.xlabel("Number of Violations")
    plt.ylabel("Frequency")
    plt.title("Violation Count Distribution")
    plt.grid(True, alpha=0.3)

    # 第四个子图：alpha vs 惩罚的关系（所有情况）
    plt.subplot(2, 3, 4)
    for case_idx in range(16):
        plt.scatter(comfort_alphas[case_idx::16], penalties_by_case[case_idx], 
                   alpha=0.3, s=10, label=f"Case {case_idx}" if case_idx < 4 else "")
    plt.xlabel("Comfort Alpha")
    plt.ylabel("Penalty")
    plt.title("Alpha vs Penalty (All Cases)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 第五个子图：惩罚的分布
    plt.subplot(2, 3, 5)
    all_penalties = penalties_by_case.flatten()
    plt.hist(all_penalties, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Comfort Penalty")
    plt.ylabel("Frequency")
    plt.title("Overall Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第六个子图：违规次数 vs 平均惩罚
    plt.subplot(2, 3, 6)
    mean_violations = np.mean(violations_by_case, axis=1)
    plt.scatter(mean_violations, mean_penalties, s=100, alpha=0.7)
    for i, (v, p) in enumerate(zip(mean_violations, mean_penalties)):
        plt.annotate(f"Case {i}", (v, p), xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Mean Violations")
    plt.ylabel("Mean Penalty")
    plt.title("Violations vs Penalty")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_lane_alignment_reward():
    print("=== 测试车道对齐奖励 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'l_align_alpha_min': 2.5e-4,
            'l_align_alpha_max': 2.5e-2,
            'vel_align_alpha_min': 0.0,
            'vel_align_alpha_max': 1.0
        }
    }

    dt = 0.3
    num_iterations = 1000
    all_results = []

    # 生成测试数据
    # theta_f 在 [-pi, pi] 之间均匀分布
    # v 在 [-2, 20] 之间均匀分布
    num_theta_samples = 100
    num_speed_samples = 100

    theta_f_range = torch.linspace(-torch.pi, torch.pi, num_theta_samples, device=device)
    speed_range = torch.linspace(-2.0, 20.0, num_speed_samples, device=device)

    # 创建网格
    theta_f_grid, speed_grid = torch.meshgrid(theta_f_range, speed_range, indexing='ij')

    print(f"生成 {num_theta_samples}x{num_speed_samples} = {num_theta_samples*num_speed_samples} 个测试点")
    print("theta_f 范围: [-π, π]")
    print("速度范围: [-2, 20] m/s")

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        l_align_alpha = reward_calculator.l_align_alpha.cpu().item()
        vel_align_alpha = reward_calculator.vel_align_alpha.cpu().item()

        # 计算所有测试点的奖励
        rewards = reward_calculator.calculate_lane_alignment_reward(
            theta_f_grid.flatten(), speed_grid.flatten(), dt
        )

        # 重塑为网格形状
        rewards_grid = rewards.view(num_theta_samples, num_speed_samples)

        iteration_results = {
            'l_align_alpha': l_align_alpha,
            'vel_align_alpha': vel_align_alpha,
            'theta_f_grid': theta_f_grid.cpu().numpy(),
            'speed_grid': speed_grid.cpu().numpy(),
            'rewards_grid': rewards_grid.cpu().numpy(),
            'rewards_flat': rewards.cpu().numpy()
        }

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    l_align_alphas = np.array([result['l_align_alpha'] for result in all_results])
    vel_align_alphas = np.array([result['vel_align_alpha'] for result in all_results])
    all_rewards = np.array([result['rewards_flat'] for result in all_results])

    # 绘制结果
    plt.figure(figsize=(20, 15))

    # 第一个子图：l_align_alpha的分布
    plt.subplot(3, 4, 1)
    plt.hist(l_align_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("L-Align Alpha")
    plt.ylabel("Frequency")
    plt.title("L-Align Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：vel_align_alpha的分布
    plt.subplot(3, 4, 2)
    plt.hist(vel_align_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Vel-Align Alpha")
    plt.ylabel("Frequency")
    plt.title("Vel-Align Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：奖励的分布
    plt.subplot(3, 4, 3)
    all_rewards_flat = all_rewards.flatten()
    plt.hist(all_rewards_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Lane Alignment Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True, alpha=0.3)

    # 第四个子图：alpha vs 平均奖励的关系
    plt.subplot(3, 4, 4)
    mean_rewards = np.mean(all_rewards, axis=1)
    plt.scatter(l_align_alphas, mean_rewards, alpha=0.6, s=10)
    plt.xlabel("L-Align Alpha")
    plt.ylabel("Mean Reward")
    plt.title("Alpha vs Mean Reward")
    plt.grid(True, alpha=0.3)

    # 第五个子图：最后一次迭代的奖励热力图
    plt.subplot(3, 4, 5)
    last_result = all_results[-1]
    im = plt.imshow(last_result['rewards_grid'], 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Heatmap (Last Iteration)")

    # 第六个子图：平均奖励热力图
    plt.subplot(3, 4, 6)
    mean_rewards_grid = np.mean([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(mean_rewards_grid, 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Mean Reward Heatmap")

    # 第七个子图：奖励标准差热力图
    plt.subplot(3, 4, 7)
    std_rewards_grid = np.std([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(std_rewards_grid, 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Std Heatmap")

    # 第八个子图：theta_f vs 平均奖励（固定速度）
    plt.subplot(3, 4, 8)
    speed_idx = num_speed_samples // 2  # 选择中间速度
    mean_rewards_at_speed = np.mean([result['rewards_grid'][:, speed_idx] for result in all_results], axis=0)
    plt.plot(theta_f_range.cpu().numpy(), mean_rewards_at_speed)
    plt.xlabel("Theta_f (rad)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs Theta_f (Speed={speed_range[speed_idx].item():.1f} m/s)")
    plt.grid(True, alpha=0.3)

    # 第九个子图：速度 vs 平均奖励（固定theta_f）
    plt.subplot(3, 4, 9)
    theta_idx = num_theta_samples - 1  # 选择最大角度 (π)
    mean_rewards_at_theta = np.mean([result['rewards_grid'][theta_idx, :] for result in all_results], axis=0)
    plt.plot(speed_range.cpu().numpy(), mean_rewards_at_theta)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs Speed (Theta_f={theta_f_range[theta_idx].item():.2f} rad)")
    plt.grid(True, alpha=0.3)

    # 第十个子图：最大奖励位置
    plt.subplot(3, 4, 10)
    max_reward_idx = np.unravel_index(np.argmax(mean_rewards_grid), mean_rewards_grid.shape)
    max_theta = theta_f_range[max_reward_idx[0]].cpu().item()
    max_speed = speed_range[max_reward_idx[1]].cpu().item()
    max_reward = mean_rewards_grid[max_reward_idx]
    plt.scatter(max_speed, max_theta, c='red', s=200, marker='*', label=f'Max: {max_reward:.4f}')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Max Reward Position")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第十一个子图：奖励范围统计
    plt.subplot(3, 4, 11)
    reward_ranges = [np.max(result['rewards_flat']) - np.min(result['rewards_flat']) for result in all_results]
    plt.hist(reward_ranges, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Reward Range")
    plt.ylabel("Frequency")
    plt.title("Reward Range Distribution")
    plt.grid(True, alpha=0.3)

    # 第十二个子图：正负奖励比例
    plt.subplot(3, 4, 12)
    positive_ratios = [np.mean(result['rewards_flat'] > 0) for result in all_results]
    plt.hist(positive_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Positive Reward Ratio")
    plt.ylabel("Frequency")
    plt.title("Positive Reward Ratio Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_lane_center_reward():
    print("=== 测试车道中心奖励 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'l_center_alpha_min': 2.5e-4,
            'l_center_alpha_max': 7.5e-3,
            'center_bias_alpha_min': -0.5,
            'center_bias_alpha_max': 0.5
        }
    }

    dt = 0.3
    num_iterations = 1000
    all_results = []

    # 生成测试数据
    # theta_f 范围 [-pi, pi]
    # d 范围 [-2, 2]
    num_theta_samples = 50
    num_d_samples = 50

    theta_f_range = torch.linspace(-torch.pi, torch.pi, num_theta_samples, device=device)
    d_range = torch.linspace(-2.0, 2.0, num_d_samples, device=device)

    # 创建网格
    theta_f_grid, d_grid = torch.meshgrid(theta_f_range, d_range, indexing='ij')

    print(f"生成 {num_theta_samples}x{num_d_samples} = {num_theta_samples*num_d_samples} 个测试点")
    print("theta_f 范围: [-π, π]")
    print("d 范围: [-2, 2] m")

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        l_center_alpha = reward_calculator.l_center_alpha.cpu().item()
        center_bias_alpha = reward_calculator.center_bias_alpha.cpu().item()

        # 计算所有测试点的奖励
        rewards = reward_calculator.calculate_lane_center_reward(
            theta_f_grid.flatten(), d_grid.flatten(), dt
        )

        # 重塑为网格形状
        rewards_grid = rewards.view(num_theta_samples, num_d_samples)

        iteration_results = {
            'l_center_alpha': l_center_alpha,
            'center_bias_alpha': center_bias_alpha,
            'theta_f_grid': theta_f_grid.cpu().numpy(),
            'd_grid': d_grid.cpu().numpy(),
            'rewards_grid': rewards_grid.cpu().numpy(),
            'rewards_flat': rewards.cpu().numpy()
        }

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    l_center_alphas = np.array([result['l_center_alpha'] for result in all_results])
    center_bias_alphas = np.array([result['center_bias_alpha'] for result in all_results])
    all_rewards = np.array([result['rewards_flat'] for result in all_results])

    # 绘制结果
    plt.figure(figsize=(20, 15))

    # 第一个子图：l_center_alpha的分布
    plt.subplot(3, 4, 1)
    plt.hist(l_center_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("L-Center Alpha")
    plt.ylabel("Frequency")
    plt.title("L-Center Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：center_bias_alpha的分布
    plt.subplot(3, 4, 2)
    plt.hist(center_bias_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Center Bias Alpha")
    plt.ylabel("Frequency")
    plt.title("Center Bias Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：奖励的分布
    plt.subplot(3, 4, 3)
    all_rewards_flat = all_rewards.flatten()
    plt.hist(all_rewards_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Lane Center Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True, alpha=0.3)

    # 第四个子图：alpha vs 平均奖励的关系
    plt.subplot(3, 4, 4)
    mean_rewards = np.mean(all_rewards, axis=1)
    plt.scatter(l_center_alphas, mean_rewards, alpha=0.6, s=10)
    plt.xlabel("L-Center Alpha")
    plt.ylabel("Mean Reward")
    plt.title("Alpha vs Mean Reward")
    plt.grid(True, alpha=0.3)

    # 第五个子图：最后一次迭代的奖励热力图
    plt.subplot(3, 4, 5)
    last_result = all_results[-1]
    im = plt.imshow(last_result['rewards_grid'], 
                   extent=[-2, 2, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("d (m)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Heatmap (Last Iteration)")

    # 第六个子图：平均奖励热力图
    plt.subplot(3, 4, 6)
    mean_rewards_grid = np.mean([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(mean_rewards_grid, 
                   extent=[-2, 2, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("d (m)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Mean Reward Heatmap")

    # 第七个子图：奖励标准差热力图
    plt.subplot(3, 4, 7)
    std_rewards_grid = np.std([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(std_rewards_grid, 
                   extent=[-2, 2, -np.pi, np.pi], 
                   aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("d (m)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Std Heatmap")

    # 第八个子图：theta_f vs 平均奖励（固定d）
    plt.subplot(3, 4, 8)
    d_idx = num_d_samples // 2  # 选择中间d值
    mean_rewards_at_d = np.mean([result['rewards_grid'][:, d_idx] for result in all_results], axis=0)
    plt.plot(theta_f_range.cpu().numpy(), mean_rewards_at_d)
    plt.xlabel("Theta_f (rad)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs Theta_f (d={d_range[d_idx].item():.1f} m)")
    plt.grid(True, alpha=0.3)

    # 第九个子图：d vs 平均奖励（固定theta_f）
    plt.subplot(3, 4, 9)
    theta_idx = num_theta_samples - 1  # 选择最大角度 (π)
    mean_rewards_at_theta = np.mean([result['rewards_grid'][theta_idx, :] for result in all_results], axis=0)
    plt.plot(d_range.cpu().numpy(), mean_rewards_at_theta)
    plt.xlabel("d (m)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs d (Theta_f={theta_f_range[theta_idx].item():.2f} rad)")
    plt.grid(True, alpha=0.3)

    # 第十个子图：最小奖励位置
    plt.subplot(3, 4, 10)
    min_reward_idx = np.unravel_index(np.argmin(mean_rewards_grid), mean_rewards_grid.shape)
    min_theta = theta_f_range[min_reward_idx[0]].cpu().item()
    min_d = d_range[min_reward_idx[1]].cpu().item()
    min_reward = mean_rewards_grid[min_reward_idx]
    plt.scatter(min_d, min_theta, c='red', s=200, marker='*', label=f'Min: {min_reward:.4f}')
    plt.xlabel("d (m)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Min Reward Position")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第十一个子图：奖励范围统计
    plt.subplot(3, 4, 11)
    reward_ranges = [np.max(result['rewards_flat']) - np.min(result['rewards_flat']) for result in all_results]
    plt.hist(reward_ranges, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Reward Range")
    plt.ylabel("Frequency")
    plt.title("Reward Range Distribution")
    plt.grid(True, alpha=0.3)

    # 第十二个子图：负奖励比例（车道中心奖励通常是负的）
    plt.subplot(3, 4, 12)
    negative_ratios = [np.mean(result['rewards_flat'] < 0) for result in all_results]
    plt.hist(negative_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Negative Reward Ratio")
    plt.ylabel("Frequency")
    plt.title("Negative Reward Ratio Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_velocity_reward():
    print("=== 测试速度奖励 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'velocity_alpha': 2.5e-3
        }
    }

    dt = 0.3
    num_iterations = 1000
    all_results = []

    # 生成测试数据
    # theta_f 在 [-pi, pi] 之间均匀分布
    # v 在 [-2, 20] 之间均匀分布
    num_theta_samples = 50
    num_speed_samples = 50

    theta_f_range = torch.linspace(-torch.pi, torch.pi, num_theta_samples, device=device)
    speed_range = torch.linspace(-2.0, 20.0, num_speed_samples, device=device)

    # 创建网格
    theta_f_grid, speed_grid = torch.meshgrid(theta_f_range, speed_range, indexing='ij')

    print(f"生成 {num_theta_samples}x{num_speed_samples} = {num_theta_samples*num_speed_samples} 个测试点")
    print("theta_f 范围: [-π, π]")
    print("速度范围: [-2, 20] m/s")

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        velocity_alpha = reward_calculator.velocity_alpha

        # 计算所有测试点的奖励
        rewards = reward_calculator.calculate_velocity_reward(
            theta_f_grid.flatten(), speed_grid.flatten(), dt
        )

        # 重塑为网格形状
        rewards_grid = rewards.view(num_theta_samples, num_speed_samples)

        iteration_results = {
            'velocity_alpha': velocity_alpha,
            'theta_f_grid': theta_f_grid.cpu().numpy(),
            'speed_grid': speed_grid.cpu().numpy(),
            'rewards_grid': rewards_grid.cpu().numpy(),
            'rewards_flat': rewards.cpu().numpy()
        }

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    velocity_alphas = np.array([result['velocity_alpha'] for result in all_results])
    all_rewards = np.array([result['rewards_flat'] for result in all_results])

    # 绘制结果
    plt.figure(figsize=(20, 15))

    # 第一个子图：velocity_alpha的分布
    plt.subplot(3, 4, 1)
    plt.hist(velocity_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Velocity Alpha")
    plt.ylabel("Frequency")
    plt.title("Velocity Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：奖励的分布
    plt.subplot(3, 4, 2)
    all_rewards_flat = all_rewards.flatten()
    plt.hist(all_rewards_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Velocity Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：alpha vs 平均奖励的关系
    plt.subplot(3, 4, 3)
    mean_rewards = np.mean(all_rewards, axis=1)
    plt.scatter(velocity_alphas, mean_rewards, alpha=0.6, s=10)
    plt.xlabel("Velocity Alpha")
    plt.ylabel("Mean Reward")
    plt.title("Alpha vs Mean Reward")
    plt.grid(True, alpha=0.3)

    # 第四个子图：正奖励比例
    plt.subplot(3, 4, 4)
    positive_ratios = [np.mean(result['rewards_flat'] > 0) for result in all_results]
    plt.hist(positive_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Positive Reward Ratio")
    plt.ylabel("Frequency")
    plt.title("Positive Reward Ratio Distribution")
    plt.grid(True, alpha=0.3)

    # 第五个子图：最后一次迭代的奖励热力图
    plt.subplot(3, 4, 5)
    last_result = all_results[-1]
    im = plt.imshow(last_result['rewards_grid'], 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Heatmap (Last Iteration)")

    # 第六个子图：平均奖励热力图
    plt.subplot(3, 4, 6)
    mean_rewards_grid = np.mean([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(mean_rewards_grid, 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Mean Reward Heatmap")

    # 第七个子图：奖励标准差热力图
    plt.subplot(3, 4, 7)
    std_rewards_grid = np.std([result['rewards_grid'] for result in all_results], axis=0)
    im = plt.imshow(std_rewards_grid, 
                   extent=[-2, 20, -np.pi, np.pi], 
                   aspect='auto', cmap='viridis')
    plt.colorbar(im)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Reward Std Heatmap")

    # 第八个子图：theta_f vs 平均奖励（固定速度）
    plt.subplot(3, 4, 8)
    speed_idx = num_speed_samples // 2  # 选择中间速度
    mean_rewards_at_speed = np.mean([result['rewards_grid'][:, speed_idx] for result in all_results], axis=0)
    plt.plot(theta_f_range.cpu().numpy(), mean_rewards_at_speed)
    plt.xlabel("Theta_f (rad)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs Theta_f (Speed={speed_range[speed_idx].item():.1f} m/s)")
    plt.grid(True, alpha=0.3)

    # 第九个子图：速度 vs 平均奖励（固定theta_f）
    plt.subplot(3, 4, 9)
    theta_idx = num_theta_samples - 1  # 选择最大角度 (π)
    mean_rewards_at_theta = np.mean([result['rewards_grid'][theta_idx, :] for result in all_results], axis=0)
    plt.plot(speed_range.cpu().numpy(), mean_rewards_at_theta)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Mean Reward")
    plt.title(f"Reward vs Speed (Theta_f={theta_f_range[theta_idx].item():.2f} rad)")
    plt.grid(True, alpha=0.3)

    # 第十个子图：最大奖励位置
    plt.subplot(3, 4, 10)
    max_reward_idx = np.unravel_index(np.argmax(mean_rewards_grid), mean_rewards_grid.shape)
    max_theta = theta_f_range[max_reward_idx[0]].cpu().item()
    max_speed = speed_range[max_reward_idx[1]].cpu().item()
    max_reward = mean_rewards_grid[max_reward_idx]
    plt.scatter(max_speed, max_theta, c='red', s=200, marker='*', label=f'Max: {max_reward:.4f}')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Theta_f (rad)")
    plt.title("Max Reward Position")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第十一个子图：奖励范围统计
    plt.subplot(3, 4, 11)
    reward_ranges = [np.max(result['rewards_flat']) - np.min(result['rewards_flat']) for result in all_results]
    plt.hist(reward_ranges, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Reward Range")
    plt.ylabel("Frequency")
    plt.title("Reward Range Distribution")
    plt.grid(True, alpha=0.3)

    # 第十二个子图：速度阈值分析
    plt.subplot(3, 4, 12)
    # 分析速度阈值2.5的影响
    speed_threshold = 2.5
    threshold_idx = np.argmin(np.abs(speed_range.cpu().numpy() - speed_threshold))

    # 分别计算低于和高于阈值的平均奖励
    below_threshold_rewards = []
    above_threshold_rewards = []

    for result in all_results:
        rewards_grid = result['rewards_grid']
        below_threshold_rewards.extend(rewards_grid[:, :threshold_idx].flatten())
        above_threshold_rewards.extend(rewards_grid[:, threshold_idx:].flatten())

    mean_rewards_below_threshold = np.mean(below_threshold_rewards)
    mean_rewards_above_threshold = np.mean(above_threshold_rewards)

    plt.bar(['Below 2.5 m/s', 'Above 2.5 m/s'], 
            [mean_rewards_below_threshold, mean_rewards_above_threshold], 
            alpha=0.7, color=['red', 'green'])
    plt.ylabel("Mean Reward")
    plt.title("Reward by Speed Threshold")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_reverse_penalty():
    print("=== 测试倒车惩罚 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'reverse_alpha_min': 2.5e-4,
            'reverse_alpha_max': 7.5e-3
        }
    }

    dt = 0.3
    num_iterations = 1000
    all_results = []

    # 生成测试数据
    # 速度在 [-2, 20] 之间均匀分布（包含负速度用于倒车测试）
    num_speed_samples = 100

    speed_range = torch.linspace(-2.0, 20.0, num_speed_samples, device=device)

    print(f"生成 {num_speed_samples} 个测试点")
    print("速度范围: [-2, 20] m/s")

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        reverse_alpha = reward_calculator.reverse_alpha.cpu().item()

        # 计算所有测试点的惩罚
        penalties = reward_calculator.calculate_reverse_penalty(speed_range, dt)

        iteration_results = {
            'reverse_alpha': reverse_alpha,
            'speed_range': speed_range.cpu().numpy(),
            'penalties': penalties.cpu().numpy()
        }

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    reverse_alphas = np.array([result['reverse_alpha'] for result in all_results])
    all_penalties = np.array([result['penalties'] for result in all_results])

    # 绘制结果
    plt.figure(figsize=(20, 12))

    # 第一个子图：reverse_alpha的分布
    plt.subplot(2, 4, 1)
    plt.hist(reverse_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Reverse Alpha")
    plt.ylabel("Frequency")
    plt.title("Reverse Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：惩罚的分布
    plt.subplot(2, 4, 2)
    all_penalties_flat = all_penalties.flatten()
    plt.hist(all_penalties_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Reverse Penalty")
    plt.ylabel("Frequency")
    plt.title("Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：alpha vs 平均惩罚的关系
    plt.subplot(2, 4, 3)
    mean_penalties = np.mean(all_penalties, axis=1)
    plt.scatter(reverse_alphas, mean_penalties, alpha=0.6, s=10)
    plt.xlabel("Reverse Alpha")
    plt.ylabel("Mean Penalty")
    plt.title("Alpha vs Mean Penalty")
    plt.grid(True, alpha=0.3)

    # 第四个子图：负惩罚比例（倒车惩罚通常是负的）
    plt.subplot(2, 4, 4)
    negative_ratios = [np.mean(result['penalties'] < 0) for result in all_results]
    plt.hist(negative_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Negative Penalty Ratio")
    plt.ylabel("Frequency")
    plt.title("Negative Penalty Ratio Distribution")
    plt.grid(True, alpha=0.3)

    # 第五个子图：最后一次迭代的惩罚曲线
    plt.subplot(2, 4, 5)
    last_result = all_results[-1]
    plt.plot(last_result['speed_range'], last_result['penalties'], 'b-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Speed = 0')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Penalty")
    plt.title("Penalty vs Speed (Last Iteration)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第六个子图：平均惩罚曲线
    plt.subplot(2, 4, 6)
    mean_penalties_curve = np.mean(all_penalties, axis=0)
    plt.plot(speed_range.cpu().numpy(), mean_penalties_curve, 'g-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Speed = 0')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Mean Penalty")
    plt.title("Mean Penalty vs Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第七个子图：惩罚标准差曲线
    plt.subplot(2, 4, 7)
    std_penalties_curve = np.std(all_penalties, axis=0)
    plt.plot(speed_range.cpu().numpy(), std_penalties_curve, 'm-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Speed = 0')
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Penalty Std")
    plt.title("Penalty Std vs Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 第八个子图：速度阈值分析（0 m/s）
    plt.subplot(2, 4, 8)
    # 分析速度阈值0的影响（正负速度）
    speed_threshold = 0.0
    threshold_idx = np.argmin(np.abs(speed_range.cpu().numpy() - speed_threshold))

    # 分别计算正速度和负速度的平均惩罚
    positive_speed_penalties = []
    negative_speed_penalties = []

    for result in all_results:
        penalties = result['penalties']
        positive_speed_penalties.extend(penalties[threshold_idx:])
        negative_speed_penalties.extend(penalties[:threshold_idx])

    mean_penalties_positive = np.mean(positive_speed_penalties)
    mean_penalties_negative = np.mean(negative_speed_penalties)

    plt.bar(['Positive Speed', 'Negative Speed'], 
            [mean_penalties_positive, mean_penalties_negative], 
            alpha=0.7, color=['green', 'red'])
    plt.ylabel("Mean Penalty")
    plt.title("Penalty by Speed Sign")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_stop_line_penalty():
    print("=== 测试停止线违规惩罚 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'stop_line_alpha_min': 0.0,
            'stop_line_alpha_max': 1.0
        }
    }

    num_iterations = 1000
    all_results = []

    # 生成测试数据
    batch_sizes = [1000]
    agent_counts = [100]

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        stop_line_alpha = reward_calculator.stop_line_alpha.cpu().item()

        # 随机选择批次大小和智能体数量
        B = np.random.choice(batch_sizes)
        M = np.random.choice(agent_counts)

        # 创建停止线违规掩码 - 全部为1（违规）
        stop_line_violation = torch.ones((B, M), dtype=torch.bool, device=device)

        # 计算惩罚
        penalties = reward_calculator.calculate_stop_line_penalty(stop_line_violation)

        iteration_results = {
            'stop_line_alpha': stop_line_alpha,
            'batch_size': B,
            'agent_count': M,
            'penalties': penalties.cpu().numpy(),
            'mean_penalty': penalties.mean().cpu().item(),
            'total_penalty': penalties.sum().cpu().item()
        }
        all_results.append(iteration_results)
        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    stop_line_alphas = np.array([result['stop_line_alpha'] for result in all_results])
    mean_penalties = np.array([result['mean_penalty'] for result in all_results])
    total_penalties = np.array([result['total_penalty'] for result in all_results])

    # 绘制结果
    plt.figure(figsize=(15, 10))

    # 第一个子图：stop_line_alpha的分布
    plt.subplot(2, 3, 1)
    plt.hist(stop_line_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Stop Line Alpha")
    plt.ylabel("Frequency")
    plt.title("Stop Line Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：平均惩罚的分布
    plt.subplot(2, 3, 2)
    plt.hist(mean_penalties, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Mean Penalty")
    plt.ylabel("Frequency")
    plt.title("Mean Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：总惩罚的分布
    plt.subplot(2, 3, 3)
    plt.hist(total_penalties, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Total Penalty")
    plt.ylabel("Frequency")
    plt.title("Total Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第四个子图：alpha vs 平均惩罚的关系
    plt.subplot(2, 3, 4)
    plt.scatter(stop_line_alphas, mean_penalties, alpha=0.6, s=10)
    plt.xlabel("Stop Line Alpha")
    plt.ylabel("Mean Penalty")
    plt.title("Alpha vs Mean Penalty")
    plt.grid(True, alpha=0.3)

    # 第五个子图：alpha vs 总惩罚的关系
    plt.subplot(2, 3, 5)
    plt.scatter(stop_line_alphas, total_penalties, alpha=0.6, s=10)
    plt.xlabel("Stop Line Alpha")
    plt.ylabel("Total Penalty")
    plt.title("Alpha vs Total Penalty")
    plt.grid(True, alpha=0.3)

    # 第六个子图：惩罚统计摘要
    plt.subplot(2, 3, 6)
    penalty_stats = {
        'Mean': np.mean(mean_penalties),
        'Std': np.std(mean_penalties),
        'Min': np.min(mean_penalties),
        'Max': np.max(mean_penalties)
    }

    plt.bar(penalty_stats.keys(), penalty_stats.values(), alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    plt.ylabel("Penalty Value")
    plt.title("Penalty Statistics Summary")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_timestep_penalty():
    print("=== 测试时间步惩罚 ===")
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试配置
    test_config = {
        'reward': {
            'timestep_alpha': 2.5e-5
        }
    }

    dt = 0.3
    num_iterations = 1000
    all_results = []

    # 生成测试数据
    # speed 范围 [-2,20]
    # along: [-15, -4, 0, 4] m/s² (纵向加速度)
    # alat: [-4, 0, 4] m/s² (横向加速度)
    speed_range = [-2.0, 20.0]
    along_values = [-15.0, -4.0, 0.0, 4.0]
    alat_values = [-4.0, 0.0, 4.0]

    print(f"生成 {num_iterations} 次测试")
    print(f"速度范围: {speed_range}")
    print(f"纵向加速度值: {along_values}")
    print(f"横向加速度值: {alat_values}")

    # 反复运行测试
    for i in range(num_iterations):
        # 每次重新初始化奖励计算器
        reward_calculator = RewardCalculator(test_config, device)
        timestep_alpha = reward_calculator.timestep_alpha

        # 随机生成批次大小和智能体数量
        B = np.random.randint(1, 101)  # 1-100
        M = np.random.randint(1, 101)  # 1-100

        # 随机生成速度
        speeds = torch.empty((B, M), device=device).uniform_(speed_range[0], speed_range[1])

        # 从固定值中随机选择 along 和 alat
        along = torch.full((B, M), np.random.choice(along_values), device=device)
        alat = torch.full((B, M), np.random.choice(alat_values), device=device)

        # 计算时间步惩罚
        penalties = reward_calculator.calculate_timestep_penalty(speeds, along, alat, dt)

        iteration_results = {
            'timestep_alpha': timestep_alpha,
            'penalties': penalties.cpu().numpy(),
            'speeds': speeds.cpu().numpy(),
            'along': along.cpu().numpy(),
            'alat': alat.cpu().numpy(),
            'acceleration_magnitude': torch.sqrt(along**2 + alat**2).cpu().numpy()
        }

        all_results.append(iteration_results)

        if i % 100 == 0:
            print(f"完成 {i+1}/{num_iterations} 次迭代")

    # 转换为numpy数组进行分析
    timestep_alphas = np.array([result['timestep_alpha'] for result in all_results])
    all_penalties = np.concatenate([result['penalties'].flatten() for result in all_results])
    all_speeds = np.concatenate([result['speeds'].flatten() for result in all_results])
    all_acceleration_magnitudes = np.concatenate([result['acceleration_magnitude'].flatten() for result in all_results])

    # 绘制结果
    plt.figure(figsize=(15, 8))

    # 第一个子图：timestep_alpha的分布
    plt.subplot(2, 2, 1)
    plt.hist(timestep_alphas, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel("Timestep Alpha")
    plt.ylabel("Frequency")
    plt.title("Timestep Alpha Distribution")
    plt.grid(True, alpha=0.3)

    # 第二个子图：惩罚的分布
    plt.subplot(2, 2, 2)
    plt.hist(all_penalties, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Penalty")
    plt.ylabel("Frequency")
    plt.title("Penalty Distribution")
    plt.grid(True, alpha=0.3)

    # 第三个子图：速度 vs 惩罚
    plt.subplot(2, 2, 3)
    plt.scatter(all_speeds, all_penalties, alpha=0.6, s=5)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Penalty")
    plt.title("Speed vs Penalty")
    plt.grid(True, alpha=0.3)

    # 第四个子图：加速度大小 vs 惩罚
    plt.subplot(2, 2, 4)
    plt.scatter(all_acceleration_magnitudes, all_penalties, alpha=0.6, s=5)
    plt.xlabel("Acceleration Magnitude (m/s²)")
    plt.ylabel("Penalty")
    plt.title("Acceleration Magnitude vs Penalty")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n=== 时间步惩罚统计 ===")
    print(f"测试迭代次数: {num_iterations}")
    print(f"平均 alpha 值: {np.mean(timestep_alphas):.8f}")
    print(f"平均惩罚值: {np.mean(all_penalties):.8f}")
    print(f"惩罚值范围: [{np.min(all_penalties):.8f}, {np.max(all_penalties):.8f}]")
    print(f"平均速度: {np.mean(all_speeds):.2f} m/s")
    print(f"平均加速度大小: {np.mean(all_acceleration_magnitudes):.2f} m/s²")

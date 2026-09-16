"""Manual diagnostics moved from simulator/simulator.py."""

from _bootstrap import PROJECT_ROOT, add_project_paths

add_project_paths()

from types import SimpleNamespace

import torch
import yaml

from simulator import TeraflowSimulator

# 这是一个简单的使用示例，用于测试模拟器的基本功能
# 从配置文件读取配置
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

# 基于文件位置解析项目根目录，避免依赖当前工作目录
config_path = PROJECT_ROOT / 'configs' / 'default_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
simulator = TeraflowSimulator(config=config, device=device)

initial_obs = simulator.reset()
print(f"Initial observation batch shape: {initial_obs.shape}")

# 可视化道路网络和智能体位置（与goals.py绘制风格保持一致）
print("\n=== 可视化道路网络和智能体位置 ===")
# 获取道路网络的四边形顶点 这里是测试road.py
quads_vertices = simulator.road_network.quads_vertices  # (num_quads, 4, 2)
quads_vertices_np = quads_vertices.cpu().numpy()
# 获取智能体状态 这里已经测试过world_initializer.py
agents_state_np = simulator.agents_state.cpu().numpy()  # (B, M, 7)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10))
# 方法1: 使用PatchCollection进行批量绘制（最快）
patches = []
# 批量创建Polygon对象
for i in range(len(quads_vertices_np)):
    vertices = quads_vertices_np[i]  # (4, 2)
    polygon = Polygon(vertices, closed=True)
    patches.append(polygon)
p = PatchCollection(patches, alpha=0.2, facecolor='lightblue', edgecolor='black', linewidth=0.1)
# 一次性添加所有quads到图形
ax.add_collection(p)

# 构建可更新的智能体绘制（仅显示第一个环境）
def build_agent_artists():
    ax_agents = []
    agents_state_np_local = simulator.agents_state.cpu().numpy()
    active_mask_local = agents_state_np_local[0, :, 6] > 0.5
    active_indices_local = np.where(active_mask_local)[0]
    if len(active_indices_local) == 0:
        return ax_agents, active_indices_local
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    import math
    for i, agent_idx in enumerate(active_indices_local):
        x, y, yaw, speed, length, width, active = agents_state_np_local[0, agent_idx]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # 智能体矩形的四个角点 (相对于中心)
        half_length = length / 2.0
        half_width = width / 2.0
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        agent_corners = corners @ rotation_matrix.T + np.array([x, y])
        poly = Polygon(agent_corners, closed=True)
        color = colors[i % len(colors)]
        ax.add_patch(poly)
        poly.set_facecolor(color)
        poly.set_alpha(0.8)
        poly.set_edgecolor('black')
        poly.set_linewidth(2)
        # 仅为第一个激活agent显示标签与速度文本
        if i == 0:
            label = f'Agent {agent_idx}'
            txt = ax.text(x, y, label, ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                          weight='bold')
        else:
            txt = None
        speed_vec = 3.0
        arr = ax.arrow(x, y, speed_vec * cos_yaw, speed_vec * sin_yaw,
                       head_width=0.5, head_length=0.5, fc=color, ec=color,
                       alpha=0.8, zorder=5, linewidth=2)
        # 仅第一个激活agent显示速度文本
        if i == 0:
            info = ax.text(x, y + half_width + 1, f'v={speed:.1f}m/s', ha='center', va='bottom',
                           fontsize=8, color=color, weight='bold')
        else:
            info = None
        ax_agents.append((agent_idx, poly, txt, arr, info, color))
    return ax_agents, active_indices_local

agent_artists, active_indices = build_agent_artists()

# 构建策略网络与初始特征（延迟导入避免循环依赖）
from ddppo import build_network_features, current_navigation
from network import create_network


def decompose_observation(observation, config):
    """Expand the simulator's legacy flat observation for this diagnostic only."""
    batch_size, max_agents, _ = observation.shape
    observation_config = config.simulator.observation
    local_state_dim = observation_config.local_state_dim
    neighbor_feature_dim = observation_config.neighbor_feature_dim
    waypoint_feature_dim = observation_config.waypoint_feature_dim
    boundary_feature_dim = observation_config.boundary_feature_dim
    num_neighbors = observation_config.num_neighbors
    num_w_lanes = observation_config.num_w_lanes
    num_w_boundaries = observation_config.num_w_boundaries

    neighbors_start = local_state_dim
    neighbors_end = neighbors_start + num_neighbors * neighbor_feature_dim
    lanes_end = neighbors_end + num_w_lanes * waypoint_feature_dim
    agents_state = observation[:, :, :local_state_dim]
    neighbors_local = observation[:, :, neighbors_start:neighbors_end].view(
        batch_size, max_agents, num_neighbors, neighbor_feature_dim
    )
    w_lanes_local = observation[:, :, neighbors_end:lanes_end].view(
        batch_size, max_agents, num_w_lanes, waypoint_feature_dim
    )
    w_boundaries_local = observation[:, :, lanes_end:].view(
        batch_size, max_agents, num_w_boundaries, boundary_feature_dim
    )
    return agents_state, neighbors_local, w_lanes_local, w_boundaries_local
import json as _json

config_ns = _json.loads(_json.dumps(config), object_hook=lambda d: SimpleNamespace(**d))
model = create_network(config=config_ns, network_type="independent").to(device)
model.eval()
with torch.no_grad():
    agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_obs, config_ns)
    features_tensor = build_network_features(
        agents_state_dec,
        neighbors_local,
        w_lanes_local,
        w_boundaries_local,
        current_navigation(simulator),
        simulator.stop_lines,
        simulator.reward_calculator.sampled_params,
        config_ns,
    )

# 在主图上绘制局部要素（第一个环境第一个激活agent）
overlay_artists = []
def clear_overlays():
    global overlay_artists
    for art in overlay_artists:
        try:
            art.remove()
        except Exception:
            pass
    overlay_artists = []

def draw_local_overlays(agents_state_dec_t, neighbors_local_t, w_lanes_local_t, w_boundaries_local_t):
    global overlay_artists, first_agent_idx
    try:
        clear_overlays()
        import numpy as np
        import math
        # 自车位姿
        ego = simulator.agents_state[0, first_agent_idx]
        ex = float(ego[0].item()); ey = float(ego[1].item()); eyaw = float(ego[2].item())
        cos_y = math.cos(eyaw); sin_y = math.sin(eyaw)
        R = np.array([[cos_y, -sin_y],[sin_y, cos_y]], dtype=float)

        # lanes
        lanes = w_lanes_local_t[0, first_agent_idx] if w_lanes_local_t is not None else None
        if lanes is not None:
            lanes_np = lanes.detach().to('cpu').numpy()
            if lanes_np.ndim >= 2 and lanes_np.shape[-1] >= 2:
                valid = (lanes_np[...,0] != -1) & (lanes_np[...,1] != -1)
                pts = lanes_np[valid][..., :2]
                if pts.size > 0:
                    world = pts @ R.T + np.array([ex, ey])
                    h = ax.scatter(world[:,0], world[:,1], s=5, c='lime', alpha=0.8, label='w_lanes_local')
                    overlay_artists.append(h)

        # boundaries
        bounds = w_boundaries_local_t[0, first_agent_idx] if w_boundaries_local_t is not None else None
        if bounds is not None:
            bounds_np = bounds.detach().to('cpu').numpy()
            if bounds_np.ndim >= 2 and bounds_np.shape[-1] >= 2:
                valid = (bounds_np[...,0] != -1) & (bounds_np[...,1] != -1)
                pts = bounds_np[valid][..., :2]
                if pts.size > 0:
                    world = pts @ R.T + np.array([ex, ey])
                    h = ax.scatter(world[:,0], world[:,1], s=4, c='k', alpha=0.5, label='w_boundaries_local')
                    overlay_artists.append(h)

        # neighbors_local: [dx, dy, heading_x, heading_y, dvx, dvy, length, width, z, active]
        neigh = neighbors_local_t[0, first_agent_idx] if neighbors_local_t is not None else None
        if neigh is not None:
            neigh_np = neigh.detach().to('cpu').numpy()
            if neigh_np.ndim >= 2 and neigh_np.shape[-1] >= 6:
                # 有效点：active>0.5 或者 长宽>0
                active_mask = neigh_np[..., -1] > 0.5 if neigh_np.shape[-1] >= 7 else np.ones(neigh_np.shape[0], dtype=bool)
                valid = active_mask
                dxdy = neigh_np[valid][..., :2]
                if dxdy.size > 0:
                    world_pts = dxdy @ R.T + np.array([ex, ey])
                    # 只为被观察到的邻居绘制标签（一次性标注）
                    h = ax.scatter(world_pts[:,0], world_pts[:,1], s=20, facecolors='none', edgecolors='red', linewidths=2, label='neighbors_local')
                    overlay_artists.append(h)
                    # 标注被观察到的邻居（仅一次图例）
                    for j, (wx, wy) in enumerate(world_pts):
                        txtn = ax.text(wx, wy, 'N', fontsize=8, color='red', weight='bold')
                        overlay_artists.append(txtn)
                    try:
                        # 仅处理前N个，避免过多图元
                        max_draw = min(world_pts.shape[0], 20)
                        # 取对应的行索引
                        valid_indices = np.nonzero(valid)[0][:max_draw]
                        # 计算自车绝对速度（世界坐标）
                        ego_speed = float(simulator.agents_state[0, first_agent_idx, 3].item())
                        vx_ego = ego_speed * cos_y
                        vy_ego = ego_speed * sin_y
                        for ii in valid_indices:
                            row = neigh_np[ii]
                            nx, ny = float(row[0]), float(row[1])
                            if row.shape[0] >= 10:
                                heading_x, heading_y = float(row[2]), float(row[3])
                                dvx_local, dvy_local = float(row[4]), float(row[5])
                                nlen = float(row[6])
                                nwid = float(row[7])
                            else:
                                heading_x = heading_y = None
                                dvx_local, dvy_local = float(row[2]), float(row[3])
                                nlen = float(row[4])
                                nwid = float(row[5])
                            # 局部中心 -> 世界中心
                            cx, cy = (R @ np.array([nx, ny])).tolist(); cx += ex; cy += ey
                            # 相对速度(局部) -> 世界相对速度
                            rvx_world, rvy_world = (R @ np.array([dvx_local, dvy_local])).tolist()
                            # 近似邻居绝对速度 = 自车绝对速度 + 相对世界速度
                            nvx_world = vx_ego + rvx_world
                            nvy_world = vy_ego + rvy_world
                            speed_mag = math.hypot(nvx_world, nvy_world)
                            if heading_x is not None and math.hypot(heading_x, heading_y) > 1e-3:
                                nyaw_world = eyaw + math.atan2(heading_y, heading_x)
                            elif speed_mag > 1e-2:
                                nyaw_world = math.atan2(nvy_world, nvx_world)
                            else:
                                nyaw_world = eyaw
                            c = math.cos(nyaw_world); s = math.sin(nyaw_world)
                            Rn = np.array([[c, -s], [s, c]], dtype=float)
                            hl = max(0.1, nlen * 0.5); hw = max(0.1, nwid * 0.5)
                            rect_local = np.array([
                                [-hl, -hw],
                                [ hl, -hw],
                                [ hl,  hw],
                                [-hl,  hw]
                            ], dtype=float)
                            rect_world = rect_local @ Rn.T + np.array([cx, cy])
                            # 邻居整体涂黑 + 金色描边
                            poly = Polygon(rect_world, closed=True, facecolor='black', edgecolor='gold', linewidth=2.0, alpha=0.9)
                            ax.add_patch(poly)
                            overlay_artists.append(poly)
                            # 绘制邻居世界速度方向（金色箭头，长度按速度幅值裁剪）
                            if speed_mag > 1e-3:
                                ux = nvx_world / speed_mag
                                uy = nvy_world / speed_mag
                                arrow_len = max(3.0, min(8.0, speed_mag))
                                arr_v = ax.arrow(cx, cy, ux * arrow_len, uy * arrow_len,
                                                 head_width=0.8, head_length=0.8, fc='gold', ec='gold',
                                                 alpha=0.95, zorder=7, linewidth=2)
                                overlay_artists.append(arr_v)
                    except Exception:
                        pass

        fig.canvas.draw_idle()
    except Exception as e:
        print(f"draw_local_overlays error: {e}")

# 初始绘制一次
try:
    draw_local_overlays(agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local)
except Exception:
    pass

# 统一图形样式
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)
ax.set_title('road graph and agent positions')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
# 绘制观测半径虚线圆（以第一个激活agent为圆心）
horizon_circle = None
try:
    horizon = float(config['simulator']['observation']['horizon'])
    if len(active_indices) > 0:
        first_idx = int(active_indices[0])
        cx = float(simulator.agents_state[0, first_idx, 0].item())
        cy = float(simulator.agents_state[0, first_idx, 1].item())
        horizon_circle = Circle((cx, cy), radius=horizon, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.add_patch(horizon_circle)
except Exception:
    pass

# 添加 Next Step 按钮
btn_ax = fig.add_axes([0.82, 0.02, 0.15, 0.05])
btn_next = Button(btn_ax, 'Next Step')

# 第二个figure：动作概率分布（仅第一个环境的第一个agent）
num_actions = simulator.dynamics_model.discrete_action_space.num_actions
first_agent_idx = 0
if len(active_indices) > 0:
    first_agent_idx = int(active_indices[0])
fig_act, ax_act = plt.subplots(figsize=(6, 3))
fig_act.canvas.manager.set_window_title('Action Probabilities (Agent 0)')
bars = ax_act.bar(np.arange(num_actions), np.zeros(num_actions), color='tab:blue')
ax_act.set_xlabel('Action Index')
ax_act.set_ylabel('Probability')
ax_act.set_title('First Agent Action Probabilities')
ax_act.set_xlim(-0.5, num_actions - 0.5)
ax_act.set_ylim(0.0, 1.0)
fig_act.tight_layout()

def refresh_agents():
    global agent_artists, active_indices, horizon_circle
    agents_state_np_local = simulator.agents_state.cpu().numpy()
    new_active_mask = agents_state_np_local[0, :, 6] > 0.5
    new_active_indices = np.where(new_active_mask)[0]
    if not np.array_equal(new_active_indices, active_indices):
        for _, poly, txt, arr, info, _ in agent_artists:
            try:
                poly.remove(); txt.remove(); info.remove(); arr.remove()
            except Exception:
                pass
        agent_artists, active_indices = build_agent_artists()
        fig.canvas.draw_idle()
        return
    import math
    for (agent_idx, poly, txt, arr, info, color) in agent_artists:
        x, y, yaw, speed, length, width, active = agents_state_np_local[0, agent_idx]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = length / 2.0
        half_width = width / 2.0
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        agent_corners = corners @ rotation_matrix.T + np.array([x, y])
        poly.set_xy(agent_corners)
        if txt is not None:
            txt.set_position((x, y))
        try:
            arr.remove()
        except Exception:
            pass
        speed_vec = 3.0
        new_arr = ax.arrow(x, y, speed_vec * cos_yaw, speed_vec * sin_yaw,
                           head_width=0.5, head_length=0.5, fc=color, ec=color,
                           alpha=0.8, zorder=5, linewidth=2)
        idx = [i for i, t in enumerate(agent_artists) if t[0] == agent_idx][0]
        agent_artists[idx] = (agent_idx, poly, txt, new_arr, info, color)
        if info is not None:
            info.set_position((x, y + half_width + 1))
            info.set_text(f'v={speed:.1f}m/s')

    # 更新虚线圆位置（跟随第一个激活agent）
    try:
        if horizon_circle is not None and len(active_indices) > 0:
            first_idx = int(active_indices[0])
            new_cx = float(simulator.agents_state[0, first_idx, 0].item())
            new_cy = float(simulator.agents_state[0, first_idx, 1].item())
            horizon_circle.center = (new_cx, new_cy)
    except Exception:
        pass

    fig.canvas.draw_idle()

def on_next_clicked(event):
    # 使用网络输出的分布采样动作并推进一步
    global features_tensor, first_agent_idx
    with torch.no_grad():
        logits = model.forward(features_tensor, mode="policy")
        dist = torch.distributions.Categorical(logits=logits)
        # 先显示当前步的动作概率分布
        try:
            probs = dist.probs.detach().to('cpu').numpy()  # (B, M, A)
            probs_first = probs[0, first_agent_idx]
            for i, b in enumerate(bars):
                b.set_height(float(probs_first[i]))
            ax_act.set_ylim(0.0, 1.0)
            fig_act.canvas.draw_idle()
        except Exception:
            pass
        actions = dist.sample()
    observation, reward, done = simulator.step(actions)
    # 显示当前观测agent的reward（B=0, M=first_agent_idx）
    try:
        cur_r = float(reward[0, first_agent_idx].item())
        print(f"当前观测agent(B=0, M={first_agent_idx}) reward: {cur_r:.4f}",'done:',done[0, first_agent_idx].item())
    except Exception:
        pass
    # 基于新观测重建特征，供下一步使用
    try:
        with torch.no_grad():
            agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(observation, config_ns)

            features_tensor = build_network_features(
                agents_state_dec,
                neighbors_local,
                w_lanes_local,
                w_boundaries_local,
                current_navigation(simulator),
                simulator.stop_lines if hasattr(simulator, 'stop_lines') else None,
                simulator.reward_calculator.sampled_params,
                config_ns, 
            )
            # 绘制局部要素，并打印本次 agents_state_dec
            try:
                draw_local_overlays(agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local)
                print('features_tensor:',features_tensor[0, first_agent_idx])
                #print('neighbors_local:',neighbors_local[0, first_agent_idx])
            except Exception:
                pass
    except Exception:
        pass
    refresh_agents()

btn_next.on_clicked(on_next_clicked)

# 绑定空格键为“下一步”
def on_key_press(event):
    try:
        if event.key in (' ', 'space'):
            on_next_clicked(event)
    except Exception:
        pass

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig_act.canvas.mpl_connect('key_press_event', on_key_press)

plt.tight_layout()
plt.show()

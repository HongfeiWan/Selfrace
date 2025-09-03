import os
import sys
import math
import json
from types import SimpleNamespace
from typing import Dict, List, Tuple
import random
import yaml
import torch
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加simulator目录到路径
simulator_dir = os.path.join(parent_dir, 'simulator')
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)
# 添加training目录到路径
training_dir = os.path.join(parent_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)
    
from simulator import TeraflowSimulator
from ddppo import decompose_observation, build_network_features
from network import create_network

class CarGame:
    def __init__(self):
        # 加载默认配置（不需要传入路径）
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default_config.yaml') 
        self.config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

        # 初始化pygame（可视化）
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("selfrace可视化训练")
        
        # 设备
        if not torch.cuda.is_available():
            raise RuntimeError("需要CUDA环境，请在GPU上运行。")
        self.device = torch.device('cuda:0')

        # 接入 TeraflowSimulator（统一管理路网/车辆/奖励/碰撞/离路等）
        if 'simulator' not in self.config:
            self.config['simulator'] = {}
        self.simulator = TeraflowSimulator(self.config, self.device)
        initial_observation = self.simulator.reset()
        # 构建网络并初始化特征
        self.config_ns = json.loads(json.dumps(self.config), object_hook=lambda d: SimpleNamespace(**d))
        self.model = create_network(config=self.config_ns, network_type="independent").to(self.device)
        self.model.eval()
        # 初始化特征（与训练保持一致）
        self.path_plan = self.simulator.agents_path_plans
        self.stop_lines = self.simulator.stop_lines

        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(initial_observation, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.path_plan,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )

        self.current_step_rewards = []
        self.agent_collision = False
        self.current_step_rewards.append(0.0)
        
        # 游戏状态
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.episode_reward = 0.0
        self.step_count = 0
        # 从配置读取最大步数：training.iteration

        self.max_steps = int(self.config.get('training').get('max_episode_length'))

        
        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # 颜色定义
        self.colors = {
            'road': (0, 0, 0),
            'grass': (255, 255, 255),
            'car': (255, 0, 0),
            'other_car': (0, 0, 255),  # 蓝色表示其他车辆
            'goal': (0, 255, 0),
            'text': (0, 0, 0),
            'lane_markings': (0, 255, 0),
            'dead_car': (255, 215, 0)
        }

    def _get_player_pose(self) -> Tuple[float, float, float]:
        """选择第一个仍然 alive 的智能体（active 且未 dead）的位姿；若无则沿用上一次视角或返回(0,0,0)。"""
        states = self.simulator.agents_state
        env0 = states[0]
        active_mask = env0[:, 6] > 0.5
        alive_mask = active_mask
        if hasattr(self, 'dead_mask_env0') and self.dead_mask_env0 is not None:
            alive_mask = active_mask & (~self.dead_mask_env0.to(active_mask.device))
        idx = torch.nonzero(alive_mask, as_tuple=False)
        if idx.numel() > 0:
            i = idx[0, 0].item()
            x = env0[i, 0].item()
            y = env0[i, 1].item()
            yaw = env0[i, 2].item()
            self._last_camera_pose = (x, y, yaw)
            return x, y, yaw
        if hasattr(self, '_last_camera_pose'):
            return self._last_camera_pose
        return 0.0, 0.0, 0.0

    def update_game_state(self):
        """使用 TeraflowSimulator 推进一步，并更新显示状态（env0）。"""
        if not hasattr(self, 'simulator') or self.simulator is None:
            return
        
        # 若处于暂停状态，直接返回（不更新步数/动作/奖励/状态）
        if hasattr(self, 'paused') and self.paused:
            return
        
        # 若所有agent都已done或active==0，则保持停止状态，不再推进
        try:
            env0_states_now_chk = self.simulator.agents_state[0]
            active_mask_now = (env0_states_now_chk[:, 6] > 0.5).detach().to('cpu')
            if hasattr(self, 'last_done_env0') and self.last_done_env0 is not None and self.last_done_env0.numel() == active_mask_now.numel():
                all_done_or_inactive = torch.all(self.last_done_env0 | (~active_mask_now))
                if bool(all_done_or_inactive.item()):
                    self.paused = True
                    return
        except Exception:
            pass

        # 使用网络输出动作分布并采样动作（与训练一致）
        with torch.no_grad():
            action_logits = self.model.forward(self.features_tensor, mode="policy")
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        # 保存本步动作以便UI显示
        self.last_actions = actions.detach().to('cpu')
        # 推进环境
        obs, reward, done = self.simulator.step(actions)
        # 更新特征以供下一个step使用
        agents_state_dec, neighbors_local, w_lanes_local, w_boundaries_local = decompose_observation(obs, self.config_ns)
        self.features_tensor = build_network_features(
            agents_state_dec,
            neighbors_local,
            w_lanes_local,
            w_boundaries_local,
            self.path_plan,
            self.stop_lines,
            self.simulator.reward_calculator.sampled_params,
            self.config_ns,
        )
        # 保存本step来自simulator的原始reward（env0，逐agent）
        self.last_reward_env0 = reward[0].detach().to('cpu')
        # 记录本步done用于下步是否暂停判定
        try:
            self.last_done_env0 = done[0].detach().to('cpu').bool()
        except Exception:
            self.last_done_env0 = None

        # 死亡（done）处理：标记并保存死亡时姿态
        env0_done = done[0].detach().to('cpu').bool()
        env0_states_now = self.simulator.agents_state[0].detach().to('cpu')
        M = env0_states_now.shape[0]
        if not hasattr(self, 'dead_mask_env0') or self.dead_mask_env0 is None or self.dead_mask_env0.numel() != M:
            self.dead_mask_env0 = torch.zeros(M, dtype=torch.bool)
            self.dead_pose_env0 = {}
        newly_dead = (~self.dead_mask_env0) & env0_done
        if newly_dead.any():
            dead_indices = torch.nonzero(newly_dead, as_tuple=False).squeeze(-1).tolist()
            for idx in dead_indices:
                x = float(env0_states_now[idx, 0].item())
                y = float(env0_states_now[idx, 1].item())
                yaw = float(env0_states_now[idx, 2].item())
                length = float(env0_states_now[idx, 4].item())
                width = float(env0_states_now[idx, 5].item())
                self.dead_pose_env0[idx] = (x, y, yaw, length, width)
        self.dead_mask_env0 = self.dead_mask_env0 | env0_done
        
        # 取env0中第一个激活体的reward作为当前step显示值
        env0_states = self.simulator.agents_state[0]
        active_mask_env0 = env0_states[:, 6] > 0.5
        if active_mask_env0.any():
            first_idx = int(torch.nonzero(active_mask_env0, as_tuple=False)[0].item())
            self.current_step_reward = float(self.last_reward_env0[first_idx].item())
        else:
            self.current_step_reward = 0.0
        self.step_count += 1


# ========================= 绘图 =================================

    def draw(self):
        """绘制游戏画面（从 TeraflowSimulator 读取状态与道路）。"""
        # 清屏
        self.screen.fill(self.colors['grass'])
        
        # 绘制道路
        self.draw_road_from_network()
        
        # 绘制目标（来自 simulator 的 agents_goal_quad_ids -> quad centers）
        try:
            goal_ids = getattr(self.simulator, 'agents_goal_quad_ids', None)
            if goal_ids is not None:
                env0_goal_ids = goal_ids[0]
                # 取玩家（第一个激活体）的目标
                env0_states = self.simulator.agents_state[0]
                active_mask = env0_states[:, 6] > 0.5
                if active_mask.any():
                    first_idx = int(torch.nonzero(active_mask, as_tuple=False)[0].item())
                    goal_quad_id = int(env0_goal_ids[first_idx].item())
                    if goal_quad_id >= 0:
                        quad_centers = self.simulator.road_network.quad_centerlines.mean(dim=1)
                        gx, gy = quad_centers[goal_quad_id]
                        goal_screen_pos = self.convert_world_to_screen(torch.tensor([[gx, gy]], dtype=torch.float32, device=self.device))
                        if goal_screen_pos:
                            pygame.draw.circle(self.screen, self.colors['goal'], goal_screen_pos[0], int(10.0))
                        else:
                            self.draw_goal_indicator()
        except Exception:
            pass
        
        # 绘制来自模拟器的智能体（env0内激活体）；死亡的车辆变色并停在死亡时姿态
        env0 = self.simulator.agents_state[0]
        active_mask = env0[:, 6] > 0.5
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1).tolist()
        for order, agent_idx in enumerate(active_indices):
            is_dead = hasattr(self, 'dead_mask_env0') and self.dead_mask_env0 is not None and agent_idx < self.dead_mask_env0.numel() and bool(self.dead_mask_env0[agent_idx].item())
            if is_dead and hasattr(self, 'dead_pose_env0') and agent_idx in self.dead_pose_env0:
                # 直接在此处按缓存姿态绘制（避免额外函数）
                x, y, yaw, length, width = self.dead_pose_env0[agent_idx]
                half_l = max(0.1, length * 0.5)
                half_w = max(0.1, width * 0.5)
                local = torch.tensor([
                    [-half_l, -half_w],
                    [ half_l, -half_w],
                    [ half_l,  half_w],
                    [-half_l,  half_w],
                ], dtype=torch.float32, device=self.device)
                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
                world = local @ rot.T + torch.tensor([x, y], dtype=torch.float32, device=self.device)
                pts = self.convert_world_to_screen(world)
                if len(pts) >= 4:
                    pygame.draw.polygon(self.screen, self.colors['dead_car'], pts, width=0)
            else:
                self.draw_car(agent_idx, is_player=(order == 0), color_override=None)
        # 绘制UI
        self.draw_ui()
        # 更新显示
        pygame.display.flip()
    
    def draw_road_from_network(self):
        """使用RoadNetwork数据绘制道路（带视野限制）"""
        # 获取汽车视野范围内的边界点
        visible_boundary_points = self.get_visible_boundary_points()
        if visible_boundary_points.shape[0] == 0:
            # 如果没有边界点，不绘制任何内容
            return
        # 将边界点转换为屏幕坐标
        screen_points = self.convert_world_to_screen(visible_boundary_points)
        # 绘制边界点作为散点
        self.draw_boundary_points(screen_points)
    
    def get_visible_boundary_points(self) -> torch.Tensor:
        """获取汽车视野范围内的边界点（优化版本）"""
        try:
            # 缓存视野点，避免每帧重复计算
            if not hasattr(self, '_cached_visible_points') or not hasattr(self, '_last_car_pos'):
                self._cached_visible_points = None
                self._last_car_pos = None
            # 检查汽车位置是否发生显著变化
            px, py, _ = self._get_player_pose()
            current_car_pos = (px, py)
            if (self._last_car_pos is not None and 
                abs(current_car_pos[0] - self._last_car_pos[0]) < 10.0 and 
                abs(current_car_pos[1] - self._last_car_pos[1]) < 10.0 and
                self._cached_visible_points is not None):
                return self._cached_visible_points
            # 获取所有边界点
            all_boundary_points = self.simulator.road_network.global_w_boundary_points
            if all_boundary_points.shape[0] == 0:
                return all_boundary_points
            # 使用更高效的距离计算
            car_pos = torch.tensor([[current_car_pos[0], current_car_pos[1]]], dtype=torch.float32, device=self.device)
            # 视野范围（米）
            vision_radius = 100.0

            # 使用更高效的距离计算和筛选
            diff = all_boundary_points - car_pos.squeeze(0)
            distances_squared = torch.sum(diff * diff, dim=1)
            visible_mask = distances_squared <= (vision_radius * vision_radius)
            visible_points = all_boundary_points[visible_mask]
            
            # 缓存结果
            self._cached_visible_points = visible_points
            self._last_car_pos = current_car_pos
            return visible_points
        
        except Exception as e:
            print(f"警告: 获取视野边界点失败: {e}")
            return torch.empty((0, 2), device=self.device)
    
    def draw_goal_indicator(self):
        """在屏幕边缘绘制目标指示器"""
        try:
            # 从模拟器获取玩家当前位置与目标位置
            px, py, _ = self._get_player_pose()
            goal_ids = getattr(self.simulator, 'agents_goal_quad_ids', None)
            
            env0_goal_ids = goal_ids[0]
            env0_states = self.simulator.agents_state[0]
            active_mask = env0_states[:, 6] > 0.5
            
            first_idx = int(torch.nonzero(active_mask, as_tuple=False)[0].item())
            goal_quad_id = int(env0_goal_ids[first_idx].item())
            
            quad_centers = self.simulator.road_network.quad_centerlines.mean(dim=1)
            gx, gy = quad_centers[goal_quad_id]
            dx = float(gx.item() - px)
            dy = float(gy.item() - py)
            # 计算方向角度
            angle = math.atan2(dy, dx)
            # 在屏幕边缘绘制指示器
            screen_center_x = self.width // 2
            screen_center_y = self.height // 2
            indicator_radius = min(self.width, self.height) // 2 - 30
            indicator_x = screen_center_x + int(indicator_radius * math.cos(angle))
            indicator_y = screen_center_y + int(indicator_radius * math.sin(angle))
            # 绘制指示器
            pygame.draw.circle(self.screen, (255, 255, 0), (indicator_x, indicator_y), 5)
        except Exception as e:
            print(f"警告: 绘制目标指示器失败: {e}")
    
    def draw_boundary_points(self, screen_points: List[Tuple[int, int]]):
        """绘制边界点作为散点（优化版本）"""
        if len(screen_points) == 0:
            return
        # 限制绘制的点数量以提高性能
        max_points = 500
        if len(screen_points) > max_points:
            # 均匀采样点
            step = len(screen_points) // max_points
            screen_points = screen_points[::step]
        # 绘制每个边界点
        point_radius = 1  # 点的大小
        point_color = self.colors['road']  # 道路颜色（黑色）
        for point in screen_points:
            pygame.draw.circle(self.screen, point_color, point, point_radius)
        
    def convert_world_to_screen(self, world_points: torch.Tensor) -> List[Tuple[int, int]]:
        """将世界坐标转换为屏幕坐标（基于汽车视野，优化版本）"""
        if world_points.shape[0] == 0:
            return []
        # 缓存转换参数，避免重复计算
        if not hasattr(self, '_cached_scale_params') or not hasattr(self, '_last_scale_car_pos'):
            self._cached_scale_params = None
            self._last_scale_car_pos = None
        # 检查汽车位置是否发生显著变化
        current_car_pos = self._get_player_pose()[:2]
        if (self._last_scale_car_pos is not None and 
            abs(current_car_pos[0] - self._last_scale_car_pos[0]) < 5.0 and 
            abs(current_car_pos[1] - self._last_scale_car_pos[1]) < 5.0 and
            self._cached_scale_params is not None):
            scale_x, scale_y, car_x, car_y = self._cached_scale_params
        else:
            # 获取汽车位置作为视野中心
            car_x, car_y, _ = self._get_player_pose()

            # 视野范围（米）
            vision_radius = 100.0
            
            # 计算缩放比例，留出边距
            margin = 50
            scale_x = (self.width - 2 * margin) / (2 * vision_radius)
            scale_y = (self.height - 2 * margin) / (2 * vision_radius)
            # 使用较小的缩放比例以保持宽高比
            scale = min(scale_x, scale_y)
            scale_x = scale_y = scale
            # 缓存参数
            self._cached_scale_params = (scale_x, scale_y, car_x, car_y)
            self._last_scale_car_pos = current_car_pos
        
        # 转换坐标（使用向量化操作）不翻转地图（世界y向上 => 屏幕y减小）
        screen_points = []
        for point in world_points:
            # 将世界坐标相对于汽车位置进行偏移
            relative_x = point[0].item() - car_x
            relative_y = point[1].item() - car_y
            # 转换为屏幕坐标
            screen_x = int(relative_x * scale_x + self.width // 2)
            screen_y = int(-relative_y * scale_y + self.height // 2)
            # 检查是否在屏幕范围内
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                screen_points.append((screen_x, screen_y))
        return screen_points
    
    def draw_car(self, agent_idx: int, is_player: bool = True, color_override: tuple = None):
        """根据 simulator.agents_state 的第5、6字段(长度/宽度)按朝向绘制车辆。"""
        env0 = self.simulator.agents_state[0]
        x = float(env0[agent_idx, 0].item())
        y = float(env0[agent_idx, 1].item())
        yaw = float(env0[agent_idx, 2].item())
        length = float(env0[agent_idx, 4].item())
        width = float(env0[agent_idx, 5].item())
        half_l = max(0.1, length * 0.5)
        half_w = max(0.1, width * 0.5)
        # 车辆局部坐标四角点
        local = torch.tensor([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w],
        ], dtype=torch.float32, device=self.device)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rot = torch.tensor([[cos_y, -sin_y],[sin_y, cos_y]], dtype=torch.float32, device=self.device)
        world = local @ rot.T + torch.tensor([x, y], dtype=torch.float32, device=self.device)
        pts = self.convert_world_to_screen(world)
        if len(pts) < 4:
            return
        color = color_override if color_override is not None else (self.colors['car'] if is_player else self.colors['other_car'])
        pygame.draw.polygon(self.screen, color, pts, width=0)
        # 朝向线
        center_screen = self.convert_world_to_screen(torch.tensor([[x, y]], dtype=torch.float32, device=self.device))
        front_world = torch.tensor([[x + half_l * cos_y, y + half_l * sin_y]], dtype=torch.float32, device=self.device)
        front_screen = self.convert_world_to_screen(front_world)
        if center_screen and front_screen:
            pygame.draw.line(self.screen, (255, 255, 0), center_screen[0], front_screen[0], 2)
  
    def draw_ui(self):
        """绘制用户界面"""
        # 展示当前step的reward（取env0第一个激活体）
        step_reward = getattr(self, 'current_step_reward', 0.0)
        reward_text = self.font.render(f"step reward: {step_reward:.4f}", True, self.colors['text'])
        step_text = self.font.render(f"step: {self.step_count}/{self.max_steps}", True, self.colors['text'])
        self.screen.blit(reward_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        # 不再输出列表

        # 显示玩家车辆状态
        px, py, pyaw = self._get_player_pose()
        # 从当前env0第一个激活体读取速度（agents_state[...,3]）
        env0_states = self.simulator.agents_state[0]
        active_mask_env0 = env0_states[:, 6] > 0.5
        if active_mask_env0.any():
            first_idx = int(torch.nonzero(active_mask_env0, as_tuple=False)[0].item())
            speed_val = float(env0_states[first_idx, 3].item())
        else:
            speed_val = 0.0
        speed_text = self.small_font.render(f"speed: {speed_val:.2f} m/s", True, self.colors['text'])
        heading_text = self.small_font.render(f"heading: {math.degrees(pyaw):.1f}°", True, self.colors['text'])
        
        # 显示离散动作信息（来自上一步采样的动作）
        if hasattr(self, 'last_actions'):
            actions_cpu = self.last_actions
            # 取env0第一个激活体动作索引
            env0_states = self.simulator.agents_state[0]
            active_mask_env0 = env0_states[:, 6] > 0.5
            if active_mask_env0.any():
                first_idx = int(torch.nonzero(active_mask_env0, as_tuple=False)[0].item())
                a_idx = int(actions_cpu[0, first_idx].item())

        action_text = self.small_font.render(f"current action idx: {a_idx}", True, self.colors['text'])
        
        self.screen.blit(speed_text, (10, 200))
        self.screen.blit(heading_text, (10, 220))
        self.screen.blit(action_text, (10, 260))
        
        # 显示汽车世界坐标
        car_world_pos = f"car world pos: ({px:.1f}, {py:.1f})"
        car_pos_text = self.small_font.render(car_world_pos, True, self.colors['text'])
        self.screen.blit(car_pos_text, (10, 300))
        
        # 显示目标世界坐标
        if hasattr(self, 'goal') and self.goal:
            goal_world_pos = f"goal world pos: ({self.goal['x']:.1f}, {self.goal['y']:.1f})"
            goal_pos_text = self.small_font.render(goal_world_pos, True, self.colors['text'])
            self.screen.blit(goal_pos_text, (10, 320))
        
        # 显示视野信息
        vision_info = f"vision radius: 100m, visible points: {len(self.get_visible_boundary_points())}"
        vision_text = self.small_font.render(vision_info, True, self.colors['text'])
        self.screen.blit(vision_text, (10, 340))
        
        # 显示车辆信息（env0 存活/死亡数量）
        env0 = self.simulator.agents_state[0]
        alive_num = int((env0[:, 6] > 0.5).sum().item())
        dead_num = int(self.dead_mask_env0.sum().item()) if hasattr(self, 'dead_mask_env0') and self.dead_mask_env0 is not None else 0
        car_info = f"alive: {alive_num}, dead: {dead_num}"
        car_text = self.small_font.render(car_info, True, self.colors['text'])
        self.screen.blit(car_text, (10, 360))
        
    def run(self):
        """运行游戏主循环（可视化）。"""
        print("可视化模式：每个step随机从12个离散动作中抽样")
        while self.running:
            # 处理事件，避免窗口无响应 别删
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
            # 推进模拟器并绘制
            self.update_game_state()
            # 绘制画面
            self.draw()
            # 帧率限制
            if self.clock is not None:
                self.clock.tick(60)
    
if __name__ == "__main__":
    game = CarGame()
    game.run()

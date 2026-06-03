import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_DIR = PROJECT_ROOT / "simulator"
TRAINING_DIR = PROJECT_ROOT / "training"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from simulator import TeraflowSimulator
from ddppo import build_features_from_simulator_state
from network import create_network


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "training" / "checkpoints_remote" / "latest.pt"


def dict_to_namespace(config: dict) -> SimpleNamespace:
    return json.loads(json.dumps(config), object_hook=lambda d: SimpleNamespace(**d))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_config(config_path: Path, device: torch.device, num_envs: int, num_agents: int) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simulator_cfg = config.setdefault("simulator", {})
    simulator_cfg["device"] = str(device)
    simulator_cfg["num_envs"] = int(num_envs)
    simulator_cfg["max_agents_num"] = int(num_agents)
    simulator_cfg["num_npc_vehicles"] = int(num_agents)
    simulator_cfg["verbose"] = False

    map_path = Path(simulator_cfg.get("map_path", ""))
    if map_path and not map_path.is_absolute():
        simulator_cfg["map_path"] = str((PROJECT_ROOT / map_path).resolve())

    training_cfg = config.setdefault("training", {})
    training_cfg["w_lane_dropout_prob"] = 0.0
    training_cfg["w_boundary_dropout_prob"] = 0.0
    profile_cfg = training_cfg.setdefault("profile", {})
    profile_cfg["enabled"] = False
    profile_cfg["cuda_sync"] = False
    return config


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}\n"
            "先从远端拉取最新模型，或用 --checkpoint 指向已有 .pt 文件。"
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.policy_network.load_state_dict(state["policy_state_dict"], strict=True)
        model.value_network.load_state_dict(state["value_state_dict"], strict=True)
        if "policy_feature_encoder_state_dict" in state:
            model.policy_feature_encoder.load_state_dict(state["policy_feature_encoder_state_dict"], strict=True)
        if "value_feature_encoder_state_dict" in state:
            model.value_feature_encoder.load_state_dict(state["value_feature_encoder_state_dict"], strict=True)
    return int(state.get("step", -1))


class InferenceGame:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = select_device(args.device)
        self.config = load_config(args.config, self.device, args.envs, args.agents)
        self.config_ns = dict_to_namespace(self.config)
        self.headless = bool(args.headless)
        self.deterministic = not bool(args.sample)
        self.max_steps = int(args.max_steps)
        self.current_world = 0
        self.selected_agent = 0
        self.step_count = 0
        self.episode_count = 0
        self.paused = False
        self.last_reward = None
        self.last_done = None
        self.cumulative_done = None
        self.last_actions = None
        self.last_values = None
        self.last_probs = None
        self.current_observation = None
        self.features_tensor = None
        self.show_all_waypoints = True
        self._visible_boundary_cache = None
        self._visible_boundary_key = None

        self.model = create_network(config=self.config_ns, network_type="independent")
        self.checkpoint_step = load_model_checkpoint(self.model, args.checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.simulator = TeraflowSimulator(self.config, self.device)
        self.reset_episode()

        self.pygame = None
        self.screen = None
        self.font = None
        self.small_font = None
        self.width = int(args.width)
        self.height = int(args.height)
        self.zoom_m = float(args.zoom)
        if not self.headless:
            self._init_pygame()

        print(
            "INFERENCE_READY",
            {
                "device": str(self.device),
                "checkpoint": str(args.checkpoint),
                "checkpoint_step": self.checkpoint_step,
                "envs": args.envs,
                "agents": args.agents,
                "mode": "argmax" if self.deterministic else "sample",
            },
            flush=True,
        )

    def _init_pygame(self):
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Selfrace checkpoint inference")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

    def reset_episode(self):
        self.simulator.reset(return_observation=False)
        self.current_observation = None
        self.step_count = 0
        self.episode_count += 1
        B, M, _ = self.simulator.agents_state.shape
        self.cumulative_done = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        self.last_reward = torch.zeros((B, M), dtype=torch.float32, device=self.device)
        self.last_done = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        self.last_actions = torch.zeros((B, M), dtype=torch.long, device=self.device)
        self.last_values = torch.zeros((B, M), dtype=torch.float32, device=self.device)
        self.last_probs = torch.zeros((B, M, self.config_ns.training.network.num_actions), dtype=torch.float32, device=self.device)
        self.current_world = min(self.current_world, B - 1)
        self.selected_agent = self._first_alive_agent(self.current_world)
        self._visible_boundary_cache = None
        self._visible_boundary_key = None
        self.features_tensor = self._build_features()

    def _build_features(self):
        return build_features_from_simulator_state(
            self.simulator,
            self.config_ns,
            alive_mask=self._alive_mask(),
            dropout_step=self.step_count,
        )

    def _alive_mask(self) -> torch.Tensor:
        active = self.simulator.agents_state[..., 6] > 0.5
        return active & (~self.cumulative_done)

    def _first_alive_agent(self, world_idx: int) -> int:
        alive = self._alive_mask()[world_idx]
        idx = torch.nonzero(alive, as_tuple=False)
        if idx.numel() > 0:
            return int(idx[0, 0].item())
        active = torch.nonzero(self.simulator.agents_state[world_idx, :, 6] > 0.5, as_tuple=False)
        return int(active[0, 0].item()) if active.numel() > 0 else 0

    def _select_next_agent(self):
        alive = self._alive_mask()[self.current_world].detach().cpu()
        candidates = torch.nonzero(alive, as_tuple=False).flatten().tolist()
        if not candidates:
            self.selected_agent = self._first_alive_agent(self.current_world)
            return
        bigger = [idx for idx in candidates if idx > self.selected_agent]
        self.selected_agent = bigger[0] if bigger else candidates[0]

    def _switch_world(self, delta: int):
        B = self.simulator.agents_state.shape[0]
        self.current_world = (self.current_world + delta) % B
        self.selected_agent = self._first_alive_agent(self.current_world)
        self._visible_boundary_cache = None
        self._visible_boundary_key = None

    def inference_step(self):
        if self._alive_mask().sum().item() == 0 or self.step_count >= self.max_steps:
            self.reset_episode()
            return

        with torch.inference_mode():
            action_logits, values = self.model(self.features_tensor, mode="both")
            probs = torch.softmax(action_logits, dim=-1)
            if self.deterministic:
                actions = torch.argmax(action_logits, dim=-1)
            else:
                actions = torch.distributions.Categorical(probs=probs).sample()

            alive = self._alive_mask()
            actions = torch.where(alive, actions, torch.zeros_like(actions))
            reward, done = self.simulator.step(actions, return_observation=False)

            self.last_actions = actions.detach()
            self.last_values = values.detach()
            self.last_probs = probs.detach()
            self.last_reward = reward.detach()
            self.last_done = done.detach().bool()
            self.cumulative_done = self.cumulative_done | self.last_done
            self.step_count += 1

            if self._alive_mask().sum().item() == 0:
                self.current_observation = None
                self.features_tensor = None
                return
            self.current_observation = None
            self.features_tensor = self._build_features()

    def run_headless(self):
        for _ in range(int(self.args.steps)):
            self.inference_step()
        self.print_summary()

    def run(self):
        while True:
            if not self._handle_events():
                break
            if not self.paused:
                self.inference_step()
            self.draw()
            self.clock.tick(int(self.args.fps))
        self.pygame.quit()

    def _handle_events(self) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_TAB:
                self._select_next_agent()
            elif event.key == pygame.K_r:
                self.reset_episode()
            elif event.key == pygame.K_m:
                self.deterministic = not self.deterministic
            elif event.key == pygame.K_w:
                self.show_all_waypoints = not self.show_all_waypoints
            elif event.key == pygame.K_LEFTBRACKET:
                self._switch_world(-1)
            elif event.key == pygame.K_RIGHTBRACKET:
                self._switch_world(1)
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self.zoom_m = max(30.0, self.zoom_m * 0.85)
                self._visible_boundary_cache = None
            elif event.key == pygame.K_MINUS:
                self.zoom_m = min(500.0, self.zoom_m * 1.15)
                self._visible_boundary_cache = None
        return True

    def _camera_pose(self):
        states = self.simulator.agents_state
        B, M, _ = states.shape
        b = min(max(self.current_world, 0), B - 1)
        m = min(max(self.selected_agent, 0), M - 1)
        state = states[b, m]
        if state[6] <= 0.5:
            m = self._first_alive_agent(b)
            self.selected_agent = m
            state = states[b, m]
        return float(state[0].item()), float(state[1].item()), float(state[2].item())

    def _world_to_screen(self, x: float, y: float, camera_xy: Optional[tuple[float, float]] = None):
        if camera_xy is None:
            camera_xy = self._camera_pose()[:2]
        scale = min(self.width, self.height) / (2.0 * self.zoom_m)
        sx = int((x - camera_xy[0]) * scale + self.width * 0.5)
        sy = int(-(y - camera_xy[1]) * scale + self.height * 0.5)
        return sx, sy

    def _polygon_points(self, x: float, y: float, yaw: float, length: float, width: float, camera_xy):
        half_l = max(0.1, length * 0.5)
        half_w = max(0.1, width * 0.5)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        local = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
        pts = []
        for lx, ly in local:
            wx = x + lx * cos_y - ly * sin_y
            wy = y + lx * sin_y + ly * cos_y
            pts.append(self._world_to_screen(wx, wy, camera_xy))
        return pts

    def draw(self):
        pygame = self.pygame
        self.screen.fill((245, 247, 250))
        camera_xy = self._camera_pose()[:2]
        self._draw_road(camera_xy)
        self._draw_navigation_targets(camera_xy)
        self._draw_agents(camera_xy)
        self._draw_action_probs()
        self._draw_ui()
        pygame.display.flip()

    def _draw_road(self, camera_xy):
        pygame = self.pygame
        points = self._visible_boundary_points(camera_xy)
        if points.numel() == 0:
            return
        pts = points.detach().cpu().tolist()
        step = max(1, len(pts) // 900)
        for x, y in pts[::step]:
            sx, sy = self._world_to_screen(float(x), float(y), camera_xy)
            if 0 <= sx < self.width and 0 <= sy < self.height:
                pygame.draw.circle(self.screen, (35, 35, 35), (sx, sy), 1)

    def _visible_boundary_points(self, camera_xy):
        key = (round(camera_xy[0] / 10.0), round(camera_xy[1] / 10.0), round(self.zoom_m))
        if self._visible_boundary_key == key and self._visible_boundary_cache is not None:
            return self._visible_boundary_cache
        all_points = self.simulator.road_network.global_w_boundary_points
        if all_points.numel() == 0:
            return all_points
        center = torch.tensor(camera_xy, dtype=all_points.dtype, device=all_points.device)
        diff = all_points - center
        visible = (diff.square().sum(dim=-1) <= (self.zoom_m * 1.35) ** 2)
        self._visible_boundary_cache = all_points[visible]
        self._visible_boundary_key = key
        return self._visible_boundary_cache

    def _draw_navigation_targets(self, camera_xy):
        pygame = self.pygame
        b = self.current_world
        if self.show_all_waypoints:
            active = (self.simulator.agents_state[b, :, 6] > 0.5).detach().cpu()
            done = self.cumulative_done[b].detach().cpu()
            for agent_idx in torch.nonzero(active & (~done), as_tuple=False).flatten().tolist():
                if agent_idx != self.selected_agent:
                    self._draw_agent_route_targets(camera_xy, b, int(agent_idx), selected=False)
        self._draw_agent_route_targets(camera_xy, b, self.selected_agent, selected=True)

        if hasattr(self.simulator, "goal_positions") and self.simulator.goal_positions is not None:
            gx, gy = self.simulator.goal_positions[b, self.selected_agent].detach().cpu().tolist()
            pygame.draw.circle(self.screen, (250, 188, 40), self._world_to_screen(float(gx), float(gy), camera_xy), 6)
        if hasattr(self.simulator, "final_goal_positions") and self.simulator.final_goal_positions is not None:
            gx, gy = self.simulator.final_goal_positions[b, self.selected_agent].detach().cpu().tolist()
            pygame.draw.circle(self.screen, (230, 90, 60), self._world_to_screen(float(gx), float(gy), camera_xy), 7, width=2)

    def _draw_agent_route_targets(self, camera_xy, world_idx: int, agent_idx: int, selected: bool):
        pygame = self.pygame
        route_points = self._remaining_route_points(world_idx, agent_idx)
        if not route_points:
            return
        for idx, (x, y) in enumerate(route_points):
            sx, sy = self._world_to_screen(x, y, camera_xy)
            if sx < -20 or sx > self.width + 20 or sy < -20 or sy > self.height + 20:
                continue
            is_current = idx == 0
            is_final = idx == len(route_points) - 1
            if selected:
                fill = (35, 115, 210) if not is_final else (230, 90, 60)
                radius = 6 if is_current else 5
                border = (245, 247, 250)
            else:
                fill = (100, 135, 170) if not is_final else (175, 110, 95)
                radius = 3 if is_current else 2
                border = (225, 230, 236)
            pygame.draw.circle(self.screen, fill, (sx, sy), radius)
            pygame.draw.circle(self.screen, border, (sx, sy), radius, width=1)
            if selected:
                label = self.small_font.render(str(idx + 1), True, (20, 45, 70))
                self.screen.blit(label, (sx + 6, sy - 6))

    def _remaining_route_points(self, world_idx: int, agent_idx: int):
        route_quads = getattr(self.simulator, "agents_route_quad_ids", None)
        target_count = getattr(self.simulator, "agents_route_target_count", None)
        current_idx = getattr(self.simulator, "agents_current_route_idx", None)
        if route_quads is None or target_count is None or current_idx is None:
            return []

        b = min(max(world_idx, 0), route_quads.shape[0] - 1)
        m = min(max(agent_idx, 0), route_quads.shape[1] - 1)
        start = int(current_idx[b, m].item())
        count = int(target_count[b, m].item())
        if count <= start:
            return []

        quads = route_quads[b, m, start:count].to(device=self.device, dtype=torch.long)
        quads = quads[quads >= 0]
        if quads.numel() == 0:
            return []
        centers = self.simulator.path_planner.get_quad_centers(quads).detach().cpu()
        return [(float(x), float(y)) for x, y in centers.tolist()]

    def _draw_agents(self, camera_xy):
        pygame = self.pygame
        b = self.current_world
        states = self.simulator.agents_state[b].detach().cpu()
        active = states[:, 6] > 0.5
        done = self.cumulative_done[b].detach().cpu()
        for m, state in enumerate(states):
            if not bool(active[m]):
                continue
            x, y, yaw, speed, length, width = [float(v) for v in state[:6].tolist()]
            if m == self.selected_agent:
                color = (225, 45, 55)
            elif bool(done[m]):
                color = (210, 156, 45)
            else:
                speed_t = min(1.0, max(0.0, speed / 20.0))
                color = (45, int(100 + 100 * speed_t), 220)
            pts = self._polygon_points(x, y, yaw, length, width, camera_xy)
            pygame.draw.polygon(self.screen, color, pts)
            cx, cy = self._world_to_screen(x, y, camera_xy)
            fx, fy = self._world_to_screen(x + math.cos(yaw) * length * 0.6, y + math.sin(yaw) * length * 0.6, camera_xy)
            pygame.draw.line(self.screen, (255, 245, 120), (cx, cy), (fx, fy), 2)

    def _draw_action_probs(self):
        b = self.current_world
        m = self.selected_agent
        if self.last_probs is None or b >= self.last_probs.shape[0] or m >= self.last_probs.shape[1]:
            return
        pygame = self.pygame
        probs = self.last_probs[b, m].detach().cpu().tolist()
        base_x = self.width - 310
        base_y = self.height - 120
        bar_w = 20
        gap = 4
        max_h = 80
        for i, prob in enumerate(probs):
            h = int(max_h * max(0.0, min(1.0, prob)))
            rect = pygame.Rect(base_x + i * (bar_w + gap), base_y + max_h - h, bar_w, h)
            color = (225, 45, 55) if self.last_actions is not None and int(self.last_actions[b, m].item()) == i else (90, 140, 210)
            pygame.draw.rect(self.screen, color, rect)
            label = self.small_font.render(str(i), True, (40, 40, 40))
            self.screen.blit(label, (base_x + i * (bar_w + gap) + 4, base_y + max_h + 4))

    def _draw_ui(self):
        b = self.current_world
        m = self.selected_agent
        states = self.simulator.agents_state[b]
        active = states[:, 6] > 0.5
        alive = self._alive_mask()[b]
        done = self.cumulative_done[b]
        speed = float(states[m, 3].item()) if m < states.shape[0] else 0.0
        reward = float(self.last_reward[b, m].item()) if self.last_reward is not None else 0.0
        value = float(self.last_values[b, m].item()) if self.last_values is not None else 0.0
        action = int(self.last_actions[b, m].item()) if self.last_actions is not None else 0
        entropy = 0.0
        if self.last_probs is not None:
            p = self.last_probs[b, m].detach()
            entropy = float((-(p * torch.log(p.clamp_min(1e-8))).sum()).item())

        lines = [
            f"ckpt step: {self.checkpoint_step}",
            f"device: {self.device} | {'argmax' if self.deterministic else 'sample'} | {'paused' if self.paused else 'running'}",
            f"episode: {self.episode_count}  step: {self.step_count}/{self.max_steps}",
            f"world {b}: alive {int(alive.sum().item())} / active {int(active.sum().item())} / done {int(done.sum().item())}",
            f"selected car: {m}  action: {action}  speed: {speed:.2f} m/s",
            f"reward: {reward:.4f}  value: {value:.4f}  entropy: {entropy:.3f}",
            f"waypoints: {'all' if self.show_all_waypoints else 'selected'}  zoom: {self.zoom_m:.0f} m",
        ]
        x, y = 14, 14
        for line in lines:
            surface = self.small_font.render(line, True, (25, 25, 25))
            self.screen.blit(surface, (x, y))
            y += 22

    def print_summary(self):
        active = self.simulator.agents_state[..., 6] > 0.5
        alive = self._alive_mask()
        done = self.cumulative_done
        mean_reward = float(self.last_reward[active].mean().item()) if bool(active.any().item()) else 0.0
        print(
            "INFERENCE_SUMMARY",
            {
                "checkpoint_step": self.checkpoint_step,
                "episode": self.episode_count,
                "step": self.step_count,
                "active": int(active.sum().item()),
                "alive": int(alive.sum().item()),
                "done": int(done.sum().item()),
                "last_mean_reward_active": round(mean_reward, 6),
            },
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Selfrace checkpoint inference in the pygame viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda:0 ...")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--zoom", type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        game = InferenceGame(args)
    except Exception as exc:
        if args.device == "auto" and select_device("auto").type == "mps":
            print(f"auto device mps failed, retrying on cpu: {exc}", flush=True)
            args.device = "cpu"
            game = InferenceGame(args)
        else:
            raise

    if args.headless:
        game.run_headless()
    else:
        game.run()


if __name__ == "__main__":
    main()

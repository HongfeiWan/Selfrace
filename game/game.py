import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_DIR = PROJECT_ROOT / "simulator"
TRAINING_DIR = PROJECT_ROOT / "training"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from ddppo import build_features_from_simulator_state
from network import create_network

from simulator import TeraflowSimulator

try:
    from .viewer import PygameViewer, RouteVisual, VehicleVisual, ViewerFrame
except ImportError:
    # ``python game/game.py`` puts the game directory, rather than the project
    # root, on sys.path.
    from viewer import PygameViewer, RouteVisual, VehicleVisual, ViewerFrame


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


def load_config(config_path: Path, num_envs: int, num_agents: int) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simulator_cfg = config.setdefault("simulator", {})
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
    CAMERA_POSITION_ALPHA = 0.22
    CAMERA_HEADING_ALPHA = 0.12

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = select_device(args.device)
        self.config = load_config(args.config, args.envs, args.agents)
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
        self.show_all_waypoints = False
        self.show_observed_boundaries = True
        # A north-up view is much easier to watch than rotating the entire
        # scene with every small steering correction from the observed car.
        self.camera_follows_heading = False
        self._smoothed_camera_pose = None
        self._camera_track_key = None
        self._road_boundary_segments = None
        self._visible_boundary_cache = None
        self._visible_boundary_key = None

        self.model = create_network(config=self.config_ns, network_type="independent")
        self.checkpoint_step = load_model_checkpoint(self.model, args.checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.simulator = TeraflowSimulator(self.config, self.device)
        self._road_boundary_segments = self._build_road_boundary_segments()
        self.action_values = (
            self.simulator.dynamics_model.discrete_action_space
            .get_all_actions()
            .detach()
            .cpu()
            .tolist()
        )
        self.reset_episode()

        self.pygame = None
        self.viewer = None
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
        self.viewer = PygameViewer(
            self.width,
            self.height,
            PROJECT_ROOT,
            title="Selfrace observation viewer",
        )
        self.pygame = self.viewer.pygame
        self.clock = self.viewer.clock
        self.width, self.height = self.viewer.width, self.viewer.height

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
        self._reset_camera_smoothing()
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

    def _build_road_boundary_segments(self) -> torch.Tensor:
        """Connect consecutive OOB samples into the continuous road edge."""
        road = self.simulator.road_network
        points = road.global_w_boundary_points
        if points.shape[0] >= 2:
            deltas = points[1:] - points[:-1]
            distances = torch.linalg.vector_norm(deltas, dim=-1)
            usable = distances[torch.isfinite(distances) & (distances > 1e-4)]
            if usable.numel() > 0:
                typical_spacing = float(torch.median(usable).item())
                link_distance = min(8.0, max(1.5, typical_spacing * 2.5))
                connected = (distances > 1e-4) & (distances <= link_distance)
                if bool(connected.any().item()):
                    return torch.stack((points[:-1][connected], points[1:][connected]), dim=1)

        # Older processed maps may not contain ordered OOB samples.  Their quad
        # sides still provide a useful road outline for the viewer.
        boundaries = []
        for name in ("left_boundaries", "right_boundaries"):
            value = getattr(road, name, None)
            if value is not None and value.numel() > 0:
                boundaries.append(value)
        if boundaries:
            return torch.cat(boundaries, dim=0)
        return torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)

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
            self._reset_camera_smoothing()
            return
        bigger = [idx for idx in candidates if idx > self.selected_agent]
        self.selected_agent = bigger[0] if bigger else candidates[0]
        self._reset_camera_smoothing()

    def _switch_world(self, delta: int):
        B = self.simulator.agents_state.shape[0]
        self.current_world = (self.current_world + delta) % B
        self.selected_agent = self._first_alive_agent(self.current_world)
        self._reset_camera_smoothing()
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
        self.viewer.close()

    def _handle_events(self) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.viewer.resize(event.w, event.h)
                self.width, self.height = self.viewer.width, self.viewer.height
                self._visible_boundary_cache = None
                self._visible_boundary_key = None
                continue
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
            elif event.key == pygame.K_b:
                self.show_observed_boundaries = not self.show_observed_boundaries
            elif event.key == pygame.K_c:
                self.camera_follows_heading = not self.camera_follows_heading
            elif event.key == pygame.K_LEFTBRACKET:
                self._switch_world(-1)
            elif event.key == pygame.K_RIGHTBRACKET:
                self._switch_world(1)
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.zoom_m = max(20.0, self.zoom_m * 0.85)
                self._visible_boundary_cache = None
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.zoom_m = min(500.0, self.zoom_m * 1.15)
                self._visible_boundary_cache = None
        return True

    def _reset_camera_smoothing(self):
        self._smoothed_camera_pose = None
        self._camera_track_key = None

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

        target_x = float(state[0].item())
        target_y = float(state[1].item())
        target_yaw = float(state[2].item()) if self.camera_follows_heading else 0.0
        track_key = (b, m)
        if self._smoothed_camera_pose is None or self._camera_track_key != track_key:
            self._smoothed_camera_pose = (target_x, target_y, target_yaw)
            self._camera_track_key = track_key
            return self._smoothed_camera_pose

        camera_x, camera_y, camera_yaw = self._smoothed_camera_pose
        position_alpha = self.CAMERA_POSITION_ALPHA
        heading_alpha = self.CAMERA_HEADING_ALPHA
        camera_x += (target_x - camera_x) * position_alpha
        camera_y += (target_y - camera_y) * position_alpha
        yaw_delta = math.atan2(
            math.sin(target_yaw - camera_yaw),
            math.cos(target_yaw - camera_yaw),
        )
        camera_yaw += yaw_delta * heading_alpha
        camera_yaw = math.atan2(math.sin(camera_yaw), math.cos(camera_yaw))
        self._smoothed_camera_pose = (camera_x, camera_y, camera_yaw)
        return self._smoothed_camera_pose

    def _visible_boundary_segments(self, camera_xy):
        viewport_key = None
        if self.viewer is not None:
            viewport_key = (self.viewer.scene_rect.width, self.viewer.scene_rect.height)
        key = (
            round(camera_xy[0] / 4.0),
            round(camera_xy[1] / 4.0),
            round(self.zoom_m / 2.0),
            viewport_key,
        )
        if self._visible_boundary_key == key and self._visible_boundary_cache is not None:
            return self._visible_boundary_cache

        segments = self._road_boundary_segments
        if segments is None or segments.numel() == 0:
            return []
        center = torch.tensor(camera_xy, dtype=segments.dtype, device=segments.device)
        midpoints = segments.mean(dim=1)
        aspect = self.viewer.scene_aspect if self.viewer is not None else 1.5
        visible_radius = self.zoom_m * max(1.6, aspect * 1.25)
        visible = (midpoints - center).square().sum(dim=-1) <= visible_radius ** 2
        self._visible_boundary_cache = [
            ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
            for start, end in segments[visible].detach().cpu().tolist()
        ]
        self._visible_boundary_key = key
        return self._visible_boundary_cache

    def _selected_boundary_observations(self):
        if not self.show_observed_boundaries:
            return [], int(self.simulator.observation_generator.num_w_boundaries)
        b = self.current_world
        m = self.selected_agent
        state = self.simulator.agents_state[b:b + 1, m:m + 1].contiguous()
        with torch.inference_mode():
            points, point_ids, _ = self.simulator.observation_generator.get_w_boundary_observation_for_agents(state)
        points = points[0, 0]
        valid = point_ids[0, 0] >= 0
        visible_points = [
            (float(x), float(y))
            for x, y in points[valid].detach().cpu().tolist()
        ]
        return visible_points, int(point_ids.shape[-1])

    def _route_visuals(self):
        routes = []
        b = self.current_world
        if self.show_all_waypoints:
            active = (self.simulator.agents_state[b, :, 6] > 0.5).detach().cpu()
            done = self.cumulative_done[b].detach().cpu()
            for agent_idx in torch.nonzero(active & (~done), as_tuple=False).flatten().tolist():
                if agent_idx != self.selected_agent:
                    points = self._remaining_route_points(b, int(agent_idx))
                    if points:
                        routes.append(RouteVisual(points=points, selected=False))
        selected_points = self._remaining_route_points(b, self.selected_agent)
        if selected_points:
            routes.append(RouteVisual(points=selected_points, selected=True))
        return routes

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

    def _vehicle_visuals(self):
        b = self.current_world
        states = self.simulator.agents_state[b].detach().cpu()
        active = states[:, 6] > 0.5
        done = self.cumulative_done[b].detach().cpu()
        vehicles = []
        for m, state in enumerate(states):
            if not bool(active[m]):
                continue
            x, y, yaw, speed, length, width = [float(v) for v in state[:6].tolist()]
            vehicles.append(VehicleVisual(
                index=m,
                x=x,
                y=y,
                yaw=yaw,
                speed=speed,
                length=length,
                width=width,
                selected=m == self.selected_agent,
                done=bool(done[m]),
            ))
        return vehicles

    def _goal_point(self, name: str):
        values = getattr(self.simulator, name, None)
        if values is None:
            return None
        x, y = values[self.current_world, self.selected_agent].detach().cpu().tolist()
        return float(x), float(y)

    def _frame_info(self):
        b = self.current_world
        m = self.selected_agent
        states = self.simulator.agents_state[b]
        active = states[:, 6] > 0.5
        alive = self._alive_mask()[b]
        speed = float(states[m, 3].item()) if m < states.shape[0] else 0.0
        reward = float(self.last_reward[b, m].item()) if self.last_reward is not None else 0.0
        value = float(self.last_values[b, m].item()) if self.last_values is not None else 0.0
        entropy = 0.0
        if self.last_probs is not None:
            p = self.last_probs[b, m].detach()
            entropy = float((-(p * torch.log(p.clamp_min(1e-8))).sum()).item())
        rows = [
            ("checkpoint", f"{self.checkpoint_step:,}"),
            ("device / mode", f"{self.device} / {'argmax' if self.deterministic else 'sample'}"),
            ("episode / step", f"{self.episode_count} / {self.step_count}:{self.max_steps}"),
            ("world / alive-active", f"{b} / {int(alive.sum().item())}:{int(active.sum().item())}"),
            ("observed car", f"#{m}  {speed:.2f} m/s"),
            ("reward / value", f"{reward:+.3f} / {value:+.3f}"),
            ("entropy / zoom", f"{entropy:.3f} / {self.zoom_m:.0f} m"),
        ]
        return rows

    def draw(self):
        camera_pose = self._camera_pose()
        observations, observation_capacity = self._selected_boundary_observations()
        b = self.current_world
        m = self.selected_agent
        probabilities = []
        if self.last_probs is not None and b < self.last_probs.shape[0] and m < self.last_probs.shape[1]:
            probabilities = [float(value) for value in self.last_probs[b, m].detach().cpu().tolist()]
        action = int(self.last_actions[b, m].item()) if self.last_actions is not None else 0
        frame = ViewerFrame(
            camera_pose=camera_pose,
            zoom_m=self.zoom_m,
            road_segments=self._visible_boundary_segments(camera_pose[:2]),
            boundary_observations=observations,
            observation_capacity=observation_capacity,
            vehicles=self._vehicle_visuals(),
            action_probabilities=probabilities,
            action_values=self.action_values,
            selected_action=action,
            info_rows=self._frame_info(),
            routes=self._route_visuals(),
            goal_point=self._goal_point("goal_positions"),
            final_goal_point=self._goal_point("final_goal_positions"),
            paused=self.paused,
            camera_aligned=self.camera_follows_heading,
        )
        self.viewer.draw(frame)

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
    parser.add_argument("--zoom", type=float, default=55.0, help="vertical half-span of the road view in metres")
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

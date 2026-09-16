"""Pygame rendering for the Selfrace inference viewer.

This module deliberately has no torch dependency.  The simulator adapter in
``game.py`` converts tensors to the small, immutable visual records below, so
the Windows UI can be rendered and tested without loading a checkpoint.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]
Segment = tuple[Point, Point]
Color = tuple[int, int, int]


@dataclass(frozen=True)
class VehicleVisual:
    index: int
    x: float
    y: float
    yaw: float
    speed: float
    length: float
    width: float
    selected: bool = False
    done: bool = False


@dataclass(frozen=True)
class RouteVisual:
    points: Sequence[Point]
    selected: bool = False


@dataclass(frozen=True)
class ViewerFrame:
    camera_pose: tuple[float, float, float]
    zoom_m: float
    road_segments: Sequence[Segment]
    boundary_observations: Sequence[Point]
    observation_capacity: int
    vehicles: Sequence[VehicleVisual]
    action_probabilities: Sequence[float]
    action_values: Sequence[Sequence[float]]
    selected_action: int
    info_rows: Sequence[tuple[str, str]]
    routes: Sequence[RouteVisual] = field(default_factory=tuple)
    goal_point: Point | None = None
    final_goal_point: Point | None = None
    paused: bool = False
    camera_aligned: bool = True


class Palette:
    BACKGROUND: Color = (250, 250, 249)
    ROAD_SHADOW: Color = (82, 83, 88)
    ROAD: Color = (126, 127, 133)
    ROAD_HIGHLIGHT: Color = (151, 152, 157)
    EGO: Color = (24, 164, 86)
    EGO_DARK: Color = (10, 105, 55)
    TRAFFIC: Color = (213, 178, 77)
    TRAFFIC_DARK: Color = (143, 112, 39)
    DONE: Color = (194, 132, 70)
    WINDOW: Color = (24, 123, 132)
    WINDOW_LIGHT: Color = (55, 169, 174)
    OBS_GLOW: Color = (171, 246, 255)
    OBS_POINT: Color = (18, 202, 231)
    OBS_CORE: Color = (242, 254, 255)
    ROUTE: Color = (69, 124, 190)
    ROUTE_OTHER: Color = (164, 181, 200)
    GOAL: Color = (238, 178, 44)
    FINAL_GOAL: Color = (225, 79, 62)
    PANEL: Color = (241, 244, 247)
    CARD: Color = (255, 255, 255)
    CARD_BORDER: Color = (218, 224, 230)
    TEXT: Color = (29, 37, 48)
    MUTED: Color = (99, 111, 124)
    FAINT: Color = (151, 161, 172)
    ACCENT: Color = (32, 113, 181)


def enable_windows_dpi_awareness() -> None:
    """Keep pygame pixels crisp when Windows display scaling is enabled."""
    if os.name != "nt":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (ImportError, AttributeError, OSError):
        # DPI awareness is cosmetic; an older Windows build should still run.
        pass


class PygameViewer:
    MIN_WIDTH = 960
    MIN_HEIGHT = 680

    def __init__(
        self,
        width: int,
        height: int,
        project_root: Path,
        title: str = "Selfrace observation viewer",
    ):
        enable_windows_dpi_awareness()
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

        self.pygame = importlib.import_module("pygame")
        self.pygame.init()
        self.pygame.font.init()
        self.flags = self.pygame.RESIZABLE | self.pygame.DOUBLEBUF
        self.width = max(self.MIN_WIDTH, int(width))
        self.height = max(self.MIN_HEIGHT, int(height))
        self.screen = self.pygame.display.set_mode(
            (self.width, self.height), self.flags
        )
        self.pygame.display.set_caption(title)
        self.clock = self.pygame.time.Clock()

        self.title_font = self._load_font(23, bold=True)
        self.heading_font = self._load_font(15, bold=True)
        self.body_font = self._load_font(15)
        self.small_font = self._load_font(13)
        self.micro_font = self._load_font(11)

        logo_path = project_root / "images" / "Logo.png"
        if logo_path.exists():
            try:
                icon = self.pygame.image.load(str(logo_path))
                self.pygame.display.set_icon(icon)
            except self.pygame.error:
                pass

    def _load_font(self, size: int, bold: bool = False):
        pygame = self.pygame
        font_path = (
            pygame.font.match_font("segoeui")
            or pygame.font.match_font("microsoftyaheiui")
            or pygame.font.match_font("arial")
        )
        font = (
            pygame.font.Font(font_path, size)
            if font_path
            else pygame.font.Font(None, size)
        )
        font.set_bold(bold)
        return font

    @property
    def sidebar_width(self) -> int:
        return max(310, min(356, round(self.width * 0.28)))

    @property
    def scene_rect(self):
        return self.pygame.Rect(0, 0, self.width - self.sidebar_width, self.height)

    @property
    def panel_rect(self):
        return self.pygame.Rect(
            self.width - self.sidebar_width, 0, self.sidebar_width, self.height
        )

    @property
    def scene_aspect(self) -> float:
        rect = self.scene_rect
        return rect.width / max(1.0, float(rect.height))

    def resize(self, width: int, height: int) -> None:
        self.width = max(self.MIN_WIDTH, int(width))
        self.height = max(self.MIN_HEIGHT, int(height))
        self.screen = self.pygame.display.set_mode(
            (self.width, self.height), self.flags
        )

    def sync_size(self) -> None:
        self.width, self.height = self.screen.get_size()

    def close(self) -> None:
        self.pygame.quit()

    def save(self, path: str | Path) -> None:
        self.pygame.image.save(self.screen, str(path))

    def scale_for_zoom(self, zoom_m: float) -> float:
        return self.scene_rect.height / max(1.0, 2.0 * float(zoom_m))

    def world_to_screen(self, point: Point, frame: ViewerFrame) -> tuple[int, int]:
        camera_x, camera_y, camera_yaw = frame.camera_pose
        dx = float(point[0]) - camera_x
        dy = float(point[1]) - camera_y
        if frame.camera_aligned:
            cos_yaw = math.cos(camera_yaw)
            sin_yaw = math.sin(camera_yaw)
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
        else:
            local_x, local_y = dx, dy

        scene = self.scene_rect
        scale = self.scale_for_zoom(frame.zoom_m)
        origin_x = scene.left + int(scene.width * 0.42)
        origin_y = scene.centery
        return round(origin_x + local_x * scale), round(origin_y - local_y * scale)

    def draw(self, frame: ViewerFrame) -> None:
        self.sync_size()
        self.screen.fill(Palette.BACKGROUND)
        self.screen.set_clip(self.scene_rect)
        self._draw_road(frame)
        self._draw_routes(frame)
        self._draw_boundary_observations(frame)
        self._draw_vehicles(frame)
        self._draw_scene_legend(frame)
        if frame.paused:
            self._draw_pause_badge()
        self.screen.set_clip(None)
        self._draw_sidebar(frame)
        self.pygame.display.flip()

    def _draw_road(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        scale = self.scale_for_zoom(frame.zoom_m)
        inner_width = max(4, min(11, round(scale * 0.72)))
        outer_width = inner_width + 3
        scene = self.scene_rect.inflate(40, 40)

        transformed = []
        for start, end in frame.road_segments:
            p1 = self.world_to_screen(start, frame)
            p2 = self.world_to_screen(end, frame)
            if not (scene.clipline(p1, p2)):
                continue
            transformed.append((p1, p2))

        for p1, p2 in transformed:
            pygame.draw.line(self.screen, Palette.ROAD_SHADOW, p1, p2, outer_width)
        for p1, p2 in transformed:
            pygame.draw.line(self.screen, Palette.ROAD, p1, p2, inner_width)
        if inner_width >= 7:
            for p1, p2 in transformed:
                pygame.draw.line(self.screen, Palette.ROAD_HIGHLIGHT, p1, p2, 1)

    def _draw_routes(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        scene = self.scene_rect.inflate(20, 20)
        for route in frame.routes:
            points = [self.world_to_screen(p, frame) for p in route.points]
            points = [p for p in points if scene.collidepoint(p)]
            if not points:
                continue
            color = Palette.ROUTE if route.selected else Palette.ROUTE_OTHER
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 1)
            radius = 4 if route.selected else 2
            for point in points:
                pygame.draw.circle(self.screen, color, point, radius)

        if frame.goal_point is not None:
            point = self.world_to_screen(frame.goal_point, frame)
            pygame.draw.circle(self.screen, Palette.GOAL, point, 7)
            pygame.draw.circle(self.screen, (255, 250, 226), point, 7, 2)
        if frame.final_goal_point is not None:
            point = self.world_to_screen(frame.final_goal_point, frame)
            pygame.draw.circle(self.screen, Palette.FINAL_GOAL, point, 8, 2)

    def _draw_boundary_observations(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        scene = self.scene_rect.inflate(-2, -2)
        scale = self.scale_for_zoom(frame.zoom_m)
        glow_radius = max(3, min(6, round(scale * 0.55)))
        point_radius = max(2, glow_radius // 2)
        for point_world in frame.boundary_observations:
            point = self.world_to_screen(point_world, frame)
            if not scene.collidepoint(point):
                continue
            pygame.draw.circle(self.screen, Palette.OBS_GLOW, point, glow_radius)
            pygame.draw.circle(self.screen, Palette.OBS_POINT, point, point_radius)
            pygame.draw.circle(self.screen, Palette.OBS_CORE, point, 1)

    @staticmethod
    def _world_box(
        center_x: float, center_y: float, yaw: float, length: float, width: float
    ) -> list[Point]:
        half_length = max(0.05, length * 0.5)
        half_width = max(0.05, width * 0.5)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ]
        return [
            (
                center_x + local_x * cos_yaw - local_y * sin_yaw,
                center_y + local_x * sin_yaw + local_y * cos_yaw,
            )
            for local_x, local_y in corners
        ]

    def _local_box(
        self,
        vehicle: VehicleVisual,
        local_x: float,
        local_y: float,
        length: float,
        width: float,
        frame: ViewerFrame,
    ) -> list[tuple[int, int]]:
        cos_yaw = math.cos(vehicle.yaw)
        sin_yaw = math.sin(vehicle.yaw)
        center_x = vehicle.x + local_x * cos_yaw - local_y * sin_yaw
        center_y = vehicle.y + local_x * sin_yaw + local_y * cos_yaw
        return [
            self.world_to_screen(p, frame)
            for p in self._world_box(center_x, center_y, vehicle.yaw, length, width)
        ]

    def _draw_vehicles(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        for vehicle in frame.vehicles:
            body_world = self._world_box(
                vehicle.x, vehicle.y, vehicle.yaw, vehicle.length, vehicle.width
            )
            body = [self.world_to_screen(point, frame) for point in body_world]
            if not self.scene_rect.inflate(40, 40).collidepoint(
                self.world_to_screen((vehicle.x, vehicle.y), frame)
            ):
                continue

            if vehicle.selected:
                halo_world = self._world_box(
                    vehicle.x,
                    vehicle.y,
                    vehicle.yaw,
                    vehicle.length + 0.5,
                    vehicle.width + 0.45,
                )
                halo = [self.world_to_screen(point, frame) for point in halo_world]
                pygame.draw.polygon(self.screen, (197, 241, 216), halo)

            shadow = [(x + 2, y + 3) for x, y in body]
            pygame.draw.polygon(self.screen, (188, 190, 191), shadow)

            if vehicle.selected:
                fill, border = Palette.EGO, Palette.EGO_DARK
            elif vehicle.done:
                fill, border = Palette.DONE, Palette.TRAFFIC_DARK
            else:
                fill, border = Palette.TRAFFIC, Palette.TRAFFIC_DARK
            pygame.draw.polygon(self.screen, fill, body)
            pygame.draw.aalines(self.screen, border, True, body)
            if vehicle.selected:
                pygame.draw.lines(self.screen, border, True, body, 2)

            if vehicle.length * self.scale_for_zoom(frame.zoom_m) >= 15:
                window_length = max(0.16, vehicle.length * 0.17)
                window_width = max(0.12, vehicle.width * 0.27)
                for local_x in (-vehicle.length * 0.12, vehicle.length * 0.14):
                    for local_y in (-vehicle.width * 0.25, vehicle.width * 0.25):
                        window = self._local_box(
                            vehicle,
                            local_x,
                            local_y,
                            window_length,
                            window_width,
                            frame,
                        )
                        pygame.draw.polygon(self.screen, Palette.WINDOW, window)
                        pygame.draw.aalines(
                            self.screen, Palette.WINDOW_LIGHT, True, window
                        )

                for local_y in (-vehicle.width * 0.28, vehicle.width * 0.28):
                    cos_yaw = math.cos(vehicle.yaw)
                    sin_yaw = math.sin(vehicle.yaw)
                    local_x = vehicle.length * 0.46
                    light_world = (
                        vehicle.x + local_x * cos_yaw - local_y * sin_yaw,
                        vehicle.y + local_x * sin_yaw + local_y * cos_yaw,
                    )
                    pygame.draw.circle(
                        self.screen,
                        (255, 239, 151),
                        self.world_to_screen(light_world, frame),
                        2,
                    )

    def _draw_scene_legend(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        rect = pygame.Rect(
            self.scene_rect.left + 14, self.scene_rect.bottom - 45, 368, 31
        )
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            surface, (255, 255, 255, 224), surface.get_rect(), border_radius=9
        )
        pygame.draw.rect(
            surface, (214, 220, 226, 230), surface.get_rect(), 1, border_radius=9
        )
        self.screen.blit(surface, rect.topleft)

        y = rect.centery
        items = [
            (Palette.EGO, "observed car", "square"),
            (Palette.TRAFFIC, "traffic", "square"),
            (Palette.OBS_POINT, "observed roadside point", "circle"),
        ]
        x = rect.left + 12
        for color, label, shape in items:
            if shape == "circle":
                pygame.draw.circle(self.screen, color, (x + 5, y), 4)
            else:
                pygame.draw.rect(
                    self.screen, color, (x, y - 5, 10, 10), border_radius=2
                )
            x += 15
            text = self.small_font.render(label, True, Palette.MUTED)
            self.screen.blit(text, (x, y - text.get_height() // 2))
            x += text.get_width() + 15

    def _draw_pause_badge(self) -> None:
        pygame = self.pygame
        label = self.heading_font.render("PAUSED", True, Palette.TEXT)
        rect = label.get_rect(center=(self.scene_rect.centerx, 32)).inflate(26, 13)
        pygame.draw.rect(self.screen, Palette.CARD, rect, border_radius=12)
        pygame.draw.rect(self.screen, Palette.CARD_BORDER, rect, 1, border_radius=12)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_card(self, rect) -> None:
        pygame = self.pygame
        shadow = rect.move(0, 2)
        pygame.draw.rect(self.screen, (226, 230, 234), shadow, border_radius=10)
        pygame.draw.rect(self.screen, Palette.CARD, rect, border_radius=10)
        pygame.draw.rect(self.screen, Palette.CARD_BORDER, rect, 1, border_radius=10)

    def _draw_sidebar(self, frame: ViewerFrame) -> None:
        pygame = self.pygame
        panel = self.panel_rect
        pygame.draw.rect(self.screen, Palette.PANEL, panel)
        pygame.draw.line(
            self.screen, Palette.CARD_BORDER, panel.topleft, panel.bottomleft, 1
        )
        margin = 16
        left = panel.left + margin
        content_width = panel.width - margin * 2

        title = self.title_font.render("SELFRACE", True, Palette.TEXT)
        self.screen.blit(title, (left, 15))
        subtitle = self.micro_font.render(
            "INFERENCE / OBSERVATION VIEW", True, Palette.MUTED
        )
        self.screen.blit(subtitle, (left, 43))
        status_text = "PAUSED" if frame.paused else "RUNNING"
        status_color = Palette.GOAL if frame.paused else Palette.EGO
        status = self.micro_font.render(status_text, True, status_color)
        status_rect = status.get_rect(topright=(panel.right - margin, 22)).inflate(
            12, 8
        )
        pygame.draw.rect(self.screen, Palette.CARD, status_rect, border_radius=9)
        pygame.draw.rect(self.screen, status_color, status_rect, 1, border_radius=9)
        self.screen.blit(status, status.get_rect(center=status_rect.center))

        stats_height = 30 + len(frame.info_rows) * 20
        stats_rect = pygame.Rect(left, 67, content_width, stats_height)
        self._draw_card(stats_rect)
        heading = self.heading_font.render("TELEMETRY", True, Palette.TEXT)
        self.screen.blit(heading, (stats_rect.left + 12, stats_rect.top + 9))
        y = stats_rect.top + 31
        for label, value in frame.info_rows:
            label_surface = self.micro_font.render(
                str(label).upper(), True, Palette.FAINT
            )
            value_surface = self.small_font.render(str(value), True, Palette.TEXT)
            self.screen.blit(label_surface, (stats_rect.left + 12, y + 1))
            self.screen.blit(
                value_surface,
                (stats_rect.right - 12 - value_surface.get_width(), y),
            )
            y += 20

        obs_rect = pygame.Rect(left, stats_rect.bottom + 10, content_width, 72)
        self._draw_card(obs_rect)
        pygame.draw.circle(
            self.screen, Palette.OBS_GLOW, (obs_rect.left + 18, obs_rect.top + 22), 7
        )
        pygame.draw.circle(
            self.screen, Palette.OBS_POINT, (obs_rect.left + 18, obs_rect.top + 22), 4
        )
        obs_title = self.heading_font.render(
            "W_BOUNDARY OBSERVATION", True, Palette.TEXT
        )
        self.screen.blit(obs_title, (obs_rect.left + 32, obs_rect.top + 13))
        obs_count = len(frame.boundary_observations)
        count_text = self.body_font.render(
            f"{obs_count} / {max(0, int(frame.observation_capacity))} points",
            True,
            Palette.ACCENT,
        )
        self.screen.blit(count_text, (obs_rect.left + 12, obs_rect.top + 40))
        camera_text = "ego-aligned" if frame.camera_aligned else "north-up"
        camera_surface = self.micro_font.render(camera_text, True, Palette.MUTED)
        self.screen.blit(
            camera_surface,
            (obs_rect.right - 12 - camera_surface.get_width(), obs_rect.top + 45),
        )

        controls_height = 124 if self.height >= 720 else 94
        controls_rect = pygame.Rect(
            left,
            panel.bottom - controls_height - 14,
            content_width,
            controls_height,
        )
        action_top = obs_rect.bottom + 11
        action_bottom = controls_rect.top - 11
        self._draw_action_grid(
            frame,
            pygame.Rect(left, action_top, content_width, action_bottom - action_top),
        )
        self._draw_controls(controls_rect)

    def _draw_action_grid(self, frame: ViewerFrame, rect) -> None:
        pygame = self.pygame
        if rect.height < 80:
            return
        heading = self.heading_font.render("ACTION POLICY", True, Palette.TEXT)
        self.screen.blit(heading, rect.topleft)
        hint = self.micro_font.render(
            "longitudinal / lateral jerk (m/s^3)", True, Palette.MUTED
        )
        self.screen.blit(hint, (rect.left, rect.top + 19))

        probabilities = [
            max(0.0, min(1.0, float(value))) for value in frame.action_probabilities
        ]
        if not probabilities:
            return
        count = len(probabilities)
        columns = 3 if count == 12 else min(4, max(1, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        gap = 6
        grid_top = rect.top + 37
        available_height = max(1, rect.bottom - grid_top)
        cell_width = (rect.width - gap * (columns - 1)) // columns
        cell_height = min(48, (available_height - gap * (rows - 1)) // rows)
        if cell_height < 25:
            return
        max_probability = max(probabilities) or 1.0

        for index, probability in enumerate(probabilities):
            row, column = divmod(index, columns)
            cell = pygame.Rect(
                rect.left + column * (cell_width + gap),
                grid_top + row * (cell_height + gap),
                cell_width,
                cell_height,
            )
            strength = probability / max_probability
            if index == frame.selected_action:
                fill = (218, 243, 229)
                border = Palette.EGO
                border_width = 2
            else:
                fill = (
                    int(247 - 24 * strength),
                    int(249 - 17 * strength),
                    int(251 - 4 * strength),
                )
                border = Palette.CARD_BORDER
                border_width = 1
            pygame.draw.rect(self.screen, fill, cell, border_radius=7)
            pygame.draw.rect(self.screen, border, cell, border_width, border_radius=7)

            top_text = self.micro_font.render(
                f"#{index}  {probability * 100:4.1f}%", True, Palette.TEXT
            )
            self.screen.blit(top_text, (cell.left + 6, cell.top + 5))
            if (
                index < len(frame.action_values)
                and len(frame.action_values[index]) >= 2
                and cell_height >= 38
            ):
                along = float(frame.action_values[index][0])
                lateral = float(frame.action_values[index][1])
                value_text = self.micro_font.render(
                    f"{along:+g} / {lateral:+g}", True, Palette.MUTED
                )
                self.screen.blit(
                    value_text,
                    (cell.left + 6, cell.bottom - value_text.get_height() - 5),
                )

    def _draw_controls(self, rect) -> None:
        self._draw_card(rect)
        heading = self.heading_font.render("CONTROLS", True, Palette.TEXT)
        self.screen.blit(heading, (rect.left + 12, rect.top + 9))
        shortcuts = [
            "Space  pause      Tab  next car",
            "[ / ]  world      + / -  zoom",
            "M  policy mode    C  camera",
            "W  route scope    B  boundary points",
            "R  reset          Esc  quit",
        ]
        max_lines = 5 if rect.height >= 120 else 3
        y = rect.top + 32
        for line in shortcuts[:max_lines]:
            surface = self.micro_font.render(line, True, Palette.MUTED)
            self.screen.blit(surface, (rect.left + 12, y))
            y += 17

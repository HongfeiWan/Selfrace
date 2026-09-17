"""Canonical layout for policy/value network input features.

The schema is shared by feature construction and the neural-network encoder so
that group order, dimensions, element widths, and padding semantics cannot
silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FEATURE_PAD_VALUE = -2.0

SIMPLE_GROUP_NAMES = ("state", "goal", "reward", "vehicle_style")
SET_GROUP_NAMES = ("road_boundary", "lane_points", "stop_lines", "other_agents")


def _get(container: Any, name: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    flat_dim: int
    element_dim: int | None = None
    active_channel: int | None = None

    def __post_init__(self) -> None:
        if self.flat_dim <= 0:
            raise ValueError(f"feature group {self.name!r} must have flat_dim > 0")
        if self.element_dim is not None:
            if self.element_dim <= 0:
                raise ValueError(f"feature group {self.name!r} must have element_dim > 0")
            if self.flat_dim % self.element_dim != 0:
                raise ValueError(
                    f"feature group {self.name!r}: flat_dim={self.flat_dim} is not "
                    f"divisible by element_dim={self.element_dim}"
                )
        if self.active_channel is not None:
            if self.element_dim is None:
                raise ValueError(f"feature group {self.name!r} has an active channel but no element_dim")
            resolved = self.resolved_active_channel
            if resolved < 0 or resolved >= self.element_dim:
                raise ValueError(
                    f"feature group {self.name!r}: active_channel={self.active_channel} "
                    f"is outside element_dim={self.element_dim}"
                )

    @property
    def element_count(self) -> int:
        if self.element_dim is None:
            return 1
        return self.flat_dim // self.element_dim

    @property
    def resolved_active_channel(self) -> int | None:
        if self.active_channel is None or self.element_dim is None:
            return None
        if self.active_channel >= 0:
            return self.active_channel
        return self.element_dim + self.active_channel


@dataclass(frozen=True)
class FeatureSchema:
    simple_groups: tuple[FeatureGroup, ...]
    set_groups: tuple[FeatureGroup, ...]
    _groups_by_name: dict[str, FeatureGroup] = field(init=False, repr=False, compare=False)
    _slices_by_name: dict[str, slice] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        simple_names = tuple(group.name for group in self.simple_groups)
        set_names = tuple(group.name for group in self.set_groups)
        if simple_names != SIMPLE_GROUP_NAMES:
            raise ValueError(f"simple feature order must be {SIMPLE_GROUP_NAMES}, got {simple_names}")
        if set_names != SET_GROUP_NAMES:
            raise ValueError(f"set feature order must be {SET_GROUP_NAMES}, got {set_names}")
        groups = self.simple_groups + self.set_groups
        object.__setattr__(self, "_groups_by_name", {group.name: group for group in groups})
        offset = 0
        slices = {}
        for group in groups:
            slices[group.name] = slice(offset, offset + group.flat_dim)
            offset += group.flat_dim
        object.__setattr__(self, "_slices_by_name", slices)

    @classmethod
    def from_config(cls, config: Any) -> "FeatureSchema":
        training = _get(config, "training")
        network = _get(training, "network")
        if network is None:
            raise ValueError("config.training.network is required")

        configured = _get(network, "feature_schema")
        if configured is None:
            raise ValueError("config.training.network.feature_schema is required")
        simple_cfg = _get(configured, "simple")
        set_cfg = _get(configured, "sets")
        if simple_cfg is None or set_cfg is None:
            raise ValueError("feature_schema requires both 'simple' and 'sets'")
        simple_groups = tuple(
            FeatureGroup(name, int(_get(simple_cfg, name)))
            for name in SIMPLE_GROUP_NAMES
        )
        set_groups = tuple(
            cls._set_group_from_config(name, _get(set_cfg, name))
            for name in SET_GROUP_NAMES
        )
        schema = cls(simple_groups=simple_groups, set_groups=set_groups)
        schema.validate_simulator_config(config)
        return schema

    @staticmethod
    def _set_group_from_config(name: str, group_config: Any) -> FeatureGroup:
        if group_config is None:
            raise ValueError(f"feature_schema.sets.{name} is required")
        active_channel = _get(group_config, "active_channel")
        return FeatureGroup(
            name=name,
            flat_dim=int(_get(group_config, "flat_dim")),
            element_dim=int(_get(group_config, "element_dim")),
            active_channel=None if active_channel is None else int(active_channel),
        )

    @property
    def groups(self) -> tuple[FeatureGroup, ...]:
        return self.simple_groups + self.set_groups

    @property
    def simple_dims(self) -> tuple[int, ...]:
        return tuple(group.flat_dim for group in self.simple_groups)

    @property
    def set_dims(self) -> tuple[int, ...]:
        return tuple(group.flat_dim for group in self.set_groups)

    @property
    def set_element_dims(self) -> tuple[int, ...]:
        return tuple(int(group.element_dim) for group in self.set_groups)

    @property
    def simple_end(self) -> int:
        return sum(self.simple_dims)

    @property
    def total_input_dim(self) -> int:
        return sum(group.flat_dim for group in self.groups)

    def group(self, name: str) -> FeatureGroup:
        try:
            return self._groups_by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown feature group: {name}") from exc

    def flat_slice(self, name: str) -> slice:
        try:
            return self._slices_by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown feature group: {name}") from exc

    def validate_tensor_width(self, width: int) -> None:
        if int(width) != self.total_input_dim:
            raise ValueError(
                f"network feature width mismatch: got {int(width)}, expected {self.total_input_dim}"
            )

    def validate_simulator_config(self, config: Any) -> None:
        simulator = _get(config, "simulator")
        if simulator is None:
            return
        observation = _get(simulator, "observation")
        traffic = _get(simulator, "traffic", {})
        if observation is None:
            return
        expected_dims = {
            "state": int(_get(observation, "local_state_dim")),
            # The simulator stores at most four remaining route targets as (dx, dy).
            "goal": 8,
            "reward": 12,
            "vehicle_style": 4,
            "road_boundary": int(_get(observation, "num_w_boundaries"))
            * int(_get(observation, "boundary_feature_dim")),
            # Two routing-distance channels are appended to each raw lane feature.
            "lane_points": int(_get(observation, "num_w_lanes"))
            * (int(_get(observation, "waypoint_feature_dim")) + 2),
            # A stop line contains two endpoints with two coordinates each.
            "stop_lines": int(_get(traffic, "stop_line_observation_count", 5)) * 4,
            "other_agents": int(_get(observation, "num_neighbors"))
            * int(_get(observation, "neighbor_feature_dim")),
        }
        mismatches = [
            f"{name}={self.group(name).flat_dim} (expected {expected})"
            for name, expected in expected_dims.items()
            if self.group(name).flat_dim != expected
        ]
        if mismatches:
            raise ValueError("feature schema does not match simulator config: " + ", ".join(mismatches))

        expected_element_dims = {
            "road_boundary": int(_get(observation, "boundary_feature_dim")),
            "lane_points": int(_get(observation, "waypoint_feature_dim")) + 2,
            "stop_lines": 2,
            "other_agents": int(_get(observation, "neighbor_feature_dim")),
        }
        element_mismatches = [
            f"{name}.element_dim={self.group(name).element_dim} (expected {expected})"
            for name, expected in expected_element_dims.items()
            if self.group(name).element_dim != expected
        ]
        other_agents = self.group("other_agents")
        for name in ("road_boundary", "lane_points", "stop_lines"):
            if self.group(name).active_channel is not None:
                element_mismatches.append(f"{name}.active_channel must be omitted")
        if other_agents.resolved_active_channel != other_agents.element_dim - 1:
            element_mismatches.append(
                "other_agents.active_channel must resolve to the final element channel"
            )
        if element_mismatches:
            raise ValueError(
                "feature schema element layout does not match simulator config: "
                + ", ".join(element_mismatches)
            )

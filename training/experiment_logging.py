"""Optional experiment tracking integrations for training."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_metrics(metrics: Mapping[str, Any]) -> dict[str, int | float]:
    """Keep only finite scalar values accepted by experiment trackers."""
    cleaned: dict[str, int | float] = {}
    for name, value in metrics.items():
        if value is None:
            continue
        if hasattr(value, "numel") and callable(value.numel):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, bool):
            cleaned[name] = int(value)
        elif isinstance(value, int):
            cleaned[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            cleaned[name] = value
    return cleaned


class SwanLabTracker:
    """Small rank-zero-only wrapper that does not make training depend on logging I/O."""

    def __init__(self, swanlab_module, run, log_interval: int):
        self._swanlab = swanlab_module
        self._run = run
        self.log_interval = max(1, int(log_interval))

    @property
    def url(self) -> str | None:
        try:
            return self._run.get_url()
        except Exception:
            return None

    def log(self, metrics: Mapping[str, Any], step: int, *, force: bool = False) -> None:
        if not force and int(step) % self.log_interval != 0:
            return
        cleaned = _clean_metrics(metrics)
        if not cleaned:
            return
        try:
            self._run.log(cleaned, step=int(step))
        except Exception as exc:
            # A transient dashboard/network problem must not discard an
            # otherwise healthy long-running PPO job.  A later step retries.
            print(f"SwanLab metric upload failed at step {step}: {exc}", flush=True)

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception as exc:
            print(f"SwanLab finish failed: {exc}", flush=True)


def initialize_swanlab(config: dict) -> SwanLabTracker | None:
    """Initialize the configured SwanLab run; callers must invoke this on rank zero only."""
    training_cfg = _mapping(config.get("training"))
    swanlab_cfg = _mapping(training_cfg.get("swanlab"))
    if not bool(swanlab_cfg.get("enabled", False)):
        return None

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "training.swanlab.enabled is true, but SwanLab is not installed; "
            "install the project's requirements before starting training"
        ) from exc

    kwargs = {
        "project": swanlab_cfg.get("project", "Selfrace"),
        "workspace": swanlab_cfg.get("workspace"),
        "experiment_name": swanlab_cfg.get("experiment_name"),
        "description": swanlab_cfg.get("description"),
        "group": swanlab_cfg.get("group"),
        "tags": swanlab_cfg.get("tags"),
        "logdir": swanlab_cfg.get("logdir", "./training/swanlog"),
        "mode": swanlab_cfg.get("mode", "online"),
        "id": swanlab_cfg.get("id"),
        "resume": swanlab_cfg.get("resume"),
        "config": config,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    run = swanlab.init(**kwargs)
    tracker = SwanLabTracker(
        swanlab,
        run,
        log_interval=int(swanlab_cfg.get("log_interval", 1)),
    )
    url = tracker.url
    print(f"SwanLab initialized{f': {url}' if url else ''}", flush=True)
    return tracker

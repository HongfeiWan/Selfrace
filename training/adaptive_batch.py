"""Stateful batch-size backoff used by PPO memory adaptation."""

from dataclasses import dataclass, field


@dataclass
class AdaptiveBatchSizer:
    """Remember a safe microbatch limit and cautiously retry larger values.

    ``maximum`` is the configured ceiling.  An OOM halves ``current`` down to
    ``minimum``.  After ``growth_interval`` successful updates at the current
    limit, the size doubles again so a temporary loss of free VRAM does not
    permanently reduce throughput.
    """

    maximum: int
    minimum: int = 256
    growth_interval: int = 20
    initial: int | None = None
    current: int = field(init=False)
    _successes_at_limit: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.maximum = int(self.maximum)
        self.minimum = int(self.minimum)
        self.growth_interval = int(self.growth_interval)
        if self.maximum <= 0:
            raise ValueError("adaptive batch maximum must be positive")
        if self.minimum <= 0:
            raise ValueError("adaptive batch minimum must be positive")
        if self.growth_interval <= 0:
            raise ValueError("adaptive batch growth_interval must be positive")
        self.minimum = min(self.minimum, self.maximum)
        if self.initial is None:
            self.initial = self.maximum
        self.initial = min(self.maximum, max(self.minimum, int(self.initial)))
        self.current = self.initial

    def choose(self, available: int) -> int:
        available = int(available)
        if available <= 0:
            return 0
        return min(available, self.current)

    def backoff(self, attempted: int) -> int | None:
        attempted = int(attempted)
        floor = min(self.minimum, attempted)
        if attempted <= floor:
            self._successes_at_limit = 0
            return None
        reduced = max(floor, attempted // 2)
        if reduced >= attempted:
            reduced = attempted - 1
        self.current = min(self.current, reduced)
        self._successes_at_limit = 0
        return self.current

    def record_success(self, attempted: int, available: int) -> None:
        attempted = int(attempted)
        available = int(available)
        if attempted < self.current or available < self.current:
            return
        self._successes_at_limit += 1
        if self._successes_at_limit < self.growth_interval:
            return
        self.current = min(self.maximum, max(self.current + 1, self.current * 2))
        self._successes_at_limit = 0

    def state_dict(self) -> dict:
        return {
            "maximum": self.maximum,
            "minimum": self.minimum,
            "growth_interval": self.growth_interval,
            "current": self.current,
            "successes_at_limit": self._successes_at_limit,
        }

    def load_state_dict(self, state: dict) -> None:
        required = {
            "maximum",
            "minimum",
            "growth_interval",
            "current",
            "successes_at_limit",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"adaptive batch state is missing keys: {sorted(missing)}")
        # Static limits belong to the current hardware/configuration.  Only the
        # learned safe point is restored, then clamped into the new limits.
        self.current = min(self.maximum, max(self.minimum, int(state["current"])))
        self._successes_at_limit = max(0, int(state["successes_at_limit"]))

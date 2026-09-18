import time
from collections.abc import Callable


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 5,
        cooldown: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._probe_in_flight:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._probe_in_flight:
            return False
        if self._clock() - self._opened_at >= self._cooldown:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._probe_in_flight or self._failures >= self._threshold:
            self._opened_at = self._clock()
            self._probe_in_flight = False

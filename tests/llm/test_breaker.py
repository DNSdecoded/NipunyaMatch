from app.llm.breaker import CircuitBreaker


class Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_opens_after_threshold_and_half_opens_after_cooldown() -> None:
    clock = Clock()
    b = CircuitBreaker(threshold=5, cooldown=60, clock=clock)
    assert b.state == "closed" and b.allow()
    for _ in range(4):
        b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clock.t = 59
    assert not b.allow()
    clock.t = 61
    assert b.allow()  # single probe
    assert b.state == "half_open"
    assert not b.allow()  # second probe denied while half-open
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_success_resets_count() -> None:
    b = CircuitBreaker(threshold=2, cooldown=1, clock=lambda: 0.0)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state == "closed"


def test_half_open_failure_reopens() -> None:
    clock = Clock()
    b = CircuitBreaker(threshold=1, cooldown=10, clock=clock)
    b.record_failure()
    clock.t = 11
    assert b.allow()
    b.record_failure()
    assert b.state == "open"
    clock.t = 15
    assert not b.allow()

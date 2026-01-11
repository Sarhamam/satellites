"""Tests for health check types."""

from data_layer.core.health import Health


def test_health_ok():
    health = Health(ok=True, details={"redis": "connected"})
    assert health.ok is True
    assert health.details["redis"] == "connected"


def test_health_not_ok():
    health = Health(ok=False, details={"postgres": "connection refused"})
    assert health.ok is False


def test_health_immutable():
    health = Health(ok=True, details={})
    # frozen dataclass should raise
    try:
        health.ok = False  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass

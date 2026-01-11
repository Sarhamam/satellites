"""Tests for event system."""

from datetime import datetime
from data_layer.core.events import DataEvent


def test_data_event_creation():
    event = DataEvent(
        type="user.created",
        resource="users",
        key="user-123",
        payload={"name": "Alice"},
        source="postgres",
    )
    assert event.type == "user.created"
    assert event.resource == "users"
    assert event.key == "user-123"
    assert event.payload["name"] == "Alice"
    assert event.source == "postgres"
    assert isinstance(event.timestamp, datetime)
    assert event.version is None


def test_data_event_with_version():
    event = DataEvent(
        type="doc.indexed",
        resource="documents",
        key="doc-1",
        payload={},
        source="elasticsearch",
        version=42,
    )
    assert event.version == 42


def test_data_event_immutable():
    event = DataEvent(
        type="test",
        resource="test",
        key="1",
        payload={},
        source="test",
    )
    try:
        event.type = "changed"  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass

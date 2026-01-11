"""Tests for event system."""

import pytest
from datetime import datetime
from data_layer.core.events import DataEvent, InProcessEventBus


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


@pytest.fixture
def event_bus() -> InProcessEventBus:
    return InProcessEventBus()


@pytest.fixture
def sample_event() -> DataEvent:
    return DataEvent(
        type="user.created",
        resource="users",
        key="user-123",
        payload={"name": "Alice"},
        source="postgres",
    )


async def test_subscribe_and_publish(event_bus: InProcessEventBus, sample_event: DataEvent):
    received: list[DataEvent] = []

    async def handler(event: DataEvent) -> None:
        received.append(event)

    event_bus.subscribe("user.created", handler)
    await event_bus.publish(sample_event)

    assert len(received) == 1
    assert received[0] == sample_event


async def test_subscribe_all(event_bus: InProcessEventBus, sample_event: DataEvent):
    received: list[DataEvent] = []

    async def handler(event: DataEvent) -> None:
        received.append(event)

    event_bus.subscribe_all(handler)
    await event_bus.publish(sample_event)

    # Publish another event type
    other_event = DataEvent(
        type="doc.deleted",
        resource="documents",
        key="doc-1",
        payload={},
        source="elasticsearch",
    )
    await event_bus.publish(other_event)

    assert len(received) == 2


async def test_unsubscribe(event_bus: InProcessEventBus, sample_event: DataEvent):
    received: list[DataEvent] = []

    async def handler(event: DataEvent) -> None:
        received.append(event)

    event_bus.subscribe("user.created", handler)
    event_bus.unsubscribe("user.created", handler)
    await event_bus.publish(sample_event)

    assert len(received) == 0


async def test_multiple_handlers(event_bus: InProcessEventBus, sample_event: DataEvent):
    received_a: list[DataEvent] = []
    received_b: list[DataEvent] = []

    async def handler_a(event: DataEvent) -> None:
        received_a.append(event)

    async def handler_b(event: DataEvent) -> None:
        received_b.append(event)

    event_bus.subscribe("user.created", handler_a)
    event_bus.subscribe("user.created", handler_b)
    await event_bus.publish(sample_event)

    assert len(received_a) == 1
    assert len(received_b) == 1


async def test_no_handlers_no_error(event_bus: InProcessEventBus, sample_event: DataEvent):
    # Should not raise
    await event_bus.publish(sample_event)

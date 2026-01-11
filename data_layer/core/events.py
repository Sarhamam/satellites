"""Event system for data layer state synchronization."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol


@dataclass(frozen=True)
class DataEvent:
    """Immutable event representing a data change."""

    type: str  # e.g. "user.created", "document.indexed"
    resource: str  # e.g. "users", "documents", "embeddings"
    key: str  # primary identifier
    payload: Mapping[str, Any]  # event data
    source: str  # "redis" | "postgres" | "elasticsearch" | "faiss"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: Optional[int] = None  # optional ordering hint


# Type alias for event handlers
EventHandler = Callable[[DataEvent], Awaitable[None]]


class EventBus(Protocol):
    """Protocol for event bus implementations (in-process now, RabbitMQ later)."""

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to specific event type."""
        ...

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        ...

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from specific event type."""
        ...

    async def publish(self, event: DataEvent) -> None:
        """Publish event to matching subscribers."""
        ...


class InProcessEventBus:
    """Simple in-process pub/sub. Replace with RabbitMQ backend later."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: DataEvent) -> None:
        handlers = self._handlers.get(event.type, []) + self._global_handlers
        for handler in handlers:
            await handler(event)

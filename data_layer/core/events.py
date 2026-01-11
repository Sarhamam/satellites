"""Event system for data layer state synchronization."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional


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

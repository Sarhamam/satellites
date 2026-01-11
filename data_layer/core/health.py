"""Health check types."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Health:
    """Aggregate health status for data layer components."""

    ok: bool
    details: Mapping[str, Any]

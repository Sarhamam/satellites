# Data Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified data layer façade binding Redis, Postgres, Elasticsearch, and FAISS with shared lifecycle and event system.

**Architecture:** Native library APIs (redis-py, asyncpg, elasticsearch-py) exposed directly through a DataLayer façade. Only FAISS gets a wrapper (needs async + metadata). In-process EventBus for state synchronization, designed for future RabbitMQ backend.

**Tech Stack:** Python 3.11+, asyncio, redis-py, asyncpg, elasticsearch[async], faiss-cpu, numpy, pytest-asyncio

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `data_layer/__init__.py`
- Create: `data_layer/py.typed`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "satellites"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "redis>=5.0",
    "asyncpg>=0.29",
    "elasticsearch[async]>=8.11",
    "faiss-cpu>=1.7",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.1",
    "mypy>=1.8",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
```

**Step 2: Create package structure**

```bash
mkdir -p data_layer tests
touch data_layer/__init__.py data_layer/py.typed tests/__init__.py
```

**Step 3: Create tests/conftest.py**

```python
"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
```

**Step 4: Install dependencies**

Run: `cd /Users/shamam/git/satellites/.worktrees/data-layer && pip install -e ".[dev]"`

**Step 5: Verify pytest runs**

Run: `pytest --collect-only`
Expected: "no tests ran" (but no errors)

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: initial project setup with dependencies"
```

---

## Task 2: Health Check Types

**Files:**
- Create: `data_layer/core/__init__.py`
- Create: `data_layer/core/health.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_health.py`

**Step 1: Create directory structure**

```bash
mkdir -p data_layer/core tests/core
touch data_layer/core/__init__.py tests/core/__init__.py
```

**Step 2: Write the failing test**

Create `tests/core/test_health.py`:

```python
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
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/core/test_health.py -v`
Expected: FAIL with "cannot import name 'Health'"

**Step 4: Write minimal implementation**

Create `data_layer/core/health.py`:

```python
"""Health check types."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Health:
    """Aggregate health status for data layer components."""

    ok: bool
    details: Mapping[str, Any]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_health.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add Health dataclass for health checks"
```

---

## Task 3: Event System - DataEvent

**Files:**
- Create: `data_layer/core/events.py`
- Create: `tests/core/test_events.py`

**Step 1: Write the failing test for DataEvent**

Create `tests/core/test_events.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_events.py::test_data_event_creation -v`
Expected: FAIL with "cannot import name 'DataEvent'"

**Step 3: Write minimal implementation**

Create `data_layer/core/events.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_events.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add DataEvent for event-driven state sync"
```

---

## Task 4: Event System - EventBus Protocol and InProcessEventBus

**Files:**
- Modify: `data_layer/core/events.py`
- Modify: `tests/core/test_events.py`

**Step 1: Write the failing test for EventBus**

Add to `tests/core/test_events.py`:

```python
import pytest
from data_layer.core.events import DataEvent, InProcessEventBus


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_events.py::test_subscribe_and_publish -v`
Expected: FAIL with "cannot import name 'InProcessEventBus'"

**Step 3: Write minimal implementation**

Add to `data_layer/core/events.py`:

```python
from typing import Protocol


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_events.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add EventBus protocol and InProcessEventBus"
```

---

## Task 5: Configuration Dataclasses

**Files:**
- Create: `data_layer/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Tests for configuration dataclasses."""

from pathlib import Path
from data_layer.config import (
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
    DataLayerConfig,
)


def test_redis_config_defaults():
    config = RedisConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 0
    assert config.password is None


def test_postgres_config_defaults():
    config = PostgresConfig()
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "satellites"
    assert config.min_connections == 2
    assert config.max_connections == 10


def test_elasticsearch_config_defaults():
    config = ElasticsearchConfig()
    assert config.hosts == ["http://localhost:9200"]


def test_faiss_config_defaults():
    config = FaissConfig()
    assert config.index_path == Path("./data/faiss")
    assert config.dimension == 1536


def test_data_layer_config():
    config = DataLayerConfig(
        redis=RedisConfig(),
        postgres=PostgresConfig(),
        elasticsearch=ElasticsearchConfig(),
        faiss=FaissConfig(),
    )
    assert config.redis.host == "localhost"
    assert config.postgres.database == "satellites"


def test_config_override():
    config = RedisConfig(host="redis.prod.internal", port=6380, password="secret")
    assert config.host == "redis.prod.internal"
    assert config.port == 6380
    assert config.password == "secret"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "cannot import name 'RedisConfig'"

**Step 3: Write minimal implementation**

Create `data_layer/config.py`:

```python
"""Configuration dataclasses for data layer backends."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None


@dataclass
class PostgresConfig:
    """Postgres connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "satellites"
    user: str = "postgres"
    password: Optional[str] = None
    min_connections: int = 2
    max_connections: int = 10


@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection configuration."""

    hosts: list[str] = field(default_factory=lambda: ["http://localhost:9200"])


@dataclass
class FaissConfig:
    """FAISS vector index configuration."""

    index_path: Path = field(default_factory=lambda: Path("./data/faiss"))
    dimension: int = 1536  # OpenAI embedding size default


@dataclass
class DataLayerConfig:
    """Aggregate configuration for all data layer backends."""

    redis: RedisConfig
    postgres: PostgresConfig
    elasticsearch: ElasticsearchConfig
    faiss: FaissConfig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add configuration dataclasses for all backends"
```

---

## Task 6: VectorIndex - Core Types and Interface

**Files:**
- Create: `data_layer/adapters/__init__.py`
- Create: `data_layer/adapters/faiss/__init__.py`
- Create: `data_layer/adapters/faiss/adapter.py`
- Create: `tests/adapters/__init__.py`
- Create: `tests/adapters/test_faiss.py`

**Step 1: Create directory structure**

```bash
mkdir -p data_layer/adapters/faiss tests/adapters
touch data_layer/adapters/__init__.py data_layer/adapters/faiss/__init__.py tests/adapters/__init__.py
```

**Step 2: Write the failing test**

Create `tests/adapters/test_faiss.py`:

```python
"""Tests for FAISS vector index adapter."""

import pytest
import tempfile
from pathlib import Path
from data_layer.adapters.faiss.adapter import VectorIndex, VectorMatch


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def vector_index(temp_dir: Path) -> VectorIndex:
    return VectorIndex(index_path=temp_dir, dimension=4)


async def test_upsert_and_query(vector_index: VectorIndex):
    await vector_index.upsert(
        namespace="test",
        vector_id="vec-1",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"label": "first"},
    )

    results = await vector_index.query(
        namespace="test",
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "vec-1"
    assert results[0].metadata == {"label": "first"}


async def test_query_returns_nearest(vector_index: VectorIndex):
    await vector_index.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0])
    await vector_index.upsert("test", "vec-2", [0.0, 1.0, 0.0, 0.0])
    await vector_index.upsert("test", "vec-3", [0.9, 0.1, 0.0, 0.0])  # closest to vec-1

    results = await vector_index.query("test", [1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    # vec-1 should be first (exact match), vec-3 second (closest)
    assert results[0].id == "vec-1"
    assert results[1].id == "vec-3"


async def test_delete(vector_index: VectorIndex):
    await vector_index.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0])
    deleted = await vector_index.delete("test", "vec-1")
    assert deleted is True

    results = await vector_index.query("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 0


async def test_delete_nonexistent(vector_index: VectorIndex):
    deleted = await vector_index.delete("test", "nonexistent")
    assert deleted is False


async def test_separate_namespaces(vector_index: VectorIndex):
    await vector_index.upsert("ns1", "vec-1", [1.0, 0.0, 0.0, 0.0])
    await vector_index.upsert("ns2", "vec-1", [0.0, 1.0, 0.0, 0.0])

    results_ns1 = await vector_index.query("ns1", [1.0, 0.0, 0.0, 0.0], top_k=1)
    results_ns2 = await vector_index.query("ns2", [1.0, 0.0, 0.0, 0.0], top_k=1)

    # Same ID, different namespaces, different vectors
    assert results_ns1[0].id == "vec-1"
    assert results_ns2[0].id == "vec-1"


async def test_save_and_load(temp_dir: Path):
    index1 = VectorIndex(index_path=temp_dir, dimension=4)
    await index1.upsert("test", "vec-1", [1.0, 0.0, 0.0, 0.0], {"label": "saved"})
    await index1.save()

    # Create new instance and load
    index2 = VectorIndex(index_path=temp_dir, dimension=4)
    await index2.load()

    results = await index2.query("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "vec-1"
    assert results[0].metadata == {"label": "saved"}


def test_vector_match_dataclass():
    match = VectorMatch(id="vec-1", score=0.95, metadata={"key": "value"})
    assert match.id == "vec-1"
    assert match.score == 0.95
    assert match.metadata == {"key": "value"}
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/adapters/test_faiss.py::test_vector_match_dataclass -v`
Expected: FAIL with "cannot import name 'VectorIndex'"

**Step 4: Write minimal implementation**

Create `data_layer/adapters/faiss/adapter.py`:

```python
"""FAISS vector index adapter with async wrapper and metadata sidecar."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorMatch:
    """Result from vector similarity search."""

    id: str
    score: float
    metadata: Optional[Mapping[str, Any]] = None


class VectorIndex:
    """Async wrapper for FAISS with metadata storage."""

    def __init__(self, index_path: Path, dimension: int) -> None:
        self._index_path = index_path
        self._dimension = dimension
        self._namespaces: dict[str, faiss.IndexFlatIP] = {}
        self._id_to_idx: dict[str, dict[str, int]] = {}  # namespace -> {id -> idx}
        self._idx_to_id: dict[str, dict[int, str]] = {}  # namespace -> {idx -> id}
        self._metadata: dict[str, dict[str, Mapping[str, Any]]] = {}  # namespace -> {id -> meta}
        self._index_path.mkdir(parents=True, exist_ok=True)

    def _get_or_create_namespace(self, namespace: str) -> faiss.IndexFlatIP:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = faiss.IndexFlatIP(self._dimension)
            self._id_to_idx[namespace] = {}
            self._idx_to_id[namespace] = {}
            self._metadata[namespace] = {}
        return self._namespaces[namespace]

    async def upsert(
        self,
        namespace: str,
        vector_id: str,
        vector: Sequence[float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Insert or update a vector with optional metadata."""

        def _upsert() -> None:
            # Delete existing if present
            if namespace in self._id_to_idx and vector_id in self._id_to_idx[namespace]:
                self._delete_sync(namespace, vector_id)

            index = self._get_or_create_namespace(namespace)
            vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(vec)  # Normalize for cosine similarity

            idx = index.ntotal
            index.add(vec)

            self._id_to_idx[namespace][vector_id] = idx
            self._idx_to_id[namespace][idx] = vector_id
            if metadata:
                self._metadata[namespace][vector_id] = metadata

        await asyncio.to_thread(_upsert)

    def _delete_sync(self, namespace: str, vector_id: str) -> bool:
        """Sync delete - rebuilds index without the deleted vector."""
        if namespace not in self._id_to_idx:
            return False
        if vector_id not in self._id_to_idx[namespace]:
            return False

        # Get all vectors except the one to delete
        old_index = self._namespaces[namespace]
        old_id_to_idx = self._id_to_idx[namespace]
        old_idx_to_id = self._idx_to_id[namespace]

        # Rebuild
        new_index = faiss.IndexFlatIP(self._dimension)
        new_id_to_idx: dict[str, int] = {}
        new_idx_to_id: dict[int, str] = {}

        for vid, old_idx in old_id_to_idx.items():
            if vid == vector_id:
                continue
            vec = old_index.reconstruct(old_idx).reshape(1, -1)
            new_idx = new_index.ntotal
            new_index.add(vec)
            new_id_to_idx[vid] = new_idx
            new_idx_to_id[new_idx] = vid

        self._namespaces[namespace] = new_index
        self._id_to_idx[namespace] = new_id_to_idx
        self._idx_to_id[namespace] = new_idx_to_id
        self._metadata[namespace].pop(vector_id, None)

        return True

    async def delete(self, namespace: str, vector_id: str) -> bool:
        """Delete a vector by ID."""
        return await asyncio.to_thread(self._delete_sync, namespace, vector_id)

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        top_k: int = 10,
    ) -> list[VectorMatch]:
        """Find nearest neighbors."""

        def _query() -> list[VectorMatch]:
            if namespace not in self._namespaces:
                return []

            index = self._namespaces[namespace]
            if index.ntotal == 0:
                return []

            vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(vec)

            k = min(top_k, index.ntotal)
            scores, indices = index.search(vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                vid = self._idx_to_id[namespace].get(int(idx))
                if vid:
                    meta = self._metadata[namespace].get(vid)
                    results.append(VectorMatch(id=vid, score=float(score), metadata=meta))

            return results

        return await asyncio.to_thread(_query)

    async def save(self) -> None:
        """Persist all indexes and metadata to disk."""

        def _save() -> None:
            for namespace, index in self._namespaces.items():
                ns_path = self._index_path / namespace
                ns_path.mkdir(exist_ok=True)

                faiss.write_index(index, str(ns_path / "index.faiss"))

                state = {
                    "id_to_idx": self._id_to_idx[namespace],
                    "idx_to_id": {str(k): v for k, v in self._idx_to_id[namespace].items()},
                    "metadata": dict(self._metadata[namespace]),
                }
                with open(ns_path / "state.json", "w") as f:
                    json.dump(state, f)

        await asyncio.to_thread(_save)

    async def load(self) -> None:
        """Load all indexes and metadata from disk."""

        def _load() -> None:
            if not self._index_path.exists():
                return

            for ns_path in self._index_path.iterdir():
                if not ns_path.is_dir():
                    continue

                index_file = ns_path / "index.faiss"
                state_file = ns_path / "state.json"

                if not index_file.exists() or not state_file.exists():
                    continue

                namespace = ns_path.name
                self._namespaces[namespace] = faiss.read_index(str(index_file))

                with open(state_file) as f:
                    state = json.load(f)

                self._id_to_idx[namespace] = state["id_to_idx"]
                self._idx_to_id[namespace] = {int(k): v for k, v in state["idx_to_id"].items()}
                self._metadata[namespace] = state.get("metadata", {})

        await asyncio.to_thread(_load)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/adapters/test_faiss.py -v`
Expected: 8 passed

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add VectorIndex FAISS adapter with async and metadata"
```

---

## Task 7: DataLayer Façade

**Files:**
- Create: `data_layer/core/facade.py`
- Create: `tests/core/test_facade.py`

**Step 1: Write the failing test**

Create `tests/core/test_facade.py`:

```python
"""Tests for DataLayer façade."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from data_layer.core.facade import DataLayer
from data_layer.config import (
    DataLayerConfig,
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def config(temp_dir: Path) -> DataLayerConfig:
    return DataLayerConfig(
        redis=RedisConfig(),
        postgres=PostgresConfig(),
        elasticsearch=ElasticsearchConfig(),
        faiss=FaissConfig(index_path=temp_dir),
    )


def test_data_layer_has_expected_attributes(config: DataLayerConfig):
    data = DataLayer(config)

    # Check all expected attributes exist
    assert hasattr(data, "redis")
    assert hasattr(data, "postgres")
    assert hasattr(data, "elasticsearch")
    assert hasattr(data, "vectors")
    assert hasattr(data, "events")


async def test_data_layer_start_stop(config: DataLayerConfig):
    """Test lifecycle methods exist and can be called."""
    data = DataLayer(config)

    # Mock the actual connections since we don't have real DBs
    with patch.object(data, "_connect_redis", new_callable=AsyncMock) as mock_redis, \
         patch.object(data, "_connect_postgres", new_callable=AsyncMock) as mock_pg, \
         patch.object(data, "_connect_elasticsearch", new_callable=AsyncMock) as mock_es:

        await data.start()

        mock_redis.assert_called_once()
        mock_pg.assert_called_once()
        mock_es.assert_called_once()

    with patch.object(data, "_disconnect_redis", new_callable=AsyncMock) as mock_redis, \
         patch.object(data, "_disconnect_postgres", new_callable=AsyncMock) as mock_pg, \
         patch.object(data, "_disconnect_elasticsearch", new_callable=AsyncMock) as mock_es:

        await data.stop()

        mock_redis.assert_called_once()
        mock_pg.assert_called_once()
        mock_es.assert_called_once()


async def test_data_layer_context_manager(config: DataLayerConfig):
    """Test async context manager protocol."""
    with patch.object(DataLayer, "start", new_callable=AsyncMock) as mock_start, \
         patch.object(DataLayer, "stop", new_callable=AsyncMock) as mock_stop:

        async with DataLayer(config) as data:
            assert data is not None

        mock_start.assert_called_once()
        mock_stop.assert_called_once()


async def test_health_check_structure(config: DataLayerConfig):
    data = DataLayer(config)

    # Mock backends as connected
    data._redis = MagicMock()
    data._redis.ping = AsyncMock(return_value=True)
    data._postgres = MagicMock()
    data._postgres.fetchval = AsyncMock(return_value=1)
    data._elasticsearch = MagicMock()
    data._elasticsearch.ping = AsyncMock(return_value=True)

    health = await data.health()

    assert hasattr(health, "ok")
    assert hasattr(health, "details")
    assert "redis" in health.details
    assert "postgres" in health.details
    assert "elasticsearch" in health.details
    assert "faiss" in health.details
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_facade.py::test_data_layer_has_expected_attributes -v`
Expected: FAIL with "cannot import name 'DataLayer'"

**Step 3: Write minimal implementation**

Create `data_layer/core/facade.py`:

```python
"""DataLayer façade - unified access to all backends."""

from types import TracebackType
from typing import Optional, Self

import redis.asyncio as redis_lib
import asyncpg
from elasticsearch import AsyncElasticsearch

from data_layer.config import DataLayerConfig
from data_layer.core.events import InProcessEventBus
from data_layer.core.health import Health
from data_layer.adapters.faiss.adapter import VectorIndex


class DataLayer:
    """Unified façade - lifecycle + access to all backends."""

    def __init__(self, config: DataLayerConfig) -> None:
        self._config = config
        self._redis: Optional[redis_lib.Redis] = None
        self._postgres: Optional[asyncpg.Pool] = None
        self._elasticsearch: Optional[AsyncElasticsearch] = None
        self._vectors = VectorIndex(
            index_path=config.faiss.index_path,
            dimension=config.faiss.dimension,
        )
        self._events = InProcessEventBus()

    @property
    def redis(self) -> redis_lib.Redis:
        """Native redis-py async client."""
        if self._redis is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._redis

    @property
    def postgres(self) -> asyncpg.Pool:
        """Native asyncpg pool."""
        if self._postgres is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._postgres

    @property
    def elasticsearch(self) -> AsyncElasticsearch:
        """Native Elasticsearch async client."""
        if self._elasticsearch is None:
            raise RuntimeError("DataLayer not started. Call start() first.")
        return self._elasticsearch

    @property
    def vectors(self) -> VectorIndex:
        """FAISS vector index."""
        return self._vectors

    @property
    def events(self) -> InProcessEventBus:
        """Event bus for pub/sub."""
        return self._events

    async def _connect_redis(self) -> None:
        cfg = self._config.redis
        self._redis = redis_lib.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password,
        )

    async def _connect_postgres(self) -> None:
        cfg = self._config.postgres
        self._postgres = await asyncpg.create_pool(
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
            user=cfg.user,
            password=cfg.password,
            min_size=cfg.min_connections,
            max_size=cfg.max_connections,
        )

    async def _connect_elasticsearch(self) -> None:
        cfg = self._config.elasticsearch
        self._elasticsearch = AsyncElasticsearch(hosts=cfg.hosts)

    async def _disconnect_redis(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def _disconnect_postgres(self) -> None:
        if self._postgres:
            await self._postgres.close()
            self._postgres = None

    async def _disconnect_elasticsearch(self) -> None:
        if self._elasticsearch:
            await self._elasticsearch.close()
            self._elasticsearch = None

    async def start(self) -> None:
        """Connect all backends."""
        await self._connect_redis()
        await self._connect_postgres()
        await self._connect_elasticsearch()
        await self._vectors.load()

    async def stop(self) -> None:
        """Graceful shutdown."""
        await self._vectors.save()
        await self._disconnect_redis()
        await self._disconnect_postgres()
        await self._disconnect_elasticsearch()

    async def health(self) -> Health:
        """Aggregate health check across all backends."""
        details: dict[str, str] = {}
        all_ok = True

        # Redis
        try:
            if self._redis:
                await self._redis.ping()
                details["redis"] = "ok"
            else:
                details["redis"] = "not connected"
                all_ok = False
        except Exception as e:
            details["redis"] = f"error: {e}"
            all_ok = False

        # Postgres
        try:
            if self._postgres:
                await self._postgres.fetchval("SELECT 1")
                details["postgres"] = "ok"
            else:
                details["postgres"] = "not connected"
                all_ok = False
        except Exception as e:
            details["postgres"] = f"error: {e}"
            all_ok = False

        # Elasticsearch
        try:
            if self._elasticsearch:
                await self._elasticsearch.ping()
                details["elasticsearch"] = "ok"
            else:
                details["elasticsearch"] = "not connected"
                all_ok = False
        except Exception as e:
            details["elasticsearch"] = f"error: {e}"
            all_ok = False

        # FAISS (always ok if initialized)
        details["faiss"] = "ok"

        return Health(ok=all_ok, details=details)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_facade.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add DataLayer façade with lifecycle and health checks"
```

---

## Task 8: Package Exports

**Files:**
- Modify: `data_layer/__init__.py`
- Modify: `data_layer/core/__init__.py`
- Modify: `data_layer/adapters/faiss/__init__.py`
- Create: `tests/test_imports.py`

**Step 1: Write the failing test**

Create `tests/test_imports.py`:

```python
"""Test that public API is importable from package root."""


def test_main_imports():
    from data_layer import (
        DataLayer,
        DataLayerConfig,
        RedisConfig,
        PostgresConfig,
        ElasticsearchConfig,
        FaissConfig,
        DataEvent,
        InProcessEventBus,
        Health,
        VectorIndex,
        VectorMatch,
    )

    # Just verify they're importable
    assert DataLayer is not None
    assert DataLayerConfig is not None
    assert DataEvent is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Update `data_layer/core/__init__.py`:

```python
"""Core components for data layer."""

from data_layer.core.events import DataEvent, EventHandler, EventBus, InProcessEventBus
from data_layer.core.facade import DataLayer
from data_layer.core.health import Health

__all__ = [
    "DataEvent",
    "EventHandler",
    "EventBus",
    "InProcessEventBus",
    "DataLayer",
    "Health",
]
```

Update `data_layer/adapters/faiss/__init__.py`:

```python
"""FAISS vector index adapter."""

from data_layer.adapters.faiss.adapter import VectorIndex, VectorMatch

__all__ = ["VectorIndex", "VectorMatch"]
```

Update `data_layer/__init__.py`:

```python
"""Unified data layer for Redis, Postgres, Elasticsearch, and FAISS."""

from data_layer.config import (
    DataLayerConfig,
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
)
from data_layer.core import (
    DataLayer,
    DataEvent,
    EventHandler,
    EventBus,
    InProcessEventBus,
    Health,
)
from data_layer.adapters.faiss import VectorIndex, VectorMatch

__all__ = [
    # Façade
    "DataLayer",
    # Config
    "DataLayerConfig",
    "RedisConfig",
    "PostgresConfig",
    "ElasticsearchConfig",
    "FaissConfig",
    # Events
    "DataEvent",
    "EventHandler",
    "EventBus",
    "InProcessEventBus",
    # Health
    "Health",
    # Vectors
    "VectorIndex",
    "VectorMatch",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_imports.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: expose public API from package root"
```

---

## Task 9: Docker Compose for Local Dev

**Files:**
- Create: `docker/docker-compose.yml`

**Step 1: Create docker directory**

```bash
mkdir -p docker
```

**Step 2: Create docker-compose.yml**

Create `docker/docker-compose.yml`:

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: satellites
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 3

  elasticsearch:
    image: elasticsearch:8.11.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q 'green\\|yellow'"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
  postgres_data:
  elasticsearch_data:
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add docker-compose for local dev (Redis, Postgres, ES)"
```

---

## Task 10: Integration Test with Real Backends

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_data_layer.py`

**Step 1: Create directory**

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

**Step 2: Create integration test**

Create `tests/integration/test_data_layer.py`:

```python
"""Integration tests for DataLayer with real backends.

Run these tests with Docker Compose:
    cd docker && docker-compose up -d
    pytest tests/integration/ -v
    cd docker && docker-compose down
"""

import pytest
import tempfile
from pathlib import Path

from data_layer import (
    DataLayer,
    DataLayerConfig,
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
    DataEvent,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def config(temp_dir: Path) -> DataLayerConfig:
    return DataLayerConfig(
        redis=RedisConfig(),
        postgres=PostgresConfig(password="postgres"),
        elasticsearch=ElasticsearchConfig(),
        faiss=FaissConfig(index_path=temp_dir),
    )


@pytest.fixture
async def data_layer(config: DataLayerConfig):
    async with DataLayer(config) as data:
        yield data


async def test_redis_operations(data_layer: DataLayer):
    """Test Redis get/set through native API."""
    await data_layer.redis.set("test:key", b"test-value")
    value = await data_layer.redis.get("test:key")
    assert value == b"test-value"

    await data_layer.redis.delete("test:key")
    value = await data_layer.redis.get("test:key")
    assert value is None


async def test_postgres_operations(data_layer: DataLayer):
    """Test Postgres queries through native API."""
    # Simple query
    result = await data_layer.postgres.fetchval("SELECT 1 + 1")
    assert result == 2

    # Create and query a temp table
    await data_layer.postgres.execute("""
        CREATE TEMPORARY TABLE test_users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    await data_layer.postgres.execute(
        "INSERT INTO test_users (name) VALUES ($1)",
        "Alice"
    )
    row = await data_layer.postgres.fetchrow(
        "SELECT * FROM test_users WHERE name = $1",
        "Alice"
    )
    assert row["name"] == "Alice"


async def test_elasticsearch_operations(data_layer: DataLayer):
    """Test Elasticsearch index/search through native API."""
    index_name = "test-docs"

    # Index a document
    await data_layer.elasticsearch.index(
        index=index_name,
        id="doc-1",
        document={"title": "Hello World", "content": "This is a test document"},
        refresh=True,  # Make immediately searchable
    )

    # Search
    result = await data_layer.elasticsearch.search(
        index=index_name,
        query={"match": {"content": "test"}},
    )
    assert result["hits"]["total"]["value"] == 1
    assert result["hits"]["hits"][0]["_id"] == "doc-1"

    # Cleanup
    await data_layer.elasticsearch.indices.delete(index=index_name)


async def test_vector_operations(data_layer: DataLayer):
    """Test FAISS vector operations."""
    await data_layer.vectors.upsert(
        namespace="embeddings",
        vector_id="vec-1",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"source": "test"},
    )

    results = await data_layer.vectors.query(
        namespace="embeddings",
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "vec-1"
    assert results[0].metadata == {"source": "test"}


async def test_event_system(data_layer: DataLayer):
    """Test event pub/sub."""
    received_events: list[DataEvent] = []

    async def handler(event: DataEvent) -> None:
        received_events.append(event)

    data_layer.events.subscribe("test.event", handler)

    event = DataEvent(
        type="test.event",
        resource="test",
        key="key-1",
        payload={"data": "value"},
        source="postgres",
    )
    await data_layer.events.publish(event)

    assert len(received_events) == 1
    assert received_events[0].type == "test.event"


async def test_health_check(data_layer: DataLayer):
    """Test aggregate health check."""
    health = await data_layer.health()

    assert health.ok is True
    assert health.details["redis"] == "ok"
    assert health.details["postgres"] == "ok"
    assert health.details["elasticsearch"] == "ok"
    assert health.details["faiss"] == "ok"
```

**Step 3: Update pytest config to handle integration marker**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: tests requiring external services (Redis, Postgres, Elasticsearch)",
]
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add integration tests for DataLayer with real backends"
```

---

## Task 11: Final Verification

**Step 1: Run all unit tests**

Run: `pytest tests/ -v --ignore=tests/integration/`
Expected: All tests pass

**Step 2: Start Docker services**

Run: `cd docker && docker-compose up -d && cd ..`

**Step 3: Run integration tests**

Run: `pytest tests/integration/ -v`
Expected: All tests pass

**Step 4: Stop Docker services**

Run: `cd docker && docker-compose down && cd ..`

**Step 5: Final commit (if any changes)**

```bash
git status
# If clean, skip this step
```

---

## Summary

After completing all tasks, you will have:

1. **Project structure** with pyproject.toml and dev dependencies
2. **Core types**: Health, DataEvent, EventBus protocol, InProcessEventBus
3. **Configuration**: Dataclasses for all backends
4. **VectorIndex**: FAISS wrapper with async and metadata
5. **DataLayer façade**: Unified lifecycle, native API access, health checks
6. **Docker Compose**: Local dev environment
7. **Tests**: Unit tests + integration tests

Total: ~11 tasks, each 5-15 minutes

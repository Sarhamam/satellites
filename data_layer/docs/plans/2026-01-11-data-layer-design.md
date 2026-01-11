# Data Layer Design

## Overview

A unified data layer façade that binds Redis, Postgres, Elasticsearch, and FAISS into one coherent interface with shared lifecycle, health checks, and an event system for state synchronization.

**Primary use case:** AI/ML pipelines with vector search as the core feature, flexible enough for analytics or SaaS workloads.

## Key Principles

1. **No unnecessary wrappers** - Use native library APIs directly (redis-py, asyncpg, elasticsearch-py). Only FAISS gets a wrapper (needs async + metadata handling).
2. **Unified lifecycle** - Single `start()` / `stop()` / `health()` for all backends.
3. **Event-driven state sync** - DataEvents bridge the data layer to application state without tight coupling.
4. **Semantic layer ready** - Interfaces designed so SemanticStore/ChangeSet abstraction can wrap them later.

## Project Structure

```
satellites/
├── data_layer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py          # DataEvent + EventBus
│   │   ├── health.py          # Health check types
│   │   └── facade.py          # DataLayer façade
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── faiss/
│   │       ├── __init__.py
│   │       └── adapter.py     # VectorIndex implementation
│   └── config.py              # Configuration dataclasses
├── docker/
│   └── docker-compose.yml     # Redis, Postgres, ES for local dev
├── tests/
│   ├── conftest.py
│   ├── test_redis.py
│   ├── test_postgres.py
│   ├── test_elasticsearch.py
│   ├── test_faiss.py
│   └── test_facade.py
├── pyproject.toml
└── README.md
```

## Core Components

### DataLayer Façade

```python
from dataclasses import dataclass
from typing import Mapping, Any
import redis.asyncio as redis
import asyncpg
from elasticsearch import AsyncElasticsearch

@dataclass
class Health:
    ok: bool
    details: Mapping[str, Any]

class DataLayer:
    """Unified façade - lifecycle + access to all backends."""

    redis: redis.Redis
    postgres: asyncpg.Pool
    elasticsearch: AsyncElasticsearch
    vectors: VectorIndex
    events: EventBus

    async def start(self) -> None:
        """Connect all backends."""
        ...

    async def stop(self) -> None:
        """Graceful shutdown."""
        ...

    async def health(self) -> Health:
        """Aggregate health check across all backends."""
        ...
```

**Access patterns:**
- `data.redis.get(...)` - native redis-py async API
- `data.postgres.fetch(...)` - native asyncpg Pool API
- `data.elasticsearch.search(...)` - native ES async API
- `data.vectors.query(...)` - our FAISS wrapper

### VectorIndex (FAISS Wrapper)

Only backend that needs a wrapper due to:
- FAISS is sync-only, needs async wrapping
- No built-in metadata storage, needs JSON sidecar
- Index persistence to local filesystem

```python
from dataclasses import dataclass
from typing import Optional, Mapping, Any, Sequence

@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    metadata: Optional[Mapping[str, Any]] = None

class VectorIndex:
    async def upsert(
        self,
        namespace: str,
        vector_id: str,
        vector: Sequence[float],
        metadata: Optional[Mapping[str, Any]] = None
    ) -> None: ...

    async def delete(self, namespace: str, vector_id: str) -> bool: ...

    async def query(
        self,
        namespace: str,
        vector: Sequence[float],
        top_k: int = 10
    ) -> list[VectorMatch]: ...

    async def save(self) -> None:
        """Persist index to disk."""
        ...

    async def load(self) -> None:
        """Load index from disk."""
        ...
```

### Events System

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Mapping, Optional, Protocol
from datetime import datetime

@dataclass(frozen=True)
class DataEvent:
    type: str                           # e.g. "user.created", "document.indexed"
    resource: str                       # e.g. "users", "documents", "embeddings"
    key: str                            # primary identifier
    payload: Mapping[str, Any]          # event data
    source: str                         # "redis" | "postgres" | "elasticsearch" | "faiss"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: Optional[int] = None       # optional ordering hint

EventHandler = Callable[[DataEvent], Awaitable[None]]

class EventBus(Protocol):
    """Protocol for event bus implementations (in-process now, RabbitMQ later)."""

    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...
    def subscribe_all(self, handler: EventHandler) -> None: ...
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None: ...
    async def publish(self, event: DataEvent) -> None: ...

class InProcessEventBus:
    """Simple in-process pub/sub. Replace with RabbitMQ backend later."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...
    def subscribe_all(self, handler: EventHandler) -> None: ...
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None: ...
    async def publish(self, event: DataEvent) -> None: ...
```

### Configuration

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "satellites"
    user: str = "postgres"
    password: Optional[str] = None
    min_connections: int = 2
    max_connections: int = 10

@dataclass
class ElasticsearchConfig:
    hosts: list[str] = None

    def __post_init__(self):
        if self.hosts is None:
            self.hosts = ["http://localhost:9200"]

@dataclass
class FaissConfig:
    index_path: Path = Path("./data/faiss")
    dimension: int = 1536  # OpenAI embedding size default

@dataclass
class DataLayerConfig:
    redis: RedisConfig
    postgres: PostgresConfig
    elasticsearch: ElasticsearchConfig
    faiss: FaissConfig
```

## Usage Example

```python
from data_layer import DataLayer, DataLayerConfig, DataEvent

# Initialize
config = DataLayerConfig(...)
data = DataLayer(config)
await data.start()

# Subscribe to events
async def on_document_indexed(event: DataEvent):
    print(f"Document {event.key} indexed")

data.events.subscribe("document.indexed", on_document_indexed)

# Use native APIs
await data.redis.set("session:123", b"user_data")
await data.postgres.execute("INSERT INTO users (id, name) VALUES ($1, $2)", user_id, name)
await data.elasticsearch.index(index="documents", id=doc_id, document={"text": "..."})
await data.vectors.upsert("embeddings", doc_id, embedding_vector, {"source": "openai"})

# Publish events after writes
await data.events.publish(DataEvent(
    type="document.indexed",
    resource="documents",
    key=doc_id,
    payload={"text": "..."},
    source="elasticsearch"
))

# Shutdown
await data.stop()
```

## Docker Compose (Local Dev)

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

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

volumes:
  redis_data:
  postgres_data:
  elasticsearch_data:
```

## Dependencies

```toml
[project]
name = "satellites"
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
    "pytest-docker>=2.0",
]
```

## Future Extensions

1. **RabbitMQ EventBus** - Implement `EventBus` protocol with RabbitMQ backend
2. **SemanticStore** - ChangeSet/SemanticOp layer on top for dynamic routing
3. **S3 FAISS persistence** - Add cloud storage option for VectorIndex
4. **Connection pooling tuning** - Per-backend pool configuration
5. **Metrics/tracing** - OpenTelemetry instrumentation on façade methods

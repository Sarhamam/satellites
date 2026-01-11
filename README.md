# Data Layer

A unified data layer façade for Redis, Postgres, Elasticsearch, and FAISS with shared lifecycle management and event-driven state synchronization.

## Features

- **Unified Interface**: Single entry point for all data operations
- **Native APIs**: Direct access to redis-py, asyncpg, and elasticsearch-py clients
- **FAISS Vector Search**: Async wrapper with metadata support and persistence
- **Event System**: In-process pub/sub for state synchronization (designed for future RabbitMQ backend)
- **Lifecycle Management**: Automatic connection/disconnection with async context manager
- **Health Checks**: Aggregate health monitoring across all backends
- **Type Safe**: Full type hints with mypy strict mode

## Installation

```bash
# Install the package
pip install -e ".[dev]"

# Start Docker services for local development
cd docker && docker compose up -d
```

## Quick Start

```python
import asyncio
from data_layer import (
    DataLayer,
    DataLayerConfig,
    RedisConfig,
    PostgresConfig,
    ElasticsearchConfig,
    FaissConfig,
)

async def main():
    # Configure the data layer
    config = DataLayerConfig(
        redis=RedisConfig(),
        postgres=PostgresConfig(password="postgres"),
        elasticsearch=ElasticsearchConfig(),
        faiss=FaissConfig(),
    )

    # Use async context manager for automatic lifecycle
    async with DataLayer(config) as data:
        # Check health
        health = await data.health()
        print(f"All services healthy: {health.ok}")

        # Use Redis
        await data.redis.set("key", b"value")
        value = await data.redis.get("key")

        # Use Postgres
        result = await data.postgres.fetchval("SELECT 1 + 1")

        # Use Elasticsearch
        await data.elasticsearch.index(
            index="docs",
            id="1",
            document={"title": "Hello World"},
            refresh=True,
        )

        # Use FAISS vectors
        embedding = [0.1] * 1536  # 1536-dimensional vector
        await data.vectors.upsert(
            namespace="embeddings",
            vector_id="doc-1",
            vector=embedding,
            metadata={"title": "Document 1"},
        )

        # Search vectors
        results = await data.vectors.query(
            namespace="embeddings",
            vector=embedding,
            top_k=5,
        )

asyncio.run(main())
```

## Architecture

### Native Library APIs

The data layer exposes native client libraries directly:

- **Redis**: `redis.asyncio.Redis` from redis-py
- **Postgres**: `asyncpg.Pool` from asyncpg
- **Elasticsearch**: `AsyncElasticsearch` from elasticsearch-py

This means you have full access to all native methods and don't need to learn a new abstraction layer.

### FAISS Adapter

FAISS requires an adapter because:
1. **Async wrapper**: FAISS is synchronous, adapter provides async interface
2. **Metadata storage**: FAISS stores only vectors, adapter adds metadata sidecar
3. **Namespaces**: Support for multiple isolated vector collections
4. **Persistence**: Save/load indexes with metadata to/from disk
5. **Thread safety**: All operations are protected with locks for concurrent access

### Event System

The event bus provides in-process pub/sub for state synchronization:

```python
from data_layer import DataEvent

# Subscribe to events
async def handle_user_created(event: DataEvent) -> None:
    print(f"User created: {event.key}")

data.events.subscribe("user.created", handle_user_created)

# Publish events
event = DataEvent(
    type="user.created",
    resource="users",
    key="user-123",
    payload={"name": "Alice"},
    source="postgres",
)
await data.events.publish(event)
```

The event system is designed for future migration to RabbitMQ without code changes.

## Configuration

### Environment-based Configuration

```python
from pathlib import Path
from data_layer import DataLayerConfig, RedisConfig, PostgresConfig

config = DataLayerConfig(
    redis=RedisConfig(
        host="redis.prod.internal",
        port=6379,
        password="secret",
    ),
    postgres=PostgresConfig(
        host="postgres.prod.internal",
        database="satellites",
        user="app",
        password="secret",
        min_connections=5,
        max_connections=20,
    ),
    elasticsearch=ElasticsearchConfig(
        hosts=["http://es-node1:9200", "http://es-node2:9200"],
    ),
    faiss=FaissConfig(
        index_path=Path("/data/vectors"),
        dimension=1536,
    ),
)
```

### Configuration Options

#### RedisConfig
- `host`: Redis server host (default: "localhost")
- `port`: Redis server port (default: 6379)
- `db`: Redis database number (default: 0)
- `password`: Redis password (default: None)

#### PostgresConfig
- `host`: Postgres server host (default: "localhost")
- `port`: Postgres server port (default: 5432)
- `database`: Database name (default: "satellites")
- `user`: Database user (default: "postgres")
- `password`: Database password (default: None)
- `min_connections`: Minimum pool size (default: 2)
- `max_connections`: Maximum pool size (default: 10)

#### ElasticsearchConfig
- `hosts`: List of Elasticsearch nodes (default: ["http://localhost:9200"])

#### FaissConfig
- `index_path`: Directory for FAISS indexes (default: "./data/faiss")
- `dimension`: Vector dimension (default: 1536)

## API Reference

### DataLayer

Main façade class providing unified access to all backends.

#### Properties

- `redis: Redis` - Native redis-py async client
- `postgres: Pool` - Native asyncpg connection pool
- `elasticsearch: AsyncElasticsearch` - Native Elasticsearch client
- `vectors: VectorIndex` - FAISS vector index adapter
- `events: InProcessEventBus` - Event bus for pub/sub

#### Methods

- `async start()` - Connect all backends
- `async stop()` - Graceful shutdown
- `async health() -> Health` - Aggregate health check

#### Context Manager

```python
async with DataLayer(config) as data:
    # Automatically calls start() on enter
    # Automatically calls stop() on exit
    pass
```

### VectorIndex

FAISS vector index with async interface and metadata support.

#### Methods

- `async upsert(namespace, vector_id, vector, metadata=None)` - Insert or update vector
- `async query(namespace, vector, top_k=10) -> list[VectorMatch]` - Find nearest neighbors
- `async delete(namespace, vector_id) -> bool` - Delete vector
- `async save()` - Persist to disk
- `async load()` - Load from disk

### InProcessEventBus

Simple pub/sub event bus.

#### Methods

- `subscribe(event_type, handler)` - Subscribe to specific event type
- `subscribe_all(handler)` - Subscribe to all events
- `unsubscribe(event_type, handler)` - Unsubscribe from event type
- `async publish(event)` - Publish event to subscribers

## Testing

```bash
# Run unit tests (no external dependencies)
pytest tests/ -v --ignore=tests/integration/

# Run integration tests (requires Docker services)
cd docker && docker compose up -d
pytest tests/integration/ -v
cd docker && docker compose down
```

## Development

### Project Structure

```
data_layer/
├── __init__.py           # Public API exports
├── config.py             # Configuration dataclasses
├── core/
│   ├── events.py         # Event system
│   ├── facade.py         # DataLayer façade
│   └── health.py         # Health check types
└── adapters/
    └── faiss/
        └── adapter.py    # FAISS vector index adapter

tests/
├── core/                 # Core functionality tests
├── adapters/             # Adapter tests
├── integration/          # Integration tests with real backends
└── conftest.py           # Pytest configuration

docker/
└── docker-compose.yml    # Local development services
```

### Code Quality

```bash
# Type checking
mypy data_layer

# Linting and formatting
ruff check data_layer
ruff format data_layer
```

## License

MIT

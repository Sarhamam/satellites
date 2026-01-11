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
    # Use dimension=1536 to match default FAISS config
    test_vector = [1.0] + [0.0] * 1535

    await data_layer.vectors.upsert(
        namespace="embeddings",
        vector_id="vec-1",
        vector=test_vector,
        metadata={"source": "test"},
    )

    results = await data_layer.vectors.query(
        namespace="embeddings",
        vector=test_vector,
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

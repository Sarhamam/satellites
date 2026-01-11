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

    # Check all expected property/attribute names exist
    assert "redis" in dir(data)
    assert "postgres" in dir(data)
    assert "elasticsearch" in dir(data)
    # These should be accessible without starting
    assert data.vectors is not None
    assert data.events is not None


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

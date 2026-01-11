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

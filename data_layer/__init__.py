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

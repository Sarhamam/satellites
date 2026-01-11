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

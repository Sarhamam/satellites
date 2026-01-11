"""Pydantic models for schema registry."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SchemaStatus(str, Enum):
    """Status of a schema version."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


# Backend configurations


class ColumnConfig(BaseModel):
    """Postgres column configuration."""

    type: str
    nullable: bool = True


class IndexConfig(BaseModel):
    """Postgres index configuration."""

    name: str
    columns: list[str]
    unique: bool = False
    using: Optional[str] = None  # e.g., "gin", "btree"


class PostgresBackend(BaseModel):
    """Postgres backend configuration."""

    table: str
    primary_key: list[str]
    columns: dict[str, ColumnConfig | dict[str, Any]]
    indexes: list[IndexConfig | dict[str, Any]] = Field(default_factory=list)


class ElasticsearchBackend(BaseModel):
    """Elasticsearch backend configuration."""

    index: str
    mappings: dict[str, Any]


class RedisBackend(BaseModel):
    """Redis backend configuration."""

    pattern: str
    ttl_seconds: Optional[int] = None
    encoding: str = "json"


class FaissBackend(BaseModel):
    """FAISS backend configuration."""

    namespace: str
    dimension: int
    id_field: str
    metadata_fields: list[str] = Field(default_factory=list)


class BackendConfig(BaseModel):
    """All backend configurations."""

    postgres: Optional[PostgresBackend] = None
    elasticsearch: Optional[ElasticsearchBackend] = None
    redis: Optional[RedisBackend] = None
    faiss: Optional[FaissBackend] = None


# Schema versions


class SchemaVersionBase(BaseModel):
    """Base fields for schema version."""

    resource: str
    tenant_id: Optional[str] = None
    extends_id: Optional[str] = None
    schema_json: dict[str, Any]
    x_backends: dict[str, Any]


class SchemaVersionCreate(SchemaVersionBase):
    """Fields for creating a new schema version."""

    pass


class SchemaVersion(SchemaVersionBase):
    """Full schema version from database."""

    id: str
    version: int
    status: SchemaStatus
    created_at: datetime
    created_by: Optional[str] = None
    checksum: str

    class Config:
        from_attributes = True

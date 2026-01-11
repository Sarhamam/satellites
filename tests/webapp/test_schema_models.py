"""Tests for schema models."""

import pytest
from datetime import datetime, timezone

from webapp.schema.models import (
    SchemaVersion,
    SchemaVersionCreate,
    SchemaStatus,
    BackendConfig,
    PostgresBackend,
    ElasticsearchBackend,
    RedisBackend,
    FaissBackend,
)


def test_schema_version_create():
    schema = SchemaVersionCreate(
        resource="users",
        schema_json={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["id", "email"],
        },
        x_backends={
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "email": {"type": "text", "nullable": False},
                },
            }
        },
    )
    assert schema.resource == "users"
    assert schema.tenant_id is None
    assert "id" in schema.schema_json["properties"]


def test_schema_version_full():
    schema = SchemaVersion(
        id="550e8400-e29b-41d4-a716-446655440000",
        resource="users",
        version=1,
        tenant_id=None,
        extends_id=None,
        schema_json={"type": "object", "properties": {}},
        x_backends={},
        status=SchemaStatus.DRAFT,
        created_at=datetime.now(timezone.utc),
        created_by="test",
        checksum="abc123",
    )
    assert schema.version == 1
    assert schema.status == SchemaStatus.DRAFT


def test_postgres_backend_config():
    config = PostgresBackend(
        table="users",
        primary_key=["id"],
        columns={
            "id": {"type": "text", "nullable": False},
            "name": {"type": "text", "nullable": True},
        },
        indexes=[
            {"name": "users_name_idx", "columns": ["name"]},
        ],
    )
    assert config.table == "users"
    assert len(config.columns) == 2


def test_elasticsearch_backend_config():
    config = ElasticsearchBackend(
        index="users",
        mappings={
            "properties": {
                "email": {"type": "keyword"},
                "name": {"type": "text"},
            }
        },
    )
    assert config.index == "users"


def test_redis_backend_config():
    config = RedisBackend(
        pattern="user:{id}",
        ttl_seconds=3600,
        encoding="json",
    )
    assert config.ttl_seconds == 3600


def test_faiss_backend_config():
    config = FaissBackend(
        namespace="user_embeddings",
        dimension=1536,
        id_field="id",
        metadata_fields=["email"],
    )
    assert config.dimension == 1536

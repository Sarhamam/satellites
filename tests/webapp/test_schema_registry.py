"""Tests for schema registry."""

import pytest
from datetime import datetime, timezone

from webapp.schema.registry import SchemaRegistry
from webapp.schema.models import SchemaVersionCreate, SchemaStatus


@pytest.fixture
async def registry():
    """Create a mock schema registry for testing."""
    # For now, use an in-memory implementation for tests
    reg = SchemaRegistry(connection_string=None)  # Will use mock
    await reg.start()
    yield reg
    await reg.stop()


@pytest.mark.asyncio
async def test_create_schema_version(registry):
    """Test creating a new schema version."""
    schema_data = SchemaVersionCreate(
        resource="users",
        schema_json={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string"},
            },
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

    version = await registry.create_schema_version(schema_data, created_by="test")

    assert version.resource == "users"
    assert version.version == 1
    assert version.status == SchemaStatus.DRAFT
    assert version.created_by == "test"
    assert version.checksum is not None


@pytest.mark.asyncio
async def test_get_schema_version_by_id(registry):
    """Test retrieving a schema version by ID."""
    # Create a schema first
    schema_data = SchemaVersionCreate(
        resource="products",
        schema_json={"type": "object", "properties": {}},
        x_backends={},
    )
    created = await registry.create_schema_version(schema_data, created_by="test")

    # Retrieve it
    retrieved = await registry.get_schema_version(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.resource == "products"


@pytest.mark.asyncio
async def test_get_latest_schema_version(registry):
    """Test retrieving the latest version of a schema."""
    # Create multiple versions
    schema_data = SchemaVersionCreate(
        resource="orders",
        schema_json={"type": "object", "properties": {"v": {"type": "integer"}}},
        x_backends={},
    )

    v1 = await registry.create_schema_version(schema_data, created_by="test")

    # Create v2
    schema_data.schema_json["properties"]["v"]["const"] = 2
    v2 = await registry.create_schema_version(schema_data, created_by="test")

    # Get latest
    latest = await registry.get_latest_schema_version("orders")

    assert latest is not None
    assert latest.version == 2
    assert latest.id == v2.id


@pytest.mark.asyncio
async def test_list_schema_versions(registry):
    """Test listing all versions of a schema."""
    # Create multiple versions
    schema_data = SchemaVersionCreate(
        resource="items",
        schema_json={"type": "object"},
        x_backends={},
    )

    await registry.create_schema_version(schema_data, created_by="test")
    await registry.create_schema_version(schema_data, created_by="test")
    await registry.create_schema_version(schema_data, created_by="test")

    versions = await registry.list_schema_versions("items")

    assert len(versions) == 3
    assert versions[0].version == 3  # Ordered by version desc
    assert versions[1].version == 2
    assert versions[2].version == 1


@pytest.mark.asyncio
async def test_publish_schema_version(registry):
    """Test publishing a draft schema."""
    schema_data = SchemaVersionCreate(
        resource="events",
        schema_json={"type": "object"},
        x_backends={},
    )

    version = await registry.create_schema_version(schema_data, created_by="test")
    assert version.status == SchemaStatus.DRAFT

    # Publish it
    published = await registry.publish_schema_version(version.id)

    assert published.status == SchemaStatus.PUBLISHED


@pytest.mark.asyncio
async def test_deprecate_schema_version(registry):
    """Test deprecating a published schema."""
    schema_data = SchemaVersionCreate(
        resource="logs",
        schema_json={"type": "object"},
        x_backends={},
    )

    version = await registry.create_schema_version(schema_data, created_by="test")
    published = await registry.publish_schema_version(version.id)

    # Deprecate it
    deprecated = await registry.deprecate_schema_version(published.id)

    assert deprecated.status == SchemaStatus.DEPRECATED


@pytest.mark.asyncio
async def test_delete_schema_version(registry):
    """Test deleting a draft schema version."""
    schema_data = SchemaVersionCreate(
        resource="temp",
        schema_json={"type": "object"},
        x_backends={},
    )

    version = await registry.create_schema_version(schema_data, created_by="test")

    # Delete it
    await registry.delete_schema_version(version.id)

    # Should not be retrievable
    deleted = await registry.get_schema_version(version.id)
    assert deleted is None

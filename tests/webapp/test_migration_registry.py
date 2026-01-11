"""Tests for migration registry."""

import pytest
from datetime import datetime, timezone

from webapp.migration.registry import MigrationRegistry
from webapp.migration.models import (
    Migration,
    MigrationOperation,
    MigrationOperationType,
    MigrationStatus,
)


@pytest.fixture
async def registry():
    """Create a mock migration registry for testing."""
    reg = MigrationRegistry(connection_string=None)  # In-memory
    await reg.start()
    yield reg
    await reg.stop()


@pytest.mark.asyncio
async def test_create_migration(registry):
    """Test creating a migration."""
    operations = [
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="users",
            details={"column": "phone", "column_type": "text", "nullable": True},
        )
    ]

    migration = await registry.create_migration(
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=operations,
        generated_sql="ALTER TABLE users ADD COLUMN phone TEXT;",
    )

    assert migration.resource == "users"
    assert migration.backend == "postgres"
    assert migration.status == MigrationStatus.PENDING
    assert len(migration.operations) == 1


@pytest.mark.asyncio
async def test_get_migration_by_id(registry):
    """Test retrieving a migration by ID."""
    operations = []
    created = await registry.create_migration(
        from_version_id=None,
        to_version_id="v1",
        resource="products",
        backend="postgres",
        operations=operations,
        generated_sql="CREATE TABLE products (...);",
    )

    retrieved = await registry.get_migration(created.id)
    assert retrieved is not None
    assert retrieved.id == created.id


@pytest.mark.asyncio
async def test_list_migrations(registry):
    """Test listing migrations for a resource."""
    operations = []

    await registry.create_migration(
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=operations,
        generated_sql="ALTER...",
    )

    await registry.create_migration(
        from_version_id="v2",
        to_version_id="v3",
        resource="users",
        backend="postgres",
        operations=operations,
        generated_sql="ALTER...",
    )

    migrations = await registry.list_migrations(resource="users")
    assert len(migrations) == 2


@pytest.mark.asyncio
async def test_mark_migration_applied(registry):
    """Test marking a migration as applied."""
    operations = []
    migration = await registry.create_migration(
        from_version_id="v1",
        to_version_id="v2",
        resource="logs",
        backend="postgres",
        operations=operations,
        generated_sql="ALTER...",
    )

    updated = await registry.mark_applied(migration.id)
    assert updated.status == MigrationStatus.APPLIED
    assert updated.applied_at is not None


@pytest.mark.asyncio
async def test_mark_migration_failed(registry):
    """Test marking a migration as failed."""
    operations = []
    migration = await registry.create_migration(
        from_version_id="v1",
        to_version_id="v2",
        resource="orders",
        backend="postgres",
        operations=operations,
        generated_sql="BAD SQL;",
    )

    updated = await registry.mark_failed(migration.id, error_message="Syntax error")
    assert updated.status == MigrationStatus.FAILED
    assert updated.error_message == "Syntax error"

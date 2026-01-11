"""Tests for migration models."""

import pytest
from datetime import datetime, timezone

from webapp.migration.models import (
    MigrationOperation,
    MigrationOperationType,
    Migration,
    MigrationStatus,
)


def test_migration_operation_create_table():
    """Test creating a create_table operation."""
    op = MigrationOperation(
        type=MigrationOperationType.CREATE_TABLE,
        table="users",
        details={
            "columns": {
                "id": {"type": "text", "nullable": False},
                "email": {"type": "text", "nullable": False},
            },
            "primary_key": ["id"],
        },
    )
    assert op.type == MigrationOperationType.CREATE_TABLE
    assert op.table == "users"
    assert "columns" in op.details


def test_migration_operation_add_column():
    """Test creating an add_column operation."""
    op = MigrationOperation(
        type=MigrationOperationType.ADD_COLUMN,
        table="users",
        details={
            "column": "phone",
            "column_type": "text",
            "nullable": True,
        },
    )
    assert op.type == MigrationOperationType.ADD_COLUMN
    assert op.details["column"] == "phone"


def test_migration_operation_drop_column():
    """Test creating a drop_column operation."""
    op = MigrationOperation(
        type=MigrationOperationType.DROP_COLUMN,
        table="users",
        details={"column": "phone"},
    )
    assert op.type == MigrationOperationType.DROP_COLUMN


def test_migration_operation_create_index():
    """Test creating a create_index operation."""
    op = MigrationOperation(
        type=MigrationOperationType.CREATE_INDEX,
        table="users",
        details={
            "name": "idx_users_email",
            "columns": ["email"],
            "unique": True,
        },
    )
    assert op.type == MigrationOperationType.CREATE_INDEX
    assert op.details["unique"] is True


def test_migration_create():
    """Test creating a migration."""
    migration = Migration(
        id="mig-001",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[
            MigrationOperation(
                type=MigrationOperationType.ADD_COLUMN,
                table="users",
                details={"column": "name", "column_type": "text", "nullable": True},
            )
        ],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )
    assert migration.resource == "users"
    assert migration.backend == "postgres"
    assert len(migration.operations) == 1
    assert migration.status == MigrationStatus.PENDING


def test_migration_with_sql():
    """Test migration with generated SQL."""
    migration = Migration(
        id="mig-002",
        from_version_id="v1",
        to_version_id="v2",
        resource="products",
        backend="postgres",
        operations=[],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        generated_sql="ALTER TABLE products ADD COLUMN price NUMERIC;",
    )
    assert migration.generated_sql is not None
    assert "ALTER TABLE" in migration.generated_sql


def test_migration_status_transitions():
    """Test migration status values."""
    assert MigrationStatus.PENDING == "pending"
    assert MigrationStatus.APPLIED == "applied"
    assert MigrationStatus.FAILED == "failed"
    assert MigrationStatus.ROLLED_BACK == "rolled_back"

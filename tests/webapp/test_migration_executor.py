"""Tests for migration executor."""

import pytest

from webapp.migration.executor import MigrationExecutor
from webapp.migration.models import Migration, MigrationOperation, MigrationOperationType, MigrationStatus
from datetime import datetime, timezone


def test_validate_migration():
    """Test migration validation."""
    executor = MigrationExecutor()

    # Valid migration
    migration = Migration(
        id="mig-1",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        generated_sql="ALTER TABLE users ADD COLUMN phone TEXT;",
    )

    # Should not raise
    executor.validate_migration(migration)


def test_validate_migration_no_sql():
    """Test validation fails without SQL."""
    executor = MigrationExecutor()

    migration = Migration(
        id="mig-1",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        generated_sql=None,  # Missing SQL
    )

    with pytest.raises(ValueError, match="no SQL"):
        executor.validate_migration(migration)


def test_validate_migration_already_applied():
    """Test validation fails if already applied."""
    executor = MigrationExecutor()

    migration = Migration(
        id="mig-1",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[],
        status=MigrationStatus.APPLIED,  # Already applied
        created_at=datetime.now(timezone.utc),
        generated_sql="ALTER TABLE users ADD COLUMN phone TEXT;",
    )

    with pytest.raises(ValueError, match="already applied"):
        executor.validate_migration(migration)


def test_dry_run():
    """Test dry run mode."""
    executor = MigrationExecutor()

    migration = Migration(
        id="mig-1",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[
            MigrationOperation(
                type=MigrationOperationType.ADD_COLUMN,
                table="users",
                details={"column": "phone", "column_type": "text"},
            )
        ],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        generated_sql="ALTER TABLE users ADD COLUMN phone TEXT;",
    )

    result = executor.dry_run(migration)

    assert result["valid"] is True
    assert "sql" in result
    assert "operations_count" in result
    assert result["operations_count"] == 1


def test_explain_migration():
    """Test migration explanation."""
    executor = MigrationExecutor()

    migration = Migration(
        id="mig-1",
        from_version_id="v1",
        to_version_id="v2",
        resource="users",
        backend="postgres",
        operations=[
            MigrationOperation(
                type=MigrationOperationType.ADD_COLUMN,
                table="users",
                details={"column": "phone", "column_type": "text"},
            ),
            MigrationOperation(
                type=MigrationOperationType.CREATE_INDEX,
                table="users",
                details={"name": "idx_users_phone", "columns": ["phone"]},
            ),
        ],
        status=MigrationStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        generated_sql="ALTER TABLE users...",
    )

    explanation = executor.explain(migration)

    assert "operations" in explanation
    assert len(explanation["operations"]) == 2
    assert explanation["operations"][0]["type"] == "add_column"
    assert explanation["operations"][1]["type"] == "create_index"

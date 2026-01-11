"""Tests for Postgres migration generator."""

import pytest

from webapp.migration.models import MigrationOperation, MigrationOperationType
from webapp.migration.generators.postgres import PostgresGenerator


def test_generate_create_table():
    """Test generating CREATE TABLE SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.CREATE_TABLE,
        table="users",
        details={
            "columns": {
                "id": {"type": "text", "nullable": False},
                "email": {"type": "text", "nullable": False},
                "name": {"type": "text", "nullable": True},
            },
            "primary_key": ["id"],
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "CREATE TABLE users" in sql
    assert "id TEXT NOT NULL" in sql
    assert "email TEXT NOT NULL" in sql
    assert "name TEXT" in sql
    assert "PRIMARY KEY (id)" in sql


def test_generate_add_column():
    """Test generating ADD COLUMN SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.ADD_COLUMN,
        table="users",
        details={
            "column": "phone",
            "column_type": "text",
            "nullable": True,
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "ALTER TABLE users" in sql
    assert "ADD COLUMN phone TEXT" in sql


def test_generate_add_column_not_null():
    """Test generating ADD COLUMN SQL with NOT NULL."""
    operation = MigrationOperation(
        type=MigrationOperationType.ADD_COLUMN,
        table="products",
        details={
            "column": "price",
            "column_type": "numeric",
            "nullable": False,
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "ALTER TABLE products" in sql
    assert "ADD COLUMN price NUMERIC NOT NULL" in sql


def test_generate_drop_column():
    """Test generating DROP COLUMN SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.DROP_COLUMN,
        table="users",
        details={"column": "phone"},
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "ALTER TABLE users" in sql
    assert "DROP COLUMN phone" in sql


def test_generate_modify_column():
    """Test generating ALTER COLUMN SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.MODIFY_COLUMN,
        table="users",
        details={
            "column": "age",
            "old_config": {"type": "integer", "nullable": True},
            "new_config": {"type": "integer", "nullable": False},
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "ALTER TABLE users" in sql
    assert "ALTER COLUMN age SET NOT NULL" in sql


def test_generate_create_index():
    """Test generating CREATE INDEX SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.CREATE_INDEX,
        table="users",
        details={
            "name": "idx_users_email",
            "columns": ["email"],
            "unique": False,
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "CREATE INDEX idx_users_email" in sql
    assert "ON users (email)" in sql


def test_generate_create_unique_index():
    """Test generating CREATE UNIQUE INDEX SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.CREATE_INDEX,
        table="users",
        details={
            "name": "idx_users_email_unique",
            "columns": ["email"],
            "unique": True,
        },
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "CREATE UNIQUE INDEX idx_users_email_unique" in sql
    assert "ON users (email)" in sql


def test_generate_drop_index():
    """Test generating DROP INDEX SQL."""
    operation = MigrationOperation(
        type=MigrationOperationType.DROP_INDEX,
        table="users",
        details={"name": "idx_users_email"},
    )

    generator = PostgresGenerator()
    sql = generator.generate_operation_sql(operation)

    assert "DROP INDEX idx_users_email" in sql


def test_generate_migration():
    """Test generating complete migration SQL."""
    operations = [
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="users",
            details={"column": "phone", "column_type": "text", "nullable": True},
        ),
        MigrationOperation(
            type=MigrationOperationType.CREATE_INDEX,
            table="users",
            details={"name": "idx_users_phone", "columns": ["phone"]},
        ),
    ]

    generator = PostgresGenerator()
    sql = generator.generate_migration_sql(operations)

    assert "ALTER TABLE users" in sql
    assert "ADD COLUMN phone TEXT" in sql
    assert "CREATE INDEX idx_users_phone" in sql
    # Should have multiple statements separated by semicolons
    assert sql.count(";") >= 2

"""Tests for schema differ."""

import pytest

from webapp.schema.models import SchemaVersion, SchemaStatus
from webapp.migration.differ import SchemaDiffer
from webapp.migration.models import MigrationOperationType
from datetime import datetime, timezone


def test_no_changes():
    """Test when schemas are identical."""
    schema_v1 = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "email": {"type": "string"},
        },
    }

    backends_v1 = {
        "postgres": {
            "table": "users",
            "primary_key": ["id"],
            "columns": {
                "id": {"type": "text", "nullable": False},
                "email": {"type": "text", "nullable": False},
            },
        }
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(backends_v1["postgres"], backends_v1["postgres"])

    assert len(operations) == 0


def test_add_column():
    """Test detecting added column."""
    old_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
        },
    }

    new_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "email": {"type": "text", "nullable": True},
        },
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(old_backend, new_backend)

    assert len(operations) == 1
    assert operations[0].type == MigrationOperationType.ADD_COLUMN
    assert operations[0].details["column"] == "email"


def test_drop_column():
    """Test detecting dropped column."""
    old_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "phone": {"type": "text", "nullable": True},
        },
    }

    new_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
        },
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(old_backend, new_backend)

    assert len(operations) == 1
    assert operations[0].type == MigrationOperationType.DROP_COLUMN
    assert operations[0].details["column"] == "phone"


def test_modify_column():
    """Test detecting modified column."""
    old_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "age": {"type": "integer", "nullable": True},
        },
    }

    new_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "age": {"type": "integer", "nullable": False},  # Changed nullable
        },
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(old_backend, new_backend)

    assert len(operations) == 1
    assert operations[0].type == MigrationOperationType.MODIFY_COLUMN
    assert operations[0].details["column"] == "age"


def test_add_index():
    """Test detecting added index."""
    old_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "email": {"type": "text", "nullable": False},
        },
        "indexes": [],
    }

    new_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "email": {"type": "text", "nullable": False},
        },
        "indexes": [
            {"name": "idx_users_email", "columns": ["email"], "unique": True}
        ],
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(old_backend, new_backend)

    assert len(operations) == 1
    assert operations[0].type == MigrationOperationType.CREATE_INDEX
    assert operations[0].details["name"] == "idx_users_email"


def test_multiple_changes():
    """Test detecting multiple changes at once."""
    old_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "name": {"type": "text", "nullable": True},
        },
        "indexes": [],
    }

    new_backend = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "name": {"type": "text", "nullable": False},  # Modified
            "email": {"type": "text", "nullable": True},  # Added
        },
        "indexes": [
            {"name": "idx_users_email", "columns": ["email"]}  # Added
        ],
    }

    differ = SchemaDiffer()
    operations = differ.diff_postgres(old_backend, new_backend)

    assert len(operations) == 3
    operation_types = {op.type for op in operations}
    assert MigrationOperationType.ADD_COLUMN in operation_types
    assert MigrationOperationType.MODIFY_COLUMN in operation_types
    assert MigrationOperationType.CREATE_INDEX in operation_types

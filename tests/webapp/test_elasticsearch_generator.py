"""Tests for Elasticsearch migration generator."""

import pytest
import json

from webapp.migration.models import MigrationOperation, MigrationOperationType
from webapp.migration.generators.elasticsearch import ElasticsearchGenerator


def test_generate_create_index():
    """Test generating create index command."""
    operation = MigrationOperation(
        type=MigrationOperationType.CREATE_TABLE,  # Maps to create index
        table="users",
        details={
            "index": "users",
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "email": {"type": "keyword"},
                    "name": {"type": "text"},
                }
            },
        },
    )

    generator = ElasticsearchGenerator()
    command = generator.generate_operation_command(operation)

    assert command["action"] == "create_index"
    assert command["index"] == "users"
    assert "properties" in command["body"]["mappings"]
    assert "id" in command["body"]["mappings"]["properties"]


def test_generate_add_field():
    """Test generating add field mapping command."""
    operation = MigrationOperation(
        type=MigrationOperationType.ADD_COLUMN,
        table="users",
        details={
            "index": "users",
            "field": "phone",
            "mapping": {"type": "keyword"},
        },
    )

    generator = ElasticsearchGenerator()
    command = generator.generate_operation_command(operation)

    assert command["action"] == "update_mapping"
    assert command["index"] == "users"
    assert "phone" in command["body"]["properties"]
    assert command["body"]["properties"]["phone"]["type"] == "keyword"


def test_generate_update_settings():
    """Test generating update settings command."""
    operation = MigrationOperation(
        type=MigrationOperationType.MODIFY_COLUMN,
        table="users",
        details={
            "index": "users",
            "settings": {
                "number_of_replicas": 2,
            },
        },
    )

    generator = ElasticsearchGenerator()
    command = generator.generate_operation_command(operation)

    assert command["action"] == "update_settings"
    assert command["index"] == "users"
    assert command["body"]["number_of_replicas"] == 2


def test_generate_migration_script():
    """Test generating complete migration script."""
    operations = [
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="products",
            details={
                "index": "products",
                "field": "price",
                "mapping": {"type": "double"},
            },
        ),
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="products",
            details={
                "index": "products",
                "field": "description",
                "mapping": {"type": "text"},
            },
        ),
    ]

    generator = ElasticsearchGenerator()
    script = generator.generate_migration_script(operations)

    # Should be valid JSON
    commands = json.loads(script)
    assert isinstance(commands, list)
    assert len(commands) == 2
    assert all(cmd["action"] == "update_mapping" for cmd in commands)


def test_python_api_format():
    """Test generating Python API calls format."""
    operations = [
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="logs",
            details={
                "index": "logs",
                "field": "level",
                "mapping": {"type": "keyword"},
            },
        ),
    ]

    generator = ElasticsearchGenerator()
    python_code = generator.generate_python_api(operations)

    assert "es.indices.put_mapping" in python_code or "client.indices.put_mapping" in python_code
    assert "logs" in python_code
    assert "level" in python_code

"""Tests for schema validator."""

import pytest
from jsonschema.exceptions import ValidationError

from webapp.schema.validator import SchemaValidator


def test_validate_valid_schema():
    """Test validation of a valid JSON Schema."""
    validator = SchemaValidator()

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["id", "email"],
    }

    # Should not raise any exception
    validator.validate_schema(schema)


def test_validate_invalid_schema():
    """Test validation of an invalid JSON Schema."""
    validator = SchemaValidator()

    # Invalid - 'type' field has invalid value
    schema = {
        "type": "invalid_type",
        "properties": {
            "id": {"type": "string"},
        },
    }

    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        validator.validate_schema(schema)


def test_validate_data_against_schema():
    """Test validating data against a schema."""
    validator = SchemaValidator()

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["id"],
    }

    # Valid data
    valid_data = {"id": "123", "age": 25}
    validator.validate_data(valid_data, schema)

    # Invalid data - missing required field
    with pytest.raises(ValidationError):
        validator.validate_data({}, schema)

    # Invalid data - wrong type
    with pytest.raises(ValidationError):
        validator.validate_data({"id": "123", "age": "not-a-number"}, schema)


def test_validate_backends_config():
    """Test validation of backend configurations."""
    validator = SchemaValidator()

    # Valid Postgres backend
    postgres_config = {
        "table": "users",
        "primary_key": ["id"],
        "columns": {
            "id": {"type": "text", "nullable": False},
            "name": {"type": "text", "nullable": True},
        },
    }
    validator.validate_postgres_backend(postgres_config)

    # Invalid - missing required field
    with pytest.raises(ValueError, match="Missing required field"):
        validator.validate_postgres_backend({"table": "users"})

    # Valid Elasticsearch backend
    es_config = {
        "index": "users",
        "mappings": {
            "properties": {
                "email": {"type": "keyword"},
            }
        },
    }
    validator.validate_elasticsearch_backend(es_config)

    # Invalid - missing mappings
    with pytest.raises(ValueError, match="Missing required field"):
        validator.validate_elasticsearch_backend({"index": "users"})


def test_validate_schema_consistency():
    """Test consistency validation between schema and backends."""
    validator = SchemaValidator()

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["id", "email"],
    }

    backends = {
        "postgres": {
            "table": "users",
            "primary_key": ["id"],
            "columns": {
                "id": {"type": "text", "nullable": False},
                "email": {"type": "text", "nullable": False},
            },
        }
    }

    # Should pass - schema fields match backend columns
    validator.validate_consistency(schema, backends)

    # Should fail - backend has column not in schema
    backends_with_extra = {
        "postgres": {
            "table": "users",
            "primary_key": ["id"],
            "columns": {
                "id": {"type": "text", "nullable": False},
                "email": {"type": "text", "nullable": False},
                "extra_field": {"type": "text", "nullable": True},
            },
        }
    }

    with pytest.raises(ValueError, match="not defined in schema"):
        validator.validate_consistency(schema, backends_with_extra)

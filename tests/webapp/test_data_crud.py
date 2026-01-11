"""Tests for data CRUD operations."""

import pytest

from webapp.data.crud import DataCRUD


def test_build_query_simple():
    """Test building a simple query."""
    crud = DataCRUD()

    filters = {"email": "test@example.com"}
    query = crud.build_query(filters)

    assert query["email"] == "test@example.com"


def test_build_query_operators():
    """Test building query with operators."""
    crud = DataCRUD()

    filters = {
        "age__gte": 18,
        "age__lt": 65,
        "name__contains": "john",
    }

    query = crud.build_query(filters)

    assert "age__gte" in query or "age" in query
    assert query.get("age__gte") == 18 or query.get("age", {}).get("$gte") == 18


def test_validate_data_against_schema():
    """Test data validation against schema."""
    crud = DataCRUD()

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["id", "email"],
    }

    valid_data = {"id": "123", "email": "test@example.com"}

    # Should not raise
    crud.validate_data(valid_data, schema)


def test_validate_data_invalid():
    """Test validation fails for invalid data."""
    crud = DataCRUD()

    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
        },
        "required": ["id"],
    }

    invalid_data = {}  # Missing required field

    with pytest.raises(Exception):  # Should raise validation error
        crud.validate_data(invalid_data, schema)


def test_prepare_for_backend():
    """Test preparing data for specific backend."""
    crud = DataCRUD()

    data = {
        "id": "123",
        "email": "test@example.com",
        "created_at": "2024-01-01T00:00:00Z",
    }

    # For Postgres
    pg_data = crud.prepare_for_backend(data, "postgres")
    assert "id" in pg_data

    # For Elasticsearch
    es_data = crud.prepare_for_backend(data, "elasticsearch")
    assert "id" in es_data

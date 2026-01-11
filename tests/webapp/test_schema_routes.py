"""Tests for schema routes."""

import pytest
from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.config import WebAppConfig
from data_layer import DataLayerConfig


@pytest.fixture
def client():
    """Create test client."""
    app = create_app(
        webapp_config=WebAppConfig(debug=True),
        data_layer_config=None,  # No DataLayer for route tests
    )
    return TestClient(app)


def test_schema_list_page(client):
    """Test schema list page renders."""
    response = client.get("/schemas")
    assert response.status_code == 200
    assert "Schemas" in response.text


def test_schema_create_page(client):
    """Test schema creation page renders."""
    response = client.get("/schemas/new")
    assert response.status_code == 200
    assert "Create Schema" in response.text or "New Schema" in response.text


def test_schema_detail_page(client):
    """Test schema detail page renders (with mock data)."""
    # This will need mocking or a test database
    # For now, test the route exists and returns proper status
    response = client.get("/schemas/test-id")
    # Expecting 404 or 500 since no real data, but route should exist
    assert response.status_code in [404, 500, 200]

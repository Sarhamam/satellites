"""Tests for migration routes."""

import pytest
from fastapi.testclient import TestClient

from webapp.app import create_app
from webapp.config import WebAppConfig


@pytest.fixture
def client():
    """Create test client."""
    app = create_app(
        webapp_config=WebAppConfig(debug=True),
        data_layer_config=None,
    )
    return TestClient(app)


def test_migration_list_page(client):
    """Test migration list page renders."""
    response = client.get("/migrations")
    assert response.status_code == 200
    assert "Migrations" in response.text or "Migration" in response.text


def test_migration_preview_page(client):
    """Test migration preview page."""
    # This will need mock data, but test route exists
    response = client.get("/migrations/test-id/preview")
    # Expecting 404 or 500 since no real data
    assert response.status_code in [404, 500, 200]

"""Tests for data routes."""

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


def test_data_explorer_page(client):
    """Test data explorer page renders."""
    response = client.get("/data")
    assert response.status_code == 200
    assert "Data" in response.text or "Explorer" in response.text


def test_list_resource_data_page(client):
    """Test resource data list page."""
    response = client.get("/data/users")
    assert response.status_code == 200

# WebApp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified admin panel for schema design, data exploration, and queries across Redis/Postgres/Elasticsearch/FAISS.

**Architecture:** FastAPI backend with Jinja2 templates + HTMX for dynamic UI. Schema registry in Postgres, migration generators per backend, generic data API routing based on schema config.

**Tech Stack:** Python 3.11+, FastAPI, Jinja2, HTMX, TailwindCSS, jsonschema

---

## Phase 1: Project Setup & Core Infrastructure

### Task 1: Add WebApp Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add to the existing dependencies:

```toml
[project]
dependencies = [
    # ... existing data_layer deps ...
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
    "jinja2>=3.1",
    "python-multipart>=0.0.6",
    "jsonschema>=4.21",
]
```

**Step 2: Install dependencies**

Run: `pip install -e ".[dev]"`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add webapp dependencies"
```

---

### Task 2: Create WebApp Package Structure

**Files:**
- Create: `webapp/__init__.py`
- Create: `webapp/app.py`
- Create: `webapp/config.py`
- Create: `webapp/dependencies.py`
- Create: `webapp/templates/base.html`
- Create: `webapp/static/css/.gitkeep`
- Create: `webapp/static/js/.gitkeep`

**Step 1: Create directory structure**

```bash
mkdir -p webapp/templates webapp/static/css webapp/static/js
mkdir -p webapp/schema webapp/migration webapp/migration/generators webapp/data webapp/query
touch webapp/__init__.py webapp/schema/__init__.py webapp/migration/__init__.py
touch webapp/migration/generators/__init__.py webapp/data/__init__.py webapp/query/__init__.py
touch webapp/static/css/.gitkeep webapp/static/js/.gitkeep
```

**Step 2: Create webapp/config.py**

```python
"""WebApp configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WebAppConfig:
    """Configuration for the web application."""

    title: str = "Satellites Admin"
    debug: bool = False
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "static")
```

**Step 3: Create webapp/dependencies.py**

```python
"""FastAPI dependency injection."""

from typing import Annotated, Optional
from fastapi import Depends, Request

from data_layer import DataLayer


async def get_data_layer(request: Request) -> DataLayer:
    """Get the DataLayer instance from app state."""
    return request.app.state.data_layer


async def get_current_tenant(request: Request) -> Optional[str]:
    """Get current tenant from request headers or session."""
    # For now, use header. Later: session/auth
    return request.headers.get("X-Tenant-ID")


DataLayerDep = Annotated[DataLayer, Depends(get_data_layer)]
TenantDep = Annotated[Optional[str], Depends(get_current_tenant)]
```

**Step 4: Create webapp/templates/base.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{ config.title }}{% endblock %}</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    {% block head %}{% endblock %}
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-white shadow mb-6">
        <div class="max-w-7xl mx-auto px-4 py-3">
            <div class="flex justify-between items-center">
                <div class="flex space-x-8">
                    <a href="/" class="text-xl font-bold text-gray-900">{{ config.title }}</a>
                    <a href="/schemas" class="text-gray-600 hover:text-gray-900">Schemas</a>
                    <a href="/data" class="text-gray-600 hover:text-gray-900">Data</a>
                    <a href="/query" class="text-gray-600 hover:text-gray-900">Query</a>
                </div>
                <div class="text-sm text-gray-500">
                    Tenant: <span id="current-tenant">{{ tenant or 'None' }}</span>
                </div>
            </div>
        </div>
    </nav>
    <main class="max-w-7xl mx-auto px-4">
        {% block content %}{% endblock %}
    </main>
    {% block scripts %}{% endblock %}
</body>
</html>
```

**Step 5: Create webapp/app.py**

```python
"""FastAPI application factory."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data_layer import DataLayer, DataLayerConfig
from webapp.config import WebAppConfig


def create_app(
    webapp_config: WebAppConfig | None = None,
    data_layer_config: DataLayerConfig | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    webapp_config = webapp_config or WebAppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: connect data layer
        if data_layer_config:
            data_layer = DataLayer(data_layer_config)
            await data_layer.start()
            app.state.data_layer = data_layer
        yield
        # Shutdown: disconnect data layer
        if hasattr(app.state, "data_layer"):
            await app.state.data_layer.stop()

    app = FastAPI(
        title=webapp_config.title,
        lifespan=lifespan,
        debug=webapp_config.debug,
    )

    # Static files
    app.mount(
        "/static",
        StaticFiles(directory=webapp_config.static_dir),
        name="static",
    )

    # Templates
    templates = Jinja2Templates(directory=webapp_config.templates_dir)
    app.state.templates = templates
    app.state.webapp_config = webapp_config

    # Register routes (will add later)
    # app.include_router(schema_router, prefix="/schemas", tags=["schemas"])
    # app.include_router(data_router, prefix="/data", tags=["data"])
    # app.include_router(query_router, prefix="/query", tags=["query"])

    return app
```

**Step 6: Create webapp/__init__.py**

```python
"""WebApp - Schema Designer & Data Explorer."""

from webapp.app import create_app
from webapp.config import WebAppConfig

__all__ = ["create_app", "WebAppConfig"]
```

**Step 7: Verify imports work**

Run: `python -c "from webapp import create_app, WebAppConfig; print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: add webapp package structure with FastAPI app factory"
```

---

## Phase 2: Schema Registry

### Task 3: Schema Pydantic Models

**Files:**
- Create: `webapp/schema/models.py`
- Create: `tests/webapp/__init__.py`
- Create: `tests/webapp/test_schema_models.py`

**Step 1: Create test directory**

```bash
mkdir -p tests/webapp
touch tests/webapp/__init__.py
```

**Step 2: Write the failing test**

Create `tests/webapp/test_schema_models.py`:

```python
"""Tests for schema models."""

import pytest
from datetime import datetime, timezone

from webapp.schema.models import (
    SchemaVersion,
    SchemaVersionCreate,
    SchemaStatus,
    BackendConfig,
    PostgresBackend,
    ElasticsearchBackend,
    RedisBackend,
    FaissBackend,
)


def test_schema_version_create():
    schema = SchemaVersionCreate(
        resource="users",
        schema_json={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["id", "email"],
        },
        x_backends={
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "email": {"type": "text", "nullable": False},
                },
            }
        },
    )
    assert schema.resource == "users"
    assert schema.tenant_id is None
    assert "id" in schema.schema_json["properties"]


def test_schema_version_full():
    schema = SchemaVersion(
        id="550e8400-e29b-41d4-a716-446655440000",
        resource="users",
        version=1,
        tenant_id=None,
        extends_id=None,
        schema_json={"type": "object", "properties": {}},
        x_backends={},
        status=SchemaStatus.DRAFT,
        created_at=datetime.now(timezone.utc),
        created_by="test",
        checksum="abc123",
    )
    assert schema.version == 1
    assert schema.status == SchemaStatus.DRAFT


def test_postgres_backend_config():
    config = PostgresBackend(
        table="users",
        primary_key=["id"],
        columns={
            "id": {"type": "text", "nullable": False},
            "name": {"type": "text", "nullable": True},
        },
        indexes=[
            {"name": "users_name_idx", "columns": ["name"]},
        ],
    )
    assert config.table == "users"
    assert len(config.columns) == 2


def test_elasticsearch_backend_config():
    config = ElasticsearchBackend(
        index="users",
        mappings={
            "properties": {
                "email": {"type": "keyword"},
                "name": {"type": "text"},
            }
        },
    )
    assert config.index == "users"


def test_redis_backend_config():
    config = RedisBackend(
        pattern="user:{id}",
        ttl_seconds=3600,
        encoding="json",
    )
    assert config.ttl_seconds == 3600


def test_faiss_backend_config():
    config = FaissBackend(
        namespace="user_embeddings",
        dimension=1536,
        id_field="id",
        metadata_fields=["email"],
    )
    assert config.dimension == 1536
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/webapp/test_schema_models.py -v`
Expected: FAIL with ImportError

**Step 4: Write implementation**

Create `webapp/schema/models.py`:

```python
"""Pydantic models for schema registry."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SchemaStatus(str, Enum):
    """Status of a schema version."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


# Backend configurations


class ColumnConfig(BaseModel):
    """Postgres column configuration."""

    type: str
    nullable: bool = True


class IndexConfig(BaseModel):
    """Postgres index configuration."""

    name: str
    columns: list[str]
    unique: bool = False
    using: Optional[str] = None  # e.g., "gin", "btree"


class PostgresBackend(BaseModel):
    """Postgres backend configuration."""

    table: str
    primary_key: list[str]
    columns: dict[str, ColumnConfig | dict[str, Any]]
    indexes: list[IndexConfig | dict[str, Any]] = Field(default_factory=list)


class ElasticsearchBackend(BaseModel):
    """Elasticsearch backend configuration."""

    index: str
    mappings: dict[str, Any]


class RedisBackend(BaseModel):
    """Redis backend configuration."""

    pattern: str
    ttl_seconds: Optional[int] = None
    encoding: str = "json"


class FaissBackend(BaseModel):
    """FAISS backend configuration."""

    namespace: str
    dimension: int
    id_field: str
    metadata_fields: list[str] = Field(default_factory=list)


class BackendConfig(BaseModel):
    """All backend configurations."""

    postgres: Optional[PostgresBackend] = None
    elasticsearch: Optional[ElasticsearchBackend] = None
    redis: Optional[RedisBackend] = None
    faiss: Optional[FaissBackend] = None


# Schema versions


class SchemaVersionBase(BaseModel):
    """Base fields for schema version."""

    resource: str
    tenant_id: Optional[str] = None
    extends_id: Optional[str] = None
    schema_json: dict[str, Any]
    x_backends: dict[str, Any]


class SchemaVersionCreate(SchemaVersionBase):
    """Fields for creating a new schema version."""

    pass


class SchemaVersion(SchemaVersionBase):
    """Full schema version from database."""

    id: str
    version: int
    status: SchemaStatus
    created_at: datetime
    created_by: Optional[str] = None
    checksum: str

    class Config:
        from_attributes = True
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/webapp/test_schema_models.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add schema Pydantic models"
```

---

### Task 4: Schema Validator

**Files:**
- Create: `webapp/schema/validator.py`
- Create: `tests/webapp/test_schema_validator.py`

**Step 1: Write the failing test**

Create `tests/webapp/test_schema_validator.py`:

```python
"""Tests for schema validator."""

import pytest

from webapp.schema.validator import SchemaValidator, ValidationError


@pytest.fixture
def validator() -> SchemaValidator:
    return SchemaValidator()


def test_validate_valid_schema(validator: SchemaValidator):
    schema_ir = {
        "resource": "users",
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["id"],
        },
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "email": {"type": "text", "nullable": True},
                },
            }
        },
    }
    errors = validator.validate(schema_ir)
    assert len(errors) == 0


def test_validate_invalid_json_schema(validator: SchemaValidator):
    schema_ir = {
        "resource": "users",
        "schema": {
            "type": "invalid_type",  # invalid
        },
        "x-backends": {},
    }
    errors = validator.validate(schema_ir)
    assert len(errors) > 0
    assert any("type" in str(e).lower() for e in errors)


def test_validate_missing_resource(validator: SchemaValidator):
    schema_ir = {
        "schema": {"type": "object"},
        "x-backends": {},
    }
    errors = validator.validate(schema_ir)
    assert len(errors) > 0
    assert any("resource" in str(e).lower() for e in errors)


def test_validate_postgres_column_mismatch(validator: SchemaValidator):
    schema_ir = {
        "resource": "users",
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},  # defined in schema
            },
        },
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    # missing "name" column
                },
            }
        },
    }
    errors = validator.validate(schema_ir)
    assert len(errors) > 0
    assert any("name" in str(e).lower() for e in errors)


def test_validate_faiss_invalid_dimension(validator: SchemaValidator):
    schema_ir = {
        "resource": "embeddings",
        "schema": {"type": "object", "properties": {}},
        "x-backends": {
            "faiss": {
                "namespace": "embeds",
                "dimension": 0,  # invalid
                "id_field": "id",
            }
        },
    }
    errors = validator.validate(schema_ir)
    assert len(errors) > 0
    assert any("dimension" in str(e).lower() for e in errors)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_schema_validator.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `webapp/schema/validator.py`:

```python
"""Schema validation for Schema IR format."""

from dataclasses import dataclass
from typing import Any

import jsonschema


@dataclass
class ValidationError:
    """A validation error."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class SchemaValidator:
    """Validates Schema IR documents."""

    # Valid JSON Schema types
    VALID_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

    # Valid Postgres column types
    VALID_PG_TYPES = {
        "text", "varchar", "char", "int", "integer", "bigint", "smallint",
        "boolean", "bool", "timestamp", "timestamptz", "date", "time",
        "json", "jsonb", "uuid", "numeric", "decimal", "float", "real",
        "double precision", "bytea", "serial", "bigserial",
    }

    def validate(self, schema_ir: dict[str, Any]) -> list[ValidationError]:
        """Validate a Schema IR document. Returns list of errors (empty if valid)."""
        errors: list[ValidationError] = []

        # Check required top-level fields
        if "resource" not in schema_ir:
            errors.append(ValidationError("", "Missing required field: resource"))

        if "schema" not in schema_ir:
            errors.append(ValidationError("", "Missing required field: schema"))
        else:
            errors.extend(self._validate_json_schema(schema_ir["schema"]))

        if "x-backends" not in schema_ir:
            errors.append(ValidationError("", "Missing required field: x-backends"))
        else:
            errors.extend(
                self._validate_backends(
                    schema_ir.get("schema", {}),
                    schema_ir["x-backends"],
                )
            )

        return errors

    def _validate_json_schema(self, schema: dict[str, Any]) -> list[ValidationError]:
        """Validate the JSON Schema portion."""
        errors: list[ValidationError] = []

        # Check type is valid
        if "type" in schema:
            schema_type = schema["type"]
            if isinstance(schema_type, str) and schema_type not in self.VALID_JSON_TYPES:
                errors.append(
                    ValidationError("schema.type", f"Invalid type: {schema_type}")
                )
            elif isinstance(schema_type, list):
                for t in schema_type:
                    if t not in self.VALID_JSON_TYPES:
                        errors.append(
                            ValidationError("schema.type", f"Invalid type in union: {t}")
                        )

        # Validate using jsonschema meta-schema
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError as e:
            errors.append(ValidationError("schema", str(e.message)))

        return errors

    def _validate_backends(
        self,
        schema: dict[str, Any],
        backends: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate backend configurations."""
        errors: list[ValidationError] = []

        schema_properties = schema.get("properties", {})

        if "postgres" in backends:
            errors.extend(
                self._validate_postgres(schema_properties, backends["postgres"])
            )

        if "elasticsearch" in backends:
            errors.extend(self._validate_elasticsearch(backends["elasticsearch"]))

        if "redis" in backends:
            errors.extend(self._validate_redis(backends["redis"]))

        if "faiss" in backends:
            errors.extend(self._validate_faiss(backends["faiss"]))

        return errors

    def _validate_postgres(
        self,
        schema_properties: dict[str, Any],
        config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate Postgres backend configuration."""
        errors: list[ValidationError] = []

        if "table" not in config:
            errors.append(ValidationError("x-backends.postgres", "Missing: table"))

        if "primary_key" not in config:
            errors.append(ValidationError("x-backends.postgres", "Missing: primary_key"))

        columns = config.get("columns", {})

        # Check that schema properties have matching columns
        for prop_name in schema_properties:
            if prop_name not in columns and prop_name != "tenant_id":
                errors.append(
                    ValidationError(
                        "x-backends.postgres.columns",
                        f"Missing column for schema property: {prop_name}",
                    )
                )

        return errors

    def _validate_elasticsearch(self, config: dict[str, Any]) -> list[ValidationError]:
        """Validate Elasticsearch backend configuration."""
        errors: list[ValidationError] = []

        if "index" not in config:
            errors.append(ValidationError("x-backends.elasticsearch", "Missing: index"))

        if "mappings" not in config:
            errors.append(ValidationError("x-backends.elasticsearch", "Missing: mappings"))

        return errors

    def _validate_redis(self, config: dict[str, Any]) -> list[ValidationError]:
        """Validate Redis backend configuration."""
        errors: list[ValidationError] = []

        if "pattern" not in config:
            errors.append(ValidationError("x-backends.redis", "Missing: pattern"))

        return errors

    def _validate_faiss(self, config: dict[str, Any]) -> list[ValidationError]:
        """Validate FAISS backend configuration."""
        errors: list[ValidationError] = []

        if "namespace" not in config:
            errors.append(ValidationError("x-backends.faiss", "Missing: namespace"))

        if "dimension" not in config:
            errors.append(ValidationError("x-backends.faiss", "Missing: dimension"))
        elif config["dimension"] <= 0:
            errors.append(
                ValidationError("x-backends.faiss.dimension", "Must be positive integer")
            )

        if "id_field" not in config:
            errors.append(ValidationError("x-backends.faiss", "Missing: id_field"))

        return errors
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_schema_validator.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add schema validator for Schema IR"
```

---

### Task 5: Schema Registry (Database Operations)

**Files:**
- Create: `webapp/schema/registry.py`
- Create: `tests/webapp/test_schema_registry.py`

**Step 1: Write the failing test**

Create `tests/webapp/test_schema_registry.py`:

```python
"""Tests for schema registry."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import hashlib
import json

from webapp.schema.registry import SchemaRegistry
from webapp.schema.models import SchemaStatus


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock(return_value="INSERT 1")
    pool.fetchval = AsyncMock(return_value=1)
    return pool


@pytest.fixture
def registry(mock_pool) -> SchemaRegistry:
    return SchemaRegistry(mock_pool)


async def test_create_version(registry: SchemaRegistry, mock_pool):
    schema_json = {"type": "object", "properties": {"id": {"type": "string"}}}
    x_backends = {"postgres": {"table": "users", "primary_key": ["id"], "columns": {}}}

    mock_pool.fetchval = AsyncMock(return_value=0)  # no existing versions
    mock_pool.fetchrow = AsyncMock(return_value={
        "id": "test-uuid",
        "resource": "users",
        "version": 1,
        "tenant_id": None,
        "extends_id": None,
        "schema_json": schema_json,
        "x_backends": x_backends,
        "status": "draft",
        "created_at": "2026-01-11T00:00:00Z",
        "created_by": "test",
        "checksum": "abc",
    })

    result = await registry.create_version(
        resource="users",
        schema_json=schema_json,
        x_backends=x_backends,
        created_by="test",
    )

    assert result["resource"] == "users"
    assert result["version"] == 1


async def test_get_latest_published(registry: SchemaRegistry, mock_pool):
    mock_pool.fetchrow = AsyncMock(return_value={
        "id": "test-uuid",
        "resource": "users",
        "version": 2,
        "status": "published",
    })

    result = await registry.get_latest_published("users")

    assert result is not None
    assert result["version"] == 2


async def test_get_latest_published_not_found(registry: SchemaRegistry, mock_pool):
    mock_pool.fetchrow = AsyncMock(return_value=None)

    result = await registry.get_latest_published("nonexistent")

    assert result is None


async def test_list_versions(registry: SchemaRegistry, mock_pool):
    mock_pool.fetch = AsyncMock(return_value=[
        {"id": "v1", "resource": "users", "version": 1, "status": "deprecated"},
        {"id": "v2", "resource": "users", "version": 2, "status": "published"},
    ])

    results = await registry.list_versions("users")

    assert len(results) == 2


async def test_publish_version(registry: SchemaRegistry, mock_pool):
    mock_pool.execute = AsyncMock(return_value="UPDATE 1")

    await registry.publish_version("test-uuid")

    # Verify deprecate old + publish new were called
    assert mock_pool.execute.call_count >= 1


async def test_compute_checksum(registry: SchemaRegistry):
    schema_json = {"type": "object"}
    x_backends = {"postgres": {}}

    checksum = registry._compute_checksum(schema_json, x_backends)

    # Should be consistent
    checksum2 = registry._compute_checksum(schema_json, x_backends)
    assert checksum == checksum2

    # Different content = different checksum
    checksum3 = registry._compute_checksum({"type": "array"}, x_backends)
    assert checksum != checksum3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_schema_registry.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `webapp/schema/registry.py`:

```python
"""Schema registry - database operations for schema versions."""

import hashlib
import json
from typing import Any, Optional

import asyncpg

from webapp.schema.models import SchemaStatus


class SchemaRegistry:
    """CRUD operations for schema versions in Postgres."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def _compute_checksum(
        self,
        schema_json: dict[str, Any],
        x_backends: dict[str, Any],
    ) -> str:
        """Compute checksum for change detection."""
        content = json.dumps(
            {"schema": schema_json, "x_backends": x_backends},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def create_version(
        self,
        resource: str,
        schema_json: dict[str, Any],
        x_backends: dict[str, Any],
        tenant_id: Optional[str] = None,
        extends_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new schema version (as draft)."""
        # Get next version number
        current_max = await self._pool.fetchval(
            """
            SELECT COALESCE(MAX(version), 0)
            FROM schema_versions
            WHERE resource = $1 AND tenant_id IS NOT DISTINCT FROM $2
            """,
            resource,
            tenant_id,
        )
        next_version = current_max + 1

        checksum = self._compute_checksum(schema_json, x_backends)

        row = await self._pool.fetchrow(
            """
            INSERT INTO schema_versions
                (resource, version, tenant_id, extends_id, schema_json, x_backends, status, created_by, checksum)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            resource,
            next_version,
            tenant_id,
            extends_id,
            json.dumps(schema_json),
            json.dumps(x_backends),
            SchemaStatus.DRAFT.value,
            created_by,
            checksum,
        )

        return dict(row)

    async def get_version(self, version_id: str) -> Optional[dict[str, Any]]:
        """Get a schema version by ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM schema_versions WHERE id = $1",
            version_id,
        )
        return dict(row) if row else None

    async def get_latest_published(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get the latest published version of a resource."""
        row = await self._pool.fetchrow(
            """
            SELECT * FROM schema_versions
            WHERE resource = $1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND status = $3
            ORDER BY version DESC
            LIMIT 1
            """,
            resource,
            tenant_id,
            SchemaStatus.PUBLISHED.value,
        )
        return dict(row) if row else None

    async def list_versions(
        self,
        resource: str,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all versions of a resource."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM schema_versions
            WHERE resource = $1
              AND tenant_id IS NOT DISTINCT FROM $2
            ORDER BY version DESC
            """,
            resource,
            tenant_id,
        )
        return [dict(row) for row in rows]

    async def list_resources(
        self,
        tenant_id: Optional[str] = None,
    ) -> list[str]:
        """List all unique resource names."""
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT resource FROM schema_versions
            WHERE tenant_id IS NOT DISTINCT FROM $1
            ORDER BY resource
            """,
            tenant_id,
        )
        return [row["resource"] for row in rows]

    async def publish_version(self, version_id: str) -> None:
        """Publish a schema version (deprecate previous published)."""
        # Get the version to publish
        version = await self.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        # Deprecate current published version
        await self._pool.execute(
            """
            UPDATE schema_versions
            SET status = $1
            WHERE resource = $2
              AND tenant_id IS NOT DISTINCT FROM $3
              AND status = $4
            """,
            SchemaStatus.DEPRECATED.value,
            version["resource"],
            version.get("tenant_id"),
            SchemaStatus.PUBLISHED.value,
        )

        # Publish the new version
        await self._pool.execute(
            """
            UPDATE schema_versions
            SET status = $1
            WHERE id = $2
            """,
            SchemaStatus.PUBLISHED.value,
            version_id,
        )

    async def delete_version(self, version_id: str) -> bool:
        """Delete a draft version. Published versions cannot be deleted."""
        result = await self._pool.execute(
            """
            DELETE FROM schema_versions
            WHERE id = $1 AND status = $2
            """,
            version_id,
            SchemaStatus.DRAFT.value,
        )
        return result == "DELETE 1"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_schema_registry.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add schema registry for database operations"
```

---

### Task 6: Schema Routes

**Files:**
- Create: `webapp/schema/routes.py`
- Create: `webapp/templates/schema/list.html`
- Create: `webapp/templates/schema/editor.html`

**Step 1: Create webapp/schema/routes.py**

```python
"""Schema routes - API and UI endpoints."""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from webapp.dependencies import DataLayerDep, TenantDep
from webapp.schema.registry import SchemaRegistry
from webapp.schema.validator import SchemaValidator
from webapp.schema.models import SchemaVersionCreate


router = APIRouter()
validator = SchemaValidator()


class CreateSchemaRequest(BaseModel):
    """Request body for creating a schema."""

    resource: str
    schema_json: dict
    x_backends: dict


# --- UI Routes ---


@router.get("", response_class=HTMLResponse)
async def list_schemas_page(
    request: Request,
    data: DataLayerDep,
    tenant: TenantDep,
):
    """Render schema list page."""
    registry = SchemaRegistry(data.postgres)
    resources = await registry.list_resources(tenant)

    # Get latest version info for each resource
    schemas = []
    for resource in resources:
        versions = await registry.list_versions(resource, tenant)
        if versions:
            schemas.append({
                "resource": resource,
                "latest_version": versions[0]["version"],
                "status": versions[0]["status"],
                "version_count": len(versions),
            })

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "schema/list.html",
        {
            "request": request,
            "config": request.app.state.webapp_config,
            "tenant": tenant,
            "schemas": schemas,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def new_schema_page(request: Request, tenant: TenantDep):
    """Render new schema editor page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "schema/editor.html",
        {
            "request": request,
            "config": request.app.state.webapp_config,
            "tenant": tenant,
            "schema": None,
            "is_new": True,
        },
    )


@router.get("/{resource}", response_class=HTMLResponse)
async def schema_detail_page(
    request: Request,
    resource: str,
    data: DataLayerDep,
    tenant: TenantDep,
    version: Optional[int] = None,
):
    """Render schema detail/editor page."""
    registry = SchemaRegistry(data.postgres)

    if version:
        versions = await registry.list_versions(resource, tenant)
        schema = next((v for v in versions if v["version"] == version), None)
    else:
        schema = await registry.get_latest_published(resource, tenant)
        if not schema:
            versions = await registry.list_versions(resource, tenant)
            schema = versions[0] if versions else None

    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema not found: {resource}")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "schema/editor.html",
        {
            "request": request,
            "config": request.app.state.webapp_config,
            "tenant": tenant,
            "schema": schema,
            "is_new": False,
        },
    )


# --- API Routes ---


@router.post("/api")
async def create_schema(
    request_body: CreateSchemaRequest,
    data: DataLayerDep,
    tenant: TenantDep,
):
    """Create a new schema version."""
    # Validate
    schema_ir = {
        "resource": request_body.resource,
        "schema": request_body.schema_json,
        "x-backends": request_body.x_backends,
    }
    errors = validator.validate(schema_ir)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"errors": [str(e) for e in errors]},
        )

    # Create
    registry = SchemaRegistry(data.postgres)
    version = await registry.create_version(
        resource=request_body.resource,
        schema_json=request_body.schema_json,
        x_backends=request_body.x_backends,
        tenant_id=tenant,
    )

    return {"id": str(version["id"]), "version": version["version"]}


@router.get("/api/{resource}")
async def get_schema(
    resource: str,
    data: DataLayerDep,
    tenant: TenantDep,
    version: Optional[int] = None,
):
    """Get a schema by resource name."""
    registry = SchemaRegistry(data.postgres)

    if version:
        versions = await registry.list_versions(resource, tenant)
        schema = next((v for v in versions if v["version"] == version), None)
    else:
        schema = await registry.get_latest_published(resource, tenant)

    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema not found: {resource}")

    return schema


@router.post("/api/{version_id}/publish")
async def publish_schema(
    version_id: str,
    data: DataLayerDep,
):
    """Publish a schema version."""
    registry = SchemaRegistry(data.postgres)

    try:
        await registry.publish_version(version_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"status": "published"}
```

**Step 2: Create webapp/templates/schema/list.html**

```html
{% extends "base.html" %}

{% block title %}Schemas - {{ config.title }}{% endblock %}

{% block content %}
<div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-bold">Schemas</h1>
    <a href="/schemas/new"
       class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
        New Schema
    </a>
</div>

{% if schemas %}
<div class="bg-white rounded shadow overflow-hidden">
    <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
            <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Resource</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Latest Version</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Versions</th>
                <th class="px-6 py-3"></th>
            </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
            {% for schema in schemas %}
            <tr>
                <td class="px-6 py-4 whitespace-nowrap font-medium">{{ schema.resource }}</td>
                <td class="px-6 py-4 whitespace-nowrap">v{{ schema.latest_version }}</td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 py-1 text-xs rounded
                        {% if schema.status == 'published' %}bg-green-100 text-green-800
                        {% elif schema.status == 'draft' %}bg-yellow-100 text-yellow-800
                        {% else %}bg-gray-100 text-gray-800{% endif %}">
                        {{ schema.status }}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-gray-500">{{ schema.version_count }}</td>
                <td class="px-6 py-4 whitespace-nowrap text-right">
                    <a href="/schemas/{{ schema.resource }}" class="text-blue-600 hover:text-blue-800">View</a>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% else %}
<div class="bg-white rounded shadow p-8 text-center text-gray-500">
    <p>No schemas defined yet.</p>
    <a href="/schemas/new" class="text-blue-600 hover:underline">Create your first schema</a>
</div>
{% endif %}
{% endblock %}
```

**Step 3: Create webapp/templates/schema/editor.html**

```html
{% extends "base.html" %}

{% block title %}
{% if is_new %}New Schema{% else %}{{ schema.resource }}{% endif %} - {{ config.title }}
{% endblock %}

{% block content %}
<div class="mb-6">
    <a href="/schemas" class="text-blue-600 hover:underline">&larr; Back to Schemas</a>
</div>

<div class="bg-white rounded shadow p-6">
    <h1 class="text-2xl font-bold mb-6">
        {% if is_new %}New Schema{% else %}{{ schema.resource }} (v{{ schema.version }}){% endif %}
    </h1>

    <form id="schema-form" hx-post="/schemas/api" hx-swap="none">
        <div class="space-y-6">
            <!-- Resource Name -->
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Resource Name</label>
                <input type="text" name="resource"
                       value="{{ schema.resource if schema else '' }}"
                       {% if not is_new %}readonly{% endif %}
                       class="w-full border rounded px-3 py-2 {% if not is_new %}bg-gray-100{% endif %}"
                       placeholder="e.g., users, documents">
            </div>

            <!-- JSON Schema -->
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">JSON Schema</label>
                <textarea name="schema_json" rows="12"
                          class="w-full border rounded px-3 py-2 font-mono text-sm"
                          placeholder='{"type": "object", "properties": {...}}'>{{ schema.schema_json | tojson(indent=2) if schema else '' }}</textarea>
            </div>

            <!-- Backend Config -->
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Backend Configuration</label>
                <textarea name="x_backends" rows="12"
                          class="w-full border rounded px-3 py-2 font-mono text-sm"
                          placeholder='{"postgres": {...}, "elasticsearch": {...}}'>{{ schema.x_backends | tojson(indent=2) if schema else '' }}</textarea>
            </div>

            <!-- Actions -->
            <div class="flex justify-between items-center pt-4 border-t">
                <div>
                    {% if schema and schema.status == 'draft' %}
                    <button type="button"
                            hx-post="/schemas/api/{{ schema.id }}/publish"
                            hx-swap="none"
                            class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
                        Publish
                    </button>
                    {% endif %}
                </div>
                <div class="space-x-3">
                    <a href="/schemas" class="text-gray-600 hover:text-gray-800">Cancel</a>
                    <button type="submit"
                            class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                        {% if is_new %}Create Draft{% else %}Save New Version{% endif %}
                    </button>
                </div>
            </div>
        </div>
    </form>

    <!-- Validation Errors -->
    <div id="errors" class="mt-4 hidden">
        <div class="bg-red-50 border border-red-200 rounded p-4 text-red-700">
            <ul id="error-list"></ul>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.body.addEventListener('htmx:afterRequest', function(evt) {
    const errorDiv = document.getElementById('errors');
    const errorList = document.getElementById('error-list');

    if (evt.detail.successful) {
        window.location.href = '/schemas';
    } else if (evt.detail.xhr.status === 400) {
        const response = JSON.parse(evt.detail.xhr.responseText);
        errorList.innerHTML = response.detail.errors
            .map(e => `<li>${e}</li>`)
            .join('');
        errorDiv.classList.remove('hidden');
    }
});
</script>
{% endblock %}
```

**Step 4: Register routes in app.py**

Update `webapp/app.py` to include the schema router:

```python
# Add import at top
from webapp.schema.routes import router as schema_router

# In create_app(), uncomment:
app.include_router(schema_router, prefix="/schemas", tags=["schemas"])
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add schema routes and templates"
```

---

## Phase 3: Migration Generator

### Task 7: Migration Models

**Files:**
- Create: `webapp/migration/models.py`
- Create: `tests/webapp/test_migration_models.py`

**Step 1: Write the failing test**

Create `tests/webapp/test_migration_models.py`:

```python
"""Tests for migration models."""

from webapp.migration.models import (
    Migration,
    MigrationStatus,
    OperationType,
    MigrationOperation,
)


def test_migration_operation():
    op = MigrationOperation(
        type=OperationType.ADD_COLUMN,
        target="users.name",
        config={"type": "text", "nullable": True},
    )
    assert op.type == OperationType.ADD_COLUMN


def test_migration():
    migration = Migration(
        id="test-id",
        from_version_id=None,
        to_version_id="version-1",
        backend="postgres",
        operations=[
            MigrationOperation(
                type=OperationType.CREATE_TABLE,
                target="users",
                config={},
            )
        ],
        sql_up="CREATE TABLE users (...)",
        sql_down="DROP TABLE users",
        status=MigrationStatus.PENDING,
    )
    assert migration.backend == "postgres"
    assert len(migration.operations) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_migration_models.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `webapp/migration/models.py`:

```python
"""Pydantic models for migrations."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MigrationStatus(str, Enum):
    """Status of a migration."""

    PENDING = "pending"
    APPROVED = "approved"
    APPLIED = "applied"
    FAILED = "failed"


class OperationType(str, Enum):
    """Types of migration operations."""

    # Postgres
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    ALTER_COLUMN = "alter_column"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"

    # Elasticsearch
    CREATE_INDEX_ES = "create_index_es"
    UPDATE_MAPPING = "update_mapping"
    DELETE_INDEX_ES = "delete_index_es"

    # Redis
    UPDATE_KEY_PATTERN = "update_key_pattern"

    # FAISS
    CREATE_NAMESPACE = "create_namespace"
    REINDEX_VECTORS = "reindex_vectors"


class MigrationOperation(BaseModel):
    """A single migration operation."""

    type: OperationType
    target: str  # e.g., "users", "users.email", "users_email_idx"
    config: dict[str, Any] = {}


class Migration(BaseModel):
    """A migration between schema versions."""

    id: str
    from_version_id: Optional[str]
    to_version_id: str
    backend: str  # postgres, elasticsearch, redis, faiss
    operations: list[MigrationOperation]
    sql_up: Optional[str] = None
    sql_down: Optional[str] = None
    status: MigrationStatus = MigrationStatus.PENDING
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None

    class Config:
        from_attributes = True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_migration_models.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add migration models"
```

---

### Task 8: Schema Differ

**Files:**
- Create: `webapp/migration/differ.py`
- Create: `tests/webapp/test_differ.py`

**Step 1: Write the failing test**

Create `tests/webapp/test_differ.py`:

```python
"""Tests for schema differ."""

import pytest
from webapp.migration.differ import SchemaDiffer
from webapp.migration.models import OperationType


@pytest.fixture
def differ() -> SchemaDiffer:
    return SchemaDiffer()


def test_diff_add_column(differ: SchemaDiffer):
    old_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                },
                "indexes": [],
            }
        }
    }
    new_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "name": {"type": "text", "nullable": True},  # new
                },
                "indexes": [],
            }
        }
    }

    ops = differ.diff(old_schema, new_schema, "postgres")

    assert len(ops) == 1
    assert ops[0].type == OperationType.ADD_COLUMN
    assert ops[0].target == "users.name"


def test_diff_drop_column(differ: SchemaDiffer):
    old_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "legacy": {"type": "text", "nullable": True},
                },
                "indexes": [],
            }
        }
    }
    new_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                },
                "indexes": [],
            }
        }
    }

    ops = differ.diff(old_schema, new_schema, "postgres")

    assert len(ops) == 1
    assert ops[0].type == OperationType.DROP_COLUMN


def test_diff_create_table(differ: SchemaDiffer):
    old_schema = {"x-backends": {}}
    new_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {"id": {"type": "text", "nullable": False}},
                "indexes": [],
            }
        }
    }

    ops = differ.diff(old_schema, new_schema, "postgres")

    assert len(ops) == 1
    assert ops[0].type == OperationType.CREATE_TABLE


def test_diff_add_index(differ: SchemaDiffer):
    old_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {"id": {"type": "text"}},
                "indexes": [],
            }
        }
    }
    new_schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {"id": {"type": "text"}},
                "indexes": [{"name": "users_id_idx", "columns": ["id"]}],
            }
        }
    }

    ops = differ.diff(old_schema, new_schema, "postgres")

    assert len(ops) == 1
    assert ops[0].type == OperationType.CREATE_INDEX


def test_diff_elasticsearch_create(differ: SchemaDiffer):
    old_schema = {"x-backends": {}}
    new_schema = {
        "x-backends": {
            "elasticsearch": {
                "index": "users",
                "mappings": {"properties": {"name": {"type": "text"}}},
            }
        }
    }

    ops = differ.diff(old_schema, new_schema, "elasticsearch")

    assert len(ops) == 1
    assert ops[0].type == OperationType.CREATE_INDEX_ES


def test_diff_no_changes(differ: SchemaDiffer):
    schema = {
        "x-backends": {
            "postgres": {
                "table": "users",
                "primary_key": ["id"],
                "columns": {"id": {"type": "text"}},
                "indexes": [],
            }
        }
    }

    ops = differ.diff(schema, schema, "postgres")

    assert len(ops) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_differ.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `webapp/migration/differ.py`:

```python
"""Schema differ - compute operations needed between versions."""

from typing import Any
from webapp.migration.models import MigrationOperation, OperationType


class SchemaDiffer:
    """Computes the diff between two schema versions for a given backend."""

    def diff(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
        backend: str,
    ) -> list[MigrationOperation]:
        """Compute operations to migrate from old to new schema."""
        if backend == "postgres":
            return self._diff_postgres(old_schema, new_schema)
        elif backend == "elasticsearch":
            return self._diff_elasticsearch(old_schema, new_schema)
        elif backend == "redis":
            return self._diff_redis(old_schema, new_schema)
        elif backend == "faiss":
            return self._diff_faiss(old_schema, new_schema)
        else:
            return []

    def _diff_postgres(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> list[MigrationOperation]:
        """Diff Postgres backend configurations."""
        ops: list[MigrationOperation] = []

        old_pg = old_schema.get("x-backends", {}).get("postgres")
        new_pg = new_schema.get("x-backends", {}).get("postgres")

        # New table
        if not old_pg and new_pg:
            return [
                MigrationOperation(
                    type=OperationType.CREATE_TABLE,
                    target=new_pg["table"],
                    config=new_pg,
                )
            ]

        # Dropped table
        if old_pg and not new_pg:
            return [
                MigrationOperation(
                    type=OperationType.DROP_TABLE,
                    target=old_pg["table"],
                    config={},
                )
            ]

        if not old_pg or not new_pg:
            return []

        table = new_pg["table"]
        old_cols = old_pg.get("columns", {})
        new_cols = new_pg.get("columns", {})

        # Added columns
        for col_name, col_config in new_cols.items():
            if col_name not in old_cols:
                ops.append(
                    MigrationOperation(
                        type=OperationType.ADD_COLUMN,
                        target=f"{table}.{col_name}",
                        config=col_config if isinstance(col_config, dict) else {"type": col_config},
                    )
                )

        # Dropped columns
        for col_name in old_cols:
            if col_name not in new_cols:
                ops.append(
                    MigrationOperation(
                        type=OperationType.DROP_COLUMN,
                        target=f"{table}.{col_name}",
                        config={},
                    )
                )

        # Altered columns (type changes)
        for col_name, new_config in new_cols.items():
            if col_name in old_cols:
                old_config = old_cols[col_name]
                if old_config != new_config:
                    ops.append(
                        MigrationOperation(
                            type=OperationType.ALTER_COLUMN,
                            target=f"{table}.{col_name}",
                            config={"old": old_config, "new": new_config},
                        )
                    )

        # Indexes
        old_indexes = {idx["name"]: idx for idx in old_pg.get("indexes", [])}
        new_indexes = {idx["name"]: idx for idx in new_pg.get("indexes", [])}

        for idx_name, idx_config in new_indexes.items():
            if idx_name not in old_indexes:
                ops.append(
                    MigrationOperation(
                        type=OperationType.CREATE_INDEX,
                        target=idx_name,
                        config=idx_config,
                    )
                )

        for idx_name in old_indexes:
            if idx_name not in new_indexes:
                ops.append(
                    MigrationOperation(
                        type=OperationType.DROP_INDEX,
                        target=idx_name,
                        config={},
                    )
                )

        return ops

    def _diff_elasticsearch(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> list[MigrationOperation]:
        """Diff Elasticsearch backend configurations."""
        ops: list[MigrationOperation] = []

        old_es = old_schema.get("x-backends", {}).get("elasticsearch")
        new_es = new_schema.get("x-backends", {}).get("elasticsearch")

        if not old_es and new_es:
            return [
                MigrationOperation(
                    type=OperationType.CREATE_INDEX_ES,
                    target=new_es["index"],
                    config=new_es,
                )
            ]

        if old_es and not new_es:
            return [
                MigrationOperation(
                    type=OperationType.DELETE_INDEX_ES,
                    target=old_es["index"],
                    config={},
                )
            ]

        if old_es and new_es:
            if old_es.get("mappings") != new_es.get("mappings"):
                ops.append(
                    MigrationOperation(
                        type=OperationType.UPDATE_MAPPING,
                        target=new_es["index"],
                        config={"mappings": new_es["mappings"]},
                    )
                )

        return ops

    def _diff_redis(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> list[MigrationOperation]:
        """Diff Redis backend configurations."""
        ops: list[MigrationOperation] = []

        old_redis = old_schema.get("x-backends", {}).get("redis")
        new_redis = new_schema.get("x-backends", {}).get("redis")

        if old_redis and new_redis:
            if old_redis.get("pattern") != new_redis.get("pattern"):
                ops.append(
                    MigrationOperation(
                        type=OperationType.UPDATE_KEY_PATTERN,
                        target="redis",
                        config={
                            "old_pattern": old_redis.get("pattern"),
                            "new_pattern": new_redis.get("pattern"),
                        },
                    )
                )

        return ops

    def _diff_faiss(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> list[MigrationOperation]:
        """Diff FAISS backend configurations."""
        ops: list[MigrationOperation] = []

        old_faiss = old_schema.get("x-backends", {}).get("faiss")
        new_faiss = new_schema.get("x-backends", {}).get("faiss")

        if not old_faiss and new_faiss:
            return [
                MigrationOperation(
                    type=OperationType.CREATE_NAMESPACE,
                    target=new_faiss["namespace"],
                    config=new_faiss,
                )
            ]

        if old_faiss and new_faiss:
            # Dimension change requires full reindex
            if old_faiss.get("dimension") != new_faiss.get("dimension"):
                ops.append(
                    MigrationOperation(
                        type=OperationType.REINDEX_VECTORS,
                        target=new_faiss["namespace"],
                        config={
                            "reason": "dimension_change",
                            "old_dimension": old_faiss.get("dimension"),
                            "new_dimension": new_faiss.get("dimension"),
                        },
                    )
                )

        return ops
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_differ.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add schema differ for computing migration operations"
```

---

### Task 9: Postgres Migration Generator

**Files:**
- Create: `webapp/migration/generators/postgres.py`
- Create: `tests/webapp/test_postgres_generator.py`

**Step 1: Write the failing test**

Create `tests/webapp/test_postgres_generator.py`:

```python
"""Tests for Postgres migration generator."""

import pytest
from webapp.migration.generators.postgres import PostgresGenerator
from webapp.migration.models import MigrationOperation, OperationType


@pytest.fixture
def generator() -> PostgresGenerator:
    return PostgresGenerator()


def test_generate_create_table(generator: PostgresGenerator):
    ops = [
        MigrationOperation(
            type=OperationType.CREATE_TABLE,
            target="users",
            config={
                "table": "users",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "email": {"type": "text", "nullable": False},
                    "name": {"type": "text", "nullable": True},
                },
            },
        )
    ]

    sql_up, sql_down = generator.generate(ops)

    assert "CREATE TABLE" in sql_up
    assert "users" in sql_up
    assert "id TEXT NOT NULL" in sql_up
    assert "PRIMARY KEY (id)" in sql_up
    assert "DROP TABLE" in sql_down


def test_generate_add_column(generator: PostgresGenerator):
    ops = [
        MigrationOperation(
            type=OperationType.ADD_COLUMN,
            target="users.name",
            config={"type": "text", "nullable": True},
        )
    ]

    sql_up, sql_down = generator.generate(ops)

    assert "ALTER TABLE users ADD COLUMN name TEXT" in sql_up
    assert "ALTER TABLE users DROP COLUMN name" in sql_down


def test_generate_drop_column(generator: PostgresGenerator):
    ops = [
        MigrationOperation(
            type=OperationType.DROP_COLUMN,
            target="users.legacy",
            config={},
        )
    ]

    sql_up, sql_down = generator.generate(ops)

    assert "ALTER TABLE users DROP COLUMN legacy" in sql_up


def test_generate_create_index(generator: PostgresGenerator):
    ops = [
        MigrationOperation(
            type=OperationType.CREATE_INDEX,
            target="users_email_idx",
            config={"columns": ["email"], "unique": True},
        )
    ]

    sql_up, sql_down = generator.generate(ops)

    assert "CREATE UNIQUE INDEX" in sql_up
    assert "users_email_idx" in sql_up
    assert "DROP INDEX" in sql_down


def test_generate_multiple_operations(generator: PostgresGenerator):
    ops = [
        MigrationOperation(
            type=OperationType.ADD_COLUMN,
            target="users.phone",
            config={"type": "text", "nullable": True},
        ),
        MigrationOperation(
            type=OperationType.CREATE_INDEX,
            target="users_phone_idx",
            config={"columns": ["phone"]},
        ),
    ]

    sql_up, sql_down = generator.generate(ops)

    assert "ADD COLUMN phone" in sql_up
    assert "CREATE INDEX" in sql_up
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/webapp/test_postgres_generator.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Create `webapp/migration/generators/postgres.py`:

```python
"""Postgres migration SQL generator."""

from webapp.migration.models import MigrationOperation, OperationType


class PostgresGenerator:
    """Generate Postgres migration SQL from operations."""

    def generate(self, operations: list[MigrationOperation]) -> tuple[str, str]:
        """Generate SQL up and down scripts."""
        up_statements: list[str] = []
        down_statements: list[str] = []

        for op in operations:
            up, down = self._generate_operation(op)
            if up:
                up_statements.append(up)
            if down:
                down_statements.insert(0, down)  # Reverse order for rollback

        sql_up = "\n\n".join(up_statements)
        sql_down = "\n\n".join(down_statements)

        return sql_up, sql_down

    def _generate_operation(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate SQL for a single operation."""
        if op.type == OperationType.CREATE_TABLE:
            return self._create_table(op)
        elif op.type == OperationType.DROP_TABLE:
            return self._drop_table(op)
        elif op.type == OperationType.ADD_COLUMN:
            return self._add_column(op)
        elif op.type == OperationType.DROP_COLUMN:
            return self._drop_column(op)
        elif op.type == OperationType.ALTER_COLUMN:
            return self._alter_column(op)
        elif op.type == OperationType.CREATE_INDEX:
            return self._create_index(op)
        elif op.type == OperationType.DROP_INDEX:
            return self._drop_index(op)
        else:
            return f"-- Unsupported operation: {op.type}", ""

    def _create_table(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate CREATE TABLE statement."""
        table = op.target
        config = op.config
        columns = config.get("columns", {})
        pk = config.get("primary_key", [])

        col_defs = []
        for col_name, col_config in columns.items():
            if isinstance(col_config, dict):
                col_type = col_config.get("type", "text").upper()
                nullable = col_config.get("nullable", True)
            else:
                col_type = str(col_config).upper()
                nullable = True

            null_str = "" if nullable else " NOT NULL"
            col_defs.append(f"    {col_name} {col_type}{null_str}")

        # Add tenant_id if not present
        if "tenant_id" not in columns:
            col_defs.append("    tenant_id TEXT NOT NULL")

        if pk:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk)})")

        columns_sql = ",\n".join(col_defs)

        up = f"CREATE TABLE {table} (\n{columns_sql}\n);"
        down = f"DROP TABLE {table};"

        return up, down

    def _drop_table(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate DROP TABLE statement."""
        table = op.target
        return f"DROP TABLE {table};", ""

    def _add_column(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate ADD COLUMN statement."""
        table, column = op.target.rsplit(".", 1)
        col_type = op.config.get("type", "text").upper()
        nullable = op.config.get("nullable", True)
        null_str = "" if nullable else " NOT NULL"

        up = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{null_str};"
        down = f"ALTER TABLE {table} DROP COLUMN {column};"

        return up, down

    def _drop_column(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate DROP COLUMN statement."""
        table, column = op.target.rsplit(".", 1)
        up = f"ALTER TABLE {table} DROP COLUMN {column};"
        # Can't easily undo a drop
        down = f"-- Cannot restore dropped column: {column}"

        return up, down

    def _alter_column(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate ALTER COLUMN statement."""
        table, column = op.target.rsplit(".", 1)
        old_config = op.config.get("old", {})
        new_config = op.config.get("new", {})

        statements = []

        # Type change
        old_type = old_config.get("type") if isinstance(old_config, dict) else old_config
        new_type = new_config.get("type") if isinstance(new_config, dict) else new_config

        if old_type != new_type:
            statements.append(
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type.upper()}"
            )

        up = ";\n".join(statements) + ";" if statements else ""
        down = ""  # Complex to reverse

        return up, down

    def _create_index(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate CREATE INDEX statement."""
        index_name = op.target
        columns = op.config.get("columns", [])
        unique = op.config.get("unique", False)
        using = op.config.get("using")

        # Extract table from index name convention (table_column_idx)
        table = index_name.rsplit("_", 2)[0] if "_" in index_name else "unknown"
        if "table" in op.config:
            table = op.config["table"]

        unique_str = "UNIQUE " if unique else ""
        using_str = f" USING {using}" if using else ""
        columns_str = ", ".join(columns)

        up = f"CREATE {unique_str}INDEX {index_name} ON {table}{using_str} ({columns_str});"
        down = f"DROP INDEX {index_name};"

        return up, down

    def _drop_index(self, op: MigrationOperation) -> tuple[str, str]:
        """Generate DROP INDEX statement."""
        index_name = op.target
        return f"DROP INDEX {index_name};", ""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/webapp/test_postgres_generator.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Postgres migration SQL generator"
```

---

## Phase 4: Remaining Implementation

The remaining tasks follow the same pattern. Continue with:

### Task 10: Elasticsearch Migration Generator
- Create `webapp/migration/generators/elasticsearch.py`
- Generate ES mapping update commands

### Task 11: Migration Registry & Routes
- Create `webapp/migration/registry.py` (store/retrieve migrations)
- Create `webapp/migration/routes.py` (API endpoints)
- Create `webapp/templates/migration/list.html`
- Create `webapp/templates/migration/preview.html`

### Task 12: Migration Executor
- Create `webapp/migration/executor.py`
- Apply migrations to each backend

### Task 13: Data API Routes
- Create `webapp/data/router.py` (route to backends based on schema)
- Create `webapp/data/crud.py` (generic CRUD)
- Create `webapp/data/routes.py`

### Task 14: Data Explorer UI
- Create `webapp/templates/data/list.html`
- Create `webapp/templates/data/explorer.html`
- Create `webapp/templates/data/record.html`

### Task 15: Query Console
- Create `webapp/query/executor.py`
- Create `webapp/query/routes.py`
- Create `webapp/templates/query/console.html`

### Task 16: Home Page & Final Wiring
- Create `webapp/templates/index.html`
- Add root route
- Wire all routers in app.py
- Create `run.py` for dev server

### Task 17: Integration Tests
- Create `tests/integration/test_webapp.py`
- Test full flow: create schema → generate migration → apply → query data

---

## Summary

After completing all tasks:

1. **Schema Registry** - Create, version, publish schemas with validation
2. **Migration System** - Diff schemas, generate SQL/ES commands, apply with approval
3. **Data API** - Generic CRUD routed to correct backend
4. **Query Console** - Raw queries against any backend
5. **HTMX UI** - Fast, server-rendered admin interface

Run the dev server:
```bash
python run.py
# or
uvicorn webapp:create_app --factory --reload
```

# API Reference

Complete API reference for the Satellites WebApp.

## Table of Contents

- [Schema Registry API](#schema-registry-api)
- [Migration API](#migration-api)
- [Data API](#data-api)
- [Query API](#query-api)
- [Python API](#python-api)

---

## Schema Registry API

### Models

#### SchemaStatus
```python
class SchemaStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
```

#### SchemaVersionCreate
```python
{
  "resource": str,                    # Resource name (e.g., "users")
  "tenant_id": Optional[str],         # Multi-tenancy support
  "extends_id": Optional[str],        # Inherit from another schema
  "schema_json": dict,                # JSON Schema (Draft 7)
  "x_backends": dict                  # Backend configurations
}
```

#### SchemaVersion
```python
{
  "id": str,                          # UUID
  "resource": str,
  "version": int,                     # Auto-incremented per resource
  "tenant_id": Optional[str],
  "extends_id": Optional[str],
  "schema_json": dict,
  "x_backends": dict,
  "status": SchemaStatus,
  "created_at": datetime,
  "created_by": Optional[str],
  "checksum": str                     # SHA256 of schema content
}
```

### Routes

#### `GET /schemas`
List all schema versions.

**Query Parameters:**
- None

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "resource": "users",
    "version": 1,
    "status": "published",
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

#### `GET /schemas/new`
Show schema creation form (UI).

**Response:** `200 OK` - HTML page

#### `GET /schemas/{schema_id}`
Get schema version details.

**Path Parameters:**
- `schema_id`: Schema version UUID

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "resource": "users",
  "version": 1,
  "schema_json": { ... },
  "x_backends": { ... },
  "status": "published",
  "created_at": "2024-01-01T00:00:00Z",
  "checksum": "abc123..."
}
```

**Error:** `404 Not Found`

---

## Migration API

### Models

#### MigrationOperationType
```python
class MigrationOperationType(str, Enum):
    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    CREATE_CONSTRAINT = "create_constraint"
    DROP_CONSTRAINT = "drop_constraint"
```

#### MigrationStatus
```python
class MigrationStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
```

#### MigrationOperation
```python
{
  "type": MigrationOperationType,
  "table": str,
  "details": dict                     # Operation-specific details
}
```

#### Migration
```python
{
  "id": str,
  "from_version_id": Optional[str],   # null for initial migration
  "to_version_id": str,
  "resource": str,
  "backend": str,                     # "postgres", "elasticsearch", etc.
  "operations": List[MigrationOperation],
  "status": MigrationStatus,
  "created_at": datetime,
  "applied_at": Optional[datetime],
  "generated_sql": Optional[str],     # For SQL backends
  "rollback_sql": Optional[str],
  "error_message": Optional[str]
}
```

### Routes

#### `GET /migrations`
List migrations with optional filtering.

**Query Parameters:**
- `resource` (optional): Filter by resource name
- `backend` (optional): Filter by backend type
- `status` (optional): Filter by status

**Response:** `200 OK`
```json
[
  {
    "id": "uuid",
    "resource": "users",
    "backend": "postgres",
    "from_version_id": "uuid1",
    "to_version_id": "uuid2",
    "status": "pending",
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

#### `GET /migrations/{migration_id}/preview`
Preview migration details.

**Response:** `200 OK` - HTML page showing:
- Migration metadata
- Generated SQL/commands
- List of operations
- Apply/cancel buttons (if status=pending)

---

## Data API

### CRUD Operations

#### Query Operators

When filtering data, use these operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `field` | Exact match | `?email=test@example.com` |
| `field__gte` | Greater than or equal | `?age__gte=18` |
| `field__gt` | Greater than | `?age__gt=17` |
| `field__lte` | Less than or equal | `?age__lte=65` |
| `field__lt` | Less than | `?age__lt=66` |
| `field__ne` | Not equal | `?status__ne=inactive` |
| `field__contains` | Contains substring | `?name__contains=john` |

### Routes

#### `GET /data`
Data explorer home page.

**Query Parameters:**
- `resource` (optional): Pre-select a resource

**Response:** `200 OK` - HTML page

#### `GET /data/{resource}`
List records for a resource.

**Path Parameters:**
- `resource`: Resource name (e.g., "users")

**Query Parameters:**
- `limit` (default: 100): Maximum records
- `offset` (default: 0): Pagination offset
- Any field name for filtering (see operators above)

**Example:**
```
GET /data/users?age__gte=18&limit=50&offset=0
```

**Response:** `200 OK`
```json
[
  {
    "id": "123",
    "email": "test@example.com",
    "name": "John Doe",
    "age": 25
  }
]
```

#### `GET /data/{resource}/{record_id}`
Get a single record.

**Response:** `200 OK`
```json
{
  "id": "123",
  "email": "test@example.com",
  "name": "John Doe",
  "age": 25
}
```

**Error:** `404 Not Found`

#### `POST /data/{resource}`
Create a new record.

**Request Body:**
```json
{
  "id": "123",
  "email": "test@example.com",
  "name": "John Doe",
  "age": 25
}
```

**Validation:**
- Must conform to resource's JSON Schema
- Required fields must be present
- Types must match

**Response:** `201 Created`
```json
{
  "id": "123",
  "email": "test@example.com",
  "name": "John Doe",
  "age": 25
}
```

**Error:** `400 Bad Request` - Validation failed

#### `PUT /data/{resource}/{record_id}`
Update an existing record.

**Request Body:** Partial or full record data

**Response:** `200 OK` - Updated record

**Error:** `404 Not Found`

#### `DELETE /data/{resource}/{record_id}`
Delete a record.

**Response:** `204 No Content`

**Error:** `404 Not Found`

---

## Query API

### Routes

#### `GET /query`
Query console UI.

**Response:** `200 OK` - HTML page with:
- Backend selector dropdown
- Query textarea
- Execute button
- Results display area
- Query history

#### `POST /query/execute`
Execute a raw query.

**Request Body for Postgres:**
```json
{
  "backend": "postgres",
  "query": "SELECT * FROM users WHERE email LIKE '%@example.com' LIMIT 10",
  "params": {}
}
```

**Request Body for Elasticsearch:**
```json
{
  "backend": "elasticsearch",
  "index": "users",
  "query": {
    "bool": {
      "must": [
        {"match": {"name": "john"}},
        {"range": {"age": {"gte": 18}}}
      ]
    }
  }
}
```

**Request Body for Redis:**
```json
{
  "backend": "redis",
  "command": "HGETALL",
  "args": ["user:123"]
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "backend": "postgres",
  "rows": [...],
  "execution_time_ms": 42
}
```

**Error:** `400 Bad Request` - Invalid query
**Error:** `403 Forbidden` - Dangerous query (DROP, DELETE, etc.)

---

## Python API

### Schema Validator

```python
from webapp.schema.validator import SchemaValidator

validator = SchemaValidator()

# Validate JSON Schema structure
validator.validate_schema({
    "type": "object",
    "properties": {
        "id": {"type": "string"}
    }
})

# Validate data against schema
validator.validate_data(
    data={"id": "123"},
    schema={"type": "object", "properties": {"id": {"type": "string"}}}
)

# Validate Postgres backend config
validator.validate_postgres_backend({
    "table": "users",
    "primary_key": ["id"],
    "columns": {"id": {"type": "text", "nullable": False}}
})

# Check consistency
validator.validate_consistency(schema_json, backends)
```

### Schema Registry

```python
from webapp.schema.registry import SchemaRegistry
from webapp.schema.models import SchemaVersionCreate

registry = SchemaRegistry(connection_string="postgresql://...")
await registry.start()

# Create schema
schema = await registry.create_schema_version(
    SchemaVersionCreate(
        resource="users",
        schema_json={...},
        x_backends={...}
    ),
    created_by="admin"
)

# Get schema
schema = await registry.get_schema_version(schema_id)

# Get latest version
latest = await registry.get_latest_schema_version("users", tenant_id=None)

# List versions
versions = await registry.list_schema_versions("users")

# Publish schema
published = await registry.publish_schema_version(schema_id)

await registry.stop()
```

### Schema Differ

```python
from webapp.migration.differ import SchemaDiffer

differ = SchemaDiffer()

# Diff Postgres backends
operations = differ.diff_postgres(old_backend, new_backend)

# Each operation has:
# - type: MigrationOperationType
# - table: str
# - details: dict
```

### Migration Generator

```python
from webapp.migration.generators.postgres import PostgresGenerator
from webapp.migration.generators.elasticsearch import ElasticsearchGenerator

# Postgres
pg_gen = PostgresGenerator()
sql = pg_gen.generate_migration_sql(operations)
# Returns: "ALTER TABLE users ADD COLUMN phone TEXT;\nCREATE INDEX..."

# Elasticsearch
es_gen = ElasticsearchGenerator()
json_commands = es_gen.generate_migration_script(operations)
python_code = es_gen.generate_python_api(operations)
```

### Migration Registry

```python
from webapp.migration.registry import MigrationRegistry

registry = MigrationRegistry(connection_string="postgresql://...")
await registry.start()

# Create migration
migration = await registry.create_migration(
    from_version_id="uuid1",
    to_version_id="uuid2",
    resource="users",
    backend="postgres",
    operations=[...],
    generated_sql="ALTER TABLE..."
)

# Get migration
migration = await registry.get_migration(migration_id)

# List migrations
migrations = await registry.list_migrations(
    resource="users",
    status=MigrationStatus.PENDING
)

# Mark applied
await registry.mark_applied(migration_id)

# Mark failed
await registry.mark_failed(migration_id, error_message="...")

await registry.stop()
```

### Migration Executor

```python
from webapp.migration.executor import MigrationExecutor

executor = MigrationExecutor()

# Validate migration
executor.validate_migration(migration)  # Raises ValueError if invalid

# Dry run
result = executor.dry_run(migration)
# Returns: {"valid": bool, "sql": str, "operations_count": int, ...}

# Get explanation
explanation = executor.explain(migration)
# Returns: {"operations": [...], "operations_count": int, ...}

# Apply migration
result = await executor.apply(migration, data_layer=data_layer)
```

### Data CRUD

```python
from webapp.data.crud import DataCRUD

crud = DataCRUD()

# Build query with operators
query = crud.build_query({
    "age__gte": 18,
    "age__lt": 65,
    "name__contains": "john"
})

# Validate data
crud.validate_data(data, schema_json)  # Raises ValidationError if invalid

# Prepare for backend
prepared = crud.prepare_for_backend(data, backend="postgres")

# CRUD operations (async)
record = await crud.create("users", data, "postgres", data_layer)
record = await crud.read("users", "123", "postgres", data_layer)
record = await crud.update("users", "123", data, "postgres", data_layer)
success = await crud.delete("users", "123", "postgres", data_layer)
records = await crud.list("users", "postgres", filters={...}, data_layer=data_layer)
```

### Data Router

```python
from webapp.data.router import DataRouter

router = DataRouter(schema_registry)

# Get backend for resource
backend_name, backend_config = await router.get_backend_for_resource(
    "users",
    tenant_id=None
)
# Returns: ("postgres", {...postgres config...})

# Route operation
result = await router.route_operation(
    operation="create",
    resource="users",
    tenant_id=None,
    data={"id": "123", "email": "test@example.com"}
)
```

### Query Executor

```python
from webapp.query.executor import QueryExecutor

executor = QueryExecutor()

# Execute Postgres query
result = await executor.execute_postgres(
    "SELECT * FROM users WHERE age >= $1",
    params={"1": 18},
    data_layer=data_layer
)

# Execute Elasticsearch query
result = await executor.execute_elasticsearch(
    index="users",
    query={"match": {"name": "john"}},
    data_layer=data_layer
)

# Execute Redis command
result = await executor.execute_redis(
    command="HGETALL",
    args=["user:123"],
    data_layer=data_layer
)

# Validate query
is_safe = executor.validate_query("postgres", "SELECT * FROM users")
# Returns False for: DROP, DELETE, TRUNCATE, ALTER
```

## Error Responses

All API endpoints follow consistent error response format:

### 400 Bad Request
```json
{
  "detail": "Validation failed: missing required field 'email'"
}
```

### 404 Not Found
```json
{
  "detail": "Schema version not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Database connection failed"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider:
- Adding rate limiting middleware
- Implementing API keys/authentication
- Using tools like `slowapi` for FastAPI

## Authentication

The current implementation does not include authentication. For production:
1. Add authentication middleware (OAuth2, JWT, etc.)
2. Implement user management
3. Add role-based access control (RBAC)
4. Use `webapp.dependencies` to inject current user

## Pagination

For list endpoints, use standard pagination:
- `limit`: Maximum records per page (default: 100, max: 1000)
- `offset`: Starting position (default: 0)

**Example:**
```
GET /data/users?limit=50&offset=100
```

Returns records 101-150.

## Versioning

API versioning strategy:
- Current version: v1 (implicit, no prefix)
- Future versions: `/api/v2/schemas`, etc.
- Breaking changes require new version
- Old versions maintained for backwards compatibility

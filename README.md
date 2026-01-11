# Satellites WebApp

A unified admin panel for schema design, data exploration, and queries across multiple backends (Redis, Postgres, Elasticsearch, FAISS).

## Features

- 📋 **Schema Registry** - Design and manage data schemas with versioning and JSON Schema validation
- ⚙️ **Migration System** - Automatic migration generation with diff detection and SQL/ES command generation
- 💾 **Data Explorer** - Browse and manage data across all backends with a unified interface
- 🔍 **Query Console** - Execute raw queries against any backend (SQL, ES queries, Redis commands)
- 🎨 **Modern UI** - Server-rendered templates with HTMX and TailwindCSS

## Architecture

```
webapp/
├── schema/          # Schema registry and validation
│   ├── models.py    # Pydantic models for schemas
│   ├── validator.py # JSON Schema validation
│   ├── registry.py  # Schema CRUD operations
│   └── routes.py    # FastAPI routes for schemas
├── migration/       # Migration system
│   ├── models.py    # Migration models
│   ├── differ.py    # Schema diff detection
│   ├── executor.py  # Migration execution
│   ├── registry.py  # Migration tracking
│   ├── routes.py    # Migration UI routes
│   └── generators/  # Backend-specific SQL/command generators
│       ├── postgres.py
│       └── elasticsearch.py
├── data/            # Data API
│   ├── crud.py      # Generic CRUD operations
│   ├── router.py    # Backend routing logic
│   └── routes.py    # Data explorer routes
├── query/           # Query console
│   ├── executor.py  # Query execution
│   └── routes.py    # Console UI routes
├── templates/       # Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── schema/
│   ├── migration/
│   ├── data/
│   └── query/
├── app.py           # FastAPI application factory
├── config.py        # Application configuration
└── dependencies.py  # FastAPI dependencies
```

## Quick Start

### Installation

```bash
# Navigate to the project directory
cd /path/to/satellites/.worktrees/webapp

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
cd data_layer
pip install -e ".[dev]"
```

### Running the Server

```bash
# From the webapp directory
python run.py
```

Or using uvicorn directly:

```bash
uvicorn webapp:create_app --factory --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## API Documentation

### Schema Registry API

#### Create Schema Version

Creates a new schema version with JSON Schema and backend configurations.

**Endpoint:** `POST /schemas`
**UI:** `GET /schemas/new`

**Request Body:**
```json
{
  "resource": "users",
  "schema_json": {
    "type": "object",
    "properties": {
      "id": {"type": "string"},
      "email": {"type": "string", "format": "email"},
      "name": {"type": "string"}
    },
    "required": ["id", "email"]
  },
  "x_backends": {
    "postgres": {
      "table": "users",
      "primary_key": ["id"],
      "columns": {
        "id": {"type": "text", "nullable": false},
        "email": {"type": "text", "nullable": false},
        "name": {"type": "text", "nullable": true}
      },
      "indexes": [
        {
          "name": "idx_users_email",
          "columns": ["email"],
          "unique": true
        }
      ]
    },
    "elasticsearch": {
      "index": "users",
      "mappings": {
        "properties": {
          "id": {"type": "keyword"},
          "email": {"type": "keyword"},
          "name": {"type": "text"}
        }
      }
    }
  }
}
```

#### List Schema Versions

**Endpoint:** `GET /schemas`
**Response:** List of all schema versions

#### Get Schema Details

**Endpoint:** `GET /schemas/{schema_id}`
**Response:** Schema version details with metadata

#### Publish Schema

**Endpoint:** `POST /schemas/{schema_id}/publish`
**Response:** Updated schema with status="published"

### Migration API

#### List Migrations

**Endpoint:** `GET /migrations`
**Query Params:**
- `resource` (optional): Filter by resource name
- `backend` (optional): Filter by backend
- `status` (optional): Filter by status (pending/applied/failed)

#### Preview Migration

**Endpoint:** `GET /migrations/{migration_id}/preview`
**Response:** Migration details with generated SQL/commands

#### Generate Migration

Automatically generates migration by diffing two schema versions.

**Process:**
1. Detect changes using `SchemaDiffer`
2. Generate operations (add_column, drop_column, etc.)
3. Generate backend-specific SQL/commands
4. Store migration with status="pending"

**Postgres Example Output:**
```sql
ALTER TABLE users ADD COLUMN phone TEXT;
CREATE INDEX idx_users_phone ON users (phone);
```

**Elasticsearch Example Output:**
```json
[
  {
    "action": "update_mapping",
    "index": "users",
    "body": {
      "properties": {
        "phone": {"type": "keyword"}
      }
    }
  }
]
```

### Data API

#### Data Explorer

**Endpoint:** `GET /data`
**Description:** Browse available resources and data

#### List Resource Data

**Endpoint:** `GET /data/{resource}`
**Query Params:**
- `limit` (default: 100): Maximum records to return
- `offset` (default: 0): Pagination offset
- Any field name for filtering (e.g., `?email=test@example.com`)

**Filter Operators:**
- `field__gte`: Greater than or equal
- `field__gt`: Greater than
- `field__lte`: Less than or equal
- `field__lt`: Less than
- `field__ne`: Not equal
- `field__contains`: Contains substring

**Example:**
```
GET /data/users?age__gte=18&age__lt=65&name__contains=john
```

#### Get Record

**Endpoint:** `GET /data/{resource}/{record_id}`
**Response:** Single record details

#### Create Record

**Endpoint:** `POST /data/{resource}`
**Request Body:** JSON object matching schema

#### Update Record

**Endpoint:** `PUT /data/{resource}/{record_id}`
**Request Body:** JSON object with updates

#### Delete Record

**Endpoint:** `DELETE /data/{resource}/{record_id}`

### Query Console API

#### Execute Query

**Endpoint:** `POST /query/execute`
**Request Body:**
```json
{
  "backend": "postgres",
  "query": "SELECT * FROM users WHERE email LIKE '%@example.com' LIMIT 10"
}
```

**For Elasticsearch:**
```json
{
  "backend": "elasticsearch",
  "index": "users",
  "query": {
    "match": {
      "name": "john"
    }
  }
}
```

**For Redis:**
```json
{
  "backend": "redis",
  "command": "HGETALL",
  "args": ["user:123"]
}
```

## Components

### Schema Validator

Validates schemas and ensures consistency between JSON Schema and backend configurations.

**Location:** `webapp/schema/validator.py`

**Features:**
- JSON Schema Draft 7 validation
- Data validation against schemas
- Backend configuration validation
- Consistency checks between schema and backends

**Example:**
```python
from webapp.schema.validator import SchemaValidator

validator = SchemaValidator()

# Validate schema structure
validator.validate_schema(schema_json)

# Validate data against schema
validator.validate_data(data, schema_json)

# Validate backend config
validator.validate_postgres_backend(postgres_config)

# Check consistency
validator.validate_consistency(schema_json, backends)
```

### Schema Differ

Detects changes between schema versions and generates migration operations.

**Location:** `webapp/migration/differ.py`

**Example:**
```python
from webapp.migration.differ import SchemaDiffer

differ = SchemaDiffer()

# Detect changes in Postgres backend
operations = differ.diff_postgres(old_backend, new_backend)

# Operations list will contain:
# - ADD_COLUMN
# - DROP_COLUMN
# - MODIFY_COLUMN
# - CREATE_INDEX
# - DROP_INDEX
```

### Migration Generators

Generate backend-specific SQL and commands from migration operations.

**Postgres Generator** (`webapp/migration/generators/postgres.py`):
```python
from webapp.migration.generators.postgres import PostgresGenerator

generator = PostgresGenerator()
sql = generator.generate_migration_sql(operations)
```

**Elasticsearch Generator** (`webapp/migration/generators/elasticsearch.py`):
```python
from webapp.migration.generators.elasticsearch import ElasticsearchGenerator

generator = ElasticsearchGenerator()
commands = generator.generate_migration_script(operations)  # JSON output
python_code = generator.generate_python_api(operations)     # Python API calls
```

### Data Router

Routes data operations to the correct backend based on schema configuration.

**Location:** `webapp/data/router.py`

**Example:**
```python
from webapp.data.router import DataRouter

router = DataRouter(schema_registry)

# Get backend for a resource
backend_name, backend_config = await router.get_backend_for_resource("users")

# Route operation
result = await router.route_operation(
    operation="create",
    resource="users",
    data={"id": "123", "email": "test@example.com"}
)
```

### Query Executor

Executes raw queries against backends with basic validation.

**Location:** `webapp/query/executor.py`

**Example:**
```python
from webapp.query.executor import QueryExecutor

executor = QueryExecutor()

# Execute Postgres query
result = await executor.execute_postgres(
    "SELECT * FROM users WHERE age >= 18",
    params=None,
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
    command="GET",
    args=["user:123"],
    data_layer=data_layer
)
```

## Configuration

### WebApp Config

**Location:** `webapp/config.py`

```python
from webapp.config import WebAppConfig

config = WebAppConfig(
    title="Satellites Admin",
    debug=True,  # Enable debug mode
    templates_dir=Path("webapp/templates"),
    static_dir=Path("webapp/static")
)
```

### DataLayer Integration

To connect to actual backends, pass a `DataLayerConfig`:

```python
from data_layer import DataLayerConfig
from webapp import create_app

data_layer_config = DataLayerConfig(
    postgres_url="postgresql://user:pass@localhost:5432/dbname",
    elasticsearch_url="http://localhost:9200",
    redis_url="redis://localhost:6379/0",
    faiss_index_dir="/path/to/faiss/indexes"
)

app = create_app(data_layer_config=data_layer_config)
```

## Testing

### Run All Tests

```bash
# From the webapp directory
source .venv/bin/activate
PYTHONPATH=/path/to/webapp:$PYTHONPATH python -m pytest tests/
```

### Run Specific Test Suites

```bash
# Schema tests
pytest tests/webapp/test_schema_*.py

# Migration tests
pytest tests/webapp/test_migration_*.py

# Data tests
pytest tests/webapp/test_data_*.py

# Route tests
pytest tests/webapp/test_*_routes.py
```

### Test Coverage

The test suite includes:
- **Unit tests** for all core components (validators, differs, generators, CRUD)
- **Route tests** for all UI endpoints
- **Integration tests** for full workflows (see `tests/integration/`)

Total test count: 50+ tests across all modules

## Development

### Project Structure

```
.
├── webapp/              # Main application code
├── tests/               # Test suites
│   ├── webapp/          # Unit tests
│   └── integration/     # Integration tests
├── data_layer/          # Backend abstraction layer
├── docs/                # Documentation
│   └── plans/           # Implementation plans
├── run.py               # Development server
├── pyproject.toml       # Dependencies (in data_layer/)
└── README.md            # This file
```

### Adding a New Backend

1. **Create generator** in `webapp/migration/generators/{backend}.py`
2. **Implement differ** in `webapp/migration/differ.py` (add `diff_{backend}` method)
3. **Update router** in `webapp/data/router.py` (add to priority list)
4. **Update query executor** in `webapp/query/executor.py` (add `execute_{backend}` method)
5. **Add tests** for the new backend

### Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Document all public APIs with docstrings
- Keep functions focused and single-purpose
- Write tests for all new features

## UI Screenshots

### Home Page
- Feature cards for Schemas, Data Explorer, Query Console
- System status indicators
- Quick start guide

### Schema Designer
- JSON Schema editor with validation
- Backend configuration (Postgres, Elasticsearch, Redis, FAISS)
- Schema versioning and publishing

### Migration Preview
- Side-by-side diff view
- Generated SQL/commands
- Apply/rollback controls

### Data Explorer
- Resource browser with sidebar
- Table view with filtering
- Record detail view with edit/delete

### Query Console
- Backend selector (Postgres/ES/Redis)
- Syntax-highlighted query editor
- Results display with formatting
- Query history

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -e ".[dev]"`
3. PYTHONPATH includes the webapp directory

### Template Not Found

Ensure `templates_dir` in `WebAppConfig` points to the correct location:
```python
WebAppConfig(templates_dir=Path(__file__).parent / "templates")
```

### Database Connection Issues

When integrating with DataLayer:
1. Verify connection strings in `DataLayerConfig`
2. Ensure backend services are running
3. Check network connectivity and credentials

## Contributing

1. Create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Follow existing code style
5. Update documentation
6. Submit pull request

## License

[Add your license information here]

## Support

For issues and questions:
- GitHub Issues: [repository URL]
- Documentation: `/docs`
- API Docs: Run server and visit `/docs` (FastAPI auto-generated)

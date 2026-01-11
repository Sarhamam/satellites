# WebApp Design - Schema Designer & Data Explorer

## Overview

A unified admin panel for schema design, data exploration, and queries across the multi-backend data layer (Redis, Postgres, Elasticsearch, FAISS).

**Primary use cases:**
- Visually define and version schemas
- Generate and apply migrations across backends
- Browse and edit data with tenant isolation
- Run ad-hoc queries against any backend

**Tech stack:** FastAPI + HTMX + Jinja2 + TailwindCSS

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         WebApp                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Schema       │  │ Data         │  │ Query                │  │
│  │ Designer     │  │ Explorer     │  │ Console              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         HTMX + Jinja templates, TailwindCSS                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Schema       │  │ Migration    │  │ Data                 │  │
│  │ Registry     │  │ Generator    │  │ API                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│     Redis    │    Postgres    │   Elasticsearch   │   FAISS     │
└─────────────────────────────────────────────────────────────────┘
```

**Three main UI modules:**

1. **Schema Designer** - Visual editor for JSON Schema + backend mappings, version management, tenant extensions
2. **Data Explorer** - Browse/filter/edit data across all backends for any resource
3. **Query Console** - Run SQL, ES queries, vector searches, Redis commands

**Backend services:**

1. **Schema Registry** - Stores schemas in Postgres, handles versioning + tenant extensions
2. **Migration Generator** - Diffs schema versions, generates SQL/ES mappings, tracks apply status
3. **Data API** - Generic CRUD that routes to correct backend based on schema's `x-backends`

## Schema IR Format

The core data structure - JSON Schema with backend extensions:

```json
{
  "resource": "users",
  "version": 3,
  "tenant_id": null,
  "extends": null,
  "schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["id", "email", "created_at"],
    "properties": {
      "id": { "type": "string" },
      "email": { "type": "string", "format": "email" },
      "name": { "type": "string" },
      "created_at": { "type": "string", "format": "date-time" },
      "attributes": { "type": "object" }
    }
  },
  "x-backends": {
    "postgres": {
      "table": "users",
      "primary_key": ["id"],
      "columns": {
        "id": { "type": "text", "nullable": false },
        "email": { "type": "text", "nullable": false },
        "name": { "type": "text", "nullable": true },
        "created_at": { "type": "timestamptz", "nullable": false },
        "tenant_id": { "type": "text", "nullable": false },
        "attributes": { "type": "jsonb", "nullable": true }
      },
      "indexes": [
        { "name": "users_tenant_email_uq", "unique": true, "columns": ["tenant_id", "email"] }
      ]
    },
    "elasticsearch": {
      "index": "users",
      "mappings": {
        "properties": {
          "email": { "type": "keyword" },
          "name": { "type": "text" },
          "tenant_id": { "type": "keyword" },
          "attributes": { "type": "object", "dynamic": true }
        }
      }
    },
    "redis": {
      "pattern": "tenant:{tenant_id}:user:{id}",
      "ttl_seconds": 3600,
      "encoding": "json"
    },
    "faiss": {
      "namespace": "user_embeddings",
      "dimension": 1536,
      "id_field": "id",
      "metadata_fields": ["tenant_id", "email"]
    }
  }
}
```

**Key design decisions:**

- `tenant_id: null` = global base schema
- `extends: "schema_version_id"` = for tenant extensions (points to parent)
- `tenant_id` column added automatically to all backends for row-level isolation
- Each backend section is optional (not all resources need all backends)

## Database Schema (Registry)

```sql
-- Tenants
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT REFERENCES tenants(id),  -- for future group tenants
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Schema versions (the core registry)
CREATE TABLE schema_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource TEXT NOT NULL,              -- e.g. "users", "documents"
    version INT NOT NULL,
    tenant_id TEXT REFERENCES tenants(id),  -- null = global base
    extends_id UUID REFERENCES schema_versions(id),  -- parent schema for extensions
    schema_json JSONB NOT NULL,          -- the full Schema IR
    status TEXT NOT NULL DEFAULT 'draft', -- draft | published | deprecated
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    checksum TEXT NOT NULL,              -- for change detection

    UNIQUE(resource, version, tenant_id)
);

-- Migrations (generated diffs between versions)
CREATE TABLE migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_version_id UUID REFERENCES schema_versions(id),  -- null for initial
    to_version_id UUID NOT NULL REFERENCES schema_versions(id),
    backend TEXT NOT NULL,               -- postgres | elasticsearch | redis | faiss
    operations JSONB NOT NULL,           -- list of operations to apply
    sql_up TEXT,                         -- generated SQL (for postgres)
    sql_down TEXT,                       -- rollback SQL
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | applied | failed
    created_at TIMESTAMPTZ DEFAULT NOW(),
    applied_at TIMESTAMPTZ,
    applied_by TEXT
);

-- Indexes
CREATE INDEX idx_schema_versions_resource ON schema_versions(resource);
CREATE INDEX idx_schema_versions_tenant ON schema_versions(tenant_id);
CREATE INDEX idx_schema_versions_status ON schema_versions(status);
CREATE INDEX idx_migrations_status ON migrations(status);
```

## Project Structure

```
satellites/
├── data_layer/                    # (already designed)
│   └── ...
├── webapp/
│   ├── __init__.py
│   ├── app.py                     # FastAPI app, mount routes
│   ├── config.py                  # WebApp config
│   ├── dependencies.py            # DI: get_data_layer, get_current_tenant
│   │
│   ├── schema/                    # Schema Registry + Designer
│   │   ├── __init__.py
│   │   ├── models.py              # Pydantic: SchemaVersion, Migration
│   │   ├── registry.py            # CRUD for schema_versions table
│   │   ├── validator.py           # JSON Schema + x-backends validation
│   │   ├── merger.py              # Merge base + tenant extensions
│   │   └── routes.py              # /schemas/* endpoints
│   │
│   ├── migration/                 # Migration Generator
│   │   ├── __init__.py
│   │   ├── differ.py              # Diff two schema versions
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py        # Generate ALTER TABLE, CREATE INDEX
│   │   │   ├── elasticsearch.py   # Generate mapping updates
│   │   │   ├── redis.py           # Key pattern changes (mostly manual)
│   │   │   └── faiss.py           # Dimension changes, reindex flags
│   │   ├── executor.py            # Apply approved migrations
│   │   └── routes.py              # /migrations/* endpoints
│   │
│   ├── data/                      # Data Explorer + API
│   │   ├── __init__.py
│   │   ├── router.py              # Route to correct backend based on schema
│   │   ├── crud.py                # Generic create/read/update/delete
│   │   └── routes.py              # /data/{resource}/* endpoints
│   │
│   ├── query/                     # Query Console
│   │   ├── __init__.py
│   │   ├── executor.py            # Run raw SQL, ES queries, etc.
│   │   └── routes.py              # /query/* endpoints
│   │
│   ├── templates/                 # Jinja2 templates
│   │   ├── base.html
│   │   ├── schema/
│   │   │   ├── list.html
│   │   │   ├── editor.html        # Visual schema designer
│   │   │   └── diff.html          # Version comparison
│   │   ├── data/
│   │   │   ├── explorer.html      # Browse/filter records
│   │   │   └── record.html        # Single record view/edit
│   │   └── query/
│   │       └── console.html       # Multi-backend query interface
│   │
│   └── static/
│       ├── css/
│       │   └── app.css            # Tailwind compiled
│       └── js/
│           └── schema-editor.js   # Visual editor interactions
│
├── tests/
│   ├── webapp/
│   │   ├── test_schema_registry.py
│   │   ├── test_migrations.py
│   │   └── test_data_api.py
│   └── ...
│
├── docker/
│   └── docker-compose.yml
│
└── pyproject.toml
```

## Key Workflows

### 1. Create/Edit Schema (Designer)

```
User opens /schemas/users/edit
    │
    ▼
Load current published version (or blank for new)
    │
    ▼
Visual editor: add fields, set types, configure backends
    │
    ▼
[Save Draft] → POST /api/schemas
    │
    ├─ Validate JSON Schema structure
    ├─ Validate x-backends config (types, constraints)
    └─ Store as new version with status=draft
    │
    ▼
[Generate Migration] → POST /api/migrations/generate
    │
    ├─ Diff draft vs current published
    ├─ Generate per-backend operations
    └─ Store migration with status=pending
    │
    ▼
Show migration preview (SQL, ES mappings, etc.)
    │
    ▼
[Approve & Apply] → POST /api/migrations/{id}/apply
    │
    ├─ Execute migration per backend
    ├─ Mark migration status=applied
    └─ Mark schema version status=published
```

### 2. Browse Data (Explorer)

```
User opens /data/users
    │
    ▼
Load schema for "users" resource
    │
    ▼
Render filter UI based on schema fields
    │
    ▼
[Search] → GET /api/data/users?tenant_id=X&email=...
    │
    ├─ Check x-backends to pick query backend
    │   (ES for search, Postgres for exact, etc.)
    ├─ Add tenant_id filter automatically
    └─ Return paginated results
    │
    ▼
[Click record] → GET /api/data/users/{id}
    │
    ├─ Fetch from primary backend (Postgres)
    ├─ Optionally enrich from other backends
    └─ Render editable form based on schema
```

### 3. Run Query (Console)

```
User opens /query
    │
    ▼
Select backend: Postgres | Elasticsearch | Redis | FAISS
    │
    ▼
Enter query (SQL, ES JSON, Redis command, vector)
    │
    ▼
[Execute] → POST /api/query
    │
    ├─ Validate query is read-only (optional safety)
    ├─ Inject tenant_id filter if applicable
    ├─ Execute against selected backend
    └─ Return results as table/JSON
```

## Multi-Tenancy Model

- **Shared tables with tenant_id column** - All tenants in same tables, filtered by tenant_id
- **Row-level security** - Postgres RLS policies enforce isolation
- **Tenant extensions on base schemas** - Global base schema, tenants can add fields
- **Future: group tenants** - `tenants.parent_id` enables tenant hierarchy

## Dependencies (additions to pyproject.toml)

```toml
dependencies = [
    # ... existing data_layer deps ...
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
    "jinja2>=3.1",
    "python-multipart>=0.0.6",  # form handling
    "jsonschema>=4.21",         # schema validation
]
```

## Future Extensions

1. **Git export** - Export schemas to `schemas/{resource}/v{n}.json` for PR review
2. **Auto-apply in dev** - Toggle to skip approval for development environments
3. **Tenant group inheritance** - Override chains: global → group → tenant
4. **Audit log** - Track all schema and data changes
5. **Schema contracts** - Validate data at write time against published schema

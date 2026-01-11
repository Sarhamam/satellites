"""Integration tests for the complete webapp flow."""

import pytest
from datetime import datetime, timezone

from webapp.schema.models import SchemaVersionCreate, SchemaStatus
from webapp.schema.registry import SchemaRegistry
from webapp.schema.validator import SchemaValidator
from webapp.migration.differ import SchemaDiffer
from webapp.migration.registry import MigrationRegistry
from webapp.migration.generators.postgres import PostgresGenerator
from webapp.migration.executor import MigrationExecutor
from webapp.data.crud import DataCRUD
from webapp.data.router import DataRouter


@pytest.fixture
async def schema_registry():
    """Create schema registry for testing."""
    registry = SchemaRegistry(connection_string=None)
    await registry.start()
    yield registry
    await registry.stop()


@pytest.fixture
async def migration_registry():
    """Create migration registry for testing."""
    registry = MigrationRegistry(connection_string=None)
    await registry.start()
    yield registry
    await registry.stop()


@pytest.mark.asyncio
async def test_full_workflow(schema_registry, migration_registry):
    """
    Test complete workflow:
    1. Create initial schema
    2. Validate schema
    3. Create updated schema with changes
    4. Generate migration
    5. Preview and validate migration
    6. Apply migration (dry run)
    7. Query data (stub)
    """

    # Step 1: Create initial schema (v1)
    validator = SchemaValidator()

    initial_schema_data = SchemaVersionCreate(
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

    # Validate initial schema
    validator.validate_schema(initial_schema_data.schema_json)
    validator.validate_postgres_backend(initial_schema_data.x_backends["postgres"])
    validator.validate_consistency(
        initial_schema_data.schema_json,
        initial_schema_data.x_backends
    )

    # Create initial schema version
    v1 = await schema_registry.create_schema_version(
        initial_schema_data,
        created_by="test_user"
    )

    assert v1.resource == "users"
    assert v1.version == 1
    assert v1.status == SchemaStatus.DRAFT

    # Publish v1
    v1_published = await schema_registry.publish_schema_version(v1.id)
    assert v1_published.status == SchemaStatus.PUBLISHED

    # Step 2: Create updated schema (v2) with changes
    updated_schema_data = SchemaVersionCreate(
        resource="users",
        schema_json={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},  # NEW FIELD
                "name": {"type": "string"},   # NEW FIELD
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
                    "phone": {"type": "text", "nullable": True},  # NEW
                    "name": {"type": "text", "nullable": True},   # NEW
                },
                "indexes": [
                    {
                        "name": "idx_users_email",
                        "columns": ["email"],
                        "unique": True
                    }
                ],
            }
        },
    )

    # Validate updated schema
    validator.validate_schema(updated_schema_data.schema_json)
    validator.validate_postgres_backend(updated_schema_data.x_backends["postgres"])
    validator.validate_consistency(
        updated_schema_data.schema_json,
        updated_schema_data.x_backends
    )

    v2 = await schema_registry.create_schema_version(
        updated_schema_data,
        created_by="test_user"
    )

    assert v2.version == 2

    # Step 3: Generate migration by diffing v1 and v2
    differ = SchemaDiffer()
    operations = differ.diff_postgres(
        v1.x_backends["postgres"],
        v2.x_backends["postgres"]
    )

    assert len(operations) == 3  # 2 ADD_COLUMN + 1 CREATE_INDEX

    # Generate SQL
    generator = PostgresGenerator()
    sql = generator.generate_migration_sql(operations)

    assert "ADD COLUMN phone TEXT" in sql
    assert "ADD COLUMN name TEXT" in sql
    assert "CREATE UNIQUE INDEX idx_users_email" in sql

    # Step 4: Create migration
    migration = await migration_registry.create_migration(
        from_version_id=v1.id,
        to_version_id=v2.id,
        resource="users",
        backend="postgres",
        operations=operations,
        generated_sql=sql,
    )

    assert migration.resource == "users"
    assert migration.backend == "postgres"
    assert len(migration.operations) == 3

    # Step 5: Preview and validate migration
    executor = MigrationExecutor()

    # Validate migration
    executor.validate_migration(migration)  # Should not raise

    # Dry run
    dry_run_result = executor.dry_run(migration)
    assert dry_run_result["valid"] is True
    assert dry_run_result["operations_count"] == 3

    # Get explanation
    explanation = executor.explain(migration)
    assert len(explanation["operations"]) == 3
    assert explanation["resource"] == "users"

    # Step 6: Apply migration (stub - would actually execute SQL)
    result = await executor.apply(migration, data_layer=None)
    assert result["success"] is True

    # Mark as applied
    applied_migration = await migration_registry.mark_applied(migration.id)
    assert applied_migration.status.value == "applied"
    assert applied_migration.applied_at is not None

    # Step 7: Data operations (stub - would use actual data layer)
    crud = DataCRUD()

    # Validate data against v2 schema
    test_data = {
        "id": "user-123",
        "email": "test@example.com",
        "phone": "+1234567890",
        "name": "Test User"
    }

    # Should not raise
    crud.validate_data(test_data, v2.schema_json)

    # Prepare for backend
    prepared_data = crud.prepare_for_backend(test_data, "postgres")
    assert "id" in prepared_data
    assert "email" in prepared_data

    # Build query
    query = crud.build_query({"email": "test@example.com"})
    assert query["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_data_routing(schema_registry):
    """Test data routing to correct backend based on schema."""

    # Create schema with multiple backends
    schema_data = SchemaVersionCreate(
        resource="products",
        schema_json={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        },
        x_backends={
            "postgres": {
                "table": "products",
                "primary_key": ["id"],
                "columns": {
                    "id": {"type": "text", "nullable": False},
                    "name": {"type": "text", "nullable": True},
                },
            },
            "elasticsearch": {
                "index": "products",
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {"type": "text"},
                    }
                },
            },
        },
    )

    schema_version = await schema_registry.create_schema_version(
        schema_data,
        created_by="test_user"
    )

    # Test routing
    router = DataRouter(schema_registry)

    backend_info = await router.get_backend_for_resource("products", tenant_id=None)

    assert backend_info is not None
    backend_name, backend_config = backend_info

    # Should prioritize postgres
    assert backend_name == "postgres"
    assert backend_config["table"] == "products"

    # Route an operation
    result = await router.route_operation(
        operation="create",
        resource="products",
        tenant_id=None,
    )

    assert result["backend"] == "postgres"
    assert result["operation"] == "create"


@pytest.mark.asyncio
async def test_schema_versioning(schema_registry):
    """Test schema versioning and retrieval."""

    # Create multiple versions
    for i in range(1, 4):
        schema_data = SchemaVersionCreate(
            resource="orders",
            schema_json={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "version": {"type": "integer", "const": i},
                },
            },
            x_backends={},
        )

        version = await schema_registry.create_schema_version(
            schema_data,
            created_by="test_user"
        )

        assert version.version == i

    # Get latest version
    latest = await schema_registry.get_latest_schema_version("orders", tenant_id=None)
    assert latest is not None
    assert latest.version == 3

    # List all versions
    versions = await schema_registry.list_schema_versions("orders", tenant_id=None)
    assert len(versions) == 3
    assert versions[0].version == 3  # Ordered desc
    assert versions[1].version == 2
    assert versions[2].version == 1


@pytest.mark.asyncio
async def test_migration_failure_handling(migration_registry):
    """Test migration failure tracking."""

    from webapp.migration.models import MigrationOperation, MigrationOperationType

    # Create a migration
    operations = [
        MigrationOperation(
            type=MigrationOperationType.ADD_COLUMN,
            table="test_table",
            details={"column": "test_col", "column_type": "text"},
        )
    ]

    migration = await migration_registry.create_migration(
        from_version_id=None,
        to_version_id="test-v1",
        resource="test",
        backend="postgres",
        operations=operations,
        generated_sql="ALTER TABLE test_table ADD COLUMN test_col TEXT;",
    )

    # Simulate failure
    failed_migration = await migration_registry.mark_failed(
        migration.id,
        error_message="Syntax error at line 1"
    )

    assert failed_migration.status.value == "failed"
    assert failed_migration.error_message == "Syntax error at line 1"

    # List failed migrations
    failed_migrations = await migration_registry.list_migrations(
        status=failed_migration.status
    )

    assert len(failed_migrations) == 1
    assert failed_migrations[0].id == migration.id


@pytest.mark.asyncio
async def test_query_operators():
    """Test query building with operators."""

    crud = DataCRUD()

    # Test various operators
    query = crud.build_query({
        "age__gte": 18,
        "age__lt": 65,
        "name__contains": "john",
        "email": "test@example.com",
        "status__ne": "inactive",
    })

    assert "age" in query or "age__gte" in query
    assert "name" in query or "name__contains" in query
    assert query.get("email") == "test@example.com"

"""Migration registry for managing migrations."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from webapp.migration.models import Migration, MigrationOperation, MigrationStatus


class MigrationRegistry:
    """Registry for managing migrations.

    For testing, can use in-memory storage (connection_string=None).
    For production, connects to Postgres via asyncpg.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the migration registry.

        Args:
            connection_string: Postgres connection string, or None for in-memory mock
        """
        self.connection_string = connection_string
        self._pool = None
        # In-memory storage for testing
        self._migrations: dict[str, Migration] = {}

    async def start(self):
        """Initialize database connection pool."""
        if self.connection_string:
            # TODO: Initialize asyncpg connection pool
            # import asyncpg
            # self._pool = await asyncpg.create_pool(self.connection_string)
            pass

    async def stop(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()

    async def create_migration(
        self,
        from_version_id: Optional[str],
        to_version_id: str,
        resource: str,
        backend: str,
        operations: list[MigrationOperation],
        generated_sql: Optional[str] = None,
        rollback_sql: Optional[str] = None,
    ) -> Migration:
        """Create a new migration.

        Args:
            from_version_id: Source schema version ID (None for initial)
            to_version_id: Target schema version ID
            resource: Resource name
            backend: Backend name (postgres, elasticsearch, etc.)
            operations: List of migration operations
            generated_sql: Generated SQL/commands
            rollback_sql: Optional rollback SQL

        Returns:
            Created migration
        """
        migration = Migration(
            id=str(uuid.uuid4()),
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            resource=resource,
            backend=backend,
            operations=operations,
            status=MigrationStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            generated_sql=generated_sql,
            rollback_sql=rollback_sql,
        )

        # Store in memory (or would insert into DB)
        self._migrations[migration.id] = migration

        return migration

    async def get_migration(self, migration_id: str) -> Optional[Migration]:
        """Get a migration by ID.

        Args:
            migration_id: Migration ID

        Returns:
            Migration or None if not found
        """
        return self._migrations.get(migration_id)

    async def list_migrations(
        self,
        resource: Optional[str] = None,
        backend: Optional[str] = None,
        status: Optional[MigrationStatus] = None,
    ) -> list[Migration]:
        """List migrations with optional filters.

        Args:
            resource: Filter by resource name
            backend: Filter by backend name
            status: Filter by status

        Returns:
            List of migrations
        """
        migrations = list(self._migrations.values())

        # Apply filters
        if resource:
            migrations = [m for m in migrations if m.resource == resource]
        if backend:
            migrations = [m for m in migrations if m.backend == backend]
        if status:
            migrations = [m for m in migrations if m.status == status]

        # Sort by created_at descending
        return sorted(migrations, key=lambda m: m.created_at, reverse=True)

    async def mark_applied(self, migration_id: str) -> Migration:
        """Mark a migration as applied.

        Args:
            migration_id: Migration ID

        Returns:
            Updated migration

        Raises:
            ValueError: If migration not found
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migration {migration_id} not found")

        migration.status = MigrationStatus.APPLIED
        migration.applied_at = datetime.now(timezone.utc)

        return migration

    async def mark_failed(self, migration_id: str, error_message: str) -> Migration:
        """Mark a migration as failed.

        Args:
            migration_id: Migration ID
            error_message: Error message

        Returns:
            Updated migration

        Raises:
            ValueError: If migration not found
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migration {migration_id} not found")

        migration.status = MigrationStatus.FAILED
        migration.error_message = error_message

        return migration

    async def mark_rolled_back(self, migration_id: str) -> Migration:
        """Mark a migration as rolled back.

        Args:
            migration_id: Migration ID

        Returns:
            Updated migration

        Raises:
            ValueError: If migration not found
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            raise ValueError(f"Migration {migration_id} not found")

        migration.status = MigrationStatus.ROLLED_BACK

        return migration

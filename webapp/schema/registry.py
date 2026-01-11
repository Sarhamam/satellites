"""Schema registry for managing schema versions."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from webapp.schema.models import (
    SchemaVersion,
    SchemaVersionCreate,
    SchemaStatus,
)


class SchemaRegistry:
    """Schema registry for managing schema versions.

    For testing, can use in-memory storage (connection_string=None).
    For production, connects to Postgres via asyncpg.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the schema registry.

        Args:
            connection_string: Postgres connection string, or None for in-memory mock
        """
        self.connection_string = connection_string
        self._pool = None
        # In-memory storage for testing
        self._schemas: dict[str, SchemaVersion] = {}
        self._version_counters: dict[str, int] = {}

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

    def _calculate_checksum(self, schema_data: SchemaVersionCreate) -> str:
        """Calculate checksum for schema version."""
        # Create a stable JSON representation for hashing
        data = {
            "resource": schema_data.resource,
            "tenant_id": schema_data.tenant_id,
            "schema_json": schema_data.schema_json,
            "x_backends": schema_data.x_backends,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def create_schema_version(
        self, schema_data: SchemaVersionCreate, created_by: str
    ) -> SchemaVersion:
        """Create a new schema version.

        Args:
            schema_data: Schema version data
            created_by: User creating the schema

        Returns:
            Created schema version
        """
        # Get next version number for this resource
        resource_key = (
            f"{schema_data.tenant_id}:{schema_data.resource}"
            if schema_data.tenant_id
            else schema_data.resource
        )
        version_num = self._version_counters.get(resource_key, 0) + 1
        self._version_counters[resource_key] = version_num

        # Create schema version
        version = SchemaVersion(
            id=str(uuid.uuid4()),
            resource=schema_data.resource,
            version=version_num,
            tenant_id=schema_data.tenant_id,
            extends_id=schema_data.extends_id,
            schema_json=schema_data.schema_json,
            x_backends=schema_data.x_backends,
            status=SchemaStatus.DRAFT,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            checksum=self._calculate_checksum(schema_data),
        )

        # Store in memory (or would insert into DB)
        self._schemas[version.id] = version

        return version

    async def get_schema_version(self, version_id: str) -> Optional[SchemaVersion]:
        """Get a schema version by ID.

        Args:
            version_id: Schema version ID

        Returns:
            Schema version or None if not found
        """
        return self._schemas.get(version_id)

    async def get_latest_schema_version(
        self, resource: str, tenant_id: Optional[str] = None
    ) -> Optional[SchemaVersion]:
        """Get the latest version of a schema.

        Args:
            resource: Resource name
            tenant_id: Optional tenant ID

        Returns:
            Latest schema version or None if not found
        """
        # Filter schemas by resource and tenant
        matching = [
            s
            for s in self._schemas.values()
            if s.resource == resource and s.tenant_id == tenant_id
        ]

        if not matching:
            return None

        # Return the one with highest version number
        return max(matching, key=lambda s: s.version)

    async def list_schema_versions(
        self, resource: str, tenant_id: Optional[str] = None
    ) -> list[SchemaVersion]:
        """List all versions of a schema.

        Args:
            resource: Resource name
            tenant_id: Optional tenant ID

        Returns:
            List of schema versions, ordered by version desc
        """
        # Filter schemas by resource and tenant
        matching = [
            s
            for s in self._schemas.values()
            if s.resource == resource and s.tenant_id == tenant_id
        ]

        # Sort by version descending
        return sorted(matching, key=lambda s: s.version, reverse=True)

    async def publish_schema_version(self, version_id: str) -> SchemaVersion:
        """Publish a draft schema version.

        Args:
            version_id: Schema version ID

        Returns:
            Updated schema version

        Raises:
            ValueError: If schema not found or not in draft status
        """
        version = self._schemas.get(version_id)
        if not version:
            raise ValueError(f"Schema version {version_id} not found")

        if version.status != SchemaStatus.DRAFT:
            raise ValueError(f"Schema version {version_id} is not in draft status")

        # Update status
        version.status = SchemaStatus.PUBLISHED

        return version

    async def deprecate_schema_version(self, version_id: str) -> SchemaVersion:
        """Deprecate a published schema version.

        Args:
            version_id: Schema version ID

        Returns:
            Updated schema version

        Raises:
            ValueError: If schema not found or not published
        """
        version = self._schemas.get(version_id)
        if not version:
            raise ValueError(f"Schema version {version_id} not found")

        if version.status != SchemaStatus.PUBLISHED:
            raise ValueError(f"Schema version {version_id} is not published")

        # Update status
        version.status = SchemaStatus.DEPRECATED

        return version

    async def delete_schema_version(self, version_id: str) -> None:
        """Delete a draft schema version.

        Args:
            version_id: Schema version ID

        Raises:
            ValueError: If schema not found or not in draft status
        """
        version = self._schemas.get(version_id)
        if not version:
            raise ValueError(f"Schema version {version_id} not found")

        if version.status != SchemaStatus.DRAFT:
            raise ValueError(
                f"Cannot delete schema version {version_id} - only drafts can be deleted"
            )

        # Remove from storage
        del self._schemas[version_id]

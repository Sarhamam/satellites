"""Route data requests to appropriate backend based on schema."""

from typing import Any, Optional


class DataRouter:
    """Routes data operations to the correct backend based on schema config."""

    def __init__(self, schema_registry=None):
        """
        Initialize the data router.

        Args:
            schema_registry: Optional SchemaRegistry instance
        """
        self.schema_registry = schema_registry

    async def get_backend_for_resource(
        self, resource: str, tenant_id: Optional[str] = None
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Get the backend configuration for a resource.

        Args:
            resource: Resource name
            tenant_id: Optional tenant ID

        Returns:
            Tuple of (backend_name, backend_config) or None if not found
        """
        if not self.schema_registry:
            return None

        # Get latest schema version for resource
        schema_version = await self.schema_registry.get_latest_schema_version(
            resource, tenant_id
        )

        if not schema_version:
            return None

        # Check which backends are configured
        backends = schema_version.x_backends

        # Priority order: postgres, elasticsearch, redis, faiss
        for backend_name in ["postgres", "elasticsearch", "redis", "faiss"]:
            if backend_name in backends:
                return (backend_name, backends[backend_name])

        return None

    async def route_operation(
        self,
        operation: str,
        resource: str,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Route an operation to the appropriate backend.

        Args:
            operation: Operation name (create, read, update, delete, list)
            resource: Resource name
            tenant_id: Optional tenant ID
            **kwargs: Operation-specific arguments

        Returns:
            Operation result

        Raises:
            ValueError: If resource not found or backend not configured
        """
        backend_info = await self.get_backend_for_resource(resource, tenant_id)

        if not backend_info:
            raise ValueError(f"No backend configured for resource: {resource}")

        backend_name, backend_config = backend_info

        return {
            "backend": backend_name,
            "config": backend_config,
            "operation": operation,
            "resource": resource,
            "message": f"Would route {operation} to {backend_name} (stub)",
        }

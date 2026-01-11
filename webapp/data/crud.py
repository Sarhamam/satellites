"""Generic CRUD operations for data."""

from typing import Any
import jsonschema


class DataCRUD:
    """Generic CRUD operations that work across backends."""

    def build_query(self, filters: dict[str, Any]) -> dict[str, Any]:
        """
        Build a query from filters.

        Args:
            filters: Filter parameters (e.g., {"email": "test@example.com", "age__gte": 18})

        Returns:
            Query dict
        """
        query = {}

        for key, value in filters.items():
            if "__" in key:
                # Handle operators like age__gte, name__contains
                field, operator = key.split("__", 1)
                if field not in query:
                    query[field] = {}

                # Map operators to MongoDB/ES style
                op_mapping = {
                    "gte": "$gte",
                    "gt": "$gt",
                    "lte": "$lte",
                    "lt": "$lt",
                    "ne": "$ne",
                    "contains": "$contains",
                }

                if operator in op_mapping:
                    if isinstance(query[field], dict):
                        query[field][op_mapping[operator]] = value
                    else:
                        query[field] = {op_mapping[operator]: value}
                else:
                    # Keep original format for backend-specific handling
                    query[key] = value
            else:
                # Simple equality
                query[key] = value

        return query

    def validate_data(self, data: dict[str, Any], schema: dict[str, Any]) -> None:
        """
        Validate data against a JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema

        Raises:
            jsonschema.ValidationError: If data is invalid
        """
        jsonschema.validate(instance=data, schema=schema)

    def prepare_for_backend(
        self, data: dict[str, Any], backend: str
    ) -> dict[str, Any]:
        """
        Prepare data for a specific backend.

        Args:
            data: Raw data
            backend: Backend name (postgres, elasticsearch, redis, faiss)

        Returns:
            Backend-specific data
        """
        # For now, just pass through
        # In production, this might do type conversions, field mapping, etc.
        prepared = data.copy()

        if backend == "postgres":
            # Could convert types, handle NULL vs None, etc.
            pass
        elif backend == "elasticsearch":
            # Could handle nested objects differently
            pass
        elif backend == "redis":
            # Might serialize to JSON string
            pass
        elif backend == "faiss":
            # Might extract embeddings
            pass

        return prepared

    async def create(
        self, resource: str, data: dict[str, Any], backend: str, data_layer=None
    ) -> dict[str, Any]:
        """
        Create a new record.

        Args:
            resource: Resource name
            data: Record data
            backend: Backend to use
            data_layer: Optional DataLayer instance

        Returns:
            Created record with generated ID
        """
        # TODO: Use data_layer to actually create the record
        # For now, stub implementation
        return {**data, "_created": True}

    async def read(
        self,
        resource: str,
        record_id: str,
        backend: str,
        data_layer=None,
    ) -> dict[str, Any] | None:
        """
        Read a record by ID.

        Args:
            resource: Resource name
            record_id: Record ID
            backend: Backend to use
            data_layer: Optional DataLayer instance

        Returns:
            Record data or None if not found
        """
        # TODO: Use data_layer to fetch the record
        return None

    async def update(
        self,
        resource: str,
        record_id: str,
        data: dict[str, Any],
        backend: str,
        data_layer=None,
    ) -> dict[str, Any]:
        """
        Update a record.

        Args:
            resource: Resource name
            record_id: Record ID
            data: Updated data
            backend: Backend to use
            data_layer: Optional DataLayer instance

        Returns:
            Updated record
        """
        # TODO: Use data_layer to update the record
        return {**data, "id": record_id, "_updated": True}

    async def delete(
        self,
        resource: str,
        record_id: str,
        backend: str,
        data_layer=None,
    ) -> bool:
        """
        Delete a record.

        Args:
            resource: Resource name
            record_id: Record ID
            backend: Backend to use
            data_layer: Optional DataLayer instance

        Returns:
            True if deleted
        """
        # TODO: Use data_layer to delete the record
        return True

    async def list(
        self,
        resource: str,
        backend: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        data_layer=None,
    ) -> list[dict[str, Any]]:
        """
        List records with optional filtering.

        Args:
            resource: Resource name
            backend: Backend to use
            filters: Optional filters
            limit: Maximum records to return
            offset: Offset for pagination
            data_layer: Optional DataLayer instance

        Returns:
            List of records
        """
        # TODO: Use data_layer to query records
        return []

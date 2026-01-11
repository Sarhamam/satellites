"""Schema differ for detecting changes between schema versions."""

from typing import Any

from webapp.migration.models import MigrationOperation, MigrationOperationType


class SchemaDiffer:
    """Detects differences between schema versions."""

    def diff_postgres(
        self, old_backend: dict[str, Any], new_backend: dict[str, Any]
    ) -> list[MigrationOperation]:
        """
        Diff Postgres backend configurations.

        Args:
            old_backend: Old Postgres backend config
            new_backend: New Postgres backend config

        Returns:
            List of migration operations
        """
        operations = []
        table_name = new_backend.get("table", old_backend.get("table", "unknown"))

        # Get columns
        old_columns = old_backend.get("columns", {})
        new_columns = new_backend.get("columns", {})

        # Detect added columns
        for col_name, col_config in new_columns.items():
            if col_name not in old_columns:
                operations.append(
                    MigrationOperation(
                        type=MigrationOperationType.ADD_COLUMN,
                        table=table_name,
                        details={
                            "column": col_name,
                            "column_type": col_config.get("type"),
                            "nullable": col_config.get("nullable", True),
                        },
                    )
                )

        # Detect dropped columns
        for col_name in old_columns:
            if col_name not in new_columns:
                operations.append(
                    MigrationOperation(
                        type=MigrationOperationType.DROP_COLUMN,
                        table=table_name,
                        details={"column": col_name},
                    )
                )

        # Detect modified columns
        for col_name in set(old_columns.keys()) & set(new_columns.keys()):
            old_col = old_columns[col_name]
            new_col = new_columns[col_name]

            # Check if column definition changed
            if old_col != new_col:
                operations.append(
                    MigrationOperation(
                        type=MigrationOperationType.MODIFY_COLUMN,
                        table=table_name,
                        details={
                            "column": col_name,
                            "old_config": old_col,
                            "new_config": new_col,
                        },
                    )
                )

        # Detect index changes
        old_indexes = old_backend.get("indexes", [])
        new_indexes = new_backend.get("indexes", [])

        # Convert to comparable format
        old_index_names = {idx.get("name") for idx in old_indexes if "name" in idx}
        new_index_names = {idx.get("name") for idx in new_indexes if "name" in idx}

        # Detect added indexes
        for idx in new_indexes:
            if "name" in idx and idx["name"] not in old_index_names:
                operations.append(
                    MigrationOperation(
                        type=MigrationOperationType.CREATE_INDEX,
                        table=table_name,
                        details={
                            "name": idx["name"],
                            "columns": idx.get("columns", []),
                            "unique": idx.get("unique", False),
                            "using": idx.get("using"),
                        },
                    )
                )

        # Detect dropped indexes
        for idx in old_indexes:
            if "name" in idx and idx["name"] not in new_index_names:
                operations.append(
                    MigrationOperation(
                        type=MigrationOperationType.DROP_INDEX,
                        table=table_name,
                        details={"name": idx["name"]},
                    )
                )

        return operations

    def diff_elasticsearch(
        self, old_backend: dict[str, Any], new_backend: dict[str, Any]
    ) -> list[MigrationOperation]:
        """
        Diff Elasticsearch backend configurations.

        Args:
            old_backend: Old Elasticsearch backend config
            new_backend: New Elasticsearch backend config

        Returns:
            List of migration operations (placeholder for now)
        """
        # TODO: Implement Elasticsearch diffing
        return []

    def diff_redis(
        self, old_backend: dict[str, Any], new_backend: dict[str, Any]
    ) -> list[MigrationOperation]:
        """
        Diff Redis backend configurations.

        Args:
            old_backend: Old Redis backend config
            new_backend: New Redis backend config

        Returns:
            List of migration operations (placeholder for now)
        """
        # TODO: Implement Redis diffing
        return []

    def diff_faiss(
        self, old_backend: dict[str, Any], new_backend: dict[str, Any]
    ) -> list[MigrationOperation]:
        """
        Diff FAISS backend configurations.

        Args:
            old_backend: Old FAISS backend config
            new_backend: New FAISS backend config

        Returns:
            List of migration operations (placeholder for now)
        """
        # TODO: Implement FAISS diffing
        return []

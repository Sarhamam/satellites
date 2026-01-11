"""Migration executor for applying migrations to backends."""

from typing import Any

from webapp.migration.models import Migration, MigrationStatus


class MigrationExecutor:
    """Execute migrations against backends."""

    def validate_migration(self, migration: Migration) -> None:
        """
        Validate that a migration can be applied.

        Args:
            migration: Migration to validate

        Raises:
            ValueError: If migration is invalid
        """
        if migration.status != MigrationStatus.PENDING:
            raise ValueError(
                f"Migration {migration.id} is not pending (status: {migration.status}). "
                "Cannot apply already applied or failed migrations."
            )

        if not migration.generated_sql:
            raise ValueError(f"Migration {migration.id} has no SQL to execute")

    def dry_run(self, migration: Migration) -> dict[str, Any]:
        """
        Perform a dry run of the migration.

        Args:
            migration: Migration to dry run

        Returns:
            Dict with validation results
        """
        try:
            self.validate_migration(migration)
            return {
                "valid": True,
                "sql": migration.generated_sql,
                "operations_count": len(migration.operations),
                "backend": migration.backend,
                "resource": migration.resource,
            }
        except ValueError as e:
            return {
                "valid": False,
                "error": str(e),
            }

    def explain(self, migration: Migration) -> dict[str, Any]:
        """
        Provide a human-readable explanation of the migration.

        Args:
            migration: Migration to explain

        Returns:
            Dict with explanation
        """
        operations = []
        for op in migration.operations:
            op_info = {
                "type": op.type.value,
                "table": op.table,
                "details": op.details,
            }

            # Add human-readable description
            if op.type.value == "add_column":
                col = op.details.get("column")
                col_type = op.details.get("column_type")
                op_info["description"] = f"Add column '{col}' ({col_type}) to table '{op.table}'"
            elif op.type.value == "drop_column":
                col = op.details.get("column")
                op_info["description"] = f"Drop column '{col}' from table '{op.table}'"
            elif op.type.value == "create_index":
                name = op.details.get("name")
                cols = op.details.get("columns", [])
                op_info["description"] = f"Create index '{name}' on {cols}"
            else:
                op_info["description"] = f"{op.type.value} on table '{op.table}'"

            operations.append(op_info)

        return {
            "migration_id": migration.id,
            "resource": migration.resource,
            "backend": migration.backend,
            "operations": operations,
            "operations_count": len(operations),
            "from_version": migration.from_version_id,
            "to_version": migration.to_version_id,
        }

    async def apply(self, migration: Migration, data_layer=None) -> dict[str, Any]:
        """
        Apply the migration to the backend.

        Args:
            migration: Migration to apply
            data_layer: Optional DataLayer instance for executing SQL

        Returns:
            Dict with execution results

        Note:
            This is a stub implementation. In production, this would:
            - Connect to the actual backend
            - Execute the SQL/commands
            - Handle errors and rollback
            - Update migration status
        """
        self.validate_migration(migration)

        # TODO: Implement actual execution
        # For now, just return success
        return {
            "success": True,
            "migration_id": migration.id,
            "message": "Migration would be applied (stub implementation)",
        }

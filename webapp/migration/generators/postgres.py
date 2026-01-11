"""Postgres migration SQL generator."""

from typing import Any

from webapp.migration.models import MigrationOperation, MigrationOperationType


class PostgresGenerator:
    """Generate Postgres SQL from migration operations."""

    def _map_type(self, json_type: str) -> str:
        """Map JSON type to Postgres type."""
        type_mapping = {
            "text": "TEXT",
            "string": "TEXT",
            "integer": "INTEGER",
            "number": "NUMERIC",
            "numeric": "NUMERIC",
            "boolean": "BOOLEAN",
            "timestamp": "TIMESTAMP",
            "date": "DATE",
            "json": "JSONB",
            "jsonb": "JSONB",
        }
        return type_mapping.get(json_type.lower(), "TEXT")

    def generate_create_table_sql(self, operation: MigrationOperation) -> str:
        """Generate CREATE TABLE SQL."""
        table = operation.table
        details = operation.details

        columns = details.get("columns", {})
        primary_key = details.get("primary_key", [])

        # Build column definitions
        col_defs = []
        for col_name, col_config in columns.items():
            col_type = self._map_type(col_config.get("type", "text"))
            nullable = col_config.get("nullable", True)
            not_null = " NOT NULL" if not nullable else ""
            col_defs.append(f"  {col_name} {col_type}{not_null}")

        # Add primary key constraint
        if primary_key:
            pk_cols = ", ".join(primary_key)
            col_defs.append(f"  PRIMARY KEY ({pk_cols})")

        columns_sql = ",\n".join(col_defs)
        return f"CREATE TABLE {table} (\n{columns_sql}\n);"

    def generate_add_column_sql(self, operation: MigrationOperation) -> str:
        """Generate ADD COLUMN SQL."""
        table = operation.table
        details = operation.details

        column = details["column"]
        column_type = self._map_type(details.get("column_type", "text"))
        nullable = details.get("nullable", True)
        not_null = " NOT NULL" if not nullable else ""

        return f"ALTER TABLE {table} ADD COLUMN {column} {column_type}{not_null};"

    def generate_drop_column_sql(self, operation: MigrationOperation) -> str:
        """Generate DROP COLUMN SQL."""
        table = operation.table
        column = operation.details["column"]
        return f"ALTER TABLE {table} DROP COLUMN {column};"

    def generate_modify_column_sql(self, operation: MigrationOperation) -> str:
        """Generate ALTER COLUMN SQL."""
        table = operation.table
        details = operation.details
        column = details["column"]

        old_config = details.get("old_config", {})
        new_config = details.get("new_config", {})

        statements = []

        # Check if nullable changed
        old_nullable = old_config.get("nullable", True)
        new_nullable = new_config.get("nullable", True)

        if old_nullable != new_nullable:
            if new_nullable:
                statements.append(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL;")
            else:
                statements.append(f"ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL;")

        # Check if type changed
        old_type = old_config.get("type")
        new_type = new_config.get("type")

        if old_type != new_type:
            pg_type = self._map_type(new_type)
            statements.append(
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {pg_type};"
            )

        return "\n".join(statements)

    def generate_create_index_sql(self, operation: MigrationOperation) -> str:
        """Generate CREATE INDEX SQL."""
        table = operation.table
        details = operation.details

        name = details["name"]
        columns = details.get("columns", [])
        unique = details.get("unique", False)
        using = details.get("using")

        unique_clause = "UNIQUE " if unique else ""
        columns_str = ", ".join(columns)
        using_clause = f" USING {using}" if using else ""

        return f"CREATE {unique_clause}INDEX {name} ON {table}{using_clause} ({columns_str});"

    def generate_drop_index_sql(self, operation: MigrationOperation) -> str:
        """Generate DROP INDEX SQL."""
        name = operation.details["name"]
        return f"DROP INDEX {name};"

    def generate_operation_sql(self, operation: MigrationOperation) -> str:
        """
        Generate SQL for a single operation.

        Args:
            operation: Migration operation

        Returns:
            SQL statement
        """
        if operation.type == MigrationOperationType.CREATE_TABLE:
            return self.generate_create_table_sql(operation)
        elif operation.type == MigrationOperationType.ADD_COLUMN:
            return self.generate_add_column_sql(operation)
        elif operation.type == MigrationOperationType.DROP_COLUMN:
            return self.generate_drop_column_sql(operation)
        elif operation.type == MigrationOperationType.MODIFY_COLUMN:
            return self.generate_modify_column_sql(operation)
        elif operation.type == MigrationOperationType.CREATE_INDEX:
            return self.generate_create_index_sql(operation)
        elif operation.type == MigrationOperationType.DROP_INDEX:
            return self.generate_drop_index_sql(operation)
        else:
            raise ValueError(f"Unsupported operation type: {operation.type}")

    def generate_migration_sql(self, operations: list[MigrationOperation]) -> str:
        """
        Generate complete migration SQL from operations.

        Args:
            operations: List of migration operations

        Returns:
            Complete SQL migration script
        """
        statements = []
        for operation in operations:
            sql = self.generate_operation_sql(operation)
            statements.append(sql)

        return "\n\n".join(statements)

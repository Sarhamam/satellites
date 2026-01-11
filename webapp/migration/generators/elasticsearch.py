"""Elasticsearch migration command generator."""

import json
from typing import Any

from webapp.migration.models import MigrationOperation, MigrationOperationType


class ElasticsearchGenerator:
    """Generate Elasticsearch API commands from migration operations."""

    def generate_create_index_command(self, operation: MigrationOperation) -> dict[str, Any]:
        """Generate create index command."""
        details = operation.details
        index = details.get("index", operation.table)
        mappings = details.get("mappings", {})

        return {
            "action": "create_index",
            "index": index,
            "body": {
                "mappings": mappings,
            },
        }

    def generate_update_mapping_command(self, operation: MigrationOperation) -> dict[str, Any]:
        """Generate update mapping command (add field)."""
        details = operation.details
        index = details.get("index", operation.table)
        field = details.get("field")
        mapping = details.get("mapping", {})

        return {
            "action": "update_mapping",
            "index": index,
            "body": {
                "properties": {
                    field: mapping,
                }
            },
        }

    def generate_update_settings_command(self, operation: MigrationOperation) -> dict[str, Any]:
        """Generate update settings command."""
        details = operation.details
        index = details.get("index", operation.table)
        settings = details.get("settings", {})

        return {
            "action": "update_settings",
            "index": index,
            "body": settings,
        }

    def generate_operation_command(self, operation: MigrationOperation) -> dict[str, Any]:
        """
        Generate Elasticsearch command for a single operation.

        Args:
            operation: Migration operation

        Returns:
            Command dict with action, index, and body
        """
        if operation.type == MigrationOperationType.CREATE_TABLE:
            return self.generate_create_index_command(operation)
        elif operation.type == MigrationOperationType.ADD_COLUMN:
            return self.generate_update_mapping_command(operation)
        elif operation.type == MigrationOperationType.MODIFY_COLUMN:
            # For ES, this typically means updating settings
            return self.generate_update_settings_command(operation)
        else:
            raise ValueError(f"Unsupported operation type for Elasticsearch: {operation.type}")

    def generate_migration_script(self, operations: list[MigrationOperation]) -> str:
        """
        Generate complete migration script as JSON.

        Args:
            operations: List of migration operations

        Returns:
            JSON string with array of commands
        """
        commands = []
        for operation in operations:
            command = self.generate_operation_command(operation)
            commands.append(command)

        return json.dumps(commands, indent=2)

    def generate_python_api(self, operations: list[MigrationOperation]) -> str:
        """
        Generate Python API calls using elasticsearch-py.

        Args:
            operations: List of migration operations

        Returns:
            Python code as string
        """
        lines = [
            "# Elasticsearch migration script",
            "# Run with: python migration_script.py",
            "",
            "from elasticsearch import Elasticsearch",
            "",
            "es = Elasticsearch(['http://localhost:9200'])",
            "",
        ]

        for i, operation in enumerate(operations, 1):
            command = self.generate_operation_command(operation)
            action = command["action"]
            index = command["index"]
            body = command.get("body", {})

            lines.append(f"# Operation {i}: {action}")

            if action == "create_index":
                lines.append(f"es.indices.create(index='{index}', body={json.dumps(body, indent=2)})")
            elif action == "update_mapping":
                lines.append(f"es.indices.put_mapping(index='{index}', body={json.dumps(body, indent=2)})")
            elif action == "update_settings":
                lines.append(f"es.indices.put_settings(index='{index}', body={json.dumps(body, indent=2)})")

            lines.append("")

        return "\n".join(lines)

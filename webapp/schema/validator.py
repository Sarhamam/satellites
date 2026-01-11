"""JSON Schema validator and backend configuration validator."""

from typing import Any
import jsonschema
from jsonschema import Draft7Validator
from jsonschema.exceptions import (
    ValidationError as JsonSchemaValidationError,
    SchemaError,
)


class SchemaValidator:
    """Validator for JSON schemas and backend configurations."""

    def validate_schema(self, schema: dict[str, Any]) -> None:
        """
        Validate that the schema is a valid JSON Schema (Draft 7).

        Args:
            schema: The JSON Schema to validate

        Raises:
            ValueError: If the schema is invalid
        """
        try:
            # Check that the schema itself is valid
            Draft7Validator.check_schema(schema)
        except (JsonSchemaValidationError, SchemaError) as e:
            raise ValueError(f"Invalid JSON Schema: {e.message}")

    def validate_data(self, data: Any, schema: dict[str, Any]) -> None:
        """
        Validate data against a JSON Schema.

        Args:
            data: The data to validate
            schema: The JSON Schema to validate against

        Raises:
            jsonschema.exceptions.ValidationError: If data doesn't match schema
        """
        jsonschema.validate(instance=data, schema=schema)

    def validate_postgres_backend(self, config: dict[str, Any]) -> None:
        """
        Validate Postgres backend configuration.

        Args:
            config: Postgres backend configuration

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["table", "primary_key", "columns"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(config["primary_key"], list):
            raise ValueError("primary_key must be a list")

        if not isinstance(config["columns"], dict):
            raise ValueError("columns must be a dictionary")

    def validate_elasticsearch_backend(self, config: dict[str, Any]) -> None:
        """
        Validate Elasticsearch backend configuration.

        Args:
            config: Elasticsearch backend configuration

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["index", "mappings"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(config["mappings"], dict):
            raise ValueError("mappings must be a dictionary")

    def validate_redis_backend(self, config: dict[str, Any]) -> None:
        """
        Validate Redis backend configuration.

        Args:
            config: Redis backend configuration

        Raises:
            ValueError: If configuration is invalid
        """
        if "pattern" not in config:
            raise ValueError("Missing required field: pattern")

    def validate_faiss_backend(self, config: dict[str, Any]) -> None:
        """
        Validate FAISS backend configuration.

        Args:
            config: FAISS backend configuration

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["namespace", "dimension", "id_field"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(config["dimension"], int) or config["dimension"] <= 0:
            raise ValueError("dimension must be a positive integer")

    def validate_consistency(
        self, schema: dict[str, Any], backends: dict[str, Any]
    ) -> None:
        """
        Validate consistency between JSON Schema and backend configurations.

        Ensures that all fields defined in backends exist in the schema.

        Args:
            schema: The JSON Schema
            backends: Backend configurations

        Raises:
            ValueError: If there are consistency issues
        """
        # Get all property names from the schema
        schema_properties = set()
        if "properties" in schema:
            schema_properties = set(schema["properties"].keys())

        # Check Postgres backend
        if "postgres" in backends:
            pg_config = backends["postgres"]
            if "columns" in pg_config:
                pg_columns = set(pg_config["columns"].keys())
                extra_columns = pg_columns - schema_properties
                if extra_columns:
                    raise ValueError(
                        f"Postgres columns {extra_columns} not defined in schema"
                    )

        # Check Elasticsearch backend
        if "elasticsearch" in backends:
            es_config = backends["elasticsearch"]
            if "mappings" in es_config and "properties" in es_config["mappings"]:
                es_fields = set(es_config["mappings"]["properties"].keys())
                extra_fields = es_fields - schema_properties
                if extra_fields:
                    raise ValueError(
                        f"Elasticsearch fields {extra_fields} not defined in schema"
                    )

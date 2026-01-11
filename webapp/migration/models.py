"""Pydantic models for migrations."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MigrationOperationType(str, Enum):
    """Types of migration operations."""

    CREATE_TABLE = "create_table"
    DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    CREATE_CONSTRAINT = "create_constraint"
    DROP_CONSTRAINT = "drop_constraint"


class MigrationOperation(BaseModel):
    """A single migration operation."""

    type: MigrationOperationType
    table: str
    details: dict[str, Any]

    class Config:
        from_attributes = True


class MigrationStatus(str, Enum):
    """Status of a migration."""

    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Migration(BaseModel):
    """A database migration."""

    id: str
    from_version_id: Optional[str] = None
    to_version_id: str
    resource: str
    backend: str
    operations: list[MigrationOperation]
    status: MigrationStatus
    created_at: datetime
    applied_at: Optional[datetime] = None
    generated_sql: Optional[str] = None
    rollback_sql: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

"""FastAPI dependency injection."""

from typing import Annotated, Optional
from fastapi import Depends, Request

from data_layer import DataLayer


async def get_data_layer(request: Request) -> DataLayer:
    """Get the DataLayer instance from app state."""
    return request.app.state.data_layer


async def get_current_tenant(request: Request) -> Optional[str]:
    """Get current tenant from request headers or session."""
    # For now, use header. Later: session/auth
    return request.headers.get("X-Tenant-ID")


DataLayerDep = Annotated[DataLayer, Depends(get_data_layer)]
TenantDep = Annotated[Optional[str], Depends(get_current_tenant)]

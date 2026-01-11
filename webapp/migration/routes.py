"""FastAPI routes for migration management."""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webapp.dependencies import DataLayerDep, TenantDep

router = APIRouter(prefix="/migrations", tags=["migrations"])


@router.get("", response_class=HTMLResponse)
async def list_migrations(
    request: Request,
    tenant: TenantDep,
    resource: Optional[str] = None,
):
    """List all migrations."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual migrations from registry
    # For now, return empty list
    migrations = []

    return templates.TemplateResponse(
        "migration/list.html",
        {
            "request": request,
            "migrations": migrations,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
            "resource_filter": resource,
        },
    )


@router.get("/{migration_id}/preview", response_class=HTMLResponse)
async def migration_preview(
    request: Request,
    migration_id: str,
    tenant: TenantDep,
):
    """Preview a migration."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual migration from registry
    raise HTTPException(status_code=404, detail="Migration not found")

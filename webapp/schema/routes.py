"""FastAPI routes for schema management."""

from typing import Optional
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webapp.dependencies import DataLayerDep, TenantDep
from webapp.schema.models import SchemaVersionCreate
from webapp.schema.registry import SchemaRegistry

router = APIRouter(prefix="/schemas", tags=["schemas"])


@router.get("", response_class=HTMLResponse)
async def list_schemas(
    request: Request,
    tenant: TenantDep,
):
    """List all schemas."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual schemas from registry
    # For now, return empty list
    schemas = []

    return templates.TemplateResponse(
        "schema/list.html",
        {
            "request": request,
            "schemas": schemas,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def create_schema_form(
    request: Request,
    tenant: TenantDep,
):
    """Show schema creation form."""
    templates: Jinja2Templates = request.app.state.templates

    return templates.TemplateResponse(
        "schema/create.html",
        {
            "request": request,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
        },
    )


@router.get("/{schema_id}", response_class=HTMLResponse)
async def schema_detail(
    request: Request,
    schema_id: str,
    tenant: TenantDep,
):
    """Show schema details."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual schema from registry
    # For now, return 404
    from fastapi import HTTPException

    raise HTTPException(status_code=404, detail="Schema not found")

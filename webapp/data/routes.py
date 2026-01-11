"""FastAPI routes for data operations."""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webapp.dependencies import DataLayerDep, TenantDep

router = APIRouter(prefix="/data", tags=["data"])


@router.get("", response_class=HTMLResponse)
async def data_explorer(
    request: Request,
    tenant: TenantDep,
    resource: Optional[str] = None,
):
    """Data explorer page."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get available resources from schema registry
    resources = []

    return templates.TemplateResponse(
        "data/explorer.html",
        {
            "request": request,
            "resources": resources,
            "selected_resource": resource,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
        },
    )


@router.get("/{resource}", response_class=HTMLResponse)
async def list_resource_data(
    request: Request,
    resource: str,
    tenant: TenantDep,
):
    """List data for a resource."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual data from backend via router
    records = []

    return templates.TemplateResponse(
        "data/list.html",
        {
            "request": request,
            "resource": resource,
            "records": records,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
        },
    )


@router.get("/{resource}/{record_id}", response_class=HTMLResponse)
async def view_record(
    request: Request,
    resource: str,
    record_id: str,
    tenant: TenantDep,
):
    """View a single record."""
    templates: Jinja2Templates = request.app.state.templates

    # TODO: Get actual record from backend
    raise HTTPException(status_code=404, detail="Record not found")

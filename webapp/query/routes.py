"""FastAPI routes for query console."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webapp.dependencies import DataLayerDep, TenantDep

router = APIRouter(prefix="/query", tags=["query"])


@router.get("", response_class=HTMLResponse)
async def query_console(
    request: Request,
    tenant: TenantDep,
):
    """Query console page."""
    templates: Jinja2Templates = request.app.state.templates

    return templates.TemplateResponse(
        "query/console.html",
        {
            "request": request,
            "tenant": tenant,
            "config": request.app.state.webapp_config,
        },
    )

"""FastAPI application factory."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data_layer import DataLayer, DataLayerConfig
from webapp.config import WebAppConfig


def create_app(
    webapp_config: WebAppConfig | None = None,
    data_layer_config: DataLayerConfig | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    webapp_config = webapp_config or WebAppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: connect data layer
        if data_layer_config:
            data_layer = DataLayer(data_layer_config)
            await data_layer.start()
            app.state.data_layer = data_layer
        yield
        # Shutdown: disconnect data layer
        if hasattr(app.state, "data_layer"):
            await app.state.data_layer.stop()

    app = FastAPI(
        title=webapp_config.title,
        lifespan=lifespan,
        debug=webapp_config.debug,
    )

    # Static files
    app.mount(
        "/static",
        StaticFiles(directory=webapp_config.static_dir),
        name="static",
    )

    # Templates
    templates = Jinja2Templates(directory=webapp_config.templates_dir)
    app.state.templates = templates
    app.state.webapp_config = webapp_config

    # Register routes
    from webapp.schema.routes import router as schema_router
    from webapp.migration.routes import router as migration_router

    app.include_router(schema_router)
    app.include_router(migration_router)
    # app.include_router(data_router, prefix="/data", tags=["data"])
    # app.include_router(query_router, prefix="/query", tags=["query"])

    return app

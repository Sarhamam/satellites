#!/usr/bin/env python3
"""Development server for WebApp."""

import uvicorn

from webapp import create_app
from webapp.config import WebAppConfig

if __name__ == "__main__":
    # Create app with debug mode
    app = create_app(
        webapp_config=WebAppConfig(debug=True),
        data_layer_config=None,  # Configure as needed
    )

    # Run development server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for auto-reload during development
        log_level="info",
    )

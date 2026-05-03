"""NeuralMem Dashboard — Web UI for monitoring and managing memories.

Provides a FastAPI/Starlette-based dashboard with health checks,
memory browsing, knowledge graph visualization, and metrics display.
"""
from __future__ import annotations

from pathlib import Path

from starlette.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"


def serve_dashboard(app) -> None:
    """Mount the dashboard static files on an existing FastAPI/Starlette app.

    Parameters
    ----------
    app:
        A FastAPI or Starlette application instance.
    """
    app.mount(
        "/static",
        StaticFiles(directory=str(_STATIC_DIR)),
        name="dashboard-static",
    )


__all__ = ["serve_dashboard"]

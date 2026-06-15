"""
API middleware — CORS configuration, request logging, and global error handling.
"""

import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# --- CORS Configuration ---

ALLOWED_ORIGINS = [
    "http://localhost:8501",    # Streamlit dev
    "http://localhost:3000",    # React/Next.js dev
    "http://localhost:7860",    # Gradio dev
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:7860",
]

ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
ALLOWED_HEADERS = ["*"]
MAX_AGE = 600  # preflight cache duration in seconds


def add_cors_middleware(app: FastAPI) -> None:
    """
    Add CORS middleware to the FastAPI application.

    Allows local development origins (Streamlit, Gradio, React).
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
        max_age=MAX_AGE,
    )
    logger.info("CORS middleware configured: origins=%s", ALLOWED_ORIGINS)


# --- Request Logging Middleware ---

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming HTTP request with method, path, status, and latency.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log incoming request
        logger.info(
            "→ %s %s (client=%s)",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(
                "✗ %s %s — ERROR: %s (%.1fms)",
                request.method,
                request.url.path,
                str(e),
                elapsed,
            )
            raise

        elapsed = (time.time() - start_time) * 1000

        logger.info(
            "← %s %s — %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        # Add custom headers
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        return response


def add_logging_middleware(app: FastAPI) -> None:
    """Add request logging middleware to the FastAPI application."""
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("Request logging middleware configured")


# --- Global Error Handler ---

def add_error_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers for common error types.
    """
    from fastapi.responses import JSONResponse

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("ValueError on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid input", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s: %s",
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred. Please try again later.",
            },
        )

    logger.info("Global error handlers registered")


def configure_middleware(app: FastAPI) -> None:
    """
    Apply all middleware and handlers to the FastAPI application.

    Call this once during app startup.
    """
    add_cors_middleware(app)
    add_logging_middleware(app)
    add_error_handlers(app)
    logger.info("All middleware configured")

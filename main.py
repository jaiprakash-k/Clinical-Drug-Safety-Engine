"""
main.py — FastAPI application for the Clinical Drug Safety Engine.

Production-grade API with:
  - Strict input validation
  - Structured JSON-only responses
  - Health checks
  - CORS support
  - Graceful error handling
  - Startup/shutdown lifecycle
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engine import DrugSafetyEngine
from models import AnalyzeRequest, AnalyzeResponse, ErrorResponse

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Engine Singleton ─────────────────────────────────────────────────────────

engine = DrugSafetyEngine(
    cache_backend=os.getenv("CACHE_BACKEND", "memory"),
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", "6379")),
    redis_password=os.getenv("REDIS_PASSWORD"),
    cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup, cleanup on shutdown."""
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   Clinical Drug Safety Engine — Starting     ║")
    logger.info("╚══════════════════════════════════════════════╝")
    engine.initialize()
    yield
    logger.info("Drug Safety Engine shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Drug Safety Engine",
    description=(
        "Production-grade API for analyzing drug-drug interactions, "
        "allergy cross-reactivity, and condition contraindications. "
        "Powered by medical-specific LLMs with validated fallback data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Logging Middleware ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        "%s %s → %d (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ─── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak raw tracebacks to client."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    error = ErrorResponse(
        error="Internal server error",
        detail="An unexpected error occurred. Please try again.",
        status_code=500,
    )
    return JSONResponse(status_code=500, content=error.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error = ErrorResponse(
        error=str(exc.detail),
        status_code=exc.status_code,
    )
    return JSONResponse(status_code=exc.status_code, content=error.model_dump())


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze drug interactions and safety",
    description=(
        "Analyze a list of medicines for drug-drug interactions, "
        "allergy cross-reactivity, and condition contraindications. "
        "Returns a comprehensive safety assessment with risk scoring."
    ),
    responses={
        200: {"model": AnalyzeResponse, "description": "Successful analysis"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def analyze_drugs(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Main endpoint: Analyze drug interactions and patient safety.

    Accepts a list of medicines and optional patient history.
    Returns structured JSON with interactions, allergy alerts,
    contraindication alerts, risk scoring, and safety decisions.
    """
    try:
        response = await engine.analyze(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.get(
    "/health",
    summary="Health check",
    description="Returns the health status of the engine and its components.",
)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "engine": "initialized",
        "llm_available": engine.is_llm_available,
        "cache_stats": engine.cache_stats,
        "version": "1.0.0",
    }


@app.get(
    "/",
    summary="Root endpoint",
    description="API information and available endpoints.",
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Clinical Drug Safety Engine",
        "version": "1.0.0",
        "description": "Healthcare-grade drug interaction analysis API",
        "endpoints": {
            "POST /analyze": "Analyze drug interactions and safety",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "API documentation (ReDoc)",
        },
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=LOG_LEVEL.lower(),
    )

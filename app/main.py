"""
main.py – FastAPI application entry point.

Endpoints
---------
POST /run         – route a task through the model pipeline
GET  /metrics     – Prometheus text exposition
GET  /health      – liveness probe
GET  /admin/*     – management routes (see admin.py)
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from auth import verify_api_key
from config_loader import load_models, load_policy, watch_configs
from metrics import generate_metrics_output
from router import run_task

# ── Structured JSON logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

# ── Rate limiter ──────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─ Startup ───────────────────────────────────────────────────────────────
    try:
        models = load_models()
        policy = load_policy()
        log.info(
            "config_loaded",
            slots=list(models.keys()),
            low_risk_path=policy.get("low_risk_path"),
            high_risk_path=policy.get("high_risk_path"),
            max_attempts=policy.get("max_attempts"),
            min_score=policy.get("min_score"),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("config_load_failed", error=str(exc))

    # Background task: hot-reload YAML configs on file change
    watcher_task = asyncio.create_task(watch_configs())

    yield  # ← app is running

    # ─ Shutdown ─────────────────────────────────────────────────────────────
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass


# ── Application ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Model Router",
    description="Provider-agnostic LLM router with cost optimisation and observability.",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate-limit state + error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# OpenTelemetry instrumentation (gracefully no-ops if SDK absent)
from tracing import setup_tracing  # noqa: E402
setup_tracing(app)

# Admin routes
from admin import router as admin_router  # noqa: E402
app.include_router(admin_router)


# ── Request / Response schemas ─────────────────────────────────────────────────────

class RunRequest(BaseModel):
    task_id: str = Field(..., description="Caller-supplied unique task identifier.")
    user_input: str = Field(..., description="The prompt / task text.")
    task_type: Optional[str] = Field(None, description="Optional hint about task type.")
    risk: str = Field(
        "low",
        description="Routing risk level: 'low' uses local→cloud path; 'high' skips local.",
    )
    budget_virtual_units: int = Field(
        50, ge=0, description="Maximum virtual cost units the router may spend."
    )
    max_latency_ms: float = Field(
        4000.0, gt=0, description="Per-attempt latency ceiling in milliseconds."
    )
    require_json: bool = Field(
        False, description="When True, evaluation requires the response to be valid JSON."
    )


# ── Routes ──────────────────────────────────────────────────────────────────────

_RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")


@app.post("/run", summary="Route a task through the model pipeline")
@limiter.limit(_RATE_LIMIT)
async def run_endpoint(request: Request, req: RunRequest) -> JSONResponse:
    """
    Route *user_input* through the configured model path and return the result.
    Requires X-API-Key header when ROUTER_API_KEY env var is set.
    """
    await verify_api_key(request)
    result = await run_task(
        task_id=req.task_id,
        user_input=req.user_input,
        task_type=req.task_type,
        risk=req.risk,
        budget_virtual_units=req.budget_virtual_units,
        max_latency_ms=req.max_latency_ms,
        require_json=req.require_json,
    )
    return JSONResponse(content=result, status_code=200 if result["success"] else 207)


@app.get("/metrics", summary="Prometheus metrics exposition")
async def metrics_endpoint() -> Response:
    return Response(content=generate_metrics_output(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


# ── Global exception handler – never crash the API ────────────────────────────────

@app.exception_handler(Exception)
async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error(
        "unhandled_exception",
        path=str(request.url.path),
        error_type=type(exc).__name__,
        error=str(exc)[:200],
    )
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)[:200]},
    )


# ---------------------------------------------------------------------------
# Global exception handler – never crash the API
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)[:200]},
    )

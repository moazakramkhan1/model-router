"""
main.py – FastAPI application entry point.

Endpoints
---------
POST /run      – submit a routing task
GET  /metrics  – Prometheus text exposition
GET  /health   – liveness probe
"""
import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from config_loader import load_models, load_policy
from router import run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Model Router",
    description="Provider-agnostic LLM router with cost optimisation.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    try:
        models = load_models()
        policy = load_policy()
        logger.info("Loaded %d model slot(s): %s", len(models), list(models.keys()))
        logger.info(
            "Policy – low_risk: %s  high_risk: %s  max_attempts: %s",
            policy.get("low_risk_path"),
            policy.get("high_risk_path"),
            policy.get("max_attempts"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Config load failed on startup: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/run",
    summary="Route a task through the model pipeline",
    response_description="Routing result with per-attempt detail",
)
async def run_endpoint(req: RunRequest) -> JSONResponse:
    """
    Route *user_input* through the configured model path and return the result.
    """
    result = run_task(
        task_id=req.task_id,
        user_input=req.user_input,
        task_type=req.task_type,
        risk=req.risk,
        budget_virtual_units=req.budget_virtual_units,
        max_latency_ms=req.max_latency_ms,
        require_json=req.require_json,
    )
    status_code = 200 if result["success"] else 207  # 207 = "partial" / all failed
    return JSONResponse(content=result, status_code=status_code)


@app.get(
    "/metrics",
    summary="Prometheus metrics exposition",
    response_class=PlainTextResponse,
)
async def metrics_endpoint() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


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

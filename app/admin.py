"""
admin.py – Management/admin API routes.

All endpoints require the same X-API-Key used by /run (when configured).

Routes
------
GET  /admin/slots   – list all configured model slots with circuit-breaker state
GET  /admin/policy  – show the active routing policy
POST /admin/reload  – hot-reload config files without restarting the container
GET  /admin/breakers– show circuit-breaker statuses for all slots
"""
from fastapi import APIRouter, Depends

from auth import verify_api_key
from circuit_breaker import circuit_breaker
from config_loader import load_models, load_policy, reload_configs

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/slots", dependencies=[Depends(verify_api_key)])
async def list_slots() -> dict:
    """List every configured model slot with its current circuit-breaker state."""
    models = load_models()
    return {
        slot: {
            "provider": cfg.get("provider"),
            "model": cfg.get("model"),
            "virtual_cost_per_1k": cfg.get("virtual_cost_per_1k", 0),
            "timeout_s": cfg.get("timeout_s"),
            "circuit_breaker": circuit_breaker.status(slot),
        }
        for slot, cfg in models.items()
    }


@router.get("/policy", dependencies=[Depends(verify_api_key)])
async def get_policy() -> dict:
    """Return the active routing policy."""
    return load_policy()


@router.post("/reload", dependencies=[Depends(verify_api_key)])
async def reload_endpoint() -> dict:
    """Force reload of models.yaml and policy.yaml without a container restart."""
    reload_configs()
    return {"status": "reloaded", "slots": list(load_models().keys())}


@router.get("/breakers", dependencies=[Depends(verify_api_key)])
async def get_breakers() -> dict:
    """Show the circuit-breaker state for every slot that has been attempted."""
    models = load_models()
    return {slot: circuit_breaker.status(slot) for slot in models}

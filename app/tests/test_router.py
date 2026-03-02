"""Tests for the router core logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider_result(text="Hello world", tokens=50):
    return {"text": text, "latency_ms": 100.0, "tokens": tokens, "provider": "mock"}


def _mock_eval_ok():
    return {"ok": True, "score": 1.0, "reasons": ["response_not_empty"]}


def _mock_eval_fail():
    return {"ok": False, "score": 0.0, "reasons": ["response_not_empty:fail"]}


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_returns_cache_hit():
    cached = {
        "success": True, "final_model": "local_small", "final_text": "Cached",
        "total_virtual_cost_units": 0, "attempts": [], "cached": True,
    }
    with patch("router.get_cached", AsyncMock(return_value=cached)):
        from router import run_task
        result = await run_task("t1", "hello", None, "low", 50, 4000, False)
    assert result["cached"] is True
    assert result["success"] is True
    assert result["final_text"] == "Cached"


# ---------------------------------------------------------------------------
# Successful first attempt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_succeeds_on_first_slot():
    provider_mock = AsyncMock()
    provider_mock.generate = AsyncMock(return_value=_mock_provider_result())

    with (
        patch("router.get_cached", AsyncMock(return_value=None)),
        patch("router.set_cached", AsyncMock()),
        patch("router.get_provider", return_value=provider_mock),
        patch("router.judge", AsyncMock(return_value=_mock_eval_ok())),
        patch("router.load_models", return_value={
            "local_small": {"provider": "ollama", "virtual_cost_per_1k": 0},
        }),
        patch("router.load_policy", return_value={
            "low_risk_path": ["local_small"],
            "high_risk_path": ["local_small"],
            "max_attempts": 3,
            "min_score": 0.5,
        }),
    ):
        from router import run_task
        result = await run_task("t1", "what is 2+2?", None, "low", 50, 4000, False)

    assert result["success"] is True
    assert result["final_model"] == "local_small"
    assert len(result["attempts"]) == 1


# ---------------------------------------------------------------------------
# Fallthrough to next slot on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_falls_through_to_second_slot():
    fail_provider = AsyncMock()
    fail_provider.generate = AsyncMock(return_value=_mock_provider_result(""))
    ok_provider = AsyncMock()
    ok_provider.generate = AsyncMock(return_value=_mock_provider_result("Good answer"))

    providers = {"local_small": fail_provider, "fast_mid": ok_provider}

    with (
        patch("router.get_cached", AsyncMock(return_value=None)),
        patch("router.set_cached", AsyncMock()),
        patch("router.get_provider", side_effect=lambda slot, _cfg: providers[slot]),
        patch("router.judge", side_effect=[
            AsyncMock(return_value=_mock_eval_fail())(),
            AsyncMock(return_value=_mock_eval_ok())(),
        ]),
        patch("router.load_models", return_value={
            "local_small": {"provider": "ollama", "virtual_cost_per_1k": 0},
            "fast_mid": {"provider": "groq", "virtual_cost_per_1k": 5},
        }),
        patch("router.load_policy", return_value={
            "low_risk_path": ["local_small", "fast_mid"],
            "high_risk_path": ["fast_mid"],
            "max_attempts": 3,
            "min_score": 0.5,
        }),
    ):
        from router import run_task
        result = await run_task("t2", "explain gravity", None, "low", 50, 4000, False)

    assert result["success"] is True
    assert result["final_model"] == "fast_mid"
    assert len(result["attempts"]) == 2


# ---------------------------------------------------------------------------
# Budget gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_budget_gate():
    """Router must stop when budget is exhausted before trying next slot."""
    provider_mock = AsyncMock()
    provider_mock.generate = AsyncMock(return_value=_mock_provider_result(tokens=10000))

    with (
        patch("router.get_cached", AsyncMock(return_value=None)),
        patch("router.set_cached", AsyncMock()),
        patch("router.get_provider", return_value=provider_mock),
        patch("router.judge", AsyncMock(return_value=_mock_eval_fail())),
        patch("router.load_models", return_value={
            "local_small": {"provider": "ollama", "virtual_cost_per_1k": 10},
            "fast_mid": {"provider": "groq", "virtual_cost_per_1k": 10},
        }),
        patch("router.load_policy", return_value={
            "low_risk_path": ["local_small", "fast_mid"],
            "high_risk_path": ["fast_mid"],
            "max_attempts": 3,
            "min_score": 0.5,
        }),
    ):
        from router import run_task
        # budget of 1 unit — first attempt costs >1 → second slot skipped
        result = await run_task("t3", "hello", None, "low", 1, 4000, False)

    assert result["success"] is False


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_blocks_open_slot():
    with (
        patch("router.get_cached", AsyncMock(return_value=None)),
        patch("router.set_cached", AsyncMock()),
        patch("router.load_models", return_value={
            "local_small": {"provider": "ollama", "virtual_cost_per_1k": 0},
        }),
        patch("router.load_policy", return_value={
            "low_risk_path": ["local_small"],
            "high_risk_path": ["local_small"],
            "max_attempts": 3,
            "min_score": 0.5,
        }),
    ):
        from circuit_breaker import circuit_breaker
        from router import run_task

        # Force breaker open
        original_is_open = circuit_breaker.is_open
        circuit_breaker.is_open = lambda _slot: True
        try:
            result = await run_task("t4", "hello", None, "low", 50, 4000, False)
        finally:
            circuit_breaker.is_open = original_is_open

    assert result["success"] is False
    assert any("circuit_breaker_open" in str(a["reasons"]) for a in result["attempts"])

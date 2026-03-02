"""
metrics.py – Prometheus instrumentation for the router.

All metrics are module-level singletons so they are registered once
and shared across requests.
"""
from prometheus_client import Counter, Histogram, make_asgi_app

# ── Counters ─────────────────────────────────────────────────────────────────

router_tasks_total = Counter(
    "router_tasks_total",
    "Total routing tasks processed",
    ["success"],           # "true" | "false"
)

router_attempts_total = Counter(
    "router_attempts_total",
    "Total provider attempts",
    ["model", "ok"],       # model = slot name, ok = "true"|"false"
)

router_virtual_cost_units_total = Counter(
    "router_virtual_cost_units_total",
    "Cumulative virtual cost units consumed",
    ["model"],             # slot name
)

# ── Histograms ────────────────────────────────────────────────────────────────

router_latency_ms = Histogram(
    "router_latency_ms",
    "Provider call latency in milliseconds",
    ["model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000],
)

# ── ASGI sub-app for /metrics ─────────────────────────────────────────────────

metrics_app = make_asgi_app()

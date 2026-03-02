"""
metrics.py – Prometheus instrumentation for the router.

Supports both single-process and multi-worker (PROMETHEUS_MULTIPROC_DIR) modes.
All metric objects are module-level singletons registered once at import time.
"""
import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# ── Counters ──────────────────────────────────────────────────────────────────

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
    ["model"],
)

# ── Histograms ────────────────────────────────────────────────────────────────

router_latency_ms = Histogram(
    "router_latency_ms",
    "Provider call latency in milliseconds",
    ["model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000],
)


# ── Metrics output ────────────────────────────────────────────────────────────

def generate_metrics_output() -> bytes:
    """
    Return the Prometheus text format bytes.

    When PROMETHEUS_MULTIPROC_DIR is set (multi-worker mode), aggregates
    metrics from all worker files before returning.
    """
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip()
    if multiproc_dir:
        from prometheus_client import CollectorRegistry  # noqa: PLC0415
        from prometheus_client import multiprocess       # noqa: PLC0415
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest()

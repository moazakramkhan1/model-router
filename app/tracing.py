"""
tracing.py – OpenTelemetry distributed tracing setup.

By default traces are emitted to stdout (ConsoleSpanExporter).
Set OTEL_EXPORTER_OTLP_ENDPOINT to forward to a collector instead.
"""
import logging
import os

log = logging.getLogger(__name__)

_tracer = None


def setup_tracing(app) -> None:
    """Instrument the FastAPI app with OpenTelemetry. Gracefully skips if SDK is absent."""
    global _tracer
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import (  # type: ignore
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": "model-router", "service.version": "2.0.0"})
        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                    OTLPSpanExporter,
                )
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                )
                log.info("OTel: exporting traces to %s", otlp_endpoint)
            except ImportError:
                log.warning("OTel OTLP exporter not installed; falling back to console.")
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            # Console exporter is handy during dev; disable in prod by setting OTEL_TRACES_EXPORTER=none
            if os.environ.get("OTEL_TRACES_EXPORTER", "console").lower() != "none":
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("model-router")

        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
            FastAPIInstrumentor.instrument_app(app)
            log.info("OTel: FastAPI auto-instrumentation enabled.")
        except ImportError:
            log.debug("opentelemetry-instrumentation-fastapi not installed, skipping.")

    except ImportError:
        log.info("opentelemetry-sdk not installed; tracing disabled.")


def get_tracer():
    """Return the configured tracer (or a no-op tracer if OTel is unavailable)."""
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace  # type: ignore
        return trace.get_tracer("model-router")
    except ImportError:
        return None

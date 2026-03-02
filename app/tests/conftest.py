"""
conftest.py – pytest configuration for the model-router test suite.

Adds the app/ directory to sys.path so tests can import modules directly
(e.g. `from eval.rules import evaluate`).
"""
import os
import sys

# Insert the app/ directory at the front of sys.path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Point CONFIG_DIR at the project-level config/ directory for integration tests
os.environ.setdefault(
    "CONFIG_DIR",
    os.path.join(os.path.dirname(APP_DIR), "config"),
)
# Disable LLM judge in unit tests (no Ollama available in CI)
os.environ.setdefault("USE_LLM_JUDGE", "false")

"""
circuit_breaker.py – lightweight per-slot circuit breaker.

States
------
closed    : all attempts pass through (normal operation)
open      : attempts are blocked until reset_timeout expires
half-open : one probe attempt is allowed; success → closed, failure → open

Configuration via environment variables:
  CB_FAIL_MAX          : consecutive failures before opening (default 3)
  CB_RESET_TIMEOUT_S   : seconds before trying a probe from open state (default 30)
"""
import os
import time
from typing import Dict


class CircuitBreaker:
    def __init__(self, fail_max: int = 3, reset_timeout: float = 30.0) -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}

    def is_open(self, slot: str) -> bool:
        """Return True if requests to *slot* should be blocked right now."""
        failures = self._failures.get(slot, 0)
        if failures < self.fail_max:
            return False
        opened = self._opened_at.get(slot)
        if opened and (time.monotonic() - opened) >= self.reset_timeout:
            # half-open: let one probe through
            return False
        return True

    def record_failure(self, slot: str) -> None:
        self._failures[slot] = self._failures.get(slot, 0) + 1
        if self._failures[slot] >= self.fail_max:
            self._opened_at.setdefault(slot, time.monotonic())

    def record_success(self, slot: str) -> None:
        self._failures[slot] = 0
        self._opened_at.pop(slot, None)

    def status(self, slot: str) -> str:
        failures = self._failures.get(slot, 0)
        if failures == 0:
            return "closed"
        if self.is_open(slot):
            return "open"
        return "half-open"

    def all_statuses(self) -> Dict[str, str]:
        all_slots = set(self._failures) | set(self._opened_at)
        return {slot: self.status(slot) for slot in all_slots}


# Module-level singleton used by the router
circuit_breaker = CircuitBreaker(
    fail_max=int(os.environ.get("CB_FAIL_MAX", "3")),
    reset_timeout=float(os.environ.get("CB_RESET_TIMEOUT_S", "30")),
)

"""Tests for the rules-based evaluator."""
import pytest

from eval.rules import evaluate


def test_empty_response_fails():
    r = evaluate("")
    assert r["ok"] is False
    assert r["score"] < 1.0


def test_whitespace_only_fails():
    r = evaluate("   \n\t  ")
    assert r["ok"] is False


def test_normal_response_passes():
    r = evaluate("Paris is the capital of France.")
    assert r["ok"] is True
    assert r["score"] == 1.0


def test_error_marker_in_response_fails():
    r = evaluate("Error: connection refused to the server")
    assert r["ok"] is False


def test_exception_marker_fails():
    r = evaluate("Exception: NullPointerException at line 42")
    assert r["ok"] is False


def test_valid_json_passes_with_flag():
    r = evaluate('{"answer": 42, "unit": "km"}', require_json=True)
    assert r["ok"] is True


def test_json_in_markdown_fence_passes():
    r = evaluate('```json\n{"key": "value"}\n```', require_json=True)
    assert r["ok"] is True


def test_invalid_json_fails_with_flag():
    r = evaluate("This is plain text, not JSON.", require_json=True)
    assert r["ok"] is False


def test_score_is_fractional_on_partial_failure():
    # error marker fails one rule but empty check passes
    r = evaluate("Error: partial failure but still some content here", require_json=False)
    assert 0.0 < r["score"] < 1.0


def test_reasons_list_populated():
    r = evaluate("Hello world")
    assert isinstance(r["reasons"], list)
    assert len(r["reasons"]) > 0


def test_require_json_false_ignores_json_rule():
    r = evaluate("plain text response", require_json=False)
    # json rule should not fail since require_json=False
    assert r["ok"] is True

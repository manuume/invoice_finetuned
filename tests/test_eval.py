"""
tests/test_eval.py

Unit tests for evaluation metrics.
No model inference needed — purely tests the metric math.
"""
import pytest
from src.eval.metrics import FinancialOCREvaluator


PERFECT_PRED = {
    "store_name": "CU Convenience",
    "total_price": "12500",
    "tax_price": "1250",
    "items": [{"name": "Coffee", "price": "3500"}],
}

PERFECT_GT = {
    "store_name": "CU Convenience",
    "total_price": "12500",
    "tax_price": "1250",
    "items": [{"name": "Coffee", "price": "3500"}],
}

WRONG_PRED = {
    "store_name": "Wrong Store",
    "total_price": "99999",
    "tax_price": None,
    "items": [],
}


def test_perfect_prediction_cer_near_zero():
    ev = FinancialOCREvaluator()
    ev.add(PERFECT_PRED, PERFECT_GT, latency_ms=100.0)
    metrics = ev.compute()
    assert metrics.cer < 0.01


def test_perfect_prediction_field_accuracy_is_one():
    ev = FinancialOCREvaluator()
    ev.add(PERFECT_PRED, PERFECT_GT, latency_ms=100.0)
    metrics = ev.compute()
    assert metrics.field_accuracy == pytest.approx(1.0)


def test_wrong_prediction_field_accuracy_is_zero():
    ev = FinancialOCREvaluator()
    ev.add(WRONG_PRED, PERFECT_GT, latency_ms=100.0)
    metrics = ev.compute()
    assert metrics.field_accuracy == pytest.approx(0.0)


def test_mixed_batch():
    ev = FinancialOCREvaluator()
    ev.add(PERFECT_PRED, PERFECT_GT, latency_ms=120.0)  # all correct
    ev.add(WRONG_PRED, PERFECT_GT, latency_ms=80.0)     # all wrong
    metrics = ev.compute()
    # 3 scalar fields × 2 samples = 6 total; 3 correct, 3 wrong → 0.5
    assert metrics.field_accuracy == pytest.approx(0.5)


def test_latency_percentiles():
    ev = FinancialOCREvaluator()
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    for lat in latencies:
        ev.add(PERFECT_PRED, PERFECT_GT, latency_ms=lat)
    metrics = ev.compute()
    assert metrics.p50_ms == pytest.approx(300.0, abs=10)
    assert metrics.p95_ms >= metrics.p50_ms
    assert metrics.p99_ms >= metrics.p95_ms


def test_to_dict_has_all_keys():
    ev = FinancialOCREvaluator()
    ev.add(PERFECT_PRED, PERFECT_GT, latency_ms=100.0)
    d = ev.compute().to_dict()
    for key in ["cer", "wer", "field_accuracy", "field_f1",
                "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
        assert key in d, f"Missing key: {key}"

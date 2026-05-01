"""
src/eval/metrics.py

All evaluation metrics for the financial OCR system.

Classes:
  FinancialOCREvaluator — computes CER, WER, field-level F1 and accuracy
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jiwer import cer, wer
from sklearn.metrics import f1_score, precision_score, recall_score


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    num_samples: int = 0
    cer: float = 0.0
    wer: float = 0.0
    field_accuracy: float = 0.0        # exact match per scalar field
    field_f1: float = 0.0
    field_precision: float = 0.0
    field_recall: float = 0.0
    per_field_accuracy: dict[str, float] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "field_accuracy": round(self.field_accuracy, 4),
            "field_f1": round(self.field_f1, 4),
            "field_precision": round(self.field_precision, 4),
            "field_recall": round(self.field_recall, 4),
            "per_field_accuracy": {k: round(v, 4) for k, v in self.per_field_accuracy.items()},
            "latency_p50_ms": round(self.p50_ms, 1),
            "latency_p95_ms": round(self.p95_ms, 1),
            "latency_p99_ms": round(self.p99_ms, 1),
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

SCALAR_FIELDS = ["store_name", "total_price", "tax_price"]


def _normalise(value: Any) -> str:
    """Lowercase, strip whitespace. None → empty string."""
    if value is None:
        return ""
    return str(value).lower().strip()


def _is_missing_label(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan"}:
        return True
    return False


def scalar_field_correctness(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, bool | None]:
    """Return exact-match correctness, or None when the ground-truth label is missing."""
    return {
        field: (
            None
            if _is_missing_label(ground_truth.get(field))
            else _normalise(prediction.get(field)) == _normalise(ground_truth.get(field))
        )
        for field in SCALAR_FIELDS
    }


def _items_to_text(items: list[dict]) -> str:
    """Flatten item list to a single comparable string."""
    parts = [f"{_normalise(i.get('name'))} {_normalise(i.get('price'))}" for i in items]
    return " ".join(parts)


class FinancialOCREvaluator:
    """
    Computes all evaluation metrics from a list of (prediction, ground_truth) pairs.

    Usage:
        evaluator = FinancialOCREvaluator()
        evaluator.add(pred_dict, gt_dict, latency_ms=120.5)
        ...
        metrics = evaluator.compute()
    """

    def __init__(self) -> None:
        self._preds: list[dict] = []
        self._gts: list[dict] = []
        self._latencies: list[float] = []

    def add(
        self,
        prediction: dict[str, Any],
        ground_truth: dict[str, Any],
        latency_ms: float = 0.0,
    ) -> None:
        self._preds.append(prediction)
        self._gts.append(ground_truth)
        self._latencies.append(latency_ms)

    def compute(self) -> EvalMetrics:
        metrics = EvalMetrics(num_samples=len(self._preds), latencies_ms=self._latencies)

        # ── CER / WER on full JSON text ────────────────────────────────────
        pred_texts = [json.dumps(p, ensure_ascii=False) for p in self._preds]
        gt_texts = [json.dumps(g, ensure_ascii=False) for g in self._gts]
        metrics.cer = cer(gt_texts, pred_texts)
        metrics.wer = wer(gt_texts, pred_texts)

        # ── Per-field exact match (scalar fields) ─────────────────────────
        per_field_correct: dict[str, int] = {f: 0 for f in SCALAR_FIELDS}
        per_field_labeled: dict[str, int] = {f: 0 for f in SCALAR_FIELDS}
        all_pred_bin: list[int] = []
        all_gt_bin: list[int] = []

        for pred, gt in zip(self._preds, self._gts):
            correctness = scalar_field_correctness(pred, gt)
            for f, is_correct in correctness.items():
                if is_correct is None:
                    continue
                match = int(is_correct)
                per_field_labeled[f] += 1
                per_field_correct[f] += match
                all_pred_bin.append(match)
                all_gt_bin.append(1)   # ground truth is always "correct"

        labeled_total = sum(per_field_labeled.values())
        metrics.per_field_accuracy = {
            f: per_field_correct[f] / per_field_labeled[f]
            for f in SCALAR_FIELDS
            if per_field_labeled[f] > 0
        }
        metrics.field_accuracy = (
            sum(per_field_correct.values()) / labeled_total if labeled_total else 0.0
        )

        # ── Field-level F1 (binary: extracted correctly or not) ───────────
        if all_gt_bin:
            metrics.field_f1 = f1_score(all_gt_bin, all_pred_bin, zero_division=0)
            metrics.field_precision = precision_score(all_gt_bin, all_pred_bin, zero_division=0)
            metrics.field_recall = recall_score(all_gt_bin, all_pred_bin, zero_division=0)

        return metrics

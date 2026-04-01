"""Evaluation metrics implementing the protocol from Section 7 of the paper.

FIX M4: evaluate_extractions() now validates list lengths and warns on mismatch.
        Removed dead hit1_scores variable.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from itertools import zip_longest
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Computes all evaluation metrics described in the paper."""

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        retrieved_set = set(retrieved_ids[:k])
        if not relevant_ids:
            return 0.0
        return len(retrieved_set & relevant_ids) / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
        retrieved_set = set(retrieved_ids[:k])
        if k == 0:
            return 0.0
        return len(retrieved_set & relevant_ids) / k

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
        dcg = sum(
            (1.0 / math.log2(i + 2)) if rid in relevant_ids else 0.0
            for i, rid in enumerate(retrieved_ids[:k])
        )
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
        return dcg / ideal if ideal > 0 else 0.0

    @staticmethod
    def exact_match(predicted: str, gold: str) -> float:
        return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0

    @staticmethod
    def token_f1(predicted: str, gold: str) -> float:
        pred_tokens = Counter(predicted.lower().split())
        gold_tokens = Counter(gold.lower().split())
        common = sum((pred_tokens & gold_tokens).values())
        if common == 0:
            return 0.0
        precision = common / sum(pred_tokens.values())
        recall = common / sum(gold_tokens.values())
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def hit_at_k(predicted_codes: list[str], gold_code: str, k: int = 1) -> float:
        return 1.0 if gold_code in predicted_codes[:k] else 0.0

    @staticmethod
    def f_beta_score(precision: float, recall: float, beta: float) -> float:
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    @classmethod
    def f05_score(cls, precision: float, recall: float) -> float:
        return cls.f_beta_score(precision, recall, beta=0.5)

    @classmethod
    def f2_score(cls, precision: float, recall: float) -> float:
        return cls.f_beta_score(precision, recall, beta=2.0)

    @staticmethod
    def auroc(y_true: list[int], y_scores: list[float]) -> float:
        from sklearn.metrics import roc_auc_score

        try:
            return float(roc_auc_score(y_true, y_scores))
        except ValueError:
            return 0.0

    @staticmethod
    def expected_calibration_error(
        confidences: list[float], accuracies: list[int], n_bins: int = 10
    ) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)
        if total == 0:
            return 0.0
        for i in range(n_bins):
            mask = [(bins[i] <= c < bins[i + 1]) for c in confidences]
            bin_size = sum(mask)
            if bin_size == 0:
                continue
            bin_acc = sum(a for a, m in zip(accuracies, mask) if m) / bin_size
            bin_conf = sum(c for c, m in zip(confidences, mask) if m) / bin_size
            ece += (bin_size / total) * abs(bin_acc - bin_conf)
        return ece

    def evaluate_extractions(
        self,
        predictions: list[dict[str, Any]],
        gold_standard: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Run full evaluation suite.

        FIX M4: Validates list lengths, warns on mismatch, counts missed gold records.
        """
        if len(predictions) != len(gold_standard):
            logger.warning(
                "Prediction/gold length mismatch: %d predictions vs %d gold. "
                "Extra items will be counted as missed.",
                len(predictions),
                len(gold_standard),
            )

        em_scores: list[float] = []
        f1_scores: list[float] = []
        missed_gold = 0

        for pred, gold in zip_longest(predictions, gold_standard):
            if gold is None:
                continue
            if pred is None:
                missed_gold += 1
                continue
            for field_name in gold:
                if field_name in pred:
                    em_scores.append(self.exact_match(str(pred[field_name]), str(gold[field_name])))
                    f1_scores.append(self.token_f1(str(pred[field_name]), str(gold[field_name])))

        return {
            "exact_match_mean": float(np.mean(em_scores)) if em_scores else 0.0,
            "token_f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "n_comparisons": len(em_scores),
            "n_missed_gold": missed_gold,
        }

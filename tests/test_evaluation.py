"""Tests for evaluation metrics.

FIX M4: Added test for length-mismatch behavior.
"""
from src.evaluation.metrics import EvaluationMetrics


class TestEvaluationMetrics:
    def test_exact_match(self):
        m = EvaluationMetrics()
        assert m.exact_match("Headache", "headache") == 1.0
        assert m.exact_match("Headache", "Nausea") == 0.0

    def test_token_f1(self):
        f1 = EvaluationMetrics.token_f1("severe headache resolved", "headache resolved")
        assert 0.5 < f1 <= 1.0

    def test_recall_at_k(self):
        r = EvaluationMetrics.recall_at_k(["a", "b", "c", "d"], {"a", "c", "e"}, k=3)
        assert r == 2 / 3

    def test_ndcg(self):
        assert EvaluationMetrics.ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0

    def test_f05(self):
        assert EvaluationMetrics.f05_score(precision=0.9, recall=0.5) > 0.5

    def test_ece(self):
        ece = EvaluationMetrics.expected_calibration_error(
            confidences=[0.9, 0.9, 0.5, 0.5], accuracies=[1, 1, 0, 1]
        )
        assert 0 <= ece <= 1

    def test_evaluate_length_mismatch(self):
        """FIX M4: Verify graceful handling when prediction/gold lengths differ."""
        m = EvaluationMetrics()
        result = m.evaluate_extractions(
            predictions=[{"AETERM": "Headache"}],
            gold_standard=[{"AETERM": "Headache"}, {"AETERM": "Nausea"}],
        )
        assert result["n_missed_gold"] == 1
        assert result["n_comparisons"] == 1

"""Confidence estimator — combines token entropy, self-consistency, and NLI support.

FIX H1: All three signals are now computed and wired into the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import ConfidenceConfig
from src.schemas.extraction import ConfidenceLevel


@dataclass
class ConfidenceSignals:
    """Raw confidence signals before aggregation."""

    token_entropy: float | None = None
    self_consistency_ratio: float | None = None
    evidence_support_score: float | None = None
    n_consistency_samples: int = 0


@dataclass
class ConfidenceScore:
    """Aggregated confidence score with component breakdown."""

    composite_score: float
    level: ConfidenceLevel
    signals: ConfidenceSignals
    is_calibrated: bool = False


class ConfidenceEstimator:
    """Computes composite confidence from multiple signals."""

    def __init__(self, config: ConfidenceConfig | None = None):
        self.config = config or ConfidenceConfig()

    def estimate(self, signals: ConfidenceSignals) -> ConfidenceScore:
        """Compute weighted composite confidence score."""
        components: list[tuple[float, float]] = []

        if signals.token_entropy is not None:
            entropy_conf = max(0.0, 1.0 - signals.token_entropy / 5.0)
            components.append((entropy_conf, self.config.token_entropy_weight))

        if signals.self_consistency_ratio is not None:
            components.append((signals.self_consistency_ratio, self.config.self_consistency_weight))

        if signals.evidence_support_score is not None:
            components.append((signals.evidence_support_score, self.config.evidence_support_weight))

        if not components:
            return ConfidenceScore(
                composite_score=0.0,
                level=ConfidenceLevel.UNABLE_TO_DETERMINE,
                signals=signals,
            )

        total_weight = sum(w for _, w in components)
        composite = sum(score * weight for score, weight in components) / total_weight

        if composite >= 0.80:
            level = ConfidenceLevel.HIGH
        elif composite >= 0.50:
            level = ConfidenceLevel.MEDIUM
        elif composite >= 0.20:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNABLE_TO_DETERMINE

        return ConfidenceScore(
            composite_score=round(composite, 4),
            level=level,
            signals=signals,
        )

    @staticmethod
    def compute_self_consistency(values: list[str]) -> float:
        """Compute fraction of samples agreeing on most common value."""
        if not values:
            return 0.0
        from collections import Counter

        counts = Counter(v.strip().lower() for v in values)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(values)

    @staticmethod
    def compute_token_entropy(logprobs: list[dict] | None) -> float | None:
        """Compute mean token-level entropy from logprobs."""
        if not logprobs:
            return None
        entropies: list[float] = []
        for token_info in logprobs:
            if isinstance(token_info, dict) and "top_logprobs" in token_info:
                probs = [math.exp(lp["logprob"]) for lp in token_info["top_logprobs"]]
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    entropies.append(entropy)
        return sum(entropies) / len(entropies) if entropies else None

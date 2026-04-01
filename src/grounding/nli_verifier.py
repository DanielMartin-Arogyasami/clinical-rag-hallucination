"""NLI-based evidence verifier — classifies support level for extractions.

FIX C2: Uses AutoModelForSequenceClassification + AutoTokenizer for proper
        NLI inference instead of the zero-shot-classification pipeline.
        DeBERTa-v3-large-mnli labels: 0=contradiction, 1=neutral, 2=entailment.

FIX L1: Removed dead NLI_LABEL_MAP constant.
FIX M7: Added device configuration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from src.config import GroundingConfig
from src.schemas.extraction import SupportLevel

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    support_level: SupportLevel
    nli_score: float
    entailment_score: float
    contradiction_score: float
    reasoning: str | None = None


class NLIVerifier:
    """Verifies extraction support using proper NLI inference.

    FIX C2: The old implementation used zero-shot-classification, which
    classifies text against topic labels — NOT NLI. This version uses
    the model as an actual NLI classifier with premise/hypothesis pairs.
    """

    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: torch.device | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        logger.info("Loading NLI model: %s (device=%s)", self.config.nli_model, self.config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.config.nli_model)
        self._device = torch.device(self.config.device)
        self._model.to(self._device)
        self._model.eval()

    def verify(self, premise: str, hypothesis: str) -> VerificationResult:
        """Verify if premise (source span) entails hypothesis (extraction claim).

        FIX C2: Proper NLI inference — encodes premise + hypothesis as a pair,
        runs through the model, and reads entailment/neutral/contradiction logits.

        Args:
            premise: The source text span cited as evidence.
            hypothesis: The claim (e.g., "The adverse event is headache").
        """
        self._load_model()

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        contradiction_score = probs[0].item()
        _ = probs[1].item()  # neutral
        entailment_score = probs[2].item()

        # Determine support level using configured thresholds
        if entailment_score >= self.config.fully_supported_threshold:
            level = SupportLevel.FULLY_SUPPORTED
        elif contradiction_score > entailment_score and contradiction_score > 0.5:
            level = SupportLevel.CONTRADICTED
        elif entailment_score >= self.config.partially_supported_threshold:
            level = SupportLevel.PARTIALLY_SUPPORTED
        else:
            level = SupportLevel.UNSUPPORTED

        return VerificationResult(
            support_level=level,
            nli_score=entailment_score,
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
        )

    def verify_extraction(
        self,
        source_span: str,
        field_name: str,
        extracted_value: str,
    ) -> VerificationResult:
        """Convenience method — builds hypothesis from field + value."""
        hypothesis = f"The {field_name} is {extracted_value}"
        return self.verify(premise=source_span, hypothesis=hypothesis)

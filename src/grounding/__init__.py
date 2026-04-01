"""Stage 4: Evidence grounding and NLI-based verification."""
from src.grounding.nli_verifier import NLIVerifier
from src.grounding.nli_verifier import VerificationResult
from src.grounding.span_finder import SpanFinder

__all__ = ["NLIVerifier", "VerificationResult", "SpanFinder"]

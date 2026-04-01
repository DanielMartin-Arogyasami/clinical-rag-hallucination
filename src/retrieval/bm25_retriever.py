"""BM25 sparse retrieval with clinical-aware tokenization.

FIX C1: Uses re.sub with callback instead of iterative string mutation,
        eliminating the offset corruption bug.
FIX M5: Narrowed regex to clinical-specific patterns only, excluding
        generic hyphenated words like "well-known" or "follow-up".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import BM25Config
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)

# FIX M5: Narrowed to clinical-specific patterns only
_CLINICAL_PATTERN = re.compile(
    r"(?:anti-[A-Za-z0-9]+)"  # anti-VEGF, anti-PD1
    r"|(?:CYP\d+[A-Za-z]\d*)"  # CYP3A4, CYP2D6
    r"|(?:\d+/\d+\s*mmHg)"  # Blood pressure 120/80 mmHg
    r"|(?:\d+\.?\d*\s*mg/[dkm]?[Ll])"  # Dosages 5 mg/dL
    r"|(?:[A-Z]{2,6}-\d{1,7})"  # Code-style: LOINC-12345
    r"|(?:MedDRA|LOINC|SNOMED|SDTM|CDISC)"  # Standard names
    r"|(?:(?:non|pre|post|intra|extra|sub)-[A-Za-z]+)"  # Clinical prefixes
)


@dataclass
class BM25Result:
    chunk: DocumentChunk
    score: float


def clinical_tokenize(text: str) -> list[str]:
    """Tokenize text preserving clinical entity boundaries.

    FIX C1: Uses re.sub with callback — no offset corruption.
    The old approach mutated the string during iteration, causing
    stale match positions after the first replacement.
    """
    protected: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        # Keys must match tokenization: tokens are taken from safe_text.lower()
        key = f"__cl{len(protected)}__"
        protected[key] = match.group().lower()
        return key

    safe_text = _CLINICAL_PATTERN.sub(_replace, text)
    tokens = re.findall(r"[a-zA-Z0-9_]+", safe_text.lower())
    return [protected.get(t, t) for t in tokens]


class BM25Retriever:
    def __init__(self, config: BM25Config | None = None):
        self.config = config or BM25Config()
        self._index: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks
        corpus = [clinical_tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(corpus, k1=self.config.k1, b=self.config.b)
        logger.info("BM25 indexed %d chunks", len(chunks))

    def retrieve(self, query: str, top_k: int | None = None) -> list[BM25Result]:
        if not self._index:
            raise RuntimeError("Call index() first")
        top_k = top_k or self.config.top_k
        scores = self._index.get_scores(clinical_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [BM25Result(chunk=self._chunks[i], score=s) for i, s in ranked if s > 0]

    @property
    def is_indexed(self) -> bool:
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

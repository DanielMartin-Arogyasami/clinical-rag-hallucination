"""Cross-encoder reranker — fine-grained relevance scoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import RerankerConfig
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    chunk: DocumentChunk
    score: float
    original_rank: int


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", self.config.model)
        self._model = CrossEncoder(self.config.model)

    def rerank(
        self, query: str, chunks: list[DocumentChunk], top_k: int | None = None
    ) -> list[RerankResult]:
        self._load_model()
        top_k = top_k or self.config.top_k_output
        chunks = chunks[: self.config.top_k_input]
        pairs = [[query, c.text] for c in chunks]
        scores = self._model.predict(pairs, batch_size=self.config.batch_size)
        ranked = sorted(enumerate(zip(chunks, scores)), key=lambda x: x[1][1], reverse=True)
        return [
            RerankResult(chunk=chunk, score=float(score), original_rank=idx)
            for idx, (chunk, score) in ranked[:top_k]
        ]

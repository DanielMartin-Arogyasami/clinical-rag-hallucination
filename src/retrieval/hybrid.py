"""Hybrid retrieval — fuses BM25 + Dense via Reciprocal Rank Fusion, then reranks.

FIX M6: Extracts key sentences from the document for better retrieval queries
        instead of using crude first-2000-char truncation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.config import RetrievalConfig
from src.ingestion.chunker import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    chunk: DocumentChunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


def build_retrieval_query(full_text: str, max_chars: int = 2000) -> str:
    """Build a retrieval query from document text.

    FIX M6: Instead of blindly truncating to first 2000 chars (which may be
    headers/boilerplate), extract key clinical sentences containing entities
    like drugs, events, lab values, dates.
    """
    clinical_keywords = re.compile(
        r"(?:adverse|event|headache|nausea|laboratory|mg|dose|treatment|study drug"
        r"|serious|severity|related|resolved|recovered|aminotransferase|creatinine"
        r"|hemoglobin|platelet|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    scored = []
    for sent in sentences:
        score = len(clinical_keywords.findall(sent))
        scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)

    query_parts: list[str] = []
    total_len = 0
    for _score, sent in scored:
        if total_len + len(sent) > max_chars:
            break
        query_parts.append(sent)
        total_len += len(sent)

    if not query_parts:
        return full_text[:max_chars]
    return " ".join(query_parts)


class HybridRetriever:
    """Fuses BM25 + Dense retrieval with cross-encoder reranking."""

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.bm25 = BM25Retriever(self.config.bm25)
        self.dense = DenseRetriever(self.config.dense)
        self.reranker = CrossEncoderReranker(self.config.reranker)

    def index(self, chunks: list[DocumentChunk]) -> None:
        logger.info("Building hybrid index over %d chunks", len(chunks))
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[HybridResult]:
        top_k = top_k or self.config.reranker.top_k_output
        bm25_results = self.bm25.retrieve(query)
        dense_results = self.dense.retrieve(query)
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)
        fused_chunks = [r.chunk for r in fused[: self.config.reranker.top_k_input]]
        if fused_chunks:
            reranked = self.reranker.rerank(query, fused_chunks, top_k=top_k)
            return [
                HybridResult(chunk=r.chunk, score=r.score, rerank_score=r.score)
                for r in reranked
            ]
        return fused[:top_k]

    def _reciprocal_rank_fusion(self, bm25_results, dense_results) -> list[HybridResult]:
        k = self.config.fusion.rrf_k
        scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}
        bm25_ranks: dict[str, int] = {}
        dense_ranks: dict[str, int] = {}

        for rank, r in enumerate(bm25_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = r.chunk
            bm25_ranks[cid] = rank + 1

        for rank, r in enumerate(dense_results):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = r.chunk
            dense_ranks[cid] = rank + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            HybridResult(
                chunk=chunk_map[cid],
                score=score,
                bm25_rank=bm25_ranks.get(cid),
                dense_rank=dense_ranks.get(cid),
            )
            for cid, score in ranked
        ]

    @property
    def is_indexed(self) -> bool:
        return self.bm25.is_indexed and self.dense.is_indexed

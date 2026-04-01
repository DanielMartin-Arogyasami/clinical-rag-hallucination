"""Dense passage retrieval — PubMedBERT embeddings + FAISS ANN search.

FIX C3: Replaced pickle with JSON serialization for chunk metadata.
FIX M7: Added device configuration for GPU support.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DenseConfig
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class DenseResult:
    chunk: DocumentChunk
    score: float


class DenseRetriever:
    def __init__(self, config: DenseConfig | None = None):
        self.config = config or DenseConfig()
        self._model: Any = None
        self._index: Any = None
        self._chunks: list[DocumentChunk] = []

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s (device=%s)", self.config.model, self.config.device)
        self._model = SentenceTransformer(self.config.model, device=self.config.device)  # FIX M7

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        emb = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=64,
        )
        return np.array(emb, dtype=np.float32)

    def index(self, chunks: list[DocumentChunk]) -> None:
        import faiss

        self._chunks = chunks
        emb = self._encode([c.text for c in chunks])
        dim = emb.shape[1]
        if self.config.index_type == "IVFFlat" and len(chunks) > 100:
            nlist = min(100, len(chunks) // 10 + 1)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(emb)
            self._index.nprobe = self.config.nprobe
        else:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb)
        logger.info("FAISS indexed %d vectors (dim=%d)", len(chunks), dim)

    def retrieve(self, query: str, top_k: int | None = None) -> list[DenseResult]:
        if not self._index:
            raise RuntimeError("Call index() first")
        top_k = top_k or self.config.top_k
        q = self._encode([query])
        scores, indices = self._index.search(q, top_k)
        return [
            DenseResult(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def save(self, path: Path) -> None:
        """Persist index and chunk metadata to disk.

        FIX C3: Uses JSON instead of pickle to prevent RCE on untrusted data.
        """
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        if self._index:
            faiss.write_index(self._index, str(path / "dense.faiss"))
        chunk_data = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "metadata": c.metadata,
                "section_heading": c.section_heading,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "token_count_approx": c.token_count_approx,
            }
            for c in self._chunks
        ]
        with open(path / "dense_chunks.json", "w") as f:
            json.dump(chunk_data, f)
        logger.info("Saved dense index: %d chunks to %s", len(self._chunks), path)

    def load(self, path: Path) -> None:
        """Load index and chunk metadata from disk.

        FIX C3: Uses JSON instead of pickle — safe against RCE.
        """
        import faiss

        self._index = faiss.read_index(str(path / "dense.faiss"))
        with open(path / "dense_chunks.json") as f:
            chunk_data = json.load(f)
        self._chunks = [
            DocumentChunk(
                chunk_id=d["chunk_id"],
                document_id=d["document_id"],
                text=d["text"],
                metadata=d.get("metadata", {}),
                section_heading=d.get("section_heading"),
                start_char=d.get("start_char"),
                end_char=d.get("end_char"),
                token_count_approx=d.get("token_count_approx", 0),
            )
            for d in chunk_data
        ]
        logger.info("Loaded dense index: %d vectors, %d chunks", self._index.ntotal, len(self._chunks))

    @property
    def is_indexed(self) -> bool:
        return self._index is not None

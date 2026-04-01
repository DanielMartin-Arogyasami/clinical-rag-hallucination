"""Document chunking — section-aware, fixed, and sentence-boundary strategies.

FIX M3: Chunk IDs use uuid4 to prevent hash collisions without global mutable state.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from src.config import ChunkingConfig
from src.ingestion.parser import ParsedDocument


def _next_chunk_id(document_id: str) -> str:
    """Generate collision-free, thread-safe chunk IDs.

    FIX M3: Uses uuid4 instead of sequential counter — no global mutable state.
    """
    return f"{document_id}_{uuid.uuid4().hex[:12]}"


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    section_heading: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    token_count_approx: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _next_chunk_id(self.document_id)
        if self.token_count_approx == 0:
            self.token_count_approx = len(self.text.split())


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk(self, doc: ParsedDocument) -> list[DocumentChunk]:
        strategy = self.config.strategy
        handler = {
            "section_aware": self._section_aware,
            "fixed": self._fixed,
            "sentence": self._sentence,
        }
        if strategy not in handler:
            raise ValueError(f"Unknown strategy: {strategy}")
        return handler[strategy](doc)

    def _section_aware(self, doc: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in doc.sections:
            text = section.text.strip()
            if not text:
                continue
            if len(text.split()) <= self.config.max_chunk_tokens:
                chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        document_id=doc.document_id,
                        text=text,
                        section_heading=section.heading,
                        metadata={"source_path": doc.source_path, "format": doc.format},
                    )
                )
            else:
                chunks.extend(self._split_overlap(text, section.heading, doc))
        for i, t in enumerate(doc.tables):
            if t.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        document_id=doc.document_id,
                        text=t,
                        section_heading=f"Table {i+1}",
                        metadata={"source_path": doc.source_path, "is_table": True},
                    )
                )
        return chunks

    def _fixed(self, doc: ParsedDocument) -> list[DocumentChunk]:
        return self._split_overlap(doc.full_text, None, doc)

    def _sentence(self, doc: ParsedDocument) -> list[DocumentChunk]:
        sentences = re.split(r"(?<=[.!?])\s+", doc.full_text)
        chunks: list[DocumentChunk] = []
        current: list[str] = []
        current_tokens = 0
        for sent in sentences:
            st = len(sent.split())
            if current_tokens + st > self.config.target_chunk_tokens and current:
                chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        document_id=doc.document_id,
                        text=" ".join(current),
                        metadata={"source_path": doc.source_path},
                    )
                )
                overlap_n = max(1, int(len(current) * self.config.overlap_fraction))
                current = current[-overlap_n:]
                current_tokens = sum(len(s.split()) for s in current)
            current.append(sent)
            current_tokens += st
        if current:
            chunks.append(
                DocumentChunk(
                    chunk_id="",
                    document_id=doc.document_id,
                    text=" ".join(current),
                    metadata={"source_path": doc.source_path},
                )
            )
        return chunks

    def _split_overlap(
        self, text: str, heading: str | None, doc: ParsedDocument
    ) -> list[DocumentChunk]:
        words = text.split()
        target = self.config.target_chunk_tokens
        overlap = int(target * self.config.overlap_fraction)
        chunks: list[DocumentChunk] = []
        start = 0
        while start < len(words):
            end = min(start + target, len(words))
            chunks.append(
                DocumentChunk(
                    chunk_id="",
                    document_id=doc.document_id,
                    text=" ".join(words[start:end]),
                    section_heading=heading,
                    metadata={"source_path": doc.source_path, "format": doc.format},
                )
            )
            if end >= len(words):
                break
            start = end - overlap
        return chunks

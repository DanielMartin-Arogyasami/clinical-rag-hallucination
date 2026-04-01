"""Span finder — identifies supporting text spans in source documents."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.extraction import SourceSpan


@dataclass
class SpanMatch:
    text: str
    start: int
    end: int
    score: float


class SpanFinder:
    """Finds supporting spans in source text for extracted values."""

    def find_spans(
        self,
        source_text: str,
        extracted_value: str,
        document_id: str,
        max_spans: int = 3,
    ) -> list[SourceSpan]:
        spans: list[SpanMatch] = []

        # Strategy 1: Exact substring match
        value_lower = extracted_value.lower()
        source_lower = source_text.lower()
        start = 0
        while True:
            idx = source_lower.find(value_lower, start)
            if idx == -1:
                break
            spans.append(
                SpanMatch(
                    text=source_text[idx : idx + len(extracted_value)],
                    start=idx,
                    end=idx + len(extracted_value),
                    score=1.0,
                )
            )
            start = idx + 1

        # Strategy 2: Token overlap windowed search
        if not spans:
            value_tokens = set(re.findall(r"\w+", value_lower))
            if value_tokens:
                window_size = max(len(extracted_value) * 3, 200)
                step = 50
                for i in range(0, max(1, len(source_text) - step), step):
                    window = source_text[i : i + window_size]
                    window_tokens = set(re.findall(r"\w+", window.lower()))
                    overlap = len(value_tokens & window_tokens) / len(value_tokens)
                    if overlap >= 0.5:
                        spans.append(
                            SpanMatch(text=window.strip(), start=i, end=i + len(window), score=overlap)
                        )

        # Deduplicate overlapping spans and take top-k
        spans.sort(key=lambda s: s.score, reverse=True)
        unique_spans: list[SpanMatch] = []
        for span in spans:
            overlaps = any(abs(span.start - u.start) < 50 for u in unique_spans)
            if not overlaps:
                unique_spans.append(span)
            if len(unique_spans) >= max_spans:
                break

        return [
            SourceSpan(
                document_id=document_id,
                text=s.text,
                start_char=s.start,
                end_char=s.end,
            )
            for s in unique_spans
        ]

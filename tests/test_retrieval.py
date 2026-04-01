"""Tests for Stage 2: Retrieval components.

FIX C1: Added test verifying clinical_tokenize handles multi-entity text correctly.
"""
import pytest

from src.ingestion.chunker import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.bm25_retriever import clinical_tokenize


def _make_chunks(texts: list[str]) -> list[DocumentChunk]:
    return [DocumentChunk(chunk_id=f"c{i}", document_id="test", text=t) for i, t in enumerate(texts)]


class TestClinicalTokenize:
    def test_preserves_clinical_terms(self):
        tokens = clinical_tokenize("Patient on anti-VEGF therapy with CYP3A4 inhibitor")
        lowered = [t.lower() for t in tokens]
        assert "anti-vegf" in lowered
        assert "cyp3a4" in lowered

    def test_basic_tokenization(self):
        tokens = clinical_tokenize("The patient had a headache")
        assert "headache" in tokens

    def test_multi_entity_no_corruption(self):
        """FIX C1: Verify no offset corruption with multiple clinical entities."""
        text = "anti-VEGF and CYP3A4 and anti-PD1 treatment with SNOMED codes"
        tokens = clinical_tokenize(text)
        lowered = [t.lower() for t in tokens]
        assert "anti-vegf" in lowered
        assert "cyp3a4" in lowered
        assert "anti-pd1" in lowered
        assert "snomed" in lowered
        # Should not contain corrupted fragments
        for t in lowered:
            assert "__cl" not in t, f"Placeholder leaked into tokens: {t}"

    def test_does_not_match_generic_hyphenated(self):
        """FIX M5: Generic words like 'well-known' should be split normally."""
        tokens = clinical_tokenize("a well-known follow-up state-of-the-art method")
        # These should NOT be preserved as single tokens
        assert "well" in tokens
        assert "known" in tokens


class TestBM25Retriever:
    def test_index_and_retrieve(self):
        chunks = _make_chunks([
            "MedDRA Preferred Term: Headache. SOC: Nervous system disorders.",
            "MedDRA Preferred Term: Nausea. SOC: Gastrointestinal disorders.",
            "LOINC 1742-6: Alanine aminotransferase in Serum.",
        ])
        retriever = BM25Retriever()
        retriever.index(chunks)
        assert retriever.is_indexed
        results = retriever.retrieve("headache", top_k=2)
        assert len(results) > 0
        assert "headache" in results[0].chunk.text.lower()

    def test_empty_query(self):
        chunks = _make_chunks(["Some text about nausea"])
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.retrieve("zzzznonexistent")
        assert isinstance(results, list)

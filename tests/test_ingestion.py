"""Tests for Stage 1: Document ingestion and chunking."""
import json

import pytest

from src.config import ChunkingConfig
from src.ingestion.chunker import Chunker
from src.ingestion.chunker import DocumentChunk
from src.ingestion.parser import DocumentParser
from src.ingestion.parser import ParsedDocument


@pytest.fixture
def sample_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(
        "ADVERSE EVENT REPORT\n\n"
        "Patient experienced moderate headache starting on 2024-03-15. "
        "The event was possibly related to study drug. "
        "The adverse event resolved on 2024-03-17.\n\n"
        "LABORATORY RESULTS\n\n"
        "ALT: 45 U/L (normal). AST: 62 U/L (high)."
    )
    return p


@pytest.fixture
def sample_json(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"subject_id": "001", "narrative": "Patient had headache on Day 3."}))
    return p


class TestDocumentParser:
    def test_parse_txt(self, sample_txt):
        doc = DocumentParser().parse(sample_txt)
        assert isinstance(doc, ParsedDocument)
        assert doc.format == "txt"
        assert "headache" in doc.full_text
        assert len(doc.sections) > 0

    def test_parse_json(self, sample_json):
        doc = DocumentParser().parse(sample_json)
        assert doc.format == "json"
        assert "headache" in doc.full_text

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentParser().parse(p)


class TestChunker:
    def test_section_aware(self, sample_txt):
        doc = DocumentParser().parse(sample_txt)
        chunks = Chunker(ChunkingConfig(strategy="section_aware")).chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.chunk_id for c in chunks)
        # FIX M3: Verify chunk IDs are unique
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_fixed(self, sample_txt):
        doc = DocumentParser().parse(sample_txt)
        chunks = Chunker(ChunkingConfig(strategy="fixed", target_chunk_tokens=20)).chunk(doc)
        assert len(chunks) >= 2

    def test_sentence(self, sample_txt):
        doc = DocumentParser().parse(sample_txt)
        chunks = Chunker(ChunkingConfig(strategy="sentence", target_chunk_tokens=20)).chunk(doc)
        assert len(chunks) >= 2

"""Tests for ontology interfaces.

FIX L2: Uses public properties (term_count, entry_count) instead of private _entries.
"""
from src.ontologies.loinc import LOINCOntology
from src.ontologies.meddra import MedDRAOntology


class TestMedDRA:
    def test_load_sample(self):
        ont = MedDRAOntology()
        ont.load()
        assert ont.term_count > 0  # FIX L2

    def test_lookup_headache(self):
        ont = MedDRAOntology()
        ont.load()
        results = ont.lookup("Headache")
        assert len(results) > 0
        assert results[0].term == "Headache"

    def test_validate_code(self):
        ont = MedDRAOntology()
        ont.load()
        assert ont.validate_code("10019211") is True
        assert ont.validate_code("99999999") is False

    def test_soc_lookup(self):
        ont = MedDRAOntology()
        ont.load()
        assert ont.get_soc_for_pt("10019211") == "Nervous system disorders"


class TestLOINC:
    def test_load_sample(self):
        ont = LOINCOntology()
        ont.load()
        assert ont.entry_count > 0  # FIX L2

    def test_lookup_alt(self):
        ont = LOINCOntology()
        ont.load()
        results = ont.lookup("Alanine aminotransferase")
        assert len(results) > 0
        assert results[0].code == "1742-6"

    def test_validate_code(self):
        ont = LOINCOntology()
        ont.load()
        assert ont.validate_code("1742-6") is True
        assert ont.validate_code("0000-0") is False

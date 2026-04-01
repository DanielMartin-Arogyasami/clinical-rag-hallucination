"""Tests for SDTM schemas and extraction models."""
from src.schemas.extraction import (
    ConfidenceLevel,
    ExtractionField,
    ExtractionProvenance,
    ExtractionRecord,
    SupportLevel,
    TASK_TERMINOLOGY_MAP,
    TaskType,
)
from src.schemas.sdtm import (
    AdverseEventExtractionOutput,
    AdverseEventRecord,
    Causality,
    Severity,
)


class TestAdverseEventRecord:
    def test_minimal_valid(self):
        ae = AdverseEventRecord(AETERM="Headache")
        assert ae.AETERM == "Headache"
        assert ae.AEDECOD is None

    def test_full_record(self):
        ae = AdverseEventRecord(
            AETERM="Headache",
            AEDECOD="Headache",
            AEBODSYS="Nervous system disorders",
            AESTDTC="2024-03-15",
            AESEV=Severity.MODERATE,
            AESER="N",
            AEREL=Causality.POSSIBLE,
            confidence="HIGH",
        )
        assert ae.AESEV == Severity.MODERATE

    def test_extraction_output(self):
        output = AdverseEventExtractionOutput(
            events=[AdverseEventRecord(AETERM="Headache"), AdverseEventRecord(AETERM="Nausea")]
        )
        assert len(output.events) == 2


class TestExtractionRecord:
    def test_needs_human_review(self):
        record = ExtractionRecord(
            record_id="t1",
            document_id="d1",
            task_type=TaskType.ADVERSE_EVENT_CODING,
            fields=[ExtractionField(field_name="AETERM", value="Headache", confidence=ConfidenceLevel.LOW)],
        )
        assert record.needs_human_review is True

    def test_auto_accept(self):
        record = ExtractionRecord(
            record_id="t2",
            document_id="d1",
            task_type=TaskType.ADVERSE_EVENT_CODING,
            fields=[
                ExtractionField(
                    field_name="AETERM",
                    value="Headache",
                    confidence=ConfidenceLevel.HIGH,
                    provenance=ExtractionProvenance(support_level=SupportLevel.FULLY_SUPPORTED),
                )
            ],
        )
        assert record.needs_human_review is False


class TestTaskTerminologyMap:
    """FIX H3: Verify terminology system mappings are correct."""

    def test_ae_uses_meddra(self):
        system, fields = TASK_TERMINOLOGY_MAP[TaskType.ADVERSE_EVENT_CODING]
        assert system == "MedDRA"
        assert "AEDECOD" in fields

    def test_lab_uses_loinc(self):
        system, fields = TASK_TERMINOLOGY_MAP[TaskType.LAB_VALUE_NORMALIZATION]
        assert system == "LOINC"
        assert "LBLOINC" in fields

    def test_cm_uses_who_dde(self):
        system, fields = TASK_TERMINOLOGY_MAP[TaskType.CONCOMITANT_MEDICATION_CODING]
        assert system == "WHO-DDE"
        assert "CMDECOD" in fields

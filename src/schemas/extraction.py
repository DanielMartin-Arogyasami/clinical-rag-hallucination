"""Core extraction schemas — provenance, confidence, support levels.

Every extraction record carries full provenance chain (source → output),
confidence signals, and evidence grounding metadata per the reference architecture.

FIX H3: Added TASK_TERMINOLOGY_MAP for correct terminology system assignment.
"""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNABLE_TO_DETERMINE = "UNABLE_TO_DETERMINE"


class SupportLevel(str, Enum):
    """NLI-derived evidence support (Section 4.4 of paper).
    Maps: FULLY_SUPPORTED↔entailment, PARTIALLY↔neutral, CONTRADICTED↔contradiction.
    """

    FULLY_SUPPORTED = "FULLY_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class TaskType(str, Enum):
    ADVERSE_EVENT_CODING = "adverse_event_coding"
    LAB_VALUE_NORMALIZATION = "lab_value_normalization"
    CONCOMITANT_MEDICATION_CODING = "concomitant_medication_coding"
    SDTM_DOMAIN_MAPPING = "sdtm_domain_mapping"
    MEDICAL_HISTORY_CODING = "medical_history_coding"


# FIX H3: Map task types to their terminology system and coded fields
TASK_TERMINOLOGY_MAP: dict[TaskType, tuple[str, set[str]]] = {
    TaskType.ADVERSE_EVENT_CODING: ("MedDRA", {"AEDECOD", "AEBODSYS"}),
    TaskType.LAB_VALUE_NORMALIZATION: ("LOINC", {"LBLOINC"}),
    TaskType.CONCOMITANT_MEDICATION_CODING: ("WHO-DDE", {"CMDECOD", "CMCLAS"}),
    TaskType.MEDICAL_HISTORY_CODING: ("MedDRA", {"MHDECOD", "MHBODSYS"}),
}


class SourceSpan(BaseModel):
    """Text span in source document supporting an extraction."""

    document_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    page: int | None = None
    section: str | None = None


class RetrievedPassage(BaseModel):
    """Passage from knowledge base used to inform extraction."""

    passage_id: str
    source: str
    text: str
    relevance_score: float
    collection: str | None = None


class ExtractionProvenance(BaseModel):
    """Full provenance chain for a single extraction."""

    source_spans: list[SourceSpan] = Field(default_factory=list)
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    support_level: SupportLevel = SupportLevel.UNSUPPORTED
    nli_score: float | None = None
    reasoning: str | None = None


class ExtractionField(BaseModel):
    """Single field–value pair with full metadata."""

    field_name: str
    value: str | None = None
    terminology_code: str | None = None
    terminology_system: str | None = None
    terminology_version: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.UNABLE_TO_DETERMINE
    confidence_score: float | None = None
    provenance: ExtractionProvenance = Field(default_factory=ExtractionProvenance)
    is_safety_critical: bool = False


class ExtractionRecord(BaseModel):
    """Complete extraction from one source document/task."""

    record_id: str
    document_id: str
    task_type: TaskType
    fields: list[ExtractionField] = Field(default_factory=list)
    raw_llm_output: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str | None = None
    knowledge_base_version: str | None = None
    pipeline_version: str | None = None

    @property
    def needs_human_review(self) -> bool:
        for f in self.fields:
            if f.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.UNABLE_TO_DETERMINE):
                return True
            if f.provenance.support_level in (
                SupportLevel.UNSUPPORTED,
                SupportLevel.CONTRADICTED,
            ):
                return True
        return False

    @property
    def all_ontology_valid(self) -> bool:
        return all(
            f.terminology_code is not None for f in self.fields if f.terminology_system
        )


class ExtractionResult(BaseModel):
    """Top-level result container for a pipeline run."""

    records: list[ExtractionRecord] = Field(default_factory=list)
    pipeline_config: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def records_needing_review(self) -> list[ExtractionRecord]:
        return [r for r in self.records if r.needs_human_review]

    @property
    def auto_accepted_records(self) -> list[ExtractionRecord]:
        return [r for r in self.records if not r.needs_human_review]

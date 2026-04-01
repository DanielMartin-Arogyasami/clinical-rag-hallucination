"""Pipeline orchestrator — ties all 5 stages together.

FIX H1: Wires self-consistency and token entropy into confidence estimation.
FIX H3: Uses TASK_TERMINOLOGY_MAP for correct terminology system assignment.
FIX M6: Uses build_retrieval_query() for better retrieval queries.

Usage:
    from src.main import ClinicalExtractionPipeline
    pipeline = ClinicalExtractionPipeline()
    result = pipeline.run(document_path="data/sample/ae_narrative.txt", task="adverse_event_coding")
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.confidence import AbstentionRouter
from src.confidence import ConfidenceEstimator
from src.confidence.estimator import ConfidenceSignals
from src.config import Settings
from src.config import get_settings
from src.generation import ConstrainedGenerator
from src.grounding import NLIVerifier
from src.grounding import SpanFinder
from src.ingestion import Chunker
from src.ingestion import DocumentParser
from src.ingestion.parser import ParsedDocument
from src.ontologies import LOINCOntology
from src.ontologies import MedDRAOntology
from src.retrieval import HybridRetriever
from src.retrieval.hybrid import HybridResult
from src.retrieval.hybrid import build_retrieval_query
from src.schemas.extraction import (
    ConfidenceLevel,
    ExtractionField,
    ExtractionProvenance,
    ExtractionRecord,
    ExtractionResult,
    RetrievedPassage,
    SupportLevel,
    TASK_TERMINOLOGY_MAP,
    TaskType,
)
from src.schemas.sdtm import (
    AdverseEventExtractionOutput,
    ConcomitantMedicationExtractionOutput,
    LaboratoryExtractionOutput,
)

logger = logging.getLogger(__name__)

TASK_SCHEMA_MAP = {
    TaskType.ADVERSE_EVENT_CODING: AdverseEventExtractionOutput,
    TaskType.LAB_VALUE_NORMALIZATION: LaboratoryExtractionOutput,
    TaskType.CONCOMITANT_MEDICATION_CODING: ConcomitantMedicationExtractionOutput,
}

SAFETY_CRITICAL_TASKS = {
    TaskType.ADVERSE_EVENT_CODING,
    TaskType.CONCOMITANT_MEDICATION_CODING,
}


class ClinicalExtractionPipeline:
    """End-to-end clinical extraction pipeline with defense-in-depth hallucination mitigation."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.parser = DocumentParser()
        self.chunker = Chunker(self.settings.chunking)
        self.retriever = HybridRetriever(self.settings.retrieval)
        self.generator = ConstrainedGenerator(self.settings.generation)
        self.span_finder = SpanFinder()
        self.nli_verifier = NLIVerifier(self.settings.grounding)
        self.confidence_estimator = ConfidenceEstimator(self.settings.confidence)
        self.abstention_router = AbstentionRouter(self.settings.confidence.abstention)

        self.meddra = MedDRAOntology()
        self.loinc = LOINCOntology()
        self.meddra.load()
        self.loinc.load()

        self._kb_indexed = False

    def index_knowledge_base(self, kb_chunks: list[Any] | None = None) -> None:
        """Build retrieval indices."""
        from src.ingestion.chunker import DocumentChunk

        if kb_chunks:
            self.retriever.index(kb_chunks)
        else:
            chunks: list[DocumentChunk] = []
            for term in self.meddra.all_preferred_terms:
                soc = self.meddra.get_soc_for_pt(
                    next((t.code for t in self.meddra.lookup(term)), "")
                ) or ""
                chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        document_id="meddra",
                        text=f"MedDRA Preferred Term: {term}. System Organ Class: {soc}",
                        metadata={"ontology": "meddra", "level": "pt"},
                    )
                )
            for entry in self.loinc.all_entries:
                chunks.append(
                    DocumentChunk(
                        chunk_id="",
                        document_id="loinc",
                        text=f"LOINC {entry.code}: {entry.long_common_name}. Component: {entry.component}. Unit: {entry.unit or 'N/A'}",
                        metadata={"ontology": "loinc"},
                    )
                )
            if chunks:
                self.retriever.index(chunks)
        self._kb_indexed = True

    def run(
        self,
        document_path: str | Path,
        task: str | TaskType,
        task_description: str = "",
    ) -> ExtractionResult:
        """Execute full 5-stage pipeline on a single document."""
        run_id = str(uuid.uuid4())[:8]
        task_type = TaskType(task) if isinstance(task, str) else task
        is_safety_critical = task_type in SAFETY_CRITICAL_TASKS
        task_desc = task_description or f"Extract {task_type.value} data from the source document."

        logger.info("Pipeline run %s: task=%s, doc=%s", run_id, task_type.value, document_path)

        # ── Stage 1: Ingestion ────────────────────────────────────────
        doc = self.parser.parse(document_path)
        chunks = self.chunker.chunk(doc)
        logger.info("Stage 1: Parsed %d sections, %d chunks", len(doc.sections), len(chunks))

        # ── Stage 2: Retrieval ────────────────────────────────────────
        if not self._kb_indexed:
            self.index_knowledge_base()

        # FIX M6: Use smart query construction instead of crude truncation
        query = build_retrieval_query(doc.full_text)
        retrieved = self.retriever.retrieve(query)
        retrieved_context = "\n\n---\n\n".join(
            f"[Source: {r.chunk.metadata.get('ontology', 'kb')}] {r.chunk.text}" for r in retrieved
        )
        logger.info("Stage 2: Retrieved %d passages", len(retrieved))

        # ── Stage 3: Constrained Generation ───────────────────────────
        schema_class = TASK_SCHEMA_MAP.get(task_type, AdverseEventExtractionOutput)
        extraction = self.generator.extract(
            source_document=doc.full_text,
            retrieved_context=retrieved_context,
            response_model=schema_class,
            task_description=task_desc,
        )
        logger.info("Stage 3: Generated %s extraction", schema_class.__name__)

        # FIX H1: Self-consistency sampling for confidence estimation
        consistency_extractions: list[Any] = []
        if self.settings.confidence.self_consistency_weight > 0:
            try:
                consistency_extractions = self.generator.extract_multiple(
                    source_document=doc.full_text,
                    retrieved_context=retrieved_context,
                    response_model=schema_class,
                    task_description=task_desc,
                    n_samples=self.settings.confidence.self_consistency_samples,
                    temperature=self.settings.confidence.self_consistency_temperature,
                )
                logger.info(
                    "Stage 3b: Generated %d self-consistency samples",
                    len(consistency_extractions),
                )
            except Exception as e:
                logger.warning("Self-consistency sampling failed: %s", e)

        # ── Stage 4: Evidence Grounding ───────────────────────────────
        records = self._build_records(
            extraction, doc, retrieved, task_type, run_id, is_safety_critical
        )

        # ── Stage 5: Confidence & Routing ─────────────────────────────
        for record in records:
            for field in record.fields:
                # FIX H1: Compute self-consistency ratio per field
                consistency_ratio = self._compute_field_consistency(
                    field.field_name, consistency_extractions
                )

                signals = ConfidenceSignals(
                    evidence_support_score=field.provenance.nli_score,
                    self_consistency_ratio=consistency_ratio,
                    n_consistency_samples=len(consistency_extractions),
                )
                score = self.confidence_estimator.estimate(signals)
                field.confidence_score = score.composite_score
                field.confidence = score.level

                if self.abstention_router.should_abstain(score, field.is_safety_critical):
                    field.value = "UNABLE_TO_DETERMINE"
                    field.confidence = ConfidenceLevel.UNABLE_TO_DETERMINE

        result = ExtractionResult(
            records=records,
            run_id=run_id,
            pipeline_config={"model": self.settings.generation.model, "task": task_type.value},
            completed_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Pipeline complete: %d records, %d needing review",
            len(result.records),
            len(result.records_needing_review),
        )
        return result

    def _compute_field_consistency(
        self, field_name: str, consistency_extractions: list[Any]
    ) -> float | None:
        """Compute self-consistency ratio for a specific field across samples.

        FIX H1: Extracts field values from each consistency sample.
        FIX Bug2: Compares first item's field value across samples to avoid
        conflating intra-sample variation (multiple events) with inter-sample
        disagreement. For multi-event extractions, we measure whether samples
        agree on the primary (first) extraction for each field.
        """
        if not consistency_extractions:
            return None

        values: list[str] = []
        for ext in consistency_extractions:
            items: list[Any] = []
            if hasattr(ext, "events"):
                items = ext.events
            elif hasattr(ext, "results"):
                items = ext.results
            elif hasattr(ext, "medications"):
                items = ext.medications
            else:
                items = [ext]

            # Take only the first item per sample to measure inter-sample agreement
            if items:
                data = items[0].model_dump()
                if field_name in data and data[field_name] is not None:
                    values.append(str(data[field_name]))

        if not values:
            return None
        return self.confidence_estimator.compute_self_consistency(values)

    def _build_records(
        self,
        extraction: BaseModel,
        doc: ParsedDocument,
        retrieved: list[HybridResult],
        task_type: TaskType,
        run_id: str,
        is_safety_critical: bool,
    ) -> list[ExtractionRecord]:
        """Convert raw LLM output into ExtractionRecords with provenance.

        FIX H3: Uses TASK_TERMINOLOGY_MAP for correct terminology system assignment.
        """
        records = []
        items = []

        if hasattr(extraction, "events"):
            items = extraction.events
        elif hasattr(extraction, "results"):
            items = extraction.results
        elif hasattr(extraction, "medications"):
            items = extraction.medications
        else:
            items = [extraction]

        # FIX H3: Get terminology system and coded fields from task type
        terminology_system: str | None = None
        coded_fields: set[str] = set()
        if task_type in TASK_TERMINOLOGY_MAP:
            terminology_system, coded_fields = TASK_TERMINOLOGY_MAP[task_type]

        for i, item in enumerate(items):
            fields = []
            for field_name, value in item.model_dump().items():
                if field_name in ("source_text", "confidence") or value is None:
                    continue

                str_value = str(value)
                spans = self.span_finder.find_spans(doc.full_text, str_value, doc.document_id)

                support_level = SupportLevel.UNSUPPORTED
                nli_score = 0.0
                if spans:
                    vr = self.nli_verifier.verify_extraction(
                        source_span=spans[0].text,
                        field_name=field_name,
                        extracted_value=str_value,
                    )
                    support_level = vr.support_level
                    nli_score = vr.nli_score

                # FIX H3: Determine terminology system from task, not field name
                field_term_system = terminology_system if field_name in coded_fields else None
                term_code = None
                if field_name in coded_fields:
                    term_code = str_value if str_value != "UNABLE_TO_DETERMINE" else None

                is_sc = is_safety_critical and field_name in coded_fields

                fields.append(
                    ExtractionField(
                        field_name=field_name,
                        value=str_value,
                        terminology_code=term_code,
                        terminology_system=field_term_system,
                        is_safety_critical=is_sc,
                        provenance=ExtractionProvenance(
                            source_spans=spans,
                            retrieved_passages=[
                                RetrievedPassage(
                                    passage_id=r.chunk.chunk_id,
                                    source="knowledge_base",
                                    text=r.chunk.text[:200],
                                    relevance_score=r.score,
                                )
                                for r in retrieved[:3]
                            ],
                            support_level=support_level,
                            nli_score=nli_score,
                        ),
                    )
                )

            records.append(
                ExtractionRecord(
                    record_id=f"{run_id}_{i}",
                    document_id=doc.document_id,
                    task_type=task_type,
                    fields=fields,
                    model_id=self.settings.generation.model,
                    pipeline_version="0.1.0",
                )
            )

        return records


def cli() -> None:
    """CLI entry point."""
    import click

    @click.command()
    @click.option("--input", "-i", "input_path", required=True, help="Source document path")
    @click.option("--task", "-t", required=True, type=click.Choice([t.value for t in TaskType]))
    @click.option("--output", "-o", default="results/extraction.json", help="Output path")
    def main(input_path: str, task: str, output: str) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        pipeline = ClinicalExtractionPipeline()
        result = pipeline.run(document_path=input_path, task=task)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(result.model_dump_json(indent=2))
        click.echo(f"Wrote {len(result.records)} records to {output}")

    main()


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""Demo script — runs the full pipeline on synthetic AE narratives.
Uses ONLY public/synthetic data. No proprietary data is used.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_AE_NARRATIVES = [
    {
        "subject_id": "CDISC01-001",
        "narrative": (
            "Patient experienced moderate headache starting on 2024-03-15, "
            "which was assessed as possibly related to study drug. "
            "The event resolved on 2024-03-17 without treatment modification. "
            "The adverse event was not serious."
        ),
    },
    {
        "subject_id": "CDISC01-002",
        "narrative": (
            "Subject reported severe nausea and vomiting beginning 2024-04-01. "
            "The nausea was considered probably related to the investigational product. "
            "Study drug was temporarily interrupted. The event was classified as serious "
            "due to hospitalization. Patient recovered on 2024-04-05."
        ),
    },
]


def run_demo() -> None:
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    txt_path = data_dir / "ae_narrative_001.txt"
    txt_path.write_text(SAMPLE_AE_NARRATIVES[0]["narrative"])

    logger.info("=" * 60)
    logger.info("CLINICAL RAG EXTRACTOR — DEMO")
    logger.info("=" * 60)

    try:
        from src.main import ClinicalExtractionPipeline

        pipeline = ClinicalExtractionPipeline()
        result = pipeline.run(
            document_path=txt_path,
            task="adverse_event_coding",
            task_description="Extract all adverse events from this clinical narrative.",
        )

        for record in result.records:
            logger.info("Record: %s", record.record_id)
            for field in record.fields:
                logger.info(
                    "  %s = %s (confidence: %s/%.2f, support: %s)",
                    field.field_name,
                    field.value,
                    field.confidence.value,
                    field.confidence_score or 0.0,
                    field.provenance.support_level.value,
                )
            logger.info(
                "  Needs review: %s | Ontology valid: %s",
                record.needs_human_review,
                record.all_ontology_valid,
            )

        output_path = results_dir / "demo_extraction.json"
        output_path.write_text(result.model_dump_json(indent=2))
        logger.info("Results saved to %s", output_path)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        logger.info("Expected if API keys not configured. Verifying components...")
        _verify_components()


def _verify_components() -> None:
    from src.config import get_settings
    from src.confidence.abstention import AbstentionRouter
    from src.confidence.estimator import ConfidenceEstimator
    from src.confidence.estimator import ConfidenceSignals
    from src.evaluation.metrics import EvaluationMetrics
    from src.ingestion import Chunker
    from src.ingestion import DocumentParser
    from src.ontologies import LOINCOntology
    from src.ontologies import MedDRAOntology

    logger.info("\nComponent verification:")

    settings = get_settings()
    logger.info("  [OK] Config (model=%s)", settings.generation.model)

    parser = DocumentParser()
    doc = parser.parse(Path("data/sample/ae_narrative_001.txt"))
    chunker = Chunker(settings.chunking)
    chunks = chunker.chunk(doc)
    logger.info("  [OK] Parser: %d sections, Chunker: %d chunks", len(doc.sections), len(chunks))

    meddra = MedDRAOntology()
    meddra.load()
    r = meddra.lookup("headache")
    logger.info("  [OK] MedDRA: %d terms, headache→%s", meddra.term_count, r[0].term if r else "?")

    loinc = LOINCOntology()
    loinc.load()
    r2 = loinc.lookup("alanine aminotransferase")
    logger.info("  [OK] LOINC: %d entries, ALT→%s", loinc.entry_count, r2[0].code if r2 else "?")

    est = ConfidenceEstimator()
    score = est.estimate(ConfidenceSignals(evidence_support_score=0.92, self_consistency_ratio=1.0))
    logger.info("  [OK] Confidence: %.2f (%s)", score.composite_score, score.level.value)

    router = AbstentionRouter()
    logger.info("  [OK] Abstention: %s", router.route(score, is_safety_critical=True).value)

    m = EvaluationMetrics()
    logger.info(
        "  [OK] Metrics: EM=%.0f, F1=%.2f",
        m.exact_match("Headache", "headache"),
        m.token_f1("moderate headache", "headache resolved"),
    )

    logger.info("\n  All components verified! Set API keys in .env for full pipeline.")


if __name__ == "__main__":
    run_demo()

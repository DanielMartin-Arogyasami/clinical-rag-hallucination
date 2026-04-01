#!/usr/bin/env python3
"""Evaluation script — compares predictions against gold standard."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    import click

    @click.command()
    @click.option("--predictions", "-p", required=True)
    @click.option("--gold", "-g", required=True)
    @click.option("--output", "-o", default="results/eval_report.json")
    def evaluate(predictions: str, gold: str, output: str) -> None:
        from src.evaluation.metrics import EvaluationMetrics

        preds = json.loads(Path(predictions).read_text())
        golds = json.loads(Path(gold).read_text())
        metrics = EvaluationMetrics()
        report = metrics.evaluate_extractions(preds, golds)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2))
        logger.info("Report: %s", json.dumps(report, indent=2))

    evaluate()


if __name__ == "__main__":
    main()

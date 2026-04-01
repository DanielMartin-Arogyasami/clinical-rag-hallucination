#!/usr/bin/env python3
"""Build retrieval indices from ontology data."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    import click

    @click.command()
    @click.option("--ontology-dir", default="data/ontologies")
    @click.option("--index-dir", default="data/indices")
    def build(ontology_dir: str, index_dir: str) -> None:
        from src.main import ClinicalExtractionPipeline

        _ = ontology_dir, index_dir
        pipeline = ClinicalExtractionPipeline()
        pipeline.index_knowledge_base()
        logger.info("Knowledge base indexed successfully")

    build()


if __name__ == "__main__":
    main()

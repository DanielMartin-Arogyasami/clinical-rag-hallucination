Topics: `rag`, `hallucination-mitigation`, `constrained-decoding`, `mimic-iv`, `nlp`

# Clinical RAG Hallucination Mitigation (clinical-rag-hallucination)

> **⚠️ INDEPENDENCE & DATA DISCLAIMER:** This reference implementation uses exclusively publicly available datasets (MIMIC-IV) and public ontologies (LOINC). It does not reflect, derive from, or use any proprietary systems, workflows, data, or trade secrets of any current or past employers. This work is an independent academic and open-source contribution.

## Overview
This repository implements a Retrieval-Augmented Generation (RAG) pipeline focused on strict factual extraction from unstructured clinical text. It uses constrained decoding (via `instructor` / `outlines`) to guarantee valid ontological outputs, eliminating structural hallucinations. This is the companion to: *"Mitigating Large Language Model Hallucinations for Clinical Data Extraction"*.

## Architecture
1. **Hybrid Retrieval:** BM25 + Dense retrieval + Cross-encoder reranking against public ontologies.
2. **Constrained Generation:** LLM generation is wrapped in a Finite State Machine (FSM) enforcing Pydantic schemas, guaranteeing the output is a valid LOINC or standard terminology code.
3. **Evidence Grounding:** A Natural Language Inference (NLI) module checks if the source text actually entails the generated code.
4. **Calibrated Abstention:** If NLI confidence is below $\tau$, the system outputs `UNABLE_TO_DETERMINE`.

## Public Dataset Acquisition
This project utilizes the **MIMIC-IV** clinical database (PhysioNet).
1. Ensure you have completed the required CITI training and signed the PhysioNet Data Use Agreement for [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/).
2. Place the de-identified discharge summaries in `data/mimic_samples/`.
3. Download the public LOINC database from [loinc.org](https://loinc.org/) and place the CSV in `data/ontologies/`.

## Quick Start
```bash
# Index the ontology
python src/retrieval/indexer.py --ontology data/ontologies/loinc.csv

# Run the constrained extraction pipeline
python src/pipeline.py --input data/mimic_samples/sample_01.txt --schema LOINC
Citation
Code snippet

@article{arogyasami2026hallucination,
  title={Mitigating Large Language Model Hallucinations for Clinical Data Extraction},
  author={Arogyasami, DanielMartin},
  year={2026},
  journal={Preprint}
}

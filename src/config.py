"""Configuration management — loads settings.yaml + environment variables.

FIX H2: Filters unknown keys before constructing Pydantic models.
FIX H5: Adds reset_settings() for test isolation and accepts injection.
FIX M7: Adds device configuration for GPU support.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _filter_fields(model_cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """Filter dict to only keys that exist as fields on the Pydantic model.

    FIX H2: Prevents crash when YAML contains keys not in the model (e.g.,
    structured_output under generation).
    """
    known = set(model_cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in known}


class ChunkingConfig(BaseModel):
    strategy: str = "section_aware"
    target_chunk_tokens: int = 256
    max_chunk_tokens: int = 512
    overlap_fraction: float = 0.15
    preserve_tables: bool = True


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 50


class DenseConfig(BaseModel):
    model: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    embedding_dim: int = 768
    top_k: int = 50
    index_type: str = "Flat"
    nprobe: int = 10
    normalize_embeddings: bool = True
    device: str = "cpu"  # FIX M7


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_k_input: int = 40
    top_k_output: int = 8
    batch_size: int = 16


class FusionConfig(BaseModel):
    method: str = "reciprocal_rank"
    rrf_k: int = 60
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


class RetrievalConfig(BaseModel):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class GenerationConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 120
    max_retries: int = 3  # FIX M2


class GroundingConfig(BaseModel):
    nli_model: str = "microsoft/deberta-v3-large-mnli"
    fully_supported_threshold: float = 0.85
    partially_supported_threshold: float = 0.50
    max_source_spans: int = 3
    device: str = "cpu"  # FIX M7


class AbstentionConfig(BaseModel):
    safety_critical_threshold: float = 0.80
    standard_threshold: float = 0.60
    calibration_method: str = "isotonic"


class ConfidenceConfig(BaseModel):
    token_entropy_weight: float = 0.2
    self_consistency_weight: float = 0.3
    self_consistency_samples: int = 5
    self_consistency_temperature: float = 0.7
    evidence_support_weight: float = 0.5
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)


class Settings(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    knowledge_base_dir: Path = Path("data/ontologies")
    index_dir: Path = Path("data/indices")
    output_dir: Path = Path("results")
    prompts_dir: Path = _PROJECT_ROOT / "config" / "prompts"


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from YAML, overlaid with environment variables.

    FIX H2: Uses _filter_fields() to avoid crashing on unknown YAML keys.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    ret = raw.get("retrieval", {})
    gen = raw.get("generation", {})
    gnd = raw.get("grounding", {})
    conf = raw.get("confidence", {})
    chk = raw.get("ingestion", {}).get("chunking", {})

    device = os.getenv("DEVICE", "cpu")

    dense_raw = {**ret.get("dense", {})}
    dense_raw["model"] = os.getenv("EMBEDDING_MODEL", dense_raw.get("model", DenseConfig().model))
    dense_raw.setdefault("device", device)

    gen_raw = {**gen}
    gen_raw["model"] = os.getenv("GENERATION_MODEL", gen_raw.get("model", "gpt-4o"))
    gen_raw["provider"] = os.getenv("LLM_PROVIDER", gen_raw.get("provider", "openai"))

    return Settings(
        retrieval=RetrievalConfig(
            bm25=BM25Config(**_filter_fields(BM25Config, ret.get("bm25", {}))),
            dense=DenseConfig(**_filter_fields(DenseConfig, dense_raw)),
            reranker=RerankerConfig(**_filter_fields(RerankerConfig, ret.get("reranker", {}))),
            fusion=FusionConfig(**_filter_fields(FusionConfig, ret.get("fusion", {}))),
        ),
        generation=GenerationConfig(**_filter_fields(GenerationConfig, gen_raw)),
        grounding=GroundingConfig(
            nli_model=os.getenv("NLI_MODEL", gnd.get("nli_model", GroundingConfig().nli_model)),
            fully_supported_threshold=gnd.get("support_thresholds", {}).get("fully_supported", 0.85),
            partially_supported_threshold=gnd.get("support_thresholds", {}).get("partially_supported", 0.50),
            max_source_spans=gnd.get("max_source_spans", 3),
            device=gnd.get("device", device),
        ),
        confidence=ConfidenceConfig(
            token_entropy_weight=conf.get("signals", {}).get("token_entropy", {}).get("weight", 0.2),
            self_consistency_weight=conf.get("signals", {}).get("self_consistency", {}).get("weight", 0.3),
            self_consistency_samples=conf.get("signals", {}).get("self_consistency", {}).get("num_samples", 5),
            self_consistency_temperature=conf.get("signals", {}).get("self_consistency", {}).get("temperature", 0.7),
            evidence_support_weight=conf.get("signals", {}).get("evidence_support", {}).get("weight", 0.5),
            abstention=AbstentionConfig(**_filter_fields(AbstentionConfig, conf.get("abstention", {}))),
        ),
        chunking=ChunkingConfig(**_filter_fields(ChunkingConfig, chk)) if chk else ChunkingConfig(),
        knowledge_base_dir=Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/ontologies")),
        index_dir=Path(os.getenv("INDEX_DIR", "data/indices")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "results")),
    )


# FIX H5: Resettable singleton with thread note
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings — use in tests for isolation.

    FIX H5: Enables test isolation without monkey-patching.
    """
    global _settings
    _settings = None

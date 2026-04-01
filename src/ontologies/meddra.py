"""MedDRA ontology interface — hierarchy traversal, PT lookup, validation.

FIX L6: Added pre-built prefix index for O(prefix_len) fuzzy lookup
        instead of O(n) linear scan over all entries.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MedDRATerm:
    code: str
    term: str
    level: str
    parent_code: str | None = None
    soc_code: str | None = None
    soc_name: str | None = None


class MedDRAOntology:
    """MedDRA ontology with hierarchy traversal and validation."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._terms: dict[str, MedDRATerm] = {}
        self._name_index: dict[str, list[MedDRATerm]] = {}
        self._prefix_index: dict[str, set[str]] = defaultdict(set)  # FIX L6

    def _build_prefix_index(self) -> None:
        """Build prefix index for O(prefix_len) fuzzy lookup.

        FIX L6: Pre-builds a map from 3-char prefixes to matching term names.
        """
        self._prefix_index.clear()
        for name in self._name_index:
            for i in range(min(3, len(name)), len(name) + 1):
                prefix = name[:i]
                self._prefix_index[prefix].add(name)

    def load(self, data_dir: str | Path | None = None) -> None:
        path = Path(data_dir) if data_dir else self.data_dir
        if path and (path / "meddra_terms.csv").exists():
            self._load_from_csv(path / "meddra_terms.csv")
        else:
            logger.warning("MedDRA data not found, loading sample terms")
            self._load_sample()
        self._build_prefix_index()

    def _load_from_csv(self, csv_path: Path) -> None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = MedDRATerm(
                    code=row["code"],
                    term=row["term"],
                    level=row["level"],
                    parent_code=row.get("parent_code"),
                    soc_code=row.get("soc_code"),
                    soc_name=row.get("soc_name"),
                )
                self._terms[term.code] = term
                key = term.term.lower()
                self._name_index.setdefault(key, []).append(term)
        logger.info("Loaded %d MedDRA terms", len(self._terms))

    def _load_sample(self) -> None:
        sample = [
            ("10019211", "Headache", "pt", "10028372", "10029205", "Nervous system disorders"),
            ("10028813", "Nausea", "pt", "10017947", "10017947", "Gastrointestinal disorders"),
            ("10047700", "Vomiting", "pt", "10017947", "10017947", "Gastrointestinal disorders"),
            ("10012735", "Diarrhoea", "pt", "10017947", "10017947", "Gastrointestinal disorders"),
            ("10016558", "Fatigue", "pt", "10018065", "10018065", "General disorders and administration site conditions"),
            ("10037660", "Pyrexia", "pt", "10018065", "10018065", "General disorders and administration site conditions"),
            ("10049816", "Dizziness", "pt", "10028372", "10029205", "Nervous system disorders"),
            ("10011224", "Cough", "pt", "10038738", "10038738", "Respiratory, thoracic and mediastinal disorders"),
            ("10037844", "Rash", "pt", "10040785", "10040785", "Skin and subcutaneous tissue disorders"),
            ("10002855", "Arthralgia", "pt", "10028395", "10028395", "Musculoskeletal and connective tissue disorders"),
            ("10003239", "Asthenia", "pt", "10018065", "10018065", "General disorders and administration site conditions"),
            ("10002034", "Anaemia", "pt", "10005329", "10005329", "Blood and lymphatic system disorders"),
            ("10020772", "Hypertension", "pt", "10047065", "10047065", "Vascular disorders"),
            ("10021097", "Hypotension", "pt", "10047065", "10047065", "Vascular disorders"),
            ("10022611", "Insomnia", "pt", "10028372", "10037175", "Psychiatric disorders"),
        ]
        for code, term, level, parent, soc_code, soc_name in sample:
            t = MedDRATerm(code=code, term=term, level=level, parent_code=parent, soc_code=soc_code, soc_name=soc_name)
            self._terms[code] = t
            self._name_index.setdefault(term.lower(), []).append(t)
        logger.info("Loaded %d sample MedDRA terms", len(self._terms))

    def lookup(self, text: str) -> list[MedDRATerm]:
        """Look up terms by name. Exact match first, then prefix-indexed fuzzy.

        FIX L6: Uses prefix index for sub-linear fuzzy search.
        """
        key = text.strip().lower()
        exact = self._name_index.get(key, [])
        if exact:
            return exact
        # Prefix-indexed fuzzy match
        candidate_names: set[str] = set()
        prefix = key[:3] if len(key) >= 3 else key
        if prefix in self._prefix_index:
            for name in self._prefix_index[prefix]:
                if key in name or name in key:
                    candidate_names.add(name)
        matches: list[MedDRATerm] = []
        for name in candidate_names:
            matches.extend(self._name_index[name])
        return matches[:10]

    def validate_code(self, code: str) -> bool:
        return code in self._terms

    def get_term(self, code: str) -> MedDRATerm | None:
        return self._terms.get(code)

    def get_soc_for_pt(self, pt_code: str) -> str | None:
        term = self._terms.get(pt_code)
        return term.soc_name if term else None

    @property
    def term_count(self) -> int:
        """FIX L2: Public accessor instead of exposing _terms."""
        return len(self._terms)

    @property
    def all_preferred_terms(self) -> list[str]:
        return [t.term for t in self._terms.values() if t.level == "pt"]

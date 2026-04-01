"""LOINC ontology interface — code lookup, validation, multi-axis search.

FIX L6: Added prefix index for faster fuzzy lookup.
FIX L2: Added public entry_count property.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LOINCEntry:
    code: str
    component: str
    long_common_name: str
    property: str | None = None
    time_aspect: str | None = None
    system: str | None = None
    scale_type: str | None = None
    method: str | None = None
    unit: str | None = None


class LOINCOntology:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._entries: dict[str, LOINCEntry] = {}
        self._name_index: dict[str, list[LOINCEntry]] = {}
        self._prefix_index: dict[str, set[str]] = defaultdict(set)  # FIX L6

    def _build_prefix_index(self) -> None:
        self._prefix_index.clear()
        for name in self._name_index:
            for i in range(min(3, len(name)), len(name) + 1):
                self._prefix_index[name[:i]].add(name)

    def load(self, data_dir: str | Path | None = None) -> None:
        path = Path(data_dir) if data_dir else self.data_dir
        if path and (path / "loinc_terms.csv").exists():
            self._load_from_csv(path / "loinc_terms.csv")
        else:
            logger.warning("LOINC data not found, loading sample")
            self._load_sample()
        self._build_prefix_index()

    def _load_from_csv(self, csv_path: Path) -> None:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                e = LOINCEntry(
                    code=row["LOINC_NUM"],
                    component=row.get("COMPONENT", ""),
                    long_common_name=row.get("LONG_COMMON_NAME", ""),
                    property=row.get("PROPERTY"),
                    system=row.get("SYSTEM"),
                    scale_type=row.get("SCALE_TYP"),
                    method=row.get("METHOD_TYP"),
                )
                self._entries[e.code] = e
                self._name_index.setdefault(e.component.lower(), []).append(e)
        logger.info("Loaded %d LOINC entries", len(self._entries))

    def _load_sample(self) -> None:
        sample = [
            ("1742-6", "Alanine aminotransferase", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "U/L"),
            ("1920-8", "Aspartate aminotransferase", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "U/L"),
            ("2160-0", "Creatinine", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
            ("2345-7", "Glucose", "Glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
            ("718-7", "Hemoglobin", "Hemoglobin [Mass/volume] in Blood", "g/dL"),
            ("4544-3", "Hematocrit", "Hematocrit [Volume Fraction] of Blood by Automated count", "%"),
            ("6690-2", "Leukocytes", "Leukocytes [#/volume] in Blood by Automated count", "10*3/uL"),
            ("777-3", "Platelets", "Platelets [#/volume] in Blood by Automated count", "10*3/uL"),
            ("2885-2", "Total protein", "Protein [Mass/volume] in Serum or Plasma", "g/dL"),
            ("1751-7", "Albumin", "Albumin [Mass/volume] in Serum or Plasma", "g/dL"),
            ("1975-2", "Total bilirubin", "Bilirubin.total [Mass/volume] in Serum or Plasma", "mg/dL"),
            ("2951-2", "Sodium", "Sodium [Moles/volume] in Serum or Plasma", "mmol/L"),
            ("2823-3", "Potassium", "Potassium [Moles/volume] in Serum or Plasma", "mmol/L"),
            ("2075-0", "Chloride", "Chloride [Moles/volume] in Serum or Plasma", "mmol/L"),
        ]
        for code, component, lcn, unit in sample:
            e = LOINCEntry(code=code, component=component, long_common_name=lcn, unit=unit)
            self._entries[code] = e
            self._name_index.setdefault(component.lower(), []).append(e)
        logger.info("Loaded %d sample LOINC entries", len(self._entries))

    def lookup(self, text: str) -> list[LOINCEntry]:
        key = text.strip().lower()
        exact = self._name_index.get(key, [])
        if exact:
            return exact
        candidate_names: set[str] = set()
        prefix = key[:3] if len(key) >= 3 else key
        if prefix in self._prefix_index:
            for name in self._prefix_index[prefix]:
                if key in name or name in key:
                    candidate_names.add(name)
        matches: list[LOINCEntry] = []
        for name in candidate_names:
            matches.extend(self._name_index[name])
        return matches[:10]

    def validate_code(self, code: str) -> bool:
        return code in self._entries

    def get_entry(self, code: str) -> LOINCEntry | None:
        return self._entries.get(code)

    @property
    def entry_count(self) -> int:
        """FIX L2: Public accessor instead of exposing _entries."""
        return len(self._entries)

    @property
    def all_entries(self) -> list[LOINCEntry]:
        """Public accessor for iterating all LOINC entries."""
        return list(self._entries.values())

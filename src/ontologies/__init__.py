"""Ontology interfaces — MedDRA, LOINC, SNOMED CT loaders and validators."""
from src.ontologies.loinc import LOINCOntology
from src.ontologies.meddra import MedDRAOntology

__all__ = ["MedDRAOntology", "LOINCOntology"]

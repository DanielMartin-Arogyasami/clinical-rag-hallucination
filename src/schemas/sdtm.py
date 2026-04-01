"""SDTM domain schemas for constrained LLM generation.

These Pydantic models are the target output schemas that the LLM must conform to.
When used with Instructor, they enforce structural validity at generation time.
Reference: CDISC SDTM Implementation Guide v3.4
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class SDTMDomain(str, Enum):
    AE = "AE"
    CM = "CM"
    DM = "DM"
    DS = "DS"
    EX = "EX"
    LB = "LB"
    MH = "MH"
    VS = "VS"


class Severity(str, Enum):
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class Causality(str, Enum):
    NOT_RELATED = "NOT RELATED"
    UNLIKELY = "UNLIKELY"
    POSSIBLE = "POSSIBLE"
    PROBABLE = "PROBABLE"
    DEFINITE = "DEFINITE"


class Outcome(str, Enum):
    RECOVERED = "RECOVERED/RESOLVED"
    RECOVERING = "RECOVERING/RESOLVING"
    NOT_RECOVERED = "NOT RECOVERED/NOT RESOLVED"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"


class AdverseEventRecord(BaseModel):
    """SDTM AE domain — structured output for AE extraction."""

    AETERM: str = Field(description="Verbatim adverse event term as reported")
    AEDECOD: str | None = Field(None, description="MedDRA Preferred Term or UNABLE_TO_DETERMINE")
    AEBODSYS: str | None = Field(None, description="MedDRA System Organ Class")
    AESTDTC: str | None = Field(None, description="Start date/time (ISO 8601)")
    AEENDTC: str | None = Field(None, description="End date/time (ISO 8601)")
    AESEV: Severity | None = Field(None, description="Severity: MILD/MODERATE/SEVERE")
    AESER: Literal["Y", "N"] | None = Field(None, description="Serious event? Y/N")
    AEREL: Causality | None = Field(None, description="Relationship to study treatment")
    AEOUT: Outcome | None = Field(None, description="Outcome of the adverse event")
    AEACN: str | None = Field(None, description="Action taken with study treatment")
    source_text: str | None = Field(None, description="Verbatim source text supporting extraction")
    confidence: str | None = Field(None, description="HIGH/MEDIUM/LOW/UNABLE_TO_DETERMINE")


class LaboratoryRecord(BaseModel):
    """SDTM LB domain — structured output for lab extraction."""

    LBTEST: str = Field(description="Laboratory test name")
    LBTESTCD: str | None = Field(None, description="Short test code")
    LBLOINC: str | None = Field(None, description="LOINC code or UNABLE_TO_DETERMINE")
    LBORRES: str | None = Field(None, description="Result as originally reported")
    LBORRESU: str | None = Field(None, description="Original result unit")
    LBSTRESN: float | None = Field(None, description="Numeric result in standard units")
    LBSTRESU: str | None = Field(None, description="Standard unit")
    LBNRIND: Literal["NORMAL", "LOW", "HIGH", "ABNORMAL"] | None = Field(
        None, description="Reference range indicator"
    )
    LBDTC: str | None = Field(None, description="Collection date/time (ISO 8601)")
    VISIT: str | None = Field(None, description="Visit name")
    source_text: str | None = None
    confidence: str | None = None


class ConcomitantMedicationRecord(BaseModel):
    """SDTM CM domain — structured output for medication extraction."""

    CMTRT: str = Field(description="Verbatim medication name")
    CMDECOD: str | None = Field(None, description="Standardized medication name (WHO-DDE)")
    CMCLAS: str | None = Field(None, description="Medication class (ATC level)")
    CMDOSE: float | None = Field(None, description="Dose per administration")
    CMDOSU: str | None = Field(None, description="Dose unit")
    CMDOSFRQ: str | None = Field(None, description="Dosing frequency (QD, BID, TID)")
    CMROUTE: str | None = Field(None, description="Route of administration")
    CMSTDTC: str | None = Field(None, description="Start date (ISO 8601)")
    CMENDTC: str | None = Field(None, description="End date (ISO 8601)")
    CMINDC: str | None = Field(None, description="Indication")
    source_text: str | None = None
    confidence: str | None = None


class AdverseEventExtractionOutput(BaseModel):
    """Wrapper for multi-AE extraction from a single document."""

    events: list[AdverseEventRecord] = Field(default_factory=list, description="All AEs extracted")


class LaboratoryExtractionOutput(BaseModel):
    results: list[LaboratoryRecord] = Field(
        default_factory=list, description="All lab results extracted"
    )


class ConcomitantMedicationExtractionOutput(BaseModel):
    medications: list[ConcomitantMedicationRecord] = Field(
        default_factory=list, description="All meds extracted"
    )

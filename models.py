"""
models.py — Pydantic models for the Clinical Drug Safety Engine.

All request/response schemas are strictly typed. No raw text is ever returned.
Healthcare-critical: every field is validated before reaching the client.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enumerations ────────────────────────────────────────────────────────────

class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AllergySeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalysisSource(str, Enum):
    LLM = "llm"
    FALLBACK = "fallback"


# ─── Request Models ──────────────────────────────────────────────────────────

class PatientHistory(BaseModel):
    """Patient medical history for contextual drug safety analysis."""

    current_medications: List[str] = Field(
        default_factory=list,
        description="List of medications the patient is currently taking",
    )
    allergies: List[str] = Field(
        default_factory=list,
        description="Known drug allergies",
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Pre-existing medical conditions",
    )
    age: int = Field(
        default=0,
        ge=0,
        le=150,
        description="Patient age in years (0 if unknown)",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=700.0,
        description="Patient weight in kg (0 if unknown)",
    )

    @field_validator("current_medications", "allergies", "conditions", mode="before")
    @classmethod
    def normalize_string_lists(cls, v: list) -> list:
        """Strip whitespace, lowercase, and deduplicate."""
        if not isinstance(v, list):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            normalized = re.sub(r"\s+", " ", normalized)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result


class AnalyzeRequest(BaseModel):
    """Incoming request to analyze drug interactions."""

    medicines: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of medicines to check for interactions",
    )
    patient_history: PatientHistory = Field(
        default_factory=PatientHistory,
        description="Patient medical history context",
    )

    @field_validator("medicines", mode="before")
    @classmethod
    def normalize_medicines(cls, v: list) -> list:
        if not isinstance(v, list):
            raise ValueError("medicines must be a list of strings")
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            normalized = re.sub(r"\s+", " ", normalized)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        if not result:
            raise ValueError("At least one valid medicine must be provided")
        return result


# ─── Response Models ─────────────────────────────────────────────────────────

class DrugInteraction(BaseModel):
    """A single drug-drug interaction record."""

    drug_a: str = Field(..., min_length=1)
    drug_b: str = Field(..., min_length=1)
    severity: Severity
    mechanism: str = Field(..., min_length=1)
    clinical_recommendation: str = Field(..., min_length=1)
    source_confidence: Confidence

    @field_validator("drug_a", "drug_b", mode="before")
    @classmethod
    def lowercase_drugs(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class AllergyAlert(BaseModel):
    """An allergy alert for a specific medicine."""

    medicine: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    severity: AllergySeverity

    @field_validator("medicine", mode="before")
    @classmethod
    def lowercase_medicine(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class ContraindicationAlert(BaseModel):
    """A condition-based contraindication alert."""

    medicine: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    risk_level: Severity
    recommendation: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    """Full structured response from the drug safety engine."""

    interactions: List[DrugInteraction] = Field(default_factory=list)
    allergy_alerts: List[AllergyAlert] = Field(default_factory=list)
    contraindication_alerts: List[ContraindicationAlert] = Field(default_factory=list)
    safe_to_prescribe: bool = True
    overall_risk_level: RiskLevel = RiskLevel.LOW
    patient_risk_score: int = Field(default=0, ge=0, le=100)
    requires_doctor_review: bool = False
    source: AnalysisSource = AnalysisSource.LLM
    cache_hit: bool = False
    processing_time_ms: float = 0.0

    @model_validator(mode="after")
    def validate_consistency(self) -> "AnalyzeResponse":
        """Ensure safe_to_prescribe and risk_level are consistent."""
        has_high_interaction = any(
            i.severity == Severity.HIGH for i in self.interactions
        )
        has_critical_allergy = any(
            a.severity == AllergySeverity.CRITICAL for a in self.allergy_alerts
        )

        if has_high_interaction or has_critical_allergy:
            self.safe_to_prescribe = False

        if not self.safe_to_prescribe:
            self.requires_doctor_review = True

        return self


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    detail: Optional[str] = None
    status_code: int = 500

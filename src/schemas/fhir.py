"""
src/schemas/fhir.py
────────────────────
HL7 FHIR R4 Pydantic models for the GoTHoMIS/OpenMRS integration.

These models validate incoming FHIR payloads at the API boundary
before they are parsed into symptoms by fhir_bridge/parser.py (Stage 5).

FHIR R4 resources used by MediAssist:
  - Patient      → age, gender, identifiers
  - Condition    → chief complaints, diagnoses from EMR
  - Observation  → vital signs (temperature, blood pressure, weight)

Reference: https://www.hl7.org/fhir/R4/

The FHIRPatientContext is what GoTHoMIS sends to
POST /api/v1/fhir/patient-context. It bundles all three resource
types in one request so the API has full context in one call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Shared FHIR building blocks
# ─────────────────────────────────────────────────────────────────────────────


class FHIRCoding(BaseModel):
    """A code defined by a terminology system (e.g. ICD-10, SNOMED)."""
    system:  Optional[str] = None
    code:    Optional[str] = None
    display: Optional[str] = None

    model_config = {"extra": "allow"}


class FHIRCodeableConcept(BaseModel):
    """A concept that may be defined by one or more codings."""
    coding: Optional[List[FHIRCoding]] = None
    text:   Optional[str]             = None

    model_config = {"extra": "allow"}

    def get_display_text(self) -> str:
        """Return the best human-readable label for this concept."""
        if self.text:
            return self.text
        if self.coding:
            for c in self.coding:
                if c.display:
                    return c.display
                if c.code:
                    return c.code
        return "Unknown"


class FHIRReference(BaseModel):
    """A reference to another FHIR resource."""
    reference: Optional[str] = None
    display:   Optional[str] = None

    model_config = {"extra": "allow"}


class FHIRQuantity(BaseModel):
    """A measured quantity with value and units."""
    value:  Optional[float] = None
    unit:   Optional[str]  = None
    system: Optional[str]  = None
    code:   Optional[str]  = None

    model_config = {"extra": "allow"}

    def as_string(self) -> str:
        """Return a human-readable measurement string."""
        if self.value is not None and self.unit:
            return f"{self.value} {self.unit}"
        if self.value is not None:
            return str(self.value)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# FHIR Patient (R4)
# ─────────────────────────────────────────────────────────────────────────────


class FHIRPatient(BaseModel):
    """
    HL7 FHIR R4 Patient resource — subset used by MediAssist.

    Required fields: resourceType, id, gender, birthDate.
    These are the minimum needed to extract patient context.
    """
    resourceType: str = Field(
        default="Patient",
        description="Must be 'Patient'.",
    )
    id: str = Field(
        ...,
        description="Patient identifier (GoTHoMIS patient ID).",
    )
    gender: str = Field(
        ...,
        description="'male' | 'female' | 'other' | 'unknown'",
    )
    birthDate: Optional[str] = Field(
        default=None,
        description="ISO 8601 date e.g. '1990-05-15'.",
    )
    name: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Patient name(s) — HumanName FHIR type.",
    )

    model_config = {"extra": "allow"}

    def get_age_years(self) -> Optional[int]:
        """Calculate age from birthDate. Returns None if birthDate missing."""
        if not self.birthDate:
            return None
        try:
            from datetime import date

            parts = self.birthDate.split("-")
            born = date(int(parts[0]), int(parts[1]), int(parts[2]))
            today = date.today()
            return (today - born).days // 365
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# FHIR Condition (R4)
# ─────────────────────────────────────────────────────────────────────────────


class FHIRCondition(BaseModel):
    """
    HL7 FHIR R4 Condition resource.

    Used by GoTHoMIS to send patient chief complaints and past diagnoses.
    The code.text field is the primary symptom/condition description.
    """
    resourceType:     str                         = Field(default="Condition")
    id:               Optional[str]               = None
    code:             Optional[FHIRCodeableConcept] = Field(
        default=None,
        description="The condition code — text field contains the description.",
    )
    clinicalStatus:   Optional[FHIRCodeableConcept] = Field(
        default=None,
        description="'active' | 'resolved' | 'inactive'",
    )
    subject:          Optional[FHIRReference]     = None
    onsetDateTime:    Optional[str]               = None
    note:             Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "allow"}

    def get_condition_text(self) -> str:
        """Extract the human-readable condition name."""
        if self.code:
            return self.code.get_display_text()
        return "Unknown condition"


# ─────────────────────────────────────────────────────────────────────────────
# FHIR Observation (R4)
# ─────────────────────────────────────────────────────────────────────────────


class FHIRObservation(BaseModel):
    """
    HL7 FHIR R4 Observation resource.

    Used by GoTHoMIS to send vital signs:
      Temperature, Blood Pressure, Weight, Pulse Rate, etc.
    The code.text identifies the observation type.
    The valueQuantity holds the measured value.
    """
    resourceType:     str                         = Field(default="Observation")
    id:               Optional[str]               = None
    status:           Optional[str]               = Field(
        default="final",
        description="'final' | 'preliminary' | 'amended'",
    )
    code:             Optional[FHIRCodeableConcept] = Field(
        default=None,
        description="What was measured e.g. 'Body temperature'.",
    )
    valueQuantity:    Optional[FHIRQuantity]       = Field(
        default=None,
        description="The measured value with units.",
    )
    valueString:      Optional[str]                = Field(
        default=None,
        description="String value for non-quantitative observations.",
    )
    subject:          Optional[FHIRReference]      = None
    effectiveDateTime: Optional[str]               = None

    model_config = {"extra": "allow"}

    def get_observation_text(self) -> str:
        """Return a human-readable observation string e.g. 'Temperature: 38.5 °C'."""
        obs_type = self.code.get_display_text() if self.code else "Observation"
        if self.valueQuantity:
            value_str = self.valueQuantity.as_string()
            return f"{obs_type}: {value_str}"
        if self.valueString:
            return f"{obs_type}: {self.valueString}"
        return obs_type


# ─────────────────────────────────────────────────────────────────────────────
# FHIRPatientContext — the top-level payload for POST /api/v1/fhir/patient-context
# ─────────────────────────────────────────────────────────────────────────────


class FHIRPatientContext(BaseModel):
    """
    Bundled FHIR context sent by GoTHoMIS to the MediAssist FHIR bridge.

    Contains the patient, their conditions (chief complaints), and
    their latest observations (vital signs) in one request.

    This is NOT a standard FHIR Bundle — it is a MediAssist-specific
    wrapper for simplicity of integration with GoTHoMIS.
    """
    patient:      FHIRPatient           = Field(
        ..., description="FHIR R4 Patient resource — required."
    )
    conditions:   List[FHIRCondition]   = Field(
        default_factory=list,
        description="Active conditions / chief complaints from GoTHoMIS.",
    )
    observations: List[FHIRObservation] = Field(
        default_factory=list,
        description="Recent vital sign observations from GoTHoMIS.",
    )

    model_config = {"extra": "allow"}

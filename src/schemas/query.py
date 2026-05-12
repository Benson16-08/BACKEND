
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):

    symptoms: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Free-text symptom description entered by the doctor.",
        examples=["fever with chills and headache for two days"],
    )

    patient_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Patient name from PatientInfo form.",
    )

    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=150,
        description="Patient age in years.",
    )

    gender: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Patient sex/gender from PatientInfo form.",
    )

    weight: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Patient weight (string to allow units, e.g. '70kg').",
    )

    temperature: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Patient temperature reading (e.g. '38.5°C').",
    )

    blood_pressure: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Patient blood pressure (e.g. '120/80').",
    )

    allergy: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Known patient allergies (optional).",
    )

    @field_validator("symptoms")
    @classmethod
    def symptoms_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Symptoms cannot be whitespace only.")
        return v.strip()

    @field_validator("gender")
    @classmethod
    def normalise_gender(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v.strip().lower()


class QueryResponse(BaseModel):
    status: str = "received"
    query: str
    message: str = "Query received. Full pipeline not yet wired."
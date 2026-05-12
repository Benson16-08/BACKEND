
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from src.core.config import settings


class EvidenceItem(BaseModel):
    source: str = Field(
        ...,
        description="Source document name e.g. 'Tanzania STG 2017 — Malaria'",
    )
    excerpt: str = Field(
        ...,
        min_length=10,
        description="Relevant excerpt from the STG guideline chunk.",
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Chapter or section of the guideline.",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance score from reranker (0–1).",
    )


class DiagnosisItem(BaseModel):
    rank: int = Field(
        ...,
        ge=1,
        description="Rank order — 1 is the most probable diagnosis.",
    )
    condition: str = Field(
        ...,
        min_length=2,
        description="Name of the clinical condition e.g. 'Malaria'.",
    )
    probability: int = Field(
        ...,
        ge=1,
        description="Estimated probability as an integer percentage (1–99).",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Clinical reasoning linking symptoms to this diagnosis.",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="STG guideline evidence supporting this diagnosis.",
    )
    icd10_code: Optional[str] = Field(
        default=None,
        description="ICD-10 code for this condition e.g. 'B50.9'.",
    )

    @field_validator("probability")
    @classmethod
    def cap_probability(cls, v: int) -> int:
        max_prob = settings.MAX_DIAGNOSIS_PROBABILITY
        if v > max_prob:
            return max_prob
        return v

    @field_validator("evidence")
    @classmethod
    def evidence_must_cite_stg(cls, v: List[EvidenceItem]) -> List[EvidenceItem]:
        if len(v) == 0:
            raise ValueError(
                "Each diagnosis must include at least one STG evidence item. "
                "Unsupported diagnoses cannot be presented to clinicians."
            )
        return v


class DiagnosticResponse(BaseModel):
    diagnoses: List[DiagnosisItem] = Field(
        ...,
        min_length=1,
        description="Ranked list of differential diagnoses.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Clarifying questions the doctor should ask the patient.",
    )
    disclaimer: str = Field(
        default=(
            "This output is a clinical decision support tool only. "
            "It does not replace the judgment of a qualified medical professional. "
            "All diagnoses and management decisions must be confirmed by a licensed clinician."
        ),
        description="Mandatory safety disclaimer shown on every response.",
    )
    retrieval_ms: Optional[int] = Field(
        default=None,
        description="Time taken for retrieval pipeline in milliseconds.",
    )
    llm_ms: Optional[int] = Field(
        default=None,
        description="Time taken for LLM generation in milliseconds.",
    )
    total_ms: Optional[int] = Field(
        default=None,
        description="Total end-to-end time in milliseconds.",
    )
    warning: Optional[str] = Field(
        default=None,
        description=(
            "Safety warning added by safety_checker.py if any rule was "
            "triggered. Shown prominently in the frontend."
        ),
    )

    @field_validator("diagnoses")
    @classmethod
    def diagnoses_ranked_correctly(cls, v: List[DiagnosisItem]) -> List[DiagnosisItem]:
        for i, item in enumerate(v, start=1):
            if item.rank != i:
                item.rank = i
        return v
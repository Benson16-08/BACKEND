
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DiagnosisItem(BaseModel):
    """
    A single ranked diagnosis — matches Peter's DiagnosisItem exactly.

    Peter's Calibrator renormalises probabilities and reassigns ranks,
    so by the time we receive this object it is already clean.
    """
    rank: int = Field(
        ge=1, le=5,
        description="Position in the differential. 1 = most likely.",
    )
    condition: str = Field(
        min_length=2,
        description="Condition name e.g. 'Malaria'.",
    )
    probability: int = Field(
        ge=0, le=100,
        description="Calibrated probability integer (Peter's Calibrator ensures sum=100).",
    )
    reasoning: str = Field(
        min_length=10,
        description="Clinical reasoning linking symptoms to this diagnosis.",
    )
    evidence: str = Field(
        min_length=10,
        description="Direct quote or close paraphrase from the STG (Peter's Auditor verifies this).",
    )
    source_section: str = Field(
        description="STG section the evidence is drawn from e.g. 'Chapter 3 — Malaria'.",
    )
    icd10_code: Optional[str] = Field(
        default=None,
        description="ICD-10 code added by Benson's bridge if available.",
    )


class DiagnosticResponse(BaseModel):
    """
    Full response contract between Peter's pipeline and Benson's API.

    Fields marked [PETER] come directly from run_mediassist_pipeline().
    Fields marked [BENSON] are added by our llm_client.py after the call.
    Jesca's (Role 5) frontend reads all fields.
    """

    # ── [PETER] Core pipeline output ─────────────────────────────────────────
    diagnoses: List[DiagnosisItem] = Field(
        min_length=1, max_length=5,
        description="[PETER] Ranked differential diagnoses, top first.",
    )
    follow_up_questions: List[str] = Field(
        min_length=1,
        description="[PETER] Clarifying questions for the doctor to ask.",
    )
    recommended_tests: List[str] = Field(
        default_factory=list,
        description="[PETER] Diagnostic tests to confirm or rule out diagnoses.",
    )
    confidence_overall: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="[PETER] LLM self-reported confidence level.",
    )
    pipeline_confidence: Optional[Literal["low", "medium", "high"] ] = Field(
        default=None,
        description="[PETER] Strategist-computed confidence (set after calibration).",
    )

    # ── [BENSON] Added by llm_client.py after pipeline returns ───────────────
    disclaimer: str = Field(
        default=(
            "This output is a clinical decision support tool only. "
            "It does not replace the judgment of a qualified medical professional. "
            "All diagnoses must be confirmed by a licensed clinician."
        ),
        description="[BENSON] Mandatory safety disclaimer shown on every response.",
    )
    retrieval_ms: Optional[int] = Field(
        default=None,
        description="[BENSON] Time taken for retrieval pipeline in milliseconds.",
    )
    llm_ms: Optional[int] = Field(
        default=None,
        description="[BENSON] Time taken for LLM generation in milliseconds.",
    )
    total_ms: Optional[int] = Field(
        default=None,
        description="[BENSON] Total end-to-end time in milliseconds.",
    )
    warning: Optional[str] = Field(
        default=None,
        description="[BENSON] Safety warning from safety_checker.py if any rule triggered.",
    )

    @field_validator("diagnoses")
    @classmethod
    def ranks_must_be_sequential(cls, v: List[DiagnosisItem]) -> List[DiagnosisItem]:
        """Auto-correct ranks to 1..N in case Peter's Calibrator missed any."""
        for i, item in enumerate(v, start=1):
            item.rank = i
        return v

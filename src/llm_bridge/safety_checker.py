from __future__ import annotations

import re
from typing import List

from src.core.config import settings
from src.core.logger import logger
from src.schemas.diagnosis import DiagnosticResponse


_PRESCRIPTION_PATTERNS = [
    r"\b(?:prescribe|prescribed|prescription)\b",
    r"\b\d+\s*mg\b",
    r"\b\d+\s*ml\b",
    r"\btake\s+\d+",
    r"\b(?:twice|three times)\s+(?:daily|a day)\b",
    r"\bonce\s+daily\b",
    r"\badminister\b",
]

_PRESCRIPTION_RE = re.compile(
    "|".join(_PRESCRIPTION_PATTERNS),
    flags=re.IGNORECASE,
)


def _check_prescription_language(response: DiagnosticResponse) -> List[str]:
    """Return list of conditions whose text contains prescription language."""
    flagged = []
    for d in response.diagnoses:
        combined = f"{d.reasoning} {d.evidence}"
        if _PRESCRIPTION_RE.search(combined):
            flagged.append(d.condition)
    return flagged


def check_response(response: DiagnosticResponse) -> DiagnosticResponse:
    """
    Apply safety rules to DiagnosticResponse before sending to frontend.

    Adds a `warning` field if any rule fires; never blocks output.

    Rules:
      1. No probability >= 100 (false certainty)
      2. Every diagnosis must have non-empty evidence
      3. At least one follow-up question present
      4. Disclaimer field must be present
      5. No prescription/dosing language in reasoning or evidence

    Args:
        response: DiagnosticResponse from llm_client.run_pipeline().

    Returns:
        The same response object, with `warning` field populated if needed.
    """
    warnings: List[str] = []

    # Rule 1: Cap probability at MAX_DIAGNOSIS_PROBABILITY
    for d in response.diagnoses:
        if d.probability >= 100:
            d.probability = settings.MAX_DIAGNOSIS_PROBABILITY
            warnings.append(
                f"Probability for '{d.condition}' was 100% — "
                f"capped to {settings.MAX_DIAGNOSIS_PROBABILITY}% (no diagnosis is certain)."
            )
            logger.warning(f"Rule 1: {d.condition} had probability 100%")

    # Rule 2: Ensure evidence field is populated
    for d in response.diagnoses:
        if not d.evidence or len(d.evidence.strip()) < 10:
            d.evidence = "Refer to Tanzania Standard Treatment Guidelines for this condition."
            warnings.append(
                f"'{d.condition}' was missing STG evidence — placeholder added."
            )
            logger.warning(f"Rule 2: {d.condition} had no evidence")

    # Rule 3: Ensure at least one follow-up question
    if not response.follow_up_questions:
        response.follow_up_questions = [
            "Please perform a full clinical assessment.",
            "What is the duration and progression of symptoms?",
        ]
        warnings.append("No follow-up questions — defaults added.")
        logger.warning("Rule 3: no follow-up questions in response")

    # Rule 4: Ensure disclaimer is present
    if not response.disclaimer or len(response.disclaimer.strip()) < 20:
        response.disclaimer = (
            "This output is a clinical decision support tool only. "
            "It does not replace the judgment of a qualified medical professional. "
            "All diagnoses must be confirmed by a licensed clinician."
        )
        logger.warning("Rule 4: disclaimer was missing")

    # Rule 5: Flag prescription language
    flagged = _check_prescription_language(response)
    if flagged:
        warnings.append(
            f"⚠️ PRESCRIPTION LANGUAGE DETECTED in: {', '.join(flagged)}. "
            "This tool does not prescribe medication."
        )
        logger.warning(f"Rule 5: prescription language in {flagged}")

    if warnings:
        response.warning = " | ".join(warnings)
        logger.info(f"safety_checker: {len(warnings)} warning(s) added")
    else:
        logger.info("safety_checker: all rules passed")

    return response

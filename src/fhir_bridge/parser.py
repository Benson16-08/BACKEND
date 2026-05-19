from __future__ import annotations

import re
from typing import Optional

from src.core.logger import logger
from src.schemas.fhir import FHIRObservation, FHIRPatientContext
from src.schemas.query import QueryRequest


def _find_observation(
    observations: list,
    keyword: str,
) -> Optional[FHIRObservation]:
    """Return first observation whose code text matches keyword (case-insensitive)."""
    kw = keyword.lower()
    for obs in observations:
        code_text = ""
        if obs.code:
            code_text = obs.code.get_display_text().lower()
        if kw in code_text:
            return obs
    return None


def _extract_observation_value(obs: Optional[FHIRObservation]) -> Optional[str]:
    """Extract the human-readable value string from an observation."""
    if obs is None:
        return None
    if obs.valueQuantity:
        return obs.valueQuantity.as_string()
    if obs.valueString:
        return obs.valueString
    return None


def parse_patient_context(ctx: FHIRPatientContext) -> QueryRequest:
    """
    Convert a FHIRPatientContext into a QueryRequest for the pipeline.

    Args:
        ctx: Validated FHIRPatientContext from validator.py.

    Returns:
        QueryRequest with symptoms string and all available patient metadata.
    """
    condition_texts = []
    for condition in ctx.conditions:
        text = condition.get_condition_text()
        if text and text != "Unknown condition":
            onset = condition.onsetDateTime
            if onset:
                condition_texts.append(f"{text} (onset: {onset[:10]})")
            else:
                condition_texts.append(text)

    vital_texts = []
    for obs in ctx.observations:
        obs_text = obs.get_observation_text()
        if obs_text and obs_text != "Observation":
            vital_texts.append(obs_text)

    symptoms_parts = []
    if condition_texts:
        symptoms_parts.append("Chief complaints: " + "; ".join(condition_texts))
    if vital_texts:
        symptoms_parts.append("Vitals: " + "; ".join(vital_texts))

    symptoms = ". ".join(symptoms_parts)

    if not symptoms.strip():
        symptoms = "Patient presenting for clinical assessment."

    patient_age = ctx.patient.get_age_years()
    gender = ctx.patient.gender

    temp_obs    = _find_observation(ctx.observations, "temperature")
    bp_obs      = _find_observation(ctx.observations, "pressure")
    weight_obs  = _find_observation(ctx.observations, "weight")

    temperature    = _extract_observation_value(temp_obs)
    blood_pressure = _extract_observation_value(bp_obs)
    weight         = _extract_observation_value(weight_obs)

    patient_name = None
    if ctx.patient.name and len(ctx.patient.name) > 0:
        name_entry = ctx.patient.name[0]
        if isinstance(name_entry, dict):
            family = name_entry.get("family", "")
            given  = " ".join(name_entry.get("given", []))
            patient_name = f"{given} {family}".strip() or None

    query = QueryRequest(
        symptoms=symptoms,
        patient_name=patient_name,
        patient_age=patient_age,
        gender=gender,
        temperature=temperature,
        blood_pressure=blood_pressure,
        weight=weight,
    )

    logger.info(
        f"parse_patient_context: patient={ctx.patient.id}  "
        f"age={patient_age}  gender={gender}  "
        f"symptoms_len={len(symptoms)}  "
        f"conditions={len(ctx.conditions)}  observations={len(ctx.observations)}"
    )
    return query

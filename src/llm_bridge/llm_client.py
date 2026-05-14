from __future__ import annotations

import asyncio
import json
import re
import time
from typing import List, Optional

from openai import OpenAI

from src.core.config import settings
from src.core.exceptions import LLMTimeoutError, LLMUnavailableError, LLMResponseInvalidError
from src.core.logger import logger
from src.schemas.diagnosis import DiagnosisItem, DiagnosticResponse
from src.schemas.query import QueryRequest


SYSTEM_PROMPT = """You are a clinical decision support assistant for \
Tanzanian healthcare facilities.

Your task: receive patient symptoms and excerpts from the Tanzania \
Standard Treatment Guidelines (STG), then produce a structured \
differential diagnosis.

Rules:
- Reason ONLY from the provided STG excerpts. Do not use general medical \
knowledge from your training data.
- Every diagnosis MUST include a direct quote from the STG in its \
`evidence` field.
- Probabilities across all diagnoses must sum to approximately 100.
- List 1 to 5 diagnoses, ranked by likelihood (rank 1 = most likely).

Respond with ONLY valid JSON in this exact schema:

{
  "diagnoses": [
    {
      "rank": 1,
      "condition": "Disease name",
      "probability": 70,
      "reasoning": "Why these symptoms suggest this condition.",
      "evidence": "A direct quote from the STG.",
      "source_section": "Section name from the STG"
    }
  ],
  "follow_up_questions": ["Question 1", "Question 2"],
  "recommended_tests": ["Test 1", "Test 2"],
  "confidence_overall": "low" | "medium" | "high"
}

Do NOT include markdown code fences, explanations, or any text outside \
the JSON object."""


def _get_openai_client():
    """Return an OpenAI client for the configured LLM provider."""
    if settings.LLM_PROVIDER == "gpt":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-placeholder"):
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not set. Add your real key to .env."
            )
        return OpenAI(api_key=settings.OPENAI_API_KEY), settings.OPENAI_MODEL
    else:
        return OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
        ), settings.OLLAMA_MODEL


def _health_check() -> bool:
    """Quick liveness check for the configured LLM provider."""
    try:
        if settings.LLM_PROVIDER == "gpt":
            client, model = _get_openai_client()
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        else:
            import requests
            root = settings.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
            r = requests.get(f"{root}/api/tags", timeout=3)
            return r.status_code == 200
        return True
    except Exception:
        return False


def _assemble_messages(symptoms: str, chunks: List[str]) -> List[dict]:
    """Build chat messages for the LLM."""
    guidelines_text = "\n\n".join(chunks)
    user_message = (
        f"Patient symptoms: {symptoms}\n\n"
        f"Relevant STG excerpts:\n{guidelines_text}\n\n"
        f"Provide a ranked differential diagnosis in the required JSON format."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def _clean_llm_output(raw: str) -> str:
    """Strip markdown fences and extract JSON from LLM output."""
    text = raw.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Parse LLM output into a dict. Returns None on failure."""
    cleaned = _clean_llm_output(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e} | cleaned='{cleaned[:200]}'")
        return None


def _call_llm_direct(symptoms: str, chunks: List[str]) -> str:
    """Call the LLM directly using the system prompt. Returns raw text."""
    client, model = _get_openai_client()
    messages = _assemble_messages(symptoms, chunks)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )

    if response.choices[0].finish_reason == "length":
        logger.warning("LLM response cut off at max_tokens — JSON may be incomplete")

    return response.choices[0].message.content


def _call_peter_pipeline(symptoms: str, chunks: List[str]) -> DiagnosticResponse:
    """
    Try to import and call Peter's run_mediassist_pipeline().
    Falls back to direct LLM call if his module is not available.
    """
    try:
        from role3.main import run_mediassist_pipeline
        peter_response = run_mediassist_pipeline(symptoms, chunks)

        our_diagnoses = []
        for d in peter_response.diagnoses:
            our_diagnoses.append(DiagnosisItem(
                rank=d.rank,
                condition=d.condition,
                probability=d.probability,
                reasoning=d.reasoning,
                evidence=d.evidence,
                source_section=d.source_section,
            ))

        return DiagnosticResponse(
            diagnoses=our_diagnoses,
            follow_up_questions=peter_response.follow_up_questions,
            recommended_tests=peter_response.recommended_tests,
            confidence_overall=peter_response.confidence_overall,
            pipeline_confidence=peter_response.pipeline_confidence,
        )

    except ImportError:
        logger.info(
            "Peter's role3 module not installed — calling LLM directly "
            "(same prompt, same output format)"
        )
        return _call_llm_direct_and_parse(symptoms, chunks)


def _call_llm_direct_and_parse(symptoms: str, chunks: List[str]) -> DiagnosticResponse:
    """Call LLM directly, parse output, and return DiagnosticResponse."""
    raw = _call_llm_direct(symptoms, chunks)
    data = _parse_llm_json(raw)

    if data is None:
        raise LLMResponseInvalidError(
            "LLM returned output that could not be parsed as valid JSON."
        )

    try:
        diagnoses_raw = data.get("diagnoses", [])
        if not diagnoses_raw:
            raise LLMResponseInvalidError("LLM returned empty diagnoses list.")

        diagnoses = []
        for d in diagnoses_raw[:5]:
            diagnoses.append(DiagnosisItem(
                rank=d.get("rank", 1),
                condition=d.get("condition", "Unknown"),
                probability=min(int(d.get("probability", 50)), 100),
                reasoning=d.get("reasoning", "See STG guidelines."),
                evidence=d.get("evidence", "Refer to Tanzania STG."),
                source_section=d.get("source_section", "Tanzania STG"),
            ))

        total = sum(d.probability for d in diagnoses)
        if total > 0 and abs(total - 100) > 5:
            for d in diagnoses:
                d.probability = max(2, min(95, round(d.probability * 100 / total)))

        diagnoses.sort(key=lambda d: d.probability, reverse=True)
        for i, d in enumerate(diagnoses, 1):
            d.rank = i

        return DiagnosticResponse(
            diagnoses=diagnoses,
            follow_up_questions=data.get("follow_up_questions", [
                "Please provide more details about the symptom onset.",
                "Any recent travel or exposure history?"
            ]),
            recommended_tests=data.get("recommended_tests", [
                "Perform clinical assessment."
            ]),
            confidence_overall=data.get("confidence_overall", "medium"),
            pipeline_confidence=None,
        )

    except Exception as exc:
        raise LLMResponseInvalidError(
            f"Failed to build DiagnosticResponse from LLM output: {exc}"
        ) from exc


async def run_pipeline(
    request: QueryRequest,
    chunks: List[dict],
    retrieval_ms: Optional[int] = None,
) -> DiagnosticResponse:
    """
    Run the full LLM pipeline and return a DiagnosticResponse.

    Args:
        request:      QueryRequest from the API (symptoms + patient metadata).
        chunks:       Retrieved STG chunks from retrieval_service (list of dicts).
        retrieval_ms: Time taken by retrieval pipeline (for performance logging).

    Returns:
        DiagnosticResponse with all fields populated.

    Raises:
        LLMUnavailableError: if LLM provider is down.
        LLMTimeoutError:     if LLM call exceeds 60 seconds.
        LLMResponseInvalidError: if LLM output cannot be parsed.
    """
    t_start = time.perf_counter()

    symptoms_parts = [request.symptoms]
    if request.patient_age:
        symptoms_parts.append(f"Patient age: {request.patient_age} years.")
    if request.gender:
        symptoms_parts.append(f"Sex: {request.gender}.")
    if request.temperature:
        symptoms_parts.append(f"Temperature: {request.temperature}.")
    if request.blood_pressure:
        symptoms_parts.append(f"Blood pressure: {request.blood_pressure}.")
    symptoms = " ".join(symptoms_parts)

    chunk_texts: List[str] = [c.get("text", "") for c in chunks if c.get("text")]

    logger.info(
        f"run_pipeline: provider={settings.LLM_PROVIDER}  "
        f"symptoms_len={len(symptoms)}  chunks={len(chunk_texts)}"
    )

    if not _health_check():
        raise LLMUnavailableError(
            f"LLM provider '{settings.LLM_PROVIDER}' is not reachable. "
            "Check your API key or Ollama server."
        )

    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, _call_peter_pipeline, symptoms, chunk_texts),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise LLMTimeoutError(
            "LLM call exceeded 60 seconds. Try again or switch to a faster provider."
        )
    except (LLMUnavailableError, LLMTimeoutError, LLMResponseInvalidError):
        raise
    except Exception as exc:
        logger.error(f"run_pipeline unexpected error: {exc}")
        raise LLMUnavailableError(f"LLM pipeline failed unexpectedly: {exc}") from exc

    llm_ms = int((time.perf_counter() - t_start) * 1000)
    total_ms = (retrieval_ms or 0) + llm_ms

    response.retrieval_ms = retrieval_ms
    response.llm_ms = llm_ms
    response.total_ms = total_ms

    logger.info(
        f"run_pipeline: done  llm_ms={llm_ms}  total_ms={total_ms}  "
        f"diagnoses={len(response.diagnoses)}  "
        f"confidence={response.confidence_overall}"
    )
    return response

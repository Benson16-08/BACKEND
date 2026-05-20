import pytest

from src.llm_bridge.safety_checker import check_response, _check_prescription_language
from src.schemas.diagnosis import DiagnosisItem, DiagnosticResponse


def _make_item(**kw) -> DiagnosisItem:
    defaults = dict(
        rank=1,
        condition="Malaria",
        probability=75,
        reasoning="Fever and chills consistent with malaria per Tanzania STG.",
        evidence="Malaria presents with fever, chills, and headache per STG guidelines.",
        source_section="Chapter Five — Malaria",
    )
    defaults.update(kw)
    return DiagnosisItem(**defaults)


def _make_response(**kw) -> DiagnosticResponse:
    defaults = dict(
        diagnoses=[_make_item()],
        follow_up_questions=["How long has the fever lasted?"],
        recommended_tests=["mRDT for Malaria"],
        confidence_overall="medium",
    )
    defaults.update(kw)
    return DiagnosticResponse(**defaults)


def _bypass(**kw) -> DiagnosticResponse:
    defaults = dict(
        diagnoses=[_make_item()],
        follow_up_questions=["Duration of fever?"],
        recommended_tests=["mRDT"],
        confidence_overall="medium",
        disclaimer="This is a clinical decision support tool only.",
        warning=None,
        retrieval_ms=None,
        llm_ms=None,
        total_ms=None,
        pipeline_confidence=None,
    )
    defaults.update(kw)
    return DiagnosticResponse.model_construct(**defaults)


class TestRule1ProbabilityCap:

    def test_probability_100_is_capped(self):
        r = _make_response()
        r.diagnoses[0].probability = 100
        result = check_response(r)
        assert result.diagnoses[0].probability <= 99

    def test_probability_100_adds_warning(self):
        r = _make_response()
        r.diagnoses[0].probability = 100
        result = check_response(r)
        assert result.warning is not None

    def test_probability_99_unchanged(self):
        r = _make_response()
        r.diagnoses[0].probability = 99
        result = check_response(r)
        assert result.diagnoses[0].probability == 99

    def test_probability_75_unchanged(self):
        r = _make_response()
        result = check_response(r)
        assert result.diagnoses[0].probability == 75
        assert result.warning is None


class TestRule2EvidenceRequired:

    def test_short_evidence_gets_placeholder(self):
        r = _bypass()
        r.diagnoses[0].evidence = "x"
        result = check_response(r)
        assert len(result.diagnoses[0].evidence) > 10

    def test_short_evidence_adds_warning(self):
        r = _bypass()
        r.diagnoses[0].evidence = "short"
        result = check_response(r)
        assert result.warning is not None

    def test_valid_evidence_unchanged(self):
        r = _make_response()
        original_evidence = r.diagnoses[0].evidence
        result = check_response(r)
        assert result.diagnoses[0].evidence == original_evidence


class TestRule3FollowUpQuestions:

    def test_empty_follow_up_gets_defaults(self):
        r = _bypass(follow_up_questions=[])
        result = check_response(r)
        assert len(result.follow_up_questions) > 0

    def test_empty_follow_up_adds_warning(self):
        r = _bypass(follow_up_questions=[])
        result = check_response(r)
        assert result.warning is not None

    def test_valid_follow_up_unchanged(self):
        r = _make_response()
        result = check_response(r)
        assert len(result.follow_up_questions) >= 1


class TestRule4Disclaimer:

    def test_empty_disclaimer_restored(self):
        r = _make_response()
        r.disclaimer = ""
        result = check_response(r)
        assert len(result.disclaimer) > 20

    def test_short_disclaimer_restored(self):
        r = _make_response()
        r.disclaimer = "Short"
        result = check_response(r)
        assert len(result.disclaimer) > 20

    def test_valid_disclaimer_present(self):
        r = _make_response()
        result = check_response(r)
        assert len(result.disclaimer) > 20


class TestRule5NoPrescriptions:

    def test_prescribe_keyword_detected(self):
        r = _make_response()
        r.diagnoses[0].reasoning = "You should prescribe artemether for this condition."
        result = check_response(r)
        assert result.warning is not None
        assert "prescription" in result.warning.lower()

    def test_dosage_mg_detected(self):
        r = _make_response()
        r.diagnoses[0].reasoning = "Give 500mg of this medication daily."
        result = check_response(r)
        assert result.warning is not None

    def test_twice_daily_detected(self):
        r = _make_response()
        r.diagnoses[0].reasoning = "Administer twice daily for 5 days."
        result = check_response(r)
        assert result.warning is not None

    def test_administer_detected(self):
        r = _make_response()
        r.diagnoses[0].reasoning = "Administer IV fluids immediately."
        result = check_response(r)
        assert result.warning is not None

    def test_clean_reasoning_no_warning(self):
        r = _make_response()
        result = check_response(r)
        assert result.warning is None

    def test_drug_name_in_context_ok(self):
        r = _make_response()
        r.diagnoses[0].reasoning = (
            "Consider antimalarial treatment per STG if mRDT confirms malaria. "
            "Artemether-lumefantrine is the first-line drug per Tanzania STG."
        )
        result = check_response(r)
        assert result.warning is None

    def test_evidence_field_also_checked(self):
        r = _make_response()
        r.diagnoses[0].evidence = "Give patient 250mg chloroquine once daily."
        result = check_response(r)
        assert result.warning is not None


class TestCleanResponse:

    def test_fully_valid_response_has_no_warning(self):
        r = _make_response()
        result = check_response(r)
        assert result.warning is None

    def test_check_response_never_raises(self):
        r = _bypass()
        r.diagnoses[0].evidence = ""
        r.diagnoses[0].probability = 100
        r.follow_up_questions = []
        r.disclaimer = ""
        result = check_response(r)
        assert result is not None

    def test_check_response_always_returns_response(self):
        r = _make_response()
        result = check_response(r)
        assert isinstance(result, DiagnosticResponse)

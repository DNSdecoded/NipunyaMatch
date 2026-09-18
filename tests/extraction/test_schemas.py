import pytest
from pydantic import ValidationError

from app.extraction.schemas import CandidateAnalysis, CandidateExtraction


def test_extraction_minimal_and_defaults() -> None:
    e = CandidateExtraction.model_validate_json('{"skills": ["Python"]}')
    assert e.name is None and e.skills == ["Python"] and e.experience == []


def test_extraction_rejects_bad_email() -> None:
    with pytest.raises(ValidationError):
        CandidateExtraction(email="not-an-email")


def test_analysis_bounds() -> None:
    base = dict(skill_matches=[], strengths=[], weaknesses=[], summary="s",
                interview_questions=["a", "b", "c"], reasoning="r")
    assert CandidateAnalysis(llm_fit_score=100, **base).llm_fit_score == 100
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=101, **base)
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=1, **{**base, "interview_questions": ["a"]})
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=1, **{**base, "strengths": list("abcdef")})

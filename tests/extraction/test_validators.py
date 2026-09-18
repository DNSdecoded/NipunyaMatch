from datetime import date

from app.extraction.schemas import (
    CandidateAnalysis,
    CandidateExtraction,
    Education,
    SkillMatch,
    WorkExperience,
)
from app.extraction.validators import validate_analysis, validate_extraction, years_from_experience


def test_years_overlap_counted_once() -> None:
    exp = [
        WorkExperience(company="a", title="t", start_date="2018-01", end_date="2020-01"),
        WorkExperience(company="b", title="t", start_date="2019-01", end_date="2021-01"),
    ]
    assert years_from_experience(exp) == 3.0


def test_years_open_ended_uses_today() -> None:
    exp = [WorkExperience(company="a", title="t", start_date="2022-01", end_date=None)]
    assert years_from_experience(exp, today=date(2024, 1, 15)) == 2.0


def test_phone_and_grad_year_and_years_rules() -> None:
    e = CandidateExtraction(
        phone="12",
        education=[Education(institution="x", graduation_year=1900),
                   Education(institution="y", graduation_year=2020)],
        total_years_experience=75,
        experience=[WorkExperience(company="a", title="t", start_date="2020-01", end_date="2022-01")],
    )
    out, flags = validate_extraction(e, "", today=date(2024, 1, 1))
    assert "phone_uncertain" in flags and out.phone == "12"
    assert out.education[0].graduation_year is None and out.education[1].graduation_year == 2020
    assert out.total_years_experience == 2.0 and "years_recomputed" in flags


def test_evidence_substring_check_is_case_and_space_insensitive() -> None:
    a = CandidateAnalysis(
        llm_fit_score=50,
        skill_matches=[
            SkillMatch(skill="Python", present=True, evidence="built   PYTHON services", required=True),
            SkillMatch(skill="Go", present=True, evidence="wrote Go daily", required=False),
            SkillMatch(skill="Rust", present=False, evidence=None, required=False),
        ],
        strengths=[], weaknesses=[], summary="s", interview_questions=["a", "b", "c"], reasoning="r",
    )
    out, halluc = validate_analysis(a, "I built Python services for years.")
    assert out.skill_matches[0].present is True
    assert out.skill_matches[1].present is False
    assert halluc == 1

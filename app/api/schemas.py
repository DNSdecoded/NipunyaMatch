from typing import Any

from pydantic import BaseModel


class JobOut(BaseModel):
    id: int
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_years: float | None
    education_level: str | None


class CandidateRow(BaseModel):
    candidate_id: int
    name: str | None
    final_score: int
    recommendation: str
    matched: int
    missing: int
    years: float | None
    extraction_method: str


class SkillMatchOut(BaseModel):
    skill: str
    present: bool
    required: bool
    evidence: str | None


class CandidateDetail(CandidateRow):
    email: str | None
    phone: str | None
    location: str | None
    char_count: int
    flags: list[str]
    components: dict[str, float]
    summary: str
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]
    interview_questions: list[str]
    skill_matches: list[SkillMatchOut]


class QueryIn(BaseModel):
    question: str


_COMPONENTS = ("required_skill_coverage", "preferred_skill_coverage", "experience_fit",
               "education_fit", "llm_fit_score")


def row_from(analysis: Any) -> CandidateRow:
    c = analysis.candidate
    return CandidateRow(
        candidate_id=c.id, name=c.name, final_score=analysis.final_score,
        recommendation=analysis.recommendation,
        matched=sum(1 for m in analysis.skill_matches if m.present),
        missing=sum(1 for m in analysis.skill_matches if not m.present),
        years=c.total_years_experience, extraction_method=c.extraction_method,
    )


def detail_from(analysis: Any) -> CandidateDetail:
    c = analysis.candidate
    return CandidateDetail(
        **row_from(analysis).model_dump(),
        email=c.email, phone=c.phone, location=c.location, char_count=c.char_count,
        flags=c.flags or [],
        components={k: getattr(analysis, k) for k in _COMPONENTS},
        summary=analysis.summary, reasoning=analysis.reasoning,
        strengths=analysis.strengths, weaknesses=analysis.weaknesses,
        interview_questions=analysis.interview_questions,
        skill_matches=[SkillMatchOut(skill=m.skill.canonical_name, present=m.present,
                                     required=m.required, evidence=m.evidence)
                       for m in analysis.skill_matches],
    )

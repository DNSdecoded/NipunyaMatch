import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, Job, SkillMatchRow
from app.extraction.persist import get_or_create_skill
from app.extraction.schemas import CandidateAnalysis
from app.extraction.validators import validate_analysis
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask
from app.scoring.components import degree_level
from app.scoring.engine import ScoreBreakdown, ScoringEngine
from app.scoring.jd import job_requirements

log = logging.getLogger(__name__)
_SCORING_KEYS = ("skills", "experience", "education", "projects", "certifications",
                 "total_years_experience")


def scoring_view(extraction_json: dict[str, Any]) -> dict[str, Any]:
    return {k: extraction_json.get(k) for k in _SCORING_KEYS}


def _candidate_inputs(candidate: Candidate) -> tuple[list[str], float | None, str | None]:
    ext = candidate.extraction_json or {}
    skills = [cs.skill.canonical_name for cs in candidate.skills]
    degrees = [e.get("degree") for e in ext.get("education", [])]
    return skills, candidate.total_years_experience, degree_level(degrees)


def _upsert(
    session: Session, candidate: Candidate, job: Job, b: ScoreBreakdown,
    llm: CandidateAnalysis | None,
) -> Analysis:
    row = session.scalar(
        select(Analysis).where(Analysis.candidate_id == candidate.id, Analysis.job_id == job.id)
    )
    if row is None:
        row = Analysis(candidate_id=candidate.id, job_id=job.id, summary="", reasoning="")
        session.add(row)
    row.final_score = b.final_score
    row.required_skill_coverage = b.required_skill_coverage
    row.preferred_skill_coverage = b.preferred_skill_coverage
    row.experience_fit = b.experience_fit
    row.education_fit = b.education_fit
    row.llm_fit_score = b.llm_fit_score
    row.recommendation = b.recommendation
    row.scoring_version = b.scoring_version
    if llm is not None:
        row.summary = llm.summary
        row.reasoning = llm.reasoning
        row.strengths = llm.strengths
        row.weaknesses = llm.weaknesses
        row.interview_questions = llm.interview_questions
        session.flush()
        row.skill_matches.clear()
        session.flush()
        for m in llm.skill_matches:
            row.skill_matches.append(SkillMatchRow(
                skill=get_or_create_skill(session, m.skill), present=m.present,
                required=m.required, evidence=m.evidence,
            ))
    session.commit()
    session.refresh(row)
    return row


async def analyze_candidate(
    gateway: Gateway, session: Session, engine: ScoringEngine, candidate: Candidate, job: Job,
    fresh: bool = False,
) -> Analysis:
    reqs = job_requirements(job)
    prompt = render(
        "analyze",
        job=reqs.model_dump_json(),
        candidate=json.dumps(scoring_view(candidate.extraction_json or {})),
        resume=candidate.raw_text[:12_000],
    )
    raw = await gateway.complete(prompt, CandidateAnalysis, LLMTask.ANALYZE, fresh=fresh)
    llm, hallucinations = validate_analysis(raw, candidate.raw_text)
    if hallucinations:
        log.warning("candidate %s: %d hallucinated evidence strings", candidate.id, hallucinations)
    skills, years, degree = _candidate_inputs(candidate)
    breakdown = engine.score(skills, years, degree, llm.llm_fit_score, reqs)
    return _upsert(session, candidate, job, breakdown, llm)


def rescore_candidate(
    session: Session, engine: ScoringEngine, candidate: Candidate, job: Job
) -> Analysis:
    existing = session.scalar(
        select(Analysis).where(Analysis.candidate_id == candidate.id, Analysis.job_id == job.id)
    )
    llm_fit = existing.llm_fit_score if existing else 0.0
    skills, years, degree = _candidate_inputs(candidate)
    breakdown = engine.score(skills, years, degree, llm_fit, job_requirements(job))
    return _upsert(session, candidate, job, breakdown, None)

from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, CandidateSkill, SkillMatchRow
from app.extraction.persist import get_or_create_skill
from app.scoring.jd import JobRequirements, create_job


def _candidate(db: Session, name: str, skills: list[str], years: float, text: str) -> Candidate:
    c = Candidate(
        name=name, resume_hash=name.lower(), raw_text=text, extraction_method="pymupdf",
        char_count=len(text), total_years_experience=years, extraction_json={"skills": skills},
    )
    db.add(c)
    db.flush()
    for s in skills:
        c.skills.append(CandidateSkill(skill=get_or_create_skill(db, s), raw_mention=s))
    return c


def _analysis(db: Session, c: Candidate, job_id: int, score: int, req: float, exp: float,
              llm: float, summary: str, present: dict[str, bool]) -> Analysis:
    a = Analysis(
        candidate_id=c.id, job_id=job_id, final_score=score,
        required_skill_coverage=req, preferred_skill_coverage=0.0, experience_fit=exp,
        education_fit=100.0, llm_fit_score=llm,
        recommendation="Shortlist" if score >= 75 else "Consider" if score >= 50 else "Reject",
        summary=summary, reasoning=f"{c.name} reasoning", strengths=["s"], weaknesses=["w"],
        interview_questions=["q1", "q2", "q3"], scoring_version="1.0",
    )
    db.add(a)
    db.flush()
    for skill, ok in present.items():
        a.skill_matches.append(SkillMatchRow(
            skill=get_or_create_skill(db, skill), present=ok, required=True,
            evidence=f"uses {skill}" if ok else None,
        ))
    return a


def seed(db: Session) -> int:
    job = create_job(db, JobRequirements(
        title="Backend", required_skills=["Python", "Docker", "FastAPI"],
        preferred_skills=["AWS"], min_years=3, education_level="bachelor"), "jd text")
    alice = _candidate(db, "Alice Smith", ["Python", "FastAPI", "Docker"], 4, "Alice builds APIs")
    bob = _candidate(db, "Bob Jones", ["Python"], 6, "Bob does data pipelines")
    carol = _candidate(db, "Carol Lee", ["Java"], 1, "Carol writes Java")
    _analysis(db, alice, job.id, 82, 100.0, 100.0, 85, "Strong API engineer",
              {"Python": True, "Docker": True, "FastAPI": True})
    _analysis(db, bob, job.id, 61, 33.3, 100.0, 70, "Data engineer, no containers",
              {"Python": True, "Docker": False, "FastAPI": False})
    _analysis(db, carol, job.id, 40, 0.0, 60.0, 30, "Junior Java developer",
              {"Python": False, "Docker": False, "FastAPI": False})
    db.commit()
    return job.id

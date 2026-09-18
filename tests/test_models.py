import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, Job


def test_tables_and_indexes(engine: Engine) -> None:
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= {
        "jobs", "candidates", "skills", "candidate_skills", "job_skills",
        "experiences", "analyses", "skill_matches", "llm_calls", "llm_cache",
    }
    names = {i["name"] for i in insp.get_indexes("analyses")}
    assert "idx_analysis_score" in names
    uniques = {u["name"] for u in insp.get_unique_constraints("analyses")}
    assert "idx_analysis_pair" in uniques


def test_resume_hash_unique(db: Session) -> None:
    db.add(Candidate(resume_hash="h1", raw_text="a", extraction_method="pymupdf", char_count=1))
    db.commit()
    db.add(Candidate(resume_hash="h1", raw_text="b", extraction_method="pymupdf", char_count=1))
    with pytest.raises(IntegrityError):
        db.commit()


def test_analysis_pair_unique(db: Session) -> None:
    job = Job(title="t", raw_text="x")
    cand = Candidate(resume_hash="h2", raw_text="a", extraction_method="pymupdf", char_count=1)
    db.add_all([job, cand])
    db.commit()
    kw = dict(
        candidate_id=cand.id, job_id=job.id, final_score=1,
        required_skill_coverage=0, preferred_skill_coverage=0, experience_fit=0,
        education_fit=0, llm_fit_score=0, recommendation="Reject",
        summary="", reasoning="", scoring_version="1.0",
    )
    db.add(Analysis(**kw))
    db.commit()
    db.add(Analysis(**kw))
    with pytest.raises(IntegrityError):
        db.commit()

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.errors import ApiError
from app.db.models import Analysis, Candidate, Job, SkillMatchRow

router = APIRouter(prefix="/api")


class CandidateSummary(BaseModel):
    candidate_id: int
    name: str | None
    email: str | None
    extraction_method: str
    char_count: int
    years: float | None
    analyses: int


def _delete_analyses(db: Session, cond: object) -> int:
    ids = db.scalars(select(Analysis.id).where(cond)).all()
    if ids:
        db.execute(delete(SkillMatchRow).where(SkillMatchRow.analysis_id.in_(ids)))
        db.execute(delete(Analysis).where(Analysis.id.in_(ids)))
    return len(ids)


@router.get("/candidates")
async def list_all_candidates(db: Session = Depends(get_db)) -> list[CandidateSummary]:
    counts = dict(
        db.execute(
            select(Analysis.candidate_id, func.count()).group_by(Analysis.candidate_id)
        ).all()
    )
    return [
        CandidateSummary(
            candidate_id=c.id, name=c.name, email=c.email, extraction_method=c.extraction_method,
            char_count=c.char_count, years=c.total_years_experience, analyses=counts.get(c.id, 0),
        )
        for c in db.scalars(select(Candidate).order_by(Candidate.id.desc())).all()
    ]


@router.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: int, db: Session = Depends(get_db)) -> dict[str, int]:
    cand = db.get(Candidate, candidate_id)
    if cand is None:
        raise ApiError(404, "NOT_FOUND", f"Candidate {candidate_id} not found")
    n = _delete_analyses(db, Analysis.candidate_id == candidate_id)
    db.delete(cand)  # skills + experiences cascade
    db.commit()
    return {"deleted_candidates": 1, "deleted_analyses": n}


@router.delete("/candidates")
async def clear_candidates(db: Session = Depends(get_db)) -> dict[str, int]:
    n = _delete_analyses(db, Analysis.id.is_not(None))
    cands = db.scalars(select(Candidate)).all()
    for c in cands:
        db.delete(c)
    db.commit()
    return {"deleted_candidates": len(cands), "deleted_analyses": n}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: int, db: Session = Depends(get_db)) -> dict[str, int]:
    job = db.get(Job, job_id)
    if job is None:
        raise ApiError(404, "NOT_FOUND", f"Job {job_id} not found")
    n = _delete_analyses(db, Analysis.job_id == job_id)
    db.delete(job)  # job_skills cascade
    db.commit()
    return {"deleted_jobs": 1, "deleted_analyses": n}

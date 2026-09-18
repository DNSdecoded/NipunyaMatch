from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.batches import BATCHES, BatchStatus
from app.api.deps import get_db, get_gateway
from app.api.errors import ApiError
from app.api.schemas import CandidateDetail, detail_from
from app.db.models import Analysis, Candidate, Job
from app.llm.gateway import Gateway
from app.scoring.analyzer import analyze_candidate

router = APIRouter(prefix="/api")


@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: int, db: Session = Depends(get_db)) -> CandidateDetail:
    a = db.scalar(select(Analysis).where(Analysis.candidate_id == candidate_id)
                  .order_by(Analysis.created_at.desc()).limit(1))
    if a is None:
        raise ApiError(404, "NOT_FOUND", f"Candidate {candidate_id} not found")
    return detail_from(a)


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> BatchStatus:
    if batch_id not in BATCHES:
        raise ApiError(404, "NOT_FOUND", f"Batch {batch_id} not found")
    return BATCHES[batch_id]


@router.post("/candidates/{candidate_id}/reanalyze")
async def reanalyze(
    candidate_id: int, job_id: int, request: Request, db: Session = Depends(get_db),
    gateway: Gateway = Depends(get_gateway),
) -> CandidateDetail:
    """Fresh LLM analysis (cache bypassed) + deterministic rescoring for one candidate."""
    cand, job = db.get(Candidate, candidate_id), db.get(Job, job_id)
    if cand is None or job is None:
        raise ApiError(404, "NOT_FOUND", f"Candidate {candidate_id} or job {job_id} not found")
    state = request.app.state
    async with state.llm_semaphore:
        a = await analyze_candidate(gateway, db, state.engine, cand, job, fresh=True)
    return detail_from(a)

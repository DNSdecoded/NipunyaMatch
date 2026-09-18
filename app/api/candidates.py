from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.batches import BATCHES, BatchStatus
from app.api.deps import get_db
from app.api.errors import ApiError
from app.api.schemas import CandidateDetail, detail_from
from app.db.models import Analysis

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

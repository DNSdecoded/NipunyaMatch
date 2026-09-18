import csv
import io
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.batches import new_batch, process_batch
from app.api.deps import forbid_in_demo, get_db, get_gateway
from app.api.errors import ApiError
from app.api.schemas import CandidateRow, JobOut, QueryIn, row_from
from app.db.models import Analysis, Candidate, CandidateSkill, ChatTurn, Job, Skill
from app.llm.gateway import Gateway
from app.parsing.pipeline import parse_pdf, parse_text
from app.parsing.validate import InvalidPDF
from app.query.schemas import QueryResponse
from app.query.service import answer_question
from app.scoring.analyzer import rescore_candidate
from app.scoring.jd import create_job, job_requirements, parse_jd

router = APIRouter(prefix="/api/jobs")
MAX_FILES = 50
DEMO_MAX_FILES = 10
DEMO_MAX_BYTES = 5 * 1024 * 1024


def _job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise ApiError(404, "NOT_FOUND", f"Job {job_id} not found")
    return job


def _job_out(job: Job) -> JobOut:
    r = job_requirements(job)
    return JobOut(id=job.id, title=r.title, required_skills=r.required_skills,
                  preferred_skills=r.preferred_skills, min_years=r.min_years,
                  education_level=r.education_level)


def _ranked(job_id: int) -> Select[tuple[Analysis]]:
    return (select(Analysis).join(Candidate).where(Analysis.job_id == job_id)
            .order_by(Analysis.final_score.desc(), Candidate.name))


@router.post("", status_code=201)
async def create(
    db: Session = Depends(get_db), gateway: Gateway = Depends(get_gateway),
    text: str | None = Form(None), file: UploadFile | None = File(None),
) -> JobOut:
    if file is not None:
        try:
            parsed = parse_pdf(await file.read())
        except InvalidPDF as e:
            raise ApiError(413 if e.code == "FILE_TOO_LARGE" else 400, e.code, e.message) from e
    elif text:
        parsed = parse_text(text)
    else:
        raise ApiError(400, "MISSING_JD", "Provide a job description as text or a PDF file.")
    reqs = await parse_jd(gateway, parsed.text)
    return _job_out(create_job(db, reqs, parsed.text))


@router.get("")
async def list_jobs(db: Session = Depends(get_db)) -> list[JobOut]:
    return [_job_out(j) for j in db.scalars(select(Job).order_by(Job.id.desc())).all()]


@router.get("/{job_id}")
async def get_job(job_id: int, db: Session = Depends(get_db)) -> JobOut:
    return _job_out(_job_or_404(db, job_id))


@router.post("/{job_id}/resumes", status_code=202)
async def upload_resumes(
    job_id: int, request: Request, background: BackgroundTasks,
    files: list[UploadFile] = File(...), db: Session = Depends(get_db),
    gateway: Gateway = Depends(get_gateway),
) -> dict[str, str]:
    demo = request.app.state.settings.demo_mode
    max_files = DEMO_MAX_FILES if demo else MAX_FILES
    if len(files) > max_files:
        raise ApiError(400, "TOO_MANY_FILES", f"Upload at most {max_files} resumes per batch.")
    _job_or_404(db, job_id)
    payload = [(f.filename or "resume.pdf", await f.read()) for f in files]
    if demo and any(len(data) >= DEMO_MAX_BYTES for _, data in payload):
        raise ApiError(413, "FILE_TOO_LARGE", "Demo limit is 5 MB per resume.")
    batch = new_batch(job_id, [n for n, _ in payload])
    background.add_task(process_batch, request.app.state, gateway, batch.id, payload)
    return {"batch_id": batch.id}


@router.get("/{job_id}/candidates")
async def list_candidates(
    job_id: int, db: Session = Depends(get_db),
    min_score: int = 0, skill: str | None = None, limit: int = 50,
) -> list[CandidateRow]:
    _job_or_404(db, job_id)
    q = _ranked(job_id).where(Analysis.final_score >= min_score)
    if skill:
        has = (select(CandidateSkill.candidate_id).join(Skill)
               .where(Skill.canonical_name.ilike(skill)))
        q = q.where(Candidate.id.in_(has))
    return [row_from(a) for a in db.scalars(q.limit(min(limit, 50))).all()]


@router.post("/{job_id}/query")
async def query(
    job_id: int, body: QueryIn, request: Request, db: Session = Depends(get_db),
    gateway: Gateway = Depends(get_gateway),
) -> QueryResponse:
    _job_or_404(db, job_id)
    state = request.app.state
    r = await answer_question(gateway, db, job_id, body.question, state.embedder)
    db.add(ChatTurn(job_id=job_id, question=body.question, answer=r.answer, intent=r.intent.value,
                    provider=r.provider_used, sources=[s.model_dump() for s in r.sources]))
    db.commit()
    return r


@router.get("/{job_id}/chat")
async def chat_history(job_id: int, db: Session = Depends(get_db)) -> list[dict[str, object]]:
    """Persisted conversation for this job, oldest first."""
    _job_or_404(db, job_id)
    q = select(ChatTurn).where(ChatTurn.job_id == job_id).order_by(ChatTurn.id)
    turns = db.scalars(q).all()
    return [{"q": t.question, "a": t.answer, "intent": t.intent, "provider": t.provider,
             "sources": t.sources, "at": t.created_at.isoformat()} for t in turns]


@router.post("/{job_id}/rescore", dependencies=[Depends(forbid_in_demo)])
async def rescore(job_id: int, request: Request, db: Session = Depends(get_db)) -> dict[str, int]:
    job = _job_or_404(db, job_id)
    engine = request.app.state.engine
    cands = db.scalars(select(Candidate).join(Analysis).where(Analysis.job_id == job_id)).all()
    for c in cands:
        rescore_candidate(db, engine, c, job)
    return {"rescored": len(cands)}


@router.get("/{job_id}/export")
async def export(
    job_id: int, db: Session = Depends(get_db), format: Literal["csv", "xlsx"] = "csv"
) -> Response:
    _job_or_404(db, job_id)
    rows = [row_from(a).model_dump() for a in db.scalars(_ranked(job_id)).all()]
    headers = list(CandidateRow.model_fields)
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=candidates.csv"})
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])
    out = io.BytesIO()
    wb.save(out)
    return Response(
        out.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=candidates.xlsx"},
    )

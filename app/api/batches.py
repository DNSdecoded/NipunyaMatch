import asyncio
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from app.db.models import Job
from app.extraction.extractor import extract_candidate
from app.extraction.persist import persist_candidate, resume_hash
from app.llm.gateway import Gateway
from app.llm.types import ExtractionFailed, NoProvider, QuotaExhausted
from app.parsing.pipeline import parse_pdf
from app.parsing.validate import InvalidPDF
from app.scoring.analyzer import analyze_candidate

log = logging.getLogger(__name__)


class FileStatus(BaseModel):
    filename: str
    status: Literal["queued", "processing", "done", "error"] = "queued"
    extraction_method: str | None = None
    error: str | None = None
    candidate_id: int | None = None


class BatchStatus(BaseModel):
    id: str
    job_id: int
    total: int
    done: int = 0
    files: list[FileStatus]
    llm_calls: int = 0


BATCHES: dict[str, BatchStatus] = {}  # ponytail: in-process; single user, single worker


def new_batch(job_id: int, filenames: list[str]) -> BatchStatus:
    b = BatchStatus(id=uuid.uuid4().hex, job_id=job_id, total=len(filenames),
                    files=[FileStatus(filename=f) for f in filenames])
    BATCHES[b.id] = b
    return b


def _friendly(e: Exception) -> str:
    if isinstance(e, QuotaExhausted):
        return "LLM quota exhausted on both providers. Retry later."
    if isinstance(e, NoProvider):
        return "No LLM provider reachable. Check API keys in .env."
    if isinstance(e, ExtractionFailed):
        return "The model could not produce a valid extraction for this resume."
    return "Unexpected error while processing this resume."


async def _process_one(
    state: Any, gateway: Gateway, batch: BatchStatus, fs: FileStatus, data: bytes
) -> None:
    fs.status = "processing"
    try:
        parsed = await asyncio.to_thread(parse_pdf, data)
        fs.extraction_method = parsed.extraction_method
        async with state.llm_semaphore:
            ext, flags = await extract_candidate(gateway, parsed.text)
        with state.session_factory() as s:
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            job = s.get(Job, batch.job_id)
            assert job is not None
            async with state.llm_semaphore:
                await analyze_candidate(gateway, s, state.engine, cand, job)
            fs.candidate_id = cand.id
        fs.status = "done"
    except InvalidPDF as e:
        fs.status, fs.error = "error", e.message
    except Exception as e:  # one failed resume never blocks the batch
        log.exception("resume %s failed", fs.filename)
        fs.status, fs.error = "error", _friendly(e)
    finally:
        batch.done += 1
        batch.llm_calls = gateway.call_count


async def process_batch(
    state: Any, gateway: Gateway, batch_id: str, files: list[tuple[str, bytes]]
) -> None:
    batch = BATCHES[batch_id]
    pairs = zip(batch.files, files, strict=True)
    await asyncio.gather(
        *(_process_one(state, gateway, batch, fs, data) for fs, (_, data) in pairs)
    )

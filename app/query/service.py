import time

from sqlalchemy.orm import Session

from app.llm.gateway import Gateway
from app.query.classifier import classify
from app.query.handlers import run_handler
from app.query.schemas import Intent, QueryResponse, Source
from app.query.synthesis import synthesize
from app.scoring.embeddings import Embedder


async def answer_question(
    gateway: Gateway, session: Session, job_id: int, question: str, embedder: Embedder
) -> QueryResponse:
    started = time.monotonic()
    plan = await classify(gateway, question)
    if plan.intent == Intent.SEMANTIC:
        plan.skill = question
    result = run_handler(session, job_id, plan, embedder)
    if not result.rows:
        answer = result.facts.get("error") or "No candidates match that question for this job."
    else:
        answer = await synthesize(gateway, question, plan, result)
    return QueryResponse(
        answer=answer,
        intent=plan.intent,
        sources=[Source(candidate_id=r["candidate_id"], name=r["name"], score=r["final_score"])
                 for r in result.rows],
        latency_ms=int((time.monotonic() - started) * 1000),
        provider_used=str(gateway.last_provider or "gemini"),
    )

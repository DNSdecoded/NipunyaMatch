from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Select, func, select, text
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, CandidateSkill, Skill
from app.query.schemas import Intent, QueryPlan
from app.scoring.embeddings import Embedder, cosine
from app.scoring.engine import load_config

ROW_CAP = 50
_WEIGHTS = load_config(Path("config/scoring.yaml")).weights
_COMPONENTS = list(_WEIGHTS)


@contextmanager
def read_only(session: Session) -> Iterator[None]:
    """Query-path guard: no writes on this connection while the handler runs.
    Reset in finally because the PRAGMA sticks to the pooled connection."""
    session.execute(text("PRAGMA query_only = ON"))
    try:
        yield
    finally:
        session.execute(text("PRAGMA query_only = OFF"))


@dataclass
class HandlerResult:
    rows: list[dict[str, Any]]
    facts: dict[str, Any] = field(default_factory=dict)


def _row(a: Analysis) -> dict[str, Any]:
    c = a.candidate
    return {
        "candidate_id": c.id, "name": c.name, "final_score": a.final_score,
        "recommendation": a.recommendation, "years": c.total_years_experience,
        "summary": a.summary, "reasoning": a.reasoning,
        "skills_present": sorted(m.skill.canonical_name for m in a.skill_matches if m.present),
        "skills_missing": sorted(m.skill.canonical_name for m in a.skill_matches if not m.present),
        **{k: getattr(a, k) for k in _COMPONENTS},
    }


def _ranked(job_id: int) -> Select[tuple[Analysis]]:
    return (
        select(Analysis).join(Candidate).where(Analysis.job_id == job_id)
        .order_by(Analysis.final_score.desc(), Candidate.name)
    )


def find_candidate(session: Session, job_id: int, name: str) -> Analysis | None:
    needle = name.strip().lower()
    if not needle:
        return None
    return session.scalar(
        _ranked(job_id).where(func.lower(Candidate.name).contains(needle)).limit(1)
    )


def explain_facts(a: Analysis, b: Analysis) -> dict[str, Any]:
    diffs = []
    for comp in _COMPONENTS:
        va, vb = float(getattr(a, comp)), float(getattr(b, comp))
        diffs.append({"component": comp, "a": va, "b": vb,
                      "weighted_gap": round(_WEIGHTS[comp] * (va - vb), 1)})
    diffs.sort(key=lambda d: abs(d["weighted_gap"]), reverse=True)
    pa = {m.skill.canonical_name for m in a.skill_matches if m.present}
    pb = {m.skill.canonical_name for m in b.skill_matches if m.present}
    ma = {m.skill.canonical_name for m in a.skill_matches if not m.present}
    mb = {m.skill.canonical_name for m in b.skill_matches if not m.present}
    return {
        "a": {"name": a.candidate.name, "score": a.final_score,
              "years": a.candidate.total_years_experience},
        "b": {"name": b.candidate.name, "score": b.final_score,
              "years": b.candidate.total_years_experience},
        "score_gap": a.final_score - b.final_score,
        "diffs": diffs,
        "a_only_skills": sorted(pa - pb),
        "b_only_skills": sorted(pb - pa),
        "both_missing": sorted(ma & mb),
    }


def run_handler(
    session: Session, job_id: int, plan: QueryPlan, embedder: Embedder
) -> HandlerResult:
    with read_only(session):
        return _dispatch(session, job_id, plan, embedder)


def _dispatch(
    session: Session, job_id: int, plan: QueryPlan, embedder: Embedder
) -> HandlerResult:
    n = min(plan.n, ROW_CAP)

    if plan.intent in (Intent.TOP_N, Intent.RECOMMEND):
        rows = session.scalars(_ranked(job_id).limit(n)).all()
        return HandlerResult([_row(a) for a in rows])

    if plan.intent == Intent.FILTER_SKILL and plan.skill:
        skill_ids = select(Skill.id).where(
            func.lower(Skill.canonical_name) == plan.skill.strip().lower()
        )
        has = select(CandidateSkill.candidate_id).where(CandidateSkill.skill_id.in_(skill_ids))
        cond = Candidate.id.in_(has) if plan.has_skill else Candidate.id.not_in(has)
        rows = session.scalars(_ranked(job_id).where(cond).limit(ROW_CAP)).all()
        return HandlerResult([_row(a) for a in rows], {"skill": plan.skill, "has": plan.has_skill})

    if (
        plan.intent == Intent.FILTER_ATTRIBUTE
        and plan.attribute and plan.op and plan.value is not None
    ):
        col = {"total_years_experience": Candidate.total_years_experience,
               "final_score": Analysis.final_score}[plan.attribute]  # whitelist
        cond = {">": col > plan.value, ">=": col >= plan.value, "<": col < plan.value,
                "<=": col <= plan.value, "=": col == plan.value}[plan.op]
        rows = session.scalars(_ranked(job_id).where(cond).limit(ROW_CAP)).all()
        return HandlerResult([_row(a) for a in rows],
                             {"attribute": plan.attribute, "op": plan.op, "value": plan.value})

    if plan.intent in (Intent.COMPARE, Intent.EXPLAIN_RANKING):
        a = find_candidate(session, job_id, plan.candidate_a or "")
        b = find_candidate(session, job_id, plan.candidate_b or "")
        missing = [nm for nm, x in ((plan.candidate_a, a), (plan.candidate_b, b)) if x is None]
        if missing or a is None or b is None:
            return HandlerResult(
                [], {"error": f"No candidate matching: {', '.join(str(m) for m in missing)}"}
            )
        if a.final_score < b.final_score:
            a, b = b, a
        return HandlerResult([_row(a), _row(b)], explain_facts(a, b))

    # semantic, and any plan whose parameters were missing
    rows = list(session.scalars(_ranked(job_id).limit(ROW_CAP)).all())
    if plan.skill:  # question text, injected by the service
        vecs = embedder.embed([plan.skill] + [a.summary or "" for a in rows])
        rows = [a for _, a in sorted(
            ((cosine(vecs[0], vecs[i + 1]), a) for i, a in enumerate(rows)),
            key=lambda x: x[0], reverse=True,
        )]
    return HandlerResult([_row(a) for a in rows[:n]], {"mode": "semantic"})

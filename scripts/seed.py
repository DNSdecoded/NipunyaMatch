"""Populate the DB from golden labels without any LLM call."""
import json
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Analysis
from app.db.session import init_db, make_engine, make_session_factory
from app.extraction.persist import persist_candidate, resume_hash
from app.extraction.schemas import CandidateExtraction, Education, WorkExperience
from app.extraction.validators import validate_extraction
from app.parsing.pipeline import ParsedDocument, parse_pdf, parse_text
from app.scoring.analyzer import rescore_candidate
from app.scoring.embeddings import LocalEmbedder
from app.scoring.engine import build_engine
from app.scoring.jd import JobRequirements, create_job

FIX = Path("tests/fixtures")


def _parse(data: bytes, lab: dict[str, object]) -> ParsedDocument:
    try:
        return parse_pdf(data)
    except Exception:  # ponytail: no Tesseract on this box -> seed from labels, mark as text
        text = f"{lab['name']} {lab['email']}\nSKILLS\n{', '.join(lab['skills'])}"  # type: ignore[arg-type]
        return parse_text(text)


def main() -> None:
    settings = get_settings()
    db_engine = make_engine(settings.database_url)
    init_db(db_engine)
    factory = make_session_factory(db_engine)
    engine = build_engine(LocalEmbedder())
    labels = json.loads((FIX / "golden/labels.json").read_text())
    with factory() as s:
        jd = parse_text((FIX / "jd.txt").read_text())
        job = create_job(s, JobRequirements(
            title="Senior Backend Engineer",
            required_skills=["Python", "FastAPI", "Docker", "PostgreSQL", "Machine Learning"],
            preferred_skills=["AWS", "Kubernetes"], min_years=3, education_level="bachelor",
        ), jd.text)
        for fname, lab in labels.items():
            data = (FIX / "resumes" / fname).read_bytes()
            parsed = _parse(data, lab)
            ext = CandidateExtraction(
                name=lab["name"], email=lab["email"], phone=lab["phone"], skills=lab["skills"],
                experience=[WorkExperience(company="Example Corp", title="Software Engineer",
                                           start_date=lab["start_date"], end_date=lab["end_date"])],
                education=[Education(institution="Example University", degree=lab["degree"])],
            )
            ext, flags = validate_extraction(ext, parsed.text)
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            s.add(Analysis(
                candidate_id=cand.id, job_id=job.id, final_score=0,
                required_skill_coverage=0, preferred_skill_coverage=0, experience_fit=0,
                education_fit=0, llm_fit_score=lab["llm_fit_score"], recommendation="Reject",
                summary="Seeded from golden labels (no LLM call)",
                reasoning="Deterministic components only; run scripts/evaluate.py for LLM analysis.",
                scoring_version="seed",
            ))
            s.commit()
            rescore_candidate(s, engine, cand, job)
        rows = s.scalars(select(Analysis).where(Analysis.job_id == job.id)
                         .order_by(Analysis.final_score.desc())).all()
        for r in rows:
            print(f"{r.final_score:3d}  {r.recommendation:9s}  {r.candidate.name}")
        print(f"\nSeeded job {job.id} with {len(rows)} candidates.")


if __name__ == "__main__":
    main()

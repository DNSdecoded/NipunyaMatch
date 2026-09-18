import hashlib

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Candidate, CandidateSkill, Experience, Skill
from app.extraction.schemas import CandidateExtraction
from app.parsing.pipeline import ParsedDocument


def resume_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_or_create_skill(session: Session, name: str) -> Skill:
    clean = name.strip()
    skill = session.scalar(
        select(Skill).where(func.lower(Skill.canonical_name) == clean.lower())
    )
    if skill is None:
        skill = Skill(canonical_name=clean)
        session.add(skill)
        session.flush()
    return skill


def persist_candidate(
    session: Session,
    parsed: ParsedDocument,
    ext: CandidateExtraction,
    flags: list[str],
    rhash: str,
) -> Candidate:
    cand = session.scalar(select(Candidate).where(Candidate.resume_hash == rhash))
    if cand is None:
        cand = Candidate(resume_hash=rhash, raw_text="", extraction_method="", char_count=0)
        session.add(cand)
    cand.name = ext.name
    cand.email = str(ext.email) if ext.email else None
    cand.phone = ext.phone
    cand.location = ext.location
    cand.total_years_experience = ext.total_years_experience
    cand.raw_text = parsed.text
    cand.extraction_method = parsed.extraction_method
    cand.char_count = parsed.char_count
    cand.extraction_json = ext.model_dump(mode="json")
    cand.flags = flags
    session.flush()

    cand.skills.clear()
    cand.experiences.clear()
    session.flush()
    seen: set[int] = set()
    for s in ext.skills:
        skill = get_or_create_skill(session, s)
        if skill.id in seen:
            continue
        seen.add(skill.id)
        cand.skills.append(CandidateSkill(skill=skill, raw_mention=s))
    for x in ext.experience:
        cand.experiences.append(
            Experience(
                company=x.company, title=x.title, start_date=x.start_date,
                end_date=x.end_date, description=x.description,
            )
        )
    session.commit()
    session.refresh(cand)
    return cand

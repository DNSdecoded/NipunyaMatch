from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Candidate, Skill
from app.extraction.persist import persist_candidate, resume_hash
from app.extraction.schemas import CandidateExtraction, WorkExperience
from app.parsing.pipeline import ParsedDocument


def _parsed() -> ParsedDocument:
    return ParsedDocument("text", "pymupdf", 4, {})


def test_upsert_on_hash_replaces_children(db: Session) -> None:
    h = resume_hash(b"pdf")
    e1 = CandidateExtraction(
        name="A", skills=["Python", "SQL"],
        experience=[WorkExperience(company="c", title="t")],
    )
    c1 = persist_candidate(db, _parsed(), e1, [], h)
    e2 = CandidateExtraction(name="A2", skills=["python", "Go"], experience=[])
    c2 = persist_candidate(db, _parsed(), e2, ["phone_uncertain"], h)
    assert c1.id == c2.id and c2.name == "A2" and c2.flags == ["phone_uncertain"]
    assert db.scalar(select(func.count()).select_from(Candidate)) == 1
    assert sorted(cs.skill.canonical_name for cs in c2.skills) == ["Go", "Python"]
    assert c2.experiences == []
    assert db.scalar(select(func.count()).select_from(Skill)) == 3  # Python, SQL, Go

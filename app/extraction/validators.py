import logging
import re
from datetime import date

from app.extraction.schemas import CandidateAnalysis, CandidateExtraction, WorkExperience

log = logging.getLogger(__name__)
_YM = re.compile(r"^(\d{4})-(\d{2})$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _months(ym: str) -> int | None:
    m = _YM.match(ym)
    return int(m.group(1)) * 12 + int(m.group(2)) - 1 if m else None


def years_from_experience(exp: list[WorkExperience], today: date | None = None) -> float | None:
    today = today or date.today()
    now = today.year * 12 + today.month - 1
    covered: set[int] = set()
    for e in exp:
        start = _months(e.start_date) if e.start_date else None
        if start is None:
            continue
        end = (_months(e.end_date) if e.end_date else now) or now
        covered.update(range(start, max(start, end)))
    return round(len(covered) / 12, 1) if covered else None


def validate_extraction(
    e: CandidateExtraction, resume_text: str, today: date | None = None
) -> tuple[CandidateExtraction, list[str]]:
    flags: list[str] = []
    out = e.model_copy(deep=True)
    if out.phone and not 7 <= len(re.sub(r"\D", "", out.phone)) <= 15:
        flags.append("phone_uncertain")
    if out.email and norm(str(out.email)) not in norm(resume_text):
        flags.append("email_uncertain")
    max_year = (today or date.today()).year + 6
    for ed in out.education:
        if ed.graduation_year is not None and not 1950 <= ed.graduation_year <= max_year:
            ed.graduation_year = None
    if out.total_years_experience is None or out.total_years_experience >= 60:
        recomputed = years_from_experience(out.experience, today)
        if recomputed != out.total_years_experience:
            flags.append("years_recomputed")
        out.total_years_experience = recomputed
    out.skills = list(dict.fromkeys(s.strip() for s in out.skills if s.strip()))
    return out, flags


def validate_analysis(a: CandidateAnalysis, resume_text: str) -> tuple[CandidateAnalysis, int]:
    haystack = norm(resume_text)
    out = a.model_copy(deep=True)
    hallucinations = 0
    for m in out.skill_matches:
        if m.present and (not m.evidence or norm(m.evidence) not in haystack):
            m.present = False
            hallucinations += 1
            log.info("hallucination: skill=%s evidence=%r", m.skill, m.evidence)
    return out, hallucinations

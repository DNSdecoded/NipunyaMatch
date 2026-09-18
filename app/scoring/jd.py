from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Job, JobSkill
from app.extraction.persist import get_or_create_skill
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask

EducationLevel = Literal["high_school", "associate", "bachelor", "master", "phd"]


class JobRequirements(BaseModel):
    title: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years: float | None = None
    education_level: EducationLevel | None = None


async def parse_jd(gateway: Gateway, text: str) -> JobRequirements:
    return await gateway.complete(render("jd", jd=text[:12_000]), JobRequirements, LLMTask.ANALYZE)


def create_job(session: Session, reqs: JobRequirements, raw_text: str) -> Job:
    job = Job(
        title=reqs.title, raw_text=raw_text,
        min_years=reqs.min_years, education_level=reqs.education_level,
    )
    session.add(job)
    session.flush()
    seen: set[int] = set()
    pairs = [(s, True) for s in reqs.required_skills] + [(s, False) for s in reqs.preferred_skills]
    for name, required in pairs:
        skill = get_or_create_skill(session, name)
        if skill.id in seen:
            continue
        seen.add(skill.id)
        job.skills.append(JobSkill(skill=skill, is_required=required))
    session.commit()
    session.refresh(job)
    return job


def job_requirements(job: Job) -> JobRequirements:
    return JobRequirements(
        title=job.title,
        required_skills=[js.skill.canonical_name for js in job.skills if js.is_required],
        preferred_skills=[js.skill.canonical_name for js in job.skills if not js.is_required],
        min_years=job.min_years,
        education_level=job.education_level,  # type: ignore[arg-type]
    )

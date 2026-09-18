from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    raw_text: Mapped[str] = mapped_column(Text)
    min_years: Mapped[float | None] = mapped_column(Float)
    education_level: Mapped[str | None] = mapped_column(String(50))
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    skills: Mapped[list[JobSkill]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[Analysis]] = relationship(back_populates="job")


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(50))
    location: Mapped[str | None] = mapped_column(String(200))
    total_years_experience: Mapped[float | None] = mapped_column(Float)
    resume_hash: Mapped[str] = mapped_column(String(64))
    extraction_method: Mapped[str] = mapped_column(String(20))
    char_count: Mapped[int] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text)
    extraction_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    skills: Mapped[list[CandidateSkill]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    experiences: Mapped[list[Experience]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[Analysis]] = relationship(back_populates="candidate")
    __table_args__ = (Index("idx_cand_hash", "resume_hash", unique=True),)


class Skill(Base):
    __tablename__ = "skills"
    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(200), unique=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)


class CandidateSkill(Base):
    __tablename__ = "candidate_skills"
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"), primary_key=True)
    raw_mention: Mapped[str] = mapped_column(String(200))
    evidence: Mapped[str | None] = mapped_column(Text)
    candidate: Mapped[Candidate] = relationship(back_populates="skills")
    skill: Mapped[Skill] = relationship()
    __table_args__ = (
        Index("idx_cs_skill", "skill_id"),
        Index("idx_cs_candidate", "candidate_id"),
    )


class JobSkill(Base):
    __tablename__ = "job_skills"
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"), primary_key=True)
    is_required: Mapped[bool] = mapped_column(Boolean)
    job: Mapped[Job] = relationship(back_populates="skills")
    skill: Mapped[Skill] = relationship()


class Experience(Base):
    __tablename__ = "experiences"
    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    company: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(200))
    start_date: Mapped[str | None] = mapped_column(String(7))
    end_date: Mapped[str | None] = mapped_column(String(7))
    description: Mapped[str | None] = mapped_column(Text)
    candidate: Mapped[Candidate] = relationship(back_populates="experiences")
    __table_args__ = (Index("idx_exp_candidate", "candidate_id"),)


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    final_score: Mapped[int] = mapped_column(Integer)
    required_skill_coverage: Mapped[float] = mapped_column(Float)
    preferred_skill_coverage: Mapped[float] = mapped_column(Float)
    experience_fit: Mapped[float] = mapped_column(Float)
    education_fit: Mapped[float] = mapped_column(Float)
    llm_fit_score: Mapped[float] = mapped_column(Float)
    recommendation: Mapped[str] = mapped_column(String(20))
    summary: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str] = mapped_column(Text)
    strengths: Mapped[list[str]] = mapped_column(JSON, default=list)
    weaknesses: Mapped[list[str]] = mapped_column(JSON, default=list)
    interview_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    scoring_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    candidate: Mapped[Candidate] = relationship(back_populates="analyses")
    job: Mapped[Job] = relationship(back_populates="analyses")
    skill_matches: Mapped[list[SkillMatchRow]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    __table_args__ = (
        UniqueConstraint("candidate_id", "job_id", name="idx_analysis_pair"),
        Index("idx_analysis_score", "job_id", "final_score"),
    )


class SkillMatchRow(Base):
    __tablename__ = "skill_matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"))
    present: Mapped[bool] = mapped_column(Boolean)
    required: Mapped[bool] = mapped_column(Boolean)
    evidence: Mapped[str | None] = mapped_column(Text)
    analysis: Mapped[Analysis] = relationship(back_populates="skill_matches")
    skill: Mapped[Skill] = relationship()


class LLMCall(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(100))
    task: Mapped[str] = mapped_column(String(20))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class LLMCache(Base):
    __tablename__ = "llm_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    response_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

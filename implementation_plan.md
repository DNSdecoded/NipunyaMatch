# NipunyaMatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, single-user recruitment assistant that parses resume PDFs, scores candidates against a job description with an auditable hybrid rubric, persists everything in SQLite, and answers recruiter questions with cited reasoning.

**Architecture:** Six layers with Pydantic boundaries: Streamlit UI → FastAPI → parsing → extraction (LLM call 1, JD-independent) → scoring (deterministic components + LLM call 2) → SQLite. A single LLM gateway owns Gemini (primary) and OpenRouter (fallback) with cache, retry, breaker and audit log. A natural-language query layer routes questions to typed SQL handlers, never text-to-SQL.

**Tech Stack:** Python 3.11+, uv, FastAPI, Streamlit, SQLAlchemy 2.0 + SQLite, Pydantic v2, pydantic-settings, httpx, PyMuPDF, pytesseract, sentence-transformers (all-MiniLM-L6-v2), numpy, PyYAML, pytest, pytest-asyncio, respx, ruff, mypy.

**Spec:** `docs/specs.md` (contracts). Rationale in `NipunyaMatch-spec.md`. Section numbers below (§N) refer to `docs/specs.md`.

## Global Constraints

- Python `>=3.11`. Package manager `uv`. All commands run as `uv run <cmd>`.
- Nothing outside `app/llm/` imports a provider client or reads `GEMINI_API_KEY` / `OPENROUTER_API_KEY`.
- Model IDs only from `Settings` (§12): `GEMINI_MODEL_EXTRACT=gemini-3.5-flash-lite`, `GEMINI_MODEL_ANALYZE=gemini-3.8-flash`, `GEMINI_MODEL_QUERY=gemini-3.5-flash`, `OPENROUTER_MODEL=google/gemini-3.8-flash`. Never `-latest`, never `gemini-2.0-*`.
- Cache key `sha256(prompt + model + schema_version)`. Retry Gemini on 429/5xx: 1s, 2s, 4s, 3 attempts. Breaker: 5 consecutive failures → open 60s → half-open one probe.
- Resume text truncated to 12,000 characters before extraction.
- PDF limits: `%PDF` magic, `< 20` pages, `< 10 MB`, not encrypted. OCR when a page has `< 100` characters.
- Scoring weights `0.35 / 0.15 / 0.20 / 0.10 / 0.20` from `config/scoring.yaml`; `scoring_version` stored on every analysis. Embedding match threshold cosine `> 0.75` credits `0.8`. Bands 75/50. Missing `> 50%` required skills caps at `Consider`.
- Worked example (§7.4) must score exactly `72` / `Consider`.
- Query path: parameterised SQL, whitelisted columns, read-only connection, 50-row cap.
- Tests never touch the network. Providers mocked with `respx`. `pytest` passes with no `.env`.
- Prompts are `.txt` files in `app/llm/prompts/`, loaded by name.
- Scoring prompt never receives name, email, phone, location, linkedin_url, age, gender markers.
- Every task ends with a conventional commit; tag `v0.<phase>` at each phase end.
- `ponytail:` embeddings stored as float32 BLOB columns and compared with numpy cosine instead of `sqlite-vec` (≤50 candidates per batch; switch to `sqlite-vec` if candidate count exceeds ~1k).
- `ponytail:` batch progress kept in an in-process dict (single-user, single-process; move to a `batches` table if a second worker ever appears).
- `ponytail:` `Base.metadata.create_all` instead of Alembic; add Alembic at the first schema change on a live DB.
- All async. LLM clients use `httpx.AsyncClient`. Tests use `pytest-asyncio` in auto mode.

---

# Phase 0 — Skeleton

### Task 0: Repository scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`, `app/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `config/scoring.yaml`

**Interfaces:**
- Produces: `uv run pytest`, `uv run ruff check .`, `uv run mypy app/llm app/scoring` runnable. `config/scoring.yaml` keys consumed by Task 23.

- [ ] **Step 1: Initialise git and uv**

```bash
cd F:/NipunyaMatch
git init -b main
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "nipunyamatch"
version = "0.1.0"
description = "AI-powered recruitment assistant"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "streamlit>=1.38",
    "sqlalchemy>=2.0",
    "pydantic>=2.8",
    "pydantic-settings>=2.4",
    "email-validator>=2.2",
    "httpx>=0.27",
    "pymupdf>=1.24",
    "pytesseract>=0.3.13",
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "python-multipart>=0.0.9",
    "openpyxl>=3.1",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "respx>=0.21",
    "ruff>=0.6",
    "mypy>=1.11",
    "types-PyYAML",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.coverage.run]
source = ["app"]
omit = ["app/ui/*"]
```

- [ ] **Step 3: Write `.gitignore` and `.env.example`**

`.gitignore`:
```
.env
.venv/
__pycache__/
*.pyc
data/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
```

`.env.example`:
```
GEMINI_API_KEY=
OPENROUTER_API_KEY=
GEMINI_MODEL_EXTRACT=gemini-3.5-flash-lite
GEMINI_MODEL_ANALYZE=gemini-3.8-flash
GEMINI_MODEL_QUERY=gemini-3.5-flash
OPENROUTER_MODEL=google/gemini-3.8-flash
DATABASE_URL=sqlite:///./data/recruiter.db
EMBEDDING_BACKEND=local
MAX_CONCURRENT_LLM=4
LOG_LEVEL=INFO
```

- [ ] **Step 4: Write `config/scoring.yaml`**

```yaml
version: "1.0"
weights:
  required_skill_coverage: 0.35
  preferred_skill_coverage: 0.15
  experience_fit: 0.20
  education_fit: 0.10
  llm_fit_score: 0.20
embedding_threshold: 0.75
embedding_credit: 0.8
bands:
  shortlist: 75
  consider: 50
```

- [ ] **Step 5: Write `Makefile`**

```makefile
.PHONY: dev test lint seed ui

dev:
	uv run uvicorn app.main:app --reload --port 8000

ui:
	uv run streamlit run app/ui/Home.py

test:
	uv run pytest --cov --cov-fail-under=70

lint:
	uv run ruff check . && uv run mypy app/llm app/scoring

seed:
	uv run python scripts/seed.py
```

- [ ] **Step 6: Create packages and a smoke test**

```bash
mkdir -p app tests config
touch app/__init__.py tests/__init__.py
```

`tests/test_smoke.py`:
```python
def test_imports() -> None:
    import app  # noqa: F401
```

- [ ] **Step 7: Install and run**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold project with uv, ruff, mypy, pytest"
```

---

### Task 1: Typed settings

**Files:**
- Create: `app/config.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `class Settings(BaseSettings)` with fields exactly as `.env.example` plus `gemini_base_url`, `openrouter_base_url`; `get_settings() -> Settings` (cached); conftest fixture `settings`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_missing_gemini_key_names_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    assert "GEMINI_API_KEY" in str(exc.value)
    assert "aistudio.google.com" in str(exc.value)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = Settings(_env_file=None)
    assert s.gemini_model_extract == "gemini-3.5-flash-lite"
    assert s.gemini_model_analyze == "gemini-3.8-flash"
    assert s.gemini_model_query == "gemini-3.5-flash"
    assert s.openrouter_model == "google/gemini-3.8-flash"
    assert s.max_concurrent_llm == 4
    assert s.openrouter_api_key is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Write `app/config.py`**

```python
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_model_extract: str = "gemini-3.5-flash-lite"
    gemini_model_analyze: str = "gemini-3.8-flash"
    gemini_model_query: str = "gemini-3.5-flash"
    openrouter_model: str = "google/gemini-3.8-flash"
    database_url: str = "sqlite:///./data/recruiter.db"
    embedding_backend: Literal["local", "gemini"] = "local"
    max_concurrent_llm: int = 4
    log_level: str = "INFO"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @field_validator("gemini_api_key")
    @classmethod
    def _require_gemini_key(cls, v: str | None) -> str:
        if not v:
            raise ValueError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and put it in .env (see .env.example)."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini",
        openrouter_api_key="test-openrouter",
        database_url="sqlite:///:memory:",
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add app/config.py tests/test_config.py tests/conftest.py
git commit -m "feat(config): typed Settings with fail-fast GEMINI_API_KEY"
```

---

### Task 2: Database models and session

**Files:**
- Create: `app/db/__init__.py`, `app/db/models.py`, `app/db/session.py`
- Test: `tests/test_models.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: ORM classes `Job, Candidate, Skill, CandidateSkill, JobSkill, Experience, Analysis, SkillMatchRow, LLMCall, LLMCache`; `Base`; `make_engine(url: str) -> Engine`; `make_session_factory(engine) -> sessionmaker[Session]`; `init_db(engine) -> None`; conftest fixtures `engine`, `db` (a `Session`), `session_factory`.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, Job


def test_tables_and_indexes(engine: Engine) -> None:
    insp = inspect(engine)
    assert set(insp.get_table_names()) >= {
        "jobs", "candidates", "skills", "candidate_skills", "job_skills",
        "experiences", "analyses", "skill_matches", "llm_calls", "llm_cache",
    }
    names = {i["name"] for i in insp.get_indexes("analyses")}
    assert "idx_analysis_score" in names
    uniques = {u["name"] for u in insp.get_unique_constraints("analyses")}
    assert "idx_analysis_pair" in uniques


def test_resume_hash_unique(db: Session) -> None:
    db.add(Candidate(resume_hash="h1", raw_text="a", extraction_method="pymupdf", char_count=1))
    db.commit()
    db.add(Candidate(resume_hash="h1", raw_text="b", extraction_method="pymupdf", char_count=1))
    with pytest.raises(IntegrityError):
        db.commit()


def test_analysis_pair_unique(db: Session) -> None:
    job = Job(title="t", raw_text="x")
    cand = Candidate(resume_hash="h2", raw_text="a", extraction_method="pymupdf", char_count=1)
    db.add_all([job, cand])
    db.commit()
    kw = dict(
        candidate_id=cand.id, job_id=job.id, final_score=1,
        required_skill_coverage=0, preferred_skill_coverage=0, experience_fit=0,
        education_fit=0, llm_fit_score=0, recommendation="Reject",
        summary="", reasoning="", scoring_version="1.0",
    )
    db.add(Analysis(**kw))
    db.commit()
    db.add(Analysis(**kw))
    with pytest.raises(IntegrityError):
        db.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: Write `app/db/models.py`**

```python
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
    skills: Mapped[list[JobSkill]] = relationship(back_populates="job", cascade="all, delete-orphan")
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
```

Notes: `idx_analysis_pair` is a named `UniqueConstraint` (SQLite backs it with a unique index). `DESC` on `idx_analysis_score` is omitted; SQLite walks an ascending index backwards at equal cost. `Candidate.extraction_json` stores the full `CandidateExtraction` dump so re-scoring never re-parses; `Candidate.flags` stores validator flags (`email_uncertain`, `phone_uncertain`).

- [ ] **Step 4: Write `app/db/session.py` and `app/db/__init__.py`**

`app/db/__init__.py`: empty.

`app/db/session.py`:
```python
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 5: Add fixtures to `tests/conftest.py`**

Replace file contents with:
```python
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.models import Base


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini",
        openrouter_api_key="test-openrouter",
        database_url="sqlite:///:memory:",
    )


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add app/db tests/test_models.py tests/conftest.py
git commit -m "feat(db): SQLAlchemy models, indexes, session factory"
```

---

### Task 3: FastAPI app, health endpoint, Docker

**Files:**
- Create: `app/main.py`, `app/api/__init__.py`, `app/api/deps.py`, `app/api/health.py`, `app/api/errors.py`, `Dockerfile`, `docker-compose.yml`
- Test: `tests/test_health.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces: `create_app(settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway | None = None) -> FastAPI`; module attribute `app`; `deps.get_db` (yields `Session`), `deps.get_gateway` (returns `request.app.state.gateway`); `errors.ApiError(status, code, message, retry_after=None)` rendered as `{code, message, retry_after}`; conftest fixture `client` (`TestClient`). `app.state.gateway` is `None` until Task 10 wires it.

- [ ] **Step 1: Write the failing test**

`tests/test_health.py`:
```python
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["gemini_configured"] is True
    assert body["openrouter_configured"] is True
    assert body["breaker_state"] == "closed"


def test_error_shape(client: TestClient) -> None:
    r = client.get("/api/_error_probe")
    assert r.status_code == 418
    assert r.json() == {"code": "PROBE", "message": "probe", "retry_after": 5}
    assert r.headers["retry-after"] == "5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_health.py -v`
Expected: FAIL with `fixture 'client' not found`

- [ ] **Step 3: Write `app/api/errors.py`**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, retry_after: int | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse(
            status_code=exc.status,
            content={"code": exc.code, "message": exc.message, "retry_after": exc.retry_after},
            headers=headers,
        )
```

- [ ] **Step 4: Write `app/api/deps.py`**

```python
from collections.abc import Iterator
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_gateway(request: Request) -> Any:
    return request.app.state.gateway
```

- [ ] **Step 5: Write `app/api/health.py`**

```python
from fastapi import APIRouter, Request

from app.api.errors import ApiError

router = APIRouter()


@router.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    gateway = request.app.state.gateway
    return {
        "gemini_configured": bool(settings.gemini_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "breaker_state": gateway.breaker.state if gateway else "closed",
    }


@router.get("/api/_error_probe", include_in_schema=False)
async def error_probe() -> None:
    raise ApiError(418, "PROBE", "probe", retry_after=5)
```

- [ ] **Step 6: Write `app/main.py`**

```python
import logging
from typing import Any

from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

from app.api.errors import install_error_handlers
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory


def create_app(
    settings: Settings,
    session_factory: sessionmaker[Session],
    gateway: Any = None,
) -> FastAPI:
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(title="NipunyaMatch", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.gateway = gateway
    install_error_handlers(app)
    app.include_router(health_router)
    return app


def _default_app() -> FastAPI:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    return create_app(settings, make_session_factory(engine))


app = _default_app()
```

- [ ] **Step 7: Add `client` fixture to `tests/conftest.py`**

Append:
```python
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(settings: Settings, session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    app = create_app(settings, session_factory)
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 8: Write `Dockerfile` and `docker-compose.yml`**

`Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8000 8501
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:
```yaml
services:
  api:
    build: .
    env_file: .env
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
  ui:
    build: .
    env_file: .env
    command: ["uv", "run", "streamlit", "run", "app/ui/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
    environment:
      API_BASE_URL: http://api:8000
    ports: ["8501:8501"]
    depends_on: [api]
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest -v`
Expected: all pass (`test_health` 2 passed)

- [ ] **Step 10: Commit and tag**

```bash
git add -A
git commit -m "feat(api): FastAPI app factory, health endpoint, error shape, Docker"
git tag v0.0
```

---

# Phase 1 — LLM Gateway

### Task 4: Gateway types

**Files:**
- Create: `app/llm/__init__.py`, `app/llm/types.py`
- Test: `tests/llm/__init__.py`, `tests/llm/test_types.py`

**Interfaces:**
- Produces:
  - `class LLMTask(StrEnum): EXTRACT="extract"; ANALYZE="analyze"; QUERY="query"`
  - `class Provider(StrEnum): GEMINI="gemini"; OPENROUTER="openrouter"`
  - `@dataclass LLMResponse(text: str, prompt_tokens: int | None, completion_tokens: int | None)`
  - `class ProviderError(Exception)` with `.status: int`, `.retry_after: float | None`, `.is_retryable`
  - `class ExtractionFailed(Exception)`, `class QuotaExhausted(Exception)` (`.retry_after`), `class NoProvider(Exception)`

- [ ] **Step 1: Write the failing test**

`tests/llm/test_types.py`:
```python
from app.llm.types import LLMTask, ProviderError


def test_task_values() -> None:
    assert LLMTask.EXTRACT == "extract"
    assert set(LLMTask) == {"extract", "analyze", "query"}


def test_provider_error_carries_status() -> None:
    e = ProviderError(429, "quota", retry_after=2.5)
    assert e.status == 429 and e.retry_after == 2.5
    assert e.is_retryable
    assert not ProviderError(400, "bad").is_retryable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_types.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/types.py`**

```python
from dataclasses import dataclass
from enum import StrEnum


class LLMTask(StrEnum):
    EXTRACT = "extract"
    ANALYZE = "analyze"
    QUERY = "query"


class Provider(StrEnum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class ProviderError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.retry_after = retry_after

    @property
    def is_retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


class ExtractionFailed(Exception):
    """Both providers returned output that failed schema validation."""


class QuotaExhausted(Exception):
    """Both providers rate-limited."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("quota exhausted")
        self.retry_after = retry_after


class NoProvider(Exception):
    """No provider configured or reachable."""
```

Create empty `app/llm/__init__.py`, `tests/llm/__init__.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/llm/test_types.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/llm tests/llm
git commit -m "feat(llm): gateway task, provider and error types"
```

---

### Task 5: Response cache

**Files:**
- Create: `app/llm/cache.py`
- Test: `tests/llm/test_cache.py`

**Interfaces:**
- Consumes: `LLMCache` model (Task 2).
- Produces: `cache_key(prompt: str, model: str, schema_version: str) -> str`; `get_cached(session, key) -> str | None`; `set_cached(session, key, text) -> None`.

- [ ] **Step 1: Write the failing test**

`tests/llm/test_cache.py`:
```python
import hashlib

from sqlalchemy.orm import Session

from app.llm.cache import cache_key, get_cached, set_cached


def test_key_is_sha256_of_parts() -> None:
    expected = hashlib.sha256(b"pMv1").hexdigest()
    assert cache_key("p", "M", "v1") == expected
    assert cache_key("p", "M", "v2") != expected


def test_roundtrip(db: Session) -> None:
    k = cache_key("prompt", "model", "1")
    assert get_cached(db, k) is None
    set_cached(db, k, '{"a": 1}')
    set_cached(db, k, '{"a": 2}')  # idempotent overwrite
    assert get_cached(db, k) == '{"a": 2}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_cache.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/cache.py`**

```python
import hashlib

from sqlalchemy.orm import Session

from app.db.models import LLMCache


def cache_key(prompt: str, model: str, schema_version: str) -> str:
    return hashlib.sha256((prompt + model + schema_version).encode()).hexdigest()


def get_cached(session: Session, key: str) -> str | None:
    row = session.get(LLMCache, key)
    return row.response_text if row else None


def set_cached(session: Session, key: str, text: str) -> None:
    session.merge(LLMCache(key=key, response_text=text))
    session.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/llm/test_cache.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/llm/cache.py tests/llm/test_cache.py
git commit -m "feat(llm): sha256 response cache backed by llm_cache table"
```

---

### Task 6: Gemini client

**Files:**
- Create: `app/llm/gemini.py`
- Test: `tests/llm/test_gemini.py`

**Interfaces:**
- Consumes: `LLMResponse`, `ProviderError`.
- Produces: `class GeminiClient(api_key: str, base_url: str, http: httpx.AsyncClient | None = None)` with `async generate(model: str, prompt: str, json_schema: dict[str, Any]) -> LLMResponse`.

- [ ] **Step 1: Write the failing test**

`tests/llm/test_gemini.py`:
```python
import json

import httpx
import pytest
import respx

from app.llm.gemini import GeminiClient
from app.llm.types import ProviderError

BASE = "https://generativelanguage.googleapis.com/v1beta"
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@respx.mock
async def test_generate_success() -> None:
    route = respx.post(f"{BASE}/models/gemini-3.8-flash:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
            },
        )
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    r = await client.generate("gemini-3.8-flash", "hi", SCHEMA)
    assert r.text == '{"x": 1}' and r.prompt_tokens == 10 and r.completion_tokens == 3
    sent = route.calls[0].request
    assert sent.headers["x-goog-api-key"] == "k"
    body = json.loads(sent.read())
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseSchema"]["properties"]["x"]["type"] == "integer"


@respx.mock
async def test_429_raises_retryable_with_retry_after() -> None:
    respx.post(f"{BASE}/models/m:generateContent").mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "7"}, json={"error": {"message": "RESOURCE_EXHAUSTED"}}
        )
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    with pytest.raises(ProviderError) as e:
        await client.generate("m", "hi", SCHEMA)
    assert e.value.status == 429 and e.value.retry_after == 7.0 and e.value.is_retryable


@respx.mock
async def test_500_raises_retryable() -> None:
    respx.post(f"{BASE}/models/m:generateContent").mock(
        return_value=httpx.Response(500, text="boom")
    )
    client = GeminiClient(api_key="k", base_url=BASE)
    with pytest.raises(ProviderError) as e:
        await client.generate("m", "hi", SCHEMA)
    assert e.value.status == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_gemini.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/gemini.py`**

```python
from typing import Any

import httpx

from app.llm.types import LLMResponse, ProviderError


def _strip_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's responseSchema rejects a few JSON-Schema keywords Pydantic emits."""
    drop = {"title", "default", "additionalProperties"}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _strip_for_gemini(v)
        elif isinstance(v, list):
            out[k] = [_strip_for_gemini(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


class GeminiClient:
    def __init__(self, api_key: str, base_url: str, http: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=60)

    async def generate(self, model: str, prompt: str, json_schema: dict[str, Any]) -> LLMResponse:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _strip_for_gemini(json_schema),
            },
        }
        try:
            resp = await self._http.post(
                f"{self._base}/models/{model}:generateContent",
                headers={"x-goog-api-key": self._key},
                json=body,
            )
        except httpx.HTTPError as e:
            raise ProviderError(503, str(e)) from e
        if resp.status_code != 200:
            ra = resp.headers.get("retry-after")
            raise ProviderError(resp.status_code, resp.text[:200], float(ra) if ra else None)
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return LLMResponse(text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))
```

Pydantic emits `$defs` + `$ref` for nested models. Gemini accepts `$ref`-free schemas only; the gateway (Task 10) inlines refs before calling providers so `_strip_for_gemini` never sees `$defs`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/llm/test_gemini.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/llm/gemini.py tests/llm/test_gemini.py
git commit -m "feat(llm): Gemini generateContent client with structured output"
```

---

### Task 7: OpenRouter client

**Files:**
- Create: `app/llm/openrouter.py`
- Test: `tests/llm/test_openrouter.py`

**Interfaces:**
- Produces: `class OpenRouterClient(api_key: str, base_url: str, http=None)` with `async generate(model, prompt, json_schema) -> LLMResponse` — same signature as `GeminiClient.generate`.

- [ ] **Step 1: Write the failing test**

`tests/llm/test_openrouter.py`:
```python
import json

import httpx
import pytest
import respx

from app.llm.openrouter import OpenRouterClient
from app.llm.types import ProviderError

BASE = "https://openrouter.ai/api/v1"
SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@respx.mock
async def test_generate_success() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"x": 2}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    r = await OpenRouterClient("k", BASE).generate("google/gemini-3.8-flash", "hi", SCHEMA)
    assert r.text == '{"x": 2}' and r.prompt_tokens == 5
    body = json.loads(route.calls[0].request.read())
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_error_status() -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(ProviderError) as e:
        await OpenRouterClient("k", BASE).generate("m", "hi", SCHEMA)
    assert e.value.status == 429
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_openrouter.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/openrouter.py`**

```python
from typing import Any

import httpx

from app.llm.types import LLMResponse, ProviderError


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, http: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=60)

    async def generate(self, model: str, prompt: str, json_schema: dict[str, Any]) -> LLMResponse:
        body = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            },
        }
        try:
            resp = await self._http.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json=body,
            )
        except httpx.HTTPError as e:
            raise ProviderError(503, str(e)) from e
        if resp.status_code != 200:
            ra = resp.headers.get("retry-after")
            raise ProviderError(resp.status_code, resp.text[:200], float(ra) if ra else None)
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResponse(
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/llm/test_openrouter.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/llm/openrouter.py tests/llm/test_openrouter.py
git commit -m "feat(llm): OpenRouter chat-completions client with json_schema"
```

---

### Task 8: Circuit breaker

**Files:**
- Create: `app/llm/breaker.py`
- Test: `tests/llm/test_breaker.py`

**Interfaces:**
- Produces: `class CircuitBreaker(threshold: int = 5, cooldown: float = 60.0, clock: Callable[[], float] = time.monotonic)` with `allow() -> bool`, `record_success() -> None`, `record_failure() -> None`, `state: str` (`"closed" | "open" | "half_open"`).

- [ ] **Step 1: Write the failing test**

`tests/llm/test_breaker.py`:
```python
from app.llm.breaker import CircuitBreaker


class Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_opens_after_threshold_and_half_opens_after_cooldown() -> None:
    clock = Clock()
    b = CircuitBreaker(threshold=5, cooldown=60, clock=clock)
    assert b.state == "closed" and b.allow()
    for _ in range(4):
        b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clock.t = 59
    assert not b.allow()
    clock.t = 61
    assert b.allow()  # single probe
    assert b.state == "half_open"
    assert not b.allow()  # second probe denied while half-open
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_success_resets_count() -> None:
    b = CircuitBreaker(threshold=2, cooldown=1, clock=lambda: 0.0)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state == "closed"


def test_half_open_failure_reopens() -> None:
    clock = Clock()
    b = CircuitBreaker(threshold=1, cooldown=10, clock=clock)
    b.record_failure()
    clock.t = 11
    assert b.allow()
    b.record_failure()
    assert b.state == "open"
    clock.t = 15
    assert not b.allow()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_breaker.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/breaker.py`**

```python
import time
from collections.abc import Callable


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 5,
        cooldown: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._probe_in_flight:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._probe_in_flight:
            return False
        if self._clock() - self._opened_at >= self._cooldown:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._probe_in_flight or self._failures >= self._threshold:
            self._opened_at = self._clock()
            self._probe_in_flight = False
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/llm/test_breaker.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/llm/breaker.py tests/llm/test_breaker.py
git commit -m "feat(llm): circuit breaker with injected clock"
```

---

### Task 9: Prompt loader

**Files:**
- Create: `app/llm/prompts.py`, `app/llm/prompts/repair.txt`
- Test: `tests/llm/test_prompts.py`

**Interfaces:**
- Produces: `load_prompt(name: str) -> str` (reads `app/llm/prompts/<name>.txt`, cached); `render(name: str, **vars: str) -> str` (`str.format_map`). Prompt files escape literal braces as `{{ }}`.

- [ ] **Step 1: Write the failing test**

`tests/llm/test_prompts.py`:
```python
import pytest

from app.llm.prompts import load_prompt, render


def test_load_and_render() -> None:
    text = load_prompt("repair")
    assert "{error}" in text
    out = render("repair", error="bad field", previous='{"a":1}')
    assert "bad field" in out and '{"a":1}' in out


def test_missing_prompt() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_prompts.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/prompts/repair.txt`**

```
Your previous response did not match the required JSON schema.

Validation error:
{error}

Previous response:
{previous}

Return ONLY corrected JSON that satisfies the schema. Do not add commentary.
```

- [ ] **Step 4: Write `app/llm/prompts.py`**

```python
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    path = _DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def render(name: str, **vars: str) -> str:
    return load_prompt(name).format_map(vars)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/llm/test_prompts.py -v`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add app/llm/prompts.py app/llm/prompts/repair.txt tests/llm/test_prompts.py
git commit -m "feat(llm): versioned prompt files and loader"
```

---

### Task 10: Gateway

**Files:**
- Create: `app/llm/gateway.py`, `app/llm/schema_utils.py`
- Test: `tests/llm/test_gateway.py`, `tests/llm/test_schema_utils.py`
- Modify: `app/main.py` (`_default_app` builds gateway), `tests/conftest.py` (add `gateway` fixture; `client` fixture passes it)

**Interfaces:**
- Consumes: `GeminiClient`, `OpenRouterClient`, `CircuitBreaker`, cache functions, `render("repair", ...)`, `Settings`, `LLMCall`.
- Produces:
  - `schema_utils.inline_refs(schema: dict) -> dict` — resolves `$ref`/`$defs` into a flat JSON schema.
  - `class Gateway(settings, session_factory, gemini: GeminiClient | None, openrouter: OpenRouterClient | None, breaker: CircuitBreaker | None = None, sleep = asyncio.sleep)`
  - `async complete(self, prompt: str, schema: type[T], task: LLMTask) -> T`
  - `last_provider: Provider | None`, `breaker: CircuitBreaker`, `call_count: int`
  - `build_gateway(settings, session_factory) -> Gateway`
- Behaviour: model chosen by task; `schema_version = schema.__name__ + ":" + sha256(json.dumps(schema.model_json_schema(), sort_keys=True))[:8]`.

- [ ] **Step 1: Write the failing schema-utils test**

`tests/llm/test_schema_utils.py`:
```python
from pydantic import BaseModel

from app.llm.schema_utils import inline_refs


class Inner(BaseModel):
    a: int


class Outer(BaseModel):
    items: list[Inner]
    one: Inner | None


def test_inline_refs_removes_defs() -> None:
    s = inline_refs(Outer.model_json_schema())
    assert "$defs" not in s
    assert "$ref" not in str(s)
    assert s["properties"]["items"]["items"]["properties"]["a"]["type"] == "integer"
```

- [ ] **Step 2: Write `app/llm/schema_utils.py`**

```python
from copy import deepcopy
from typing import Any


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(i) for i in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result
```

Run: `uv run pytest tests/llm/test_schema_utils.py -v` → `1 passed`.

- [ ] **Step 3: Write the failing gateway test**

`tests/llm/test_gateway.py`:
```python
from collections.abc import Iterator

import httpx
import pytest
import respx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import LLMCall
from app.llm.breaker import CircuitBreaker
from app.llm.gateway import Gateway
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient
from app.llm.types import ExtractionFailed, LLMTask, Provider, QuotaExhausted

GEM = "https://generativelanguage.googleapis.com/v1beta"
ORT = "https://openrouter.ai/api/v1"


class Out(BaseModel):
    x: int


def gem_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def or_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def gw(settings: Settings, session_factory: sessionmaker[Session], sleeps: list[float]) -> Gateway:
    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    return Gateway(
        settings,
        session_factory,
        GeminiClient("k", GEM),
        OpenRouterClient("k", ORT),
        CircuitBreaker(threshold=5, cooldown=60, clock=lambda: 0.0),
        sleep=fake_sleep,
    )


@pytest.fixture(autouse=True)
def _mock() -> Iterator[None]:
    with respx.mock:
        yield


async def test_gemini_success_and_cache_hit(gw: Gateway, db: Session) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok('{"x": 1}')
    )
    r1 = await gw.complete("p", Out, LLMTask.EXTRACT)
    r2 = await gw.complete("p", Out, LLMTask.EXTRACT)
    assert r1 == r2 == Out(x=1)
    assert route.call_count == 1
    assert gw.last_provider == Provider.GEMINI
    calls = db.scalars(select(LLMCall)).all()
    assert len(calls) == 1 and calls[0].status == "ok" and calls[0].task == "extract"


async def test_retries_then_falls_back(gw: Gateway, db: Session, sleeps: list[float]) -> None:
    gem = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=httpx.Response(429, json={})
    )
    ort = respx.post(f"{ORT}/chat/completions").mock(return_value=or_ok('{"x": 9}'))
    r = await gw.complete("p", Out, LLMTask.ANALYZE)
    assert r == Out(x=9)
    assert gem.call_count == 3 and ort.call_count == 1
    assert [round(s) for s in sleeps] == [1, 2]
    assert all(1.0 <= sleeps[0] < 1.1 for _ in [0]) and 2.0 <= sleeps[1] < 2.1
    assert gw.last_provider == Provider.OPENROUTER
    statuses = [c.status for c in db.scalars(select(LLMCall)).all()]
    assert statuses == ["error", "error", "error", "ok"]


async def test_retry_after_header_honoured(gw: Gateway, sleeps: list[float]) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[httpx.Response(503, headers={"retry-after": "3"}), gem_ok('{"x": 4}')]
    )
    assert await gw.complete("p", Out, LLMTask.QUERY) == Out(x=4)
    assert sleeps == [3.0]


async def test_breaker_open_skips_gemini(gw: Gateway) -> None:
    gem = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent")
    respx.post(f"{ORT}/chat/completions").mock(return_value=or_ok('{"x": 5}'))
    for _ in range(5):
        gw.breaker.record_failure()
    assert gw.breaker.state == "open"
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=5)
    assert gem.call_count == 0


async def test_bad_json_repaired_once(gw: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        side_effect=[gem_ok('{"x": "not int"}'), gem_ok('{"x": 7}')]
    )
    assert await gw.complete("p", Out, LLMTask.EXTRACT) == Out(x=7)
    assert route.call_count == 2
    second = route.calls[1].request.read().decode()
    assert "did not match the required JSON schema" in second


async def test_repair_fails_raises_extraction_failed(gw: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok("not json at all")
    )
    with pytest.raises(ExtractionFailed):
        await gw.complete("p", Out, LLMTask.EXTRACT)


async def test_both_rate_limited_raises_quota(gw: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=httpx.Response(429)
    )
    respx.post(f"{ORT}/chat/completions").mock(return_value=httpx.Response(429))
    with pytest.raises(QuotaExhausted):
        await gw.complete("p", Out, LLMTask.EXTRACT)


async def test_no_openrouter_still_works_on_gemini(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok('{"x": 1}')
    )
    g = Gateway(settings, session_factory, GeminiClient("k", GEM), None)
    assert await g.complete("p", Out, LLMTask.EXTRACT) == Out(x=1)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/llm/test_gateway.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'app.llm.gateway'`

- [ ] **Step 5: Write `app/llm/gateway.py`**

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import LLMCall
from app.llm.breaker import CircuitBreaker
from app.llm.cache import cache_key, get_cached, set_cached
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient
from app.llm.prompts import render
from app.llm.schema_utils import inline_refs
from app.llm.types import (
    ExtractionFailed,
    LLMResponse,
    LLMTask,
    NoProvider,
    Provider,
    ProviderError,
    QuotaExhausted,
)

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

BACKOFF = (1.0, 2.0, 4.0)
MAX_ATTEMPTS = 3
JITTER = 0.09


def schema_version(schema: type[BaseModel]) -> str:
    digest = hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode())
    return f"{schema.__name__}:{digest.hexdigest()[:8]}"


class Gateway:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        gemini: GeminiClient | None,
        openrouter: OpenRouterClient | None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if gemini is None and openrouter is None:
            raise NoProvider("no LLM provider configured")
        self._settings = settings
        self._sessions = session_factory
        self._gemini = gemini
        self._openrouter = openrouter
        self.breaker = breaker or CircuitBreaker()
        self._sleep = sleep
        self.last_provider: Provider | None = None
        self.call_count = 0

    def _model_for(self, task: LLMTask) -> str:
        s = self._settings
        return {
            LLMTask.EXTRACT: s.gemini_model_extract,
            LLMTask.ANALYZE: s.gemini_model_analyze,
            LLMTask.QUERY: s.gemini_model_query,
        }[task]

    def _log(
        self,
        provider: Provider,
        model: str,
        task: LLMTask,
        attempt: int,
        started: float,
        status: str,
        resp: LLMResponse | None = None,
    ) -> None:
        self.call_count += 1
        with self._sessions() as s:
            s.add(
                LLMCall(
                    provider=provider.value,
                    model=model,
                    task=task.value,
                    attempt=attempt,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status=status,
                    prompt_tokens=resp.prompt_tokens if resp else None,
                    completion_tokens=resp.completion_tokens if resp else None,
                )
            )
            s.commit()

    async def _gemini_with_retry(
        self, model: str, prompt: str, js: dict[str, Any], task: LLMTask
    ) -> LLMResponse | None:
        """None when Gemini is skipped (breaker) or exhausts retries."""
        if self._gemini is None or not self.breaker.allow():
            return None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                resp = await self._gemini.generate(model, prompt, js)
            except ProviderError as e:
                self._log(Provider.GEMINI, model, task, attempt, started, "error")
                self.breaker.record_failure()
                if not e.is_retryable or attempt == MAX_ATTEMPTS:
                    return None
                delay = e.retry_after or BACKOFF[attempt - 1] + random.random() * JITTER
                await self._sleep(delay)
                continue
            self._log(Provider.GEMINI, model, task, attempt, started, "ok", resp)
            self.breaker.record_success()
            self.last_provider = Provider.GEMINI
            return resp
        return None

    async def _openrouter_once(
        self, prompt: str, js: dict[str, Any], task: LLMTask
    ) -> LLMResponse:
        if self._openrouter is None:
            raise NoProvider("Gemini unavailable and OPENROUTER_API_KEY not set")
        model = self._settings.openrouter_model
        started = time.monotonic()
        try:
            resp = await self._openrouter.generate(model, prompt, js)
        except ProviderError as e:
            self._log(Provider.OPENROUTER, model, task, 1, started, "error")
            if e.status == 429:
                raise QuotaExhausted(e.retry_after) from e
            raise NoProvider(str(e)) from e
        self._log(Provider.OPENROUTER, model, task, 1, started, "ok", resp)
        self.last_provider = Provider.OPENROUTER
        return resp

    async def _raw(self, prompt: str, js: dict[str, Any], task: LLMTask) -> str:
        resp = await self._gemini_with_retry(self._model_for(task), prompt, js, task)
        if resp is None:
            resp = await self._openrouter_once(prompt, js, task)
        return resp.text

    async def complete(self, prompt: str, schema: type[T], task: LLMTask) -> T:
        key = cache_key(prompt, self._model_for(task), schema_version(schema))
        with self._sessions() as s:
            cached = get_cached(s, key)
        if cached is not None:
            return schema.model_validate_json(cached)

        js = inline_refs(schema.model_json_schema())
        text = await self._raw(prompt, js, task)
        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as first:
            repair = prompt + "\n\n" + render("repair", error=str(first)[:2000], previous=text[:4000])
            text = await self._raw(repair, js, task)
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as second:
                raise ExtractionFailed(str(second)[:500]) from second
        with self._sessions() as s:
            set_cached(s, key, parsed.model_dump_json())
        return parsed


def build_gateway(settings: Settings, session_factory: sessionmaker[Session]) -> Gateway:
    gemini = (
        GeminiClient(settings.gemini_api_key, settings.gemini_base_url)
        if settings.gemini_api_key
        else None
    )
    openrouter = (
        OpenRouterClient(settings.openrouter_api_key, settings.openrouter_base_url)
        if settings.openrouter_api_key
        else None
    )
    return Gateway(settings, session_factory, gemini, openrouter)
```

- [ ] **Step 6: Wire gateway in `app/main.py` and conftest**

In `app/main.py` `_default_app`:
```python
def _default_app() -> FastAPI:
    from app.llm.gateway import build_gateway

    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    return create_app(settings, factory, build_gateway(settings, factory))
```

In `tests/conftest.py` append and update `client`:
```python
from app.llm.breaker import CircuitBreaker
from app.llm.gateway import Gateway
from app.llm.gemini import GeminiClient
from app.llm.openrouter import OpenRouterClient

GEM_URL = "https://generativelanguage.googleapis.com/v1beta"
ORT_URL = "https://openrouter.ai/api/v1"


@pytest.fixture
def gateway(settings: Settings, session_factory: sessionmaker[Session]) -> Gateway:
    async def no_sleep(_: float) -> None:
        return None

    return Gateway(
        settings,
        session_factory,
        GeminiClient("k", GEM_URL),
        OpenRouterClient("k", ORT_URL),
        CircuitBreaker(clock=lambda: 0.0),
        sleep=no_sleep,
    )


@pytest.fixture
def client(
    settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway
) -> Iterator[TestClient]:
    app = create_app(settings, session_factory, gateway)
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/llm -v`
Expected: all pass (8 gateway tests).

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check --fix . && uv run mypy app/llm`
Expected: clean.

- [ ] **Step 9: Commit and tag**

```bash
git add -A
git commit -m "feat(llm): gateway with cache, retry, breaker, failover, repair pass, audit log"
git tag v0.1
```

---

# Phase 2 — Parsing

### Task 11: PDF validation

**Files:**
- Create: `app/parsing/__init__.py`, `app/parsing/validate.py`
- Test: `tests/parsing/__init__.py`, `tests/parsing/test_validate.py`, `tests/parsing/helpers.py`

**Interfaces:**
- Produces: `class InvalidPDF(Exception)` with `.code: str` (`"INVALID_PDF" | "FILE_TOO_LARGE"`) and `.message`; `validate_pdf(data: bytes) -> fitz.Document`. Constants `MAX_PAGES = 20`, `MAX_BYTES = 10 * 1024 * 1024`.
- `tests/parsing/helpers.py`: `make_pdf(pages: list[list[tuple[float, float, str]]]) -> bytes` (each page = list of `(x, y, text)`), `make_blank_pdf(n: int) -> bytes`.

- [ ] **Step 1: Write `tests/parsing/helpers.py`**

```python
import fitz  # PyMuPDF


def make_pdf(pages: list[list[tuple[float, float, str]]]) -> bytes:
    doc = fitz.open()
    for items in pages:
        page = doc.new_page(width=595, height=842)
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=11)
    return doc.tobytes()


def make_blank_pdf(n: int) -> bytes:
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    return doc.tobytes()
```

- [ ] **Step 2: Write the failing test**

`tests/parsing/test_validate.py`:
```python
import fitz
import pytest

from app.parsing.validate import InvalidPDF, validate_pdf
from tests.parsing.helpers import make_blank_pdf, make_pdf


def test_valid_pdf_returns_doc() -> None:
    doc = validate_pdf(make_pdf([[(72, 72, "hello")]]))
    assert doc.page_count == 1


def test_not_a_pdf() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(b"hello world")
    assert e.value.code == "INVALID_PDF" and "not a PDF" in e.value.message


def test_too_many_pages() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(make_blank_pdf(20))
    assert "20 pages" in e.value.message


def test_too_large() -> None:
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(b"%PDF-" + b"0" * (10 * 1024 * 1024))
    assert e.value.code == "FILE_TOO_LARGE"


def test_encrypted() -> None:
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="x")
    with pytest.raises(InvalidPDF) as e:
        validate_pdf(data)
    assert "encrypted" in e.value.message
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/parsing/test_validate.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Write `app/parsing/validate.py`**

```python
import fitz

MAX_PAGES = 20
MAX_BYTES = 10 * 1024 * 1024


class InvalidPDF(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_pdf(data: bytes) -> fitz.Document:
    if len(data) >= MAX_BYTES:
        raise InvalidPDF("FILE_TOO_LARGE", "File is over 10 MB. Please upload a smaller PDF.")
    if not data.startswith(b"%PDF"):
        raise InvalidPDF("INVALID_PDF", "This file is not a PDF.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # fitz raises generic errors on corrupt input
        raise InvalidPDF("INVALID_PDF", "This PDF could not be opened; it may be corrupt.") from e
    if doc.is_encrypted:
        raise InvalidPDF("INVALID_PDF", "This PDF is encrypted. Remove the password and retry.")
    if doc.page_count >= MAX_PAGES:
        raise InvalidPDF(
            "INVALID_PDF", f"This PDF has {doc.page_count} pages; the limit is {MAX_PAGES - 1}."
        )
    return doc
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/parsing/test_validate.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add app/parsing tests/parsing
git commit -m "feat(parsing): PDF validation with recruiter-facing messages"
```

---

### Task 12: Column-aware text extraction

**Files:**
- Create: `app/parsing/pdf.py`
- Test: `tests/parsing/test_pdf.py`

**Interfaces:**
- Produces: `@dataclass PageText(number: int, text: str, char_count: int)`; `extract_pages(doc: fitz.Document) -> list[PageText]`; `order_blocks(blocks: list[tuple[float, float, float, float, str]], page_width: float) -> list[str]` (pure; blocks are `(x0, y0, x1, y1, text)`).

- [ ] **Step 1: Write the failing test**

`tests/parsing/test_pdf.py`:
```python
from app.parsing.pdf import extract_pages, order_blocks
from app.parsing.validate import validate_pdf
from tests.parsing.helpers import make_pdf


def test_order_blocks_two_columns() -> None:
    blocks = [
        (300, 100, 500, 120, "R1"),
        (50, 100, 250, 120, "L1"),
        (50, 200, 250, 220, "L2"),
        (300, 200, 500, 220, "R2"),
    ]
    assert order_blocks(blocks, page_width=595) == ["L1", "L2", "R1", "R2"]


def test_order_blocks_single_column_by_y() -> None:
    blocks = [(50, 300, 500, 320, "B"), (50, 100, 500, 120, "A")]
    assert order_blocks(blocks, page_width=595) == ["A", "B"]


def test_extract_pages_two_column_pdf() -> None:
    pdf = make_pdf([[(72, 100, "Left top"), (350, 100, "Right top"),
                     (72, 200, "Left bottom"), (350, 200, "Right bottom")]])
    pages = extract_pages(validate_pdf(pdf))
    assert len(pages) == 1
    t = pages[0].text
    assert t.index("Left top") < t.index("Left bottom") < t.index("Right top") < t.index("Right bottom")
    assert pages[0].char_count == len(t)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parsing/test_pdf.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/parsing/pdf.py`**

```python
from dataclasses import dataclass

import fitz

Block = tuple[float, float, float, float, str]


@dataclass
class PageText:
    number: int
    text: str
    char_count: int


def order_blocks(blocks: list[Block], page_width: float) -> list[str]:
    """Cluster blocks into column bands by x-midpoint, then read each band top-to-bottom."""
    if not blocks:
        return []
    band_width = page_width / 2  # ponytail: two bands; add k-means if 3-column resumes appear
    keyed = sorted(
        blocks,
        key=lambda b: (int(((b[0] + b[2]) / 2) // band_width), b[1], b[0]),
    )
    return [b[4].strip() for b in keyed if b[4].strip()]


def extract_pages(doc: fitz.Document) -> list[PageText]:
    pages: list[PageText] = []
    for i, page in enumerate(doc):
        raw = page.get_text("blocks")
        blocks: list[Block] = [(b[0], b[1], b[2], b[3], b[4]) for b in raw if b[6] == 0]
        text = "\n".join(order_blocks(blocks, page.rect.width))
        pages.append(PageText(i + 1, text, len(text)))
    return pages
```

`b[6] == 0` keeps text blocks and drops image blocks. Full-width blocks (single-column resumes) have midpoint in band 0 and sort by y, which is the single-column case.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/parsing/test_pdf.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/parsing/pdf.py tests/parsing/test_pdf.py
git commit -m "feat(parsing): column-aware PyMuPDF block ordering"
```

---

### Task 13: OCR fallback

**Files:**
- Create: `app/parsing/ocr.py`
- Test: `tests/parsing/test_ocr.py`
- Modify: `pyproject.toml` (add `"pillow>=10"`)

**Interfaces:**
- Produces: `OCR_THRESHOLD = 100`; `needs_ocr(page: PageText) -> bool`; `ocr_page(doc: fitz.Document, page_number: int, run_tesseract: Callable[[Any], str] = _tesseract) -> str`; `_tesseract(img) -> str` (lazy-imports pytesseract).

- [ ] **Step 1: Write the failing test**

`tests/parsing/test_ocr.py`:
```python
from typing import Any

from app.parsing.ocr import needs_ocr, ocr_page
from app.parsing.pdf import PageText
from app.parsing.validate import validate_pdf
from tests.parsing.helpers import make_blank_pdf


def test_needs_ocr_threshold() -> None:
    assert needs_ocr(PageText(1, "x" * 99, 99))
    assert not needs_ocr(PageText(1, "x" * 100, 100))


def test_ocr_page_rasterises_at_300dpi_and_calls_tesseract() -> None:
    seen: list[Any] = []

    def fake(img: Any) -> str:
        seen.append(img)
        return "OCR TEXT"

    doc = validate_pdf(make_blank_pdf(1))
    assert ocr_page(doc, 0, run_tesseract=fake) == "OCR TEXT"
    img = seen[0]
    # A4 at 300 DPI ≈ 2480 x 3508
    assert 2400 < img.width < 2560 and 3400 < img.height < 3600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parsing/test_ocr.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/parsing/ocr.py`**

```python
import io
from collections.abc import Callable
from typing import Any

import fitz
from PIL import Image

from app.parsing.pdf import PageText

OCR_THRESHOLD = 100
DPI = 300


def _tesseract(img: Any) -> str:
    import pytesseract

    return str(pytesseract.image_to_string(img))


def needs_ocr(page: PageText) -> bool:
    return page.char_count < OCR_THRESHOLD


def ocr_page(
    doc: fitz.Document, page_number: int, run_tesseract: Callable[[Any], str] = _tesseract
) -> str:
    pix = doc[page_number].get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return run_tesseract(img)
```

Add `"pillow>=10",` to `[project].dependencies` in `pyproject.toml`, then `uv sync`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/parsing/test_ocr.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/parsing/ocr.py tests/parsing/test_ocr.py pyproject.toml uv.lock
git commit -m "feat(parsing): per-page Tesseract OCR fallback"
```

---

### Task 14: Normalisation

**Files:**
- Create: `app/parsing/normalise.py`
- Test: `tests/parsing/test_normalise.py`

**Interfaces:**
- Produces: `normalise(pages: list[str]) -> str` — ligature/smart-quote fixes; header/footer stripping (a line repeated on ≥ 2 pages, and bare page-number lines); mid-sentence line joining (previous line ends with lowercase letter or comma, next starts lowercase, neither is a bullet); bullet markers preserved; whitespace collapsed; ≥3 newlines → 2.

- [ ] **Step 1: Write the failing test**

`tests/parsing/test_normalise.py`:
```python
from app.parsing.normalise import normalise


def test_ligatures_and_quotes() -> None:
    assert normalise(["ﬁnance “quoted” it’s"]) == 'finance "quoted" it\'s'


def test_repeated_header_footer_removed() -> None:
    pages = ["John Doe Resume\nSkills: Python\nPage 1", "John Doe Resume\nExperience\nPage 2"]
    out = normalise(pages)
    assert out.count("John Doe Resume") == 0
    assert "Page 1" not in out
    assert "Skills: Python" in out and "Experience" in out


def test_join_broken_lines_keep_bullets() -> None:
    out = normalise(["Built a scalable data\npipeline for analytics\n• Led team\n• Shipped API"])
    assert "Built a scalable data pipeline for analytics" in out
    assert "• Led team\n• Shipped API" in out


def test_collapse_whitespace() -> None:
    assert normalise(["a   b\n\n\n\nc"]) == "a b\n\nc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/parsing/test_normalise.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/parsing/normalise.py`**

```python
import re
from collections import Counter

_CHARS = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " ",
}
_BULLET = re.compile(r"^\s*([•\-\*▪●○◦‣]|\d+[.)])\s+")
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)


def _fix_chars(s: str) -> str:
    for k, v in _CHARS.items():
        s = s.replace(k, v)
    return s


def _strip_repeated(pages: list[list[str]]) -> list[list[str]]:
    repeated: set[str] = set()
    if len(pages) >= 2:
        counts = Counter(ln for p in pages for ln in set(p) if ln)
        repeated = {ln for ln, c in counts.items() if c >= 2}
    return [[ln for ln in p if ln not in repeated and not _PAGE_NO.match(ln)] for p in pages]


def _join_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if (
            out
            and ln
            and not _BULLET.match(ln)
            and not _BULLET.match(out[-1])
            and ln[0].islower()
            and re.search(r"[a-z,]$", out[-1])
        ):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def normalise(pages: list[str]) -> str:
    split = [
        [re.sub(r"[ \t]+", " ", ln).strip() for ln in _fix_chars(p).splitlines()] for p in pages
    ]
    split = _strip_repeated(split)
    lines = _join_lines([ln for p in split for ln in p])
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/parsing/test_normalise.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add app/parsing/normalise.py tests/parsing/test_normalise.py
git commit -m "feat(parsing): text normalisation with header/footer stripping"
```

---

### Task 15: Section tagging and parse pipeline

**Files:**
- Create: `app/parsing/sections.py`, `app/parsing/pipeline.py`
- Test: `tests/parsing/test_sections.py`, `tests/parsing/test_pipeline.py`

**Interfaces:**
- Produces:
  - `tag_sections(text: str) -> dict[str, str]` — keys ⊆ `{"education","experience","skills","projects","certifications"}`; value = text under that heading until the next heading.
  - `@dataclass ParsedDocument(text: str, extraction_method: str, char_count: int, sections: dict[str, str])`
  - `parse_pdf(data: bytes, run_tesseract=_tesseract) -> ParsedDocument` (raises `InvalidPDF`)
  - `parse_text(text: str) -> ParsedDocument` (`extraction_method="text"`)

- [ ] **Step 1: Write the failing tests**

`tests/parsing/test_sections.py`:
```python
from app.parsing.sections import tag_sections


def test_tags_known_headings_with_synonyms() -> None:
    text = "WORK EXPERIENCE\nAcme 2020-2022\nTechnical Skills\nPython, SQL\nEDUCATION\nB.Tech 2019"
    s = tag_sections(text)
    assert s["experience"].strip() == "Acme 2020-2022"
    assert s["skills"].strip() == "Python, SQL"
    assert s["education"].strip() == "B.Tech 2019"


def test_no_headings_gives_empty_dict() -> None:
    assert tag_sections("just a paragraph about me") == {}
```

`tests/parsing/test_pipeline.py`:
```python
from typing import Any

from app.parsing.pipeline import parse_pdf, parse_text
from tests.parsing.helpers import make_pdf


def test_parse_text_pdf() -> None:
    pdf = make_pdf([[(72, 72, "SKILLS"), (72, 100, "Python and Docker " * 8)]])
    doc = parse_pdf(pdf)
    assert doc.extraction_method == "pymupdf"
    assert "Python and Docker" in doc.text
    assert doc.char_count == len(doc.text)
    assert "skills" in doc.sections


def test_parse_scanned_pdf_uses_ocr_only_on_empty_pages() -> None:
    calls: list[Any] = []

    def fake(img: Any) -> str:
        calls.append(img)
        return "OCR RESULT " * 20

    pdf = make_pdf([[(72, 100, "real text " * 20)], []])  # page 2 blank
    doc = parse_pdf(pdf, run_tesseract=fake)
    assert doc.extraction_method == "pymupdf+ocr"
    assert len(calls) == 1
    assert "real text" in doc.text and "OCR RESULT" in doc.text


def test_parse_text_plain() -> None:
    doc = parse_text("  Skills\nPython  ")
    assert doc.extraction_method == "text" and doc.text == "Skills\nPython"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_sections.py tests/parsing/test_pipeline.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/parsing/sections.py`**

```python
import re

_SYNONYMS: dict[str, list[str]] = {
    "education": ["education", "academic background", "academics", "qualifications"],
    "experience": ["experience", "work experience", "employment", "work history",
                   "professional experience", "career history"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tech stack"],
    "projects": ["projects", "personal projects", "key projects", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses", "licences"],
}
_LOOKUP = {syn: key for key, syns in _SYNONYMS.items() for syn in syns}
_HEADING = re.compile(r"^\s*([A-Za-z &/]{3,40})\s*:?\s*$")


def tag_sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _HEADING.match(line)
        key = _LOOKUP.get(m.group(1).strip().lower()) if m else None
        if key:
            current = key
            out.setdefault(current, [])
            continue
        if current:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}
```

- [ ] **Step 4: Write `app/parsing/pipeline.py`**

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.parsing.normalise import normalise
from app.parsing.ocr import _tesseract, needs_ocr, ocr_page
from app.parsing.pdf import extract_pages
from app.parsing.sections import tag_sections
from app.parsing.validate import validate_pdf


@dataclass
class ParsedDocument:
    text: str
    extraction_method: str
    char_count: int
    sections: dict[str, str]


def parse_pdf(data: bytes, run_tesseract: Callable[[Any], str] = _tesseract) -> ParsedDocument:
    doc = validate_pdf(data)
    pages = extract_pages(doc)
    used_ocr = False
    texts: list[str] = []
    for p in pages:
        if needs_ocr(p):
            used_ocr = True
            texts.append(ocr_page(doc, p.number - 1, run_tesseract))
        else:
            texts.append(p.text)
    text = normalise(texts)
    return ParsedDocument(
        text=text,
        extraction_method="pymupdf+ocr" if used_ocr else "pymupdf",
        char_count=len(text),
        sections=tag_sections(text),
    )


def parse_text(text: str) -> ParsedDocument:
    clean = normalise([text])
    return ParsedDocument(clean, "text", len(clean), tag_sections(clean))
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/parsing -v`
Expected: all pass.

- [ ] **Step 6: Commit and tag**

```bash
git add app/parsing tests/parsing
git commit -m "feat(parsing): section tagging and parse_pdf/parse_text pipeline"
git tag v0.2
```

---

# Phase 3 — Extraction

### Task 16: Extraction and analysis schemas

**Files:**
- Create: `app/extraction/__init__.py`, `app/extraction/schemas.py`
- Test: `tests/extraction/__init__.py`, `tests/extraction/test_schemas.py`

**Interfaces:**
- Produces (§6): `WorkExperience`, `Education`, `CandidateExtraction`, `SkillMatch`, `CandidateAnalysis`. Optional fields default `None`; list fields default `[]`; `extraction_confidence` defaults `0.5`.

- [ ] **Step 1: Write the failing test**

`tests/extraction/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from app.extraction.schemas import CandidateAnalysis, CandidateExtraction


def test_extraction_minimal_and_defaults() -> None:
    e = CandidateExtraction.model_validate_json('{"skills": ["Python"]}')
    assert e.name is None and e.skills == ["Python"] and e.experience == []


def test_extraction_rejects_bad_email() -> None:
    with pytest.raises(ValidationError):
        CandidateExtraction(email="not-an-email")


def test_analysis_bounds() -> None:
    base = dict(skill_matches=[], strengths=[], weaknesses=[], summary="s",
                interview_questions=["a", "b", "c"], reasoning="r")
    assert CandidateAnalysis(llm_fit_score=100, **base).llm_fit_score == 100
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=101, **base)
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=1, **{**base, "interview_questions": ["a"]})
    with pytest.raises(ValidationError):
        CandidateAnalysis(llm_fit_score=1, **{**base, "strengths": list("abcdef")})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extraction/test_schemas.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/extraction/schemas.py`**

```python
from pydantic import BaseModel, EmailStr, Field


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str | None = None  # ISO YYYY-MM where derivable
    end_date: str | None = None  # None means current
    description: str | None = None


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    graduation_year: int | None = None


class CandidateExtraction(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    skills: list[str] = Field(default_factory=list)  # verbatim from the resume, deduped
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_years_experience: float | None = None
    extraction_confidence: float = Field(default=0.5, ge=0, le=1)


class SkillMatch(BaseModel):
    skill: str
    present: bool
    evidence: str | None = None  # the resume phrase that proves it
    required: bool  # required vs nice-to-have in the JD


class CandidateAnalysis(BaseModel):
    llm_fit_score: int = Field(ge=0, le=100)
    skill_matches: list[SkillMatch]
    strengths: list[str] = Field(max_length=5)
    weaknesses: list[str] = Field(max_length=5)
    summary: str  # 2-3 sentences
    interview_questions: list[str] = Field(min_length=3, max_length=5)
    reasoning: str  # why this score, cited to resume content
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/extraction/test_schemas.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/extraction tests/extraction
git commit -m "feat(extraction): CandidateExtraction and CandidateAnalysis schemas"
```

---

### Task 17: Post-parse validators

**Files:**
- Create: `app/extraction/validators.py`
- Test: `tests/extraction/test_validators.py`

**Interfaces:**
- Produces:
  - `norm(s: str) -> str` — lowercase, whitespace collapsed.
  - `years_from_experience(exp: list[WorkExperience], today: date | None = None) -> float | None` — union of month ranges, overlaps counted once; `end_date=None` = today; entries without `start_date` ignored.
  - `validate_extraction(e, resume_text, today=None) -> tuple[CandidateExtraction, list[str]]` — flags ⊆ `{"email_uncertain","phone_uncertain","years_recomputed"}`.
  - `validate_analysis(a: CandidateAnalysis, resume_text: str) -> tuple[CandidateAnalysis, int]` — evidence-failing matches set `present=False`; returns hallucination count.

- [ ] **Step 1: Write the failing test**

`tests/extraction/test_validators.py`:
```python
from datetime import date

from app.extraction.schemas import (
    CandidateAnalysis,
    CandidateExtraction,
    Education,
    SkillMatch,
    WorkExperience,
)
from app.extraction.validators import validate_analysis, validate_extraction, years_from_experience


def test_years_overlap_counted_once() -> None:
    exp = [
        WorkExperience(company="a", title="t", start_date="2018-01", end_date="2020-01"),
        WorkExperience(company="b", title="t", start_date="2019-01", end_date="2021-01"),
    ]
    assert years_from_experience(exp) == 3.0


def test_years_open_ended_uses_today() -> None:
    exp = [WorkExperience(company="a", title="t", start_date="2022-01", end_date=None)]
    assert years_from_experience(exp, today=date(2024, 1, 15)) == 2.0


def test_phone_and_grad_year_and_years_rules() -> None:
    e = CandidateExtraction(
        phone="12",
        education=[Education(institution="x", graduation_year=1900),
                   Education(institution="y", graduation_year=2020)],
        total_years_experience=75,
        experience=[WorkExperience(company="a", title="t", start_date="2020-01", end_date="2022-01")],
    )
    out, flags = validate_extraction(e, "", today=date(2024, 1, 1))
    assert "phone_uncertain" in flags and out.phone == "12"
    assert out.education[0].graduation_year is None and out.education[1].graduation_year == 2020
    assert out.total_years_experience == 2.0 and "years_recomputed" in flags


def test_evidence_substring_check_is_case_and_space_insensitive() -> None:
    a = CandidateAnalysis(
        llm_fit_score=50,
        skill_matches=[
            SkillMatch(skill="Python", present=True, evidence="built   PYTHON services", required=True),
            SkillMatch(skill="Go", present=True, evidence="wrote Go daily", required=False),
            SkillMatch(skill="Rust", present=False, evidence=None, required=False),
        ],
        strengths=[], weaknesses=[], summary="s", interview_questions=["a", "b", "c"], reasoning="r",
    )
    out, halluc = validate_analysis(a, "I built Python services for years.")
    assert out.skill_matches[0].present is True
    assert out.skill_matches[1].present is False
    assert halluc == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extraction/test_validators.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/extraction/validators.py`**

```python
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
```

Email RFC shape is already enforced by `EmailStr` in the schema; `email_uncertain` flags an email that validates but does not appear in the resume text.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/extraction/test_validators.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add app/extraction/validators.py tests/extraction/test_validators.py
git commit -m "feat(extraction): validators incl. evidence substring hallucination check"
```

---

### Task 18: Extractor (LLM call 1)

**Files:**
- Create: `app/extraction/extractor.py`, `app/llm/prompts/extract.txt`
- Test: `tests/extraction/test_extractor.py`

**Interfaces:**
- Consumes: `Gateway.complete`, `render("extract", resume=...)`, `validate_extraction`.
- Produces: `MAX_CHARS = 12_000`; `async extract_candidate(gateway: Gateway, resume_text: str) -> tuple[CandidateExtraction, list[str]]`.

- [ ] **Step 1: Write the failing test**

`tests/extraction/test_extractor.py`:
```python
import json
from typing import Any

import httpx
import respx

from app.extraction.extractor import MAX_CHARS, extract_candidate
from app.llm.gateway import Gateway

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_extract_truncates_and_validates(gateway: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok({"name": "A", "skills": ["Python", "Python ", "SQL"], "phone": "123"})
    )
    text = "x" * (MAX_CHARS + 500)
    ext, flags = await extract_candidate(gateway, text)
    assert ext.name == "A" and ext.skills == ["Python", "SQL"]
    assert "phone_uncertain" in flags
    body = json.loads(route.calls[0].request.read())
    prompt = body["contents"][0]["parts"][0]["text"]
    assert prompt.count("x") <= MAX_CHARS + 50  # allow for the letter x in instructions
    assert "never invent" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extraction/test_extractor.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/prompts/extract.txt`**

```
You are extracting structured data from a resume. Return JSON matching the schema.

Rules:
- Never invent a field. If information is absent, return null (or an empty list).
- Copy skills verbatim from the resume; do not paraphrase or expand abbreviations. Deduplicate.
- Dates as YYYY-MM when derivable; end_date null means the role is current.
- total_years_experience: derive from date ranges; count overlapping roles once.
- extraction_confidence: your 0-1 estimate of how complete and accurate this extraction is.

Resume:
<<<
{resume}
>>>
```

- [ ] **Step 4: Write `app/extraction/extractor.py`**

```python
from app.extraction.schemas import CandidateExtraction
from app.extraction.validators import validate_extraction
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask

MAX_CHARS = 12_000


async def extract_candidate(
    gateway: Gateway, resume_text: str
) -> tuple[CandidateExtraction, list[str]]:
    prompt = render("extract", resume=resume_text[:MAX_CHARS])
    raw = await gateway.complete(prompt, CandidateExtraction, LLMTask.EXTRACT)
    return validate_extraction(raw, resume_text)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/extraction/test_extractor.py -v`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add app/extraction/extractor.py app/llm/prompts/extract.txt tests/extraction/test_extractor.py
git commit -m "feat(extraction): extract_candidate via gateway with 12k truncation"
```

---

### Task 19: Persist candidate

**Files:**
- Create: `app/extraction/persist.py`
- Test: `tests/extraction/test_persist.py`

**Interfaces:**
- Consumes: ORM models, `ParsedDocument`, `CandidateExtraction`.
- Produces:
  - `resume_hash(data: bytes) -> str` (sha256 hex)
  - `get_or_create_skill(session, name: str) -> Skill` (case-insensitive lookup on `canonical_name`)
  - `persist_candidate(session, parsed: ParsedDocument, ext: CandidateExtraction, flags: list[str], rhash: str) -> Candidate` — upsert on `resume_hash`; replaces skills and experiences; commits.

- [ ] **Step 1: Write the failing test**

`tests/extraction/test_persist.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extraction/test_persist.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/extraction/persist.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/extraction/test_persist.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit and tag**

```bash
git add app/extraction/persist.py tests/extraction/test_persist.py
git commit -m "feat(extraction): persist candidate with upsert on resume hash"
git tag v0.3
```

---

# Phase 4 — Scoring

### Task 20: Embeddings and skill matcher

**Files:**
- Create: `app/scoring/__init__.py`, `app/scoring/embeddings.py`, `app/scoring/skills.py`, `config/skill_aliases.yaml`
- Test: `tests/scoring/__init__.py`, `tests/scoring/test_skills.py`, `tests/scoring/fakes.py`

**Interfaces:**
- Produces:
  - `embeddings.py`: `class Embedder(Protocol): def embed(self, texts: list[str]) -> np.ndarray` (shape `(n, d)`); `class LocalEmbedder` (lazy-loads `all-MiniLM-L6-v2`); `cosine(a, b) -> float`; `to_blob(v) -> bytes`; `from_blob(b) -> np.ndarray`.
  - `skills.py`: `load_aliases(path: Path) -> dict[str, str]` (lowercased alias → canonical); `canon(name, aliases) -> str`; `@dataclass MatchResult(target: str, matched: bool, credit: float, method: str, via: str | None = None)` with `method ∈ {"exact","alias","embedding","none"}`; `class SkillMatcher(aliases, embedder, threshold=0.75, credit=0.8)` with `match(target, candidate_skills) -> MatchResult` and `coverage(targets, candidate_skills) -> tuple[float, list[MatchResult]]` (0–100; `100.0` when `targets` empty).
  - `tests/scoring/fakes.py`: `FakeEmbedder(vectors: dict[str, list[float]], dim=4)` — unknown strings get a zero vector.

- [ ] **Step 1: Write `config/skill_aliases.yaml`**

```yaml
# alias: canonical  (case-insensitive on lookup)
js: JavaScript
javascript: JavaScript
ts: TypeScript
py: Python
python3: Python
postgres: PostgreSQL
postgresql: PostgreSQL
k8s: Kubernetes
kubernetes: Kubernetes
ml: Machine Learning
machine learning: Machine Learning
aws: AWS
amazon web services: AWS
gcp: Google Cloud
google cloud platform: Google Cloud
node: Node.js
nodejs: Node.js
node.js: Node.js
react.js: React
reactjs: React
sklearn: scikit-learn
scikit learn: scikit-learn
tf: TensorFlow
```

- [ ] **Step 2: Write `tests/scoring/fakes.py`**

```python
import numpy as np


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]], dim: int = 4) -> None:
        self._v = {k.lower(): np.array(v, dtype=np.float32) for k, v in vectors.items()}
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self._v.get(t.lower(), np.zeros(self._dim, dtype=np.float32)) for t in texts]
        return np.vstack(rows)
```

- [ ] **Step 3: Write the failing test**

`tests/scoring/test_skills.py`:
```python
import math
from pathlib import Path

import numpy as np

from app.scoring.embeddings import cosine, from_blob, to_blob
from app.scoring.skills import SkillMatcher, load_aliases
from tests.scoring.fakes import FakeEmbedder

ALIASES = load_aliases(Path("config/skill_aliases.yaml"))
# ML ↔ TensorFlow at cosine 0.81; everything else orthogonal or zero.
VEC = {
    "Machine Learning": [1, 0, 0, 0],
    "TensorFlow": [0.81, math.sqrt(1 - 0.81**2), 0, 0],
    "Docker": [0, 0, 1, 0],
    "Kubernetes": [0, 0, 0, 1],
}


def matcher() -> SkillMatcher:
    return SkillMatcher(ALIASES, FakeEmbedder(VEC))


def test_exact_case_insensitive() -> None:
    r = matcher().match("Python", ["python", "SQL"])
    assert r.matched and r.credit == 1.0 and r.method == "exact"


def test_alias_hit() -> None:
    r = matcher().match("JavaScript", ["JS"])
    assert r.matched and r.credit == 1.0 and r.method == "alias" and r.via == "JS"


def test_embedding_hit_above_threshold_gets_partial_credit() -> None:
    r = matcher().match("ML", ["TensorFlow"])
    assert r.matched and r.credit == 0.8 and r.method == "embedding" and r.via == "TensorFlow"


def test_embedding_below_threshold_misses() -> None:
    r = matcher().match("Docker", ["Kubernetes"])
    assert not r.matched and r.credit == 0.0 and r.method == "none"


def test_coverage_worked_example_required() -> None:
    cov, results = matcher().coverage(
        ["Python", "FastAPI", "Docker", "PostgreSQL", "ML"],
        ["Python", "FastAPI", "TensorFlow", "scikit-learn", "PostgreSQL"],
    )
    assert cov == 76.0  # (1 + 1 + 0 + 1 + 0.8) / 5
    assert [r.matched for r in results] == [True, True, False, True, True]


def test_coverage_empty_targets_is_100() -> None:
    assert matcher().coverage([], ["x"])[0] == 100.0


def test_blob_roundtrip_and_cosine() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert np.allclose(from_blob(to_blob(v)), v)
    assert cosine(v, v) == 1.0
    assert cosine(v, np.zeros(2)) == 0.0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_skills.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 5: Write `app/scoring/embeddings.py`**

```python
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """all-MiniLM-L6-v2 on CPU; model loaded on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._name = model_name
        self._model: Any = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._name)
        out: np.ndarray = self._model.encode(texts, normalize_embeddings=True)
        return out.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
```

- [ ] **Step 6: Write `app/scoring/skills.py`**

```python
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.scoring.embeddings import Embedder, cosine


def load_aliases(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k).lower(): str(v) for k, v in data.items()}


def canon(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name.strip().lower(), name.strip())


@dataclass
class MatchResult:
    target: str
    matched: bool
    credit: float
    method: str  # exact | alias | embedding | none
    via: str | None = None


class SkillMatcher:
    def __init__(
        self,
        aliases: dict[str, str],
        embedder: Embedder,
        threshold: float = 0.75,
        credit: float = 0.8,
    ) -> None:
        self._aliases = aliases
        self._embedder = embedder
        self._threshold = threshold
        self._credit = credit

    def match(self, target: str, candidate_skills: list[str]) -> MatchResult:
        t = target.strip().lower()
        for s in candidate_skills:
            if s.strip().lower() == t:
                return MatchResult(target, True, 1.0, "exact", s)
        tc = canon(target, self._aliases).lower()
        for s in candidate_skills:
            if canon(s, self._aliases).lower() == tc:
                return MatchResult(target, True, 1.0, "alias", s)
        if candidate_skills:
            names = [canon(target, self._aliases)] + [canon(s, self._aliases) for s in candidate_skills]
            vecs = self._embedder.embed(names)
            sims = [(cosine(vecs[0], vecs[i + 1]), s) for i, s in enumerate(candidate_skills)]
            best, via = max(sims, key=lambda x: x[0])
            if best > self._threshold:
                return MatchResult(target, True, self._credit, "embedding", via)
        return MatchResult(target, False, 0.0, "none")

    def coverage(
        self, targets: list[str], candidate_skills: list[str]
    ) -> tuple[float, list[MatchResult]]:
        if not targets:
            return 100.0, []
        results = [self.match(t, candidate_skills) for t in targets]
        return round(100.0 * sum(r.credit for r in results) / len(results), 1), results
```

`ponytail:` re-embeds the candidate list per target; fine for ≤ 50 skills. Memoise `embed()` if profiling says so.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/scoring/test_skills.py -v`
Expected: `7 passed`

- [ ] **Step 8: Commit**

```bash
git add app/scoring config/skill_aliases.yaml tests/scoring
git commit -m "feat(scoring): alias + embedding skill matcher with partial credit"
```

---

### Task 21: JD parsing and job persistence

**Files:**
- Create: `app/scoring/jd.py`, `app/llm/prompts/jd.txt`
- Test: `tests/scoring/test_jd.py`

**Interfaces:**
- Consumes: `Gateway.complete`, `get_or_create_skill`, `Job`, `JobSkill`.
- Produces:
  - `EducationLevel = Literal["high_school", "associate", "bachelor", "master", "phd"]`
  - `class JobRequirements(BaseModel)`: `title: str`, `required_skills: list[str]`, `preferred_skills: list[str]`, `min_years: float | None`, `education_level: EducationLevel | None`
  - `async parse_jd(gateway, text: str) -> JobRequirements`
  - `create_job(session, reqs, raw_text) -> Job` (writes `job_skills`; commits)
  - `job_requirements(job: Job) -> JobRequirements` (rebuilds from ORM rows)

- [ ] **Step 1: Write the failing test**

`tests/scoring/test_jd.py`:
```python
import json
from typing import Any

import httpx
import respx
from sqlalchemy.orm import Session

from app.llm.gateway import Gateway
from app.scoring.jd import JobRequirements, create_job, job_requirements, parse_jd

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_parse_jd_uses_analyze_model(gateway: Gateway) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok({
            "title": "Backend Engineer", "required_skills": ["Python", "Docker"],
            "preferred_skills": ["AWS"], "min_years": 3, "education_level": "bachelor",
        })
    )
    r = await parse_jd(gateway, "We need a backend engineer...")
    assert r.title == "Backend Engineer" and r.min_years == 3 and route.call_count == 1


def test_create_job_and_roundtrip(db: Session) -> None:
    reqs = JobRequirements(
        title="T", required_skills=["Python", "Docker"], preferred_skills=["AWS"],
        min_years=3, education_level="bachelor",
    )
    job = create_job(db, reqs, "raw jd")
    assert job.id and job.min_years == 3 and job.education_level == "bachelor"
    back = job_requirements(job)
    assert sorted(back.required_skills) == ["Docker", "Python"]
    assert back.preferred_skills == ["AWS"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_jd.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/prompts/jd.txt`**

```
Extract hiring requirements from this job description. Return JSON matching the schema.

Rules:
- required_skills: technologies/skills stated as must-have or required. Copy names verbatim.
- preferred_skills: nice-to-have, bonus, or "familiarity with" items.
- min_years: minimum years of experience if stated, else null.
- education_level: minimum degree if stated: high_school, associate, bachelor, master, phd; else null.
- title: the role title.

Job description:
<<<
{jd}
>>>
```

- [ ] **Step 4: Write `app/scoring/jd.py`**

```python
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/scoring/test_jd.py -v`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add app/scoring/jd.py app/llm/prompts/jd.txt tests/scoring/test_jd.py
git commit -m "feat(scoring): JD parsing to JobRequirements and job persistence"
```

---

### Task 22: Deterministic components

**Files:**
- Create: `app/scoring/components.py`
- Test: `tests/scoring/test_components.py`

**Interfaces:**
- Produces:
  - `DEGREE_RANK: dict[str, int] = {"high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "phd": 5}`
  - `degree_level(degrees: list[str | None]) -> str | None` — highest level inferred from degree strings.
  - `experience_fit(years, min_years) -> float` — `100` if no minimum or `years >= min`; `40` if years unknown; else `40 + 60 * years / min_years` (1 dp).
  - `education_fit(candidate_level, required_level) -> float` — `100` if no requirement or meets; `70` one level below; else `40`.

- [ ] **Step 1: Write the failing test**

`tests/scoring/test_components.py`:
```python
import pytest

from app.scoring.components import degree_level, education_fit, experience_fit


@pytest.mark.parametrize(
    "degrees,expected",
    [
        (["B.Tech in CS"], "bachelor"),
        (["BSc", "M.S. Computer Science"], "master"),
        (["PhD Physics"], "phd"),
        (["MBA"], "master"),
        (["Associate of Arts"], "associate"),
        (["High School Diploma"], "high_school"),
        ([None, "certificate"], None),
    ],
)
def test_degree_level(degrees: list[str | None], expected: str | None) -> None:
    assert degree_level(degrees) == expected


@pytest.mark.parametrize(
    "years,min_years,expected",
    [(4, 3, 100.0), (3, 3, 100.0), (1.5, 3, 70.0), (0, 3, 40.0), (None, 3, 40.0),
     (None, None, 100.0), (20, 3, 100.0)],
)
def test_experience_fit(years: float | None, min_years: float | None, expected: float) -> None:
    assert experience_fit(years, min_years) == expected


@pytest.mark.parametrize(
    "cand,req,expected",
    [("bachelor", "bachelor", 100.0), ("master", "bachelor", 100.0),
     ("associate", "bachelor", 70.0), ("high_school", "bachelor", 40.0),
     (None, "bachelor", 40.0), (None, None, 100.0), ("bachelor", None, 100.0)],
)
def test_education_fit(cand: str | None, req: str | None, expected: float) -> None:
    assert education_fit(cand, req) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_components.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/scoring/components.py`**

```python
import re

DEGREE_RANK: dict[str, int] = {
    "high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "phd": 5,
}
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("phd", re.compile(r"\b(ph\.?d|doctor(ate)?|d\.?phil)\b", re.I)),
    ("master", re.compile(r"\b(m\.?s\.?c?|m\.?tech|m\.?e\.?|mba|master'?s?|m\.?a\.?)\b", re.I)),
    ("bachelor", re.compile(r"\b(b\.?s\.?c?|b\.?tech|b\.?e\.?|bachelor'?s?|b\.?a\.?|b\.?eng)\b", re.I)),
    ("associate", re.compile(r"\bassociate\b", re.I)),
    ("high_school", re.compile(r"\b(high school|secondary|diploma)\b", re.I)),
]


def degree_level(degrees: list[str | None]) -> str | None:
    best: str | None = None
    for d in degrees:
        if not d:
            continue
        for level, pat in _PATTERNS:
            if pat.search(d):
                if best is None or DEGREE_RANK[level] > DEGREE_RANK[best]:
                    best = level
                break
    return best


def experience_fit(years: float | None, min_years: float | None) -> float:
    if not min_years:
        return 100.0
    if years is None:
        return 40.0
    if years >= min_years:
        return 100.0
    return round(40.0 + 60.0 * years / min_years, 1)


def education_fit(candidate_level: str | None, required_level: str | None) -> float:
    if required_level is None:
        return 100.0
    if candidate_level is None:
        return 40.0
    gap = DEGREE_RANK[required_level] - DEGREE_RANK[candidate_level]
    if gap <= 0:
        return 100.0
    return 70.0 if gap == 1 else 40.0
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/scoring/test_components.py -v`
Expected: all parametrised cases pass.

- [ ] **Step 5: Commit**

```bash
git add app/scoring/components.py tests/scoring/test_components.py
git commit -m "feat(scoring): experience and education fit components"
```

---

### Task 23: Scoring engine

**Files:**
- Create: `app/scoring/engine.py`
- Test: `tests/scoring/test_engine.py`

**Interfaces:**
- Consumes: `SkillMatcher`, `JobRequirements`, components.
- Produces:
  - `class ScoringConfig(BaseModel)`: `version: str`, `weights: dict[str, float]`, `embedding_threshold: float`, `embedding_credit: float`, `bands: dict[str, int]`
  - `load_config(path: Path = Path("config/scoring.yaml")) -> ScoringConfig`
  - `class ScoreBreakdown(BaseModel)`: `final_score: int`, `required_skill_coverage`, `preferred_skill_coverage`, `experience_fit`, `education_fit`, `llm_fit_score` (all `float`), `recommendation: str`, `scoring_version: str`, `required_matches: list[MatchResult]`, `preferred_matches: list[MatchResult]`
  - `class ScoringEngine(config, matcher)` with `score(candidate_skills, years, degree, llm_fit, job) -> ScoreBreakdown`
  - `build_engine(embedder: Embedder, config_path=..., aliases_path=...) -> ScoringEngine`

- [ ] **Step 1: Write the failing test**

`tests/scoring/test_engine.py`:
```python
import math
from pathlib import Path

from app.scoring.engine import ScoringEngine, load_config
from app.scoring.jd import JobRequirements
from app.scoring.skills import SkillMatcher, load_aliases
from tests.scoring.fakes import FakeEmbedder

VEC = {
    "Machine Learning": [1, 0, 0, 0],
    "TensorFlow": [0.81, math.sqrt(1 - 0.81**2), 0, 0],
}
JOB = JobRequirements(
    title="Backend", required_skills=["Python", "FastAPI", "Docker", "PostgreSQL", "ML"],
    preferred_skills=["AWS", "Kubernetes"], min_years=3, education_level="bachelor",
)


def engine() -> ScoringEngine:
    cfg = load_config(Path("config/scoring.yaml"))
    matcher = SkillMatcher(
        load_aliases(Path("config/skill_aliases.yaml")), FakeEmbedder(VEC),
        cfg.embedding_threshold, cfg.embedding_credit,
    )
    return ScoringEngine(cfg, matcher)


def test_worked_example_reproduces() -> None:
    b = engine().score(
        ["Python", "FastAPI", "TensorFlow", "scikit-learn", "PostgreSQL"],
        years=4, degree="bachelor", llm_fit=78, job=JOB,
    )
    assert b.required_skill_coverage == 76.0
    assert b.preferred_skill_coverage == 0.0
    assert b.experience_fit == 100.0 and b.education_fit == 100.0
    assert b.final_score == 72 and b.recommendation == "Consider"
    assert b.scoring_version == "1.0"


def test_bands() -> None:
    e = engine()
    perfect = e.score(JOB.required_skills + JOB.preferred_skills, 5, "master", 100, JOB)
    assert perfect.final_score == 100 and perfect.recommendation == "Shortlist"
    weak = e.score([], None, None, 0, JOB)
    assert weak.recommendation == "Reject"


def test_missing_over_half_required_caps_at_consider() -> None:
    b = engine().score(["Python", "FastAPI", "AWS", "Kubernetes"], 10, "phd", 100, JOB)
    # 2 of 5 required present; total would be Shortlist without the cap
    assert b.recommendation == "Consider"


def test_weights_change_moves_score() -> None:
    cfg = load_config(Path("config/scoring.yaml"))
    cfg.weights = {**cfg.weights, "llm_fit_score": 0.0, "required_skill_coverage": 0.55}
    e = ScoringEngine(cfg, SkillMatcher({}, FakeEmbedder({})))
    job = JobRequirements(title="t", required_skills=["Python"])
    assert e.score(["Python"], 4, "bachelor", 0, job).final_score == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_engine.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/scoring/engine.py`**

```python
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from app.scoring.components import education_fit, experience_fit
from app.scoring.embeddings import Embedder
from app.scoring.jd import JobRequirements
from app.scoring.skills import MatchResult, SkillMatcher, load_aliases


class ScoringConfig(BaseModel):
    version: str
    weights: dict[str, float]
    embedding_threshold: float = 0.75
    embedding_credit: float = 0.8
    bands: dict[str, int]


def load_config(path: Path = Path("config/scoring.yaml")) -> ScoringConfig:
    return ScoringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_score: int
    required_skill_coverage: float
    preferred_skill_coverage: float
    experience_fit: float
    education_fit: float
    llm_fit_score: float
    recommendation: str
    scoring_version: str
    required_matches: list[MatchResult]
    preferred_matches: list[MatchResult]


class ScoringEngine:
    def __init__(self, config: ScoringConfig, matcher: SkillMatcher) -> None:
        self.config = config
        self.matcher = matcher

    def score(
        self,
        candidate_skills: list[str],
        years: float | None,
        degree: str | None,
        llm_fit: float,
        job: JobRequirements,
    ) -> ScoreBreakdown:
        req_cov, req_matches = self.matcher.coverage(job.required_skills, candidate_skills)
        pref_cov, pref_matches = self.matcher.coverage(job.preferred_skills, candidate_skills)
        comps = {
            "required_skill_coverage": req_cov,
            "preferred_skill_coverage": pref_cov,
            "experience_fit": experience_fit(years, job.min_years),
            "education_fit": education_fit(degree, job.education_level),
            "llm_fit_score": float(llm_fit),
        }
        w = self.config.weights
        final = round(sum(w[k] * v for k, v in comps.items()))
        bands = self.config.bands
        if final >= bands["shortlist"]:
            rec = "Shortlist"
        elif final >= bands["consider"]:
            rec = "Consider"
        else:
            rec = "Reject"
        missing = sum(1 for m in req_matches if not m.matched)
        if job.required_skills and missing > len(job.required_skills) / 2 and rec == "Shortlist":
            rec = "Consider"
        return ScoreBreakdown(
            final_score=final,
            recommendation=rec,
            scoring_version=self.config.version,
            required_matches=req_matches,
            preferred_matches=pref_matches,
            **comps,
        )


def build_engine(
    embedder: Embedder,
    config_path: Path = Path("config/scoring.yaml"),
    aliases_path: Path = Path("config/skill_aliases.yaml"),
) -> ScoringEngine:
    cfg = load_config(config_path)
    matcher = SkillMatcher(
        load_aliases(aliases_path), embedder, cfg.embedding_threshold, cfg.embedding_credit
    )
    return ScoringEngine(cfg, matcher)
```

Note on §7.4: the spec table shows required coverage `80` ("4 of 5") while its own rule credits an embedding match at `0.8`, which gives `76`. Final score is `72` either way (`0.35·76 + 0 + 20 + 10 + 15.6 = 72.2`). The plan follows the credit rule; the README should quote coverage `76`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/scoring/test_engine.py -v`
Expected: `4 passed`

- [ ] **Step 5: Type-check**

Run: `uv run mypy app/scoring`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add app/scoring/engine.py tests/scoring/test_engine.py
git commit -m "feat(scoring): weighted engine with bands and missing-required cap"
```

---

### Task 24: Analyzer (LLM call 2) and analysis persistence

**Files:**
- Create: `app/scoring/analyzer.py`, `app/llm/prompts/analyze.txt`
- Test: `tests/scoring/test_analyzer.py`

**Interfaces:**
- Consumes: `Gateway.complete`, `CandidateAnalysis`, `validate_analysis`, `ScoringEngine`, `job_requirements`, `degree_level`, ORM.
- Produces:
  - `scoring_view(extraction_json: dict) -> dict` — PII-stripped subset for the LLM: keys `skills, experience, education, projects, certifications, total_years_experience` only.
  - `async analyze_candidate(gateway, session, engine, candidate: Candidate, job: Job) -> Analysis` — LLM call 2, evidence check, deterministic scoring, upserts `analyses` + `skill_matches`, commits.
  - `rescore_candidate(session, engine, candidate, job) -> Analysis` — no LLM call; reuses stored `llm_fit_score`, recomputes deterministic parts.

- [ ] **Step 1: Write the failing test**

`tests/scoring/test_analyzer.py`:
```python
import json
from typing import Any

import httpx
import respx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Analysis
from app.extraction.persist import persist_candidate
from app.extraction.schemas import CandidateExtraction, Education
from app.llm.gateway import Gateway
from app.parsing.pipeline import ParsedDocument
from app.scoring.analyzer import analyze_candidate, rescore_candidate, scoring_view
from app.scoring.engine import ScoringEngine, load_config
from app.scoring.jd import JobRequirements, create_job
from app.scoring.skills import SkillMatcher
from tests.scoring.fakes import FakeEmbedder

GEM = "https://generativelanguage.googleapis.com/v1beta"
RESUME = "Alice Smith. Built FastAPI services in Python on PostgreSQL. B.Tech 2019."


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def test_scoring_view_strips_pii() -> None:
    v = scoring_view({"name": "A", "email": "a@x.com", "phone": "1", "location": "L",
                      "linkedin_url": "u", "skills": ["Python"], "experience": [],
                      "education": [], "projects": [], "certifications": [],
                      "total_years_experience": 4})
    assert set(v) == {"skills", "experience", "education", "projects", "certifications",
                      "total_years_experience"}


@respx.mock
async def test_analyze_persists_and_rescore_reuses_llm_fit(gateway: Gateway, db: Session) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok({
            "llm_fit_score": 78,
            "skill_matches": [
                {"skill": "Python", "present": True, "evidence": "services in Python", "required": True},
                {"skill": "Docker", "present": True, "evidence": "deployed with Docker", "required": True},
            ],
            "strengths": ["APIs"], "weaknesses": ["No Docker"], "summary": "ok",
            "interview_questions": ["a", "b", "c"], "reasoning": "r",
        })
    )
    engine = ScoringEngine(load_config(), SkillMatcher({}, FakeEmbedder({})))
    job = create_job(db, JobRequirements(title="t", required_skills=["Python", "Docker"],
                                         min_years=3, education_level="bachelor"), "jd")
    ext = CandidateExtraction(name="Alice", skills=["Python", "FastAPI"], total_years_experience=4,
                              education=[Education(institution="X", degree="B.Tech")])
    cand = persist_candidate(db, ParsedDocument(RESUME, "pymupdf", len(RESUME), {}), ext, [], "h")

    a = await analyze_candidate(gateway, db, engine, cand, job)
    prompt = json.loads(route.calls[0].request.read())["contents"][0]["parts"][0]["text"]
    assert "Python" in prompt
    assert '"name"' not in prompt and '"email"' not in prompt
    assert a.llm_fit_score == 78 and a.required_skill_coverage == 50.0
    assert a.final_score == round(0.35 * 50 + 0.15 * 100 + 20 + 10 + 0.2 * 78)
    rows = {m.skill.canonical_name: m.present for m in a.skill_matches}
    assert rows == {"Python": True, "Docker": False}  # Docker evidence not in resume

    a2 = await analyze_candidate(gateway, db, engine, cand, job)  # upsert, cached LLM
    assert a2.id == a.id and route.call_count == 1
    assert db.scalar(select(func.count()).select_from(Analysis)) == 1

    engine.config.weights["llm_fit_score"] = 0.0
    r = rescore_candidate(db, engine, cand, job)
    assert r.id == a.id and r.llm_fit_score == 78
    assert r.final_score == round(0.35 * 50 + 0.15 * 100 + 20 + 10)
```

The resume text itself (containing the name) is still passed for evidence quoting — §6.4 withholds identifiers from the *structured judgement input*; the prompt instructs the model to ignore them.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_analyzer.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/prompts/analyze.txt`**

```
You are evaluating a candidate against a job. Score on skills, experience and education ONLY.
Do not consider or infer name, gender, age, nationality, photo, or location.

Job requirements (JSON):
{job}

Candidate profile (JSON, personal identifiers removed):
{candidate}

Candidate resume text (for evidence quotes only):
<<<
{resume}
>>>

Return JSON matching the schema:
- llm_fit_score: 0-100 holistic fit, weighing project relevance and career trajectory.
- skill_matches: one entry per job skill (required and preferred). present=true ONLY if the resume
  text contains evidence; evidence must be an exact phrase copied from the resume text.
- strengths / weaknesses: up to 5 each, concrete.
- summary: 2-3 sentences.
- interview_questions: 3-5 questions probing gaps or verifying claims.
- reasoning: why this score, citing resume content.
```

- [ ] **Step 4: Write `app/scoring/analyzer.py`**

```python
import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, Job, SkillMatchRow
from app.extraction.persist import get_or_create_skill
from app.extraction.schemas import CandidateAnalysis
from app.extraction.validators import validate_analysis
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask
from app.scoring.components import degree_level
from app.scoring.engine import ScoreBreakdown, ScoringEngine
from app.scoring.jd import job_requirements

log = logging.getLogger(__name__)
_SCORING_KEYS = ("skills", "experience", "education", "projects", "certifications",
                 "total_years_experience")


def scoring_view(extraction_json: dict[str, Any]) -> dict[str, Any]:
    return {k: extraction_json.get(k) for k in _SCORING_KEYS}


def _candidate_inputs(candidate: Candidate) -> tuple[list[str], float | None, str | None]:
    ext = candidate.extraction_json or {}
    skills = [cs.skill.canonical_name for cs in candidate.skills]
    degrees = [e.get("degree") for e in ext.get("education", [])]
    return skills, candidate.total_years_experience, degree_level(degrees)


def _upsert(
    session: Session, candidate: Candidate, job: Job, b: ScoreBreakdown,
    llm: CandidateAnalysis | None,
) -> Analysis:
    row = session.scalar(
        select(Analysis).where(Analysis.candidate_id == candidate.id, Analysis.job_id == job.id)
    )
    if row is None:
        row = Analysis(candidate_id=candidate.id, job_id=job.id, summary="", reasoning="")
        session.add(row)
    row.final_score = b.final_score
    row.required_skill_coverage = b.required_skill_coverage
    row.preferred_skill_coverage = b.preferred_skill_coverage
    row.experience_fit = b.experience_fit
    row.education_fit = b.education_fit
    row.llm_fit_score = b.llm_fit_score
    row.recommendation = b.recommendation
    row.scoring_version = b.scoring_version
    if llm is not None:
        row.summary = llm.summary
        row.reasoning = llm.reasoning
        row.strengths = llm.strengths
        row.weaknesses = llm.weaknesses
        row.interview_questions = llm.interview_questions
        session.flush()
        row.skill_matches.clear()
        session.flush()
        for m in llm.skill_matches:
            row.skill_matches.append(SkillMatchRow(
                skill=get_or_create_skill(session, m.skill), present=m.present,
                required=m.required, evidence=m.evidence,
            ))
    session.commit()
    session.refresh(row)
    return row


async def analyze_candidate(
    gateway: Gateway, session: Session, engine: ScoringEngine, candidate: Candidate, job: Job
) -> Analysis:
    reqs = job_requirements(job)
    prompt = render(
        "analyze",
        job=reqs.model_dump_json(),
        candidate=json.dumps(scoring_view(candidate.extraction_json or {})),
        resume=candidate.raw_text[:12_000],
    )
    raw = await gateway.complete(prompt, CandidateAnalysis, LLMTask.ANALYZE)
    llm, hallucinations = validate_analysis(raw, candidate.raw_text)
    if hallucinations:
        log.warning("candidate %s: %d hallucinated evidence strings", candidate.id, hallucinations)
    skills, years, degree = _candidate_inputs(candidate)
    breakdown = engine.score(skills, years, degree, llm.llm_fit_score, reqs)
    return _upsert(session, candidate, job, breakdown, llm)


def rescore_candidate(
    session: Session, engine: ScoringEngine, candidate: Candidate, job: Job
) -> Analysis:
    existing = session.scalar(
        select(Analysis).where(Analysis.candidate_id == candidate.id, Analysis.job_id == job.id)
    )
    llm_fit = existing.llm_fit_score if existing else 0.0
    skills, years, degree = _candidate_inputs(candidate)
    breakdown = engine.score(skills, years, degree, llm_fit, job_requirements(job))
    return _upsert(session, candidate, job, breakdown, None)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/scoring -v && uv run mypy app/llm app/scoring`
Expected: all pass; mypy clean.

- [ ] **Step 6: Commit and tag**

```bash
git add app/scoring/analyzer.py app/llm/prompts/analyze.txt tests/scoring/test_analyzer.py
git commit -m "feat(scoring): analyzer with PII-stripped prompt, evidence check, analysis upsert"
git tag v0.4
```

---

# Phase 5 — Query layer

### Task 25: Query schemas and intent classifier

**Files:**
- Create: `app/query/__init__.py`, `app/query/schemas.py`, `app/query/classifier.py`, `app/llm/prompts/classify.txt`
- Test: `tests/query/__init__.py`, `tests/query/test_classifier.py`

**Interfaces:**
- Produces (`schemas.py`):
  - `class Intent(StrEnum)`: `TOP_N="top_n"`, `FILTER_SKILL="filter_skill"`, `FILTER_ATTRIBUTE="filter_attribute"`, `COMPARE="compare"`, `EXPLAIN_RANKING="explain_ranking"`, `RECOMMEND="recommend"`, `SEMANTIC="semantic"`
  - `Attribute = Literal["total_years_experience", "final_score"]`, `Op = Literal[">", ">=", "<", "<=", "="]`
  - `class QueryPlan(BaseModel)`: `intent: Intent`, `n: int = 5` (1–50), `skill: str | None`, `has_skill: bool = True`, `attribute: Attribute | None`, `op: Op | None`, `value: float | None`, `candidate_a: str | None`, `candidate_b: str | None`
  - `class Source(BaseModel)`: `candidate_id: int`, `name: str | None`, `score: int`
  - `class QueryResponse(BaseModel)`: `answer: str`, `intent: Intent`, `sources: list[Source]`, `latency_ms: int`, `provider_used: str`
- Produces (`classifier.py`): `async classify(gateway: Gateway, question: str) -> QueryPlan` — on `ExtractionFailed` returns `QueryPlan(intent=Intent.SEMANTIC)`.

- [ ] **Step 1: Write the failing test**

`tests/query/test_classifier.py`:
```python
import json
from typing import Any

import httpx
import respx

from app.llm.gateway import Gateway
from app.query.classifier import classify
from app.query.schemas import Intent, QueryPlan

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_classify_uses_query_model_and_parses(gateway: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "filter_skill", "skill": "Docker", "has_skill": False})
    )
    plan = await classify(gateway, "Which candidates are missing Docker?")
    assert plan == QueryPlan(intent=Intent.FILTER_SKILL, skill="Docker", has_skill=False)


@respx.mock
async def test_unparseable_falls_back_to_semantic(gateway: Gateway) -> None:
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "nonsense"})
    )
    plan = await classify(gateway, "tell me something")
    assert plan.intent == Intent.SEMANTIC
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/query/test_classifier.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/query/schemas.py`**

```python
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    TOP_N = "top_n"
    FILTER_SKILL = "filter_skill"
    FILTER_ATTRIBUTE = "filter_attribute"
    COMPARE = "compare"
    EXPLAIN_RANKING = "explain_ranking"
    RECOMMEND = "recommend"
    SEMANTIC = "semantic"


Attribute = Literal["total_years_experience", "final_score"]
Op = Literal[">", ">=", "<", "<=", "="]


class QueryPlan(BaseModel):
    intent: Intent
    n: int = Field(default=5, ge=1, le=50)
    skill: str | None = None
    has_skill: bool = True
    attribute: Attribute | None = None
    op: Op | None = None
    value: float | None = None
    candidate_a: str | None = None
    candidate_b: str | None = None


class Source(BaseModel):
    candidate_id: int
    name: str | None
    score: int


class QueryResponse(BaseModel):
    answer: str
    intent: Intent
    sources: list[Source]
    latency_ms: int
    provider_used: str
```

- [ ] **Step 4: Write `app/llm/prompts/classify.txt`**

```
Classify a recruiter's question about ranked candidates into one intent with parameters.
Return JSON matching the schema.

Intents:
- top_n: "show me the top N candidates" -> n
- filter_skill: "who knows X" (has_skill=true) / "who is missing X" (has_skill=false) -> skill
- filter_attribute: numeric comparison -> attribute (total_years_experience | final_score), op, value
- compare: "compare A and B" -> candidate_a, candidate_b (names as written)
- explain_ranking: "why is A ranked above B" -> candidate_a, candidate_b
- recommend: "who should we interview / best candidate" -> n (default 1)
- semantic: anything else

Question: {question}
```

- [ ] **Step 5: Write `app/query/classifier.py`**

```python
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import ExtractionFailed, LLMTask
from app.query.schemas import Intent, QueryPlan


async def classify(gateway: Gateway, question: str) -> QueryPlan:
    try:
        return await gateway.complete(
            render("classify", question=question[:1000]), QueryPlan, LLMTask.QUERY
        )
    except ExtractionFailed:
        return QueryPlan(intent=Intent.SEMANTIC)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/query/test_classifier.py -v`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add app/query app/llm/prompts/classify.txt tests/query
git commit -m "feat(query): intent schemas and LLM classifier with semantic fallback"
```

---

### Task 26: Typed handlers

**Files:**
- Create: `app/query/handlers.py`
- Test: `tests/query/test_handlers.py`, `tests/query/seed.py`

**Interfaces:**
- Consumes: ORM, `Embedder`, `cosine`, `load_config`.
- Produces:
  - `@dataclass HandlerResult(rows: list[dict[str, Any]], facts: dict[str, Any])` — every row has `candidate_id, name, final_score, recommendation, years, summary, reasoning, skills_present, skills_missing` plus the five component values.
  - `ROW_CAP = 50`
  - `run_handler(session, job_id: int, plan: QueryPlan, embedder: Embedder) -> HandlerResult` — sets `PRAGMA query_only = ON` on the session connection first. `semantic` reads the question text from `plan.skill` (the service copies it there).
  - `find_candidate(session, job_id, name) -> Analysis | None` (case-insensitive substring on candidate name; highest score first).
  - `explain_facts(a: Analysis, b: Analysis) -> dict` — component diffs sorted by absolute weighted gap; `a` is the higher-scored.
- `tests/query/seed.py`: `seed(db) -> int` — one job (required Python, Docker, FastAPI; preferred AWS; 3 years; bachelor) and three candidates: Alice (82, Python+FastAPI+Docker, 4y), Bob (61, Python, 6y), Carol (40, Java, 1y), with analyses and skill_matches. Returns `job_id`.

- [ ] **Step 1: Write `tests/query/seed.py`**

```python
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate, CandidateSkill, SkillMatchRow
from app.extraction.persist import get_or_create_skill
from app.scoring.jd import JobRequirements, create_job


def _candidate(db: Session, name: str, skills: list[str], years: float, text: str) -> Candidate:
    c = Candidate(
        name=name, resume_hash=name.lower(), raw_text=text, extraction_method="pymupdf",
        char_count=len(text), total_years_experience=years, extraction_json={"skills": skills},
    )
    db.add(c)
    db.flush()
    for s in skills:
        c.skills.append(CandidateSkill(skill=get_or_create_skill(db, s), raw_mention=s))
    return c


def _analysis(db: Session, c: Candidate, job_id: int, score: int, req: float, exp: float,
              llm: float, summary: str, present: dict[str, bool]) -> Analysis:
    a = Analysis(
        candidate_id=c.id, job_id=job_id, final_score=score,
        required_skill_coverage=req, preferred_skill_coverage=0.0, experience_fit=exp,
        education_fit=100.0, llm_fit_score=llm,
        recommendation="Shortlist" if score >= 75 else "Consider" if score >= 50 else "Reject",
        summary=summary, reasoning=f"{c.name} reasoning", strengths=["s"], weaknesses=["w"],
        interview_questions=["q1", "q2", "q3"], scoring_version="1.0",
    )
    db.add(a)
    db.flush()
    for skill, ok in present.items():
        a.skill_matches.append(SkillMatchRow(
            skill=get_or_create_skill(db, skill), present=ok, required=True,
            evidence=f"uses {skill}" if ok else None,
        ))
    return a


def seed(db: Session) -> int:
    job = create_job(db, JobRequirements(
        title="Backend", required_skills=["Python", "Docker", "FastAPI"],
        preferred_skills=["AWS"], min_years=3, education_level="bachelor"), "jd text")
    alice = _candidate(db, "Alice Smith", ["Python", "FastAPI", "Docker"], 4, "Alice builds APIs")
    bob = _candidate(db, "Bob Jones", ["Python"], 6, "Bob does data pipelines")
    carol = _candidate(db, "Carol Lee", ["Java"], 1, "Carol writes Java")
    _analysis(db, alice, job.id, 82, 100.0, 100.0, 85, "Strong API engineer",
              {"Python": True, "Docker": True, "FastAPI": True})
    _analysis(db, bob, job.id, 61, 33.3, 100.0, 70, "Data engineer, no containers",
              {"Python": True, "Docker": False, "FastAPI": False})
    _analysis(db, carol, job.id, 40, 0.0, 60.0, 30, "Junior Java developer",
              {"Python": False, "Docker": False, "FastAPI": False})
    db.commit()
    return job.id
```

- [ ] **Step 2: Write the failing test**

`tests/query/test_handlers.py`:
```python
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate
from app.query.handlers import HandlerResult, explain_facts, run_handler
from app.query.schemas import Intent, QueryPlan
from tests.query.seed import seed
from tests.scoring.fakes import FakeEmbedder

EMB = FakeEmbedder({
    "who has container experience": [0, 1, 0, 0],
    "Strong API engineer": [0.2, 0.9, 0, 0],
    "Data engineer, no containers": [0.9, 0.1, 0, 0],
    "Junior Java developer": [0, 0, 1, 0],
})


def names(result: HandlerResult) -> list[str]:
    return [r["name"] for r in result.rows]


@pytest.fixture
def job_id(db: Session) -> int:
    return seed(db)


def test_top_n(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.TOP_N, n=2), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_filter_skill_has(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.FILTER_SKILL, skill="python"), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_filter_skill_missing(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.FILTER_SKILL, skill="Docker", has_skill=False)
    assert names(run_handler(db, job_id, plan, EMB)) == ["Bob Jones", "Carol Lee"]


def test_filter_attribute(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.FILTER_ATTRIBUTE, attribute="total_years_experience",
                     op=">", value=2)
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith", "Bob Jones"]


def test_explain_ranking_facts(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.EXPLAIN_RANKING, candidate_a="Alice", candidate_b="Bob")
    r = run_handler(db, job_id, plan, EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]
    f = r.facts
    assert f["a"]["name"] == "Alice Smith" and f["score_gap"] == 21
    assert f["diffs"][0]["component"] == "required_skill_coverage"
    assert f["diffs"][0]["a"] == 100.0 and f["diffs"][0]["b"] == 33.3
    assert f["a_only_skills"] == ["Docker", "FastAPI"]


def test_compare_orders_higher_first(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.COMPARE, candidate_a="Bob", candidate_b="Alice")
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith", "Bob Jones"]


def test_compare_unknown_name_gives_empty(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.COMPARE, candidate_a="Zed", candidate_b="Bob")
    r = run_handler(db, job_id, plan, EMB)
    assert r.rows == [] and "Zed" in r.facts["error"]


def test_recommend(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.RECOMMEND, n=1), EMB)
    assert names(r) == ["Alice Smith"] and r.rows[0]["reasoning"] == "Alice Smith reasoning"


def test_semantic_ranks_by_similarity(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.SEMANTIC, n=1, skill="who has container experience")
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith"]


def test_semantic_without_question_falls_back_to_rank(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.SEMANTIC, n=2), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_read_only_connection(db: Session, job_id: int) -> None:
    run_handler(db, job_id, QueryPlan(intent=Intent.TOP_N), EMB)
    with pytest.raises(OperationalError):
        db.execute(text("DELETE FROM analyses"))


def test_explain_facts_ordering() -> None:
    a = Analysis(candidate=Candidate(name="A"), final_score=80, required_skill_coverage=100,
                 preferred_skill_coverage=0, experience_fit=60, education_fit=100, llm_fit_score=80)
    b = Analysis(candidate=Candidate(name="B"), final_score=70, required_skill_coverage=60,
                 preferred_skill_coverage=50, experience_fit=100, education_fit=100, llm_fit_score=80)
    comps = [d["component"] for d in explain_facts(a, b)["diffs"]]
    assert comps[0] == "required_skill_coverage"  # 0.35 * 40 = 14 weighted gap, largest
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/query/test_handlers.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Write `app/query/handlers.py`**

```python
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
    session.execute(text("PRAGMA query_only = ON"))
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/query/test_handlers.py -v`
Expected: `12 passed`

- [ ] **Step 6: Commit**

```bash
git add app/query/handlers.py tests/query
git commit -m "feat(query): typed read-only handlers for all seven intents"
```

---

### Task 27: Synthesis and query service

**Files:**
- Create: `app/query/synthesis.py`, `app/query/service.py`, `app/llm/prompts/synthesize.txt`
- Test: `tests/query/test_service.py`

**Interfaces:**
- Produces:
  - `class Answer(BaseModel)`: `answer: str`
  - `async synthesize(gateway, question, plan: QueryPlan, result: HandlerResult) -> str`
  - `async answer_question(gateway, session, job_id, question, embedder) -> QueryResponse` — classify → (semantic: `plan.skill = question`) → `run_handler` → synthesize; empty rows short-circuit to a fixed answer without an LLM call; `provider_used = gateway.last_provider`.

- [ ] **Step 1: Write the failing test**

`tests/query/test_service.py`:
```python
import json
from typing import Any

import httpx
import respx
from sqlalchemy.orm import Session

from app.llm.gateway import Gateway
from app.query.service import answer_question
from tests.query.seed import seed
from tests.scoring.fakes import FakeEmbedder

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@respx.mock
async def test_explain_ranking_end_to_end(gateway: Gateway, db: Session) -> None:
    job_id = seed(db)
    route = respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[
            gem_ok({"intent": "explain_ranking", "candidate_a": "Alice", "candidate_b": "Bob"}),
            gem_ok({"answer": "Alice ranks above Bob mainly on required-skill coverage, 100 vs 33."}),
        ]
    )
    r = await answer_question(gateway, db, job_id, "Why is Alice above Bob?", FakeEmbedder({}))
    assert r.intent == "explain_ranking"
    assert [s.name for s in r.sources] == ["Alice Smith", "Bob Jones"]
    assert r.provider_used == "gemini" and r.latency_ms >= 0
    synth_prompt = json.loads(route.calls[1].request.read())["contents"][0]["parts"][0]["text"]
    assert '"score_gap": 21' in synth_prompt
    assert "answer only from the facts" in synth_prompt.lower()


@respx.mock
async def test_empty_result_skips_synthesis(gateway: Gateway, db: Session) -> None:
    job_id = seed(db)
    route = respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=gem_ok({"intent": "filter_skill", "skill": "COBOL", "has_skill": True})
    )
    r = await answer_question(gateway, db, job_id, "Who knows COBOL?", FakeEmbedder({}))
    assert r.sources == [] and "No candidates" in r.answer and route.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/query/test_service.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `app/llm/prompts/synthesize.txt`**

```
You are answering a recruiter's question about ranked candidates.
Answer ONLY from the facts below. If the facts do not contain the answer, say so plainly.
Numbers in the facts are authoritative; do not recompute or invent them.
Refer to candidates by name. Be concise: 2-5 sentences, with the reasoning visible.

Question: {question}
Intent: {intent}

Facts (JSON):
{facts}

Rows (JSON, ranked):
{rows}

Return JSON: {{"answer": "..."}}
```

- [ ] **Step 4: Write `app/query/synthesis.py`**

```python
import json

from pydantic import BaseModel

from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask
from app.query.handlers import HandlerResult
from app.query.schemas import QueryPlan


class Answer(BaseModel):
    answer: str


async def synthesize(
    gateway: Gateway, question: str, plan: QueryPlan, result: HandlerResult
) -> str:
    prompt = render(
        "synthesize",
        question=question[:1000],
        intent=plan.intent.value,
        facts=json.dumps(result.facts, default=str),
        rows=json.dumps(result.rows, default=str)[:20_000],
    )
    return (await gateway.complete(prompt, Answer, LLMTask.QUERY)).answer
```

- [ ] **Step 5: Write `app/query/service.py`**

```python
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
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/query -v`
Expected: all pass.

- [ ] **Step 7: Commit and tag**

```bash
git add app/query app/llm/prompts/synthesize.txt tests/query
git commit -m "feat(query): synthesis from structured facts and answer_question service"
git tag v0.5
```

---

# Phase 6 — API and UI

### Task 28: REST API and batch processing

**Files:**
- Create: `app/api/batches.py`, `app/api/jobs.py`, `app/api/candidates.py`, `app/api/schemas.py`
- Modify: `app/main.py` (`create_app` takes `engine`; routers; exception mapping), `app/scoring/skills.py` (rename `_embedder` → public `embedder`), `tests/conftest.py` (`client` passes a `ScoringEngine` with `FakeEmbedder`)
- Test: `tests/api/__init__.py`, `tests/api/test_jobs.py`

**Interfaces:**
- Consumes: everything from phases 1–5.
- Produces:
  - `create_app(settings, session_factory, gateway=None, engine: ScoringEngine | None = None) -> FastAPI`; `app.state.engine`, `app.state.embedder`, `app.state.llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm)`.
  - `batches.py`: `FileStatus(filename, status: "queued"|"processing"|"done"|"error", extraction_method, error, candidate_id)`; `BatchStatus(id, job_id, total, done, files, llm_calls)`; `BATCHES: dict[str, BatchStatus]`; `new_batch(job_id, filenames) -> BatchStatus`; `async process_batch(state, batch_id, files: list[tuple[str, bytes]]) -> None` (per file: parse → extract → persist → analyze; errors recorded per file, never raised).
  - Routes per §10. Error mapping: `InvalidPDF` → 400/413 by `.code`; `ExtractionFailed` → 422; `QuotaExhausted` → 429 with `retry_after`; `NoProvider` → 503; unknown job/candidate/batch → 404 `NOT_FOUND`; >50 files → 400 `TOO_MANY_FILES`.
  - `schemas.py`: `JobOut`, `CandidateRow`, `CandidateDetail`, `SkillMatchOut`, `QueryIn`, `row_from(analysis)`, `detail_from(analysis)`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_jobs.py`:
```python
import json
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from tests.parsing.helpers import make_pdf

GEM = "https://generativelanguage.googleapis.com/v1beta"
JD_RESP = {"title": "Backend", "required_skills": ["Python", "Docker"], "preferred_skills": [],
           "min_years": 3, "education_level": "bachelor"}
EXT_RESP = {"name": "Alice Smith", "email": "alice@example.com", "skills": ["Python", "FastAPI"],
            "total_years_experience": 4, "education": [{"institution": "X", "degree": "B.Tech"}]}
AN_RESP = {"llm_fit_score": 80,
           "skill_matches": [{"skill": "Python", "present": True, "evidence": "Python", "required": True},
                             {"skill": "Docker", "present": False, "evidence": None, "required": True}],
           "strengths": ["s"], "weaknesses": ["w"], "summary": "ok",
           "interview_questions": ["a", "b", "c"], "reasoning": "r"}


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def _resume_pdf() -> bytes:
    return make_pdf([[(72, 72, "Alice Smith alice@example.com"),
                      (72, 100, "Skills: Python, FastAPI. Built services in Python. " * 3),
                      (72, 130, "B.Tech 2019")]])


@respx.mock
def test_full_flow(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        side_effect=[gem_ok(JD_RESP), gem_ok(AN_RESP)]
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        return_value=gem_ok(EXT_RESP)
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=[gem_ok({"intent": "top_n", "n": 5}), gem_ok({"answer": "Alice leads."})]
    )

    r = client.post("/api/jobs", data={"text": "We need Python and Docker, 3+ years, bachelor's."})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["required_skills"] == ["Python", "Docker"]
    job_id = job["id"]

    r = client.post(f"/api/jobs/{job_id}/resumes",
                    files=[("files", ("alice.pdf", _resume_pdf(), "application/pdf")),
                           ("files", ("bad.pdf", b"not a pdf", "application/pdf"))])
    assert r.status_code == 202
    batch_id = r.json()["batch_id"]

    b = client.get(f"/api/batches/{batch_id}").json()
    assert b["total"] == 2 and b["done"] == 2
    by_name = {f["filename"]: f for f in b["files"]}
    assert by_name["alice.pdf"]["status"] == "done"
    assert by_name["alice.pdf"]["extraction_method"] == "pymupdf"
    assert by_name["bad.pdf"]["status"] == "error" and "not a PDF" in by_name["bad.pdf"]["error"]

    rows = client.get(f"/api/jobs/{job_id}/candidates").json()
    assert len(rows) == 1 and rows[0]["name"] == "Alice Smith"
    assert rows[0]["matched"] == 1 and rows[0]["missing"] == 1
    cid = rows[0]["candidate_id"]

    d = client.get(f"/api/candidates/{cid}").json()
    assert d["components"]["required_skill_coverage"] == 50.0
    assert d["interview_questions"] == ["a", "b", "c"]

    assert client.get(f"/api/jobs/{job_id}/candidates?min_score=99").json() == []
    assert client.get(f"/api/jobs/{job_id}/candidates?skill=Python").json()[0]["name"] == "Alice Smith"

    q = client.post(f"/api/jobs/{job_id}/query", json={"question": "Top candidates?"}).json()
    assert q["answer"] == "Alice leads." and q["intent"] == "top_n"
    assert q["sources"][0]["candidate_id"] == cid and q["provider_used"] == "gemini"

    r = client.post(f"/api/jobs/{job_id}/rescore")
    assert r.status_code == 200 and r.json()["rescored"] == 1

    r = client.get(f"/api/jobs/{job_id}/export?format=csv")
    assert r.status_code == 200 and "Alice Smith" in r.text
    assert r.headers["content-type"].startswith("text/csv")
    r = client.get(f"/api/jobs/{job_id}/export?format=xlsx")
    assert r.headers["content-type"].startswith("application/vnd.openxmlformats")


def test_job_not_found(client: TestClient) -> None:
    r = client.get("/api/jobs/999")
    assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"


def test_too_many_files(client: TestClient) -> None:
    files = [("files", (f"{i}.pdf", b"%PDF", "application/pdf")) for i in range(51)]
    r = client.post("/api/jobs/1/resumes", files=files)
    assert r.status_code == 400 and r.json()["code"] == "TOO_MANY_FILES"


@respx.mock
def test_query_quota_exhausted_maps_429(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(return_value=gem_ok(JD_RESP))
    job_id = client.post("/api/jobs", data={"text": "jd"}).json()["id"]
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        return_value=httpx.Response(429))
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, headers={"retry-after": "30"}))
    r = client.post(f"/api/jobs/{job_id}/query", json={"question": "top 5"})
    assert r.status_code == 429 and r.json()["code"] == "QUOTA_EXHAUSTED"
    assert r.json()["retry_after"] == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api -v`
Expected: FAIL (`404` / `ModuleNotFoundError`)

- [ ] **Step 3: Write `app/api/schemas.py`**

```python
from typing import Any

from pydantic import BaseModel


class JobOut(BaseModel):
    id: int
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_years: float | None
    education_level: str | None


class CandidateRow(BaseModel):
    candidate_id: int
    name: str | None
    final_score: int
    recommendation: str
    matched: int
    missing: int
    years: float | None
    extraction_method: str


class SkillMatchOut(BaseModel):
    skill: str
    present: bool
    required: bool
    evidence: str | None


class CandidateDetail(CandidateRow):
    email: str | None
    phone: str | None
    location: str | None
    char_count: int
    flags: list[str]
    components: dict[str, float]
    summary: str
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]
    interview_questions: list[str]
    skill_matches: list[SkillMatchOut]


class QueryIn(BaseModel):
    question: str


_COMPONENTS = ("required_skill_coverage", "preferred_skill_coverage", "experience_fit",
               "education_fit", "llm_fit_score")


def row_from(analysis: Any) -> CandidateRow:
    c = analysis.candidate
    return CandidateRow(
        candidate_id=c.id, name=c.name, final_score=analysis.final_score,
        recommendation=analysis.recommendation,
        matched=sum(1 for m in analysis.skill_matches if m.present),
        missing=sum(1 for m in analysis.skill_matches if not m.present),
        years=c.total_years_experience, extraction_method=c.extraction_method,
    )


def detail_from(analysis: Any) -> CandidateDetail:
    c = analysis.candidate
    return CandidateDetail(
        **row_from(analysis).model_dump(),
        email=c.email, phone=c.phone, location=c.location, char_count=c.char_count,
        flags=c.flags or [],
        components={k: getattr(analysis, k) for k in _COMPONENTS},
        summary=analysis.summary, reasoning=analysis.reasoning,
        strengths=analysis.strengths, weaknesses=analysis.weaknesses,
        interview_questions=analysis.interview_questions,
        skill_matches=[SkillMatchOut(skill=m.skill.canonical_name, present=m.present,
                                     required=m.required, evidence=m.evidence)
                       for m in analysis.skill_matches],
    )
```

- [ ] **Step 4: Write `app/api/batches.py`**

```python
import asyncio
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from app.db.models import Job
from app.extraction.extractor import extract_candidate
from app.extraction.persist import persist_candidate, resume_hash
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


async def _process_one(state: Any, batch: BatchStatus, fs: FileStatus, data: bytes) -> None:
    fs.status = "processing"
    try:
        parsed = await asyncio.to_thread(parse_pdf, data)
        fs.extraction_method = parsed.extraction_method
        async with state.llm_semaphore:
            ext, flags = await extract_candidate(state.gateway, parsed.text)
        with state.session_factory() as s:
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            job = s.get(Job, batch.job_id)
            assert job is not None
            async with state.llm_semaphore:
                await analyze_candidate(state.gateway, s, state.engine, cand, job)
            fs.candidate_id = cand.id
        fs.status = "done"
    except InvalidPDF as e:
        fs.status, fs.error = "error", e.message
    except Exception as e:  # one failed resume never blocks the batch
        log.exception("resume %s failed", fs.filename)
        fs.status, fs.error = "error", _friendly(e)
    finally:
        batch.done += 1
        batch.llm_calls = state.gateway.call_count


async def process_batch(state: Any, batch_id: str, files: list[tuple[str, bytes]]) -> None:
    batch = BATCHES[batch_id]
    await asyncio.gather(*(
        _process_one(state, batch, fs, data) for fs, (_, data) in zip(batch.files, files)
    ))
```

- [ ] **Step 5: Write `app/api/jobs.py`**

```python
import csv
import io
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.api.batches import new_batch, process_batch
from app.api.deps import get_db
from app.api.errors import ApiError
from app.api.schemas import CandidateRow, JobOut, QueryIn, row_from
from app.db.models import Analysis, Candidate, CandidateSkill, Job, Skill
from app.parsing.pipeline import parse_pdf, parse_text
from app.parsing.validate import InvalidPDF
from app.query.schemas import QueryResponse
from app.query.service import answer_question
from app.scoring.analyzer import rescore_candidate
from app.scoring.jd import create_job, job_requirements, parse_jd

router = APIRouter(prefix="/api/jobs")
MAX_FILES = 50


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
    request: Request, db: Session = Depends(get_db),
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
    reqs = await parse_jd(request.app.state.gateway, parsed.text)
    return _job_out(create_job(db, reqs, parsed.text))


@router.get("/{job_id}")
async def get_job(job_id: int, db: Session = Depends(get_db)) -> JobOut:
    return _job_out(_job_or_404(db, job_id))


@router.post("/{job_id}/resumes", status_code=202)
async def upload_resumes(
    job_id: int, request: Request, background: BackgroundTasks,
    files: list[UploadFile] = File(...), db: Session = Depends(get_db),
) -> dict[str, str]:
    if len(files) > MAX_FILES:
        raise ApiError(400, "TOO_MANY_FILES", f"Upload at most {MAX_FILES} resumes per batch.")
    _job_or_404(db, job_id)
    payload = [(f.filename or "resume.pdf", await f.read()) for f in files]
    batch = new_batch(job_id, [n for n, _ in payload])
    background.add_task(process_batch, request.app.state, batch.id, payload)
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
    job_id: int, body: QueryIn, request: Request, db: Session = Depends(get_db)
) -> QueryResponse:
    _job_or_404(db, job_id)
    state = request.app.state
    return await answer_question(state.gateway, db, job_id, body.question, state.embedder)


@router.post("/{job_id}/rescore")
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
```

- [ ] **Step 6: Write `app/api/candidates.py`**

```python
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.batches import BATCHES, BatchStatus
from app.api.deps import get_db
from app.api.errors import ApiError
from app.api.schemas import CandidateDetail, detail_from
from app.db.models import Analysis

router = APIRouter(prefix="/api")


@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: int, db: Session = Depends(get_db)) -> CandidateDetail:
    a = db.scalar(select(Analysis).where(Analysis.candidate_id == candidate_id)
                  .order_by(Analysis.created_at.desc()).limit(1))
    if a is None:
        raise ApiError(404, "NOT_FOUND", f"Candidate {candidate_id} not found")
    return detail_from(a)


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> BatchStatus:
    if batch_id not in BATCHES:
        raise ApiError(404, "NOT_FOUND", f"Batch {batch_id} not found")
    return BATCHES[batch_id]
```

- [ ] **Step 7: Rewrite `app/main.py`**

```python
import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from app.api.candidates import router as candidates_router
from app.api.errors import install_error_handlers
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.config import Settings, get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.types import ExtractionFailed, NoProvider, QuotaExhausted
from app.scoring.engine import ScoringEngine


def _err(status: int, code: str, message: str, retry_after: int | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


def create_app(
    settings: Settings,
    session_factory: sessionmaker[Session],
    gateway: Any = None,
    engine: ScoringEngine | None = None,
) -> FastAPI:
    logging.basicConfig(level=settings.log_level)
    app = FastAPI(title="NipunyaMatch", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.gateway = gateway
    if engine is None:
        from app.scoring.embeddings import LocalEmbedder
        from app.scoring.engine import build_engine

        engine = build_engine(LocalEmbedder())
    app.state.engine = engine
    app.state.embedder = engine.matcher.embedder
    app.state.llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm)
    install_error_handlers(app)

    @app.exception_handler(QuotaExhausted)
    async def _quota(_: Request, e: QuotaExhausted) -> JSONResponse:
        ra = int(e.retry_after) if e.retry_after else None
        return _err(429, "QUOTA_EXHAUSTED",
                    "Both LLM providers are rate-limited. Try again later.", ra)

    @app.exception_handler(NoProvider)
    async def _noprov(_: Request, e: NoProvider) -> JSONResponse:
        return _err(503, "NO_PROVIDER", "No LLM provider is configured or reachable. Check .env.")

    @app.exception_handler(ExtractionFailed)
    async def _extract(_: Request, e: ExtractionFailed) -> JSONResponse:
        return _err(422, "EXTRACTION_FAILED", "The model returned unusable output. Try again.")

    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(candidates_router)
    return app


def _default_app() -> FastAPI:
    from app.llm.gateway import build_gateway

    settings = get_settings()
    db_engine = make_engine(settings.database_url)
    init_db(db_engine)
    factory = make_session_factory(db_engine)
    return create_app(settings, factory, build_gateway(settings, factory))


app = _default_app()
```

In `app/scoring/skills.py` rename `self._embedder` → `self.embedder` (three occurrences).

- [ ] **Step 8: Update `tests/conftest.py`**

Add at top: `from pathlib import Path`. Append:
```python
from app.scoring.engine import ScoringEngine, load_config
from app.scoring.skills import SkillMatcher, load_aliases
from tests.scoring.fakes import FakeEmbedder


@pytest.fixture
def scoring_engine() -> ScoringEngine:
    cfg = load_config(Path("config/scoring.yaml"))
    matcher = SkillMatcher(load_aliases(Path("config/skill_aliases.yaml")), FakeEmbedder({}),
                           cfg.embedding_threshold, cfg.embedding_credit)
    return ScoringEngine(cfg, matcher)
```

Replace the `client` fixture:
```python
@pytest.fixture
def client(
    settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway,
    scoring_engine: ScoringEngine,
) -> Iterator[TestClient]:
    from app.api.batches import BATCHES

    BATCHES.clear()
    app = create_app(settings, session_factory, gateway, scoring_engine)
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest -v`
Expected: all pass, including `tests/api` (4 tests) and the earlier `tests/test_health.py`.

- [ ] **Step 10: Commit**

```bash
git add app/api app/main.py app/scoring/skills.py tests/api tests/conftest.py
git commit -m "feat(api): jobs, resume batches, candidates, query, rescore, export"
```

---

### Task 29: Streamlit UI

**Files:**
- Create: `app/ui/__init__.py`, `app/ui/client.py`, `app/ui/Home.py`, `app/ui/pages/1_Setup.py`, `app/ui/pages/2_Processing.py`, `app/ui/pages/3_Candidates.py`, `app/ui/pages/4_Assistant.py`
- Test: `tests/test_ui_compiles.py`

**Interfaces:**
- `client.py`: `API = os.environ.get("API_BASE_URL", "http://localhost:8000")`; `get(path, **params)`, `post(path, **kw)` — thin `httpx` wrappers raising `UiError(message)` with the API's `message` on non-2xx; `health() -> dict`.
- Session state keys: `job` (JobOut dict), `batch_id`, `chat` (list of `{q, a, sources, provider}`).

- [ ] **Step 1: Write the compile-check test**

`tests/test_ui_compiles.py`:
```python
import py_compile
from pathlib import Path

import pytest

UI = sorted(Path("app/ui").rglob("*.py"))


@pytest.mark.parametrize("path", UI, ids=[p.name for p in UI])
def test_ui_file_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)
```

- [ ] **Step 2: Write `app/ui/client.py`**

```python
import os
from typing import Any

import httpx

API = os.environ.get("API_BASE_URL", "http://localhost:8000")


class UiError(Exception):
    pass


def _check(r: httpx.Response) -> Any:
    if r.is_success:
        return r.json()
    try:
        message = r.json().get("message", r.text)
    except ValueError:
        message = r.text
    raise UiError(message)


def get(path: str, **params: Any) -> Any:
    return _check(httpx.get(f"{API}{path}", params=params, timeout=60))


def post(path: str, **kw: Any) -> Any:
    return _check(httpx.post(f"{API}{path}", timeout=300, **kw))


def health() -> dict[str, Any]:
    try:
        result: dict[str, Any] = get("/api/health")
        return result
    except Exception:
        return {"gemini_configured": False, "openrouter_configured": False, "breaker_state": "down"}
```

- [ ] **Step 3: Write `app/ui/Home.py`**

```python
import streamlit as st

from app.ui.client import get, health


def sidebar() -> None:
    h = health()
    dot = "🟢" if h["gemini_configured"] else "🔴"
    extra = " + OpenRouter" if h["openrouter_configured"] else ""
    st.sidebar.markdown(f"{dot} Provider: Gemini{extra}")
    st.sidebar.caption(f"Breaker: {h['breaker_state']}")
    job = st.session_state.get("job")
    if job:
        st.sidebar.markdown(f"**Job:** {job['title']}")
        st.sidebar.markdown(f"**Candidates:** {len(get(f'/api/jobs/{job['id']}/candidates'))}")
    else:
        st.sidebar.info("No job yet. Start on the Setup page.")
    if h["breaker_state"] == "down":
        st.sidebar.error("API not reachable. Run `make dev`.")
    elif not h["gemini_configured"]:
        st.sidebar.error("GEMINI_API_KEY not configured. Set it in .env and restart the API.")


if __name__ == "__main__" or st.runtime.exists():
    st.set_page_config(page_title="NipunyaMatch", layout="wide")
    sidebar()
    st.title("NipunyaMatch")
    st.write("1. Setup a job → 2. Upload resumes → 3. Review candidates → 4. Ask questions.")
```

- [ ] **Step 4: Write `app/ui/pages/1_Setup.py`**

```python
import streamlit as st

from app.ui.client import UiError, post
from app.ui.Home import sidebar

sidebar()
st.header("1 · Setup")

with st.form("jd"):
    text = st.text_area("Paste the job description", height=200)
    pdf = st.file_uploader("…or upload a JD PDF", type=["pdf"])
    if st.form_submit_button("Parse job"):
        try:
            if pdf is not None:
                job = post("/api/jobs", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
            elif text.strip():
                job = post("/api/jobs", data={"text": text})
            else:
                st.warning("Paste a JD or upload a PDF.")
                st.stop()
            st.session_state["job"] = job
            st.success(f"Parsed: {job['title']}")
        except UiError as e:
            st.error(str(e))

job = st.session_state.get("job")
if job:
    st.subheader("Parsed requirements")
    st.write("**Required:**", ", ".join(job["required_skills"]) or "—")
    st.write("**Preferred:**", ", ".join(job["preferred_skills"]) or "—")
    st.write(f"**Min years:** {job['min_years'] or '—'} · **Education:** {job['education_level'] or '—'}")
    st.caption("Mis-parsed? Edit the JD text above and parse again before uploading resumes.")

    files = st.file_uploader("Upload resumes (PDF, up to 50)", type=["pdf"], accept_multiple_files=True)
    if files and st.button(f"Process {len(files)} resume(s)"):
        try:
            r = post(f"/api/jobs/{job['id']}/resumes",
                     files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files])
            st.session_state["batch_id"] = r["batch_id"]
            st.switch_page("pages/2_Processing.py")
        except UiError as e:
            st.error(str(e))
```

- [ ] **Step 5: Write `app/ui/pages/2_Processing.py`**

```python
import time

import streamlit as st

from app.ui.client import get
from app.ui.Home import sidebar

sidebar()
st.header("2 · Processing")

if not st.session_state.get("job"):
    st.info("No job yet. Start on the Setup page.")
    st.stop()
batch_id = st.session_state.get("batch_id")
if not batch_id:
    st.info("No resumes uploaded yet for this job. Upload them on the Setup page.")
    st.stop()

slot = st.empty()
while True:
    b = get(f"/api/batches/{batch_id}")
    with slot.container():
        st.progress(b["done"] / max(b["total"], 1), text=f"{b['done']} / {b['total']} processed")
        st.caption(f"LLM calls so far: {b['llm_calls']}")
        st.dataframe(
            [{"file": f["filename"], "status": f["status"], "method": f["extraction_method"] or "",
              "error": f["error"] or ""} for f in b["files"]],
            use_container_width=True,
        )
        if b["done"] == b["total"]:
            errors = [f for f in b["files"] if f["status"] == "error"]
            if errors and len(errors) == b["total"]:
                st.error("Every resume failed. Errors above name the cause (bad PDF, quota, key).")
            else:
                st.success("Batch complete.")
            if st.button("View candidates"):
                st.switch_page("pages/3_Candidates.py")
            break
    time.sleep(2)
```

- [ ] **Step 6: Write `app/ui/pages/3_Candidates.py`**

```python
import streamlit as st

from app.ui.client import API, get
from app.ui.Home import sidebar

sidebar()
st.header("3 · Candidates")
job = st.session_state.get("job")
if not job:
    st.info("No job yet. Start on the Setup page.")
    st.stop()

c1, c2, c3 = st.columns(3)
min_score = c1.slider("Min score", 0, 100, 0)
skill = c2.text_input("Required skill filter")
min_years = c3.number_input("Min years", 0.0, 50.0, 0.0)
rows = get(f"/api/jobs/{job['id']}/candidates", min_score=min_score, skill=skill or None)
rows = [r for r in rows if (r["years"] or 0) >= min_years]
if not rows:
    st.info("No candidates yet. Upload resumes on the Setup page.")
    st.stop()

COLOR = {"Shortlist": "🟢", "Consider": "🟡", "Reject": "🔴"}
st.dataframe(
    [{"Name": r["name"], "Score": r["final_score"],
      "Rec": f"{COLOR[r['recommendation']]} {r['recommendation']}",
      "Matched": r["matched"], "Missing": r["missing"], "Years": r["years"],
      "Parse": r["extraction_method"]} for r in rows],
    use_container_width=True,
)
st.markdown(f"[Export CSV]({API}/api/jobs/{job['id']}/export?format=csv) · "
            f"[Export XLSX]({API}/api/jobs/{job['id']}/export?format=xlsx)")

for r in rows:
    with st.expander(f"{r['name']} — {r['final_score']} ({r['recommendation']})"):
        d = get(f"/api/candidates/{r['candidate_id']}")
        st.write(d["summary"])
        st.bar_chart(d["components"])
        a, b = st.columns(2)
        a.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in d["strengths"]))
        b.markdown("**Weaknesses**\n" + "\n".join(f"- {w}" for w in d["weaknesses"]))
        st.markdown("**Skills** (evidence quoted from resume text; ❌ = not found in resume)")
        for m in d["skill_matches"]:
            mark = "✅" if m["present"] else "❌"
            req = " *(required)*" if m["required"] else ""
            ev = f" — _{m['evidence']}_" if m["evidence"] else ""
            st.markdown(f"{mark} {m['skill']}{req}{ev}")
        st.markdown("**Interview questions**\n"
                    + "\n".join(f"1. {q}" for q in d["interview_questions"]))
        st.caption(f"Reasoning: {d['reasoning']}")
        st.caption(f"Parse: {d['extraction_method']}, {d['char_count']} chars"
                   + (f" · flags: {', '.join(d['flags'])}" if d["flags"] else ""))
```

- [ ] **Step 7: Write `app/ui/pages/4_Assistant.py`**

```python
import streamlit as st

from app.ui.client import UiError, post
from app.ui.Home import sidebar

sidebar()
st.header("4 · Assistant")
job = st.session_state.get("job")
if not job:
    st.info("No job yet. Start on the Setup page.")
    st.stop()

STARTERS = ["Show me the top 5 candidates", "Which candidates know Python?",
            "Which candidates are missing Docker?", "Recommend the best candidate for interview"]
chat = st.session_state.setdefault("chat", [])


def ask(q: str) -> None:
    try:
        r = post(f"/api/jobs/{job['id']}/query", json={"question": q})
        chat.append({"q": q, "a": r["answer"], "sources": r["sources"], "provider": r["provider_used"]})
    except UiError as e:
        chat.append({"q": q, "a": f"⚠️ {e}", "sources": [], "provider": "—"})


for col, s in zip(st.columns(len(STARTERS)), STARTERS):
    if col.button(s):
        ask(s)

for turn in chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["sources"]:
            st.markdown(" ".join(f"`{s['name']} ({s['score']})`" for s in turn["sources"]))
        st.caption(f"provider: {turn['provider']}")

if q := st.chat_input("Ask about the candidates"):
    ask(q)
    st.rerun()
```

- [ ] **Step 8: Run tests and a manual smoke**

Run: `uv run pytest tests/test_ui_compiles.py -v`
Expected: 6 files compile.

Manual: `make dev` in one terminal, `make ui` in another, open `http://localhost:8501`. Walk Setup → Processing → Candidates → Assistant with two small PDFs. Check the six states: no job; job but no resumes; batch in progress; all resumes failed (upload a `.txt` renamed `.pdf`); no API key (unset `GEMINI_API_KEY`, restart API → sidebar error); quota exhausted (invalid key, no OpenRouter key → per-file "No LLM provider reachable").

- [ ] **Step 9: Commit and tag**

```bash
git add app/ui tests/test_ui_compiles.py
git commit -m "feat(ui): four Streamlit pages with polling, empty states and starter questions"
git tag v0.6
```

---

### Task 30: Integration test — cold start to answer

**Files:**
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the test**

`tests/test_integration.py`:
```python
import json
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from tests.parsing.helpers import make_pdf

GEM = "https://generativelanguage.googleapis.com/v1beta"


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


def pdf(name: str, skills: str) -> bytes:
    return make_pdf([[(72, 72, f"{name} {name.lower()}@example.com"),
                      (72, 100, f"Skills: {skills}. Built systems using {skills}. " * 3),
                      (72, 130, "B.Tech 2018")]])


def analysis(fit: int, present: dict[str, bool]) -> dict[str, Any]:
    return {"llm_fit_score": fit,
            "skill_matches": [{"skill": k, "present": v, "evidence": k if v else None,
                               "required": True} for k, v in present.items()],
            "strengths": ["s"], "weaknesses": ["w"], "summary": f"fit {fit}",
            "interview_questions": ["a", "b", "c"], "reasoning": "r"}


def _route_by_prompt(prompt_key: str, responses: dict[str, dict[str, Any]]) -> Any:
    """Pick a canned response by a substring of the prompt so concurrency order doesn't matter."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())["contents"][0]["parts"][0]["text"]
        for needle, resp in responses.items():
            if needle in body:
                return gem_ok(resp)
        raise AssertionError(f"no canned {prompt_key} response for prompt: {body[:200]}")

    return handler


@respx.mock
def test_cold_start_to_grounded_answer(client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        side_effect=_route_by_prompt("analyze", {
            "Job description:": {"title": "Backend", "required_skills": ["Python", "Docker"],
                                 "preferred_skills": [], "min_years": 2, "education_level": "bachelor"},
            "alice@example.com": analysis(90, {"Python": True, "Docker": True}),
            "bob@example.com": analysis(60, {"Python": True, "Docker": False}),
        })
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash-lite:generateContent").mock(
        side_effect=_route_by_prompt("extract", {
            "alice@example.com": {"name": "Alice", "skills": ["Python", "Docker"],
                                  "total_years_experience": 5,
                                  "education": [{"institution": "X", "degree": "B.Tech"}]},
            "bob@example.com": {"name": "Bob", "skills": ["Python"], "total_years_experience": 3,
                                "education": [{"institution": "X", "degree": "B.Tech"}]},
        })
    )
    respx.post(f"{GEM}/models/gemini-3.5-flash:generateContent").mock(
        side_effect=_route_by_prompt("query", {
            "Classify": {"intent": "explain_ranking", "candidate_a": "Alice", "candidate_b": "Bob"},
            "Facts (JSON)": {"answer": "Alice ranks above Bob on required-skill coverage: Docker present."},
        })
    )

    job_id = client.post("/api/jobs", data={"text": "Python, Docker, 2+ years"}).json()["id"]
    batch = client.post(f"/api/jobs/{job_id}/resumes", files=[
        ("files", ("alice.pdf", pdf("Alice", "Python, Docker"), "application/pdf")),
        ("files", ("bob.pdf", pdf("Bob", "Python"), "application/pdf")),
    ]).json()["batch_id"]
    assert client.get(f"/api/batches/{batch}").json()["done"] == 2

    rows = client.get(f"/api/jobs/{job_id}/candidates").json()
    assert [r["name"] for r in rows] == ["Alice", "Bob"]
    assert rows[0]["final_score"] > rows[1]["final_score"]

    q = client.post(f"/api/jobs/{job_id}/query", json={"question": "Why is Alice above Bob?"}).json()
    assert q["intent"] == "explain_ranking"
    assert [s["name"] for s in q["sources"]] == ["Alice", "Bob"]
    assert "Docker" in q["answer"]
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_integration.py -v`
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: cold-start integration from JD to grounded answer"
```

---

# Phase 7 — Hardening

### Task 31: Fixtures, seed, evaluation script

**Files:**
- Create: `scripts/make_fixtures.py`, `scripts/seed.py`, `scripts/evaluate.py`, `tests/fixtures/jd.txt`, `tests/fixtures/golden/labels.json` (generated), `tests/fixtures/golden/ranking.json` (generated), `tests/fixtures/resumes/*.pdf` (generated, 12 files)
- Test: `tests/test_fixtures.py`

**Interfaces:**
- `labels.json`: `{"<file>.pdf": {"name", "email", "phone", "skills": [...], "start_date": "YYYY-MM", "end_date": "YYYY-MM" | null, "degree", "llm_fit_score"}}`.
- `ranking.json`: `["<file>.pdf", ...]` human ranking, best first.
- `scripts/seed.py`: no LLM. Parses each fixture PDF, builds `CandidateExtraction` from labels, persists, scores with `ScoringEngine` using the labelled `llm_fit_score`; prints the ranked table.
- `scripts/evaluate.py`: needs `GEMINI_API_KEY`. Runs real extraction + analysis, prints per-field accuracy, hallucination rate, Spearman vs `ranking.json`.

- [ ] **Step 1: Write `tests/fixtures/jd.txt`**

```
Senior Backend Engineer

Required: Python, FastAPI, Docker, PostgreSQL, machine learning experience.
Nice to have: AWS, Kubernetes.
Minimum 3 years of professional experience. Bachelor's degree in Computer Science or related field.
```

- [ ] **Step 2: Write `scripts/make_fixtures.py`**

```python
"""Generate 12 synthetic resume PDFs (one two-column, one image-only) plus golden labels."""
import json
from pathlib import Path

import fitz

OUT = Path("tests/fixtures/resumes")
OUT.mkdir(parents=True, exist_ok=True)

PEOPLE = [
    ("Asha Rao", "asha.rao@example.com", "+91 98765 43210",
     ["Python", "FastAPI", "Docker", "PostgreSQL", "TensorFlow"], "2018-06", None, "B.Tech Computer Science", 85),
    ("Ben Carter", "ben.carter@example.com", "+1 415 555 0101",
     ["Python", "Django", "PostgreSQL"], "2016-01", "2023-12", "BSc Software Engineering", 70),
    ("Chen Wei", "chen.wei@example.com", "+86 138 0000 0000",
     ["Java", "Spring", "Kubernetes", "AWS"], "2015-03", None, "M.S. Computer Science", 55),
    ("Dana Ortiz", "dana.ortiz@example.com", "+34 600 000 000",
     ["Python", "scikit-learn", "Pandas", "SQL"], "2021-09", None, "MSc Data Science", 65),
    ("Eli Novak", "eli.novak@example.com", "+420 600 000 000",
     ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"], "2019-02", None, "B.Eng", 88),
    ("Fatima Khan", "fatima.khan@example.com", "+92 300 0000000",
     ["JavaScript", "React", "Node.js"], "2020-05", None, "BS Computer Science", 30),
    ("Grace Obi", "grace.obi@example.com", "+234 800 000 0000",
     ["Python", "Flask", "Docker", "MySQL"], "2017-08", None, "B.Tech IT", 62),
    ("Hiro Tanaka", "hiro.tanaka@example.com", "+81 90 0000 0000",
     ["Go", "Docker", "Kubernetes", "PostgreSQL"], "2014-04", None, "Bachelor of Engineering", 50),
    ("Ivy Larsen", "ivy.larsen@example.com", "+45 20 00 00 00",
     ["Python", "PyTorch", "FastAPI", "Docker"], "2022-01", None, "PhD Machine Learning", 80),
    ("Jonas Meyer", "jonas.meyer@example.com", "+49 151 00000000",
     ["C++", "Python", "Linux"], "2012-10", None, "Diploma Informatik", 40),
    ("Kavya Iyer", "kavya.iyer@example.com", "+91 90000 00000",
     ["Python", "FastAPI", "PostgreSQL", "ML", "Docker", "Kubernetes"], "2019-07", None, "B.Tech CSE", 92),
    ("Leo Santos", "leo.santos@example.com", "+55 11 90000 0000",
     ["PHP", "Laravel", "MySQL"], "2018-03", None, "Associate Degree", 20),
]


def body(name: str, email: str, phone: str, skills: list[str], start: str,
         end: str | None, degree: str) -> list[str]:
    return [
        name, f"{email} | {phone}", "",
        "SKILLS", ", ".join(skills), "",
        "EXPERIENCE", f"Software Engineer, Example Corp ({start} - {end or 'Present'})",
        f"Built and maintained services using {', '.join(skills[:3])}.", "",
        "EDUCATION", f"{degree}, Example University, 2016",
    ]


def write_pdf(path: Path, lines: list[str], two_column: bool = False, image_only: bool = False) -> None:
    doc = fitz.open()
    page = doc.new_page()
    if two_column:
        half = len(lines) // 2
        for i, ln in enumerate(lines[:half]):
            page.insert_text((40, 60 + 16 * i), ln, fontsize=10)
        for i, ln in enumerate(lines[half:]):
            page.insert_text((320, 60 + 16 * i), ln, fontsize=10)
    else:
        for i, ln in enumerate(lines):
            page.insert_text((60, 60 + 16 * i), ln, fontsize=10)
    if image_only:
        pix = page.get_pixmap(dpi=150)
        img_doc = fitz.open()
        p = img_doc.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, pixmap=pix)
        img_doc.save(path)
        return
    doc.save(path)


labels: dict[str, dict[str, object]] = {}
for i, (name, email, phone, skills, start, end, degree, fit) in enumerate(PEOPLE):
    fname = f"{i + 1:02d}_{name.split()[0].lower()}.pdf"
    write_pdf(OUT / fname, body(name, email, phone, skills, start, end, degree),
              two_column=(i == 2), image_only=(i == 9))
    labels[fname] = {"name": name, "email": email, "phone": phone, "skills": skills,
                     "start_date": start, "end_date": end, "degree": degree, "llm_fit_score": fit}

Path("tests/fixtures/golden").mkdir(parents=True, exist_ok=True)
Path("tests/fixtures/golden/labels.json").write_text(json.dumps(labels, indent=2))
Path("tests/fixtures/golden/ranking.json").write_text(json.dumps([
    "11_kavya.pdf", "05_eli.pdf", "01_asha.pdf", "09_ivy.pdf", "02_ben.pdf", "04_dana.pdf",
    "07_grace.pdf", "03_chen.pdf", "08_hiro.pdf", "10_jonas.pdf", "06_fatima.pdf", "12_leo.pdf",
], indent=2))
print(f"wrote {len(labels)} resumes")
```

- [ ] **Step 3: Write `scripts/seed.py`**

```python
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
from app.parsing.pipeline import parse_pdf, parse_text
from app.scoring.analyzer import rescore_candidate
from app.scoring.embeddings import LocalEmbedder
from app.scoring.engine import build_engine
from app.scoring.jd import JobRequirements, create_job

FIX = Path("tests/fixtures")


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
            parsed = parse_pdf(data)
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
```

`seed.py` needs `GEMINI_API_KEY` set to *any* value because `Settings` fails fast; the README says `GEMINI_API_KEY=dummy` is enough for `make seed`.

- [ ] **Step 4: Write `scripts/evaluate.py`**

```python
"""Golden-set evaluation. Requires a real GEMINI_API_KEY. Prints three numbers for the README."""
import asyncio
import json
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.extraction.extractor import extract_candidate
from app.extraction.persist import persist_candidate, resume_hash
from app.extraction.validators import norm
from app.llm.gateway import build_gateway
from app.parsing.pipeline import parse_pdf, parse_text
from app.scoring.analyzer import analyze_candidate
from app.scoring.embeddings import LocalEmbedder
from app.scoring.engine import build_engine
from app.scoring.jd import create_job, parse_jd

FIX = Path("tests/fixtures")


def spearman(a: list[int], b: list[int]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def digits(s: str | None) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


async def main() -> None:
    settings = get_settings()
    db = make_engine("sqlite:///./data/eval.db")
    init_db(db)
    factory = make_session_factory(db)
    gateway = build_gateway(settings, factory)
    engine = build_engine(LocalEmbedder())
    labels = json.loads((FIX / "golden/labels.json").read_text())
    human = json.loads((FIX / "golden/ranking.json").read_text())

    hits = {"name": 0, "email": 0, "phone": 0, "skills": 0}
    claimed = hallucinated = 0
    scores: dict[str, int] = {}
    with factory() as s:
        jd_text = parse_text((FIX / "jd.txt").read_text()).text
        job = create_job(s, await parse_jd(gateway, jd_text), jd_text)
        for fname, lab in labels.items():
            data = (FIX / "resumes" / fname).read_bytes()
            parsed = parse_pdf(data)
            ext, flags = await extract_candidate(gateway, parsed.text)
            hits["name"] += norm(ext.name or "") == norm(lab["name"])
            hits["email"] += (ext.email or "").lower() == lab["email"].lower()
            hits["phone"] += digits(ext.phone) == digits(lab["phone"])
            hits["skills"] += {x.lower() for x in ext.skills} >= {x.lower() for x in lab["skills"]}
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            a = await analyze_candidate(gateway, s, engine, cand, job)
            claimed += len(a.skill_matches)
            hallucinated += sum(1 for m in a.skill_matches
                                if m.evidence and norm(m.evidence) not in norm(parsed.text))
            scores[fname] = a.final_score

    n = len(labels)
    print(f"Field extraction accuracy (n={n}):")
    for k, v in hits.items():
        print(f"  {k:7s} {v / n:.0%}")
    print(f"Hallucination rate: {hallucinated}/{claimed} = {hallucinated / max(claimed, 1):.1%}")
    system_rank = sorted(scores, key=lambda f: scores[f], reverse=True)
    rho = spearman([human.index(f) for f in labels], [system_rank.index(f) for f in labels])
    print(f"Spearman vs human ranking: {rho:.2f}  (12 resumes; wide error bars)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Write `tests/test_fixtures.py`**

```python
import json
from pathlib import Path

import pytest

from app.parsing.pipeline import parse_pdf

FIX = Path("tests/fixtures")
LABELS = json.loads((FIX / "golden/labels.json").read_text())


@pytest.mark.parametrize("fname", sorted(LABELS))
def test_fixture_parses_with_expected_fields(fname: str) -> None:
    if fname == "10_jonas.pdf":
        pytest.skip("image-only fixture needs Tesseract; OCR path covered by unit tests")
    doc = parse_pdf((FIX / "resumes" / fname).read_bytes())
    lab = LABELS[fname]
    assert lab["email"] in doc.text
    for skill in lab["skills"]:
        assert skill in doc.text
    assert doc.extraction_method == "pymupdf"
    assert "skills" in doc.sections and "experience" in doc.sections


def test_two_column_fixture_keeps_columns_ordered() -> None:
    text = parse_pdf((FIX / "resumes" / "03_chen.pdf").read_bytes()).text
    assert text.index("SKILLS") < text.index("EXPERIENCE") < text.index("EDUCATION")
```

- [ ] **Step 6: Run**

Run: `uv run python scripts/make_fixtures.py && uv run pytest tests/test_fixtures.py -v && GEMINI_API_KEY=dummy uv run python scripts/seed.py`
Expected: 12 PDFs written; fixture tests pass (11 pass, 1 skip, +1); seed prints a 12-row ranked table with no network call.

- [ ] **Step 7: Commit**

```bash
git add scripts tests/fixtures tests/test_fixtures.py
git commit -m "feat: golden fixtures, LLM-free seed, evaluation script"
```

---

### Task 32: CI, docs, README, clean-clone check

**Files:**
- Create: `.github/workflows/ci.yml`, `docs/architecture.md`, `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run mypy app/llm app/scoring
      - run: uv run pytest --cov --cov-fail-under=70
```

- [ ] **Step 2: Write `docs/architecture.md`**

Copy the Mermaid diagrams from `docs/specs.md` §2 (layers), §4.4 (failover), §5 (parsing) and §8 (ER). Under each, two sentences: what the diagram shows and the one decision it encodes — gateway is the only provider-aware module; extraction and analysis are separate calls; column-band block sorting; skills normalised into a table.

- [ ] **Step 3: Write `README.md`**

Ten required sections, each sourced from `docs/specs.md`:

1. **Project description and objectives** — §1 goal + definition of done.
2. **System architecture** — §2 diagram; the `generateContent` vs Interactions API decision (§4.2) and its trade-off.
3. **Setup and installation** — `cp .env.example .env`; set `GEMINI_API_KEY` (`https://aistudio.google.com/apikey`); optional `OPENROUTER_API_KEY`; `uv sync`; Tesseract install note for OCR.
4. **Resume parsing approach and library choices** — §5 pipeline; PyMuPDF/pdfplumber/pytesseract; `extraction_method` and `char_count` surfaced in the UI.
5. **Candidate scoring logic and methodology** — §7 formula, components, bands, cap; worked example with `required_skill_coverage = 76` and final `72`; LLM 20% varies slightly between runs.
6. **Database schema overview** — §8 ER diagram, tables, indexes, JSON-column policy; embeddings as float32 BLOBs (sqlite-vec deferred).
7. **NL query handling approach** — §9 intents table; "why not text-to-SQL"; guardrails.
8. **How to run** — `docker compose up` (API :8000, UI :8501); or `make dev` + `make ui`; `GEMINI_API_KEY=dummy make seed` for a populated demo with no real key; `make test`; `uv run python scripts/evaluate.py` for golden-set numbers.
9. **Assumptions** — §1, all eight verbatim.
10. **Known limitations** — §14, all seven verbatim.

Extras: the four starter questions with answers from the seeded DB; evaluation numbers pasted from `scripts/evaluate.py` with the "12 resumes, wide error bars" caveat; a paragraph on the failover design and the `llm_calls` table as evidence; the fairness statement (identifiers withheld from the scoring prompt; assists a human decision, not a hiring decision system); "model IDs verified 2026-09-17".

- [ ] **Step 4: Update `CLAUDE.md`**

Replace the "Planned commands (from spec — verify once scaffolded)" block with:

```
make dev / make ui                      # API :8000, Streamlit :8501
docker compose up                       # same, containerised
GEMINI_API_KEY=dummy make seed          # golden fixtures → ranked table, no LLM call
make test                               # pytest --cov, 70% floor; no network
uv run pytest tests/scoring/test_engine.py::test_worked_example_reproduces
make lint                               # ruff + mypy --strict on app/llm app/scoring
uv run python scripts/evaluate.py       # golden-set metrics (needs a real GEMINI_API_KEY)
```

Point at `docs/specs.md` (contracts) and `implementation_plan.md` (build order). In the architecture line replace `SQLite + sqlite-vec` with `SQLite (embeddings as float32 BLOB, numpy cosine)`.

- [ ] **Step 5: Clean-clone check**

```bash
cd /tmp && rm -rf nm-check && git clone F:/NipunyaMatch nm-check && cd nm-check
cp .env.example .env && echo "GEMINI_API_KEY=dummy" >> .env
uv sync && uv run pytest -q && uv run python scripts/seed.py
```

Expected: tests pass; seed prints 12 ranked rows. Under five minutes on a warm network (first MiniLM download ≈ 90 MB).

- [ ] **Step 6: Commit and tag**

```bash
git add -A
git commit -m "docs: README with all required sections, architecture diagrams, CI"
git tag v0.7
```

---

## Self-review

**Spec coverage** (`docs/specs.md` → tasks): §1 → 32; §2 → 0–3, 32; §3 → 0; §4 → 4–10; §5 → 11–15; §6 → 16–18, 24; §7 → 20–24; §8 → 2, 19, 24; §9 → 25–27; §10 → 28; §11 → 29; §12 → 1; §13 → every task's test, 30, 31, 32; §14 → 32; fairness (§6.4) → 24.

**Names used identically across tasks:** `Gateway.complete`, `LLMTask`, `Provider`, `ExtractionFailed`, `QuotaExhausted`, `NoProvider`, `ParsedDocument`, `parse_pdf`, `parse_text`, `CandidateExtraction`, `CandidateAnalysis`, `validate_extraction`, `validate_analysis`, `norm`, `persist_candidate`, `resume_hash`, `get_or_create_skill`, `SkillMatcher.coverage`, `SkillMatcher.embedder`, `MatchResult`, `JobRequirements`, `create_job`, `job_requirements`, `parse_jd`, `ScoringEngine.score`, `ScoreBreakdown`, `load_config`, `build_engine`, `analyze_candidate`, `rescore_candidate`, `scoring_view`, `QueryPlan`, `Intent`, `run_handler`, `HandlerResult`, `explain_facts`, `answer_question`, `QueryResponse`, `create_app(settings, session_factory, gateway, engine)`, `BATCHES`, `new_batch`, `process_batch`, conftest fixtures `settings`, `engine`, `session_factory`, `db`, `gateway`, `scoring_engine`, `client`.

**Deliberate deviations from `NipunyaMatch-spec.md`** (each marked inline): no `sqlite-vec` (BLOB + numpy), no Alembic (`create_all`), in-process batch dict, `pdfplumber` not wired (PyMuPDF blocks handle the fixtures; add when a table-heavy resume fails), required coverage `76` not `80` in the worked example (final `72` unchanged).

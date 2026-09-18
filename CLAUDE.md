# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Implemented through Phase 7 (tags v0.0–v0.7). Three documents:

- `docs/specs.md` — engineering contracts (schemas, API, DB, scoring, gateway rules). Source of truth for *what* to build.
- `implementation_plan.md` — 33 TDD tasks across 8 phases with test + implementation code per task. Source of truth for *how* and *in what order*. Execute with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
- `NipunyaMatch-spec.md` — original design doc with rationale. Read when a contract needs justification.

This file summarises the decisions that are easy to violate by accident. Deliberate deviations from the original spec (marked `ponytail:` in the plan): embeddings as float32 BLOB + numpy cosine instead of `sqlite-vec`; `create_all` instead of Alembic; in-process batch dict; worked-example required coverage is `76` (0.8 embedding credit), final still `72`.

## Commands

```
make dev / make ui                      # API :8000, Streamlit :8501
docker compose up                       # same, containerised
GEMINI_API_KEY=dummy make seed          # golden fixtures -> ranked table, no LLM call
make test                               # pytest --cov, 70% floor; no network
uv run pytest tests/scoring/test_engine.py::test_worked_example_reproduces
make lint                               # ruff + mypy --strict on app/llm app/scoring
uv run python scripts/evaluate.py       # golden-set metrics (needs a real GEMINI_API_KEY)
uv run python scripts/make_fixtures.py  # regenerate tests/fixtures/resumes + golden labels
```

No `make` on Windows: run the `uv run ...` lines from the Makefile directly.

## Operational gotchas (learned live)

- Gemini free tier: ~20 requests/day **per model**; `gemini-3.8-flash` often 503s. Retry delay is in the 429 *body* (`retry in Ns`), parsed by `app/llm/gemini.py:_retry_after`. Swap models in `.env`, never in code.
- `_strip_for_gemini` must not touch keys inside `properties` — a field named `title` was once deleted. Regression test in `tests/llm/test_gemini.py`.
- OpenRouter `402` (no credits) maps to `QuotaExhausted`, not `NoProvider`.
- Breaker is in-process: once open it stays open 60 s across batches; restarting the API resets it.
- `PRAGMA query_only` must be scoped (`app/query/handlers.py:read_only`) — it sticks to the pooled SQLite connection.
- Streamlit caches imported modules; after editing `app/ui/client.py` or `Home.py` restart `streamlit`, not just reload.
- Windows dev box has no `make`/`tesseract`; run the `uv run …` lines from the Makefile.

## Architecture (six layers, typed boundaries)

`Streamlit UI -> FastAPI -> Ingestion (PDF->text) -> Extraction (LLM, JD-independent) -> Scoring (hybrid) -> SQLite (embeddings as float32 BLOB, numpy cosine)`, plus a Query router over SQL + vector retrieval. Layout: `app/{api,llm,parsing,extraction,scoring,query,db,ui}` (`api/admin.py` = list/delete candidates and jobs; `ui/pages/5_Data.py` fronts it), `config/scoring.yaml`, `tests/fixtures/`.

Load-bearing rules:

- **LLM gateway is the only provider-aware module.** Everything calls `llm/gateway.py: complete(prompt, schema, task) -> ParsedModel`. Gemini primary via `generateContent`, OpenRouter fallback via chat completions. Cache (`sha256(prompt+model+schema_version)`) → retry Gemini on 429/5xx (1s/2s/4s, 3 attempts) → breaker after 5 failures (60s open) → OpenRouter → one JSON repair pass → typed `ExtractionFailed`. Every call logged to `llm_calls`.
- **Model IDs live in config/env, never in code.** Gemini 2.0 models are shut down; do not copy tutorial IDs. Pin explicit stable IDs, not `-latest`.
- **Structured output from one schema.** Pydantic `model_json_schema()` feeds both Gemini `response_schema` and OpenRouter `response_format.json_schema`.
- **Two LLM calls per candidate**, separate schemas: `CandidateExtraction` (cached by resume hash, JD-independent) and `CandidateAnalysis` (per JD). Changing the JD re-scores without re-parsing.
- **Evidence check is non-negotiable.** Every `SkillMatch.evidence` must be a substring of the resume text (whitespace/case-normalised); otherwise `present=false` and log a hallucination event.
- **Scoring is deterministic-first.** `final = 0.35*required + 0.15*preferred + 0.20*experience + 0.10*education + 0.20*llm_fit`; weights in `config/scoring.yaml`, `scoring_version` stored on every `analyses` row. Scores are stored, never recomputed at read time. Skill matching order: exact → alias table → embedding (cosine > 0.75 at 0.8 credit), stop at first hit. Candidates missing >50% required skills capped at "Consider".
- **Query layer is intent routing, not text-to-SQL.** One classifier call → typed handler (`top_n`, `filter_skill`, `filter_attribute`, `compare`, `explain_ranking`, `recommend`, `semantic` fallback). Parameterised SQL on whitelisted columns, read-only connection, 50-row cap. `explain_ranking` diffs component scores in Python before synthesis.
- **Parsing:** PyMuPDF blocks sorted by column band then y (fixes two-column resumes); OCR only when <100 chars/page; store `extraction_method` and `char_count` per candidate.
- **Fairness:** name, gender markers, age, photo, address are extracted for display but withheld from the scoring prompt.
- **Prompts are versioned text files under `app/llm/prompts/`**, not inline f-strings.
- **Tests never hit the network.** Both providers mocked with `respx`.

## Build order

Phases 0–7 in spec: skeleton → gateway → parsing → extraction → scoring → query → UI → hardening. 1–4 are sequential. Gateway first so every later phase gets retry/cache for free. Tag each phase boundary; conventional commits.

## Config

`.env` validated by a Pydantic `Settings` class at startup; missing `GEMINI_API_KEY` must fail fast with a named message. Keep `.env.example` complete and `.env` gitignored.

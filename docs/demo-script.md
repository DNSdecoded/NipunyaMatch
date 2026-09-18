# Demo video script (target 6–8 min)

Covers every item in the brief: upload, extraction, scoring for one resume, data in the DB,
three-plus NL questions. Record at 1920×1080, browser zoom 110 %, mic on.

## Before recording

```
uv run uvicorn app.main:app --port 8000          # terminal 1
uv run streamlit run app/ui/Home.py              # terminal 2
```

- `.env`: `GEMINI_MODEL_ANALYZE=gemini-3.5-flash-lite`, `GEMINI_MODEL_QUERY=gemini-3.5-flash-lite`,
  `MAX_CONCURRENT_LLM=1` (free tier; ~20 calls/day/model). Budget for the take: 1 (JD) + 3
  (resumes; extractions are cached) + 8 (4 questions) = 12 calls. Do one take.
- Open http://localhost:8501 on **Data** page so the "before" state is visible.
- Have `tests/fixtures/jd.txt` open in a text editor and `tests/fixtures/resumes/` in Explorer.
- Optional second window: `sqlite3 data/recruiter.db` or DB Browser for SQLite.

## 0:00 — Intro (30 s)

Home page. Say: local recruitment assistant — FastAPI + Streamlit, SQLite, Gemini via a single
gateway with retry/breaker/fallback. Point at sidebar: provider dot, breaker state, active-job
picker. "Everything the model claims is checked against the resume text; every score decomposes."

## 0:30 — Job description upload (45 s)

**Setup** page. Paste `tests/fixtures/jd.txt` → **Parse job**. Show parsed chips: required
Python, FastAPI, Docker, PostgreSQL, machine learning; preferred AWS, Kubernetes; 3 yrs; bachelor.
Say: one LLM call, structured output from a Pydantic schema, stored in `jobs` + `job_skills`.

## 1:15 — Resume upload + processing (1 min 30 s)

Same page, drag in **three** PDFs: `01_asha.pdf`, `05_eli.pdf`, `06_fatima.pdf` (one strong,
one perfect, one off-profile) → **Process 3 resume(s)**. **Processing** page polls every 2 s:
show status → done, `pymupdf` extraction method, LLM call counter. Say: parse → extract (LLM 1,
JD-independent, cached by resume hash) → score (deterministic) + analyse (LLM 2, per JD).
Mention: bad PDFs get a per-file error row, never block the batch (optionally drop a `.txt`
renamed `.pdf` to show it).

## 2:45 — Candidate extraction + scoring for one resume (2 min)

**Candidates** page. Ranked table: colour bands, matched/missing counts, years, parse method.
Expand **Eli Novak**:
- summary, component bar chart — read the formula: 0.35 required + 0.15 preferred +
  0.20 experience + 0.10 education + 0.20 LLM fit; weights in `config/scoring.yaml`.
- skills list with ✅ + quoted evidence; point at a ❌ and say "evidence wasn't a substring of the
  resume → flipped to not-present and logged as a hallucination".
- strengths / weaknesses / interview questions / reasoning.
- footer: parse method, char count, validator flags.
Expand **Fatima Khan**: Reject, 0/5 required — "missing more than half the required skills caps
at Consider even if the LLM likes them".
Click **Export CSV** briefly.

## 4:45 — Data in the database (45 s)

**Data** page: jobs, all candidates with analyses count. Then (optional) SQLite window:
```sql
SELECT id, name, email, total_years_experience, extraction_method FROM candidates;
SELECT candidate_id, final_score, recommendation, scoring_version FROM analyses WHERE job_id = <id>;
SELECT provider, model, task, status, latency_ms FROM llm_calls ORDER BY id DESC LIMIT 5;
```
Say: analyses store every component + `scoring_version`; `skill_matches` keep the evidence;
`llm_calls` is the audit trail of retries and providers; `chat_turns` persists the assistant.

## 5:30 — Assistant, four questions (2 min)

**Assistant** page. Ask, in this order (starter buttons for the first two):
1. **Show me the top 5 candidates** → `top_n`, source chips.
2. **Which candidates are missing Docker?** → `filter_skill`, has=false.
3. **Which candidates have Machine Learning experience?** → alias-aware (`ML` → Machine Learning).
4. **Why is Eli ranked higher than Asha?** → `explain_ranking`; say "the numbers in the answer
   are computed in Python from the stored components and handed to the model as facts — it
   narrates, it doesn't calculate."
Point at provider caption under each answer. Refresh the page: history persists (from `chat_turns`).

## 7:30 — Wrap (30 s)

README sections list, `pytest` count (142, no network), CI badge, Docker compose, limitations
(free-tier quota, OCR quality, 12-resume eval set). Repo URL on screen.

## If something fails on camera

- `429 QUOTA_EXHAUSTED`: say it out loud — the failover chain and per-file error rows are a
  feature — then switch `.env` model and restart API (`Ctrl-C`, rerun).
- Streamlit stale import: restart `streamlit`, not just reload.

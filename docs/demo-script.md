# Demo video — speaking notes (target 7–9 min)

Covers every item the brief asks for: JD + resume upload, extraction, scoring for one resume,
data in the database, ≥3 natural-language questions. Bold text = what to say; plain = what to do.

## Before recording

```
uv run uvicorn app.main:app --port 8000          # terminal 1
uv run streamlit run app/ui/Home.py              # terminal 2
```

- `.env`: real `GEMINI_API_KEY`, `MAX_CONCURRENT_LLM=1`. Free tier ≈ 20 calls/day/model; this
  take needs ~12 (1 JD + 3 resumes × 2 + 4 questions × 2 minus cache hits). One take.
- Data page → delete any test jobs so the picker is clean. Keep job "Senior Backend Engineer"
  (11 scored) as the pre-existing job to show multi-JD support.
- Have ready: `tests/fixtures/jd.txt` in a text editor; a folder with `05_eli.pdf`,
  `01_asha.pdf`, `06_fatima.pdf`, plus one scanned PDF (e.g. Swapnika's) to show OCR.
- 1920×1080, browser zoom 110 %, mic check.

---

## 0:00 — Opening (30 s) — Home page, no job selected or the existing job

**"This is NipunyaMatch, a recruitment assistant I built for this assessment. It parses resume
PDFs, scores every candidate against a job description with a transparent hybrid formula, stores
everything in SQLite, and answers recruiter questions in plain English with cited sources.
Stack: FastAPI backend, Streamlit UI, Gemini through a single gateway with retry, circuit
breaker and model fallback. Everything runs locally with one API key; there's also a public demo
on Cloud Run."**

Point at the sidebar: provider dot, breaker state, Active-job picker.

## 0:30 — Job description (45 s) — Setup page

Paste `jd.txt` → **Parse job**.

**"One LLM call, structured output validated by a Pydantic schema. It pulled required skills —
Python, FastAPI, Docker, PostgreSQL, machine learning — preferred AWS and Kubernetes, three
years, bachelor's. That's stored in `jobs` and `job_skills`. If the model misreads the JD I
just edit the text and re-parse."**

## 1:15 — Resume upload + processing (1 min 30 s)

Drag in the 3 text PDFs + the scanned one → **Process 4 resumes**. Processing page appears.

**"Each file goes parse → extract → score. Parsing is PyMuPDF with column detection for
two-column layouts, and per-page OCR when a page has under 100 characters — you can see this
one shows `pymupdf+ocr`, that's the scanned resume going through Tesseract. Extraction is LLM
call one: contact details, skills copied verbatim, experience with dates, education. It's
independent of the job and cached by the resume's hash, so changing the JD later re-scores
without re-parsing. Then the scoring engine and LLM call two run per job. The counter is live LLM
calls. A bad PDF gets an error row and never blocks the batch."**

Wait for done → **View candidates**.

## 2:45 — Scoring for one resume (2 min) — Candidates page

**"Ranked table: score, colour-banded recommendation, matched and missing required skills,
years, parse method."** Expand **Eli Novak**.

**"Every score decomposes. Thirty-five percent required-skill coverage, fifteen preferred, twenty
experience fit, ten education, and only twenty percent is the model's holistic judgement — so
eighty percent of the ranking is deterministic and reproducible. Weights live in a YAML file and
the scoring version is stored on every row."**

Point at skills list. **"This is the part I care most about. Every skill the model claims must be
backed by a phrase that literally appears in the resume text. If it isn't — here's one with a
cross — the claim is flipped to not-present and logged as a hallucination. The model can't
invent evidence."**

Scroll: strengths, weaknesses, interview questions, reasoning. **"Skill matching is exact, then an
alias table — JS equals JavaScript, ML equals Machine Learning — then embedding similarity above
0.75 at partial credit. Missing more than half the required skills caps you at Consider no matter
what the LLM thinks."**

Expand **Fatima Khan** briefly: **"Frontend profile against a backend JD: zero of five required,
Reject, and the reasoning says why."** Click **Export CSV**.

## 4:45 — Data in the database (45 s) — Data page

**"All of it is in SQLite, eleven tables."** Tables section: pick `analyses`.

**"One row per candidate per job with all five components, recommendation, scoring version.
`skill_matches` keeps the evidence quotes. `llm_calls` is the audit trail — provider, model,
attempt, latency, status — so retries and fallbacks are visible, not just logged. `llm_cache`
is why re-uploads are free. `chat_turns` persists the assistant history."** Show `llm_calls`.

## 5:30 — Assistant, four questions (2 min) — Assistant page

Ask in order (use the starter buttons for 1 and 2):

1. **Show me the top 5 candidates** — *"intent `top_n`, straight SQL, sources as chips."*
2. **Which candidates are missing Docker?** — *"`filter_skill` with has=false, a join on
   `candidate_skills`."*
3. **Which candidates have Machine Learning experience?** — *"alias-aware: a resume that says
   'ML' or 'TensorFlow' still matches."*
4. **Why is Eli ranked higher than Asha?** — *"`explain_ranking`. The component differences and
   the score gap are computed in Python and handed to the model as facts. It narrates; it doesn't
   do arithmetic. That's deliberate — this is intent routing to typed handlers, not text-to-SQL,
   so the model never writes a query and can't touch a table it shouldn't."*

Refresh the page: **"History persists — it's in the database."**

## 7:30 — Dashboard + wrap (45 s) — Home page

**"The home page summarises the active job: bands, score distribution, coverage per candidate,
top five."** Switch the job picker to the other JD: **"Multiple job descriptions coexist; a
resume uploaded to two jobs is stored once and scored twice."**

**"Repo: README with all ten sections, 157 tests that never touch the network, CI, Docker
Compose, one-command Cloud Run deploy, and a live demo where visitors bring their own key.
Known limits are in the README — OCR quality, non-English resumes, free-tier quotas, and a
twelve-resume evaluation set with wide error bars. Thanks."** Repo URL on screen.

---

## If something goes wrong on camera

- **429 / quota**: say it — *"free-tier quota; the gateway falls back to another Gemini model and
  reports honestly when both are spent"* — then switch `GEMINI_MODEL_ANALYZE` in `.env` and
  restart the API.
- **Sidebar says API not reachable**: terminal 1 isn't up.
- **Page shows stale code**: restart `streamlit`, not just reload.

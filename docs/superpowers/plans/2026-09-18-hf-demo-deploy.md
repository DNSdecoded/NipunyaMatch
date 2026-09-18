# HF Spaces Public Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship NipunyaMatch as a free public Hugging Face Docker Space where visitors use their own Gemini key, redeployed from GitHub `main` on every green CI run.

**Architecture:** One container runs uvicorn (127.0.0.1:8000) and Streamlit (0.0.0.0:7860). The API resolves a per-visitor `Gateway` from an `X-Gemini-Key` header (bounded in-process pool); `DEMO_MODE=true` turns off deletes/rescore and caps uploads. A GitHub Action prepends HF front-matter to the README and force-pushes `main` to the Space.

**Tech Stack:** existing FastAPI + Streamlit + SQLite app; Docker; GitHub Actions; Hugging Face Spaces (Docker SDK).

**Spec:** `docs/superpowers/specs/2026-09-18-hf-demo-deploy-design.md`

## Global Constraints

- `GEMINI_API_KEY` stays required at startup (Settings fail-fast unchanged). On the Space it is a placeholder.
- Visitor traffic never uses the server key when `DEMO_MODE=true`; missing header → `401 NO_API_KEY`.
- Per-key gateway pool bounded to 100 entries (drop oldest). OpenRouter is `None` for visitor gateways.
- Demo caps: 10 files per batch, 5 MB per file. Local defaults unchanged (50 files, 10 MB).
- Demo forbids `DELETE /api/jobs/{id}`, `DELETE /api/candidates`, `DELETE /api/candidates/{id}`, `POST /api/jobs/{id}/rescore` → `403 DEMO_READ_ONLY`.
- Single exposed port 7860; SQLite is ephemeral, seeded on boot by `scripts/seed.py` (no LLM).
- All commands `uv run …`; tests never hit the network (`respx`); `ruff check .` and `mypy app/llm app/scoring` clean.
- Conventional commits, one per task.

---

### Task 1: `demo_mode` setting and health flag

**Files:**
- Modify: `app/config.py` (after `log_level`)
- Modify: `app/api/health.py`
- Modify: `.env.example`
- Test: `tests/api/test_demo.py` (new)

**Interfaces:**
- Produces: `Settings.demo_mode: bool = False` (env `DEMO_MODE`); `GET /api/health` body gains `"demo_mode": bool`; test fixtures `demo_settings`, `demo_client`.

- [ ] **Step 1: Write the failing test**

`tests/api/test_demo.py`:
```python
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.llm.gateway import Gateway
from app.main import create_app
from app.scoring.engine import ScoringEngine

GEM = "https://generativelanguage.googleapis.com/v1beta"
JD_RESP = {"title": "Backend", "required_skills": ["Python"], "preferred_skills": [],
           "min_years": None, "education_level": None}


def gem_ok(obj: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}
    )


@pytest.fixture
def demo_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"demo_mode": True})


@pytest.fixture
def demo_client(
    demo_settings: Settings, session_factory: sessionmaker[Session], gateway: Gateway,
    scoring_engine: ScoringEngine,
) -> Iterator[TestClient]:
    from app.api.batches import BATCHES

    BATCHES.clear()
    app = create_app(demo_settings, session_factory, gateway, scoring_engine)
    with TestClient(app) as c:
        yield c


def test_health_reports_demo_mode(client: TestClient, demo_client: TestClient) -> None:
    assert client.get("/api/health").json()["demo_mode"] is False
    assert demo_client.get("/api/health").json()["demo_mode"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_demo.py -v`
Expected: FAIL with `KeyError: 'demo_mode'`.

- [ ] **Step 3: Add the setting**

In `app/config.py`, after `log_level: str = "INFO"` add:
```python
    demo_mode: bool = False  # public demo: BYO key, no deletes, small upload caps
```

In `app/api/health.py` replace the return dict with:
```python
    return {
        "gemini_configured": bool(settings.gemini_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "breaker_state": gateway.breaker.state if gateway else "closed",
        "demo_mode": settings.demo_mode,
    }
```

Append to `.env.example`:
```
DEMO_MODE=false
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/api/test_demo.py tests/test_health.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/api/health.py .env.example tests/api/test_demo.py
git commit -m "feat(config): DEMO_MODE setting surfaced on /api/health"
```

---

### Task 2: Per-visitor gateway dependency

**Files:**
- Modify: `app/api/deps.py` (replace whole file)
- Modify: `app/main.py` (add `app.state.gateways = {}` after `app.state.gateway = gateway`)
- Modify: `app/api/jobs.py` (`create`, `upload_resumes`, `query`)
- Modify: `app/api/candidates.py` (`reanalyze`)
- Modify: `app/api/batches.py` (`_process_one`, `process_batch`)
- Test: `tests/api/test_demo.py`

**Interfaces:**
- Produces: `deps.get_gateway(request) -> Gateway`; `deps.MAX_VISITOR_GATEWAYS = 100`; `app.state.gateways: dict[str, Gateway]`; `process_batch(state, gateway, batch_id, files)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_demo.py`:
```python
@respx.mock
def test_header_key_builds_per_visitor_gateway(client: TestClient) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok(JD_RESP)
    )
    for _ in range(2):
        r = client.post("/api/jobs", data={"text": "jd A"}, headers={"X-Gemini-Key": "visitor-a"})
        assert r.status_code == 201, r.text
    client.post("/api/jobs", data={"text": "jd B"}, headers={"X-Gemini-Key": "visitor-b"})
    sent = [c.request.headers["x-goog-api-key"] for c in route.calls]
    assert sent == ["visitor-a", "visitor-b"]  # second A call was a cache hit
    assert len(client.app.state.gateways) == 2


@respx.mock
def test_no_header_uses_server_gateway_outside_demo(client: TestClient) -> None:
    route = respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
        return_value=gem_ok(JD_RESP)
    )
    assert client.post("/api/jobs", data={"text": "jd"}).status_code == 201
    assert route.calls[0].request.headers["x-goog-api-key"] == "k"  # conftest gateway key


def test_demo_without_key_is_401(demo_client: TestClient) -> None:
    r = demo_client.post("/api/jobs", data={"text": "jd"})
    assert r.status_code == 401 and r.json()["code"] == "NO_API_KEY"
    assert "aistudio.google.com" in r.json()["message"]
    r = demo_client.post("/api/jobs/1/query", json={"question": "top 5"})
    assert r.status_code == 401


def test_gateway_pool_is_bounded(client: TestClient) -> None:
    from app.api.deps import MAX_VISITOR_GATEWAYS

    with respx.mock:
        respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(
            return_value=gem_ok(JD_RESP)
        )
        for i in range(MAX_VISITOR_GATEWAYS + 5):
            client.post("/api/jobs", data={"text": f"jd {i}"}, headers={"X-Gemini-Key": f"k{i}"})
    assert len(client.app.state.gateways) == MAX_VISITOR_GATEWAYS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_demo.py -v`
Expected: the four new tests FAIL (`AttributeError: ... 'gateways'`, wrong header, 201 instead of 401).

- [ ] **Step 3: Replace `app/api/deps.py`**

```python
import hashlib
from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.api.errors import ApiError
from app.llm.gateway import Gateway, build_gateway

MAX_VISITOR_GATEWAYS = 100  # ponytail: in-process pool, drop-oldest; enough for a demo


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_gateway(request: Request) -> Gateway:
    """Server gateway locally; per-visitor gateway (own key, no OpenRouter) when X-Gemini-Key
    is sent. In demo mode the header is mandatory so the server key is never spent."""
    state = request.app.state
    key = request.headers.get("x-gemini-key", "").strip()
    if not key:
        if state.settings.demo_mode:
            raise ApiError(
                401, "NO_API_KEY",
                "Enter your Gemini API key in the sidebar "
                "(free at https://aistudio.google.com/apikey).",
            )
        gateway: Gateway = state.gateway
        return gateway
    pool: dict[str, Gateway] = state.gateways
    digest = hashlib.sha256(key.encode()).hexdigest()
    gw = pool.get(digest)
    if gw is None:
        if len(pool) >= MAX_VISITOR_GATEWAYS:
            pool.pop(next(iter(pool)))  # dicts keep insertion order: drop the oldest
        visitor = state.settings.model_copy(
            update={"gemini_api_key": key, "openrouter_api_key": None}
        )
        gw = pool[digest] = build_gateway(visitor, state.session_factory)
    return gw
```

- [ ] **Step 4: Initialise the pool in `app/main.py`**

After `app.state.gateway = gateway` add:
```python
    app.state.gateways = {}  # visitor key digest -> Gateway (see api/deps.get_gateway)
```

- [ ] **Step 5: Route the LLM-calling endpoints through the dependency**

`app/api/jobs.py` — change the import line to `from app.api.deps import get_db, get_gateway`
and add `from app.llm.gateway import Gateway`.

`create` — new signature (drop `request`), and the `parse_jd` call:
```python
@router.post("", status_code=201)
async def create(
    db: Session = Depends(get_db), gateway: Gateway = Depends(get_gateway),
    text: str | None = Form(None), file: UploadFile | None = File(None),
) -> JobOut:
```
```python
    reqs = await parse_jd(gateway, parsed.text)
```

`upload_resumes` — add `gateway: Gateway = Depends(get_gateway),` to the parameters; change
the background call:
```python
    background.add_task(process_batch, request.app.state, gateway, batch.id, payload)
```

`query` — add `gateway: Gateway = Depends(get_gateway),`; replace the `answer_question` call:
```python
    r = await answer_question(gateway, db, job_id, body.question, state.embedder)
```
(the `ChatTurn` persistence lines stay).

`app/api/candidates.py` — add `get_gateway` to the deps import and
`from app.llm.gateway import Gateway`; `reanalyze` gains
`gateway: Gateway = Depends(get_gateway),` and calls
`analyze_candidate(gateway, db, state.engine, cand, job, fresh=True)`.

`app/api/batches.py` — add `from app.llm.gateway import Gateway`; new signatures and bodies:
```python
async def _process_one(
    state: Any, gateway: Gateway, batch: BatchStatus, fs: FileStatus, data: bytes
) -> None:
    fs.status = "processing"
    try:
        parsed = await asyncio.to_thread(parse_pdf, data)
        fs.extraction_method = parsed.extraction_method
        async with state.llm_semaphore:
            ext, flags = await extract_candidate(gateway, parsed.text)
        with state.session_factory() as s:
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            job = s.get(Job, batch.job_id)
            assert job is not None
            async with state.llm_semaphore:
                await analyze_candidate(gateway, s, state.engine, cand, job)
            fs.candidate_id = cand.id
        fs.status = "done"
    except InvalidPDF as e:
        fs.status, fs.error = "error", e.message
    except Exception as e:  # one failed resume never blocks the batch
        log.exception("resume %s failed", fs.filename)
        fs.status, fs.error = "error", _friendly(e)
    finally:
        batch.done += 1
        batch.llm_calls = gateway.call_count


async def process_batch(
    state: Any, gateway: Gateway, batch_id: str, files: list[tuple[str, bytes]]
) -> None:
    batch = BATCHES[batch_id]
    pairs = zip(batch.files, files, strict=True)
    await asyncio.gather(
        *(_process_one(state, gateway, batch, fs, data) for fs, (_, data) in pairs)
    )
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy app/llm app/scoring`
Expected: all pass (existing API tests send no header and still use the conftest gateway).

- [ ] **Step 7: Commit**

```bash
git add app/api/deps.py app/main.py app/api/jobs.py app/api/candidates.py app/api/batches.py tests/api/test_demo.py
git commit -m "feat(api): per-visitor Gemini key via X-Gemini-Key header; 401 in demo mode"
```

---

### Task 3: Demo-mode guards and upload caps

**Files:**
- Modify: `app/api/deps.py` (append `forbid_in_demo`)
- Modify: `app/api/admin.py` (three delete routes)
- Modify: `app/api/jobs.py` (`rescore`, `upload_resumes`, constants)
- Test: `tests/api/test_demo.py`

**Interfaces:**
- Produces: `deps.forbid_in_demo(request) -> None` (FastAPI dependency); `DEMO_MAX_FILES = 10`, `DEMO_MAX_BYTES = 5 * 1024 * 1024` in `app/api/jobs.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_demo.py`:
```python
def test_demo_forbids_deletes_and_rescore(demo_client: TestClient) -> None:
    for method, path in [("DELETE", "/api/jobs/1"), ("DELETE", "/api/candidates"),
                         ("DELETE", "/api/candidates/1"), ("POST", "/api/jobs/1/rescore")]:
        r = demo_client.request(method, path)
        assert r.status_code == 403 and r.json()["code"] == "DEMO_READ_ONLY", path


@respx.mock
def test_demo_upload_caps(demo_client: TestClient) -> None:
    respx.post(f"{GEM}/models/gemini-3.8-flash:generateContent").mock(return_value=gem_ok(JD_RESP))
    h = {"X-Gemini-Key": "visitor"}
    job_id = demo_client.post("/api/jobs", data={"text": "jd"}, headers=h).json()["id"]
    many = [("files", (f"{i}.pdf", b"%PDF", "application/pdf")) for i in range(11)]
    r = demo_client.post(f"/api/jobs/{job_id}/resumes", files=many, headers=h)
    assert r.status_code == 400 and r.json()["code"] == "TOO_MANY_FILES"
    big = [("files", ("big.pdf", b"%PDF" + b"0" * (5 * 1024 * 1024), "application/pdf"))]
    r = demo_client.post(f"/api/jobs/{job_id}/resumes", files=big, headers=h)
    assert r.status_code == 413 and r.json()["code"] == "FILE_TOO_LARGE"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_demo.py -v -k "forbids or caps"`
Expected: FAIL (404/200 instead of 403; 202 instead of 400/413).

- [ ] **Step 3: Append the dependency to `app/api/deps.py`**

```python
def forbid_in_demo(request: Request) -> None:
    if request.app.state.settings.demo_mode:
        raise ApiError(
            403, "DEMO_READ_ONLY", "Deletes and rescoring are disabled on the public demo."
        )
```

- [ ] **Step 4: Apply it**

`app/api/admin.py`: change the import to `from app.api.deps import forbid_in_demo, get_db`
and add `dependencies=[Depends(forbid_in_demo)]` to each of the three `@router.delete(...)`
decorators, e.g.
```python
@router.delete("/candidates/{candidate_id}", dependencies=[Depends(forbid_in_demo)])
```

`app/api/jobs.py`: add `forbid_in_demo` to the deps import; decorate rescore:
```python
@router.post("/{job_id}/rescore", dependencies=[Depends(forbid_in_demo)])
```
Add under `MAX_FILES = 50`:
```python
DEMO_MAX_FILES = 10
DEMO_MAX_BYTES = 5 * 1024 * 1024
```
In `upload_resumes` replace the file-count check and payload build with:
```python
    demo = request.app.state.settings.demo_mode
    max_files = DEMO_MAX_FILES if demo else MAX_FILES
    if len(files) > max_files:
        raise ApiError(400, "TOO_MANY_FILES", f"Upload at most {max_files} resumes per batch.")
    _job_or_404(db, job_id)
    payload = [(f.filename or "resume.pdf", await f.read()) for f in files]
    if demo and any(len(data) >= DEMO_MAX_BYTES for _, data in payload):
        raise ApiError(413, "FILE_TOO_LARGE", "Demo limit is 5 MB per resume.")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/api -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app/api/deps.py app/api/admin.py app/api/jobs.py tests/api/test_demo.py
git commit -m "feat(api): demo mode forbids deletes/rescore and caps uploads"
```

---

### Task 4: UI — visitor key input, demo banner, hidden destructive buttons

**Files:**
- Modify: `app/ui/client.py`
- Modify: `app/ui/Home.py` (`sidebar`)
- Modify: `app/ui/pages/3_Candidates.py` (Rescore button)
- Modify: `app/ui/pages/5_Data.py` (delete buttons, danger zone)
- Test: `tests/test_ui_compiles.py` (already parametrised over `app/ui/**/*.py`; no new test file)

**Interfaces:**
- Produces: `client._headers() -> dict[str, str]` sending `X-Gemini-Key` from `st.session_state["api_key"]`; `client.health()` result includes `demo_mode`; session keys `api_key`, `demo_mode`.

- [ ] **Step 1: Send the key from `app/ui/client.py`**

Add after `class UiError`:
```python
def _headers() -> dict[str, str]:
    """Visitor key lives only in the Streamlit session; sent as X-Gemini-Key when present."""
    try:
        import streamlit as st

        key = st.session_state.get("api_key")
    except Exception:  # not running under streamlit (tests, scripts)
        key = None
    return {"X-Gemini-Key": str(key)} if key else {}
```
Change the three request helpers to pass headers:
```python
def get(path: str, **params: Any) -> Any:
    return _check(httpx.get(f"{API}{path}", params=params, headers=_headers(), timeout=60))


def post(path: str, **kw: Any) -> Any:
    return _check(httpx.post(f"{API}{path}", headers=_headers(), timeout=300, **kw))


def delete(path: str) -> Any:
    return _check(httpx.delete(f"{API}{path}", headers=_headers(), timeout=60))
```
In `health()` change the fallback dict to also include `"demo_mode": False`.

- [ ] **Step 2: Sidebar in `app/ui/Home.py`**

Inside `sidebar()`, right after `h = health()` add:
```python
    demo = bool(h.get("demo_mode"))
    st.session_state["demo_mode"] = demo
    if demo:
        st.sidebar.warning(
            "Public demo — bring your own Gemini key. Data is shared and resets on restart."
        )
    if demo or st.session_state.get("api_key"):
        st.session_state["api_key"] = st.sidebar.text_input(
            "Your Gemini API key", type="password", key="api_key_input",
            value=st.session_state.get("api_key", ""),
            help="Free at https://aistudio.google.com/apikey — kept only in this browser session.",
        ).strip()
        if not st.session_state["api_key"]:
            st.sidebar.caption("Enter a key to parse jobs, upload resumes, or ask questions.")
```
Keep the rest of `sidebar()` as is.

- [ ] **Step 3: Hide destructive buttons**

`app/ui/pages/3_Candidates.py` — wrap the Rescore block:
```python
t1, t2 = st.columns([1, 4])
if not st.session_state.get("demo_mode") and t1.button(
    "Rescore all (no LLM)", help="Recompute deterministic parts with current weights"
):
```
(the `try:` body under it is unchanged).

`app/ui/pages/5_Data.py` — after the `st.caption("Everything in the local SQLite DB ...")` line add:
```python
READ_ONLY = bool(st.session_state.get("demo_mode"))
if READ_ONLY:
    st.info("Public demo: deletes are disabled. Run the app locally for full control.")
```
Then guard each delete button and the danger zone:
```python
    if not READ_ONLY and c2.button("Delete", key=f"job-{j['id']}"):
```
```python
    if not READ_ONLY and c2.button("Delete", key=f"cand-{r['candidate_id']}"):
```
```python
if not READ_ONLY:
    st.divider()
    st.subheader("Danger zone")
    confirm = st.checkbox(f"Yes, delete all {len(rows)} candidates, their resumes and analyses")
    if st.button("Clear all candidates", type="primary", disabled=not confirm):
        try:
            r = delete("/api/candidates")
            st.toast(f"Deleted {r['deleted_candidates']} candidates, {r['deleted_analyses']} analyses")
            _refresh()
        except UiError as e:
            st.error(str(e))
```

- [ ] **Step 4: Verify**

Run: `uv run pytest tests/test_ui_compiles.py -q && uv run ruff check .`
Expected: pass. Manual: `DEMO_MODE=true uv run uvicorn app.main:app --port 8000` + `uv run streamlit run app/ui/Home.py`; sidebar shows the banner and key field; Setup without a key shows the 401 message; with a real key the flow works; Data page has no buttons.

- [ ] **Step 5: Commit**

```bash
git add app/ui
git commit -m "feat(ui): visitor API key input, demo banner, read-only demo pages"
```

---

### Task 5: Single-container entrypoint and Dockerfile

**Files:**
- Create: `entrypoint.sh`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Create: `.hf/README-frontmatter.yml`

**Interfaces:**
- Produces: image serving Streamlit on `$PORT` (default 7860) with the API on 127.0.0.1:8000; `.hf/README-frontmatter.yml` consumed by Task 6.

- [ ] **Step 1: Write `entrypoint.sh`**

```bash
#!/usr/bin/env sh
set -e
export API_BASE_URL="http://127.0.0.1:8000"
if [ ! -f data/recruiter.db ]; then
  echo "seeding demo data (no LLM calls)"
  uv run python scripts/seed.py
fi
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
exec uv run streamlit run app/ui/Home.py \
  --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
```
After `git add`, run `git update-index --chmod=+x entrypoint.sh` (Windows checkouts do not
preserve the bit; the index flag is what the Docker build sees).

- [ ] **Step 2: Replace `Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV HF_HOME=/app/.cache UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Bake the embedding model so the first request is not a 90 MB download.
RUN uv run --no-project python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
RUN uv sync --frozen --no-dev && chmod +x entrypoint.sh && mkdir -p data && chmod -R a+rwX /app
EXPOSE 7860 8000 8501
CMD ["./entrypoint.sh"]
```

- [ ] **Step 3: Keep compose two-service**

`docker-compose.yml` — the `api` service needs its command back explicitly now that `CMD` is
the entrypoint:
```yaml
services:
  api:
    build: .
    env_file: .env
    command: ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
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

- [ ] **Step 4: Write `.hf/README-frontmatter.yml`**

```yaml
---
title: NipunyaMatch
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Resume screening with evidence-checked AI scoring
---
```

- [ ] **Step 5: Build and smoke locally**

Run: `docker build -t nipunyamatch . && docker run --rm -p 7860:7860 -e GEMINI_API_KEY=demo-unused -e DEMO_MODE=true nipunyamatch`
Expected: log shows "seeding demo data", 12-row ranked table, then the Streamlit URL. Open
http://localhost:7860 — sidebar banner + key field; Candidates shows the seeded job. (Skip if
Docker is not installed locally; the Space build is the real test.)

- [ ] **Step 6: Commit**

```bash
git add entrypoint.sh Dockerfile docker-compose.yml .hf/README-frontmatter.yml
git update-index --chmod=+x entrypoint.sh
git commit -m "build: single-container entrypoint for HF Spaces, baked MiniLM, HF front-matter"
```

---

### Task 6: Deploy workflow and README section

**Files:**
- Create: `.github/workflows/deploy-hf.yml`
- Modify: `README.md` (add "Live demo" under Demo; note in §8)

**Interfaces:**
- Consumes: `.hf/README-frontmatter.yml`; GitHub secret `HF_TOKEN` (write); Space `DNSdecoded/NipunyaMatch` with secrets `GEMINI_API_KEY=demo-unused`, `DEMO_MODE=true`.

- [ ] **Step 1: Write `.github/workflows/deploy-hf.yml`**

```yaml
name: deploy-hf
on:
  workflow_run:
    workflows: [ci]
    types: [completed]
    branches: [main]
jobs:
  push-space:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
          fetch-depth: 0
      - name: Prepend Hugging Face front-matter to README
        run: |
          { cat .hf/README-frontmatter.yml; echo; cat README.md; } > README.hf && mv README.hf README.md
          git config user.name "github-actions"
          git config user.email "actions@users.noreply.github.com"
          git commit -am "hf: add Space front-matter"
      - name: Push to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push --force "https://DNSdecoded:${HF_TOKEN}@huggingface.co/spaces/DNSdecoded/NipunyaMatch" HEAD:main
```

- [ ] **Step 2: README**

Under `## Demo`, after the GIF paragraph, add:
```markdown
**Live demo:** https://huggingface.co/spaces/DNSdecoded/NipunyaMatch — bring your own free
Gemini key (sidebar). Data is shared between visitors and resets when the Space restarts;
deletes are disabled there.
```
In `## 8. How to run` add a final line to the command block:
```
docker build -t nipunyamatch . && docker run -p 7860:7860 -e GEMINI_API_KEY=x -e DEMO_MODE=true nipunyamatch   # single container (HF Spaces layout)
```

- [ ] **Step 3: One-time manual setup (maintainer)**

1. https://huggingface.co/new-space → owner `DNSdecoded`, name `NipunyaMatch`, SDK **Docker**, public.
2. Space → Settings → Variables and secrets: `GEMINI_API_KEY=demo-unused`, `DEMO_MODE=true`.
3. https://huggingface.co/settings/tokens → new token, **write** scope.
4. GitHub repo → Settings → Secrets → Actions → `HF_TOKEN` = that token.

- [ ] **Step 4: Verify**

Push `main`; wait for `ci` then `deploy-hf` in the Actions tab; Space builds (~10 min first time,
torch + model). Open the Space URL: banner, key field, seeded candidates, ask a question with a
real key.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/deploy-hf.yml README.md
git commit -m "ci: deploy main to Hugging Face Space after green CI"
```

---

## Self-review

**Spec coverage:** §1 per-visitor key → Task 2 (+ Task 4 UI). §2 demo mode → Tasks 1, 3, 4.
§3 container → Task 5. §4 pipeline → Task 6. §5 tests → Tasks 1–3 (`test_demo.py`), Task 4
compile test, Tasks 5–6 manual smoke. `.env.example` → Task 1. Every file in the spec's list is
touched by some task.

**Placeholders:** none. **Type consistency:** `get_gateway` returns `Gateway` everywhere;
`process_batch(state, gateway, batch_id, files)` matches its single caller in `upload_resumes`;
`demo_client` fixture defined in Task 1, used in Tasks 2–3; `_headers()` used by all three
client helpers; `st.session_state["demo_mode"]` written in Task 4 Step 2, read in Step 3.

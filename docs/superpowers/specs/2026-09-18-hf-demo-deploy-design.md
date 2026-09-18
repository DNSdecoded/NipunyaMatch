# Public demo deployment on Hugging Face Spaces — design

Date: 2026-09-18. Status: approved.

## Goal

A free, always-linkable public demo of NipunyaMatch where visitors bring their own Gemini API key.
GitHub `main` stays the source of truth; every push redeploys the Space.

## Constraints

- Free tier only. Hugging Face Docker Space: 2 vCPU, 16 GB RAM, 50 GB ephemeral disk, one
  exposed port (7860), sleeps after 48 h idle. No persistent storage.
- The maintainer's Gemini quota must never be spent by visitors.
- Shared SQLite, no accounts. Visitors see each other's jobs. Acceptable for a demo.
- No changes to scoring, parsing, extraction or query semantics.

## Non-goals

Accounts, per-visitor data isolation, persistent storage, rate limiting beyond upload caps,
OpenRouter fallback for visitors, Render/Fly/Streamlit Cloud targets.

## 1. Per-visitor Gemini key

**UI.** Sidebar gains `st.text_input("Your Gemini API key", type="password")` stored only in
`st.session_state["api_key"]`. `app/ui/client.py` sends it as header `X-Gemini-Key` on every
request when present. Sidebar shows a one-line hint with the AI Studio link when empty.

**API.** `app/api/deps.py:get_gateway(request)` returns:

- the app-level gateway (built from `Settings.gemini_api_key`) when no header is present **and**
  `DEMO_MODE` is off (local use unchanged);
- a per-key `Gateway` when the header is present: `request.app.state.gateways: dict[str, Gateway]`
  keyed by `sha256(key)`, created on first use via
  `build_gateway(settings.model_copy(update={"gemini_api_key": key, "openrouter_api_key": None}), session_factory)`.
  Cache, breaker and `llm_calls` audit therefore work per visitor key. Dict is bounded to 100
  entries (drop oldest);
- `ApiError(401, "NO_API_KEY", "Enter your Gemini API key in the sidebar (free at https://aistudio.google.com/apikey).")`
  when `DEMO_MODE` is on and no header is present.

Routes that call the LLM (`POST /api/jobs`, `POST /api/jobs/{id}/resumes`,
`POST /api/jobs/{id}/query`, `POST /api/candidates/{id}/reanalyze`) take the gateway from
`get_gateway`. `process_batch` receives the resolved gateway explicitly instead of reading
`state.gateway`.

**Settings.** `GEMINI_API_KEY` stays required at startup (fail-fast contract unchanged). On the
Space it is set to a placeholder and is never used for visitor traffic.

## 2. Demo mode

New `Settings.demo_mode: bool = False` (`DEMO_MODE` env). When true:

| Surface | Behaviour |
| --- | --- |
| delete job, delete candidate(s), rescore endpoints | `403 DEMO_READ_ONLY` "Deletes and rescoring are disabled on the public demo." |
| `POST /api/jobs/{id}/resumes` | max 10 files per batch (`TOO_MANY_FILES`), 5 MB per file (`FILE_TOO_LARGE`) |
| `GET /api/health` | adds `"demo_mode": true` |
| Data page | tables only; delete buttons and danger zone hidden with a caption |
| Candidates page | "Rescore all" hidden |
| Sidebar | key input shown; banner "Public demo — bring your own Gemini key; data is shared and resets on restart" |

Caps are constants in `app/api/jobs.py` chosen by `settings.demo_mode`.

## 3. Container

- `entrypoint.sh` (repo root, executable): if `data/recruiter.db` is absent run
  `python scripts/seed.py` (no LLM); start `uvicorn app.main:app --host 127.0.0.1 --port 8000` in
  the background; exec `streamlit run app/ui/Home.py --server.port ${PORT:-7860}
  --server.address 0.0.0.0 --server.headless true`. `API_BASE_URL=http://127.0.0.1:8000` exported
  for the UI.
- `Dockerfile`: keep `tesseract-ocr`; add a build step that pre-downloads `all-MiniLM-L6-v2`
  into the image (`HF_HOME=/app/.cache`) so the first request is not a 90 MB fetch;
  `EXPOSE 7860`; `CMD ["./entrypoint.sh"]`. `docker-compose.yml` keeps its two-service layout for
  local use by overriding `command` per service (behaviour unchanged).
- README gains Hugging Face front-matter at the very top (`title`, `sdk: docker`,
  `app_port: 7860`, `pinned: false`). GitHub hides it; HF requires it.

## 4. Deploy pipeline

- Space: `DNSdecoded/NipunyaMatch`, Docker SDK, secrets `GEMINI_API_KEY=demo-unused`,
  `DEMO_MODE=true`.
- `.github/workflows/deploy-hf.yml`: on push to `main`, after the `ci` job passes, push the
  checked-out `main` (full history) to `https://huggingface.co/spaces/DNSdecoded/NipunyaMatch`
  using the `HF_TOKEN` repository secret. Media files are 1.4 MB and 1.6 MB — under HF's 10 MB
  non-LFS limit, so no LFS.
- The maintainer creates the Space and the `HF_TOKEN` (write scope) GitHub secret by hand.

## 5. Testing

Unit (`tests/api/test_demo.py`):

- header present → per-key gateway used; same key twice → same object; different keys → different.
- demo on, no header → 401 `NO_API_KEY` on `POST /api/jobs` and `/query`.
- demo off, no header → app gateway used (existing tests keep passing).
- demo on → delete and rescore return 403; 11 files → 400; 6 MB file → 413.
- health reports `demo_mode`.

Manual: `docker build -t nm .` then run with `-p 7860:7860 -e GEMINI_API_KEY=x -e DEMO_MODE=true`;
open :7860, enter a real key in the sidebar, run Setup → Processing → Candidates → Assistant.

## Files touched

`app/config.py`, `app/api/deps.py`, `app/api/jobs.py`, `app/api/candidates.py`, `app/api/admin.py`,
`app/api/batches.py`, `app/api/health.py`, `app/main.py`, `app/ui/client.py`, `app/ui/Home.py`,
`app/ui/pages/3_Candidates.py`, `app/ui/pages/5_Data.py`, `entrypoint.sh`, `Dockerfile`,
`README.md`, `.github/workflows/deploy-hf.yml`, `.env.example` (`DEMO_MODE=false`), tests.

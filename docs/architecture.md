# Architecture

Diagrams copied from `docs/specs.md`; each followed by the one decision it encodes.

## Layers (§2)

```mermaid
flowchart TD
  UI[Streamlit UI] --> API[FastAPI layer]
  API --> ING[Ingestion<br/>PDF to text]
  ING --> EXT[Extraction<br/>structured JSON]
  EXT --> SCORE[Scoring engine<br/>hybrid rubric]
  SCORE --> DB[(SQLite<br/>+ embeddings)]
  API --> QRY[Query router]
  QRY --> DB
  EXT --> GW[LLM Gateway]
  SCORE --> GW
  QRY --> GW
  GW --> GEM[Gemini API<br/>primary]
  GW --> OR[OpenRouter<br/>fallback]
```

Six layers with a Pydantic type at every boundary. The gateway (`app/llm/gateway.py`) is the only
module that holds provider credentials or knows which provider answered; extraction (JD-independent,
cached by resume hash) and analysis (per JD) are two separate LLM calls so a JD change re-scores
without re-parsing.

## Failover (§4.4)

```mermaid
flowchart TD
  C[complete] --> CACHE{cached?}
  CACHE -->|yes| RET[return]
  CACHE -->|no| BRK{breaker open?}
  BRK -->|yes| ORP[OpenRouter]
  BRK -->|no| GEM[Gemini call]
  GEM -->|valid| RET
  GEM -->|429 or 5xx| RTY{retries left?}
  RTY -->|yes| GEM
  RTY -->|no| ORP
  GEM -->|bad JSON| REP[repair pass]
  REP --> RET
  ORP -->|ok| RET
  ORP -->|fail| ERR[typed error to UI]
```

Every call goes cache → Gemini with 1s/2s/4s retry → breaker (5 failures, 60s open, one probe) →
OpenRouter → one JSON repair pass → typed error. Every attempt is a row in `llm_calls`, so the
audit trail is the evidence that failover works, not a log line.

## Parsing (§5)

```mermaid
flowchart LR
  PDF[PDF bytes] --> V{valid PDF?}
  V -->|no| REJ[reject with reason]
  V -->|yes| MU[PyMuPDF text]
  MU --> Q{chars per page<br/>above 100?}
  Q -->|yes| NORM[normalise]
  Q -->|no| OCR[Tesseract OCR]
  OCR --> NORM
  NORM --> OUT[clean text + metadata]
```

Text spans are clustered into column bands by x-midpoint and read band-by-band, top to bottom,
which is what keeps two-column resumes from interleaving. OCR runs per page and only when a page
yields fewer than 100 characters; `extraction_method` and `char_count` are stored per candidate.

## Data model (§8)

```mermaid
erDiagram
  JOBS ||--o{ ANALYSES : scored_against
  JOBS ||--o{ JOB_SKILLS : requires
  CANDIDATES ||--o{ ANALYSES : has
  CANDIDATES ||--o{ CANDIDATE_SKILLS : has
  CANDIDATES ||--o{ EXPERIENCES : has
  SKILLS ||--o{ CANDIDATE_SKILLS : appears_in
  SKILLS ||--o{ JOB_SKILLS : appears_in
  ANALYSES ||--o{ SKILL_MATCHES : contains
```

Skills are normalised into one `skills` table so "who knows Docker" is a join, not a text scan.
Lists only ever read whole (`strengths`, `weaknesses`, `interview_questions`) are JSON columns;
embeddings are float32 BLOBs compared with numpy cosine (sqlite-vec deferred until >1k candidates).

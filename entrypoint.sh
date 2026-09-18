#!/usr/bin/env sh
# Streamlit binds $PORT immediately (Cloud Run startup probe); seed + API come up behind it.
set -e
export API_BASE_URL="http://127.0.0.1:8000"
PY=/app/.venv/bin/python
(
  if [ ! -f data/recruiter.db ]; then
    echo "seeding demo data (no LLM calls)"
    "$PY" scripts/seed.py
  fi
  exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) &
exec "$PY" -m streamlit run app/ui/Home.py \
  --server.port "${PORT:-8080}" --server.address 0.0.0.0 --server.headless true

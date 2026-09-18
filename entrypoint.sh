#!/usr/bin/env sh
set -e
export API_BASE_URL="http://127.0.0.1:8000"
if [ ! -f data/recruiter.db ]; then
  echo "seeding demo data (no LLM calls)"
  uv run python scripts/seed.py
fi
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
exec uv run streamlit run app/ui/Home.py \
  --server.port "${PORT:-8080}" --server.address 0.0.0.0 --server.headless true

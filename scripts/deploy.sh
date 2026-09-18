#!/usr/bin/env sh
# One-command deploy/redeploy of the public demo. Requires: gcloud auth login; billing on the project.
set -e
PROJECT="${GCP_PROJECT:-nipunyamatch-demo}"
REGION="${GCP_REGION:-us-central1}"
gcloud run deploy nipunyamatch \
  --source . \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 --concurrency 20 \
  --timeout 3600 --session-affinity --no-cpu-throttling \
  --set-env-vars "GEMINI_API_KEY=demo-unused,DEMO_MODE=true,MAX_CONCURRENT_LLM=2,LOG_LEVEL=INFO"
gcloud run services describe nipunyamatch --project "$PROJECT" --region "$REGION" \
  --format "value(status.url)"

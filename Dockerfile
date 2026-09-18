FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV HF_HOME=/app/.cache UV_LINK_MODE=copy UV_NO_SYNC=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Bake the embedding model so the first request is not a 90 MB download.
RUN uv run --no-project python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
# Model is baked above; never call out to huggingface.co at runtime (rate-limited from Cloud Run).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN uv sync --frozen --no-dev && chmod +x entrypoint.sh && mkdir -p data
EXPOSE 8080 8000 8501
CMD ["./entrypoint.sh"]

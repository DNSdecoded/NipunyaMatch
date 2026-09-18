.PHONY: dev test lint seed ui

dev:
	uv run uvicorn app.main:app --reload --port 8000

ui:
	uv run streamlit run app/ui/Home.py

test:
	uv run pytest --cov --cov-fail-under=70

lint:
	uv run ruff check . && uv run mypy app/llm app/scoring

seed:
	uv run python scripts/seed.py

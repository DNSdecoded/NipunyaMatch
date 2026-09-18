import hashlib

from sqlalchemy.orm import Session

from app.db.models import LLMCache


def cache_key(prompt: str, model: str, schema_version: str) -> str:
    return hashlib.sha256((prompt + model + schema_version).encode()).hexdigest()


def get_cached(session: Session, key: str) -> str | None:
    row = session.get(LLMCache, key)
    return row.response_text if row else None


def set_cached(session: Session, key: str, text: str) -> None:
    session.merge(LLMCache(key=key, response_text=text))
    session.commit()

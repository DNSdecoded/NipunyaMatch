import hashlib

from sqlalchemy.orm import Session

from app.llm.cache import cache_key, get_cached, set_cached


def test_key_is_sha256_of_parts() -> None:
    expected = hashlib.sha256(b"pMv1").hexdigest()
    assert cache_key("p", "M", "v1") == expected
    assert cache_key("p", "M", "v2") != expected


def test_roundtrip(db: Session) -> None:
    k = cache_key("prompt", "model", "1")
    assert get_cached(db, k) is None
    set_cached(db, k, '{"a": 1}')
    set_cached(db, k, '{"a": 2}')  # idempotent overwrite
    assert get_cached(db, k) == '{"a": 2}'

from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """all-MiniLM-L6-v2 on CPU; model loaded on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._name = model_name
        self._model: Any = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._name)
        out: np.ndarray = self._model.encode(texts, normalize_embeddings=True)
        return out.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

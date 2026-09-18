import numpy as np


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]], dim: int = 4) -> None:
        self._v = {k.lower(): np.array(v, dtype=np.float32) for k, v in vectors.items()}
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self._v.get(t.lower(), np.zeros(self._dim, dtype=np.float32)) for t in texts]
        return np.vstack(rows)

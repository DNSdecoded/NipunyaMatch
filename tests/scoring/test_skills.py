import math
from pathlib import Path

import numpy as np

from app.scoring.embeddings import cosine, from_blob, to_blob
from app.scoring.skills import SkillMatcher, load_aliases
from tests.scoring.fakes import FakeEmbedder

ALIASES = load_aliases(Path("config/skill_aliases.yaml"))
# ML <-> TensorFlow at cosine 0.81; everything else orthogonal or zero.
VEC = {
    "Machine Learning": [1, 0, 0, 0],
    "TensorFlow": [0.81, math.sqrt(1 - 0.81**2), 0, 0],
    "Docker": [0, 0, 1, 0],
    "Kubernetes": [0, 0, 0, 1],
}


def matcher() -> SkillMatcher:
    return SkillMatcher(ALIASES, FakeEmbedder(VEC))


def test_exact_case_insensitive() -> None:
    r = matcher().match("Python", ["python", "SQL"])
    assert r.matched and r.credit == 1.0 and r.method == "exact"


def test_alias_hit() -> None:
    r = matcher().match("JavaScript", ["JS"])
    assert r.matched and r.credit == 1.0 and r.method == "alias" and r.via == "JS"


def test_embedding_hit_above_threshold_gets_partial_credit() -> None:
    r = matcher().match("ML", ["TensorFlow"])
    assert r.matched and r.credit == 0.8 and r.method == "embedding" and r.via == "TensorFlow"


def test_embedding_below_threshold_misses() -> None:
    r = matcher().match("Docker", ["Kubernetes"])
    assert not r.matched and r.credit == 0.0 and r.method == "none"


def test_coverage_worked_example_required() -> None:
    cov, results = matcher().coverage(
        ["Python", "FastAPI", "Docker", "PostgreSQL", "ML"],
        ["Python", "FastAPI", "TensorFlow", "scikit-learn", "PostgreSQL"],
    )
    assert cov == 76.0  # (1 + 1 + 0 + 1 + 0.8) / 5
    assert [r.matched for r in results] == [True, True, False, True, True]


def test_coverage_empty_targets_is_100() -> None:
    assert matcher().coverage([], ["x"])[0] == 100.0


def test_blob_roundtrip_and_cosine() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert np.allclose(from_blob(to_blob(v)), v)
    assert cosine(v, v) == 1.0
    assert cosine(v, np.zeros(2)) == 0.0

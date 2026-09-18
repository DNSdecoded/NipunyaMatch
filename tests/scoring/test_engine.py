import math
from pathlib import Path

from app.scoring.engine import ScoringEngine, load_config
from app.scoring.jd import JobRequirements
from app.scoring.skills import SkillMatcher, load_aliases
from tests.scoring.fakes import FakeEmbedder

VEC = {
    "Machine Learning": [1, 0, 0, 0],
    "TensorFlow": [0.81, math.sqrt(1 - 0.81**2), 0, 0],
}
JOB = JobRequirements(
    title="Backend", required_skills=["Python", "FastAPI", "Docker", "PostgreSQL", "ML"],
    preferred_skills=["AWS", "Kubernetes"], min_years=3, education_level="bachelor",
)


def engine() -> ScoringEngine:
    cfg = load_config(Path("config/scoring.yaml"))
    matcher = SkillMatcher(
        load_aliases(Path("config/skill_aliases.yaml")), FakeEmbedder(VEC),
        cfg.embedding_threshold, cfg.embedding_credit,
    )
    return ScoringEngine(cfg, matcher)


def test_worked_example_reproduces() -> None:
    b = engine().score(
        ["Python", "FastAPI", "TensorFlow", "scikit-learn", "PostgreSQL"],
        years=4, degree="bachelor", llm_fit=78, job=JOB,
    )
    assert b.required_skill_coverage == 76.0
    assert b.preferred_skill_coverage == 0.0
    assert b.experience_fit == 100.0 and b.education_fit == 100.0
    assert b.final_score == 72 and b.recommendation == "Consider"
    assert b.scoring_version == "1.0"


def test_bands() -> None:
    e = engine()
    perfect = e.score(JOB.required_skills + JOB.preferred_skills, 5, "master", 100, JOB)
    assert perfect.final_score == 100 and perfect.recommendation == "Shortlist"
    weak = e.score([], None, None, 0, JOB)
    assert weak.recommendation == "Reject"


def test_missing_over_half_required_caps_at_consider() -> None:
    b = engine().score(["Python", "FastAPI", "AWS", "Kubernetes"], 10, "phd", 100, JOB)
    # 2 of 5 required present; total would be Shortlist without the cap
    assert b.recommendation == "Consider"


def test_weights_change_moves_score() -> None:
    cfg = load_config(Path("config/scoring.yaml"))
    cfg.weights = {**cfg.weights, "llm_fit_score": 0.0, "required_skill_coverage": 0.55}
    e = ScoringEngine(cfg, SkillMatcher({}, FakeEmbedder({})))
    job = JobRequirements(title="t", required_skills=["Python"])
    assert e.score(["Python"], 4, "bachelor", 0, job).final_score == 100

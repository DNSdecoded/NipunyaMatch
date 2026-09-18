from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from app.scoring.components import education_fit, experience_fit
from app.scoring.embeddings import Embedder
from app.scoring.jd import JobRequirements
from app.scoring.skills import MatchResult, SkillMatcher, load_aliases


class ScoringConfig(BaseModel):
    version: str
    weights: dict[str, float]
    embedding_threshold: float = 0.75
    embedding_credit: float = 0.8
    bands: dict[str, int]


def load_config(path: Path = Path("config/scoring.yaml")) -> ScoringConfig:
    return ScoringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_score: int
    required_skill_coverage: float
    preferred_skill_coverage: float
    experience_fit: float
    education_fit: float
    llm_fit_score: float
    recommendation: str
    scoring_version: str
    required_matches: list[MatchResult]
    preferred_matches: list[MatchResult]


class ScoringEngine:
    def __init__(self, config: ScoringConfig, matcher: SkillMatcher) -> None:
        self.config = config
        self.matcher = matcher

    def score(
        self,
        candidate_skills: list[str],
        years: float | None,
        degree: str | None,
        llm_fit: float,
        job: JobRequirements,
    ) -> ScoreBreakdown:
        req_cov, req_matches = self.matcher.coverage(job.required_skills, candidate_skills)
        pref_cov, pref_matches = self.matcher.coverage(job.preferred_skills, candidate_skills)
        comps = {
            "required_skill_coverage": req_cov,
            "preferred_skill_coverage": pref_cov,
            "experience_fit": experience_fit(years, job.min_years),
            "education_fit": education_fit(degree, job.education_level),
            "llm_fit_score": float(llm_fit),
        }
        w = self.config.weights
        final = round(sum(w[k] * v for k, v in comps.items()))
        bands = self.config.bands
        if final >= bands["shortlist"]:
            rec = "Shortlist"
        elif final >= bands["consider"]:
            rec = "Consider"
        else:
            rec = "Reject"
        missing = sum(1 for m in req_matches if not m.matched)
        if job.required_skills and missing > len(job.required_skills) / 2 and rec == "Shortlist":
            rec = "Consider"
        return ScoreBreakdown(
            final_score=final,
            recommendation=rec,
            scoring_version=self.config.version,
            required_matches=req_matches,
            preferred_matches=pref_matches,
            **comps,
        )


def build_engine(
    embedder: Embedder,
    config_path: Path = Path("config/scoring.yaml"),
    aliases_path: Path = Path("config/skill_aliases.yaml"),
) -> ScoringEngine:
    cfg = load_config(config_path)
    matcher = SkillMatcher(
        load_aliases(aliases_path), embedder, cfg.embedding_threshold, cfg.embedding_credit
    )
    return ScoringEngine(cfg, matcher)

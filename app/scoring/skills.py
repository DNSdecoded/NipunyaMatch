from dataclasses import dataclass
from pathlib import Path

import yaml

from app.scoring.embeddings import Embedder, cosine


def load_aliases(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k).lower(): str(v) for k, v in data.items()}


def canon(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name.strip().lower(), name.strip())


@dataclass
class MatchResult:
    target: str
    matched: bool
    credit: float
    method: str  # exact | alias | embedding | none
    via: str | None = None


class SkillMatcher:
    def __init__(
        self,
        aliases: dict[str, str],
        embedder: Embedder,
        threshold: float = 0.75,
        credit: float = 0.8,
    ) -> None:
        self._aliases = aliases
        self.embedder = embedder
        self._threshold = threshold
        self._credit = credit

    def match(self, target: str, candidate_skills: list[str]) -> MatchResult:
        t = target.strip().lower()
        for s in candidate_skills:
            if s.strip().lower() == t:
                return MatchResult(target, True, 1.0, "exact", s)
        tc = canon(target, self._aliases).lower()
        for s in candidate_skills:
            if canon(s, self._aliases).lower() == tc:
                return MatchResult(target, True, 1.0, "alias", s)
        if candidate_skills:
            names = [canon(target, self._aliases)] + [
                canon(s, self._aliases) for s in candidate_skills
            ]
            vecs = self.embedder.embed(names)
            sims = [(cosine(vecs[0], vecs[i + 1]), s) for i, s in enumerate(candidate_skills)]
            best, via = max(sims, key=lambda x: x[0])
            if best > self._threshold:
                return MatchResult(target, True, self._credit, "embedding", via)
        return MatchResult(target, False, 0.0, "none")

    def coverage(
        self, targets: list[str], candidate_skills: list[str]
    ) -> tuple[float, list[MatchResult]]:
        if not targets:
            return 100.0, []
        results = [self.match(t, candidate_skills) for t in targets]
        return round(100.0 * sum(r.credit for r in results) / len(results), 1), results

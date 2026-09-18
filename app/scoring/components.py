import re

DEGREE_RANK: dict[str, int] = {
    "high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "phd": 5,
}
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("phd", re.compile(r"\b(ph\.?d|doctor(ate)?|d\.?phil)\b", re.I)),
    ("master", re.compile(r"\b(m\.?s\.?c?|m\.?tech|m\.?e\.?|mba|master'?s?|m\.?a\.?)\b", re.I)),
    (
        "bachelor",
        re.compile(r"\b(b\.?s\.?c?|b\.?tech|b\.?e\.?|bachelor'?s?|b\.?a\.?|b\.?eng)\b", re.I),
    ),
    ("associate", re.compile(r"\bassociate\b", re.I)),
    ("high_school", re.compile(r"\b(high school|secondary|diploma)\b", re.I)),
]


def degree_level(degrees: list[str | None]) -> str | None:
    best: str | None = None
    for d in degrees:
        if not d:
            continue
        for level, pat in _PATTERNS:
            if pat.search(d):
                if best is None or DEGREE_RANK[level] > DEGREE_RANK[best]:
                    best = level
                break
    return best


def experience_fit(years: float | None, min_years: float | None) -> float:
    if not min_years:
        return 100.0
    if years is None:
        return 40.0
    if years >= min_years:
        return 100.0
    return round(40.0 + 60.0 * years / min_years, 1)


def education_fit(candidate_level: str | None, required_level: str | None) -> float:
    if required_level is None:
        return 100.0
    if candidate_level is None:
        return 40.0
    gap = DEGREE_RANK[required_level] - DEGREE_RANK[candidate_level]
    if gap <= 0:
        return 100.0
    return 70.0 if gap == 1 else 40.0

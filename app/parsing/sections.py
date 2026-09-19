import re

_SYNONYMS: dict[str, list[str]] = {
    "education": ["education", "academic background", "academics", "qualifications"],
    "experience": ["experience", "work experience", "employment", "work history",
                   "professional experience", "career history"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tech stack"],
    "projects": ["projects", "personal projects", "key projects", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses", "licences"],
}
_LOOKUP = {syn: key for key, syns in _SYNONYMS.items() for syn in syns}
_HEADING = re.compile(r"^\s*([A-Za-z &/]{3,40})\s*:?\s*$")


def _lookup(heading: str) -> str | None:
    """Exact synonym first; then a heading that *starts* with one ("Skills and Interests",
    "Education & Training", "Projects / Publications")."""
    h = heading.strip().lower()
    if h in _LOOKUP:
        return _LOOKUP[h]
    for syn, key in _LOOKUP.items():
        if h.startswith(syn + " ") or h.startswith(syn + "/") or h.startswith(syn + " &"):
            return key
    return None


def tag_sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _HEADING.match(line)
        key = _lookup(m.group(1)) if m else None
        if key:
            current = key
            out.setdefault(current, [])
            continue
        if current:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}

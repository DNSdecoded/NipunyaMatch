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


def tag_sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _HEADING.match(line)
        key = _LOOKUP.get(m.group(1).strip().lower()) if m else None
        if key:
            current = key
            out.setdefault(current, [])
            continue
        if current:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}

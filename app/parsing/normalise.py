import re
from collections import Counter

_CHARS = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " ",
}
_BULLET = re.compile(r"^\s*([•\-\*▪●○◦‣]|\d+[.)])\s+")
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.I)


def _fix_chars(s: str) -> str:
    for k, v in _CHARS.items():
        s = s.replace(k, v)
    return s


def _strip_repeated(pages: list[list[str]]) -> list[list[str]]:
    repeated: set[str] = set()
    if len(pages) >= 2:
        counts = Counter(ln for p in pages for ln in set(p) if ln)
        repeated = {ln for ln, c in counts.items() if c >= 2}
    return [[ln for ln in p if ln not in repeated and not _PAGE_NO.match(ln)] for p in pages]


def _join_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if (
            out
            and ln
            and not _BULLET.match(ln)
            and not _BULLET.match(out[-1])
            and ln[0].islower()
            and re.search(r"[a-z,]$", out[-1])
        ):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return out


def normalise(pages: list[str]) -> str:
    split = [
        [re.sub(r"[ \t]+", " ", ln).strip() for ln in _fix_chars(p).splitlines()] for p in pages
    ]
    split = _strip_repeated(split)
    lines = _join_lines([ln for p in split for ln in p])
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

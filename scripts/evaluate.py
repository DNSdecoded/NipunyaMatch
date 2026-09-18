"""Golden-set evaluation. Requires a real GEMINI_API_KEY. Prints three numbers for the README."""
import asyncio
import json
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.extraction.extractor import extract_candidate
from app.extraction.persist import persist_candidate, resume_hash
from app.extraction.validators import norm
from app.llm.gateway import build_gateway
from app.parsing.pipeline import parse_pdf, parse_text
from app.scoring.analyzer import analyze_candidate
from app.scoring.embeddings import LocalEmbedder
from app.scoring.engine import build_engine
from app.scoring.jd import create_job, parse_jd

FIX = Path("tests/fixtures")


def spearman(a: list[int], b: list[int]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def digits(s: str | None) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


async def main() -> None:
    settings = get_settings()
    db = make_engine("sqlite:///./data/eval.db")
    init_db(db)
    factory = make_session_factory(db)
    gateway = build_gateway(settings, factory)
    engine = build_engine(LocalEmbedder())
    labels = json.loads((FIX / "golden/labels.json").read_text())
    human = json.loads((FIX / "golden/ranking.json").read_text())

    hits = {"name": 0, "email": 0, "phone": 0, "skills": 0}
    claimed = hallucinated = 0
    scores: dict[str, int] = {}
    with factory() as s:
        jd_text = parse_text((FIX / "jd.txt").read_text()).text
        job = create_job(s, await parse_jd(gateway, jd_text), jd_text)
        for fname, lab in labels.items():
            data = (FIX / "resumes" / fname).read_bytes()
            parsed = parse_pdf(data)
            ext, flags = await extract_candidate(gateway, parsed.text)
            hits["name"] += norm(ext.name or "") == norm(lab["name"])
            hits["email"] += (ext.email or "").lower() == lab["email"].lower()
            hits["phone"] += digits(ext.phone) == digits(lab["phone"])
            hits["skills"] += {x.lower() for x in ext.skills} >= {x.lower() for x in lab["skills"]}
            cand = persist_candidate(s, parsed, ext, flags, resume_hash(data))
            a = await analyze_candidate(gateway, s, engine, cand, job)
            claimed += len(a.skill_matches)
            hallucinated += sum(1 for m in a.skill_matches
                                if m.evidence and norm(m.evidence) not in norm(parsed.text))
            scores[fname] = a.final_score

    n = len(labels)
    print(f"Field extraction accuracy (n={n}):")
    for k, v in hits.items():
        print(f"  {k:7s} {v / n:.0%}")
    print(f"Hallucination rate: {hallucinated}/{claimed} = {hallucinated / max(claimed, 1):.1%}")
    system_rank = sorted(scores, key=lambda f: scores[f], reverse=True)
    rho = spearman([human.index(f) for f in labels], [system_rank.index(f) for f in labels])
    print(f"Spearman vs human ranking: {rho:.2f}  (12 resumes; wide error bars)")


if __name__ == "__main__":
    asyncio.run(main())

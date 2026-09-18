from app.extraction.schemas import CandidateExtraction
from app.extraction.validators import validate_extraction
from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask

MAX_CHARS = 12_000


async def extract_candidate(
    gateway: Gateway, resume_text: str
) -> tuple[CandidateExtraction, list[str]]:
    prompt = render("extract", resume=resume_text[:MAX_CHARS])
    raw = await gateway.complete(prompt, CandidateExtraction, LLMTask.EXTRACT)
    return validate_extraction(raw, resume_text)

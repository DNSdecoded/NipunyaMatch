import json

from pydantic import BaseModel

from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import LLMTask
from app.query.handlers import HandlerResult
from app.query.schemas import QueryPlan


class Answer(BaseModel):
    answer: str


async def synthesize(
    gateway: Gateway, question: str, plan: QueryPlan, result: HandlerResult
) -> str:
    prompt = render(
        "synthesize",
        question=question[:1000],
        intent=plan.intent.value,
        facts=json.dumps(result.facts, default=str),
        rows=json.dumps(result.rows, default=str)[:20_000],
    )
    return (await gateway.complete(prompt, Answer, LLMTask.QUERY)).answer

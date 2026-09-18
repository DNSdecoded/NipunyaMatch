from app.llm.gateway import Gateway
from app.llm.prompts import render
from app.llm.types import ExtractionFailed, LLMTask
from app.query.schemas import Intent, QueryPlan


async def classify(gateway: Gateway, question: str) -> QueryPlan:
    try:
        return await gateway.complete(
            render("classify", question=question[:1000]), QueryPlan, LLMTask.QUERY
        )
    except ExtractionFailed:
        return QueryPlan(intent=Intent.SEMANTIC)

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    TOP_N = "top_n"
    FILTER_SKILL = "filter_skill"
    FILTER_ATTRIBUTE = "filter_attribute"
    COMPARE = "compare"
    EXPLAIN_RANKING = "explain_ranking"
    RECOMMEND = "recommend"
    SEMANTIC = "semantic"


Attribute = Literal["total_years_experience", "final_score"]
Op = Literal[">", ">=", "<", "<=", "="]


class QueryPlan(BaseModel):
    intent: Intent
    n: int = Field(default=5, ge=1, le=50)
    skill: str | None = None
    has_skill: bool = True
    attribute: Attribute | None = None
    op: Op | None = None
    value: float | None = None
    candidate_a: str | None = None
    candidate_b: str | None = None


class Source(BaseModel):
    candidate_id: int
    name: str | None
    score: int


class QueryResponse(BaseModel):
    answer: str
    intent: Intent
    sources: list[Source]
    latency_ms: int
    provider_used: str

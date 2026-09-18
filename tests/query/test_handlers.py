import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.models import Analysis, Candidate
from app.query.handlers import HandlerResult, explain_facts, run_handler
from app.query.schemas import Intent, QueryPlan
from tests.query.seed import seed
from tests.scoring.fakes import FakeEmbedder

EMB = FakeEmbedder({
    "who has container experience": [0, 1, 0, 0],
    "Strong API engineer": [0.2, 0.9, 0, 0],
    "Data engineer, no containers": [0.9, 0.1, 0, 0],
    "Junior Java developer": [0, 0, 1, 0],
})


def names(result: HandlerResult) -> list[str]:
    return [r["name"] for r in result.rows]


@pytest.fixture
def job_id(db: Session) -> int:
    return seed(db)


def test_top_n(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.TOP_N, n=2), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_filter_skill_has(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.FILTER_SKILL, skill="python"), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_filter_skill_missing(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.FILTER_SKILL, skill="Docker", has_skill=False)
    assert names(run_handler(db, job_id, plan, EMB)) == ["Bob Jones", "Carol Lee"]


def test_filter_attribute(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.FILTER_ATTRIBUTE, attribute="total_years_experience",
                     op=">", value=2)
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith", "Bob Jones"]


def test_explain_ranking_facts(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.EXPLAIN_RANKING, candidate_a="Alice", candidate_b="Bob")
    r = run_handler(db, job_id, plan, EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]
    f = r.facts
    assert f["a"]["name"] == "Alice Smith" and f["score_gap"] == 21
    assert f["diffs"][0]["component"] == "required_skill_coverage"
    assert f["diffs"][0]["a"] == 100.0 and f["diffs"][0]["b"] == 33.3
    assert f["a_only_skills"] == ["Docker", "FastAPI"]


def test_compare_orders_higher_first(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.COMPARE, candidate_a="Bob", candidate_b="Alice")
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith", "Bob Jones"]


def test_compare_unknown_name_gives_empty(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.COMPARE, candidate_a="Zed", candidate_b="Bob")
    r = run_handler(db, job_id, plan, EMB)
    assert r.rows == [] and "Zed" in r.facts["error"]


def test_recommend(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.RECOMMEND, n=1), EMB)
    assert names(r) == ["Alice Smith"] and r.rows[0]["reasoning"] == "Alice Smith reasoning"


def test_semantic_ranks_by_similarity(db: Session, job_id: int) -> None:
    plan = QueryPlan(intent=Intent.SEMANTIC, n=1, skill="who has container experience")
    assert names(run_handler(db, job_id, plan, EMB)) == ["Alice Smith"]


def test_semantic_without_question_falls_back_to_rank(db: Session, job_id: int) -> None:
    r = run_handler(db, job_id, QueryPlan(intent=Intent.SEMANTIC, n=2), EMB)
    assert names(r) == ["Alice Smith", "Bob Jones"]


def test_read_only_connection(db: Session, job_id: int) -> None:
    run_handler(db, job_id, QueryPlan(intent=Intent.TOP_N), EMB)
    with pytest.raises(OperationalError):
        db.execute(text("DELETE FROM analyses"))


def test_explain_facts_ordering() -> None:
    a = Analysis(candidate=Candidate(name="A"), final_score=80, required_skill_coverage=100,
                 preferred_skill_coverage=0, experience_fit=60, education_fit=100, llm_fit_score=80)
    b = Analysis(candidate=Candidate(name="B"), final_score=70, required_skill_coverage=60,
                 preferred_skill_coverage=50, experience_fit=100, education_fit=100, llm_fit_score=80)
    comps = [d["component"] for d in explain_facts(a, b)["diffs"]]
    assert comps[0] == "required_skill_coverage"  # 0.35 * 40 = 14 weighted gap, largest

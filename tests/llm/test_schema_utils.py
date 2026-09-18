from pydantic import BaseModel

from app.llm.schema_utils import inline_refs


class Inner(BaseModel):
    a: int


class Outer(BaseModel):
    items: list[Inner]
    one: Inner | None


def test_inline_refs_removes_defs() -> None:
    s = inline_refs(Outer.model_json_schema())
    assert "$defs" not in s
    assert "$ref" not in str(s)
    assert s["properties"]["items"]["items"]["properties"]["a"]["type"] == "integer"

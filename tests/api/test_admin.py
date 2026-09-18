from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.query.seed import seed


def test_list_delete_clear(client: TestClient, db: Session) -> None:
    job_id = seed(db)
    rows = client.get("/api/candidates").json()
    assert len(rows) == 3 and all(r["analyses"] == 1 for r in rows)

    cid = next(r["candidate_id"] for r in rows if r["name"] == "Bob Jones")
    assert client.delete(f"/api/candidates/{cid}").json() == {
        "deleted_candidates": 1, "deleted_analyses": 1
    }
    assert client.get(f"/api/candidates/{cid}").status_code == 404
    assert len(client.get(f"/api/jobs/{job_id}/candidates").json()) == 2

    assert client.delete(f"/api/jobs/{job_id}").json()["deleted_analyses"] == 2
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get("/api/candidates").json()[0]["analyses"] == 0

    assert client.delete("/api/candidates").json()["deleted_candidates"] == 2
    assert client.get("/api/candidates").json() == []
    assert client.delete("/api/candidates/999").status_code == 404

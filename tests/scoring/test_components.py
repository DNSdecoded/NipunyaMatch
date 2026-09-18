import pytest

from app.scoring.components import degree_level, education_fit, experience_fit


@pytest.mark.parametrize(
    "degrees,expected",
    [
        (["B.Tech in CS"], "bachelor"),
        (["BSc", "M.S. Computer Science"], "master"),
        (["PhD Physics"], "phd"),
        (["MBA"], "master"),
        (["Associate of Arts"], "associate"),
        (["High School Diploma"], "high_school"),
        ([None, "certificate"], None),
    ],
)
def test_degree_level(degrees: list[str | None], expected: str | None) -> None:
    assert degree_level(degrees) == expected


@pytest.mark.parametrize(
    "years,min_years,expected",
    [(4, 3, 100.0), (3, 3, 100.0), (1.5, 3, 70.0), (0, 3, 40.0), (None, 3, 40.0),
     (None, None, 100.0), (20, 3, 100.0)],
)
def test_experience_fit(years: float | None, min_years: float | None, expected: float) -> None:
    assert experience_fit(years, min_years) == expected


@pytest.mark.parametrize(
    "cand,req,expected",
    [("bachelor", "bachelor", 100.0), ("master", "bachelor", 100.0),
     ("associate", "bachelor", 70.0), ("high_school", "bachelor", 40.0),
     (None, "bachelor", 40.0), (None, None, 100.0), ("bachelor", None, 100.0)],
)
def test_education_fit(cand: str | None, req: str | None, expected: float) -> None:
    assert education_fit(cand, req) == expected

from collections.abc import Iterator
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_gateway(request: Request) -> Any:
    return request.app.state.gateway

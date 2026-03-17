from __future__ import annotations

from fastapi import HTTPException, Request, status


class SessionAuth:
    def __init__(self, session_token: str):
        self._token = session_token

    def verify_request(self, request: Request) -> None:
        incoming = request.headers.get("x-session-token")
        if incoming != self._token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token",
            )

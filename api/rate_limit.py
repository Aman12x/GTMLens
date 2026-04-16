"""
In-process sliding-window rate limiter for Claude API endpoints.

Why not a library (e.g. slowapi): avoids an extra dependency and is
sufficient for cost-protection on a single-process demo deployment.
For multi-process or multi-instance production, replace with a Redis-backed
limiter (e.g. slowapi + Redis storage).

Usage (as a FastAPI dependency):
    from api.rate_limit import claude_rate_limit
    from fastapi import Depends

    @router.post("/generate", dependencies=[Depends(claude_rate_limit)])
    def generate(...): ...
"""

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request, status


class _SlidingWindow:
    """
    Thread-safe sliding-window counter keyed by an arbitrary string (e.g. IP).

    Args:
        limit:          Maximum number of calls allowed in the window.
        window_seconds: Width of the rolling window in seconds.
    """

    def __init__(self, limit: int, window_seconds: int) -> None:
        self._limit = limit
        self._window = float(window_seconds)
        self._store: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """
        Return True and record the call if the key is within its limit,
        False if the limit has been reached.
        """
        now = time.monotonic()
        with self._lock:
            timestamps = self._store[key]
            # Evict expired entries
            self._store[key] = [t for t in timestamps if now - t < self._window]
            if len(self._store[key]) >= self._limit:
                return False
            self._store[key].append(now)
            return True


# 20 Claude API calls per minute per client IP — protects API key budget
_claude_limiter = _SlidingWindow(limit=20, window_seconds=60)


def claude_rate_limit(request: Request) -> None:
    """
    FastAPI dependency that enforces a per-IP rate limit on Claude API routes.

    Raises HTTP 429 when the limit is exceeded.

    Attach with:
        @router.post("/path", dependencies=[Depends(claude_rate_limit)])
    """
    client_ip = request.client.host if request.client else "anonymous"
    if not _claude_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "detail": "Max 20 Claude API requests per minute per IP. Retry after 60s.",
            },
        )

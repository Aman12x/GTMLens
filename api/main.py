"""
GTMLens FastAPI application.

Route namespaces:
    /api/          — demo endpoints (synthetic data, known ground truth)
    /api/alpha/    — real customer endpoints (auth-gated, Phase 2+)

Startup: initialises the auth database table if it does not exist.

Demo reset safety:
    POST /api/demo/reset acquires a threading.Lock before dropping and
    recreating DuckDB tables. Concurrent reset requests receive HTTP 409
    rather than racing into a corrupt half-written state.
    Changed from GET to POST: a GET that destroys and recreates data
    violates HTTP safety semantics and can be triggered by prefetchers.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()  # loads .env from the working directory; no-op if file doesn't exist

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import analyze as analyze_router
from api.routes import auth as auth_router
from api.routes import contacts as contacts_router
from api.routes import data as data_router
from api.routes import experiment as experiment_router
from api.routes import narrative as narrative_router
from api.routes import outreach as outreach_router
from api.routes import segment as segment_router
from core.auth import init_auth_db

logger = logging.getLogger(__name__)

# Lock prevents concurrent demo resets from racing on the DuckDB tables.
_reset_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise and migrate databases on startup."""
    from data.seed_db import migrate_sqlite, seed_sqlite, _DEFAULT_SQLITE, _DEFAULT_DB_URL

    init_auth_db()

    # Auto-seed demo DuckDB on first boot (e.g. fresh Railway deploy).
    db_url = os.getenv("DATABASE_URL", _DEFAULT_DB_URL)
    if not Path(db_url).exists():
        logger.info("Demo DB not found — seeding synthetic data (first boot)…")
        from data.seed_db import main as seed_main
        seed_main()

    # Ensure all SQLite tables exist (idempotent — CREATE TABLE IF NOT EXISTS).
    # Run before migrate_sqlite so the migration can assume tables are present.
    sqlite_path = os.getenv("SQLITE_PATH", _DEFAULT_SQLITE)
    seed_sqlite(sqlite_path)
    migrate_sqlite(sqlite_path)

    logger.info("GTMLens API started.")
    yield


app = FastAPI(
    title="GTMLens",
    description="Causal targeting engine for GTM funnels.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow localhost dev origins; tighten for production via env var
# ---------------------------------------------------------------------------

_allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health check (required by Railway deploy checklist)
# ---------------------------------------------------------------------------


@app.get("/health", tags=["infra"])
def health() -> dict:
    """Returns 200 when the API is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Demo reset
#
# POST (not GET): this operation drops and recreates database tables.
# GET must be safe and idempotent per RFC 9110 — data destruction is neither.
# Using GET risks triggering the reset via browser prefetch, link crawler,
# or reverse-proxy health checks.
#
# Lock: prevents a concurrent /analyze request from seeing a half-rebuilt
# DuckDB schema while tables are being dropped and recreated.
# ---------------------------------------------------------------------------


@app.post("/api/demo/reset", tags=["demo"])
def demo_reset() -> dict:
    """
    Reload the synthetic dataset with known ground truth.

    This endpoint is intentionally unauthenticated so the live demo
    can be reset without credentials.

    Returns HTTP 409 if a reset is already in progress.
    """
    if not _reset_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "Reset in progress", "detail": "A reset is already running. Retry in a moment."},
        )
    try:
        from data.seed_db import main as seed_main
        seed_main()
        return {"status": "reset", "message": "Demo data reloaded from synthetic generator."}
    finally:
        _reset_lock.release()


# ---------------------------------------------------------------------------
# Demo / analysis routes (unauthenticated — synthetic data)
# ---------------------------------------------------------------------------

app.include_router(analyze_router.router,    prefix="/api")
app.include_router(experiment_router.router, prefix="/api")
app.include_router(segment_router.router,    prefix="/api")
app.include_router(outreach_router.router,   prefix="/api")
app.include_router(narrative_router.router,  prefix="/api")
app.include_router(contacts_router.router,   prefix="/api")
app.include_router(data_router.router,       prefix="/api")

# ---------------------------------------------------------------------------
# Alpha router (auth-gated real-customer routes)
# ---------------------------------------------------------------------------

app.include_router(auth_router.router, prefix="/api/alpha")

# ---------------------------------------------------------------------------
# Serve built React app as SPA — MUST come last so all /api/* routes above
# take priority over the catch-all static file handler.
# In development the Vite dev server (port 5173) proxies /api to FastAPI,
# so this block is a no-op (dist/ won't exist). On Railway the build step
# produces ui/dist/ and FastAPI serves it from the same process/port.
# ---------------------------------------------------------------------------

_ui_dist = Path(__file__).parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="frontend")

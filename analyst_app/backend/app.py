import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .contracts import (
    AIResult,
    DashboardResult,
    AnalyzeRequest,
    EventInput,
    EventList,
    EventRecord,
    FeedbackInput,
    SearchRequest,
    SimilarResult,
    ThemeResult,
    TrendResult,
)
from .services import Services
from .storage import Database
from .dashboard import event_rows, matches_filter, overview


def create_app(settings=None, encoder=None):
    settings = settings or Settings.from_env()
    # No network model lookup, telemetry or cloud inference, even during startup.
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )

    @asynccontextmanager
    async def lifespan(app):
        app.state.db = Database(settings.database)
        app.state.services = Services(settings, app.state.db, encoder)
        yield

    app = FastAPI(title="MIDAS local analyst MVP", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in (
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ):
            return JSONResponse({"detail": "Cross-origin access disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default response echoes invalid input values, possibly PHI.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    dict(loc=e["loc"], type=e["type"], msg="Invalid or missing field")
                    for e in exc.errors()
                ]
            },
        )

    def event(key):
        try:
            UUID(key)
        except ValueError:
            raise HTTPException(404, "Event not found")
        value = app.state.db.get("events", key)
        if value is None:
            raise HTTPException(404, "Event not found")
        return value

    def theme(key):
        value = app.state.db.get("themes", key)
        if value is None:
            raise HTTPException(404, "Pattern not found")
        return {k: v for k, v in value.items() if k != "centroid"}

    @app.get("/api/health")
    def health():
        s = app.state.services
        return dict(
            status="local-development",
            fixture_mode=settings.fixture_mode,
            models={
                name: getattr(s, name).version for name in ("severity", "event_type", "pattern")
            },
            errors=s.startup_errors,
            rebuild=s.rebuild_status,
        )

    @app.post("/api/events", status_code=201, response_model=EventRecord)
    def create_event(data: EventInput, background: BackgroundTasks):
        try:
            record = app.state.db.create_event(data)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Source event ID already exists")
        app.state.services.pending(record["id"])
        app.state.services.queue.enqueue(background, app.state.services.analyze, record["id"])
        return dict(**record, analysis_status="pending")

    @app.get("/api/events", response_model=EventList)
    def events(
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
        filter: Literal["all", "unreviewed", "priority", "reviewed", "analysis_issues"] = "all",
    ):
        records = [
            r
            for r in event_rows(app.state.db.snapshot("events", "ai_results", "analyst_feedback"))
            if matches_filter(r, filter)
        ]
        return dict(
            total=len(records),
            items=records[offset : offset + limit],
        )

    @app.get("/api/events/{key}", response_model=EventRecord)
    def get_event(key: str):
        decisions = [r for r in app.state.db.all("analyst_feedback") if r["event_id"] == key]
        return dict(
            **event(key),
            analyst_decisions=decisions,
            review_status="reviewed" if decisions else "unreviewed",
        )

    @app.post("/api/ai/analyze", status_code=202)
    def analyze(data: AnalyzeRequest, background: BackgroundTasks):
        for key in data.event_ids:
            event(key)
        for key in set(data.event_ids):
            app.state.services.pending(key)
            app.state.services.queue.enqueue(background, app.state.services.analyze, key)
        return {"status": "queued", "event_ids": data.event_ids}

    @app.get("/api/events/{key}/ai", response_model=AIResult)
    def get_ai(key: str):
        event(key)
        with app.state.services.lock:
            return app.state.services.result(key)

    @app.get("/api/events/{key}/similar", response_model=List[SimilarResult])
    def similar(key: str, top_n: int = Query(5, ge=1, le=10000)):
        record = event(key)
        with app.state.services.lock:
            pattern = app.state.services.pattern
            if not hasattr(pattern, "analyze"):
                raise HTTPException(503, "Local embedding model unavailable")
            stored = app.state.db.get("event_embeddings", key)
            if not stored or stored["model_version"] != pattern.version:
                raise HTTPException(409, "Report embedding pending; queue analysis first")
            vector = stored["vector"]
            pattern.refresh()
            return pattern.store.search(vector, top_n, before=record["event_date"], exclude=key)

    @app.post("/api/search", response_model=List[SimilarResult])
    def search(data: SearchRequest):
        with app.state.services.lock:
            pattern = app.state.services.pattern
            if not hasattr(pattern, "search"):
                raise HTTPException(503, "Local embedding model unavailable")
            return pattern.search(data.query, data.top_n)

    @app.post("/api/events/{key}/feedback", status_code=201)
    def feedback(key: str, data: FeedbackInput):
        event(key)
        return app.state.db.feedback(key, data)

    @app.get("/api/patterns", response_model=List[ThemeResult])
    def patterns():
        return [
            dict(
                **theme(t["theme_id"]),
                trend=app.state.db.get("trend_results", t["theme_id"]),
            )
            for t in app.state.db.all("themes")
        ]

    @app.post("/api/patterns/rebuild", status_code=202)
    def rebuild(background: BackgroundTasks):
        s = app.state.services
        if s.rebuild_status["status"] in ("queued", "running"):
            raise HTTPException(409, "Rebuild already queued")
        s.rebuild_status = {"status": "queued"}
        s.queue.enqueue(background, s.rebuild)
        return {"status": "queued"}

    @app.get("/api/patterns/{key}", response_model=ThemeResult)
    def get_pattern(key: str):
        return dict(**theme(key), trend=app.state.db.get("trend_results", key))

    @app.get("/api/patterns/{key}/events", response_model=EventList)
    def pattern_events(
        key: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)
    ):
        theme(key)
        ids = {m["event_id"] for m in app.state.db.all("event_themes") if m["theme_id"] == key}
        records = [
            r
            for r in event_rows(app.state.db.snapshot("events", "ai_results", "analyst_feedback"))
            if r["id"] in ids
        ]
        return {
            "total": len(records),
            "items": records[offset : offset + limit],
        }

    @app.get("/api/trends", response_model=List[TrendResult])
    def trends():
        return app.state.db.all("trend_results")

    @app.get("/api/trends/{key}", response_model=TrendResult)
    def trend(key: str):
        value = app.state.db.get("trend_results", key)
        if value is None:
            raise HTTPException(404, "Trend not found")
        return value

    @app.get("/api/dashboard", response_model=DashboardResult)
    def dashboard():
        return dict(
            fixture_mode=settings.fixture_mode,
            **overview(
                app.state.db.snapshot(
                    "events", "ai_results", "analyst_feedback", "themes", "trend_results"
                )
            ),
        )

    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def client(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            return FileResponse(dist / "index.html")

    return app

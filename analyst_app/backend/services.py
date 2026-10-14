import threading
from typing import Protocol

from .contracts import AIResult, EventInput, now
from .inference import (
    EventTypeService,
    FixtureEventType,
    FixtureSeverity,
    SeverityService,
    UnavailableModel,
)
from .patterns import PatternService, weekly_trend


class JobQueue(Protocol):
    def enqueue(self, tasks, function, *args) -> None: ...


class BackgroundQueue:
    def enqueue(self, tasks, function, *args):
        tasks.add_task(self.run, function, *args)

    @staticmethod
    def run(function, *args):
        # Background exceptions must never bubble into event submission or log PHI.
        # Services persist expected per-model failures themselves. For unexpected
        # worker failures, retain the stored event and expose a sanitized failed job.
        try:
            function(*args)
        except Exception:
            owner = getattr(function, "__self__", None)
            if owner is not None and args:
                try:
                    key = args[0]
                    owner.db.put(
                        "ai_results",
                        key,
                        AIResult(
                            event_id=key,
                            status="failed",
                            errors={
                                "worker": "Background processing failed; queue analysis to retry"
                            },
                        ).model_dump(mode="json"),
                    )
                    owner.db.audit("ai.worker_failed", key)
                except Exception:
                    pass  # storage failure cannot be recovered in an in-process queue


class Services:
    def __init__(self, settings, db, encoder=None):
        self.settings, self.db = settings, db
        self.lock = threading.RLock()
        self.queue = BackgroundQueue()
        self.startup_errors = {}
        if settings.fixture_mode:
            self.severity, self.event_type = FixtureSeverity(), FixtureEventType()
        else:
            self.severity = self._load(
                "severity",
                lambda: SeverityService.load_model(
                    settings.severity_repo, settings.severity_artifacts
                ),
            )
            self.event_type = self._load(
                "event_type",
                lambda: EventTypeService.load_model(
                    settings.event_type_repo, settings.event_type_artifacts
                ),
            )
        self.pattern = self._load("pattern", lambda: PatternService(settings, db, encoder))
        self.rebuild_status = {"status": "idle"}

    def _load(self, name, function):
        try:
            return function()
        except Exception as exc:
            # No exception messages/stack traces: third-party errors can echo input.
            self.startup_errors[name] = (
                type(exc).__name__ + ": check local model configuration/artifacts"
            )
            return UnavailableModel()

    def pending(self, key):
        result = AIResult(
            event_id=key,
            model_versions={
                name: getattr(self, name).version for name in ("severity", "event_type", "pattern")
            },
        ).model_dump(mode="json")
        self.db.put("ai_results", key, result)

    def analyze(self, key):
        with self.lock:
            record = self.db.get("events", key)
            if not record:
                return
            event = EventInput(**{k: record[k] for k in EventInput.model_fields})
            result = AIResult(event_id=key)
            for name in ("severity", "event_type"):
                model = getattr(self, name)
                result.model_versions[name] = model.version
                try:
                    setattr(result, name, model.predict_event(event))
                except Exception as exc:
                    result.errors[name] = type(exc).__name__ + ": analysis unavailable"
            result.model_versions["pattern"] = self.pattern.version
            try:
                result.similar_events, result.themes = self.pattern.analyze(record)
            except Exception as exc:
                result.errors["pattern"] = type(exc).__name__ + ": analysis unavailable"
            result.status = (
                "complete"
                if not result.errors
                else ("failed" if len(result.errors) == 3 else "partial")
            )
            self.db.put("ai_results", key, result.model_dump(mode="json"))
            self.refresh_summaries()
            self.db.audit("ai." + result.status, key)

    def refresh_summaries(self):
        """Incremental membership changes update counts without reclustering."""
        from collections import Counter
        from datetime import date

        events = {r["id"]: r for r in self.db.all("events")}
        ai = {r["event_id"]: r for r in self.db.all("ai_results")}
        membership = self.db.all("event_themes")
        for theme in self.db.all("themes"):
            members = [
                events[m["event_id"]] for m in membership if m["theme_id"] == theme["theme_id"]
            ]
            dates = [date.fromisoformat(r["event_date"]) for r in members]
            theme.update(
                event_count=len(members),
                first_seen=min(dates).isoformat() if dates else None,
                last_seen=max(dates).isoformat() if dates else None,
                location_distribution=dict(
                    Counter(r["location_name"] or "Unspecified" for r in members)
                ),
                facility_distribution=dict(
                    Counter(r["facility"] or "Unspecified" for r in members)
                ),
                event_type_distribution=dict(
                    Counter(
                        (ai.get(r["id"], {}).get("event_type") or {}).get(
                            "event_type_code", "unavailable"
                        )
                        for r in members
                    )
                ),
                severity_distribution=dict(
                    Counter(
                        (ai.get(r["id"], {}).get("severity") or {}).get("grade", "unavailable")
                        for r in members
                    )
                ),
            )
            self.db.put("themes", theme["theme_id"], theme)
            self.db.put(
                "trend_results",
                theme["theme_id"],
                weekly_trend(theme["theme_id"], dates, self.settings),
            )

    def result(self, key):
        result = self.db.get("ai_results", key)
        if not result:
            return AIResult(event_id=key)
        # Memberships are current even if the classifier result predates a rebuild.
        themes = [m for m in self.db.all("event_themes") if m["event_id"] == key]
        result["themes"] = themes
        result["trend_context"] = [self.db.get("trend_results", m["theme_id"]) for m in themes]
        return AIResult(**result)

    def rebuild(self):
        with self.lock:
            self.rebuild_status = {"status": "running", "created_at": now()}
            try:
                themes = self.pattern.rebuild()
                self.rebuild_status = {
                    "status": "complete",
                    "theme_count": len(themes),
                    "created_at": now(),
                }
            except Exception as exc:
                self.rebuild_status = {
                    "status": "failed",
                    "error": type(exc).__name__ + ": pattern rebuild unavailable",
                    "created_at": now(),
                }

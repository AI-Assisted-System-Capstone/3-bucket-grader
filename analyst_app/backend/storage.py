"""Local JSON-domain records in relational tables; no report content is logged."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .contracts import now

TABLES = (
    "events",
    "ai_results",
    "event_embeddings",
    "similar_events",
    "themes",
    "event_themes",
    "trend_results",
    "analyst_feedback",
    "audit_log",
)


class AuditSink(Protocol):
    def audit(self, action: str, subject_id: str) -> None: ...


class Database:
    def __init__(self, path):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, source_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ai_results (
                    id TEXT PRIMARY KEY REFERENCES events(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS event_embeddings (
                    id TEXT PRIMARY KEY REFERENCES events(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS similar_events (
                    id TEXT PRIMARY KEY REFERENCES events(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS themes (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS event_themes (
                    id TEXT PRIMARY KEY, event_id TEXT REFERENCES events(id),
                    theme_id TEXT REFERENCES themes(id), payload TEXT NOT NULL,
                    UNIQUE(event_id, theme_id));
                CREATE INDEX IF NOT EXISTS memberships_event ON event_themes(event_id);
                CREATE TABLE IF NOT EXISTS trend_results (
                    id TEXT PRIMARY KEY REFERENCES themes(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analyst_feedback (
                    id TEXT PRIMARY KEY, event_id TEXT REFERENCES events(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit_log (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
        Path(path).chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, table, key):
        assert table in TABLES
        with self.connect() as db:
            row = db.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, table):
        assert table in TABLES
        with self.connect() as db:
            rows = db.execute(f"SELECT payload FROM {table} ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def snapshot(self, *tables):
        """Read related collections from one SQLite snapshot, even during a rebuild."""
        assert all(table in TABLES for table in tables)
        with self.connect() as db:
            db.execute("BEGIN")
            return {
                table: [
                    json.loads(row[0])
                    for row in db.execute(f"SELECT payload FROM {table} ORDER BY rowid").fetchall()
                ]
                for table in tables
            }

    def put(self, table, key, payload):
        assert table in TABLES and table not in (
            "events",
            "event_themes",
            "analyst_feedback",
        )
        with self.connect() as db:
            db.execute(
                f"INSERT INTO {table}(id,payload) VALUES(?,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (key, json.dumps(payload, allow_nan=False)),
            )

    def create_event(self, event):
        key = str(uuid4())
        record = dict(id=key, **event.model_dump(mode="json"), created_at=now())
        with self.connect() as db:
            db.execute(
                "INSERT INTO events VALUES(?,?,?)",
                (key, event.event_id, json.dumps(record, allow_nan=False)),
            )
            self._audit(db, "event.created", key)
        return record

    def feedback(self, event_id, data):
        record = dict(
            id=str(uuid4()),
            event_id=event_id,
            **data.model_dump(),
            created_at=now(),
            actor="local-analyst",
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO analyst_feedback VALUES(?,?,?)",
                (record["id"], event_id, json.dumps(record)),
            )
            self._audit(db, "analyst.decision_recorded", event_id)
        return record

    def replace_patterns(self, themes, memberships, trends):
        """Atomic snapshot replacement, so readers never see partially rebuilt themes."""
        with self.connect() as db:
            for table in ("trend_results", "event_themes", "themes"):
                db.execute(f"DELETE FROM {table}")
            for theme in themes:
                db.execute(
                    "INSERT INTO themes VALUES(?,?)",
                    (theme["theme_id"], json.dumps(theme, allow_nan=False)),
                )
            for membership in memberships:
                db.execute(
                    "INSERT INTO event_themes VALUES(?,?,?,?)",
                    (
                        membership["event_id"] + ":" + membership["theme_id"],
                        membership["event_id"],
                        membership["theme_id"],
                        json.dumps(membership, allow_nan=False),
                    ),
                )
            for trend in trends:
                db.execute(
                    "INSERT INTO trend_results VALUES(?,?)",
                    (trend["theme_id"], json.dumps(trend, allow_nan=False)),
                )
            self._audit(db, "patterns.rebuilt", "patterns")

    def set_memberships(self, event_id, memberships):
        with self.connect() as db:
            db.execute("DELETE FROM event_themes WHERE event_id=?", (event_id,))
            for membership in memberships:
                record = dict(event_id=event_id, **membership)
                db.execute(
                    "INSERT INTO event_themes VALUES(?,?,?,?)",
                    (
                        event_id + ":" + record["theme_id"],
                        event_id,
                        record["theme_id"],
                        json.dumps(record),
                    ),
                )

    @staticmethod
    def _audit(db, action, subject_id):
        # Opaque internal IDs only; never source IDs, narratives, errors or notes.
        record = dict(
            action=action,
            subject_id=subject_id,
            actor="local-session",
            created_at=now(),
        )
        db.execute("INSERT INTO audit_log VALUES(?,?)", (str(uuid4()), json.dumps(record)))

    def audit(self, action, subject_id):
        with self.connect() as db:
            self._audit(db, action, subject_id)

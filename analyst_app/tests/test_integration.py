import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import create_app
from backend.config import Settings
from backend.contracts import (
    CODES,
    GRADES,
    AIResult,
    EventInput,
    model_frame,
    normalized_text,
)
from backend.inference import (
    EventTypeService,
    FixtureEventType,
    FixtureSeverity,
    SeverityService,
    import_source,
    probabilities,
)
from backend.patterns import (
    CosineStore,
    PatternService,
    memberships_for,
    weekly_trend,
)
from backend.seed import synthetic_events
from backend.storage import TABLES, Database


def event(identifier="SYN-1", **changes):
    return EventInput(
        event_id=identifier,
        event_comments="Synthetic barcode scanner failed before medication dose.",
        event_date="2026-09-20",
        **changes,
    )


class SyntheticVectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 1))


class SyntheticGradeModel:
    classes_ = np.array([0, 5, 9])

    def predict_proba(self, x):
        return np.tile([0.6, 0.3, 0.1], (len(x), 1))


class SyntheticCalibrator:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        return np.tile([0.95, 0.05], (len(x), 1))


class ContractTests(unittest.TestCase):
    def test_mapping_exact_intake_only(self):
        e = event(
            location_name="Unit A",
            encounter_service="Medicine",
            age_at_encounter=55,
            prescribed_medication="SyntheticDrug",
            prescribed_dose="5 units",
        )
        row = model_frame([e]).iloc[0]
        self.assertEqual(row["Location Name"], "Unit A")
        self.assertEqual(row["ME - Prescribed - Name *... Name"], "SyntheticDrug")
        self.assertEqual(len(row), 10)
        self.assertEqual(
            normalized_text(e),
            "Unit: Unit A. Service: Medicine. Age: 55. Prescribed: SyntheticDrug 5 units.\n"
            + e.event_comments,
        )
        for field in [
            "manager_comments",
            "Event Type",
            "Significance (PSRS Harm score)",
            "grade",
        ]:
            with self.assertRaises(ValidationError):
                EventInput(**e.model_dump(), **{field: "SECRET TARGET"})

    def test_fixture_probabilities_and_labels(self):
        s = FixtureSeverity().predict_event(event())
        t = FixtureEventType().predict_event(event())
        self.assertEqual(set(s.grade_probabilities), set(GRADES))
        self.assertEqual(set(t.probabilities), set(CODES))
        self.assertAlmostEqual(sum(s.grade_probabilities.values()), 1)
        self.assertAlmostEqual(sum(s.bucket_probabilities.values()), 1)
        self.assertAlmostEqual(sum(t.probabilities.values()), 1)
        self.assertTrue(s.fixture and t.fixture)
        self.assertEqual(s.model_dump(), FixtureSeverity().predict_event(event()).model_dump())

    def test_invalid_probability_contract(self):
        for p in [[[np.nan, 1]], [[-0.1, 1.1]], [[0.1, 0.1]]]:
            with self.assertRaises(ValueError):
                probabilities(p, 2)

    def test_missing_artifacts_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            SeverityService.load_model("", "")
        with self.assertRaises(FileNotFoundError):
            EventTypeService.load_model("", "")

    def test_severity_mapping_and_saved_raw_cutoff(self):
        class Vectorizer:
            def transform(self, text):
                return text

        class Model:
            classes_ = np.array([9, 5, 0])  # intentionally noncanonical ordering/subset

            def predict_proba(self, x):
                return np.array([[0.1, 0.3, 0.6]] * len(x))

        class Calibrator:
            classes_ = np.array([1, 0])

            def predict_proba(self, x):
                return np.array([[0.05, 0.95]] * len(x))

        class Upstream:
            @staticmethod
            def bucket_probs(p, classes):
                return p[:, [2, 1, 0]]

            @staticmethod
            def triage_score(p, classes):
                return p[:, :2].sum(axis=1)

        service = SeverityService()
        service.vectorizer, service.model, service.calibrator = (
            Vectorizer(),
            Model(),
            Calibrator(),
        )
        service.upstream, service.builder, service.cutoff, service.version = (
            Upstream(),
            lambda x: x,
            0.4,
            "test-only",
        )
        value = service.predict_event(event())
        self.assertEqual(value.grade, "A")
        self.assertAlmostEqual(value.grade_probabilities["I"], 0.1)
        self.assertTrue(value.priority_flag)  # raw .4 >= .4, despite calibrated .05
        self.assertEqual(value.harm_probability_calibrated, 0.05)
        service.cutoff = 0.40001
        self.assertFalse(service.predict_event(event()).priority_flag)


class PatternTests(unittest.TestCase):
    def test_similarity_sorted_self_and_future_excluded(self):
        store = CosineStore()
        records = [
            dict(id=str(i), event_date=d, location_name="Synthetic")
            for i, d in enumerate(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
        ]
        store.replace(records, [[1, 0], [0.8, 0.2], [0, 1], [1, 0]])
        hits = store.search(np.array([1, 0]), 99, before="2026-01-04", exclude="0")
        self.assertEqual([h["event_id"] for h in hits], ["1", "2"])
        self.assertGreater(hits[0]["similarity"], hits[1]["similarity"])
        self.assertEqual(len(store.search(np.array([1, 0]), 1)), 1)
        with self.assertRaises(ValueError):
            store.search(np.array([1, 0]), 0)

    def test_multilabel_assignment(self):
        themes = [
            dict(theme_id="one", name="Scanner", centroid=[1, 0]),
            dict(theme_id="two", name="Handoff", centroid=[0, 1]),
        ]
        vector = np.array([1, 1]) / np.sqrt(2)
        self.assertEqual(len(memberships_for(vector, themes, 0.7)), 2)
        self.assertEqual(len(memberships_for(vector, themes, 0.8)), 0)

    def test_trend_thresholds_and_complete_weeks(self):
        settings = Settings(
            baseline_weeks=4,
            minimum_event_count=3,
            minimum_percent_increase=50,
            z_score_threshold=2,
        )
        as_of = date(2026, 9, 30)
        week = date(2026, 9, 21)
        baseline = [week - timedelta(weeks=i) for i in range(1, 5)]
        trend = weekly_trend("x", baseline + [week] * 6 + [as_of] * 100, settings, as_of)
        self.assertEqual(trend["current_count"], 6)  # excludes current partial week
        self.assertEqual(trend["baseline_count"], 1)
        self.assertEqual(trend["percent_change"], 500)
        self.assertEqual(trend["status"], "emerging")
        self.assertEqual(weekly_trend("x", [week], settings, as_of)["status"], "stable")
        zero = weekly_trend("x", [week] * 3, settings, as_of)
        self.assertIsNone(zero["percent_change"])
        self.assertTrue(zero["new_from_zero"])
        json.dumps(zero, allow_nan=False)

    def test_cluster_creation_persistence_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(
                database=str(Path(folder) / "db"),
                embedding_backend="lexical",
                cluster_method="agglomerative",
                membership_threshold=0.75,
                cluster_distance=0.2,
            )
            db = Database(settings.database)
            for e in synthetic_events():
                db.create_event(e)
            service = PatternService(settings, db)
            themes = service.rebuild()
            self.assertGreaterEqual(len(themes), 3)
            self.assertEqual(len(db.all("event_embeddings")), len(db.all("events")))
            self.assertEqual(len(db.all("trend_results")), len(themes))
            for t in themes:
                self.assertGreaterEqual(t["event_count"], 3)
                self.assertTrue(t["keywords"])
                self.assertTrue(t["first_seen"])
            restarted = PatternService(settings, db)
            self.assertEqual(
                restarted.search("barcode scanner medication dose", 5),
                service.search("barcode scanner medication dose", 5),
            )
            self.assertEqual(
                {t["theme_id"] for t in restarted.rebuild()},
                {t["theme_id"] for t in themes},
            )


class APITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.settings = Settings(
            database=str(Path(self.folder.name) / "db"),
            fixture_mode=True,
            embedding_backend="lexical",
            cluster_method="agglomerative",
        )
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.folder.cleanup()

    def post(self, identifier="SYN-1"):
        response = self.client.post("/api/events", json=event(identifier).model_dump(mode="json"))
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_all_required_routes_and_contract(self):
        key = self.post()
        self.assertNotEqual(key, "SYN-1")
        result = self.client.get("/api/events/" + key + "/ai")
        ai = AIResult(**result.json())
        self.assertEqual(ai.status, "complete")
        self.assertTrue(ai.severity.fixture)
        self.assertIn("created_at", result.json())
        self.assertEqual(self.client.get("/api/events").json()["total"], 1)
        self.assertEqual(self.client.get("/api/events/" + key).status_code, 200)
        self.assertEqual(self.client.get("/api/events/" + key + "/similar?top_n=10000").json(), [])
        self.assertEqual(
            self.client.get("/api/events/" + key + "/similar?top_n=0").status_code, 422
        )
        self.assertEqual(self.client.get("/api/events/missing").status_code, 404)
        self.assertEqual(
            self.client.post("/api/ai/analyze", json={"event_ids": [key]}).status_code,
            202,
        )
        self.assertEqual(self.client.post("/api/patterns/rebuild").status_code, 202)
        self.assertEqual(self.client.get("/api/patterns").json(), [])
        self.assertEqual(self.client.get("/api/trends").json(), [])
        self.assertEqual(self.client.get("/api/trends/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/patterns/missing/events").status_code, 404)
        self.assertEqual(
            self.client.post(
                "/api/search", json={"query": "barcode scanner", "top_n": 20}
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post("/api/events", json=event().model_dump(mode="json")).status_code,
            409,
        )
        schema = self.client.get("/openapi.json").json()
        self.assertIn("EventInput", schema["components"]["schemas"])

    def test_all_ai_failures_still_persist_event(self):
        services = self.app.state.services
        with (
            patch.object(
                services.severity,
                "predict_event",
                side_effect=RuntimeError("PRIVATE CONTENT"),
            ),
            patch.object(
                services.event_type,
                "predict_event",
                side_effect=RuntimeError("PRIVATE CONTENT"),
            ),
            patch.object(services.pattern, "analyze", side_effect=RuntimeError("PRIVATE CONTENT")),
        ):
            key = self.post()
        result = self.client.get("/api/events/" + key + "/ai").json()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["errors"]), 3)
        self.assertNotIn("PRIVATE CONTENT", json.dumps(result))
        self.assertIsNotNone(self.app.state.db.get("events", key))

    def test_unexpected_worker_failure_does_not_break_submission(self):
        with patch.object(
            self.app.state.services,
            "refresh_summaries",
            side_effect=RuntimeError("SECRET"),
        ):
            key = self.post("SYN-WORKER-FAIL")
        self.assertEqual(self.client.get("/api/events/" + key).status_code, 200)
        result = self.client.get("/api/events/" + key + "/ai").json()
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("SECRET", json.dumps(result))

    def test_pattern_routes_and_incremental_memberships(self):
        keys = []
        for i in range(3):
            keys.append(self.post("SYN-CLUSTER-" + str(i)))
        self.client.post("/api/patterns/rebuild")
        patterns = self.client.get("/api/patterns").json()
        self.assertEqual(len(patterns), 1)
        key = patterns[0]["theme_id"]
        self.assertEqual(self.client.get("/api/patterns/" + key).status_code, 200)
        self.assertEqual(self.client.get("/api/patterns/" + key + "/events").json()["total"], 3)
        self.assertEqual(self.client.get("/api/trends/" + key).status_code, 200)
        new = self.post("SYN-NEW-MEMBER")
        self.assertEqual(self.client.get("/api/patterns/" + key).json()["event_count"], 4)
        self.assertEqual(
            self.client.get("/api/events/" + new + "/ai").json()["themes"][0]["theme_id"],
            key,
        )

    def test_missing_models_are_not_silently_replaced(self):
        settings = Settings(
            database=str(Path(self.folder.name) / "missing.db"),
            embedding_backend="lexical",
            severity_repo="",
            severity_artifacts="",
        )
        with TestClient(create_app(settings)) as client:
            r = client.post("/api/events", json=event().model_dump(mode="json"))
            self.assertEqual(r.status_code, 201)
            ai = client.get("/api/events/" + r.json()["id"] + "/ai").json()
            self.assertIsNone(ai["severity"])
            self.assertIsNone(ai["event_type"])
            self.assertEqual(ai["status"], "partial")
            self.assertFalse(client.get("/api/health").json()["fixture_mode"])

    def test_decisions_separate_and_audit_has_no_content(self):
        key = self.post()
        before = self.client.get("/api/events/" + key + "/ai").json()["severity"]
        response = self.client.post(
            "/api/events/" + key + "/feedback",
            json={
                "grade": "A",
                "event_type_code": "EQ",
                "note": "Synthetic analyst correction",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(before, self.client.get("/api/events/" + key + "/ai").json()["severity"])
        audit = json.dumps(self.app.state.db.all("audit_log"))
        self.assertNotIn("scanner", audit)
        self.assertNotIn("Synthetic analyst correction", audit)
        self.assertNotIn("SYN-1", audit)
        self.assertIn("analyst.decision_recorded", audit)

    def test_local_boundary_and_redacted_validation(self):
        response = self.client.post(
            "/api/events",
            json={"event_comments": "SENSITIVE"},
            headers={"Origin": "https://outside.invalid"},
        )
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/events", json={"event_comments": "SENSITIVE"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("SENSITIVE", response.text)
        self.assertEqual(self.client.get("/api/health").headers["cache-control"], "no-store")

    def test_database_domain_tables(self):
        with self.app.state.db.connect() as db:
            names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(set(TABLES).issubset(names))

    def test_dashboard_review_counts_and_filters(self):
        first, second = self.post("DASH-ONE"), self.post("DASH-TWO")
        for _ in range(2):
            self.client.post(
                f"/api/events/{first}/feedback",
                json={"grade": "A", "event_type_code": "EQ", "note": "PRIVATE ANALYST NOTE"},
            )
        result = self.client.get("/api/dashboard").json()
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["reviewed_count"], 1)
        self.assertEqual(result["unreviewed_count"], 1)
        self.assertEqual(result["high_priority_count"], 2)
        self.assertEqual(result["priority_unreviewed_count"], 1)
        self.assertEqual([r["id"] for r in result["review_queue"]], [second])
        self.assertNotIn("PRIVATE ANALYST NOTE", json.dumps(result))
        self.assertNotIn("event_comments", result["review_queue"][0])
        self.assertNotIn("prescribed_medication", result["recent_events"][0])
        for filter_name, expected in [
            ("reviewed", first),
            ("unreviewed", second),
            ("priority", second),
        ]:
            listing = self.client.get(f"/api/events?filter={filter_name}&limit=1").json()
            self.assertEqual(listing["total"], 1)
            self.assertEqual(listing["items"][0]["id"], expected)
        self.assertEqual(self.client.get("/api/events?filter=invalid").status_code, 422)

    def test_dashboard_pending_failed_and_empty(self):
        empty = self.client.get("/api/dashboard").json()
        self.assertEqual(empty["event_count"], 0)
        self.assertEqual(empty["review_queue"], [])
        key = self.post("DASH-FAIL")
        value = self.app.state.db.get("ai_results", key)
        value["status"] = "failed"
        value["severity"] = None
        self.app.state.db.put("ai_results", key, value)
        pending = self.app.state.db.create_event(event("DASH-PENDING"))
        result = self.client.get("/api/dashboard").json()
        self.assertEqual(result["analysis_issue_count"], 1)
        self.assertEqual(result["pending_analysis_count"], 1)
        self.assertEqual(result["unreviewed_count"], 2)
        self.assertEqual(
            self.client.get("/api/events?filter=analysis_issues").json()["items"][0]["id"], key
        )
        self.assertIn(pending["id"], [r["id"] for r in result["review_queue"]])

    def test_dashboard_filtering_happens_before_pagination(self):
        # Direct authored fixtures keep this pagination test fast and model-independent.
        from backend.contracts import FeedbackInput

        for index in range(55):
            row = self.app.state.db.create_event(event(f"DASH-PAGE-{index}"))
            self.app.state.db.feedback(row["id"], FeedbackInput(grade="A", event_type_code="EQ"))
        result = self.client.get("/api/events?filter=reviewed&offset=50&limit=50").json()
        self.assertEqual(result["total"], 55)
        self.assertEqual(len(result["items"]), 5)
        self.assertTrue(all(r["review_status"] == "reviewed" for r in result["items"]))
        self.assertEqual(self.client.get("/api/events?filter=priority").json()["total"], 0)

    def test_dashboard_marks_old_trends_as_stale(self):
        from backend.dashboard import overview

        base = dict(
            events=[],
            ai_results=[],
            analyst_feedback=[],
            themes=[dict(theme_id="old", centroid=[1]), dict(theme_id="new", centroid=[1])],
            trend_results=[
                dict(theme_id="old", status="emerging", period_end="2026-10-04"),
                dict(theme_id="new", status="emerging", period_end="2026-10-11"),
            ],
        )
        result = overview(base, today=date(2026, 10, 13))
        self.assertTrue(result["patterns"][0]["trend_stale"])
        self.assertFalse(result["patterns"][1]["trend_stale"])
        self.assertEqual(result["emerging_pattern_count"], 1)
        self.assertNotIn("centroid", result["patterns"][0])

    def test_complete_workflow_survives_service_restart(self):
        # Authored synthetic reports exercise every persistence/service boundary.
        for report in synthetic_events()[:3]:
            response = self.client.post("/api/events", json=report.model_dump(mode="json"))
            self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.post("/api/patterns/rebuild").status_code, 202)
        report = synthetic_events()[3].model_copy(update={"event_id": "DEMO-E2E-RESTART"})
        response = self.client.post("/api/events", json=report.model_dump(mode="json"))
        key = response.json()["id"]
        ai = self.client.get(f"/api/events/{key}/ai").json()
        self.assertEqual(ai["status"], "complete")
        self.assertEqual(len(ai["similar_events"]), 3)
        self.assertEqual(len(ai["themes"]), 1)
        theme_id = ai["themes"][0]["theme_id"]
        self.assertEqual(self.client.get(f"/api/patterns/{theme_id}/events").json()["total"], 4)
        self.assertEqual(ai["trend_context"][0]["theme_id"], theme_id)
        self.assertEqual(self.client.get("/api/dashboard").json()["event_count"], 4)
        self.assertEqual(
            self.client.post(
                f"/api/events/{key}/feedback",
                json={"grade": "B2", "event_type_code": "ME", "note": "Synthetic review"},
            ).status_code,
            201,
        )
        before = self.client.post(
            "/api/search", json={"query": report.event_comments, "top_n": 4}
        ).json()
        # A fresh app reconstructs its vector index and all UI data from SQLite.
        with TestClient(create_app(self.settings)) as restarted:
            after = restarted.post(
                "/api/search", json={"query": report.event_comments, "top_n": 4}
            ).json()
            self.assertEqual(before, after)
            stored = restarted.get(f"/api/events/{key}").json()
            self.assertEqual(stored["analyst_decisions"][0]["grade"], "B2")
            self.assertEqual(
                restarted.get(f"/api/events/{key}/ai").json()["severity"]["grade"], "E"
            )
            self.assertEqual(restarted.get(f"/api/patterns/{theme_id}").json()["event_count"], 4)
            self.assertEqual(restarted.get(f"/api/trends/{theme_id}").status_code, 200)


class UpstreamContractTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MIDAS_TEST_SEVERITY_REPO"),
        "Optional upstream checkout not configured",
    )
    def test_severity_artifact_loading_and_standalone_entrypoint(self):
        import joblib

        root = Path(os.environ["MIDAS_TEST_SEVERITY_REPO"])
        pipeline = import_source("standalone_pipeline", root / "data_pipeline.py")
        upstream = import_source(
            "standalone_model", root / "severity_model.py", {"data_pipeline": pipeline}
        )
        entry = import_source(
            "standalone_inference",
            Path(__file__).resolve().parents[1] / "integrations/severity_inference.py",
            {"data_pipeline": pipeline, "severity_model": upstream},
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name, obj in [
                ("vectorizer", SyntheticVectorizer()),
                ("model", SyntheticGradeModel()),
                ("calibrator", SyntheticCalibrator()),
            ]:
                joblib.dump(obj, path / (name + ".joblib"))
            (path / "config.json").write_text(json.dumps({"cutoff_raw_triage": 0.4}))
            adapter = SeverityService.load_model(str(root), folder)
            standalone = entry.load_model(folder)
            result = adapter.predict_event(event()).model_dump()
            direct = entry.predict_event(standalone, model_frame([event()]).iloc[0].to_dict())
            for field in [
                "grade",
                "bucket",
                "grade_probabilities",
                "bucket_probabilities",
                "triage_score_raw",
                "harm_probability_calibrated",
                "priority_flag",
            ]:
                self.assertEqual(result[field], direct[field])
            self.assertTrue(direct["priority_flag"])
            self.assertEqual(
                len(
                    entry.predict_batch(standalone, [model_frame([event()]).iloc[0].to_dict()] * 2)
                ),
                2,
            )
            with self.assertRaises(ValueError):
                entry.predict_event(
                    standalone,
                    dict(model_frame([event()]).iloc[0], manager_comments="TARGET"),
                )

    @unittest.skipUnless(
        os.environ.get("MIDAS_TEST_SEVERITY_REPO"),
        "Optional upstream checkout not configured",
    )
    def test_exact_builders_and_severity_functions(self):
        root = Path(os.environ["MIDAS_TEST_SEVERITY_REPO"])
        pipeline = import_source("test_severity_data", root / "data_pipeline.py")
        model = import_source(
            "test_severity_model",
            root / "severity_model.py",
            {"data_pipeline": pipeline},
        )
        for e in [
            event(),
            event(
                age_at_encounter=71,
                location_name="Demo ICU",
                administered_medication="Demo drug",
                administered_dose="5 mg",
            ),
        ]:
            self.assertEqual(
                normalized_text(e), pipeline.build_input_text(model_frame([e])).iloc[0]
            )
        p = np.array([[0.6, 0.1, 0.3]])
        np.testing.assert_allclose(model.bucket_probs(p, np.array([0, 9, 5])), [[0.6, 0.3, 0.1]])
        np.testing.assert_allclose(model.triage_score(p, np.array([0, 9, 5])), [0.4])

    @unittest.skipUnless(
        os.environ.get("MIDAS_TEST_EVENT_TYPE_REPO"),
        "Optional upstream checkout not configured",
    )
    def test_exact_event_builder_and_checkpoint_loader(self):
        import torch

        root = Path(os.environ["MIDAS_TEST_EVENT_TYPE_REPO"])
        pipeline = import_source("test_event_data", root / "src/pipeline.py")
        mtl = import_source("test_event_mtl", root / "src/mtl.py", {"pipeline": pipeline})
        self.assertEqual(
            normalized_text(event()),
            pipeline.build_input_text(model_frame([event()])).iloc[0],
        )
        model = mtl.EmbeddingMultiTask(384, hidden=256, dropout=0.2)
        # Synthetic zero-weight contract fixture, never a substitute for trained weights.
        for parameter in model.parameters():
            parameter.data.zero_()

        class Encoder:
            def encode(self, texts, **kwargs):
                assert kwargs["normalize_embeddings"] is False
                return np.ones((len(texts), 384), dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            torch.save(model.state_dict(), path / "model_seed0.pt")
            (path / "train_log_seed0.json").write_text(
                json.dumps(
                    {
                        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                        "use_event_probs": True,
                        "config": {"hidden": 256, "dropout": 0.2},
                    }
                )
            )
            service = EventTypeService.load_model(str(root), folder, Encoder())
            output = service.predict_event(event())
            self.assertEqual(tuple(output.probabilities), CODES)
            self.assertAlmostEqual(sum(output.probabilities.values()), 1, places=5)


if __name__ == "__main__":
    unittest.main()

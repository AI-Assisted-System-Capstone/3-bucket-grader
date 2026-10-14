from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_engine import PatternEngine, build_report_text, normalize_schema


class PatternEngineTests(unittest.TestCase):
    def test_schema_aliases_and_no_target_leakage(self):
        raw = pd.DataFrame({
            "Event No.": ["E1"],
            "Event Date": ["2024-01-01"],
            "Event Comments": ["Wrong insulin dose was prepared."],
            "Event Type": ["Medication error"],
            "MANAGER COMMENTS": ["Manager says severe harm category."],
            "Significance (PSRS Harm score)": ["I-Harm-Death"],
        })
        df = normalize_schema(raw)
        text = build_report_text(df, include_review_fields=False).iloc[0]
        self.assertIn("Wrong insulin dose", text)
        self.assertIn("Medication error", text)
        self.assertNotIn("Manager says", text)
        self.assertNotIn("I-Harm-Death", text)

    def test_retrieval_and_novelty(self):
        df = pd.DataFrame({
            "event_id": ["A", "B", "C", "D"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "date_received": pd.NaT,
            "location": ["", "", "", ""],
            "service": ["", "", "", ""],
            "event_type": ["Medication", "Medication", "Imaging", "Fall"],
            "age": ["", "", "", ""],
            "prescribed_med": ["insulin", "insulin", "", ""],
            "prescribed_dose": ["10 units", "12 units", "", ""],
            "admin_med": ["", "", "", ""],
            "admin_dose": ["", "", "", ""],
            "adr_med": ["", "", "", ""],
            "adr_dose": ["", "", "", ""],
            "event_comments": [
                "Wrong insulin dose was prepared for the patient.",
                "Patient almost received an incorrect insulin dose.",
                "CT scan scheduling was delayed by the radiology queue.",
                "Patient slipped while walking in the hallway.",
            ],
            "manager_comments": ["", "", "", ""],
            "unit_actions": ["", "", "", ""],
            "shareable_lessons": ["", "", "", ""],
            "harm_score": ["", "", "", ""],
            "hpi_designation": ["", "", "", ""],
            "investigation_level": ["", "", "", ""],
            "analyst_report_type": ["", "", "", ""],
        })
        engine = PatternEngine(force_tfidf=True, n_clusters=2, retrieval_top_k=3)
        engine.fit(df)
        result = engine.search("An incorrect insulin dose was nearly administered", top_k=3)
        json.dumps(result)
        ids = [x["event_id"] for x in result["matches"]]
        self.assertTrue("A" in ids or "B" in ids)

        neighbors = engine.retrieval_neighbors(top_k=engine.retrieval_top_k)
        self.assertEqual(len(neighbors), len(df) * 3)

        with self.assertRaises(ValueError):
            engine.search("anything", top_k=0)

        novel = engine.search("A helicopter struck the hospital roof during a storm", top_k=1)
        self.assertGreaterEqual(novel["novelty_score"], result["novelty_score"])

    def test_trend_fills_zero_weeks(self):
        base_cols = {
            "date_received": pd.NaT, "location": "", "service": "", "event_type": "Medication",
            "age": "", "prescribed_med": "", "prescribed_dose": "", "admin_med": "", "admin_dose": "",
            "adr_med": "", "adr_dose": "", "manager_comments": "", "unit_actions": "", "shareable_lessons": "",
            "harm_score": "", "hpi_designation": "", "investigation_level": "", "analyst_report_type": "",
        }
        rows = []
        dates = ["2024-01-01", "2024-01-15", "2024-02-12", "2024-02-19", "2024-02-20", "2024-02-21"]
        for i, d in enumerate(dates):
            row = dict(base_cols)
            row.update(event_id=f"E{i}", event_date=pd.Timestamp(d), event_comments=f"Medication barcode error number {i}")
            rows.append(row)
        engine = PatternEngine(force_tfidf=True, n_clusters=2)
        engine.fit(pd.DataFrame(rows))
        trends = engine.trend_table()
        self.assertTrue((trends["count"] == 0).any())

        monthly = engine.trend_table(freq="M")
        self.assertEqual(set(monthly["period"].dt.day), {1})
        self.assertEqual(
            set(monthly["period"].dt.strftime("%Y-%m")),
            {"2024-01", "2024-02"},
        )


if __name__ == "__main__":
    unittest.main()

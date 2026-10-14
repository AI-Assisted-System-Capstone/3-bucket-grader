"""Round-trip check on the saved model: the files in severity_model/ load
from disk and score a report on their own, through predict.py.

The .joblib files are gitignored, so these tests are skipped until
severity_model.py has been run on this machine."""

import json
import os

import pandas as pd
import pytest

from data_pipeline import FEATURE_COLUMNS, FULL_LABELS, LABELS
from predict import load_model, score_reports
from severity_model import OUT_DIR

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), OUT_DIR)
ARTIFACTS = ("vectorizer.joblib", "model.joblib", "calibrator.joblib", "config.json")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in ARTIFACTS),
    reason="saved model not found in severity_model/ -- run severity_model.py first",
)


@pytest.fixture
def reports():
    blank = {c: None for c in FEATURE_COLUMNS}
    return pd.DataFrame([
        {**blank,
         "event_comments": "Patient found on floor at 0300, alert and oriented, "
                           "complaining of left hip pain. X-ray confirmed fracture. "
                           "Transferred to OR for repair.",
         "Location Name": "ICU", "Encounter Service": "Cardiology", "Age at Encounter": 74,
         "ME - Prescribed - Name *... Name": "Warfarin", "ME - Prescribed - Dose *": "5mg"},
        {**blank,
         "event_comments": "Wrong label printed for specimen, caught before it left the unit.",
         "Location Name": "Lab"},
    ])


def test_saved_model_scores_reports(reports):
    _, _, _, cutoff = load_model(MODEL_DIR)
    out = score_reports(reports, MODEL_DIR)

    assert len(out) == len(reports)
    assert set(out["grade_label"]) <= set(FULL_LABELS)
    assert all(label.split("-", 1)[0] == grade for grade, label in zip(out["grade"], out["grade_label"]))
    assert set(out["bucket"]) <= set(LABELS)
    assert out["triage_score_raw"].between(0, 1).all()
    assert out["triage_probability"].between(0, 1).all()
    assert (out["flagged"].astype(bool) == (out["triage_score_raw"] >= cutoff)).all()


def test_harm_report_outranks_near_miss(reports):
    out = score_reports(reports, MODEL_DIR)
    fracture, near_miss = out["triage_score_raw"]
    assert fracture > near_miss


def test_scoring_is_repeatable(reports):
    pd.testing.assert_frame_equal(score_reports(reports, MODEL_DIR), score_reports(reports, MODEL_DIR))


def test_post_review_columns_do_not_change_scores(reports):
    with_leaky = reports.assign(
        manager_comments="Patient died. Reported to the state.",
        unit_actions_taken="RCA opened.",
        shareable_lessons="Serious harm event.",
    )
    pd.testing.assert_frame_equal(score_reports(with_leaky, MODEL_DIR), score_reports(reports, MODEL_DIR))


def test_report_without_text_is_not_scored(reports):
    reports.loc[1, list(FEATURE_COLUMNS)] = None
    out = score_reports(reports, MODEL_DIR)
    assert pd.notna(out.loc[0, "grade"])
    assert out.loc[1].isna().all()


def test_config_cutoff_matches_evaluation():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, "evaluation.json")) as f:
        evaluation = json.load(f)
    assert config["cutoff_raw_triage"] == evaluation["triage"]["cutoff_chosen_on_validation"]

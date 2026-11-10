import json
import os

import numpy as np
import pandas as pd
import pytest

from data_pipeline import BUCKET_GROUPS, FULL_LABELS, LABELS, build_input_text
from predict import load_model, main as predict_main, score_reports
from severity_model import OUT_DIR, bucket_probs, save_model, triage_score
from conftest import make_reports

N_GRADES = len(FULL_LABELS)


# --- bucket_probs / triage_score --------------------------------------------

def _random_proba(n, k, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(k), size=n)


def test_bucket_probs_sum_to_one():
    proba = _random_proba(500, N_GRADES)
    out = bucket_probs(proba, np.arange(N_GRADES))
    assert out.shape == (500, len(LABELS))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_bucket_probs_with_missing_class():
    # A model trained without any grade-I rows has only 9 classes.
    classes = np.arange(N_GRADES - 1)
    proba = _random_proba(200, len(classes), seed=1)
    np.testing.assert_allclose(bucket_probs(proba, classes).sum(axis=1), 1.0)


def test_bucket_probs_respects_class_order():
    classes = np.random.default_rng(2).permutation(N_GRADES)
    proba = _random_proba(100, N_GRADES, seed=3)
    out = bucket_probs(proba, classes)
    for bi, bucket in enumerate(LABELS):
        cols = [i for i, c in enumerate(classes) if c in BUCKET_GROUPS[bucket]]
        np.testing.assert_allclose(out[:, bi], proba[:, cols].sum(axis=1))


def test_triage_is_some_plus_serious():
    classes = np.arange(N_GRADES)
    proba = _random_proba(300, N_GRADES, seed=4)
    b = bucket_probs(proba, classes)
    np.testing.assert_allclose(triage_score(proba, classes), b[:, 1] + b[:, 2])


# --- Save / load round trip -------------------------------------------------

@pytest.fixture
def saved_dir(tmp_path, tiny_model):
    save_model(str(tmp_path), *tiny_model)
    return tmp_path


def test_round_trip_matches_in_memory(saved_dir, tiny_model):
    vectorizer, clf, calibrator, cutoff = tiny_model
    reports = make_reports([f"grade{g}word event0 narrative" for g in range(N_GRADES)])

    model = load_model(str(saved_dir))
    assert model.cutoff == cutoff
    np.testing.assert_array_equal(model.clf.classes_, clf.classes_)

    out = score_reports(reports, model)

    expected = clf.predict_proba(vectorizer.transform(build_input_text(reports)))
    np.testing.assert_allclose(out[[f"p_{b}" for b in LABELS]].to_numpy(),
                               bucket_probs(expected, clf.classes_))
    assert list(out["grade"]) == [FULL_LABELS[c] for c in clf.predict(vectorizer.transform(build_input_text(reports)))]
    np.testing.assert_allclose(out[[f"p_{b}" for b in LABELS]].sum(axis=1), 1.0)
    assert (out["flagged"] == (out["triage_score_raw"] >= cutoff)).all()
    assert out["triage_probability"].between(0, 1).all()


def test_load_model_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="severity_model.py"):
        load_model(str(tmp_path))


def test_score_ignores_post_investigation_columns(saved_dir):
    model = load_model(str(saved_dir))
    clean = make_reports(["grade0word event1 narrative", "grade9word event2 narrative"])
    leaky = clean.assign(
        manager_comments=["grade9word patient died", "grade9word patient died"],
        unit_actions_taken=["grade9word", "grade9word"],
    )
    pd.testing.assert_frame_equal(score_reports(clean, model), score_reports(leaky, model))


def test_score_without_optional_intake_columns(saved_dir):
    model = load_model(str(saved_dir))
    out = score_reports(pd.DataFrame({"event_comments": ["grade3word"]}), model)
    assert len(out) == 1


def test_score_requires_narrative(saved_dir):
    with pytest.raises(ValueError, match="event_comments"):
        score_reports(pd.DataFrame({"Location Name": ["ICU"]}), load_model(str(saved_dir)))


@pytest.mark.parametrize("in_ext, out_ext", [(".csv", ".parquet"), (".parquet", ".csv")])
def test_cli_round_trip(saved_dir, tmp_path, in_ext, out_ext):
    reports = make_reports(["grade0word", "grade7word"]).assign(**{"Event No.": ["E-1", "E-2"]})
    src, dst = tmp_path / f"in{in_ext}", tmp_path / f"out{out_ext}"
    reports.to_csv(src, index=False) if in_ext == ".csv" else reports.to_parquet(src, index=False)

    predict_main([str(src), "-o", str(dst), "--model-dir", str(saved_dir)])

    out = pd.read_csv(dst) if out_ext == ".csv" else pd.read_parquet(dst)
    assert list(out["event_no"]) == ["E-1", "E-2"]
    assert {"grade", "bucket", "triage_probability", "flagged"} <= set(out.columns)
    assert "event_comments" not in out.columns


# --- The real trained artifacts, when present -------------------------------

needs_trained = pytest.mark.skipif(
    not os.path.exists(os.path.join(OUT_DIR, "model.joblib")),
    reason="severity_model/*.joblib not trained (run severity_model.py)",
)


@needs_trained
def test_saved_production_model_loads_and_scores():
    model = load_model(OUT_DIR)
    assert list(model.clf.classes_) == list(range(N_GRADES))
    with open(os.path.join(OUT_DIR, "evaluation.json")) as f:
        assert model.cutoff == pytest.approx(json.load(f)["triage"]["cutoff_chosen_on_validation"])

    out = score_reports(make_reports([
        "Patient found on floor, X-ray confirmed hip fracture, transferred to OR.",
        "Wrong dose drawn up but caught by second nurse before administration.",
    ]), model)
    # Structure only: which way a given sentence scores changes with every retrain.
    assert len(out) == 2
    assert {"grade", "bucket", "p_none", "p_some", "p_serious", "triage_score_raw",
            "triage_probability", "flagged"} <= set(out.columns)
    np.testing.assert_allclose(out[[f"p_{b}" for b in LABELS]].sum(axis=1), 1.0)
    assert set(out["grade"]) <= set(FULL_LABELS)
    assert set(out["bucket"]) <= set(LABELS)
    assert out["triage_score_raw"].between(0, 1).all()
    assert out["triage_probability"].between(0, 1).all()
    assert (out["flagged"] == (out["triage_score_raw"] >= model.cutoff)).all()

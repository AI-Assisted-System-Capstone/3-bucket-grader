"""Checks on the probability sums that derive the 3-bucket label and triage
score from the 10-grade model. No dataset or saved model needed."""

import numpy as np
import pytest

from data_pipeline import FULL_LABELS, LABELS
from severity_model import bucket_probs, pick_cutoff_for_recall, triage_score

ALL_CLASSES = np.arange(len(FULL_LABELS))
# A classifier only has a column for grades it saw in training.
MISSING_ONE_GRADE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])


def random_proba(n_classes, n_rows=200):
    return np.random.default_rng(0).dirichlet(np.ones(n_classes), size=n_rows)


@pytest.mark.parametrize("classes", [ALL_CLASSES, MISSING_ONE_GRADE])
def test_bucket_probs_sum_to_one(classes):
    out = bucket_probs(random_proba(len(classes)), classes)
    assert out.shape == (200, len(LABELS))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_bucket_probs_known_row():
    #                  A    B1   B2   C    D    E    F    G     H     I
    proba = np.array([[.1, .05, .05, .1, .2, .25, .1, .05, .06, .04]])
    assert np.allclose(bucket_probs(proba, ALL_CLASSES), [[0.5, 0.35, 0.15]])


@pytest.mark.parametrize("classes", [ALL_CLASSES, MISSING_ONE_GRADE])
def test_triage_score_is_some_plus_serious(classes):
    proba = random_proba(len(classes))
    buckets = bucket_probs(proba, classes)
    expected = buckets[:, LABELS.index("some")] + buckets[:, LABELS.index("serious")]
    assert np.allclose(triage_score(proba, classes), expected)


def test_cutoff_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.4 * y, 0.15), 0, 1)
    cutoff = pick_cutoff_for_recall(y, scores, 0.95)
    assert (scores[y == 1] >= cutoff).mean() >= 0.95

import os
import sys

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import FEATURE_COLUMNS, FULL_LABELS, build_input_text  # noqa: E402


def make_reports(narratives, **extra):
    """Raw-report frame with every feature column, blanks for the intake fields."""
    df = pd.DataFrame({c: [None] * len(narratives) for c in FEATURE_COLUMNS})
    df["event_comments"] = narratives
    df["Location Name"] = "ICU"
    for col, vals in extra.items():
        df[col] = vals
    return df


@pytest.fixture(scope="session")
def tiny_model():
    """A small model in the production artifact format, trained on toy text so
    the round-trip tests don't need the dataset or a trained severity_model/."""
    narratives, labels = [], []
    for gid in range(len(FULL_LABELS)):
        for k in range(4):
            narratives.append(f"grade{gid}word event{k} narrative")
            labels.append(gid)
    texts = build_input_text(make_reports(narratives))
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000).fit(X, labels)
    hurt = [int(l >= 5) for l in labels]
    calibrator = LogisticRegression().fit(
        clf.predict_proba(X)[:, 5:].sum(axis=1).reshape(-1, 1), hurt
    )
    return vectorizer, clf, calibrator, 0.3

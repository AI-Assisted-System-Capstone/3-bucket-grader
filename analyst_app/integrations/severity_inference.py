"""Drop-in inference.py for the inspected 3-bucket-grader repository.

Copy this file next to severity_model.py. No training occurs on import. Call
load_model once at startup, then predict_event/predict_batch with mappings of the
upstream intake column names. This standalone entrypoint does not need FastAPI.
Only trusted joblib artifacts may be loaded.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data_pipeline import FEATURE_COLUMNS, FULL_LABELS, LABELS, build_input_text
from severity_model import bucket_probs, triage_score


class SeverityPredictor:
    def __init__(self, artifact_dir):
        folder = Path(artifact_dir)
        paths = [
            folder / name
            for name in (
                "vectorizer.joblib",
                "model.joblib",
                "calibrator.joblib",
                "config.json",
            )
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(
                "Severity vectorizer, model, calibrator and config are required"
            )
        self.vectorizer, self.model, self.calibrator = (joblib.load(path) for path in paths[:3])
        self.cutoff = float(json.loads(paths[3].read_text())["cutoff_raw_triage"])
        if not 0 <= self.cutoff <= 1 or set(self.model.classes_) - set(range(len(FULL_LABELS))):
            raise ValueError("Invalid severity artifact contract")
        digest = hashlib.sha256()
        for path in paths:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        self.model_version = "severity-" + digest.hexdigest()[:16]

    def predict_batch(self, events):
        if not events:
            return []
        # Reject label/review fields rather than silently consuming them.
        if any(set(event) - set(FEATURE_COLUMNS) for event in events):
            raise ValueError("Only upstream intake FEATURE_COLUMNS are accepted")
        text = build_input_text(pd.DataFrame(events))
        values = self.model.predict_proba(self.vectorizer.transform(text))
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1)
        ):
            raise ValueError("Invalid grade probabilities")
        buckets, raw = (
            bucket_probs(values, self.model.classes_),
            triage_score(values, self.model.classes_),
        )
        calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[
            :, list(self.calibrator.classes_).index(1)
        ]
        if not np.isfinite(calibrated).all() or (calibrated < 0).any() or (calibrated > 1).any():
            raise ValueError("Invalid calibrated probabilities")
        results = []
        for i, row in enumerate(values):
            grades = {label.split("-")[0]: 0.0 for label in FULL_LABELS}
            grades.update(
                {
                    FULL_LABELS[int(c)].split("-")[0]: float(p)
                    for c, p in zip(self.model.classes_, row)
                }
            )
            bp = dict(zip(LABELS, map(float, buckets[i])))
            results.append(
                dict(
                    grade=FULL_LABELS[int(self.model.classes_[row.argmax()])].split("-")[0],
                    grade_probabilities=grades,
                    bucket=max(bp, key=bp.get),
                    bucket_probabilities=bp,
                    triage_score_raw=float(raw[i]),
                    harm_probability_calibrated=float(calibrated[i]),
                    priority_flag=bool(raw[i] >= self.cutoff),
                    model_version=self.model_version,
                )
            )
        return results

    def predict_event(self, event):
        return self.predict_batch([event])[0]


def load_model(artifact_dir=None):
    return SeverityPredictor(artifact_dir or Path(__file__).resolve().parent / "severity_model")


def predict_event(model, event):
    return model.predict_event(event)


def predict_batch(model, events):
    return model.predict_batch(events)

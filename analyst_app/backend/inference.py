"""Inference-only integration: upstream training modules are imported, never executed.

Only load trusted, locally supplied repositories and pickle artifacts. Functions
are intentionally exposed here so neither upstream training pipeline must change.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .contracts import (
    CODES,
    GRADES,
    LABELS,
    EventTypeResult,
    SeverityResult,
    model_frame,
)


def import_source(name, path, dependencies=None):
    """Isolated source import with upstream's absolute imports bound temporarily."""
    dependencies = dependencies or {}
    saved = {key: sys.modules.get(key) for key in dependencies}
    try:
        sys.modules.update(dependencies)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def probabilities(values, count):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != count
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid model probability contract")
    return values


class UnavailableModel:
    version = "unavailable"

    def predict_event(self, event):
        raise RuntimeError("Model artifacts unavailable; consult local startup status")

    def predict_batch(self, events):
        return [self.predict_event(event) for event in events]


class SeverityService:
    @classmethod
    def load_model(cls, repo, artifacts):
        import joblib

        self = cls()
        root, folder = Path(repo), Path(artifacts)
        names = [
            "vectorizer.joblib",
            "model.joblib",
            "calibrator.joblib",
            "config.json",
        ]
        if not repo or not artifacts or not all((folder / name).is_file() for name in names):
            raise FileNotFoundError("Missing severity repository or artifacts")
        pipeline = import_source("midas_severity_pipeline", root / "data_pipeline.py")
        self.upstream = import_source(
            "midas_severity_model",
            root / "severity_model.py",
            {"data_pipeline": pipeline},
        )
        self.builder = pipeline.build_input_text
        self.vectorizer = joblib.load(folder / names[0])
        self.model = joblib.load(folder / names[1])
        self.calibrator = joblib.load(folder / names[2])
        self.cutoff = float(json.loads((folder / names[3]).read_text())["cutoff_raw_triage"])
        feature_names = getattr(self.vectorizer, "get_feature_names_out", None)
        self.terms = np.asarray(feature_names()) if feature_names else None  # None: no words
        if not 0 <= self.cutoff <= 1 or set(self.model.classes_) - set(range(10)):
            raise ValueError("Invalid severity artifact contract")
        self.version = "severity-" + fingerprint(
            [folder / name for name in names]
            + [root / "data_pipeline.py", root / "severity_model.py"]
        )
        return self

    def predict_batch(self, events):
        if not events:
            return []
        text = self.builder(model_frame(events))
        features = self.vectorizer.transform(text)
        p = probabilities(self.model.predict_proba(features), len(self.model.classes_))
        buckets = probabilities(self.upstream.bucket_probs(p, self.model.classes_), 3)
        raw = self.upstream.triage_score(p, self.model.classes_)
        calibrated = probabilities(self.calibrator.predict_proba(raw.reshape(-1, 1)), 2)
        positive = list(self.calibrator.classes_).index(1)
        explainable = getattr(self, "terms", None) is not None and hasattr(self.model, "coef_")
        results = []
        for i in range(len(events)):
            harm_words, safe_words = (
                self.explain(features[i], p[i], raw[i]) if explainable else ([], [])
            )
            grades = dict.fromkeys(GRADES, 0.0)
            grades.update(
                {GRADES[int(c)]: float(value) for c, value in zip(self.model.classes_, p[i])}
            )
            bucket = dict(zip(("none", "some", "serious"), map(float, buckets[i])))
            results.append(
                SeverityResult(
                    grade=GRADES[int(self.model.classes_[p[i].argmax()])],
                    grade_probabilities=grades,
                    bucket=max(bucket, key=bucket.get),
                    bucket_probabilities=bucket,
                    triage_score_raw=float(raw[i]),
                    harm_probability_calibrated=float(calibrated[i, positive]),
                    priority_flag=bool(raw[i] >= self.cutoff),
                    model_version=self.version,
                    harm_words=harm_words,
                    safe_words=safe_words,
                )
            )
        return results

    def explain(self, row, p, triage, top=5):
        """Which words in this report moved its harm score, and by how much.

        The harm score is the summed probability of grades E-I. For a softmax
        model its slope along word j is sum_k p_k (harm_k - score) w_kj, so a
        word's effect is its TF-IDF value times that slope. Exact for this
        model's linear layer; an estimate of the effect on the probability.
        """
        harm = np.isin(self.model.classes_, [GRADES.index(g) for g in "EFGHI"]).astype(float)
        slope = (p * (harm - triage)) @ self.model.coef_
        row = row.tocsr()
        effect = row.data * slope[row.indices]
        order = np.argsort(effect)
        def words(indices):
            return [
                dict(term=str(self.terms[row.indices[j]]), effect=round(float(effect[j]), 4))
                for j in indices
            ]
        up = [j for j in order[::-1][:top] if effect[j] > 0]
        down = [j for j in order[:top] if effect[j] < 0]
        return words(up), words(down)

    def predict_event(self, event):
        return self.predict_batch([event])[0]


class EventTypeService:
    @classmethod
    def load_model(cls, repo, artifacts, encoder=None):
        import torch
        from sentence_transformers import SentenceTransformer

        self = cls()
        root, folder = Path(repo) / "src", Path(artifacts)
        checkpoint, log = folder / "model_seed0.pt", folder / "train_log_seed0.json"
        if not repo or not artifacts or not checkpoint.is_file() or not log.is_file():
            raise FileNotFoundError("Missing event-type baseline artifacts")
        metadata = json.loads(log.read_text())
        if metadata["encoder"] != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("Only the documented frozen MiniLM baseline is supported")
        pipeline = import_source("midas_event_pipeline", root / "pipeline.py")
        mtl = import_source("midas_event_mtl", root / "mtl.py", {"pipeline": pipeline})
        if tuple(pipeline.EVENT_BUCKETS) != CODES:
            raise ValueError("Upstream class order changed")
        self.builder = pipeline.build_input_text
        cfg = metadata["config"]
        self.model = mtl.EmbeddingMultiTask(
            384,
            use_event_probs=metadata["use_event_probs"],
            hidden=cfg["hidden"],
            dropout=cfg["dropout"],
        )
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.model.eval()
        self.encoder = encoder or SentenceTransformer(
            metadata["encoder"], device="cpu", local_files_only=True
        )
        self.version = "event-type-" + fingerprint(
            [checkpoint, log, root / "pipeline.py", root / "mtl.py"]
        )
        return self

    def predict_batch(self, events):
        import torch

        if not events:
            return []
        texts = self.builder(model_frame(events)).tolist()
        # Training used unnormalized vectors. Do NOT reuse cosine-normalized pattern vectors.
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        with torch.inference_mode():
            logits, _auxiliary_hurt = self.model(torch.as_tensor(embeddings, dtype=torch.float32))
            values = probabilities(torch.softmax(logits, dim=-1).numpy(), 11)
        return [
            EventTypeResult(
                event_type_code=CODES[int(p.argmax())],
                event_type_label=LABELS[CODES[int(p.argmax())]],
                confidence=float(p.max()),
                probabilities=dict(zip(CODES, map(float, p))),
                model_version=self.version,
            )
            for p in values
        ]

    def predict_event(self, event):
        return self.predict_batch([event])[0]


class FixtureSeverity:
    """Fixed synthetic UI values, NOT a clinical prediction or calibrated estimate."""

    version = "FIXTURE-severity-v1"

    def predict_event(self, event):
        # Explicit fixed test vector: no random predictions, no invented trained weights.
        p = dict(zip(GRADES, [0.04, 0.03, 0.08, 0.08, 0.12, 0.4, 0.15, 0.04, 0.04, 0.02]))
        return SeverityResult(
            grade="E",
            grade_probabilities=p,
            bucket="some",
            bucket_probabilities={"none": 0.35, "some": 0.55, "serious": 0.10},
            triage_score_raw=0.65,
            harm_probability_calibrated=0.2,
            priority_flag=True,
            model_version=self.version,
            fixture=True,
        )

    def predict_batch(self, events):
        return [self.predict_event(event) for event in events]


class FixtureEventType:
    version = "FIXTURE-event-type-v1"

    def predict_event(self, event):
        code = "FALL" if "fall" in event.event_comments.lower() else "ME"
        p = {c: 0.02 for c in CODES}
        p[code] = 0.8
        return EventTypeResult(
            event_type_code=code,
            event_type_label=LABELS[code],
            confidence=0.8,
            probabilities=p,
            model_version=self.version,
            fixture=True,
        )

    def predict_batch(self, events):
        return [self.predict_event(event) for event in events]

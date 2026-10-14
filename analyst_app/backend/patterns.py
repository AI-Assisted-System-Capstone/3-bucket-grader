import hashlib
from collections import Counter
from datetime import date, timedelta
from typing import Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .contracts import EventInput, normalized_text, now


class VectorStore(Protocol):
    def replace(self, records: list, vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, top_n: int, before=None, exclude=None) -> list: ...


class CosineStore:
    """Exact O(Nd) search, no NxN matrix. Replace with pgvector behind this interface."""

    def __init__(self):
        self.records, self.vectors = [], np.empty((0, 384), dtype=np.float32)

    def replace(self, records, vectors):
        self.records = records
        self.vectors = (
            normalize(np.asarray(vectors, dtype=np.float32)) if len(vectors) else np.empty((0, 384))
        )

    def search(self, vector, top_n, before=None, exclude=None):
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if not self.records:
            return []
        vector = normalize(np.asarray(vector).reshape(1, -1))[0]
        scores = self.vectors @ vector
        eligible = [
            i
            for i, r in enumerate(self.records)
            if r["id"] != exclude and (before is None or r["event_date"] < before)
        ]
        # Stable secondary key makes ties reproducible. Strictly earlier dates avoid
        # claiming same-day events are historical when only a date is available.
        eligible.sort(key=lambda i: (-float(scores[i]), self.records[i]["id"]))
        return [
            dict(
                event_id=self.records[i]["id"],
                source_event_id=self.records[i].get("event_id"),
                similarity=float(np.clip(scores[i], -1, 1)),
                location_name=self.records[i]["location_name"],
                event_date=self.records[i]["event_date"],
            )
            for i in eligible[:top_n]
        ]


class LocalEncoder:
    def __init__(self, backend="minilm"):
        self.backend, self.model = backend, None
        if backend == "minilm":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                local_files_only=True,
            )
            self.version = "all-MiniLM-L6-v2-384-intake-v1"
        elif backend == "lexical":
            # Explicit offline lexical fallback. Stable vocabulary-free hashing means
            # persisted vectors remain comparable across inserts and process restarts.
            self.model = HashingVectorizer(
                n_features=384,
                alternate_sign=False,
                stop_words="english",
                ngram_range=(1, 2),
                norm="l2",
            )
            self.version = "LEXICAL-hashing-384-intake-v1"
        else:
            raise ValueError("MIDAS_EMBEDDING_BACKEND must be minilm or lexical")

    def encode(self, texts):
        if self.backend == "minilm":
            values = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            values = self.model.transform(texts).toarray()
        if values.shape != (len(texts), 384) or not np.isfinite(values).all():
            raise ValueError("Invalid embedding dimensions or values")
        return normalize(values).astype(np.float32)


class ThemeNamer(Protocol):
    def name(self, texts: list) -> dict: ...


class KeywordNamer:
    def name(self, texts):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        try:
            matrix = vectorizer.fit_transform(texts)
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], terms[i]))
            keywords = [str(terms[i]) for i in order[:6]]
        except ValueError:
            keywords = ["Recurring reports"]
        return dict(name=" · ".join(keywords[:3]), keywords=keywords)


def memberships_for(vector, themes, threshold):
    # Multi-label: independent threshold for every theme, not argmax.
    return [
        dict(theme_id=t["theme_id"], label=t["name"], membership_score=float(score))
        for t in themes
        for score in [np.clip(np.dot(vector, t["centroid"]), -1, 1)]
        if score >= threshold
    ]


def weekly_trend(theme_id, dates, settings, as_of=None):
    as_of = as_of or date.today()
    # Compare complete weeks to complete weeks; never a partial current week.
    week = as_of - timedelta(days=as_of.weekday() + 7)
    counts = Counter(d - timedelta(days=d.weekday()) for d in dates if d <= as_of)
    start = week - timedelta(weeks=settings.baseline_weeks)
    timeline = [
        dict(
            week=(start + timedelta(weeks=i)).isoformat(),
            count=counts[start + timedelta(weeks=i)],
        )
        for i in range(settings.baseline_weeks + 1)
    ]
    baseline_values = [point["count"] for point in timeline[:-1]]
    baseline, current = float(np.mean(baseline_values)), timeline[-1]["count"]
    # Poisson variance floor keeps zero/constant baselines finite and explainable.
    scale = max(float(np.std(baseline_values)), float(np.sqrt(baseline)), 1.0)
    z = (current - baseline) / scale
    percent = (current - baseline) / baseline * 100 if baseline > 0 else None
    support = current >= settings.minimum_event_count
    increase = current > 0 if baseline == 0 else percent >= settings.minimum_percent_increase
    status = (
        "emerging"
        if support and increase and z >= settings.z_score_threshold
        else ("watch" if support and increase else "stable")
    )
    return dict(
        theme_id=theme_id,
        current_count=current,
        baseline_count=baseline,
        percent_change=percent,
        new_from_zero=baseline == 0 and current > 0,
        trend_score=z,
        status=status,
        period_start=week.isoformat(),
        period_end=(week + timedelta(days=6)).isoformat(),
        as_of=as_of.isoformat(),
        timeline=timeline,
        created_at=now(),
        explanation="Last complete week versus prior zero-filled weeks; association, not causation.",
    )


class PatternService:
    def __init__(self, settings, database, encoder=None, store=None, namer=None):
        self.settings, self.db = settings, database
        self.encoder = encoder or LocalEncoder(settings.embedding_backend)
        self.version = self.encoder.version
        self.store = store or CosineStore()
        self.namer = namer or KeywordNamer()
        self.refresh()

    def refresh(self):
        records = {r["id"]: r for r in self.db.all("events")}
        stored = [
            e
            for e in self.db.all("event_embeddings")
            if e["model_version"] == self.version and e["event_id"] in records
        ]
        self.store.replace([records[e["event_id"]] for e in stored], [e["vector"] for e in stored])

    def embed(self, record):
        old = self.db.get("event_embeddings", record["id"])
        if old and old["model_version"] == self.version:
            return np.asarray(old["vector"], dtype=np.float32)
        event = EventInput(**{key: record[key] for key in EventInput.model_fields})
        vector = self.encoder.encode([normalized_text(event)])[0]
        self.db.put(
            "event_embeddings",
            record["id"],
            dict(
                event_id=record["id"],
                vector=vector.tolist(),
                model_version=self.version,
                created_at=now(),
            ),
        )
        return vector

    def analyze(self, record, top_n=None):
        vector = self.embed(record)
        self.refresh()
        similar = self.store.search(
            vector,
            top_n or self.settings.top_n,
            before=record["event_date"],
            exclude=record["id"],
        )
        themes = memberships_for(vector, self.db.all("themes"), self.settings.membership_threshold)
        self.db.set_memberships(record["id"], themes)
        self.db.put(
            "similar_events",
            record["id"],
            dict(
                event_id=record["id"],
                matches=similar,
                model_version=self.version,
                created_at=now(),
            ),
        )
        return similar, themes

    def search(self, query, top_n):
        # Queries use the same narrative-only form of the normalized intake representation.
        return self.store.search(self.encoder.encode([query])[0], top_n)

    def rebuild(self, as_of=None):
        records = self.db.all("events")
        if len(records) > self.settings.max_pattern_events:
            raise ValueError(
                "Pattern rebuild exceeds local MVP capacity; use a larger vector service"
            )
        vectors = np.array([self.embed(r) for r in records])
        self.refresh()
        if len(records) < self.settings.minimum_cluster_size:
            self.db.replace_patterns([], [], [])
            return []
        method = self.settings.cluster_method
        if method == "hdbscan":
            try:
                import hdbscan

                labels = hdbscan.HDBSCAN(
                    min_cluster_size=self.settings.minimum_cluster_size,
                    min_samples=1,
                    metric="euclidean",
                ).fit_predict(vectors)
            except ImportError:
                method = "agglomerative"
        if method == "agglomerative":
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=self.settings.cluster_distance,
            ).fit_predict(vectors)
        elif method != "hdbscan":
            raise ValueError("Unsupported cluster method")
        themes = []
        for label in sorted(set(labels) - {-1}):
            indices = np.flatnonzero(labels == label)
            if len(indices) < self.settings.minimum_cluster_size:
                continue
            centroid = normalize(vectors[indices].mean(axis=0).reshape(1, -1))[0]
            members = [records[i] for i in indices]
            representative = sorted(indices, key=lambda i: -float(vectors[i] @ centroid))[:3]
            # Content-derived identity is deterministic for identical rebuilds.
            identity = "|".join(sorted(r["id"] for r in members))
            theme_id = "theme-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            themes.append(
                dict(
                    theme_id=theme_id,
                    **self.namer.name([r["event_comments"] for r in members]),
                    centroid=centroid.tolist(),
                    representative_event_ids=[records[i]["id"] for i in representative],
                    cluster_method=method,
                    model_version=self.version,
                    created_at=now(),
                )
            )
        memberships = [
            dict(event_id=r["id"], **m)
            for r, vector in zip(records, vectors)
            for m in memberships_for(vector, themes, self.settings.membership_threshold)
        ]
        ai = {result["event_id"]: result for result in self.db.all("ai_results")}
        trends = []
        for theme in themes:
            member_ids = {m["event_id"] for m in memberships if m["theme_id"] == theme["theme_id"]}
            members = [r for r in records if r["id"] in member_ids]
            dates = [date.fromisoformat(r["event_date"]) for r in members]
            theme.update(
                event_count=len(members),
                first_seen=min(dates).isoformat() if dates else None,
                last_seen=max(dates).isoformat() if dates else None,
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
                location_distribution=dict(
                    Counter(r["location_name"] or "Unspecified" for r in members)
                ),
                facility_distribution=dict(
                    Counter(r["facility"] or "Unspecified" for r in members)
                ),
            )
            trends.append(weekly_trend(theme["theme_id"], dates, self.settings, as_of))
        self.db.replace_patterns(themes, memberships, trends)
        return themes

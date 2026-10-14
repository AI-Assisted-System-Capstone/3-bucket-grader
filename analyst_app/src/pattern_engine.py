from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ALIASES = {
    "event_id": ["Event No.", "event_no", "event id", "event_id"],
    "event_date": ["Event Date", "event_date", "date"],
    "date_received": ["Date Received", "date_received"],
    "location": ["Location Name", "Location  Name", "location_name", "location"],
    "service": ["Encounter Service", "encounter_service", "service"],
    "event_type": ["Event Type", "event_type"],
    "age": ["Age at Encounter", "age_at_encounter", "age"],
    "prescribed_med": ["ME - Prescribed - Name *... Name", "ME - Prescribed - Name", "prescribed_med"],
    "prescribed_dose": ["ME - Prescribed - Dose *", "prescribed_dose"],
    "admin_med": ["ME - Admin - Name", "admin_med"],
    "admin_dose": ["ME - Admin - Dose", "admin_dose"],
    "adr_med": ["ADR - Suspect Med Name", "adr_med"],
    "adr_dose": ["ADR - Dose *", "adr_dose"],
    "event_comments": ["event_comments", "Event Comments", "event comments"],
    "manager_comments": ["manager_comments", "MANAGER COMMENTS", "manager comments"],
    "unit_actions": ["unit_actions_taken", "UNIT_ACTIONS_TAKEN", "unit actions taken"],
    "shareable_lessons": ["shareable_lessons", "SHAREABLE_LESSONS", "shareable lessons"],
    "harm_score": ["Significance (PSRS Harm score)", "harm_score"],
    "hpi_designation": ["HPI Designation... Name", "hpi_designation"],
    "investigation_level": ["Level of Invet", "level_of_investigation", "investigation_level"],
    "analyst_report_type": ["Analyst-Report Type*", "analyst_report_type"],
}

# Default embedding text uses only information expected to be available when a new report arrives.
# Post-review fields are retained in the normalized data but excluded by default.
DEFAULT_TEXT_FIELDS = [
    "event_comments",
    "event_type",
    "prescribed_med",
    "prescribed_dose",
    "admin_med",
    "admin_dose",
    "adr_med",
    "adr_dose",
]
REVIEW_FIELDS = ["manager_comments", "unit_actions", "shareable_lessons"]
TARGET_OR_REVIEW_LABELS = ["harm_score", "hpi_designation", "investigation_level", "analyst_report_type"]

GENERIC_TERMS = {
    "patient", "patients", "event", "events", "report", "reports", "reported", "unit",
    "staff", "review", "reviewed", "noted", "occurred", "hospital", "care", "clinical",
    "no", "routine", "time", "following", "issue", "team", "required", "identified",
}


def clean_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\\_", "_").replace("\\*", "*")).strip()


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {clean_col_name(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = clean_col_name(cand).lower()
        if key in normalized:
            return normalized[key]
    return None


def load_markdown_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found in {path}")

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = split_row(table_lines[0])
    rows: list[list[str]] = []
    for ln in table_lines[2:]:
        vals = split_row(ln)
        if len(vals) < len(header):
            vals += [""] * (len(header) - len(vals))
        elif len(vals) > len(header):
            vals = vals[: len(header) - 1] + [" | ".join(vals[len(header) - 1 :])]
        rows.append(vals)
    return pd.DataFrame(rows, columns=[clean_col_name(x) for x in header])


def load_reports(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".md", ".markdown"}:
        df = load_markdown_table(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=sheet_name or 0)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(p, lines=True)
    elif suffix == ".json":
        df = pd.read_json(p)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    df.columns = [clean_col_name(c) for c in df.columns]
    return df


def normalize_schema(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for canonical, candidates in ALIASES.items():
        col = _find_column(raw, candidates)
        out[canonical] = raw[col] if col is not None else ""

    for c in out.columns:
        if c in {"event_date", "date_received"}:
            continue
        out[c] = out[c].fillna("").astype(str).map(lambda x: re.sub(r"\s+", " ", x).strip())

    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["date_received"] = pd.to_datetime(out["date_received"], errors="coerce")
    out["event_id"] = out["event_id"].replace("", np.nan)
    fallback = pd.Series([f"ROW-{i:06d}" for i in range(len(out))], index=out.index)
    out["event_id"] = out["event_id"].fillna(fallback)
    return out


def build_report_text(df: pd.DataFrame, include_review_fields: bool = False) -> pd.Series:
    """Build semantic text without leaking PSRS/HPI/investigation labels.

    Location/service/age are intentionally not embedded by default. They remain metadata
    for filtering/faceting, preventing same-unit reports from looking similar merely because
    they came from the same place.
    """

    def one(row: pd.Series) -> str:
        parts: list[str] = []
        narrative = str(row.get("event_comments", "") or "").strip()
        if narrative:
            parts.append(narrative)

        event_type = str(row.get("event_type", "") or "").strip()
        if event_type:
            parts.append(f"Event type: {event_type}")

        med_bits = []
        for label, field in [
            ("prescribed medication", "prescribed_med"),
            ("prescribed dose", "prescribed_dose"),
            ("administered medication", "admin_med"),
            ("administered dose", "admin_dose"),
            ("suspected adverse-reaction medication", "adr_med"),
            ("adverse-reaction dose", "adr_dose"),
        ]:
            value = str(row.get(field, "") or "").strip()
            if value:
                med_bits.append(f"{label}: {value}")
        if med_bits:
            parts.append("; ".join(med_bits))

        if include_review_fields:
            for label, field in [
                ("Manager comments", "manager_comments"),
                ("Actions taken", "unit_actions"),
                ("Shareable lessons", "shareable_lessons"),
            ]:
                value = str(row.get(field, "") or "").strip()
                if value:
                    parts.append(f"{label}: {value}")
        return "\n".join(parts).strip()

    return df.apply(one, axis=1)


class LocalTextEmbedder:
    """Local semantic embedder with an offline TF-IDF fallback.

    No hosted model API is used. SentenceTransformer is loaded from local cache
    unless allow_model_download=True is explicitly requested.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        force_tfidf: bool = False,
        allow_model_download: bool = False,
        max_tfidf_features: int = 40000,
    ):
        self.model_name = model_name
        self.force_tfidf = force_tfidf
        self.allow_model_download = allow_model_download
        self.max_tfidf_features = max_tfidf_features
        self.backend: Optional[str] = None
        self.model = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.fallback_reason: Optional[str] = None

    def fit_transform(self, texts: list[str]):
        if not self.force_tfidf:
            try:
                if not self.allow_model_download:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer

                kwargs = {}
                if not self.allow_model_download:
                    kwargs["local_files_only"] = True
                self.model = SentenceTransformer(self.model_name, **kwargs)
                X = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 1000,
                    batch_size=64,
                )
                self.backend = "sentence-transformers"
                return np.asarray(X, dtype=np.float32)
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            max_features=self.max_tfidf_features,
        )
        X = self.vectorizer.fit_transform(texts)
        self.backend = "tfidf"
        return normalize(X, norm="l2", copy=False)

    def transform(self, texts: list[str]):
        if self.backend == "sentence-transformers":
            X = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            return np.asarray(X, dtype=np.float32)
        if self.backend == "tfidf" and self.vectorizer is not None:
            return normalize(self.vectorizer.transform(texts), norm="l2", copy=False)
        raise RuntimeError("Embedder has not been fit")


@dataclass
class PatternArtifacts:
    df: pd.DataFrame
    embeddings: object
    cluster_labels: np.ndarray
    cluster_names: dict[int, str]
    clusterer: object
    cluster_method: str
    novelty_threshold: float


class PatternEngine:
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        model_name: str = DEFAULT_MODEL,
        force_tfidf: bool = False,
        allow_model_download: bool = False,
        include_review_fields: bool = False,
        cluster_method: str = "auto",
        retrieval_top_k: int = 5,
    ):
        self.requested_clusters = n_clusters
        self.embedder = LocalTextEmbedder(
            model_name=model_name,
            force_tfidf=force_tfidf,
            allow_model_download=allow_model_download,
        )
        self.include_review_fields = include_review_fields
        self.cluster_method_requested = cluster_method
        if int(retrieval_top_k) < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        self.retrieval_top_k = int(retrieval_top_k)
        self.artifacts: Optional[PatternArtifacts] = None
        self.nn: Optional[NearestNeighbors] = None
        self._neighbor_distances: Optional[np.ndarray] = None
        self._neighbor_indices: Optional[np.ndarray] = None

    @staticmethod
    def _matrix_len(X) -> int:
        return int(X.shape[0])

    @staticmethod
    def _candidate_k(n: int) -> list[int]:
        if n < 6:
            return [2] if n >= 3 else [1]
        upper = min(14, max(3, int(round(math.sqrt(n)) + 2)), n - 1)
        vals = sorted(set([2, 3, 4, 5, 6, 8, 10, 12, upper]))
        return [k for k in vals if 2 <= k <= upper]

    def _choose_k_by_silhouette(self, X) -> int:
        n = self._matrix_len(X)
        if self.requested_clusters is not None:
            return max(2, min(int(self.requested_clusters), n - 1))
        if n < 3:
            return 1

        sample_size = min(2500, n)
        best_k, best_score = 2, -np.inf
        for k in self._candidate_k(n):
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=10,
                batch_size=min(2048, max(64, n)),
            )
            labels = model.fit_predict(X)
            if len(set(labels.tolist())) < 2:
                continue
            try:
                score = silhouette_score(
                    X,
                    labels,
                    metric="cosine",
                    sample_size=sample_size if n > sample_size else None,
                    random_state=SEED,
                )
            except Exception:
                score = -np.inf
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def _fit_clusters(self, X) -> tuple[np.ndarray, object, str]:
        n = self._matrix_len(X)
        if n < 3:
            model = MiniBatchKMeans(n_clusters=1, random_state=SEED, n_init=10).fit(X)
            return np.zeros(n, dtype=int), model, "minibatch-kmeans"

        # HDBSCAN is preferred when available for medium/large dense semantic-embedding sets
        # because it can leave true outliers as noise rather than forcing every event into a cluster.
        wants_hdbscan = self.cluster_method_requested in {"auto", "hdbscan"}
        if wants_hdbscan and not sparse.issparse(X) and n >= 50:
            try:
                import hdbscan

                n_comp = min(50, X.shape[1], max(2, n - 1))
                Z = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)
                min_cluster_size = max(8, int(round(math.sqrt(n))))
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(3, min_cluster_size // 3),
                    metric="euclidean",
                    prediction_data=True,
                )
                labels = model.fit_predict(Z)
                # If HDBSCAN collapses almost everything into noise, use KMeans fallback.
                non_noise = set(labels.tolist()) - {-1}
                if non_noise and np.mean(labels == -1) < 0.85:
                    model._midas_reduced_X = Z
                    return labels.astype(int), model, "hdbscan"
            except Exception:
                if self.cluster_method_requested == "hdbscan":
                    raise

        k = self._choose_k_by_silhouette(X)
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=20,
            batch_size=min(2048, max(64, n)),
        )
        labels = model.fit_predict(X)
        return labels.astype(int), model, "minibatch-kmeans"

    def _name_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
        names: dict[int, str] = {}
        for cluster in sorted(set(labels.tolist())):
            if cluster == -1:
                names[cluster] = "Noise / unassigned"
                continue
            texts = df.loc[labels == cluster, "report_text"].astype(str).tolist()
            if not texts:
                names[cluster] = f"Cluster {cluster}"
                continue
            joined = " ".join(texts)
            try:
                vec = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=3000,
                    sublinear_tf=True,
                )
                # Use the cluster corpus as one document plus a small pseudo-background constructed
                # from all non-cluster reports, producing more discriminative cluster descriptors.
                background = " ".join(df.loc[labels != cluster, "report_text"].astype(str).tolist()) or "background"
                M = vec.fit_transform([joined, background])
                scores = np.asarray(M[0].todense()).ravel()
                terms = np.asarray(vec.get_feature_names_out())
                order = scores.argsort()[::-1]
                selected = []
                for idx in order:
                    term = terms[idx]
                    toks = set(term.lower().split())
                    if toks and toks.issubset(GENERIC_TERMS):
                        continue
                    if any(tok in GENERIC_TERMS for tok in toks) and len(toks) == 1:
                        continue
                    selected.append(term)
                    if len(selected) == 4:
                        break
                names[cluster] = " / ".join(selected) if selected else f"Cluster {cluster}"
            except ValueError:
                names[cluster] = f"Cluster {cluster}"
        return names

    def fit(self, df: pd.DataFrame) -> PatternArtifacts:
        work = df.copy()
        work["report_text"] = build_report_text(work, include_review_fields=self.include_review_fields)
        work = work[work["report_text"].str.len() > 20].reset_index(drop=True)
        if len(work) < 2:
            raise ValueError("Need at least 2 non-empty reports for pattern analysis")

        X = self.embedder.fit_transform(work["report_text"].tolist())

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.nn.fit(X)
        # Compute enough neighbors once for novelty and the standard top-five export.
        # Exact brute-force cosine retrieval is the expensive step on large datasets;
        # caching avoids repeating the same all-report search during save/metrics.
        n_neighbors = min(6, len(work))
        distances, indices = self.nn.kneighbors(X, n_neighbors=n_neighbors)
        self._neighbor_distances = distances
        self._neighbor_indices = indices
        nn_distance = np.empty(len(work), dtype=float)
        for i in range(len(work)):
            non_self = distances[i][indices[i] != i]
            nn_distance[i] = float(non_self[0]) if len(non_self) else 1.0
        nn_similarity = 1.0 - nn_distance
        novelty = nn_distance
        novelty_threshold = float(np.quantile(novelty, 0.95)) if len(novelty) >= 10 else float(np.quantile(novelty, 0.90))

        labels, clusterer, cluster_method = self._fit_clusters(X)
        cluster_names = self._name_clusters(work, labels)

        work["cluster"] = labels
        work["cluster_name"] = [cluster_names[int(c)] for c in labels]
        work["nearest_neighbor_similarity"] = nn_similarity
        work["novelty_score"] = novelty
        work["novelty_flag"] = novelty >= novelty_threshold

        self.artifacts = PatternArtifacts(
            df=work,
            embeddings=X,
            cluster_labels=labels,
            cluster_names=cluster_names,
            clusterer=clusterer,
            cluster_method=cluster_method,
            novelty_threshold=novelty_threshold,
        )
        return self.artifacts

    def _cluster_query(self, q) -> tuple[int, str]:
        assert self.artifacts is not None
        method = self.artifacts.cluster_method
        model = self.artifacts.clusterer
        if method == "hdbscan":
            try:
                import hdbscan
                # HDBSCAN needs the same PCA space used during fit. For the prototype we return the
                # cluster of the nearest historical report rather than refitting a PCA transform.
                d, idx = self.nn.kneighbors(q, n_neighbors=1)
                label = int(self.artifacts.cluster_labels[int(idx[0, 0])])
                return label, self.artifacts.cluster_names.get(label, f"Cluster {label}")
            except Exception:
                pass
        label = int(model.predict(q)[0])
        return label, self.artifacts.cluster_names.get(label, f"Cluster {label}")

    def search(
        self,
        report_text: str,
        top_k: int = 5,
        as_of_date: Optional[str | pd.Timestamp] = None,
        lookback_days: Optional[int] = None,
        location: Optional[str] = None,
        service: Optional[str] = None,
    ) -> dict:
        if self.artifacts is None or self.nn is None:
            raise RuntimeError("Call fit first")
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1")
        top_k = int(top_k)
        q = self.embedder.transform([report_text])
        df = self.artifacts.df

        mask = np.ones(len(df), dtype=bool)
        if as_of_date is not None:
            as_of = pd.Timestamp(as_of_date)
            mask &= df["event_date"].notna().to_numpy()
            mask &= (df["event_date"] <= as_of).to_numpy()
            if lookback_days is not None:
                lower = as_of - pd.Timedelta(days=int(lookback_days))
                mask &= (df["event_date"] >= lower).to_numpy()
        if location:
            mask &= df["location"].fillna("").str.casefold().eq(location.casefold()).to_numpy()
        if service:
            mask &= df["service"].fillna("").str.casefold().eq(service.casefold()).to_numpy()

        candidate_idx = np.flatnonzero(mask)
        if len(candidate_idx) == 0:
            raise ValueError("No historical reports match the requested filters/lookback")

        if len(candidate_idx) == len(df):
            k = min(top_k, len(df))
            distances, indices = self.nn.kneighbors(q, n_neighbors=k)
            order = indices[0]
            similarities = 1.0 - distances[0]
        else:
            Xc = self.artifacts.embeddings[candidate_idx]
            sims = cosine_similarity(q, Xc)[0]
            local_order = np.argsort(sims)[::-1][: min(top_k, len(candidate_idx))]
            order = candidate_idx[local_order]
            similarities = sims[local_order]

        cluster, cluster_name = self._cluster_query(q)
        nearest = float(similarities[0]) if len(similarities) else 0.0
        novelty = 1.0 - nearest

        matches = []
        cols = ["event_id", "event_date", "event_type", "location", "service", "cluster", "cluster_name"]
        for idx, sim in zip(order, similarities):
            row = df.iloc[int(idx)]
            rec = {c: (None if pd.isna(row.get(c)) else row.get(c)) for c in cols}
            # Pandas returns NumPy scalar types for numeric columns; convert those so
            # callers can serialize the search result directly as JSON.
            rec = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in rec.items()
            }
            if isinstance(rec.get("event_date"), pd.Timestamp):
                rec["event_date"] = rec["event_date"].date().isoformat()
            rec["similarity"] = float(sim)
            rec["snippet"] = row["report_text"][:320].replace("\n", " ")
            matches.append(rec)

        return {
            "cluster": cluster,
            "cluster_name": cluster_name,
            "nearest_similarity": nearest,
            "novelty_score": float(novelty),
            "novelty_threshold": float(self.artifacts.novelty_threshold),
            "novelty_flag": bool(novelty >= self.artifacts.novelty_threshold),
            "matches": matches,
        }

    def retrieval_neighbors(self, top_k: int = 5) -> pd.DataFrame:
        if self.artifacts is None or self.nn is None:
            raise RuntimeError("Call fit first")
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1")
        top_k = int(top_k)
        n = len(self.artifacts.df)
        k = min(top_k + 1, n)
        if (
            self._neighbor_distances is not None
            and self._neighbor_indices is not None
            and self._neighbor_distances.shape[1] >= k
        ):
            distances = self._neighbor_distances[:, :k]
            indices = self._neighbor_indices[:, :k]
        else:
            distances, indices = self.nn.kneighbors(self.artifacts.embeddings, n_neighbors=k)
        rows = []
        for i in range(n):
            rank = 0
            for dist, j in zip(distances[i], indices[i]):
                if int(j) == i:
                    continue
                rank += 1
                if rank > top_k:
                    break
                src = self.artifacts.df.iloc[i]
                dst = self.artifacts.df.iloc[int(j)]
                rows.append({
                    "event_id": src["event_id"],
                    "neighbor_event_id": dst["event_id"],
                    "rank": rank,
                    "similarity": float(1.0 - dist),
                    "same_event_type": bool(src["event_type"] and src["event_type"] == dst["event_type"]),
                    "same_service": bool(src["service"] and src["service"] == dst["service"]),
                    "source_event_type": src["event_type"],
                    "neighbor_event_type": dst["event_type"],
                    "source_service": src["service"],
                    "neighbor_service": dst["service"],
                })
        return pd.DataFrame(rows)

    def trend_table(self, freq: str = "W-SUN", baseline_periods: int = 8) -> pd.DataFrame:
        if self.artifacts is None:
            raise RuntimeError("Call fit first")
        df = self.artifacts.df[self.artifacts.df["event_date"].notna()].copy()
        if df.empty:
            return pd.DataFrame()

        period = df["event_date"].dt.to_period(freq).dt.start_time
        df["period"] = period
        # Use the requested period frequency so this works for weekly and monthly
        # trend tables (and fills missing periods correctly for each).
        global_periods = pd.period_range(
            start=period.min(),
            end=period.max(),
            freq=freq,
        ).start_time
        rows = []

        for cluster, g in df.groupby("cluster"):
            cname = self.artifacts.cluster_names[int(cluster)]
            counts = g.groupby("period").size().reindex(global_periods, fill_value=0).astype(int)
            frame = pd.DataFrame({"period": global_periods, "count": counts.to_numpy()})
            frame["cluster"] = int(cluster)
            frame["cluster_name"] = cname
            shifted = frame["count"].shift(1)
            frame["baseline_mean"] = shifted.rolling(baseline_periods, min_periods=4).mean()
            frame["baseline_std"] = shifted.rolling(baseline_periods, min_periods=4).std(ddof=0)
            # Count data are often sparse; a Poisson-style 3-sigma upper control limit behaves
            # better than dividing by a near-zero sample standard deviation.
            frame["upper_control_limit"] = frame["baseline_mean"] + 3.0 * np.sqrt(frame["baseline_mean"].clip(lower=0.25))
            frame["relative_to_baseline"] = np.where(
                frame["baseline_mean"] > 0,
                frame["count"] / frame["baseline_mean"],
                np.nan,
            )
            frame["spike_flag"] = (
                frame["baseline_mean"].notna()
                & (frame["count"] > frame["upper_control_limit"])
                & (frame["count"] >= frame["baseline_mean"] + 2)
                & (frame["count"] >= 3)
            )
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    def sanity_metrics(self) -> dict:
        if self.artifacts is None:
            raise RuntimeError("Call fit first")
        X = self.artifacts.embeddings
        labels = self.artifacts.cluster_labels
        n = len(labels)
        metrics = {
            "n_reports": int(n),
            "embedding_backend": self.embedder.backend,
            "embedding_model": self.embedder.model_name if self.embedder.backend == "sentence-transformers" else None,
            "fallback_reason": self.embedder.fallback_reason,
            "cluster_method": self.artifacts.cluster_method,
            "n_clusters_including_noise": int(len(set(labels.tolist()))),
            "noise_fraction": float(np.mean(labels == -1)),
            "mean_nearest_neighbor_similarity": float(self.artifacts.df["nearest_neighbor_similarity"].mean()),
            "novelty_threshold": float(self.artifacts.novelty_threshold),
        }

        valid_labels = labels >= 0
        unique = set(labels[valid_labels].tolist())
        if len(unique) > 1 and valid_labels.sum() > len(unique):
            try:
                metrics["silhouette_cosine"] = float(
                    silhouette_score(
                        X[valid_labels],
                        labels[valid_labels],
                        metric="cosine",
                        sample_size=min(2500, int(valid_labels.sum())) if valid_labels.sum() > 2500 else None,
                        random_state=SEED,
                    )
                )
            except Exception:
                pass

        neighbors = self.retrieval_neighbors(top_k=self.retrieval_top_k)
        if not neighbors.empty:
            et = neighbors[neighbors["source_event_type"].astype(str).str.len() > 0]
            if not et.empty:
                metrics[f"weak_label_event_type_precision_at_{self.retrieval_top_k}"] = float(et["same_event_type"].mean())
            svc = neighbors[neighbors["source_service"].astype(str).str.len() > 0]
            if not svc.empty:
                metrics[f"weak_label_same_service_rate_at_{self.retrieval_top_k}"] = float(svc["same_service"].mean())
        metrics["retrieval_top_k"] = int(self.retrieval_top_k)
        return metrics

    def save(self, output_dir: str | Path) -> None:
        if self.artifacts is None:
            raise RuntimeError("Call fit first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.artifacts.df.to_csv(out / "pattern_reports.csv", index=False)
        dense_embedding_path = out / "embeddings.npy"
        sparse_embedding_path = out / "embeddings_sparse.npz"
        if sparse.issparse(self.artifacts.embeddings):
            sparse.save_npz(sparse_embedding_path, self.artifacts.embeddings)
            dense_embedding_path.unlink(missing_ok=True)
        else:
            np.save(dense_embedding_path, self.artifacts.embeddings)
            sparse_embedding_path.unlink(missing_ok=True)

        self.retrieval_neighbors(top_k=self.retrieval_top_k).to_csv(out / "retrieval_neighbors.csv", index=False)
        self.trend_table().to_csv(out / "weekly_cluster_trends.csv", index=False)
        self.trend_table(freq="M").to_csv(out / "monthly_cluster_trends.csv", index=False)
        (out / "pattern_metrics.json").write_text(json.dumps(self.sanity_metrics(), indent=2), encoding="utf-8")
        (out / "cluster_names.json").write_text(
            json.dumps({str(k): v for k, v in self.artifacts.cluster_names.items()}, indent=2),
            encoding="utf-8",
        )

        X = self.artifacts.embeddings
        n = X.shape[0]
        method = "TruncatedSVD" if sparse.issparse(X) else "PCA"
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                random_state=SEED,
                n_jobs=1,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
            )
            xy = reducer.fit_transform(X)
            method = "UMAP"
        except Exception:
            if sparse.issparse(X):
                n_comp = min(2, max(1, X.shape[1] - 1), max(1, n - 1))
                if n_comp >= 2:
                    xy = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(X)
                else:
                    one = TruncatedSVD(n_components=1, random_state=SEED).fit_transform(X).ravel()
                    xy = np.column_stack([one, np.zeros(n)])
            elif X.shape[1] >= 2 and n >= 2:
                xy = PCA(n_components=2, random_state=SEED).fit_transform(X)
            else:
                xy = np.column_stack([np.asarray(X)[:, 0], np.zeros(n)])

        viz = self.artifacts.df[["event_id", "cluster", "cluster_name"]].copy()
        viz["x"] = xy[:, 0]
        viz["y"] = xy[:, 1]
        viz["projection_method"] = method
        viz.to_csv(out / "embedding_projection.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 7))
            for c in sorted(set(self.artifacts.cluster_labels.tolist())):
                mask = self.artifacts.cluster_labels == c
                label = self.artifacts.cluster_names[int(c)]
                plt.scatter(xy[mask, 0], xy[mask, 1], label=f"{c}: {label}", alpha=0.75)
            plt.xlabel(f"{method} 1")
            plt.ylabel(f"{method} 2")
            plt.title(f"MIDAS report embedding projection ({method}; visualization only)")
            plt.legend(fontsize=7, loc="best")
            plt.tight_layout()
            plt.savefig(out / "embedding_projection.png", dpi=160)
            plt.close()
        except Exception:
            pass


def print_demo(result: dict) -> None:
    print("\n=== Pattern result ===")
    print(f"Cluster/theme proxy: {result['cluster']} — {result['cluster_name']}")
    print(f"Nearest similarity:  {result['nearest_similarity']:.3f}")
    print(f"Novelty score:       {result['novelty_score']:.3f}")
    print(f"Novelty threshold:   {result['novelty_threshold']:.3f}")
    print(f"Novelty flag:        {result['novelty_flag']}")
    print("\nTop similar historical reports:")
    for i, m in enumerate(result["matches"], 1):
        print(f"{i}. {m['event_id']}  sim={m['similarity']:.3f}  cluster={m['cluster']}")
        print(f"   {m['event_type']} | {m['location']} | {m['service']}")
        print(f"   {m['snippet']}")

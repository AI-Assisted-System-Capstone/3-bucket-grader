import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# analyst_app/ lives inside the severity repo, so its model is the default.
SEVERITY_REPO = Path(__file__).resolve().parents[2]
# The event-type model lives in the separate 11-buckets repo; use it when it is
# checked out next to this one (MIDAS_EVENT_TYPE_REPO overrides).
_EVENT_TYPE_REPO = SEVERITY_REPO.parent / "11-buckets"
EVENT_TYPE_REPO = str(_EVENT_TYPE_REPO) if _EVENT_TYPE_REPO.is_dir() else ""
EVENT_TYPE_ARTIFACTS = str(_EVENT_TYPE_REPO / "outputs" / "baseline") if EVENT_TYPE_REPO else ""


class Settings(BaseModel):
    database: str = "local_state/midas.sqlite3"
    fixture_mode: bool = False
    embedding_backend: Literal["minilm", "lexical"] = "minilm"
    severity_repo: str = str(SEVERITY_REPO)
    severity_artifacts: str = str(SEVERITY_REPO / "severity_model")
    event_type_repo: str = EVENT_TYPE_REPO
    event_type_artifacts: str = EVENT_TYPE_ARTIFACTS
    top_n: int = Field(default=5, ge=1, le=10000)
    # Similar reports come from the year before the report (sponsor: "last year").
    similar_lookback_days: int = Field(default=365, ge=1)
    # A report is flagged novel when its closest report from the past year is
    # less similar than this share of reports' closest matches. The cut-off is
    # recomputed on each pattern rebuild, so it adapts to how much history exists.
    novelty_quantile: float = Field(default=0.05, gt=0, lt=1)
    # Below this many earlier reports there is too little history to judge novelty.
    novelty_min_history: int = Field(default=200, ge=1)
    membership_threshold: float = Field(default=0.75, ge=0, le=1)
    cluster_distance: float = Field(default=0.50, gt=0, le=2)
    minimum_cluster_size: int = Field(default=3, ge=2)
    cluster_method: Literal["hdbscan", "agglomerative"] = "agglomerative"
    minimum_event_count: int = Field(default=3, ge=1)
    minimum_percent_increase: float = Field(default=50, ge=0)
    z_score_threshold: float = Field(default=2, ge=0)
    baseline_weeks: int = Field(default=8, ge=2)
    max_pattern_events: int = Field(default=10000, ge=10)

    @classmethod
    def from_env(cls):
        # Pydantic validates strings into the configured numeric/boolean types.
        return cls(
            **{
                key: os.environ["MIDAS_" + key.upper()]
                for key in cls.model_fields
                if "MIDAS_" + key.upper() in os.environ
            }
        )

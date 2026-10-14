import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# analyst_app/ lives inside the severity repo, so its model is the default.
SEVERITY_REPO = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    database: str = "local_state/midas.sqlite3"
    fixture_mode: bool = False
    embedding_backend: Literal["minilm", "lexical"] = "minilm"
    severity_repo: str = str(SEVERITY_REPO)
    severity_artifacts: str = str(SEVERITY_REPO / "severity_model")
    event_type_repo: str = ""
    event_type_artifacts: str = ""
    top_n: int = Field(default=5, ge=1, le=10000)
    membership_threshold: float = Field(default=0.80, ge=0, le=1)
    cluster_distance: float = Field(default=0.40, gt=0, le=2)
    minimum_cluster_size: int = Field(default=3, ge=2)
    cluster_method: Literal["hdbscan", "agglomerative"] = "hdbscan"
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

from datetime import date, datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def now():
    return datetime.now(timezone.utc).isoformat()


GRADES = ("A", "B1", "B2", "C", "D", "E", "F", "G", "H", "I")
CODES = ("ADR", "C", "E", "EQ", "FALL", "I", "ME", "O", "SH", "SI", "T")
LABELS = dict(
    zip(
        CODES,
        (
            "Adverse drug reaction",
            "Care complication",
            "General / laboratory / documentation",
            "Equipment / product",
            "Fall",
            "Security / violence / administration",
            "Medication error",
            "Outcome",
            "Self-harm",
            "Skin integrity",
            "Transfusion / blood product",
        ),
    )
)


class EventInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    event_id: str = Field(min_length=1, max_length=100)
    event_comments: str = Field(min_length=10, max_length=30000)
    location_name: str = Field(default="", max_length=200)
    encounter_service: str = Field(default="", max_length=200)
    age_at_encounter: Optional[float] = Field(default=None, ge=0, le=130)
    prescribed_medication: str = Field(default="", max_length=200)
    prescribed_dose: str = Field(default="", max_length=100)
    administered_medication: str = Field(default="", max_length=200)
    administered_dose: str = Field(default="", max_length=100)
    suspect_medication: str = Field(default="", max_length=200)
    suspect_dose: str = Field(default="", max_length=100)
    event_date: date
    facility: str = Field(default="", max_length=200)


class SeverityResult(BaseModel):
    grade: Literal["A", "B1", "B2", "C", "D", "E", "F", "G", "H", "I"]
    grade_probabilities: Dict[str, float]
    bucket: Literal["none", "some", "serious"]
    bucket_probabilities: Dict[str, float]
    triage_score_raw: float = Field(ge=0, le=1)
    harm_probability_calibrated: float = Field(ge=0, le=1)
    priority_flag: bool
    model_version: str
    fixture: bool = False


class EventTypeResult(BaseModel):
    event_type_code: str
    event_type_label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float]
    model_version: str
    fixture: bool = False


class AIResult(BaseModel):
    event_id: str  # opaque internal UUID, never the source MIDAS identifier
    severity: Optional[SeverityResult] = None
    event_type: Optional[EventTypeResult] = None
    themes: List[dict] = Field(default_factory=list)
    similar_events: List[dict] = Field(default_factory=list)
    trend_context: List[dict] = Field(default_factory=list)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    status: Literal["pending", "complete", "partial", "failed"] = "pending"
    created_at: str = Field(default_factory=now)


class AnalyzeRequest(BaseModel):
    event_ids: List[str] = Field(min_length=1, max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=30000)
    top_n: int = Field(default=5, ge=1, le=10000)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal["A", "B1", "B2", "C", "D", "E", "F", "G", "H", "I"]
    event_type_code: Literal["ADR", "C", "E", "EQ", "FALL", "I", "ME", "O", "SH", "SI", "T"]
    note: str = Field(default="", max_length=2000)


class EventRecord(EventInput):
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    id: str
    created_at: str
    ai: Optional[AIResult] = None
    analyst_decisions: List[dict] = Field(default_factory=list)
    analysis_status: Optional[str] = None


class EventList(BaseModel):
    total: int
    items: List[EventRecord]


class SimilarResult(BaseModel):
    event_id: str
    source_event_id: Optional[str] = None
    similarity: float = Field(ge=-1, le=1)
    location_name: str
    event_date: date


class TrendResult(BaseModel):
    theme_id: str
    current_count: int
    baseline_count: float
    percent_change: Optional[float]
    new_from_zero: bool
    trend_score: float
    status: Literal["stable", "watch", "emerging"]
    period_start: date
    period_end: date
    as_of: date
    timeline: List[dict]
    created_at: str
    explanation: str


class ThemeResult(BaseModel):
    theme_id: str
    name: str
    keywords: List[str]
    event_count: int
    first_seen: Optional[date]
    last_seen: Optional[date]
    representative_event_ids: List[str]
    event_type_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    location_distribution: Dict[str, int]
    facility_distribution: Dict[str, int]
    cluster_method: str
    model_version: str
    created_at: str
    trend: Optional[TrendResult] = None


class DashboardPattern(ThemeResult):
    trend_stale: bool


class DashboardSignal(BaseModel):
    priority_flag: bool
    fixture: bool


class DashboardAnalysis(BaseModel):
    status: Literal["pending", "complete", "partial", "failed"]
    severity: Optional[DashboardSignal] = None


class DashboardReport(BaseModel):
    # Dashboard needs navigation and status, not narratives, notes or medications.
    id: str
    event_id: str
    event_date: date
    location_name: str
    review_status: Literal["unreviewed", "reviewed"]
    ai: Optional[DashboardAnalysis] = None


class DashboardResult(BaseModel):
    generated_at: str
    fixture_mode: bool
    event_count: int = Field(ge=0)
    reviewed_count: int = Field(ge=0)
    unreviewed_count: int = Field(ge=0)
    high_priority_count: int = Field(ge=0)
    priority_unreviewed_count: int = Field(ge=0)
    analysis_issue_count: int = Field(ge=0)
    pending_analysis_count: int = Field(ge=0)
    emerging_pattern_count: int = Field(ge=0)
    review_queue: List[DashboardReport]
    recent_events: List[DashboardReport]
    patterns: List[DashboardPattern]


FIELD_MAP = {
    "event_comments": "event_comments",
    "location_name": "Location Name",
    "encounter_service": "Encounter Service",
    "age_at_encounter": "Age at Encounter",
    "prescribed_medication": "ME - Prescribed - Name *... Name",
    "prescribed_dose": "ME - Prescribed - Dose *",
    "administered_medication": "ME - Admin - Name",
    "administered_dose": "ME - Admin - Dose",
    "suspect_medication": "ADR - Suspect Med Name",
    "suspect_dose": "ADR - Dose *",
}


def model_frame(events):
    import pandas as pd

    return pd.DataFrame(
        [
            {raw: getattr(event, normalized) for normalized, raw in FIELD_MAP.items()}
            for event in events
        ]
    )


def normalized_text(event):
    """Same submission-time prefix/order as both upstream builders; no labels or review fields."""
    groups = [
        ("Unit", [event.location_name]),
        ("Service", [event.encounter_service]),
        ("Age", [event.age_at_encounter]),
        ("Prescribed", [event.prescribed_medication, event.prescribed_dose]),
        ("Administered", [event.administered_medication, event.administered_dose]),
        ("ADR suspect medication", [event.suspect_medication, event.suspect_dose]),
    ]
    parts = []
    for label, values in groups:
        values = [
            str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()
            for v in values
            if v is not None and str(v).strip()
        ]
        if values:
            parts.append(label + ": " + " ".join(values) + ".")
    return (" ".join(parts) + "\n" + event.event_comments).lstrip()

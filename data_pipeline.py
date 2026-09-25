"""
Leakage-safe input pipeline for the 3-bucket severity grader, aligned with the
teammate's 11-buckets pipeline (see AI-Assisted-System-Capstone/11-buckets,
src/pipeline.py) so both models are trained/evaluated on the same rows,
the same features, and the same time-based split.

Two things changed from the original baseline_grader.py / finetune_distilbert.py:

1. Dropped `manager_comments` and `unit_actions_taken` as inputs. Both are
   filled in *after* a safety officer investigates the report, so a model
   reading them is scoring hindsight, not predicting from what's available
   at submission time. This is exactly the leakage the teammate's data audit
   flagged and hard-blocks in their pipeline. It also explains why this
   project's earlier numbers (97% acc, 0.82 serious recall) looked stronger
   than the 11-buckets baseline (PR-AUC 0.47) -- different, easier task.

2. Switched from a random stratified split to the dataset's real time-based
   split (Jan-Aug train / Sep validation / Oct test), which is encoded in
   each report's `Event No.` (e.g. SYNPROD-TRAIN-0000001). This was already
   an open To-Do here ("Time-based validation") and is required to compare
   directly against the teammate's event-type/hurt model on the same test
   rows.

Model input is now: a short intake-field prefix (unit, service, age,
prescribed/administered/suspect medication) + `event_comments`, same as the
teammate's `build_input_text`. `manager_comments`, `unit_actions_taken`, and
`shareable_lessons` are never read from disk.
"""

import re

import numpy as np
import pandas as pd

DATA_PATH = "/Users/devanshigupta/Downloads/Capstone Engineering Files/LOCAL_ONLY_student_facing_candidate.parquet"

ID_COL = "Event No."
DATE_COL = "Event Date"
HARM_COL = "Significance (PSRS Harm score)"
EVENT_TYPE_COL = "Event Type"
TEXT_COL = "event_comments"

# The teammate's 11-bucket event-type task, added here as an auxiliary label
# (see combined_mtl.py) so a shared model can use event type as a clue for
# severity, same idea as their Head A -> Head B setup. Same 11 codes/order as
# their src/pipeline.py EVENT_BUCKETS.
EVENT_BUCKETS = ("ADR", "C", "E", "EQ", "FALL", "I", "ME", "O", "SH", "SI", "T")
EVENT_BUCKET2ID = {b: i for i, b in enumerate(EVENT_BUCKETS)}
N_EVENT_CLASSES = len(EVENT_BUCKETS)
UNLABELED = -1

# Post-investigation fields: must never reach the model. Same list as the
# teammate's src/pipeline.py EXCLUDED_COLUMNS (minus Event Type/harm score,
# which are label sources here, not blocked features).
EXCLUDED_COLUMNS = (
    "manager_comments",
    "unit_actions_taken",
    "shareable_lessons",
    "HPI Designation... Name",
    "Analyst-Report Type*",
    "Level of Invet",
)

# Intake fields available at submission time, used as a structured prefix
# ahead of the narrative. Same fields/order as the teammate's PREFIX_FIELDS.
PREFIX_FIELDS = (
    ("Unit", ("Location Name",)),
    ("Service", ("Encounter Service",)),
    ("Age", ("Age at Encounter",)),
    ("Prescribed", ("ME - Prescribed - Name *... Name", "ME - Prescribed - Dose *")),
    ("Administered", ("ME - Admin - Name", "ME - Admin - Dose")),
    ("ADR suspect medication", ("ADR - Suspect Med Name", "ADR - Dose *")),
)
FEATURE_COLUMNS = (TEXT_COL,) + tuple(c for _, cols in PREFIX_FIELDS for c in cols)

BUCKET_MAP = {
    "A-Unsafe Condition": "none",
    # B1 before B2: matches the sponsor's own code (B1=2, B2=3), not a guess.
    "B1-Near Miss (by chance)": "none",
    "B2-Near Miss (avoided)": "none",
    "C-No Harm (no monitoring)": "none",
    "D-No Harm (intervened)": "none",
    "E-Harm-Temp (treated/intervened)": "some",
    "F-Harm-Temp (add hospitalization)": "some",
    "G-Harm-Permanent": "serious",
    "H-Harm-Near Death": "serious",
    "I-Harm-Death": "serious",
}
LABELS = ["none", "some", "serious"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Full ordinal PSRS scale (NCC MERP-style, least to most severe). This
# synthetic dataset has no "Deleted" row, but the real data (in the sponsor's
# container) will have one -- DELETED_VALUES lists the raw strings to treat
# as "not a severity level" rather than crash on. See handle_deleted().
FULL_LABELS = list(BUCKET_MAP.keys())  # 10 grades, A .. I
FULL_LABEL2ID = {l: i for i, l in enumerate(FULL_LABELS)}
DELETED_VALUES = {"Deleted", "DELETED", "deleted"}

# Two more pre-fill targets from the sponsor's Approach 1 deck, alongside the
# harm score. Level of Invet is deliberately NOT modeled here: 78,475 / 80,000
# rows (98%) are the single value "Pt. Safety Department Review" -- a
# majority-class baseline already gets ~98%, so a model adds no real value.
REPORT_TYPE_COL = "Analyst-Report Type*"
REPORT_TYPES = ["Incident", "Serious Event", "Infrastructure", "Other"]
REPORT_TYPE2ID = {r: i for i, r in enumerate(REPORT_TYPES)}

HPI_COL = "HPI Designation... Name"
HPI_DESIGNATIONS = [
    "PSE 4 - No Harm", "PSE 3 - No Detectable Harm", "PSE 2 - Minimal Temporary Harm",
    "NME 1 - Unplanned Barrier Catch", "NME 2 - Last Strong Barrier Catch", "NME 3 - Early Barrier Catch",
    "SSE 4 - Severe Temporary Harm", "SSE 5 - Moderate Temporary Harm",
]
HPI2ID = {h: i for i, h in enumerate(HPI_DESIGNATIONS)}

# Subgroup metadata for fairness/error-rate breakdowns (evaluation only --
# never model input; Encounter Service and Location Name ARE also feature
# columns, so this reads the same values, not new leakage).
AGE_BANDS = [(0, 1, "infant <1"), (1, 18, "1-17"), (18, 41, "18-40"), (41, 65, "41-64"), (65, 200, "65+")]


def age_band(age):
    if pd.isna(age):
        return "unknown"
    for lo, hi, label in AGE_BANDS:
        if lo <= age < hi:
            return label
    return "unknown"


def handle_deleted(df, harm_col):
    """Real data (sponsor's container) can have a 'Deleted' harm score --
    not a severity level. Drop those rows from severity training/scoring,
    same as a missing score, but count them instead of silently vanishing."""
    is_deleted = df[harm_col].isin(DELETED_VALUES)
    n_deleted = int(is_deleted.sum())
    if n_deleted:
        print(f"[data_pipeline] Dropping {n_deleted} 'Deleted' rows from severity population "
              f"(not a severity level).")
    return df[~is_deleted].copy(), n_deleted


# Which of the 10 grade indices fall in each of the original 3 buckets, so a
# model trained on the full scale can also answer the original none/some/
# serious question by summing probabilities over these index groups -- same
# trick as the E..I triage score, just with 3 groups instead of 2.
BUCKET_GROUPS = {
    bucket: [FULL_LABEL2ID[g] for g in FULL_LABELS if BUCKET_MAP[g] == bucket]
    for bucket in LABELS
}
# E-I = "hurt", same cut point as the teammate's binary harm_label.
HURT_GRADES = {"E-Harm-Temp (treated/intervened)", "F-Harm-Temp (add hospitalization)",
               "G-Harm-Permanent", "H-Harm-Near Death", "I-Harm-Death"}


class DataContractError(RuntimeError):
    pass


class LeakageError(DataContractError):
    pass


def assert_no_excluded_columns(columns):
    leaked = sorted(set(columns) & set(EXCLUDED_COLUMNS))
    if leaked:
        raise LeakageError(f"Excluded (post-investigation) columns in model features: {leaked}")


def event_bucket(event_type):
    """Code before the first '-', uppercased. None if there is no '-'."""
    if not isinstance(event_type, str) or "-" not in event_type:
        return None
    return event_type.split("-", 1)[0].strip().upper()


def split_from_event_no(event_no):
    m = re.match(r"^SYNPROD-(TRAIN|VALIDATION|TEST)-\d+$", str(event_no))
    if not m:
        raise DataContractError(f"Event No. without a split prefix: {event_no!r}")
    return m.group(1)


def build_input_text(features):
    """'Unit: X. Service: Y. ...' prefix + narrative. No post-investigation fields."""
    assert_no_excluded_columns(features.columns)
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise DataContractError(f"Feature columns missing: {sorted(missing)}")

    def fmt(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()

    parts = []
    for label, cols in PREFIX_FIELDS:
        vals = features[list(cols)]
        joined = vals.apply(lambda r: " ".join(fmt(v) for v in r if pd.notna(v) and str(v).strip()), axis=1)
        parts.append(np.where(joined != "", label + ": " + joined + ". ", ""))
    prefix = pd.Series(["".join(p) for p in zip(*parts)], index=features.index)
    return (prefix.str.strip() + "\n" + features[TEXT_COL].fillna("").astype(str)).str.lstrip()


def load_dataset(path=DATA_PATH):
    """Load the parquet file and return one row per report with `split`,
    `bucket` (3-way severity label) and `text` (leakage-safe model input).
    Excluded columns are read only to assert they never leak into `text`.
    """
    df = pd.read_parquet(path)

    df["split"] = df[ID_COL].map(split_from_event_no)

    df, n_deleted = handle_deleted(df, HARM_COL)
    df = df[df[HARM_COL].notna()].copy()
    df["bucket"] = df[HARM_COL].map(BUCKET_MAP)
    unmapped = df.loc[df["bucket"].isna(), HARM_COL].unique()
    if len(unmapped):
        raise DataContractError(f"Harm scores outside BUCKET_MAP: {sorted(unmapped)}")

    df["text"] = build_input_text(df[list(FEATURE_COLUMNS)])
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["bucket"].map(LABEL2ID)

    # Full 10-grade ordinal scale (A..I) and the binary "hurt" cut (E-I), for
    # comparing a fine-grained model's marginalized triage score against a
    # model trained directly on the coarse target -- see combined_mtl.py's
    # sibling experiment, ordinal_vs_coarse.py.
    df["grade"] = df[HARM_COL]
    df["full_label"] = df["grade"].map(FULL_LABEL2ID)
    df["hurt_label"] = df["grade"].isin(HURT_GRADES).astype(int)

    # Auxiliary 11-bucket event-type label (UNLABELED for the ~64 reports with
    # no event code). Only computed within rows that already have a harm
    # score, since that's the population this project scores -- a slightly
    # smaller set than the teammate's full-dataset event-type task, but the
    # same 11 codes and the same rule.
    df["event_bucket"] = df[EVENT_TYPE_COL].map(event_bucket)
    unknown = sorted(set(df["event_bucket"].dropna()) - set(EVENT_BUCKETS))
    if unknown:
        raise DataContractError(f"Event-type codes outside the 11 buckets: {unknown}")
    df["event_type_label"] = df["event_bucket"].map(EVENT_BUCKET2ID).fillna(UNLABELED).astype(int)

    # Two more pre-fill targets (Approach 1 deck). UNLABELED where the field
    # is missing/None or an unrecognized value -- excluded from those two
    # tasks' training/scoring, not dropped from the dataset entirely.
    df["report_type_label"] = df[REPORT_TYPE_COL].map(REPORT_TYPE2ID).fillna(UNLABELED).astype(int)
    df["hpi_label"] = df[HPI_COL].map(HPI2ID).fillna(UNLABELED).astype(int)

    # Subgroup metadata for fairness/error breakdowns -- read-only, not model
    # input (Encounter Service and Location Name already are, via the prefix).
    df["encounter_service"] = df["Encounter Service"]
    df["location_name"] = df["Location Name"]
    df["age_band"] = df["Age at Encounter"].map(age_band)

    return df[
        ["event_no" if "event_no" in df.columns else ID_COL, "split", "bucket", "label",
         "grade", "full_label", "hurt_label",
         "event_bucket", "event_type_label",
         "report_type_label", "hpi_label",
         "encounter_service", "location_name", "age_band",
         "text"]
    ].rename(columns={ID_COL: "event_no"})


def train_val_test(path=DATA_PATH):
    df = load_dataset(path)
    return (
        df[df.split == "TRAIN"].reset_index(drop=True),
        df[df.split == "VALIDATION"].reset_index(drop=True),
        df[df.split == "TEST"].reset_index(drop=True),
    )


if __name__ == "__main__":
    train_df, val_df, test_df = train_val_test()
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name} bucket distribution:")
        print(split_df["bucket"].value_counts())
        print(split_df["bucket"].value_counts(normalize=True).round(3))

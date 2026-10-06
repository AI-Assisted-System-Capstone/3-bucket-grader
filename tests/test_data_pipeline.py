import numpy as np
import pytest

from data_pipeline import (
    BUCKET_GROUPS,
    BUCKET_MAP,
    EXCLUDED_COLUMNS,
    FULL_LABELS,
    HURT_GRADES,
    LABELS,
    DataContractError,
    LeakageError,
    assert_no_excluded_columns,
    build_input_text,
    split_from_event_no,
)
from conftest import make_reports


# --- Excluded (post-investigation) columns ---------------------------------

@pytest.mark.parametrize("col", EXCLUDED_COLUMNS)
def test_assert_rejects_each_excluded_column(col):
    with pytest.raises(LeakageError, match=col.replace("*", r"\*")):
        assert_no_excluded_columns(["event_comments", col])


def test_assert_allows_clean_columns():
    assert_no_excluded_columns(["event_comments", "Location Name"])


@pytest.mark.parametrize("col", ["manager_comments", "unit_actions_taken", "shareable_lessons"])
def test_build_input_text_refuses_leaked_column(col):
    df = make_reports(["fall"], **{col: ["patient died after the fall"]})
    with pytest.raises(LeakageError):
        build_input_text(df)


def test_excluded_list_covers_the_documented_leaks():
    for col in ("manager_comments", "unit_actions_taken", "shareable_lessons", "Analyst-Report Type*"):
        assert col in EXCLUDED_COLUMNS


# --- Event No. split decoding -----------------------------------------------

@pytest.mark.parametrize("event_no, split", [
    ("SYNPROD-TRAIN-0000001", "TRAIN"),
    ("SYNPROD-VALIDATION-0012345", "VALIDATION"),
    ("SYNPROD-TEST-9", "TEST"),
])
def test_split_decoding(event_no, split):
    assert split_from_event_no(event_no) == split


@pytest.mark.parametrize("event_no", [
    "SYNPROD-train-0000001",   # lowercase split
    "SYNPROD-DEV-0000001",     # unknown split
    "SYNPROD-TEST-",           # no number
    "SYNPROD-TEST-12a",        # trailing junk
    "XSYNPROD-TEST-1",         # wrong prefix
    "SYNPROD-TEST-1 ",         # trailing space
    "", None, float("nan"),
])
def test_split_decoding_rejects_malformed(event_no):
    with pytest.raises(DataContractError):
        split_from_event_no(event_no)


# --- Grade -> bucket mapping ------------------------------------------------

EXPECTED_BUCKET = {
    "A": "none", "B1": "none", "B2": "none", "C": "none", "D": "none",
    "E": "some", "F": "some",
    "G": "serious", "H": "serious", "I": "serious",
}


def test_grade_to_bucket_mapping():
    got = {g.split("-", 1)[0]: b for g, b in BUCKET_MAP.items()}
    assert got == EXPECTED_BUCKET


def test_full_labels_are_ordinal_a_to_i():
    assert [g.split("-", 1)[0] for g in FULL_LABELS] == list(EXPECTED_BUCKET)


def test_bucket_groups_partition_all_grades():
    ids = sorted(i for b in LABELS for i in BUCKET_GROUPS[b])
    assert ids == list(range(len(FULL_LABELS)))
    for bucket, group in BUCKET_GROUPS.items():
        assert all(BUCKET_MAP[FULL_LABELS[i]] == bucket for i in group)


def test_hurt_grades_are_some_plus_serious():
    hurt_from_buckets = {g for g, b in BUCKET_MAP.items() if b in ("some", "serious")}
    assert HURT_GRADES == hurt_from_buckets


def test_buckets_are_contiguous_in_severity_order():
    order = [LABELS.index(BUCKET_MAP[g]) for g in FULL_LABELS]
    assert order == sorted(order)

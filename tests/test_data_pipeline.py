"""Checks on data_pipeline.py's leakage guard, split decoding and label
mappings. None of these read the dataset."""

import pandas as pd
import pytest

from data_pipeline import (
    BUCKET_GROUPS,
    BUCKET_MAP,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    FULL_LABELS,
    HURT_GRADES,
    LABELS,
    DataContractError,
    LeakageError,
    assert_no_excluded_columns,
    build_input_text,
    split_from_event_no,
)


def one_report(**overrides):
    row = {c: None for c in FEATURE_COLUMNS}
    row["event_comments"] = "Patient found on floor, no injury noted."
    row.update(overrides)
    return pd.DataFrame([row])


# --- Excluded-column assertion ---

@pytest.mark.parametrize("column", EXCLUDED_COLUMNS)
def test_excluded_column_raises(column):
    with pytest.raises(LeakageError):
        assert_no_excluded_columns(list(FEATURE_COLUMNS) + [column])


@pytest.mark.parametrize("column", EXCLUDED_COLUMNS)
def test_build_input_text_rejects_excluded_column(column):
    with pytest.raises(LeakageError):
        build_input_text(one_report(**{column: "resolved after review"}))


def test_feature_columns_pass_the_assertion():
    assert not set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS)
    assert_no_excluded_columns(FEATURE_COLUMNS)


def test_build_input_text_requires_every_feature_column():
    with pytest.raises(DataContractError):
        build_input_text(one_report().drop(columns=["Location Name"]))


def test_build_input_text_prefix_and_narrative():
    text = build_input_text(one_report(**{"Location Name": "ICU", "Age at Encounter": 74.0}))
    assert text.iloc[0] == "Unit: ICU. Age: 74.\nPatient found on floor, no injury noted."


# --- Event No. split decoding ---

@pytest.mark.parametrize("event_no, split", [
    ("SYNPROD-TRAIN-0000001", "TRAIN"),
    ("SYNPROD-VALIDATION-0051234", "VALIDATION"),
    ("SYNPROD-TEST-0006980", "TEST"),
])
def test_split_decoded_from_event_no(event_no, split):
    assert split_from_event_no(event_no) == split


@pytest.mark.parametrize("event_no", [
    "SYNPROD-HOLDOUT-0000001",   # unknown split
    "synprod-train-0000001",     # wrong case
    "SYNPROD-TRAIN-",            # no number
    "SYNPROD-TRAIN-12AB",        # non-digits
    "TRAIN-0000001",             # no SYNPROD prefix
    "SYNPROD-TEST-0000001-X",    # trailing text
    "",
    None,
])
def test_malformed_event_no_raises(event_no):
    with pytest.raises(DataContractError):
        split_from_event_no(event_no)


# --- Grade -> bucket mapping ---

def test_ten_grades_in_ordinal_order():
    letters = [g.split("-", 1)[0] for g in FULL_LABELS]
    assert letters == ["A", "B1", "B2", "C", "D", "E", "F", "G", "H", "I"]


@pytest.mark.parametrize("letter, bucket", [
    ("A", "none"), ("B1", "none"), ("B2", "none"), ("C", "none"), ("D", "none"),
    ("E", "some"), ("F", "some"),
    ("G", "serious"), ("H", "serious"), ("I", "serious"),
])
def test_grade_maps_to_bucket(letter, bucket):
    grade = next(g for g in FULL_LABELS if g.split("-", 1)[0] == letter)
    assert BUCKET_MAP[grade] == bucket


def test_bucket_groups_cover_each_grade_once():
    assert list(BUCKET_GROUPS) == LABELS
    assert BUCKET_GROUPS == {"none": [0, 1, 2, 3, 4], "some": [5, 6], "serious": [7, 8, 9]}


def test_hurt_grades_are_some_plus_serious():
    assert HURT_GRADES == {g for g, b in BUCKET_MAP.items() if b in ("some", "serious")}

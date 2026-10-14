"""Checks on the rolling-evaluation folds and the paired comparison. None of
these read the dataset or train a model."""

import pandas as pd
import pytest

from data_pipeline import DataContractError, assert_months_match_splits, rolling_folds
from rolling_evaluation import compare

MONTHS = [f"2024-{m:02d}" for m in range(1, 11)]
SPLITS = ["TRAIN"] * 8 + ["VALIDATION", "TEST"]


def ten_months(rows_per_month=3):
    return pd.DataFrame([
        {"event_no": f"{month}-{i}", "event_month": month, "split": split}
        for month, split in zip(MONTHS, SPLITS) for i in range(rows_per_month)
    ]).sample(frac=1, random_state=0).reset_index(drop=True)


# --- Fold construction ---

def test_folds_start_in_march_and_end_in_october():
    test_months = [test["event_month"].iloc[0] for _, _, test in rolling_folds(ten_months())]
    assert test_months == MONTHS[2:]


def test_train_precedes_cutoff_precedes_test():
    for train, cutoff, test in rolling_folds(ten_months()):
        assert cutoff["event_month"].nunique() == 1 and test["event_month"].nunique() == 1
        assert train["event_month"].max() < cutoff["event_month"].iloc[0] < test["event_month"].iloc[0]


def test_window_expands_from_january():
    for train, cutoff, _ in rolling_folds(ten_months()):
        expected = [m for m in MONTHS if m < cutoff["event_month"].iloc[0]]
        assert sorted(train["event_month"].unique()) == expected


def test_no_report_is_tested_twice():
    tested = pd.concat(test for _, _, test in rolling_folds(ten_months()))
    assert tested["event_no"].is_unique


def test_last_fold_is_the_production_split():
    df = ten_months()
    train, cutoff, test = list(rolling_folds(df))[-1]
    assert set(train["event_no"]) == set(df.loc[df.split == "TRAIN", "event_no"])
    assert set(cutoff["event_no"]) == set(df.loc[df.split == "VALIDATION", "event_no"])
    assert set(test["event_no"]) == set(df.loc[df.split == "TEST", "event_no"])


def test_min_train_months_delays_the_first_fold():
    test_months = [test["event_month"].iloc[0] for _, _, test in rolling_folds(ten_months(), min_train_months=2)]
    assert test_months == MONTHS[3:]


# --- Event Date vs. Event No. split ---

def test_months_matching_splits_pass():
    assert_months_match_splits(ten_months())


def test_month_shared_by_two_splits_raises():
    df = ten_months()
    df.loc[df["event_month"] == "2024-09", "event_month"] = "2024-08"
    with pytest.raises(DataContractError):
        assert_months_match_splits(df)


# --- Paired comparison ---

def scored(flags):
    return pd.DataFrame({
        "event_no": [f"e{i}" for i in range(len(flags))],
        "bucket": "serious",
        "flagged": flags,
    })


def test_compare_counts_disagreements():
    a = scored([True, True, False, False, True])
    b = scored([True, False, True, True, True])
    out = compare(a, b)
    assert (out["both"], out["only_a"], out["only_b"], out["neither"]) == (2, 1, 2, 0)
    assert out["n_serious"] == 5


def test_compare_identical_runs_is_not_significant():
    a = scored([True, False, True])
    assert compare(a, a)["mcnemar_exact_p"] == 1.0


def test_compare_rejects_different_cases():
    a = scored([True, False, True])
    with pytest.raises(ValueError):
        compare(a, scored([True, False, True]).assign(event_no=["x", "y", "z"]))

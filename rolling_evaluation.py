"""
Rolling evaluation of the production model, to score the serious tier
(G/H/I) on far more than October's 20 cases.

The production split tests on one month. This script repeats that split
shape across the year with an expanding window:

    train Jan            cutoff Feb   test Mar
    train Jan-Feb        cutoff Mar   test Apr
    ...
    train Jan-Aug        cutoff Sep   test Oct   <- the production split

Each fold trains what severity_model.py trains (fit_model), picks the
>=95%-recall triage cutoff on the cutoff month only, freezes it, and scores
the test month. Every Mar-Oct report is scored exactly once, by a model that
never saw its month or any later one, so the per-report results can be
pooled: 303 serious cases instead of 20.

How to read the output:

- This is a DEVELOPMENT metric, not a second test set. Jan-Aug informed
  every modelling choice so far. October alone stays the final check.
- Jan-Aug deliberately oversamples harm (~13% of reports vs ~4% in Sep/Oct),
  so precision and flag rate in those months are not what a natural month
  would give. Pooled numbers are therefore limited to rates that do not
  depend on the harm rate: recall on a harm tier, the false-alarm rate on
  unharmed reports, and where serious cases rank among unharmed reports.
  Precision and flag rate are reported per fold, next to that month's harm
  rate.
- Early folds train on much less data than production (the March fold
  trains on January alone), so the pooled figures understate the production
  model. The per-fold table shows the trend.
- Pooled results are also split into oversampled months (Mar-Aug) and
  natural-rate months (Sep-Oct). If the oversampled months look clearly
  easier, the pooled number is optimistic for real traffic.

To measure a change, pass a different fit function to run_rolling() and
hand both results to compare(), which pairs the two models on the same
serious cases.

Writes severity_model/rolling_evaluation.json (aggregates only) and
severity_model/rolling_serious_misses.csv (row-level, gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import precision_score

from data_pipeline import FULL_LABELS, LABELS, load_dataset, rolling_folds
from severity_model import (
    OUT_DIR,
    TARGET_RECALL,
    bucket_probs,
    fit_model,
    pick_cutoff_for_recall,
    triage_score,
)
from severity_model_extended import wilson_interval

MIN_TRAIN_MONTHS = 1  # first test month is March (train Jan, cutoff Feb)
SERIOUS_GRADES = [g for g in FULL_LABELS if g[0] in "GHI"]
# False-alarm budgets for the cutoff-free view: share of serious cases that
# outscore all but this share of the same month's unharmed reports.
FALSE_ALARM_RATES = [0.05, 0.10, 0.20]


def month_span(months):
    months = sorted(months)
    return months[0] if len(months) == 1 else f"{months[0]}..{months[-1]}"


def run_rolling(fit_fn=fit_model, df=None, min_train_months=MIN_TRAIN_MONTHS, verbose=True):
    """Score every test month with a model fit on earlier months only.

    fit_fn(train_df) must return (vectorizer, clf), where clf has
    predict_proba and classes_ indexing into FULL_LABELS.

    Returns (oof, folds): oof has one row per scored report (each report
    appears once), folds has one dict per fold describing how it was fit.
    """
    if df is None:
        df = load_dataset()

    oof, folds = [], []
    for train_df, cutoff_df, test_df in rolling_folds(df, min_train_months):
        vectorizer, clf = fit_fn(train_df)
        triage_cutoff_month = triage_score(clf.predict_proba(vectorizer.transform(cutoff_df["text"])), clf.classes_)
        cutoff = pick_cutoff_for_recall(cutoff_df["hurt_label"].to_numpy(), triage_cutoff_month, TARGET_RECALL)

        proba = clf.predict_proba(vectorizer.transform(test_df["text"]))
        triage = triage_score(proba, clf.classes_)

        out = test_df[["event_no", "split", "event_month", "grade", "full_label", "bucket", "hurt_label", "text"]].copy()
        out["triage"] = triage
        out["cutoff"] = cutoff
        out["flagged"] = triage >= cutoff
        out["bucket_pred"] = np.array(LABELS)[bucket_probs(proba, clf.classes_).argmax(axis=1)]
        out["grade_pred"] = clf.classes_[proba.argmax(axis=1)]
        # Share of this month's unharmed reports scoring below each report.
        none_scores = np.sort(triage[(test_df["bucket"] == "none").to_numpy()])
        out["none_pct"] = np.searchsorted(none_scores, triage, side="left") / len(none_scores)
        oof.append(out)

        fold = {
            "test_month": test_df["event_month"].iloc[0],
            "cutoff_month": cutoff_df["event_month"].iloc[0],
            "train_months": month_span(train_df["event_month"].unique()),
            "n_train": len(train_df),
            "cutoff": cutoff,
        }
        folds.append(fold)
        if verbose:
            print(f"  fold: train {fold['train_months']} ({fold['n_train']:,} reports), "
                  f"cutoff {fold['cutoff_month']} = {cutoff:.4f}, test {fold['test_month']}")

    return pd.concat(oof, ignore_index=True), folds


def rate(successes, n):
    lo, hi = wilson_interval(successes, n)
    return {"n": int(n), "count": int(successes), "rate": successes / n if n else float("nan"),
            "wilson_95": [lo, hi]}


def serious_summary(rows):
    """Counts and Wilson intervals for the G/H/I reports in `rows`."""
    s = rows[rows["bucket"] == "serious"]
    out = {
        "flagged_by_triage_cutoff": rate(s["flagged"].sum(), len(s)),
        "labelled_serious_by_bucket": rate((s["bucket_pred"] == "serious").sum(), len(s)),
        "by_grade": {
            g.split("-", 1)[0]: {
                "flagged_by_triage_cutoff": rate((s["flagged"] & (s["grade"] == g)).sum(), (s["grade"] == g).sum()),
                "labelled_serious_by_bucket": rate(((s["bucket_pred"] == "serious") & (s["grade"] == g)).sum(),
                                                   (s["grade"] == g).sum()),
            }
            for g in SERIOUS_GRADES
        },
        # Cutoff-free: how serious cases rank against the same month's
        # unharmed reports. The mean is the serious-vs-none AUROC.
        "mean_share_of_unharmed_outscored": float(s["none_pct"].mean()) if len(s) else float("nan"),
        "caught_at_false_alarm_rate": {
            f"{fa:.0%}": rate((s["none_pct"] >= 1 - fa).sum(), len(s)) for fa in FALSE_ALARM_RATES
        },
    }
    return out


def harm_independent_summary(rows):
    """Rates that don't move with the month's harm rate."""
    hurt = rows[rows["hurt_label"] == 1]
    none = rows[rows["bucket"] == "none"]
    return {
        "serious": serious_summary(rows),
        "hurt_recall_at_cutoff": rate(hurt["flagged"].sum(), len(hurt)),
        "unharmed_flagged_at_cutoff": rate(none["flagged"].sum(), len(none)),
    }


def summarize(oof, folds):
    per_fold = []
    for fold in folds:
        rows = oof[oof["event_month"] == fold["test_month"]]
        per_fold.append({
            **fold,
            "n_test": len(rows),
            "test_hurt_rate": float(rows["hurt_label"].mean()),
            **harm_independent_summary(rows),
            # These two depend on the month's harm rate; see test_hurt_rate.
            "precision_at_cutoff": float(precision_score(rows["hurt_label"], rows["flagged"], zero_division=0)),
            "flag_rate": float(rows["flagged"].mean()),
        })

    oversampled = oof[oof["split"] == "TRAIN"]
    natural = oof[oof["split"] != "TRAIN"]
    return {
        "config": {
            "target_recall": TARGET_RECALL,
            "min_train_months": MIN_TRAIN_MONTHS,
            "test_months": month_span(oof["event_month"].unique()),
            "note": "Development metric. Jan-Aug oversamples harm and informed modelling choices; "
                    "October alone remains the final test.",
        },
        "pooled": harm_independent_summary(oof),
        "pooled_oversampled_months": {"months": month_span(oversampled["event_month"].unique()),
                                      **harm_independent_summary(oversampled)},
        "pooled_natural_rate_months": {"months": month_span(natural["event_month"].unique()),
                                       **harm_independent_summary(natural)},
        "per_fold": per_fold,
    }


def compare(oof_a, oof_b, column="flagged"):
    """Pair two run_rolling() results on the same serious cases. `column` is
    a boolean per-report outcome ('flagged', or a column you add). Returns
    the 2x2 counts and an exact McNemar p-value on the cases where the two
    models disagree."""
    a = oof_a[oof_a["bucket"] == "serious"].set_index("event_no")[column].astype(bool)
    b = oof_b[oof_b["bucket"] == "serious"].set_index("event_no")[column].astype(bool).reindex(a.index)
    if b.isna().any():
        raise ValueError("The two runs did not score the same serious cases")
    b = b.astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "n_serious": len(a),
        "both": int((a & b).sum()), "only_a": only_a, "only_b": only_b, "neither": int((~a & ~b).sum()),
        "mcnemar_exact_p": float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0,
    }


# Per-report outcomes on serious cases that compare_models() pairs. The
# triage flag is near its ceiling (6 misses in 303), so the other two are
# where a change can show up.
OUTCOMES = {
    "flagged_by_triage_cutoff": lambda oof: oof["flagged"],
    "labelled_serious_by_bucket": lambda oof: oof["bucket_pred"] == "serious",
    "caught_at_5pct_false_alarms": lambda oof: oof["none_pct"] >= 0.95,
}


def cost_summary(oof):
    """What a model's serious-tier results cost on everything else."""
    hurt, none = oof[oof["hurt_label"] == 1], oof[oof["bucket"] == "none"]
    serious = oof[oof["bucket"] == "serious"]
    return {
        "hurt_recall_at_cutoff": float(hurt["flagged"].mean()),
        "unharmed_flagged_at_cutoff": float(none["flagged"].mean()),
        "non_serious_labelled_serious": int(((oof["bucket_pred"] == "serious") & (oof["bucket"] != "serious")).sum()),
        "serious_vs_unharmed_auroc": float(serious["none_pct"].mean()),
    }


def compare_models(oof_a, oof_b):
    """Pair model a (baseline) and model b (variant) on every outcome in
    OUTCOMES, for all test months and for the natural-rate months alone, with
    each model's costs alongside."""
    def paired(a, b):
        return {name: compare(a.assign(_outcome=fn(a)), b.assign(_outcome=fn(b)), column="_outcome")
                for name, fn in OUTCOMES.items()}

    natural_a, natural_b = oof_a[oof_a["split"] != "TRAIN"], oof_b[oof_b["split"] != "TRAIN"]
    return {
        "all_test_months": paired(oof_a, oof_b),
        "natural_rate_months": paired(natural_a, natural_b),
        "costs_a": cost_summary(oof_a),
        "costs_b": cost_summary(oof_b),
    }


def print_comparison(name, cmp):
    print(f"\n--- {name} (a = baseline, b = variant) ---")
    for outcome, c in cmp["all_test_months"].items():
        nat = cmp["natural_rate_months"][outcome]
        print(f"  {outcome:30s} a {c['both'] + c['only_a']:>3}/{c['n_serious']}  b {c['both'] + c['only_b']:>3}/{c['n_serious']}"
              f"  (only a {c['only_a']:>2}, only b {c['only_b']:>2}, McNemar p={c['mcnemar_exact_p']:.3f})"
              f"   Sep-Oct: a {nat['both'] + nat['only_a']}/{nat['n_serious']}, b {nat['both'] + nat['only_b']}/{nat['n_serious']}")
    for key in cmp["costs_a"]:
        a, b = cmp["costs_a"][key], cmp["costs_b"][key]
        shown = f"a {a}  b {b}" if isinstance(a, int) else f"a {a:.4f}  b {b:.4f}"
        print(f"  {key:30s} {shown}")


def check_last_fold_is_production(folds):
    """The last fold is the production split, so its cutoff must match the
    one severity_model.py saved. Catches the folds drifting from production."""
    path = f"{OUT_DIR}/config.json"
    if not os.path.exists(path):
        print(f"({path} not found, skipping the production-split check)")
        return
    saved = json.load(open(path))["cutoff_raw_triage"]
    if not np.isclose(folds[-1]["cutoff"], saved, atol=1e-3):
        raise RuntimeError(f"Last fold cutoff {folds[-1]['cutoff']:.4f} does not match the saved "
                           f"production cutoff {saved:.4f}: the last fold is not the production split")
    print(f"Last fold reproduces the production cutoff ({folds[-1]['cutoff']:.4f} vs saved {saved:.4f}).")


def fmt(r):
    lo, hi = r["wilson_95"]
    return f"{r['count']:>3}/{r['n']:<3} = {r['rate']:6.1%}  (95% CI {lo:.1%}-{hi:.1%})"


def print_report(results):
    print("\n" + "=" * 78)
    print("PER FOLD (precision and flag rate depend on the month's harm rate)")
    print("=" * 78)
    print(f"{'test':8s} {'train':>7s} {'cutoff':>7s} {'hurt%':>6s} {'serious flagged':>16s} "
          f"{'as serious':>11s} {'hurt recall':>12s} {'precision':>10s} {'flag rate':>10s}")
    for f in results["per_fold"]:
        s = f["serious"]
        fl, lab = s["flagged_by_triage_cutoff"], s["labelled_serious_by_bucket"]
        print(f"{f['test_month']:8s} {f['n_train']:>7,} {f['cutoff']:>7.3f} {f['test_hurt_rate']:>6.1%} "
              f"{fl['count']:>9}/{fl['n']:<2} {fl['rate']:>4.0%} "
              f"{lab['count']:>5}/{lab['n']:<2} {lab['rate']:>3.0%} "
              f"{f['hurt_recall_at_cutoff']['rate']:>12.1%} {f['precision_at_cutoff']:>10.1%} {f['flag_rate']:>10.1%}")

    for key, title in [("pooled", f"POOLED, {results['config']['test_months']}"),
                       ("pooled_oversampled_months", "OVERSAMPLED MONTHS ONLY"),
                       ("pooled_natural_rate_months", "NATURAL-RATE MONTHS ONLY")]:
        block = results[key]
        s = block["serious"]
        print("\n" + "=" * 78)
        print(title + (f" ({block['months']})" if "months" in block else ""))
        print("=" * 78)
        print(f"Serious flagged by triage cutoff:  {fmt(s['flagged_by_triage_cutoff'])}")
        print(f"Serious labelled `serious`:        {fmt(s['labelled_serious_by_bucket'])}")
        for letter, g in s["by_grade"].items():
            print(f"  grade {letter}: flagged {fmt(g['flagged_by_triage_cutoff'])}")
            print(f"           as serious {fmt(g['labelled_serious_by_bucket'])}")
        for fa, r in s["caught_at_false_alarm_rate"].items():
            print(f"Serious caught at {fa:>3s} false alarms:  {fmt(r)}")
        print(f"Serious-vs-unharmed AUROC:         {s['mean_share_of_unharmed_outscored']:.4f}")
        print(f"Hurt (E-I) recall at cutoff:       {fmt(block['hurt_recall_at_cutoff'])}")
        print(f"Unharmed reports flagged:          {fmt(block['unharmed_flagged_at_cutoff'])}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Rolling evaluation, expanding window:")
    oof, folds = run_rolling()
    check_last_fold_is_production(folds)

    results = summarize(oof, folds)
    print_report(results)

    with open(f"{OUT_DIR}/rolling_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    misses = oof[(oof["bucket"] == "serious") & ~oof["flagged"]]
    misses = misses[["event_no", "event_month", "grade", "triage", "cutoff", "bucket_pred", "text"]]
    misses.sort_values(["event_month", "triage"]).to_csv(f"{OUT_DIR}/rolling_serious_misses.csv", index=False)

    print(f"\nSaved {OUT_DIR}/rolling_evaluation.json and "
          f"{OUT_DIR}/rolling_serious_misses.csv ({len(misses)} serious cases below their fold's cutoff).")


if __name__ == "__main__":
    main()

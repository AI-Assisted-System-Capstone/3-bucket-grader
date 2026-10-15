# Project Guide

Read this first. It explains what the project is for, where everything lives,
what works today, what's left, and the rules everyone follows when changing
the code. [README.md](README.md) has the full results and run instructions;
[PIPELINE.md](PIPELINE.md) has the history of every experiment.

## Goal

Penn State Health's MIDAS team receives thousands of patient-safety event
reports. Reviewers can't read them all quickly. This capstone builds tools
that help a safety reviewer, without making the official decision for them:

1. **Severity triage:** predict how badly the patient was harmed (PSRS grade
   A–I), and flag likely-harm reports for priority review.
2. **Patterns and themes:** group reports into recurring themes without
   predefined categories, find similar past reports, and spot weekly spikes.
3. **One place to review it:** a local analyst app that shows all of the above
   for each report.

Sponsor contact: Christopher Jackson. Advisor: Al Verbanoc.

The sponsor-facing requirements (from the Beta Prototype Report):

| # | Requirement | Where it's addressed |
|---|---|---|
| 1 | Find recurring themes in event comments without predefined categories | `analyst_app/src/pattern_engine.py`, `analyst_app/backend/patterns.py` |
| 2 | Find patterns associated with different PSRS harm levels | `severity_model.py` |
| 3 | Estimate severity using only information available at submission | `data_pipeline.py` (excluded columns, date split) |

## Who owns what

| Person | Area |
|---|---|
| Devanshi | Severity pipeline (`data_pipeline.py`, `severity_model.py`, `severity_model_extended.py`), repo upkeep |
| Madhura | `predict.py`, `tests/`, `experiments/finetune_distilbert.py` |
| Alana | `rolling_evaluation.py` and its tests |
| Rushil | `analyst_app/` (pattern engine, backend, frontend) |
| Event-type teammate | Separate `11-buckets` repo (11-way event-type classifier) |

## Repo layout

```
data_pipeline.py            loads data, drops post-review fields, builds model input,
                            date split (Jan–Aug train / Sep tune / Oct test), labels
severity_model.py           production model: TF-IDF + logistic regression on 10 grades,
                            cutoff and calibration fit on September, evaluated on October
severity_model_extended.py  sponsor comparison, operating points, subgroup breakdowns
rolling_evaluation.py       retrains month by month to score 303 serious cases (dev metric)
predict.py                  scores a CSV/parquet of new reports with the saved model
prefill_extra_fields.py     HPI Designation classifier
severity_model/             evaluation JSONs + config (committed); .joblib files (not committed)
tests/                      pytest checks for the safety-critical pieces (no dataset needed)
experiments/                retired or in-progress experiments; see experiments/README.md
analyst_app/                FastAPI + React analyst app and pattern engine; see its README
data/                       put the dataset here (gitignored)
requirements.txt            pinned Python packages for everything outside analyst_app/
```

## Setup

```bash
pip install -r requirements.txt
# dataset: copy or symlink the sponsor's parquet file to
#   data/LOCAL_ONLY_student_facing_candidate.parquet
# or set SEVERITY_DATA_PATH=/path/to/file.parquet
python3 severity_model.py     # trains and saves severity_model/*.joblib (needed by predict.py and the app)
python3 -m pytest             # 51 tests, no dataset needed
```

The analyst app has its own requirements and setup: see
[analyst_app/MVP_GUIDE.md](analyst_app/MVP_GUIDE.md). Its tests:
`cd analyst_app && ./run_tests.sh`.

Run experiments from the repo root as modules, e.g.
`python3 -m experiments.error_analysis`.

## Current status (October 2026)

All numbers are on the October 2024 test month (8,312 reports) unless noted.
They come from `severity_model/evaluation.json` and
`severity_model/rolling_evaluation.json`; if you retrain, update them from
the JSON.

The current model (word-counting, `severity_model.py`):

| What | Result |
|---|---|
| Harm flag recall (cutoff 0.2361, set on September) | 96.0% (387 of 403 harmed patients) |
| Reports flagged for review | 16.1% |
| Precision of the flag | 28.9% |
| Letter-grade accuracy (A–I) | 76.6% |
| 3-bucket macro F1 | 0.82 |
| Sensitivity at 96% specificity | 90.3% (sponsor reports 95.2% for their model) |
| Serious cases flagged, rolling Mar–Oct | 297 of 303 (98.0%) |
| Serious cases labelled `serious`, rolling Mar–Oct | 157 of 303 (51.8%); grade G only 13 of 69 |

Head-to-head with the sponsor's model, same data, same split, same October
reports. "Caught" = harmed patients caught while 96% of no-harm reports are
left alone. Details in README ("Compared with the sponsor's model") and
PIPELINE.md.

| Model | Caught |
|---|---|
| Our current model | 90.3% |
| Our word-counting model, improved settings (not adopted yet) | 91.6% |
| Sponsor's model, report as submitted | 91.6% |
| Sponsor's model with manager comments and unit actions | 93.8% |
| **Our DistilBERT** (report as submitted) | **94.3%** |

In short: given the same information, our DistilBERT beats the sponsor's
model, and it ties their model even when theirs reads the after-review
fields. Their 95.2% comes from their real data and random split.

Working:

- Production severity model, saved and reproducible from the committed code.
- `predict.py` scores new report files.
- Analyst app runs locally, using this repo's severity model by default.
  Pattern themes use local MiniLM embeddings.

Not done yet:

- Decide whether DistilBERT replaces the current model. If yes, adjust its
  cutoff on September and make `predict.py` and the app load it.
- Train DistilBERT more than once, to see how much the numbers move between
  runs. Each run takes about 40 minutes on a server GPU, about 3 hours on a
  laptop.
- The analyst app's event-type adapter needs the `11-buckets` repo and its
  trained model, which aren't in this repo.

The full task list, including questions that need the sponsor, is under
"Open tasks" in [README.md](README.md). Keep that list as the one source of
truth: tick items there when they're done.

## Rules

These protect the results. Breaking one usually doesn't crash anything; it
just quietly makes the numbers wrong.

1. **Never use post-review fields as model input.** `manager_comments`,
   `unit_actions_taken`, `shareable_lessons`, `HPI Designation`,
   `Analyst-Report Type` and `Level of Invet` are written after review and
   leak the answer. `data_pipeline.EXCLUDED_COLUMNS` enforces this; don't
   weaken it.
2. **Keep the date split.** Train on January–August, tune on September,
   test on October. No random splits.
3. **Never tune on October.** Cutoffs, calibration and model choices are
   picked on September (or with the rolling folds). October is scored once.
4. **Never commit data.** No parquet, CSV, report narratives or `.joblib`
   files. `.gitignore` covers these; don't force-add around it.
5. **Numbers in docs come from the JSON.** After a retrain, regenerate the
   JSON and update every table that quotes it (README, PIPELINE.md, this
   file).
6. **Run the tests before pushing:** `python3 -m pytest`, plus
   `analyst_app/run_tests.sh` if you touched the app.
7. **One branch per change, merged by pull request.** Pull `main` first.
   Keep commits small, with plain messages that say what changed and why.
   Don't add tool-generated attribution lines or trailers.
8. **Keep the README current.** If you add a script, add it to the README's
   workflow and this file's layout; if you finish a task, tick it.

## Data

The dataset is synthetic (IDs start with `SYNPROD-`), provided by the
sponsor: 80,000 reports, 67,003 usable. It is not in the repo. Get the
parquet file from the team's shared capstone files and put it in `data/`.
Harm grades are PSRS A–I, grouped into buckets: `none` (A–D), `some` (E–F),
`serious` (G–I).

# MIDAS Pattern Recognition — Local Prototype

## Analyst web MVP

The repository now also contains a local FastAPI + React/TypeScript analyst app:
independent severity and event-type adapters, recurring semantic themes, configurable
top-N historical case linkage, weekly trends, and separate analyst decisions.

Start with [MVP_GUIDE.md](MVP_GUIDE.md) for setup, environment variables, model
artifact requirements, architecture and limitations. [PLAN.md](PLAN.md) records
the upstream repository inspection. The existing CLI below remains available.

The trained severity/event-type weights are **not included in either upstream
repository**. Normal mode reports these models as unavailable; fixture mode must
be explicitly enabled and is visibly labeled. Pattern embeddings can use the real
locally cached MiniLM independently of those missing classifier weights.

This folder implements the pattern-recognition phase of the Penn State Health MIDAS capstone entirely locally. It does not send report data to any hosted model API.

## Why this version is stronger than the first baseline

- **No post-review leakage by default.** Similarity text uses the incoming event narrative, event type, and medication fields. Manager comments, actions taken, PSRS score, HPI designation, investigation level, and analyst classification are not used for semantic matching by default.
- **Scales better.** Retrieval uses `NearestNeighbors` instead of materializing a full N×N cosine-similarity matrix. This matters for the ~24k-report dataset.
- **Sparse TF-IDF stays sparse.** The offline fallback does not convert a 24k × vocabulary matrix into a huge dense array.
- **Data-calibrated novelty.** Novelty is based on nearest-neighbor distance, with a threshold learned from the historical distribution instead of a hard-coded similarity cutoff.
- **Better clustering behavior.** The pipeline prefers HDBSCAN for sufficiently large dense semantic-embedding datasets when available; otherwise it uses MiniBatchKMeans and selects k by silhouette score.
- **Better trend calculations.** Missing weeks are explicitly filled with zero counts, and spike detection uses a count-oriented upper control limit rather than unstable z-scores on sparse weekly data.
- **Time-aware search.** You can restrict similar cases to a historical lookback window, matching the project idea of showing recent similar events.

## Folder layout

- `data/` — local-only input directory; report data is excluded from version control
- `src/pattern_engine.py` — pattern pipeline
- `src/cli.py` — command-line interface
- `tests/test_pattern_engine.py` — local tests
- `output/` — generated results; excluded from version control
- `run_local.sh` — one-command local demo

## Run locally

```bash
./run_local.sh --input /absolute/path/to/local_midas_reports.csv --force-tfidf
```

The `--force-tfidf` option is a fully offline smoke-test mode. For the intended semantic version, install `sentence-transformers` and cache the model locally, then run without `--force-tfidf`.

The default semantic model is the same one already described in the project materials:

```text
sentence-transformers/all-MiniLM-L6-v2
```

No API token is required. By default the code only loads a locally cached model. If the workstation has approved internet access and you intentionally want the first model download, add `--allow-model-download` once.

MIDAS source files, report text, embeddings, and generated outputs must remain on the
local workstation. The repository ignore rules exclude common data formats, `data/`,
`output*/`, and embedding files from version control.

## Outputs

- `pattern_reports.csv` — normalized reports + cluster/novelty fields
- `retrieval_neighbors.csv` — configurable top-k historical neighbors for manual reviewer evaluation
- `embeddings.npy` or `embeddings_sparse.npz`
- `cluster_names.json`
- `weekly_cluster_trends.csv`
- `monthly_cluster_trends.csv`
- `pattern_metrics.json`
- `embedding_projection.csv`
- `embedding_projection.png`

The number of retrieved/exported similar reports is configurable with `--top-k`.
It defaults to 5 and is capped by the number of available reports:

```bash
./run_local.sh --input /absolute/path/to/local_midas_reports.csv --top-k 25
```

## What “optimal” means here

There is no defensible way to call one embedding model optimal without a reviewer-labeled validation set. The right production process is to have safety reviewers label whether returned historical cases are genuinely related, then compare local embedding models using Precision@K / NDCG@K for the selected top-k value. `all-MiniLM-L6-v2` is a good fast baseline, not a guaranteed winner.

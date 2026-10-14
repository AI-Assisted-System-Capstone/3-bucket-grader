# Local test results

Tested on the supplied 23-report synthetic MIDAS sample on 2026-09-13.

## Unit tests

All 3 local test cases passed:

- schema aliases + target/review-field leakage prevention
- similarity retrieval + novelty behavior + JSON-serializable results
- weekly/monthly trend period generation and zero-count filling

## End-to-end run

After caching the public model once with:

```bash
./run_local.sh --allow-model-download
```

the normal offline command completed successfully:

```bash
./run_local.sh
```

Result:

- 23 reports loaded and normalized
- local `sentence-transformers/all-MiniLM-L6-v2` embeddings generated (23 x 384)
- exact cosine nearest-neighbor index built
- 3 MiniBatchKMeans exploratory clusters selected
- configurable top-k neighbors exported for every report (5 in this recorded run)
- novelty scores and a data-calibrated threshold generated
- weekly and monthly trend tables generated with zero-count periods
- anomaly spike flags generated (no spikes in this small sample)
- 2-D UMAP projection and PNG generated
- demo query completed successfully

For the query:

```text
A medication was nearly administered at the wrong dose after a barcode scanner problem.
```

the top historical match was `SYNPROD-TRAIN-0000010` with cosine similarity
`0.762059`, describing a barcode scanner failure/manual override followed by
incorrect medication administration. The query novelty score was `0.237941`
against a threshold of `0.504119`, so it was not flagged as novel.

## Limitations

The 23-row sample is too small and heterogeneous to judge cluster quality or tune
spike detection. Retrieval and clustering should be evaluated on the full dataset
with patient-safety reviewers labeling whether top-five matches are genuinely related.
The macOS system Python 3.9 runtime uses LibreSSL 2.8.3, which produces a harmless
`urllib3` compatibility notice; cached-model inference remains fully local.

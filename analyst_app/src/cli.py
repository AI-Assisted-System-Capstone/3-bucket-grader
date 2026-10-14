from __future__ import annotations

import argparse
import json
from pathlib import Path

from pattern_engine import PatternEngine, load_reports, normalize_schema, print_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MIDAS pattern-recognition pipeline")
    parser.add_argument("--input", required=True, help="CSV/XLSX/Markdown/JSON input")
    parser.add_argument("--sheet", default=None, help="Excel sheet name")
    parser.add_argument("--output", default="output")
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cluster-method", choices=["auto", "kmeans", "hdbscan"], default="auto")
    parser.add_argument("--force-tfidf", action="store_true", help="Use local TF-IDF instead of a transformer")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Hugging Face model download; off by default")
    parser.add_argument("--include-review-fields", action="store_true", help="Include manager/action/lesson text; off by default to prevent post-review leakage")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar reports to return and export (default: 5; capped at available reports)",
    )
    parser.add_argument("--query", default=None, help="Optional new report text")
    parser.add_argument("--as-of-date", default=None, help="Optional YYYY-MM-DD date for historical lookback")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--service", default=None)
    args = parser.parse_args()

    raw = load_reports(args.input, args.sheet)
    df = normalize_schema(raw)
    engine = PatternEngine(
        n_clusters=args.clusters,
        model_name=args.model,
        force_tfidf=args.force_tfidf,
        allow_model_download=args.allow_model_download,
        include_review_fields=args.include_review_fields,
        cluster_method=args.cluster_method,
        retrieval_top_k=args.top_k,
    )
    artifacts = engine.fit(df)
    engine.save(args.output)

    print(json.dumps(engine.sanity_metrics(), indent=2))
    print(f"Saved pattern outputs to: {Path(args.output).resolve()}")
    print("Cluster names:")
    for c, name in artifacts.cluster_names.items():
        print(f"  {c}: {name}")

    if args.query:
        result = engine.search(
            args.query,
            top_k=args.top_k,
            as_of_date=args.as_of_date,
            lookback_days=args.lookback_days,
            location=args.location,
            service=args.service,
        )
        print_demo(result)


if __name__ == "__main__":
    main()

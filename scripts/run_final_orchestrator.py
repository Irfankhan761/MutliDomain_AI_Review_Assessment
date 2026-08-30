from pathlib import Path
import argparse
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pipeline.final_orchestrator import FinalOrchestrator


def print_orchestrator_trace(state):
    print("\nFINAL ORCHESTRATOR EXECUTION TRACE")
    print("=" * 90)

    for i, item in enumerate(state.get("execution_trace", []), start=1):
        print(f"{i}. {item.get('step')} -> {item.get('message')}")
        output = item.get("output", {})
        if output:
            print(f"   Output: {output}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13: Final Orchestrator for Modular AI Framework For Multi-Domain Review Trust Assessment."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", default=None, help="Auto-detected input: CSV path, Google Play URL/app id, or single review text.")
    input_group.add_argument("--csv", default=None, help="CSV dataset path.")
    input_group.add_argument("--input-text", default=None, help="Single review text.")
    input_group.add_argument("--url", default=None, help="Google Play app URL.")
    input_group.add_argument("--app-id", default=None, help="Google Play app id, e.g. com.anydo.")

    parser.add_argument(
        "--input-type",
        default="auto",
        choices=["auto", "csv", "single_review", "google_play_url", "app_id"],
        help="Optional manual input type override.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=["mobile_app", "hotel", "ecommerce", "restaurant", "multidomain"],
        help="Optional domain override.",
    )
    parser.add_argument("--rating", type=float, default=3.0, help="Rating for single-review workflow.")
    parser.add_argument("--entity-id", default="manual_entity", help="Entity id for single-review workflow.")
    parser.add_argument("--entity-name", default="Manual Entity", help="Entity name for single-review workflow.")
    parser.add_argument("--sample-size", type=int, default=200, help="Rows used for CSV workflow. 0 means full dataset.")
    parser.add_argument("--max-reviews", type=int, default=200, help="Reviews to scrape for Google Play workflow.")
    parser.add_argument("--model-path", default="outputs/models/distilbert_sentiment")
    parser.add_argument("--output-dir", default="outputs/final_orchestrator_runs")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG for faster run.")
    parser.add_argument("--no-groq", action="store_true", help="Disable Groq final summary.")
    parser.add_argument("--preview-rows", type=int, default=8)

    args = parser.parse_args()

    print("\nPHASE 13: FINAL ORCHESTRATOR RUN")
    print("=" * 90)

    orchestrator = FinalOrchestrator(
        model_path=args.model_path,
        base_output_dir=args.output_dir,
        use_rag=not args.no_rag,
        use_groq=not args.no_groq,
        sample_size=args.sample_size,
        max_reviews=args.max_reviews,
    )

    result = orchestrator.run(
        input_value=args.input,
        input_path=args.csv,
        input_text=args.input_text,
        url=args.url,
        app_id=args.app_id,
        input_type=args.input_type,
        domain=args.domain,
        rating=args.rating,
        entity_id=args.entity_id,
        entity_name=args.entity_name,
    )

    state = result["state"]
    results = result["results"]

    print_orchestrator_trace(state)

    review_df = results["review_level_results"]
    entity_df = results["entity_level_summary"]

    print("\nORCHESTRATOR STATE SUMMARY")
    print("=" * 90)
    for key in [
        "run_id",
        "status",
        "input_type",
        "domain",
        "selected_workflow",
        "next_agent",
        "app_id",
        "prepared_dataset_path",
        "output_dir",
        "groq_report_path",
    ]:
        print(f"{key}: {state.get(key)}")

    print("\nREVIEW-LEVEL RESULT PREVIEW")
    print("=" * 90)
    review_cols = [
        "domain",
        "entity_name",
        "rating",
        "predicted_sentiment",
        "predicted_star_rating",
        "discrepancy_status",
        "primary_issue",
        "issue_severity_level",
        "trust_score",
        "risk_level",
        "recommendation_level",
    ]
    review_cols = [c for c in review_cols if c in review_df.columns]
    print(review_df[review_cols].head(args.preview_rows).to_string())

    print("\nENTITY-LEVEL RESULT PREVIEW")
    print("=" * 90)
    entity_cols = [
        "domain",
        "entity_type",
        "entity_name",
        "total_reviews",
        "average_trust_score",
        "overall_risk_level",
        "high_risk_percentage",
        "mismatch_percentage",
        "top_issues",
    ]
    entity_cols = [c for c in entity_cols if c in entity_df.columns]
    print(entity_df[entity_cols].head(args.preview_rows).to_string())

    if result.get("final_report"):
        print("\nFINAL GROQ REPORT PREVIEW")
        print("=" * 90)
        print(result["final_report"][:2500])

    print("\nPHASE 13 COMPLETE")
    print("Run output folder:")
    print(result["output_dir"])
    print("\nImportant files:")
    print(Path(result["output_dir"]) / "orchestrator_state.json")
    print(Path(result["output_dir"]) / "prepared_standardised_dataset.csv")
    print(Path(result["output_dir"]) / "analysis_pipeline" / "multidomain_review_level_results.csv")
    print(Path(result["output_dir"]) / "analysis_pipeline" / "multidomain_entity_level_summary.csv")
    print(Path(result["output_dir"]) / "final_groq_report.txt")


if __name__ == "__main__":
    main()

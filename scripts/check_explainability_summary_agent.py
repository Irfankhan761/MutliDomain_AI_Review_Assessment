from pathlib import Path
import argparse
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pipeline.multidomain_review_analysis_pipeline import MultiDomainReviewAnalysisPipeline


def balanced_sample_by_domain(df: pd.DataFrame, sample_size: int):
    if sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)

    if "domain" not in df.columns:
        return df.sample(sample_size, random_state=42).reset_index(drop=True)

    per_domain = max(1, sample_size // df["domain"].nunique())
    parts = []
    for _, group in df.groupby("domain"):
        parts.append(group.sample(min(len(group), per_domain), random_state=42))

    return pd.concat(parts, ignore_index=True)


def save_phase11_reports(review_df: pd.DataFrame, entity_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_distribution = (
        review_df["warning_flags"]
        .value_counts()
        .rename_axis("warning_flags")
        .reset_index(name="count")
    )
    warning_distribution.to_csv(output_dir / "phase11_warning_flag_distribution.csv", index=False)

    recommendation_distribution = (
        review_df["recommendation_level"]
        .value_counts()
        .rename_axis("recommendation_level")
        .reset_index(name="count")
    )
    recommendation_distribution.to_csv(output_dir / "phase11_recommendation_distribution.csv", index=False)

    key_reason_distribution = (
        review_df["key_reasons"]
        .value_counts()
        .rename_axis("key_reasons")
        .reset_index(name="count")
        .head(100)
    )
    key_reason_distribution.to_csv(output_dir / "phase11_key_reason_distribution.csv", index=False)

    explain_cols = [
        "domain", "entity_name", "rating", "predicted_sentiment", "predicted_star_rating",
        "discrepancy_status", "primary_issue", "issue_severity_level",
        "trust_score", "risk_level", "key_reasons", "warning_flags",
        "evidence_based_explanation", "recommendation"
    ]
    explain_cols = [c for c in explain_cols if c in review_df.columns]
    review_df[explain_cols].head(500).to_csv(
        output_dir / "phase11_explainability_examples.csv",
        index=False,
        encoding="utf-8",
    )

    entity_cols = [
        "domain", "entity_type", "entity_name", "total_reviews",
        "average_rating", "average_trust_score", "overall_risk_level",
        "overall_reliability_level", "high_risk_percentage", "mismatch_percentage",
        "top_issues", "evidence_examples", "entity_explanation", "entity_recommendation"
    ]
    entity_cols = [c for c in entity_cols if c in entity_df.columns]
    entity_df[entity_cols].head(500).to_csv(
        output_dir / "phase11_entity_explainability_summary.csv",
        index=False,
        encoding="utf-8",
    )

    return {
        "warning_distribution": warning_distribution,
        "recommendation_distribution": recommendation_distribution,
        "key_reason_distribution": key_reason_distribution,
    }


def print_trace(trace):
    print("\nPHASE 11 PIPELINE EXECUTION TRACE")
    print("=" * 90)
    for i, item in enumerate(trace, start=1):
        print(f"{i}. {item['step']} -> {item['message']}")
        if "output" in item:
            print(f"   Output: {item['output']}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 11 test: advanced explainability and entity summary."
    )
    parser.add_argument("--input", default="data/processed/combined_multidomain_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=200, help="0 means full dataset")
    parser.add_argument("--model-path", default="outputs/models/distilbert_sentiment")
    parser.add_argument("--output-dir", default="outputs/phase11_explainability_summary_pipeline")
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    model_path = PROJECT_ROOT / args.model_path
    output_dir = PROJECT_ROOT / args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"DistilBERT model not found: {model_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = balanced_sample_by_domain(df, args.sample_size)

    print("\nPHASE 11: ADVANCED EXPLAINABILITY AND ENTITY SUMMARY")
    print("=" * 90)
    print("Input:", input_path)
    print("Rows used:", len(df))
    print("RAG enabled:", not args.no_rag)
    print("Output dir:", output_dir)

    if "domain" in df.columns:
        print("\nDomain distribution:")
        print(df["domain"].value_counts())

    pipeline = MultiDomainReviewAnalysisPipeline(
        model_path=str(model_path),
        use_transformer=True,
        use_discrepancy_model=True,
        use_semantic_issue_model=True,
        use_rag=not args.no_rag,
        output_dir=str(output_dir),
    )

    results = pipeline.analyze(df, save_outputs=True)

    review_df = results["review_level_results"]
    entity_df = results["entity_level_summary"]

    reports = save_phase11_reports(review_df, entity_df, output_dir)

    print_trace(results["execution_trace"])

    print("\nEXPLAINABILITY OUTPUT CHECK")
    print("=" * 90)
    check_cols = [
        "explanation_text",
        "evidence_based_explanation",
        "key_reasons",
        "warning_flags",
        "recommendation",
        "recommendation_level",
        "explanation_factors",
    ]
    for col in check_cols:
        print(f"{col}:", "YES" if col in review_df.columns else "NO")

    print("\nWARNING FLAG DISTRIBUTION")
    print("=" * 90)
    print(reports["warning_distribution"].head(20))

    print("\nRECOMMENDATION DISTRIBUTION")
    print("=" * 90)
    print(reports["recommendation_distribution"])

    print("\nREVIEW-LEVEL EXPLANATION PREVIEW")
    print("=" * 90)
    preview_cols = [
        "domain", "rating", "predicted_sentiment", "predicted_star_rating",
        "discrepancy_status", "primary_issue", "trust_score", "risk_level",
        "key_reasons", "warning_flags", "recommendation_level",
        "evidence_based_explanation"
    ]
    preview_cols = [c for c in preview_cols if c in review_df.columns]
    print(review_df[preview_cols].head(8).to_string())

    print("\nENTITY-LEVEL SUMMARY PREVIEW")
    print("=" * 90)
    entity_cols = [
        "domain", "entity_type", "entity_name", "total_reviews",
        "average_trust_score", "overall_risk_level", "high_risk_percentage",
        "mismatch_percentage", "top_issues", "entity_recommendation"
    ]
    entity_cols = [c for c in entity_cols if c in entity_df.columns]
    print(entity_df[entity_cols].head(12).to_string())

    print("\nPHASE 11 COMPLETE")
    print("Outputs saved in:", output_dir)
    print("Important files:")
    print(output_dir / "phase11_explainability_examples.csv")
    print(output_dir / "phase11_entity_explainability_summary.csv")
    print(output_dir / "phase11_warning_flag_distribution.csv")
    print(output_dir / "phase11_recommendation_distribution.csv")


if __name__ == "__main__":
    main()

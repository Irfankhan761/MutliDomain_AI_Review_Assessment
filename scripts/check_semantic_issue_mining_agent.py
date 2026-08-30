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


def print_trace(trace):
    print("\nPHASE 8 PIPELINE EXECUTION TRACE")
    print("=" * 90)

    for i, item in enumerate(trace, start=1):
        print(f"{i}. {item['step']} -> {item['message']}")
        if "output" in item:
            print(f"   Output: {item['output']}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 test: DistilBERT sentiment + BERT discrepancy + MiniLM semantic issue mining."
    )
    parser.add_argument("--input", default="data/processed/combined_multidomain_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--model-path", default="outputs/models/distilbert_sentiment")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    model_path = PROJECT_ROOT / args.model_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"DistilBERT sentiment model not found: {model_path}\n"
            "Run Phase 6 training first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    df = balanced_sample_by_domain(df, args.sample_size)

    print("\nPHASE 8: DOMAIN-AWARE SEMANTIC ISSUE MINING PIPELINE TEST")
    print("=" * 90)
    print("Input file:", input_path)
    print("DistilBERT sentiment model:", model_path)
    print("Rows requested:", args.sample_size)
    print("Rows used before preprocessing:", len(df))

    if "domain" in df.columns:
        print("\nDomain distribution:")
        print(df["domain"].value_counts())

    pipeline = MultiDomainReviewAnalysisPipeline(
        model_path=str(model_path),
        use_transformer=True,
        use_discrepancy_model=True,
        use_semantic_issue_model=True,
        use_rag=False,
        output_dir="outputs/phase8_semantic_issue_pipeline",
    )

    results = pipeline.analyze(df, save_outputs=True)
    review_df = results["review_level_results"]
    entity_df = results["entity_level_summary"]

    print_trace(results["execution_trace"])

    print("\nSENTIMENT MODEL USED DISTRIBUTION")
    print("=" * 90)
    print(review_df["sentiment_model_used"].value_counts())

    print("\nDISCREPANCY MODEL USED DISTRIBUTION")
    print("=" * 90)
    print(review_df["discrepancy_model_used"].value_counts())

    print("\nISSUE MODEL USED DISTRIBUTION")
    print("=" * 90)
    print(review_df["issue_model_used"].value_counts())

    print("\nPRIMARY ISSUE DISTRIBUTION")
    print("=" * 90)
    print(review_df["primary_issue"].value_counts())

    print("\nPRIMARY ISSUE BY DOMAIN")
    print("=" * 90)
    print(pd.crosstab(review_df["domain"], review_df["primary_issue"]))

    print("\nISSUE SEVERITY DISTRIBUTION")
    print("=" * 90)
    print(review_df["issue_severity_level"].value_counts())

    print("\nREVIEW-LEVEL PREVIEW")
    print("=" * 90)
    preview_cols = [
        "domain",
        "rating",
        "predicted_sentiment",
        "sentiment_confidence",
        "predicted_star_rating",
        "discrepancy_status",
        "primary_issue",
        "issue_label",
        "issue_severity_level",
        "primary_issue_similarity",
        "evidence_phrase",
        "trust_score",
        "risk_level",
    ]
    preview_cols = [c for c in preview_cols if c in review_df.columns]
    print(review_df[preview_cols].head(15).to_string())

    print("\nHIGH-SEVERITY ISSUE EXAMPLES")
    print("=" * 90)
    high_df = review_df[review_df["issue_severity_level"] == "high"].copy()
    if len(high_df) == 0:
        print("No high-severity issues found in this sample.")
    else:
        high_cols = [
            "domain",
            "rating",
            "predicted_sentiment",
            "primary_issue",
            "issue_label",
            "primary_issue_similarity",
            "evidence_phrase",
            "review_text",
        ]
        high_cols = [c for c in high_cols if c in high_df.columns]
        print(high_df[high_cols].head(8).to_string())

    print("\nENTITY SUMMARY PREVIEW")
    print("=" * 90)
    summary_cols = [
        "domain",
        "entity_type",
        "entity_name",
        "total_reviews",
        "average_trust_score",
        "overall_risk_level",
        "top_issues",
        "mismatch_percentage",
    ]
    summary_cols = [c for c in summary_cols if c in entity_df.columns]
    print(entity_df[summary_cols].head(10).to_string())

    print("\nPHASE 8 PIPELINE TEST COMPLETE")
    print("Outputs saved in: outputs/phase8_semantic_issue_pipeline/")


if __name__ == "__main__":
    main()

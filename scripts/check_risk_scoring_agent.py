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


def save_phase10_reports(review_df: pd.DataFrame, entity_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_distribution = review_df["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="count")
    risk_distribution.to_csv(output_dir / "phase10_risk_distribution.csv", index=False)

    risk_by_domain = pd.crosstab(review_df["domain"], review_df["risk_level"])
    risk_by_domain.to_csv(output_dir / "phase10_risk_by_domain.csv")

    reliability_by_domain = pd.crosstab(review_df["domain"], review_df["reliability_level"])
    reliability_by_domain.to_csv(output_dir / "phase10_reliability_by_domain.csv")

    penalty_cols = [
        "rating_penalty",
        "sentiment_penalty",
        "issue_penalty",
        "discrepancy_penalty",
        "domain_critical_penalty",
        "uncertainty_penalty",
        "total_penalty",
        "trust_score",
    ]
    penalty_cols = [c for c in penalty_cols if c in review_df.columns]
    penalty_summary = review_df.groupby("domain")[penalty_cols].mean().round(2).reset_index()
    penalty_summary.to_csv(output_dir / "phase10_penalty_summary_by_domain.csv", index=False)

    dominant_factor = (
        review_df["dominant_factor"]
        .value_counts()
        .rename_axis("dominant_factor")
        .reset_index(name="count")
    )
    dominant_factor.to_csv(output_dir / "phase10_dominant_factor_distribution.csv", index=False)

    evidence_strength = (
        review_df["evidence_strength"]
        .value_counts()
        .rename_axis("evidence_strength")
        .reset_index(name="count")
    )
    evidence_strength.to_csv(output_dir / "phase10_evidence_strength_distribution.csv", index=False)

    top_risky_cols = [
        "domain", "entity_type", "entity_name", "total_reviews",
        "average_trust_score", "overall_risk_level", "top_issues",
        "mismatch_percentage"
    ]
    top_risky_cols = [c for c in top_risky_cols if c in entity_df.columns]
    entity_df[top_risky_cols].head(50).to_csv(output_dir / "phase10_top_risky_entities.csv", index=False)

    return {
        "risk_distribution": risk_distribution,
        "risk_by_domain": risk_by_domain,
        "reliability_by_domain": reliability_by_domain,
        "penalty_summary": penalty_summary,
        "dominant_factor": dominant_factor,
        "evidence_strength": evidence_strength,
    }


def print_trace(trace):
    print("\nPHASE 10 PIPELINE EXECUTION TRACE")
    print("=" * 90)
    for i, item in enumerate(trace, start=1):
        print(f"{i}. {item['step']} -> {item['message']}")
        if "output" in item:
            print(f"   Output: {item['output']}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 10 test: advanced risk scoring using sentiment, discrepancy, semantic issue severity and RAG evidence."
    )
    parser.add_argument("--input", default="data/processed/combined_multidomain_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=200, help="0 means full dataset")
    parser.add_argument("--model-path", default="outputs/models/distilbert_sentiment")
    parser.add_argument("--output-dir", default="outputs/phase10_risk_scoring_pipeline")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG for faster test")
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

    print("\nPHASE 10: ADVANCED RISK SCORING PIPELINE")
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

    reports = save_phase10_reports(review_df, entity_df, output_dir)

    print_trace(results["execution_trace"])

    print("\nMODEL USAGE CHECK")
    print("=" * 90)
    print("Sentiment model:")
    print(review_df["sentiment_model_used"].value_counts())
    print("\nDiscrepancy model:")
    print(review_df["discrepancy_model_used"].value_counts())
    print("\nIssue model:")
    print(review_df["issue_model_used"].value_counts())

    print("\nRISK LEVEL DISTRIBUTION")
    print("=" * 90)
    print(reports["risk_distribution"])

    print("\nRISK BY DOMAIN")
    print("=" * 90)
    print(reports["risk_by_domain"])

    print("\nAVERAGE PENALTY SUMMARY BY DOMAIN")
    print("=" * 90)
    print(reports["penalty_summary"])

    print("\nDOMINANT FACTOR DISTRIBUTION")
    print("=" * 90)
    print(reports["dominant_factor"])

    print("\nREVIEW-LEVEL PREVIEW")
    print("=" * 90)
    preview_cols = [
        "domain", "rating", "predicted_sentiment", "predicted_star_rating",
        "discrepancy_status", "primary_issue", "issue_severity_level",
        "evidence_strength", "rating_penalty", "sentiment_penalty",
        "issue_penalty", "discrepancy_penalty", "domain_critical_penalty",
        "total_penalty", "trust_score", "risk_level", "reliability_level",
        "dominant_factor"
    ]
    preview_cols = [c for c in preview_cols if c in review_df.columns]
    print(review_df[preview_cols].head(15).to_string())

    print("\nTOP RISKY ENTITIES")
    print("=" * 90)
    summary_cols = [
        "domain", "entity_type", "entity_name", "total_reviews",
        "average_trust_score", "overall_risk_level",
        "top_issues", "mismatch_percentage"
    ]
    summary_cols = [c for c in summary_cols if c in entity_df.columns]
    print(entity_df[summary_cols].head(15).to_string())

    print("\nPHASE 10 COMPLETE")
    print("Outputs saved in:", output_dir)
    print("Important files:")
    print(output_dir / "multidomain_review_level_results.csv")
    print(output_dir / "multidomain_entity_level_summary.csv")
    print(output_dir / "phase10_risk_distribution.csv")
    print(output_dir / "phase10_risk_by_domain.csv")
    print(output_dir / "phase10_penalty_summary_by_domain.csv")
    print(output_dir / "phase10_top_risky_entities.csv")


if __name__ == "__main__":
    main()

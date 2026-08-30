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
    print("\nPHASE 9 PIPELINE EXECUTION TRACE")
    print("=" * 90)

    for i, item in enumerate(trace, start=1):
        print(f"{i}. {item['step']} -> {item['message']}")
        if "output" in item:
            print(f"   Output: {item['output']}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 test: DistilBERT + BERT discrepancy + MiniLM issue mining + RAG evidence retrieval."
    )
    parser.add_argument("--input", default="data/processed/combined_multidomain_reviews.csv")
    parser.add_argument("--sample-size", type=int, default=120)
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

    print("\nPHASE 9: RAG EVIDENCE RETRIEVAL PIPELINE TEST")
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
        use_rag=True,
        output_dir="outputs/phase9_rag_pipeline",
    )

    results = pipeline.analyze(df, save_outputs=True)
    review_df = results["review_level_results"]
    entity_df = results["entity_level_summary"]
    evidence = results.get("issue_evidence", {})

    print_trace(results["execution_trace"])

    print("\nMODEL USAGE CHECK")
    print("=" * 90)
    print("Sentiment model:")
    print(review_df["sentiment_model_used"].value_counts())
    print("\nDiscrepancy model:")
    print(review_df["discrepancy_model_used"].value_counts())
    print("\nIssue model:")
    print(review_df["issue_model_used"].value_counts())

    print("\nRAG EVIDENCE SUMMARY")
    print("=" * 90)
    if not evidence:
        print("No RAG evidence retrieved. This can happen if the sample has no detected issues.")
    else:
        for issue, records in evidence.items():
            print(f"\nIssue: {issue}")
            print(f"Evidence records retrieved: {len(records)}")
            for rec in records[:3]:
                print(
                    f"  Rank {rec.get('rank')} | backend={rec.get('retrieval_backend')} "
                    f"| score={rec.get('similarity_score')} | domain={rec.get('domain')} "
                    f"| risk={rec.get('risk_level')}"
                )
                print(f"  Evidence: {str(rec.get('evidence_text', ''))[:250]}")

    print("\nREVIEW-LEVEL RAG PREVIEW")
    print("=" * 90)
    preview_cols = [
        "domain",
        "rating",
        "predicted_sentiment",
        "predicted_star_rating",
        "discrepancy_status",
        "primary_issue",
        "issue_severity_level",
        "rag_similarity_score",
        "rag_evidence_text",
        "trust_score",
        "risk_level",
        "evidence_based_explanation",
    ]
    preview_cols = [c for c in preview_cols if c in review_df.columns]
    print(review_df[preview_cols].head(12).to_string())

    print("\nROWS WITH DETECTED ISSUES + RAG EVIDENCE")
    print("=" * 90)
    issue_rows = review_df[review_df["primary_issue"].fillna("no_issue") != "no_issue"].copy()
    if len(issue_rows) == 0:
        print("No detected issue rows found in this sample.")
    else:
        issue_cols = [
            "domain",
            "rating",
            "primary_issue",
            "issue_severity_level",
            "evidence_phrase",
            "rag_evidence_text",
            "rag_similarity_score",
            "trust_score",
            "risk_level",
        ]
        issue_cols = [c for c in issue_cols if c in issue_rows.columns]
        print(issue_rows[issue_cols].head(12).to_string())

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

    print("\nPHASE 9 PIPELINE TEST COMPLETE")
    print("Outputs saved in: outputs/phase9_rag_pipeline/")
    print("Vector index saved in: outputs/vector_index/")


if __name__ == "__main__":
    main()

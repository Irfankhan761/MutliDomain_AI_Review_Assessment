from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agents.domain_issue_taxonomy import GLOBAL_PROBLEM_CUES, get_domain_taxonomy
from services.local_model_registry import enforce_offline_mode, get_minilm_path, require_local_model

# These are the actual configured domain base thresholds used by the project.
ACTUAL_BASE_THRESHOLDS = {
    "mobile_app": 0.30,
    "hotel": 0.31,
    "ecommerce": 0.32,
    "restaurant": 0.31,
}


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def has_problem_cue(text: str) -> bool:
    t = str(text).lower()
    return any(cue in t for cue in GLOBAL_PROBLEM_CUES)


def build_issue_text(domain: str, issue: str, info: dict) -> str:
    label = info.get("label", issue)
    descriptions = " ".join(info.get("descriptions", []))
    keywords = " ".join(info.get("keywords", []))
    return f"{domain} issue: {label}. {descriptions}. Related terms: {keywords}"


def load_minilm():
    enforce_offline_mode()
    model_path = require_local_model(get_minilm_path(), "MiniLM semantic issue model")
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(str(model_path), local_files_only=True)
    except TypeError:
        model = SentenceTransformer(str(model_path))
    return model, model_path


def score_reviews(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    model, model_path = load_minilm()
    parts = []

    for domain, group in df.groupby("domain", sort=True):
        domain = str(domain)
        taxonomy = get_domain_taxonomy(domain)
        issue_names = list(taxonomy.keys())
        issue_texts = [build_issue_text(domain, issue, taxonomy[issue]) for issue in issue_names]

        issue_embeddings = model.encode(
            issue_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        review_embeddings = model.encode(
            group["review_text"].map(clean_text).tolist(),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        similarity = np.matmul(review_embeddings, issue_embeddings.T)
        top_idx = np.argmax(similarity, axis=1)
        top_score = similarity[np.arange(len(group)), top_idx]

        scored = group.copy()
        scored["minilm_top_issue"] = [issue_names[int(i)] for i in top_idx]
        scored["minilm_top_similarity"] = top_score.astype(float)
        scored["minilm_model_path"] = str(model_path)
        parts.append(scored)

    return pd.concat(parts, ignore_index=True)


def actual_system_prediction(row, base_threshold: float) -> tuple[str, float]:
    """Replicate the project's current IssueMiningAgent threshold logic.

    The configured domain threshold is the base value. The live system adjusts the
    effective cutoff by review rating/problem cues exactly as the application does:
      rating <= 2 -> max(0.26, base - 0.03)
      rating == 3 -> base
      high rating + problem cue -> base + 0.03
      high rating + no problem cue -> no_issue (base + 0.10 recorded)
    """
    rating = float(row.get("rating", 3) or 3)
    text = clean_text(row.get("review_text", ""))
    problem = has_problem_cue(text)

    if rating <= 2:
        threshold = max(0.26, float(base_threshold) - 0.03)
    elif rating == 3:
        threshold = float(base_threshold)
    elif problem:
        threshold = float(base_threshold) + 0.03
    else:
        threshold = float(base_threshold) + 0.10
        return "no_issue", threshold

    pred = row["minilm_top_issue"] if float(row["minilm_top_similarity"]) >= threshold else "no_issue"
    return pred, threshold


def add_actual_predictions(df: pd.DataFrame, base_threshold: float) -> pd.DataFrame:
    out = df.copy()
    pairs = out.apply(lambda r: actual_system_prediction(r, base_threshold), axis=1)
    out["prediction"] = [x[0] for x in pairs]
    out["effective_threshold_used"] = [float(x[1]) for x in pairs]
    out["configured_base_threshold"] = float(base_threshold)
    return out


def evaluate_predictions(df: pd.DataFrame) -> dict:
    y_true_issue = df["human_has_issue"].astype(int).to_numpy()
    y_pred_issue = (df["prediction"].astype(str) != "no_issue").astype(int).to_numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true_issue, y_pred_issue)),
        "precision": float(precision_score(y_true_issue, y_pred_issue, zero_division=0)),
        "recall": float(recall_score(y_true_issue, y_pred_issue, zero_division=0)),
        "f1": float(f1_score(y_true_issue, y_pred_issue, zero_division=0)),
    }
    return metrics


# def make_holdout(domain_df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
#     """Create an untouched holdout evaluation split. No threshold tuning is performed."""
#     _, test_df = train_test_split(
#         domain_df,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=domain_df["human_has_issue"].astype(int),
#     )
#     return test_df.reset_index(drop=True)


def save_binary_confusion_png(df: pd.DataFrame, path: Path, title: str):
    import matplotlib.pyplot as plt

    y_true = df["human_has_issue"].astype(int).to_numpy()
    y_pred = (df["prediction"].astype(str) != "no_issue").astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["No Issue", "Issue"])
    ax.set_yticks([0, 1], labels=["No Issue", "Issue"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground-truth label")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_dataset(df: pd.DataFrame):
    required = {"review_text", "domain", "rating", "human_has_issue", "human_issue", "human_verified"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    unverified = ~df["human_verified"].astype(str).str.lower().isin({"yes", "true", "1", "verified"})
    if int(unverified.sum()):
        raise RuntimeError(
            f"{int(unverified.sum())} rows are not marked human_verified=yes. "
            "Verify the evaluation labels before using the results as evidence."
        )

    for i, row in df.iterrows():
        domain = str(row["domain"])
        label = str(row["human_issue"])
        valid_labels = set(get_domain_taxonomy(domain).keys()) | {"no_issue"}
        if label not in valid_labels:
            raise ValueError(f"Invalid issue label at row {i}: domain={domain}, label={label}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniLM using only the project's actual configured thresholds; no threshold tuning."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "minilm_human_labelled_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "minilm_actual_threshold_evaluation",
    )
    # parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    validate_dataset(df)

    print(f"Loaded {len(df)} labelled benchmark reviews from: {args.input}")
    print(f"Configured thresholds: {ACTUAL_BASE_THRESHOLDS}")

    scored = score_reviews(df, args.batch_size)
    scored.to_csv(args.output_dir / "labelled_reviews_with_minilm_similarity.csv", index=False)

    summary_rows = []
    test_parts = []
    per_issue_rows = []

    for domain, domain_df in scored.groupby("domain", sort=True):
        domain = str(domain)
        base_threshold = ACTUAL_BASE_THRESHOLDS[domain]
        # test_df = make_holdout(domain_df, args.test_size, args.random_state)
        # test_df = add_actual_predictions(test_df, base_threshold)
        # test_df["split"] = "holdout_test"
        test_df = domain_df.copy().reset_index(drop=True)
        test_df = add_actual_predictions(test_df, base_threshold)
        test_df["split"] = "full_controlled_benchmark"
        metrics = evaluate_predictions(test_df)
        test_parts.append(test_df)

        summary_rows.append({
            "domain": domain,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

        labels = sorted(set(test_df["human_issue"].astype(str)) | set(test_df["prediction"].astype(str)))
        report = classification_report(
            test_df["human_issue"].astype(str),
            test_df["prediction"].astype(str),
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        for label, vals in report.items():
            if isinstance(vals, dict) and label not in {"accuracy", "macro avg", "weighted avg"}:
                per_issue_rows.append({"domain": domain, "issue": label, **vals})

        save_binary_confusion_png(
            test_df,
            args.output_dir / f"{domain}_binary_confusion_matrix.png",
            f"{domain}: MiniLM issue detection (threshold={base_threshold:.2f})",
        )

    summary = pd.DataFrame(summary_rows)
    test_all = pd.concat(test_parts, ignore_index=True)
    per_issue = pd.DataFrame(per_issue_rows)

    # Overall human-ground-truth holdout metrics across all four domains.
    # This is kept separate from the per-domain table so the dissertation chart
    # generator can report one clean overall MiniLM performance figure.
    overall_metrics = evaluate_predictions(test_all)
    y_true_all = test_all["human_has_issue"].astype(int).to_numpy()
    y_pred_all = (test_all["prediction"].astype(str) != "no_issue").astype(int).to_numpy()
    overall_cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    overall_metrics_df = pd.DataFrame([{
        "accuracy": overall_metrics["accuracy"],
        "precision": overall_metrics["precision"],
        "recall": overall_metrics["recall"],
        "f1": overall_metrics["f1"],
        "holdout_rows": int(len(test_all)),
    }])
    overall_cm_df = pd.DataFrame(
        overall_cm,
        index=["actual_no_issue", "actual_issue"],
        columns=["pred_no_issue", "pred_issue"],
    )

    summary.to_csv(args.output_dir / "minilm_actual_threshold_summary.csv", index=False)
    test_all.to_csv(args.output_dir / "minilm_actual_threshold_test_predictions.csv", index=False)
    per_issue.to_csv(args.output_dir / "minilm_actual_threshold_per_issue_metrics.csv", index=False)
    overall_metrics_df.to_csv(args.output_dir / "minilm_actual_threshold_overall_metrics.csv", index=False)
    overall_cm_df.to_csv(args.output_dir / "minilm_actual_threshold_overall_confusion_matrix.csv")

    # run_info = {
    #     "input": str(args.input),
    #     "output_dir": str(args.output_dir),
    #     "total_rows": int(len(df)),
    #     "test_size": args.test_size,
    #     "random_state": args.random_state,
    #     "threshold_tuning_performed": False,
    #     "actual_configured_base_thresholds": ACTUAL_BASE_THRESHOLDS,
    #     "evaluation": "Configured project thresholds evaluated on a holdout test split.",
    #     "note": (
    #         "The evaluator reproduces the project's live dynamic threshold adjustments around each configured "
    #         "domain base threshold. No candidate thresholds are searched or selected."
    #     ),
    run_info = {
    "input": str(args.input),
    "output_dir": str(args.output_dir),
    "total_rows": int(len(df)),
    "evaluated_rows": int(len(test_all)),
    "evaluation_mode": "full_controlled_benchmark",
    "random_state": args.random_state,
    "threshold_tuning_performed": False,
    "actual_configured_base_thresholds": ACTUAL_BASE_THRESHOLDS,
    "evaluation": (
        "Configured project thresholds evaluated on the complete "
        "controlled manually constructed benchmark."
    ),
    }
    (args.output_dir / "minilm_actual_threshold_run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    # report_lines = [
        # "MiniLM Actual-Threshold Evaluation",
        # "=" * 35,
        # "Threshold tuning: NOT PERFORMED",
        # f"Configured thresholds: {ACTUAL_BASE_THRESHOLDS}",
        # f"Holdout test size: {args.test_size:.0%}",
    report_lines = [
        "MiniLM Actual-Threshold Evaluation",
        "=" * 35,
        "Threshold tuning: NOT PERFORMED",
        f"Configured thresholds: {ACTUAL_BASE_THRESHOLDS}",
        f"Controlled benchmark rows evaluated: {len(test_all)}",
        "",
        "The table below reports only the thresholds already configured in the project.",
        "",
        "Overall controlled benchmark metrics across all domains:",
        pd.DataFrame([overall_metrics]).to_string(index=False),
        "",
        "Per-domain controlled benchmark metrics:",
        summary.to_string(index=False),
    ]
    (args.output_dir / "minilm_actual_threshold_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    # print("\nMiniLM evaluation using ACTUAL PROJECT THRESHOLDS (HOLDOUT TEST split):")
    print("\nMiniLM evaluation using ACTUAL PROJECT THRESHOLDS (FULL CONTROLLED BENCHMARK):")
    print("\nOverall controlled benchmark metrics:")
    print(pd.DataFrame([overall_metrics]).to_string(index=False))
    print("\nPer-domain controlled benchmark metrics:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

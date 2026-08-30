"""
Phase 15 — Evaluation and Results

Reads the final orchestrator/pipeline outputs and creates dissertation-ready evaluation files:
- sentiment metrics
- rating prediction and discrepancy metrics
- risk distribution
- issue mining distribution
- RAG evidence coverage
- domain-level comparison
- top risky entities
- charts
- Chapter 5 results draft

Recommended:
    python scripts/run_phase15_evaluation_results.py --latest
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = df[col].fillna(0)
    return df


def find_latest_run(root: Path) -> Path:
    runs_root = root / "outputs" / "final_orchestrator_runs"
    if not runs_root.exists():
        raise FileNotFoundError("outputs/final_orchestrator_runs folder not found.")

    candidates = []
    for p in runs_root.iterdir():
        review_file = p / "analysis_pipeline" / "multidomain_review_level_results.csv"
        if p.is_dir() and review_file.exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No completed final orchestrator run found.")

    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def resolve_paths(args, root: Path) -> Tuple[Path, Optional[Path], Path]:
    if args.review_csv:
        review_csv = Path(args.review_csv)
        entity_csv = Path(args.entity_csv) if args.entity_csv else None
        return review_csv, entity_csv, review_csv.parent

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(root)
    review_csv = run_dir / "analysis_pipeline" / "multidomain_review_level_results.csv"
    entity_csv = run_dir / "analysis_pipeline" / "multidomain_entity_level_summary.csv"
    return review_csv, entity_csv if entity_csv.exists() else None, run_dir


def rating_to_sentiment(value) -> str:
    try:
        r = float(value)
    except Exception:
        return "neutral"
    if r <= 2:
        return "negative"
    if r < 4:
        return "neutral"
    return "positive"


def normalise_sentiment(value) -> str:
    value = str(value).lower().strip()
    if "neg" in value:
        return "negative"
    if "neu" in value:
        return "neutral"
    if "pos" in value:
        return "positive"
    return "neutral"


def normalise_rating(value) -> Optional[int]:
    try:
        r = float(value)
        if math.isnan(r):
            return None
        return int(round(max(1, min(5, r))))
    except Exception:
        return None


def prf(conf: pd.DataFrame, label: str) -> Dict[str, float]:
    tp = float(conf.loc[label, label])
    fp = float(conf[label].sum() - tp)
    fn = float(conf.loc[label].sum() - tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_sentiment(df: pd.DataFrame):
    work = df.copy()
    if "sentiment_label" in work.columns:
        work["actual_sentiment_eval"] = work["sentiment_label"].apply(normalise_sentiment)
        label_source = "existing sentiment_label column"
    else:
        work["actual_sentiment_eval"] = work["rating"].apply(rating_to_sentiment)
        label_source = "rating-derived sentiment label"

    if "predicted_sentiment" not in work.columns:
        work["predicted_sentiment"] = "neutral"

    work["predicted_sentiment_eval"] = work["predicted_sentiment"].apply(normalise_sentiment)
    conf = pd.crosstab(work["actual_sentiment_eval"], work["predicted_sentiment_eval"], rownames=["actual"], colnames=["predicted"])

    for label in LABELS:
        if label not in conf.index:
            conf.loc[label] = 0
        if label not in conf.columns:
            conf[label] = 0
    conf = conf.loc[LABELS, LABELS]

    total = float(conf.to_numpy().sum())
    correct = float(np.trace(conf.to_numpy()))
    accuracy = correct / total if total else 0.0

    metric_rows = []
    f1s = []
    for label in LABELS:
        item = prf(conf, label)
        f1s.append(item["f1"])
        metric_rows.append({
            "label": label,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1"],
            "support": int(conf.loc[label].sum()),
        })

    summary = {
        "actual_label_source": label_source,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "total_evaluated_reviews": int(total),
    }
    return conf, pd.DataFrame(metric_rows), summary


def evaluate_rating(df: pd.DataFrame):
    if "rating" not in df.columns or "predicted_star_rating" not in df.columns:
        return pd.DataFrame(), {"available": False, "reason": "rating or predicted_star_rating column missing"}

    work = df.copy()
    work["actual_rating_eval"] = work["rating"].apply(normalise_rating)
    work["predicted_rating_eval"] = work["predicted_star_rating"].apply(normalise_rating)
    valid = work.dropna(subset=["actual_rating_eval", "predicted_rating_eval"]).copy()

    if valid.empty:
        return pd.DataFrame(), {"available": False, "reason": "no valid rating rows"}

    valid["rating_error"] = (valid["actual_rating_eval"] - valid["predicted_rating_eval"]).abs()
    valid["exact_match"] = valid["rating_error"] == 0
    valid["within_one"] = valid["rating_error"] <= 1

    conf = pd.crosstab(valid["actual_rating_eval"], valid["predicted_rating_eval"], rownames=["actual_rating"], colnames=["predicted_rating"])
    for r in [1, 2, 3, 4, 5]:
        if r not in conf.index:
            conf.loc[r] = 0
        if r not in conf.columns:
            conf[r] = 0
    conf = conf.loc[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    mismatch = None
    if "discrepancy_status" in valid.columns:
        mismatch = round(float((valid["discrepancy_status"].astype(str) == "mismatched").mean() * 100), 2)

    summary = {
        "available": True,
        "total_evaluated_reviews": int(len(valid)),
        "exact_rating_accuracy": round(float(valid["exact_match"].mean()), 4),
        "within_one_rating_accuracy": round(float(valid["within_one"].mean()), 4),
        "mean_absolute_error": round(float(valid["rating_error"].mean()), 4),
        "discrepancy_rate_percent": mismatch,
    }
    return conf, summary


def domain_performance(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "domain" not in work.columns:
        work["domain"] = "unknown"

    rows = []
    for domain, part in work.groupby("domain"):
        row = {"domain": domain, "reviews": int(len(part))}
        if "rating" in part.columns:
            row["average_rating"] = round(float(pd.to_numeric(part["rating"], errors="coerce").mean()), 2)
        if "trust_score" in part.columns:
            trust = pd.to_numeric(part["trust_score"], errors="coerce")
            row["average_trust_score"] = round(float(trust.mean()), 2)
            row["min_trust_score"] = round(float(trust.min()), 2)
            row["max_trust_score"] = round(float(trust.max()), 2)
        if "risk_level" in part.columns:
            total = max(len(part), 1)
            row["high_risk_percent"] = round(float((part["risk_level"].astype(str) == "high_risk").sum() / total * 100), 2)
            row["medium_risk_percent"] = round(float((part["risk_level"].astype(str) == "medium_risk").sum() / total * 100), 2)
            row["low_risk_percent"] = round(float((part["risk_level"].astype(str) == "low_risk").sum() / total * 100), 2)
        if "primary_issue" in part.columns:
            total = max(len(part), 1)
            row["issue_detection_percent"] = round(float((part["primary_issue"].astype(str) != "no_issue").sum() / total * 100), 2)
            vc = part["primary_issue"].astype(str).value_counts()
            row["top_issue"] = vc.index[0] if len(vc) else "none"
        if "discrepancy_status" in part.columns:
            total = max(len(part), 1)
            row["mismatch_percent"] = round(float((part["discrepancy_status"].astype(str) == "mismatched").sum() / total * 100), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("domain")


def count_table(df: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    out = df[column].astype(str).value_counts().reset_index()
    out.columns = [name, "count"]
    out["percentage"] = round(out["count"] / max(len(df), 1) * 100, 2)
    return out


def cross_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    if row not in df.columns or col not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[row].astype(str), df[col].astype(str)).reset_index()


def rag_metrics(df: pd.DataFrame) -> pd.DataFrame:
    issue_df = df.copy()
    if "primary_issue" in issue_df.columns:
        issue_df = issue_df[issue_df["primary_issue"].astype(str) != "no_issue"].copy()

    total_issue_rows = len(issue_df)
    evidence_col = None
    for candidate in ["rag_evidence_text", "evidence_phrase"]:
        if candidate in issue_df.columns:
            evidence_col = candidate
            break

    if evidence_col:
        coverage = (issue_df[evidence_col].astype(str).str.strip() != "").mean() * 100 if total_issue_rows else 0
    else:
        coverage = 0

    mean_similarity = "not_available"
    if "rag_similarity_score" in issue_df.columns:
        sim = pd.to_numeric(issue_df["rag_similarity_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
        if not pd.isna(sim):
            mean_similarity = round(float(sim), 4)

    return pd.DataFrame([
        {
            "metric": "issue_rows_evaluated",
            "value": int(total_issue_rows),
            "description": "Rows where an issue was detected and RAG evidence could be checked.",
        },
        {
            "metric": "issue_rows_with_evidence_percent",
            "value": round(float(coverage), 2),
            "description": "Percentage of detected issue rows with supporting evidence text.",
        },
        {
            "metric": "mean_rag_similarity_score",
            "value": mean_similarity,
            "description": "Mean similarity score for retrieved evidence where available.",
        },
    ])


def top_risky_entities(entity_df: Optional[pd.DataFrame], review_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if entity_df is not None and not entity_df.empty:
        work = entity_df.copy()
        if "average_trust_score" in work.columns:
            work["average_trust_score"] = pd.to_numeric(work["average_trust_score"], errors="coerce")
            work = work.sort_values("average_trust_score", ascending=True)
        cols = [
            "domain", "entity_type", "entity_name", "total_reviews", "average_rating",
            "average_trust_score", "overall_risk_level", "high_risk_percentage",
            "mismatch_percentage", "top_issues", "entity_recommendation"
        ]
        return work[[c for c in cols if c in work.columns]].head(top_n)

    fallback = review_df.copy()
    for col in ["domain", "entity_name"]:
        if col not in fallback.columns:
            fallback[col] = "unknown"
    if "trust_score" not in fallback.columns:
        fallback["trust_score"] = 0
    return fallback.groupby(["domain", "entity_name"]).agg(
        total_reviews=("entity_name", "count"),
        average_trust_score=("trust_score", "mean"),
    ).reset_index().sort_values("average_trust_score").head(top_n)


def save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# def make_charts(out_dir: Path, tables: Dict[str, pd.DataFrame]):
#     try:
#         import matplotlib.pyplot as plt
#     except Exception:
#         print("matplotlib not installed. Charts skipped.")
#         return

#     charts = ensure_dir(out_dir / "charts")


def make_charts(out_dir: Path, tables: Dict[str, pd.DataFrame]):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("matplotlib not installed. Charts skipped.")
        return

    charts = ensure_dir(out_dir / "charts")

    def clean_label(value):
        return str(value).replace("_", " ").title()

    def add_labels(ax, bars, suffix="", decimals=2):
        """Show the actual value above every bar."""
        for bar in bars:
            value = bar.get_height()

            if pd.isna(value):
                continue

            if suffix == "%":
                text = f"{value:.{decimals}f}%"
            elif float(value).is_integer():
                text = f"{int(value):,}"
            else:
                text = f"{value:.{decimals}f}"

            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    def save_single_bar(
        df,
        x,
        y,
        title,
        filename,
        ylabel,
        percent=False,
        limit=None,
        rotate=0,
    ):
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return

        p = df.copy()
        if limit is not None:
            p = p.head(limit)

        labels = [clean_label(v) for v in p[x]]
        values = pd.to_numeric(p[y], errors="coerce").fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        ax.tick_params(axis="x", rotation=rotate)

        add_labels(ax, bars, suffix="%" if percent else "")

        max_value = float(values.max()) if len(values) else 0
        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)

        fig.tight_layout()
        fig.savefig(charts / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------
    # Figure 5.8 — Average Trust Score by Domain
    # ---------------------------------------------------------
    domain_df = tables.get("domain_performance")

    save_single_bar(
        domain_df,
        "domain",
        "average_trust_score",
        "Average Trust Score by Domain",
        "trust_by_domain.png",
        "Average Trust Score",
    )

    # ---------------------------------------------------------
    # Figure 5.9 — High-Risk Review Percentage by Domain
    # ---------------------------------------------------------
    save_single_bar(
        domain_df,
        "domain",
        "high_risk_percent",
        "High-Risk Review Percentage by Domain",
        "high_risk_by_domain.png",
        "High-Risk Reviews (%)",
        percent=True,
    )

    # ---------------------------------------------------------
    # Figure 5.10 — Rating-Review Mismatch Percentage
    # ---------------------------------------------------------
    save_single_bar(
        domain_df,
        "domain",
        "mismatch_percent",
        "Rating-Review Mismatch Percentage by Domain",
        "mismatch_by_domain.png",
        "Mismatch (%)",
        percent=True,
    )

    # ---------------------------------------------------------
    # Figure 5.11 — Issue Detection Percentage
    # ---------------------------------------------------------
    save_single_bar(
        domain_df,
        "domain",
        "issue_detection_percent",
        "Issue Detection Percentage by Domain",
        "issue_detection_by_domain.png",
        "Reviews with Detected Issue (%)",
        percent=True,
    )

    # ---------------------------------------------------------
    # Figure 5.12 — Overall Risk Level Distribution
    # count + percentage on each bar
    # ---------------------------------------------------------
    risk_df = tables.get("risk_distribution")

    if risk_df is not None and not risk_df.empty:
        risk_df = risk_df.copy()

        order = ["low_risk", "medium_risk", "high_risk"]
        risk_df["_order"] = risk_df["risk_level"].map(
            {name: i for i, name in enumerate(order)}
        )
        risk_df = risk_df.sort_values("_order")

        labels = [clean_label(v) for v in risk_df["risk_level"]]
        counts = pd.to_numeric(risk_df["count"], errors="coerce").fillna(0)
        percentages = pd.to_numeric(
            risk_df["percentage"], errors="coerce"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, counts)

        ax.set_title(
            "Overall Risk Level Distribution",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_ylabel("Number of Reviews", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.30)

        for bar, count, pct in zip(bars, counts, percentages):
            ax.annotate(
                f"{int(count):,}\n({pct:.2f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, count),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylim(0, counts.max() * 1.20)

        fig.tight_layout()
        fig.savefig(
            charts / "risk_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---------------------------------------------------------
    # Figure 5.13 — Risk Level Composition by Domain
    # ---------------------------------------------------------
    risk_domain = tables.get("risk_by_domain")

    if risk_domain is not None and not risk_domain.empty:
        p = risk_domain.copy()

        domains = [clean_label(v) for v in p["domain"]]

        low = (
            pd.to_numeric(p["low_risk"], errors="coerce").fillna(0)
            if "low_risk" in p.columns
            else pd.Series([0] * len(p))
        )
        medium = (
            pd.to_numeric(p["medium_risk"], errors="coerce").fillna(0)
            if "medium_risk" in p.columns
            else pd.Series([0] * len(p))
        )
        high = (
            pd.to_numeric(p["high_risk"], errors="coerce").fillna(0)
            if "high_risk" in p.columns
            else pd.Series([0] * len(p))
        )

        x = np.arange(len(domains))
        width = 0.25

        fig, ax = plt.subplots(figsize=(11, 6))

        bars_low = ax.bar(x - width, low, width, label="Low Risk")
        bars_medium = ax.bar(x, medium, width, label="Medium Risk")
        bars_high = ax.bar(x + width, high, width, label="High Risk")

        ax.set_title(
            "Risk Level Composition by Domain",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_ylabel("Number of Reviews", fontweight="bold")
        ax.set_xticks(x, domains)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.30)

        add_labels(ax, bars_low)
        add_labels(ax, bars_medium)
        add_labels(ax, bars_high)

        highest = max(low.max(), medium.max(), high.max())
        ax.set_ylim(0, highest * 1.20 if highest > 0 else 1)

        fig.tight_layout()
        fig.savefig(
            charts / "risk_by_domain.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---------------------------------------------------------
    # Figure 5.14 — Issue Severity Distribution
    # ---------------------------------------------------------
    severity_df = tables.get("severity_distribution")

    if severity_df is not None and not severity_df.empty:
        severity_df = severity_df.copy()

        order = ["none", "low", "medium", "high"]
        severity_df["_order"] = severity_df[
            "issue_severity_level"
        ].map({name: i for i, name in enumerate(order)})

        severity_df = severity_df.sort_values("_order")

        labels = [
            clean_label(v)
            for v in severity_df["issue_severity_level"]
        ]
        counts = pd.to_numeric(
            severity_df["count"], errors="coerce"
        ).fillna(0)

        percentages = pd.to_numeric(
            severity_df["percentage"], errors="coerce"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, counts)

        ax.set_title(
            "Issue Severity Distribution",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_ylabel("Number of Reviews", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.30)

        for bar, count, pct in zip(bars, counts, percentages):
            ax.annotate(
                f"{int(count):,}\n({pct:.2f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, count),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylim(0, counts.max() * 1.20)

        fig.tight_layout()
        fig.savefig(
            charts / "severity_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---------------------------------------------------------
    # Extra — Top detected issues
    # ---------------------------------------------------------
    issue_df = tables.get("issue_distribution")

    if issue_df is not None and not issue_df.empty:
        issue_df = issue_df[
            ~issue_df["primary_issue"]
            .astype(str)
            .isin(["no_issue", "none"])
        ].copy()

        issue_df = issue_df.sort_values(
            "count", ascending=False
        ).head(12)

        labels = [
            clean_label(v) for v in issue_df["primary_issue"]
        ]
        counts = pd.to_numeric(
            issue_df["count"], errors="coerce"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(11, 7))

        y = np.arange(len(labels))
        bars = ax.barh(y, counts)

        ax.set_yticks(y, labels)
        ax.invert_yaxis()

        ax.set_title(
            "Top Detected Issues Across Reviews",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel("Number of Reviews", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.30)

        for bar, value in zip(bars, counts):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {int(value):,}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(
            charts / "issue_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ---------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------
    def matrix(df: pd.DataFrame, title: str, filename: str):
        if df is None or df.empty:
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(df.to_numpy())

        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xticks(range(len(df.columns)), df.columns)
        ax.set_yticks(range(len(df.index)), df.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{int(df.iloc[i, j]):,}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    matrix(
        tables.get("sentiment_confusion_matrix"),
        "Sentiment Confusion Matrix",
        "sentiment_confusion_matrix.png",
    )

    matrix(
        tables.get("rating_confusion_matrix"),
        "Rating Prediction Confusion Matrix",
        "rating_confusion_matrix.png",
    )

    def bar(df: pd.DataFrame, x: str, y: str, title: str, filename: str, limit: int = 12):
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return
        p = df.head(limit).copy()
        plt.figure(figsize=(11, 6))
        plt.bar(p[x].astype(str), p[y])
        plt.title(title)
        plt.xlabel(x.replace("_", " ").title())
        plt.ylabel(y.replace("_", " ").title())
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(charts / filename, dpi=160)
        plt.close()

    bar(tables.get("risk_distribution"), "risk_level", "count", "Risk Level Distribution", "risk_distribution.png")
    bar(tables.get("issue_distribution"), "primary_issue", "count", "Detected Issue Distribution", "issue_distribution.png")
    bar(tables.get("severity_distribution"), "issue_severity_level", "count", "Issue Severity Distribution", "severity_distribution.png")
    bar(tables.get("domain_performance"), "domain", "average_trust_score", "Average Trust Score by Domain", "trust_by_domain.png")

    def matrix(df: pd.DataFrame, title: str, filename: str):
        if df is None or df.empty:
            return
        plt.figure(figsize=(7, 6))
        plt.imshow(df.to_numpy())
        plt.title(title)
        plt.xticks(range(len(df.columns)), df.columns)
        plt.yticks(range(len(df.index)), df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, int(df.iloc[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(charts / filename, dpi=160)
        plt.close()

    matrix(tables.get("sentiment_confusion_matrix"), "Sentiment Confusion Matrix", "sentiment_confusion_matrix.png")
    matrix(tables.get("rating_confusion_matrix"), "Rating Prediction Confusion Matrix", "rating_confusion_matrix.png")


def build_reports(out_dir: Path, review_df: pd.DataFrame, entity_df: Optional[pd.DataFrame], sentiment_summary: Dict, rating_summary: Dict, domain_df: pd.DataFrame, tables: Dict[str, pd.DataFrame]):
    total_reviews = len(review_df)
    total_entities = len(entity_df) if entity_df is not None else "not_available"
    avg_trust = "not_available"
    high_risk_pct = "not_available"

    if "trust_score" in review_df.columns:
        avg_trust = round(float(pd.to_numeric(review_df["trust_score"], errors="coerce").mean()), 2)
    if "risk_level" in review_df.columns:
        high_risk_pct = round(float((review_df["risk_level"].astype(str) == "high_risk").mean() * 100), 2)

    risk_md = tables.get("risk_distribution", pd.DataFrame()).to_markdown(index=False) if not tables.get("risk_distribution", pd.DataFrame()).empty else "Not available."
    issue_md = tables.get("issue_distribution", pd.DataFrame()).head(15).to_markdown(index=False) if not tables.get("issue_distribution", pd.DataFrame()).empty else "Not available."
    domain_md = domain_df.to_markdown(index=False) if not domain_df.empty else "Not available."
    rag_md = tables.get("rag_metrics", pd.DataFrame()).to_markdown(index=False) if not tables.get("rag_metrics", pd.DataFrame()).empty else "Not available."
    top_md = tables.get("top_risky_entities", pd.DataFrame()).head(10).to_markdown(index=False) if not tables.get("top_risky_entities", pd.DataFrame()).empty else "Not available."

    md = f"""# Phase 15 Evaluation and Results Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Used for Evaluation

- Total review-level rows: **{total_reviews}**
- Total entity-level rows: **{total_entities}**
- Average trust score: **{avg_trust}**
- High-risk review percentage: **{high_risk_pct}%**

## 2. Transformer Sentiment Agent Evaluation

The sentiment agent was evaluated by comparing predicted sentiment with the available evaluation label.

- Label source: **{sentiment_summary.get('actual_label_source')}**
- Accuracy: **{sentiment_summary.get('accuracy')}**
- Macro F1-score: **{sentiment_summary.get('macro_f1')}**
- Evaluated rows: **{sentiment_summary.get('total_evaluated_reviews')}**

This is an internal consistency evaluation because the available label is based on dataset rating/sentiment fields rather than a separately hand-labelled test set.

## 3. Rating Prediction and Discrepancy Agent Evaluation

- Exact rating accuracy: **{rating_summary.get('exact_rating_accuracy', 'not_available')}**
- Within-one-star accuracy: **{rating_summary.get('within_one_rating_accuracy', 'not_available')}**
- Mean absolute error: **{rating_summary.get('mean_absolute_error', 'not_available')}**
- Rating-review discrepancy rate: **{rating_summary.get('discrepancy_rate_percent', 'not_available')}%**

The discrepancy metric is important because it identifies cases where the written review and the numerical star rating do not communicate the same reliability signal.

## 4. Domain-Level Results

{domain_md}

## 5. Risk Scoring Results

{risk_md}

## 6. Semantic Issue Mining Results

{issue_md}

## 7. RAG Evidence Retrieval Results

{rag_md}

## 8. Top Risky Entities

{top_md}

## 9. Overall Interpretation

The evaluation shows that the system is not only a single sentiment classifier. It combines Transformer sentiment classification, BERT-based rating prediction, discrepancy detection, MiniLM semantic issue mining, RAG evidence retrieval, risk scoring and explainable reporting. This makes the result more suitable for review trust assessment because the final decision is based on multiple coordinated agent outputs rather than one baseline model.
"""
    (out_dir / "phase15_evaluation_report.md").write_text(md, encoding="utf-8")

    chapter5 = f"""Phase 15 Results Summary for Chapter 5

The final Sequentical Modular Pipeline system was evaluated using the completed review-level and entity-level pipeline outputs. A total of {total_reviews} review-level records were assessed. The system generated sentiment predictions, star-rating predictions, discrepancy flags, semantic issue labels, RAG evidence fields, trust scores, risk levels and final recommendations.

The Transformer Sentiment Agent achieved an accuracy of {sentiment_summary.get('accuracy')} and a macro F1-score of {sentiment_summary.get('macro_f1')} when compared with the internal evaluation label. This demonstrates that the sentiment module provides a useful signal for the wider trust assessment process. However, this result should be treated as internal validation because the comparison label is derived from available dataset fields.

The Rating Prediction and Discrepancy Agent achieved an exact rating accuracy of {rating_summary.get('exact_rating_accuracy', 'not_available')} and a within-one-star accuracy of {rating_summary.get('within_one_rating_accuracy', 'not_available')}. The mean absolute error was {rating_summary.get('mean_absolute_error', 'not_available')}, while the rating-review discrepancy rate was {rating_summary.get('discrepancy_rate_percent', 'not_available')}%. This shows that the system can detect cases where the text meaning and the numerical rating may not be fully aligned.

The risk scoring stage produced an average trust score of {avg_trust}. High-risk reviews represented {high_risk_pct}% of the evaluated records. The semantic issue mining and RAG stages strengthened explainability by identifying domain-specific issues and retrieving supporting evidence.

Overall, the results demonstrate the value of the Modular design. Each specialised agent contributes a separate analytical signal, and the orchestrator combines these outputs into an explainable review trust assessment.
"""
    (out_dir / "phase15_chapter5_results_summary.txt").write_text(chapter5, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 15 Evaluation and Results")
    parser.add_argument("--latest", action="store_true", help="Use the latest final orchestrator run.")
    parser.add_argument("--run-dir", default=None, help="Specific final_orchestrator_runs folder.")
    parser.add_argument("--review-csv", default=None, help="Direct review-level result CSV path.")
    parser.add_argument("--entity-csv", default=None, help="Direct entity-level summary CSV path.")
    parser.add_argument("--output-dir", default=None, help="Output folder.")
    parser.add_argument("--top-n", type=int, default=20, help="Top risky entities to save.")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation.")
    args = parser.parse_args()

    root = project_root()
    review_csv, entity_csv, source_dir = resolve_paths(args, root)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else root / "outputs" / "phase15_evaluation_results" / stamp())

    print("\nPHASE 15: EVALUATION AND RESULTS")
    print("=" * 88)
    print(f"Review-level input: {review_csv}")
    print(f"Entity-level input: {entity_csv if entity_csv else 'not_available'}")
    print(f"Output folder: {out_dir}")

    review_df = read_csv(review_csv)
    entity_df = read_csv(entity_csv) if entity_csv and Path(entity_csv).exists() else None

    print("\nINPUT CHECK")
    print("=" * 88)
    print(f"Review rows: {len(review_df)}")
    print(f"Review columns: {list(review_df.columns)}")
    print(f"Entity rows: {len(entity_df) if entity_df is not None else 'not_available'}")

    sent_conf, sent_metrics, sent_summary = evaluate_sentiment(review_df)
    rating_conf, rating_summary = evaluate_rating(review_df)
    domain_df = domain_performance(review_df)

    tables = {
        "sentiment_confusion_matrix": sent_conf,
        "sentiment_metrics": sent_metrics,
        "rating_confusion_matrix": rating_conf,
        "domain_performance": domain_df,
        "risk_distribution": count_table(review_df, "risk_level", "risk_level"),
        "risk_by_domain": cross_table(review_df, "domain", "risk_level"),
        "issue_distribution": count_table(review_df, "primary_issue", "primary_issue"),
        "issue_by_domain": cross_table(review_df, "domain", "primary_issue"),
        "severity_distribution": count_table(review_df, "issue_severity_level", "issue_severity_level"),
        "severity_by_domain": cross_table(review_df, "domain", "issue_severity_level"),
        "dominant_factor_distribution": count_table(review_df, "dominant_factor", "dominant_factor"),
        "rag_metrics": rag_metrics(review_df),
        "top_risky_entities": top_risky_entities(entity_df, review_df, args.top_n),
    }

    print("\nEVALUATION SUMMARY")
    print("=" * 88)
    print("Sentiment:")
    print(json.dumps(sent_summary, indent=2))
    print("\nRating / discrepancy:")
    print(json.dumps(rating_summary, indent=2))
    print("\nDomain performance:")
    print(domain_df.to_string(index=False))

    summary_json = {
        "source_review_csv": str(review_csv),
        "source_entity_csv": str(entity_csv) if entity_csv else None,
        "source_folder": str(source_dir),
        "review_rows": int(len(review_df)),
        "entity_rows": int(len(entity_df)) if entity_df is not None else None,
        "sentiment_summary": sent_summary,
        "rating_discrepancy_summary": rating_summary,
    }
    save_json(out_dir / "phase15_evaluation_summary.json", summary_json)

    print("\nSAVING OUTPUT FILES")
    print("=" * 88)
    for name, table in tables.items():
        if table is not None and isinstance(table, pd.DataFrame) and not table.empty:
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=True if "confusion_matrix" in name else False)
            print(f"Saved: {path}")

    if not args.no_charts:
        make_charts(out_dir, tables)
        print(f"Charts saved to: {out_dir / 'charts'}")

    build_reports(out_dir, review_df, entity_df, sent_summary, rating_summary, domain_df, tables)
    print(f"Saved: {out_dir / 'phase15_evaluation_report.md'}")
    print(f"Saved: {out_dir / 'phase15_chapter5_results_summary.txt'}")

    print("\nPHASE 15 COMPLETE")
    print("=" * 88)
    print(f"Evaluation outputs saved in: {out_dir}")
    print("\nMost important files:")
    print(out_dir / "phase15_evaluation_report.md")
    print(out_dir / "phase15_chapter5_results_summary.txt")
    print(out_dir / "domain_performance.csv")
    print(out_dir / "top_risky_entities.csv")


if __name__ == "__main__":
    main()

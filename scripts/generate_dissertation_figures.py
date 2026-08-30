from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR_NAME = "01_model_performance"
OPERATIONAL_DIR_NAME = "02_operational_results"
TABLES_DIR_NAME = "03_tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def newest_existing(paths: list[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def latest_subdir(parent: Path) -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"Folder not found: {parent}")
    dirs = [p for p in parent.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run folders found inside: {parent}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_percent(value) -> float:
    x = float(value)
    return x * 100.0 if abs(x) <= 1.5 else x


def clean_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def add_value_labels(ax, bars, suffix="", decimals=1):
    for bar in bars:
        h = float(bar.get_height())
        if np.isnan(h):
            continue
        if abs(h - round(h)) < 1e-9 and not suffix:
            text = f"{int(round(h)):,}"
        else:
            text = f"{h:.{decimals}f}{suffix}"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def save_metric_bars(labels, values, title, output_path: Path, note: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    add_value_labels(ax, bars, suffix="%", decimals=2)
    ax.set_ylim(0, max(100, max(values) * 1.16 if values else 100))
    ax.set_ylabel("Performance (%)")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=9)
        fig.subplots_adjust(bottom=0.16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_heatmap(matrix, labels, title, output_path: Path, xlabel, ylabel):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = f"{int(val):,}" if float(val).is_integer() else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_simple_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    percent_values: bool = False,
    percent_of_total: bool = False,
    top_n: Optional[int] = None,
):
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    chart = df.copy()
    chart[y_col] = pd.to_numeric(chart[y_col], errors="coerce").fillna(0)
    if top_n:
        chart = chart.sort_values(y_col, ascending=False).head(top_n)

    labels = [clean_label(x) for x in chart[x_col].astype(str)]
    values = chart[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, values)

    total = float(values.sum()) if percent_of_total else None
    for bar, value in zip(bars, values):
        if percent_of_total and total and total > 0:
            text = f"{int(value):,}\n({value/total*100:.1f}%)"
        elif percent_values:
            text = f"{value:.2f}%"
        elif abs(value - round(value)) < 1e-9:
            text = f"{int(round(value)):,}"
        else:
            text = f"{value:.2f}"

        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.tick_params(axis="x", rotation=20)

    vmax = max(values) if len(values) else 0
    if vmax > 0:
        ax.set_ylim(0, vmax * 1.22)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grouped_risk_chart(df: pd.DataFrame, output_path: Path):
    if df is None or df.empty or "domain" not in df.columns:
        return

    candidate_groups = [
        ["low_risk", "medium_risk", "high_risk"],
        ["low", "medium", "high"],
        ["Low Risk", "Medium Risk", "High Risk"],
    ]
    cols = next((g for g in candidate_groups if all(c in df.columns for c in g)), None)

    if cols is None:
        # Try any columns containing low/medium/high.
        lower_map = {c.lower(): c for c in df.columns}
        found = []
        for keyword in ["low", "medium", "high"]:
            match = next((orig for low, orig in lower_map.items() if keyword in low and "risk" in low), None)
            if match:
                found.append(match)
        if len(found) == 3:
            cols = found
        else:
            return

    categories = [clean_label(x) for x in df["domain"].astype(str)]
    x = np.arange(len(categories))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for idx, col in enumerate(cols):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=clean_label(col))
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{int(val):,}",
                (bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("Risk Level Composition by Domain", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Reviews")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_table(source: Path, tables_dir: Path):
    if source.exists():
        shutil.copy2(source, tables_dir / source.name)


# ---------------------------------------------------------------------------
# Model-performance sources
# ---------------------------------------------------------------------------

def generate_distilbert(model_dir: Path, tables_dir: Path, manifest: dict):
    reports = PROJECT_ROOT / "outputs" / "reports"
    metrics_path = reports / "phase6_distilbert_metrics.json"
    conf_path = reports / "phase6_distilbert_confusion_matrix.csv"

    if not metrics_path.exists() or not conf_path.exists():
        raise FileNotFoundError(
            "Final DistilBERT validation reports are missing. "
            "Run scripts/train_distilbert_sentiment.py first."
        )

    metrics = read_json(metrics_path)
    accuracy = as_percent(metrics["eval_accuracy"])
    macro_f1 = as_percent(metrics["eval_f1_macro"])
    weighted_f1 = as_percent(metrics["eval_f1_weighted"])
    val_loss = float(metrics["eval_loss"])

    save_metric_bars(
        ["Accuracy", "Macro F1", "Weighted F1"],
        [accuracy, macro_f1, weighted_f1],
        "DistilBERT Sentiment — Held-Out Validation Performance",
        model_dir / "01_distilbert_validation_performance.png",
        note=f"Held-out validation split. Validation loss = {val_loss:.4f}.",
    )

    conf = pd.read_csv(conf_path, index_col=0)
    save_confusion_heatmap(
        conf.to_numpy(dtype=float),
        ["Negative", "Neutral", "Positive"],
        "DistilBERT Sentiment — Validation Confusion Matrix",
        model_dir / "02_distilbert_validation_confusion_matrix.png",
        "Predicted sentiment",
        "Actual sentiment",
    )

    copy_table(conf_path, tables_dir)

    manifest["distilbert"] = {
        "metrics_source": str(metrics_path),
        "confusion_source": str(conf_path),
        "accuracy_percent": accuracy,
        "macro_f1_percent": macro_f1,
        "weighted_f1_percent": weighted_f1,
        "validation_loss": val_loss,
    }


def choose_rating_files() -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "outputs" / "reports"

    metric_candidates = [
        reports / "phase7_rating_bert_metrics.json",
        reports / "phase7_rating_bert_exp2_metrics.json",
    ]
    metrics_path = newest_existing(metric_candidates)
    if metrics_path is None:
        raise FileNotFoundError(
            "No held-out Rating-BERT metrics found. Expected "
            "phase7_rating_bert_metrics.json (preferred final name) or "
            "phase7_rating_bert_exp2_metrics.json."
        )

    if "exp2" in metrics_path.name:
        conf_candidates = [
            reports / "phase7_rating_bert_exp2_confusion_matrix.csv",
            reports / "phase7_rating_bert_confusion_matrix.csv",
        ]
    else:
        conf_candidates = [
            reports / "phase7_rating_bert_confusion_matrix.csv",
            reports / "phase7_rating_bert_exp2_confusion_matrix.csv",
        ]

    conf_path = newest_existing(conf_candidates)
    if conf_path is None:
        raise FileNotFoundError("Rating-BERT held-out confusion matrix not found.")

    return metrics_path, conf_path


def generate_rating_bert(model_dir: Path, tables_dir: Path, manifest: dict):
    metrics_path, conf_path = choose_rating_files()
    metrics = read_json(metrics_path)

    exact = as_percent(
        metrics.get("exact_rating_accuracy", metrics.get("eval_exact_accuracy"))
    )
    within = as_percent(
        metrics.get("within_one_rating_accuracy", metrics.get("eval_within_one_accuracy"))
    )
    mae = float(
        metrics.get("mean_absolute_error", metrics.get("eval_mae"))
    )
    macro_f1_raw = metrics.get("macro_f1", metrics.get("eval_f1_macro"))
    macro_f1 = as_percent(macro_f1_raw) if macro_f1_raw is not None else None

    note = f"Mean Absolute Error (MAE) = {mae:.4f} stars. Evaluation uses the held-out validation split."
    save_metric_bars(
        ["Exact 5-Star Accuracy", "Within ±1 Star Accuracy"],
        [exact, within],
        "NLP Town BERT — Held-Out Star-Rating Prediction Performance",
        model_dir / "03_rating_bert_validation_performance.png",
        note=note,
    )

    conf = pd.read_csv(conf_path, index_col=0)
    save_confusion_heatmap(
        conf.to_numpy(dtype=float),
        ["1", "2", "3", "4", "5"],
        "NLP Town BERT — Validation Confusion Matrix",
        model_dir / "04_rating_bert_validation_confusion_matrix.png",
        "Predicted star rating",
        "Actual star rating",
    )

    copy_table(conf_path, tables_dir)

    manifest["rating_bert"] = {
        "metrics_source": str(metrics_path),
        "confusion_source": str(conf_path),
        "exact_accuracy_percent": exact,
        "within_one_accuracy_percent": within,
        "mae_stars": mae,
        "macro_f1_percent": macro_f1,
    }


def minilm_metrics_from_predictions(pred_df: pd.DataFrame):
    y_true = pred_df["human_has_issue"].astype(int).to_numpy()
    y_pred = (pred_df["prediction"].astype(str) != "no_issue").astype(int).to_numpy()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, matrix


def generate_minilm(model_dir: Path, tables_dir: Path, manifest: dict):
    base = PROJECT_ROOT / "outputs" / "minilm_actual_threshold_evaluation"
    pred_path = base / "minilm_actual_threshold_test_predictions.csv"
    summary_path = base / "minilm_actual_threshold_summary.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            "MiniLM controlled benchmark predictions are missing. "
            "Run scripts/evaluate_minilm_ground_truth.py first."
        )

    pred_df = pd.read_csv(pred_path)
    required = {"human_has_issue", "prediction", "domain"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"MiniLM predictions missing columns: {sorted(missing)}")

    metrics, matrix = minilm_metrics_from_predictions(pred_df)

    save_metric_bars(
        ["Accuracy", "Precision", "Recall", "F1"],
        [metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"]],
        "MiniLM Issue Detection — Controlled Benchmark Performance",
        model_dir / "05_minilm_controlled_benchmark_overall.png",
        note=f"Controlled manually constructed validation benchmark; n = {len(pred_df)}.",
    )

    per_domain_rows = []
    for domain, group in pred_df.groupby("domain", sort=True):
        m, _ = minilm_metrics_from_predictions(group)
        per_domain_rows.append({"domain": domain, **m})
    per_domain = pd.DataFrame(per_domain_rows)

    # Domain F1 chart (keeps the domain comparison readable).
    save_simple_bar(
        per_domain.rename(columns={"F1": "f1"}),
        "domain",
        "f1",
        "MiniLM Issue Detection — Controlled Benchmark F1 by Domain",
        "F1 (%)",
        model_dir / "06_minilm_controlled_benchmark_by_domain.png",
        percent_values=True,
    )

    save_confusion_heatmap(
        matrix,
        ["No Issue", "Issue"],
        "MiniLM Issue Detection — Controlled Benchmark Confusion Matrix",
        model_dir / "07_minilm_controlled_benchmark_confusion_matrix.png",
        "Predicted class",
        "Actual class",
    )

    per_domain.to_csv(tables_dir / "minilm_controlled_benchmark_per_domain.csv", index=False)
    copy_table(summary_path, tables_dir)
    copy_table(pred_path, tables_dir)

    manifest["minilm"] = {
        "predictions_source": str(pred_path),
        "summary_source": str(summary_path) if summary_path.exists() else None,
        "benchmark_rows": int(len(pred_df)),
        "accuracy_percent": metrics["Accuracy"],
        "precision_percent": metrics["Precision"],
        "recall_percent": metrics["Recall"],
        "f1_percent": metrics["F1"],
    }


# ---------------------------------------------------------------------------
# Operational Phase-15 figures
# ---------------------------------------------------------------------------

def generate_operational(eval_dir: Path, operational_dir: Path, tables_dir: Path, manifest: dict):
    domain_df = read_csv(eval_dir / "domain_performance.csv")
    risk_df = read_csv(eval_dir / "risk_distribution.csv")
    risk_domain_df = read_csv(eval_dir / "risk_by_domain.csv")
    issue_df = read_csv(eval_dir / "issue_distribution.csv")
    severity_df = read_csv(eval_dir / "severity_distribution.csv")

    if domain_df is not None:
        save_simple_bar(
            domain_df,
            "domain",
            "average_trust_score",
            "Average Trust Score by Domain",
            "Average Trust Score",
            operational_dir / "08_average_trust_score_by_domain.png",
        )
        save_simple_bar(
            domain_df,
            "domain",
            "high_risk_percent",
            "High-Risk Review Percentage by Domain",
            "High-Risk Reviews (%)",
            operational_dir / "09_high_risk_percentage_by_domain.png",
            percent_values=True,
        )
        save_simple_bar(
            domain_df,
            "domain",
            "mismatch_percent",
            "Rating-Review Mismatch Percentage by Domain",
            "Mismatch (%)",
            operational_dir / "10_rating_review_mismatch_percentage_by_domain.png",
            percent_values=True,
        )
        save_simple_bar(
            domain_df,
            "domain",
            "issue_detection_percent",
            "Issue Detection Percentage by Domain",
            "Issue Detection (%)",
            operational_dir / "11_issue_detection_percentage_by_domain.png",
            percent_values=True,
        )

    if risk_df is not None:
        # Support both risk_level and label column names.
        xcol = "risk_level" if "risk_level" in risk_df.columns else "label"
        save_simple_bar(
            risk_df,
            xcol,
            "count",
            "Overall Risk Level Distribution",
            "Number of Reviews",
            operational_dir / "12_overall_risk_level_distribution.png",
            percent_of_total=True,
        )

    if risk_domain_df is not None:
        save_grouped_risk_chart(
            risk_domain_df,
            operational_dir / "13_risk_level_composition_by_domain.png",
        )

    if severity_df is not None:
        xcol = (
            "issue_severity_level"
            if "issue_severity_level" in severity_df.columns
            else ("severity" if "severity" in severity_df.columns else severity_df.columns[0])
        )
        ycol = "count" if "count" in severity_df.columns else severity_df.columns[-1]
        save_simple_bar(
            severity_df,
            xcol,
            ycol,
            "Issue Severity Distribution",
            "Number of Reviews",
            operational_dir / "14_issue_severity_distribution.png",
            percent_of_total=True,
        )

    if issue_df is not None:
        xcol = "primary_issue" if "primary_issue" in issue_df.columns else issue_df.columns[0]
        ycol = "count" if "count" in issue_df.columns else issue_df.columns[-1]
        issue_plot = issue_df[
            issue_df[xcol].astype(str).str.lower().ne("no_issue")
        ].copy()
        save_simple_bar(
            issue_plot,
            xcol,
            ycol,
            "Top Detected Issues",
            "Number of Reviews",
            operational_dir / "15_top_detected_issues.png",
            percent_of_total=False,
            top_n=12,
        )

    # Copy the core Phase-15 tables used in Chapter 5.
    for name in [
        "domain_performance.csv",
        "risk_distribution.csv",
        "risk_by_domain.csv",
        "issue_distribution.csv",
        "issue_by_domain.csv",
        "severity_distribution.csv",
        "severity_by_domain.csv",
        "rag_metrics.csv",
        "top_risky_entities.csv",
        "phase15_evaluation_summary.json",
    ]:
        src = eval_dir / name
        if src.exists():
            shutil.copy2(src, tables_dir / src.name)

    manifest["operational_phase15"] = {
        "source_dir": str(eval_dir),
        "source_modified_at": datetime.fromtimestamp(eval_dir.stat().st_mtime).isoformat(timespec="seconds"),
    }


def write_model_summary(manifest: dict, tables_dir: Path):
    rows = []

    d = manifest.get("distilbert", {})
    if d:
        rows.extend([
            ["DistilBERT sentiment", "Accuracy", d.get("accuracy_percent")],
            ["DistilBERT sentiment", "Macro F1", d.get("macro_f1_percent")],
            ["DistilBERT sentiment", "Weighted F1", d.get("weighted_f1_percent")],
            ["DistilBERT sentiment", "Validation loss", d.get("validation_loss")],
        ])

    r = manifest.get("rating_bert", {})
    if r:
        rows.extend([
            ["NLP Town rating BERT", "Exact 5-star accuracy", r.get("exact_accuracy_percent")],
            ["NLP Town rating BERT", "Within ±1 star accuracy", r.get("within_one_accuracy_percent")],
            ["NLP Town rating BERT", "MAE (stars)", r.get("mae_stars")],
            ["NLP Town rating BERT", "Macro F1", r.get("macro_f1_percent")],
        ])

    m = manifest.get("minilm", {})
    if m:
        rows.extend([
            ["MiniLM issue detection", "Accuracy", m.get("accuracy_percent")],
            ["MiniLM issue detection", "Precision", m.get("precision_percent")],
            ["MiniLM issue detection", "Recall", m.get("recall_percent")],
            ["MiniLM issue detection", "F1", m.get("f1_percent")],
        ])

    pd.DataFrame(rows, columns=["component", "metric", "value"]).to_csv(
        tables_dir / "final_model_metrics_summary.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ONE clean dissertation-figure package from final held-out model "
            "evaluations plus the latest Phase-15 operational results."
        )
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Optional Phase-15 folder. If omitted, the latest folder is used.",
    )
    args = parser.parse_args()

    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        if not eval_dir.is_absolute():
            eval_dir = PROJECT_ROOT / eval_dir
    else:
        eval_dir = latest_subdir(PROJECT_ROOT / "outputs" / "phase15_evaluation_results")

    output_root = PROJECT_ROOT / "outputs" / "dissertation_figures"
    model_dir = ensure_clean_dir(output_root / MODEL_DIR_NAME)
    operational_dir = ensure_clean_dir(output_root / OPERATIONAL_DIR_NAME)
    tables_dir = ensure_clean_dir(output_root / TABLES_DIR_NAME)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "principle": (
            "Model-performance figures use held-out/component-specific evaluation sources. "
            "Operational trust/risk figures use the latest Phase-15 orchestrator evaluation."
        ),
    }

    print("\nFINAL DISSERTATION FIGURE PACKAGE")
    print("=" * 92)
    print(f"Phase-15 operational source: {eval_dir}")
    print(f"Output root: {output_root}")
    print("\nGenerating held-out model-performance figures...")

    generate_distilbert(model_dir, tables_dir, manifest)
    generate_rating_bert(model_dir, tables_dir, manifest)
    generate_minilm(model_dir, tables_dir, manifest)

    print("Generating operational trust/risk figures...")
    generate_operational(eval_dir, operational_dir, tables_dir, manifest)

    write_model_summary(manifest, tables_dir)

    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""FINAL DISSERTATION FIGURES

1. {MODEL_DIR_NAME}
   Held-out / controlled benchmark model-performance figures only.
   These must be used for claims about DistilBERT, NLP Town BERT and MiniLM accuracy/F1.

2. {OPERATIONAL_DIR_NAME}
   Latest Phase-15 operational trust/risk/domain figures.
   These must be used for Chapter 5 domain/risk/discrepancy/issue distributions.

3. {TABLES_DIR_NAME}
   Final model metrics plus copied Phase-15 tables used by the figures.

IMPORTANT:
- Do not quote model accuracy from the operational Phase-15 run when that dataset overlaps model training.
- DistilBERT performance comes from its held-out validation report.
- Rating-BERT performance comes from its held-out validation report.
- MiniLM performance comes from the controlled 320-example benchmark.
- Phase-15 is for operational system findings such as trust, risk, mismatch, issue and entity distributions.
"""
    (output_root / "README_FINAL_FIGURES.txt").write_text(readme, encoding="utf-8")

    print("\nDONE")
    print("=" * 92)

    d = manifest["distilbert"]
    r = manifest["rating_bert"]
    m = manifest["minilm"]

    print(
        f"DistilBERT: Accuracy={d['accuracy_percent']:.2f}%, "
        f"Macro F1={d['macro_f1_percent']:.2f}%"
    )
    print(
        f"Rating BERT: Exact={r['exact_accuracy_percent']:.2f}%, "
        f"Within ±1={r['within_one_accuracy_percent']:.2f}%, "
        f"MAE={r['mae_stars']:.4f}"
    )
    print(
        f"MiniLM: Accuracy={m['accuracy_percent']:.2f}%, "
        f"F1={m['f1_percent']:.2f}%"
    )

    print(f"\nModel figures:       {model_dir}")
    print(f"Operational figures: {operational_dir}")
    print(f"Tables:              {tables_dir}")
    print(f"Source manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
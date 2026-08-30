from __future__ import annotations

import json
import os
import inspect
import importlib.util
import shutil
import sys
import threading
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ui_runs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Local/offline model safety
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

JOBS: Dict[str, Dict[str, Any]] = {}

ALLOWED_MODES = {"csv", "single", "google_url", "app_id", "google_maps_url"}
ALLOWED_DOMAINS = {"auto", "mobile_app", "hotel", "ecommerce", "restaurant"}
ALLOWED_SORTS = {"newest", "most_relevant", "highest_rating", "lowest_rating"}


def _parse_int_form(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = request.form.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name.replace('_', ' ').title()} must be a whole number.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name.replace('_', ' ').title()} must be between {minimum} and {maximum}."
        )
    return value


def _parse_rating_form() -> float:
    raw = request.form.get("rating", "3")
    try:
        rating = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rating must be a number from 1 to 5.") from exc
    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5.")
    return rating


def _model_health() -> Dict[str, bool]:
    from services.local_model_registry import local_model_status

    status = local_model_status()
    return {
        "distilbert": bool(status["distilbert_sentiment"]["ready"]),
        "minilm": bool(status["minilm"]["ready"]),
        "rating": bool(status["rating_model"]["ready"]),
    }


def _json_safe(obj: Any) -> Any:
    """Convert pandas/numpy/path objects into JSON-safe values."""
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _read_text(path: Path, default: str = "") -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return default

def _extract_google_play_app_id(value: str) -> str:
    """
    Google Play URL ya direct package name se app id nikalta hai.
    Example:
    https://play.google.com/store/apps/details?id=com.whatsapp
    -> com.whatsapp
    """
    text = str(value or "").strip()

    if not text:
        raise ValueError("Google Play URL or App ID is required.")

    # Full URL case
    try:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        if "id" in query and query["id"]:
            return query["id"][0].strip()
    except Exception:
        pass

    # URL text me id= search
    match = re.search(r"[?&]id=([^&\s]+)", text)
    if match:
        return match.group(1).strip()

    # Direct package name case
    package_match = re.search(r"([a-zA-Z][\w]*\.)+[a-zA-Z][\w]*", text)
    if package_match:
        return package_match.group(0).strip()

    raise ValueError("Could not extract Google Play App ID from the provided input.")


def _scrape_google_play_to_common_csv(
    app_input: str,
    run_dir: Path,
    max_reviews: int = 200,
    sort_order: str = "newest",
) -> tuple[Path, str, str, int]:
 
    try:
        from google_play_scraper import app as gp_app
        from google_play_scraper import reviews as gp_reviews
        from google_play_scraper import Sort
    except Exception as exc:
        raise ImportError(
            "google-play-scraper is not installed. Run: pip install google-play-scraper"
        ) from exc

    app_id = _extract_google_play_app_id(app_input)
    max_reviews = int(max_reviews or 200)
    if max_reviews <= 0:
        max_reviews = 200

    # For Safety limit demo
    max_reviews = min(max_reviews, 1000)

    sort_map = {
        "newest": Sort.NEWEST,
        "most_relevant": Sort.MOST_RELEVANT,
        "relevant": Sort.MOST_RELEVANT,
    }
    selected_sort = sort_map.get(str(sort_order or "newest").lower(), Sort.NEWEST)

    app_name = app_id
    try:
        details = gp_app(app_id, lang="en", country="us")
        app_name = details.get("title") or app_id
    except Exception:
        app_name = app_id

    scraped_reviews, _ = gp_reviews(
        app_id,
        lang="en",
        country="us",
        sort=selected_sort,
        count=max_reviews,
    )

    if not scraped_reviews:
        raise ValueError(f"No Google Play reviews found for app id: {app_id}")

    raw_path = run_dir / f"raw_google_play_{app_id}.csv"
    pd.DataFrame(scraped_reviews).to_csv(raw_path, index=False, encoding="utf-8-sig")

    common_rows = []

    for index, row in enumerate(scraped_reviews, start=1):
        review_text = str(row.get("content") or "").strip()
        rating = row.get("score")

        if not review_text:
            continue

        review_date = row.get("at")
        if hasattr(review_date, "date"):
            review_date = review_date.date().isoformat()
        else:
            review_date = str(review_date or "")

        common_rows.append({
            "review_id": row.get("reviewId") or f"{app_id}_review_{index}",
            "domain": "mobile_app",
            "entity_id": app_id,
            "entity_name": app_name,
            "review_text": review_text,
            "rating": rating,
            "rating_original": rating,
            "review_date": review_date,
            "source": "google_play_scraper",
            "raw_source_path": str(raw_path),
        })

    if not common_rows:
        raise ValueError(f"Google Play reviews were scraped, but no valid review text was found for {app_id}.")

    prepared_path = run_dir / f"prepared_google_play_{app_id}.csv"
    pd.DataFrame(common_rows).to_csv(prepared_path, index=False, encoding="utf-8-sig")

    return prepared_path, app_id, app_name, len(common_rows)


def _latest_run_folder() -> Optional[Path]:
    root = PROJECT_ROOT / "outputs" / "final_orchestrator_runs"
    if not root.exists():
        return None
    folders = [p for p in root.iterdir() if p.is_dir()]
    if not folders:
        return None
    return sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_csv_rows(path: Path, limit: int = 8) -> list[dict]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        return df.head(limit).fillna("").to_dict(orient="records")
    except Exception:
        return []


def _safe_counts(rows: list[dict], key: str, limit: int = 8) -> list[dict]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return [{"label": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]]



def _normalise_label(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("_", " ").title() if text else "Not Available"


def _first_existing_text(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def _find_final_report_text(run_dir: Path) -> str:
    """Find the Groq report even if the orchestrator saved it with a slightly different path/name."""
    candidates = [
        run_dir / "final_groq_report.txt",
        run_dir / "groq_final_report.txt",
        run_dir / "final_report.txt",
    ]
    try:
        candidates.extend(sorted(run_dir.rglob("*groq*report*.txt"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(run_dir.rglob("*final*report*.txt"), key=lambda p: p.stat().st_mtime, reverse=True))
    except Exception:
        pass

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text and "no final report" not in text.lower() and "no report" not in text.lower():
                    return text
        except Exception:
            continue
    return ""


def _build_fallback_report(
    summary: dict,
    headline: dict,
    distributions: dict,
    issue_examples: list[dict],
    use_rag: bool = False,
) -> str:
    """Create a local deterministic summary when Groq is disabled/unavailable.

    The wording must reflect the switches used in the current run. RAG is only
    described as active when the user actually enabled semantic retrieval.
    """
    entity = headline.get("entity_name") or "Selected entity"
    entity_type = _normalise_label(headline.get("entity_type") or "entity")
    score = headline.get("average_trust_score") or summary.get("average_trust") or 0
    risk = _normalise_label(headline.get("overall_risk_level") or "not available")
    reviews = headline.get("total_reviews") or summary.get("total_reviews") or 0
    issues = headline.get("top_issues") or "no major issue detected"
    high_risk = summary.get("high_risk") or 0
    recommendation = headline.get("entity_recommendation") or "Review the detected issues and review examples before making a trust decision."

    example_lines = []
    for i, row in enumerate(issue_examples[:4], 1):
        if use_rag:
            evidence_text = _first_existing_text(
                row, ["rag_evidence_text"]
            )
            evidence_label = "RAG evidence"
        else:
            evidence_text = _first_existing_text(
                row, ["evidence_phrase", "review_text", "content", "clean_review"]
            )
            evidence_label = "Review text"
        example_lines.append(
            f"Example {i}: rating {row.get('rating', '-')}, sentiment {_normalise_label(row.get('predicted_sentiment'))}, "
            f"issue {_normalise_label(row.get('primary_issue'))}, trust score {row.get('trust_score', '-')}/100. "
            f"{evidence_label}: {evidence_text or 'No independent corpus evidence retrieved.'}"
        )

    if not example_lines:
        example_lines.append("No major issue examples were available in the selected sample.")

    signal_text = (
        "rating signals, transformer sentiment, predicted star rating, discrepancy checks, "
        "domain-aware issue severity and risk scoring"
    )
    if use_rag:
        signal_text += ", with MiniLM/FAISS RAG evidence retrieval enabled"

    evidence_heading = "3. RAG-Supported Review Evidence Examples" if use_rag else "3. Review Examples"

    return "\n".join([
        "1. Overall Trust Assessment",
        f"Entity: {entity} ({entity_type})",
        f"Analysed reviews: {reviews}",
        f"Average trust score: {score}/100",
        f"Overall risk level: {risk}",
        f"High-risk reviews: {high_risk}",
        f"Main detected issues: {issues}",
        "",
        "2. Main Interpretation",
        f"The system combined {signal_text}. The final judgement is {risk} because the calculated risk factors produced an average trust score of {score}/100.",
        "",
        evidence_heading,
        *example_lines,
        "",
        "4. Recommendation",
        str(recommendation),
    ])


def _select_issue_examples(review_df: Optional[pd.DataFrame], limit: int = 4) -> list[dict]:
    if review_df is None or review_df.empty:
        return []

    df = review_df.copy().fillna("")
    for col in ["primary_issue", "risk_level", "issue_severity_level", "trust_score"]:
        if col not in df.columns:
            df[col] = ""

    issue_col = df["primary_issue"].astype(str).str.lower()
    risk_col = df["risk_level"].astype(str).str.lower()
    severity_col = df["issue_severity_level"].astype(str).str.lower()

    # Priority: real issues first, then high-risk, then medium-risk, then any row.
    issue_mask = (~issue_col.isin(["", "none", "no_issue", "nan", "not_available"]))
    risky_mask = risk_col.eq("high_risk") | severity_col.isin(["high", "critical"])

    candidates = []
    for subset in [df[issue_mask & risky_mask], df[issue_mask], df[risky_mask], df]:
        if subset.empty:
            continue
        temp = subset.copy()
        if "trust_score" in temp.columns:
            temp["_trust_sort"] = pd.to_numeric(temp["trust_score"], errors="coerce").fillna(999)
            temp = temp.sort_values("_trust_sort", ascending=True)
        for row in temp.to_dict(orient="records"):
            key = (row.get("review_id"), row.get("review_text"), row.get("content"))
            if key not in candidates:
                candidates.append(key)
        if len(candidates) >= limit:
            break

    selected = []
    seen = set()
    for subset in [df[issue_mask & risky_mask], df[issue_mask], df[risky_mask], df]:
        if subset.empty:
            continue
        temp = subset.copy()
        if "trust_score" in temp.columns:
            temp["_trust_sort"] = pd.to_numeric(temp["trust_score"], errors="coerce").fillna(999)
            temp = temp.sort_values("_trust_sort", ascending=True)
        for row in temp.to_dict(orient="records"):
            key = str(row.get("review_id") or row.get("review_text") or row.get("content") or len(selected))[:200]
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= limit:
                return _json_safe(selected)
    return _json_safe(selected[:limit])


def _clean_label(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "not_available"}:
        return "Not available"
    return text.replace("_", " ").title()


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if str(value).strip().lower() in {"", "nan", "none", "null"}:
            return default
        return round(float(value), 2)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        if str(value).strip().lower() in {"", "nan", "none", "null"}:
            return default
        return int(float(value))
    except Exception:
        return default


def _normalise_distribution(value, label_key=None, count_key=None):
    """
    UI-safe distribution converter.

    Accepts:
    - dict: {"low_risk": 10}
    - list of dicts: [{"label": "low_risk", "count": 10}]
    - list of dicts: [{"risk_level": "low_risk", "count": 10}]
    - list of pairs: [["low_risk", 10]]

    Returns:
    - clean dict: {"low_risk": 10}

    This prevents the UI/API crash:
    'list' object has no attribute 'items'
    """
    output = {}

    if value is None:
        return output

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                count = (
                    v.get("count")
                    or v.get("reviews")
                    or v.get("total_reviews")
                    or v.get("value")
                    or 0
                )
                output[str(k)] = _safe_int(count)
            else:
                output[str(k)] = _safe_int(v)
        return output

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                key = None

                if label_key and label_key in item:
                    key = item.get(label_key)
                else:
                    for possible_key in [
                        "label",
                        "risk_level",
                        "domain",
                        "primary_issue",
                        "issue",
                        "name",
                        "category",
                    ]:
                        if possible_key in item:
                            key = item.get(possible_key)
                            break

                count = None
                if count_key and count_key in item:
                    count = item.get(count_key)
                else:
                    for possible_count in [
                        "count",
                        "reviews",
                        "total_reviews",
                        "value",
                        "frequency",
                    ]:
                        if possible_count in item:
                            count = item.get(possible_count)
                            break

                if key is not None:
                    output[str(key)] = output.get(str(key), 0) + _safe_int(count, 1)

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                output[str(item[0])] = output.get(str(item[0]), 0) + _safe_int(item[1])

            else:
                output[str(item)] = output.get(str(item), 0) + 1

        return output

    return output


def _top_items(distribution, limit=5, exclude=None):
    exclude = set(str(x).lower() for x in (exclude or []))
    dist = _normalise_distribution(distribution)
    cleaned = {
        k: v
        for k, v in dist.items()
        if str(k).lower() not in exclude and _safe_int(v) > 0
    }
    return sorted(cleaned.items(), key=lambda x: x[1], reverse=True)[:limit]


def _build_ui_final_summary(summary, headline, distributions, issue_examples, final_report, use_rag: bool = False):
    """
    Makes Final UI object. This function's output gives the frontend clean cards,
    readable summary, issue examples and final report.
    """
    summary = summary or {}
    headline = headline or {}
    distributions = distributions or {}
    issue_examples = issue_examples or []

    entity_name = (
        headline.get("entity_name")
        or headline.get("name")
        or headline.get("entity")
        or "Selected entity"
    )

    domain = (
        headline.get("domain")
        or headline.get("entity_type")
        or "auto domain"
    )

    trust_score = _safe_float(
        headline.get("average_trust_score")
        or headline.get("trust_score")
        or summary.get("average_trust")
        or summary.get("average_trust_score")
        or 0
    )

    total_reviews = _safe_int(
        headline.get("total_reviews")
        or summary.get("total_reviews")
        or 0
    )

    high_risk = _safe_int(
        summary.get("high_risk")
        or headline.get("high_risk")
        or 0
    )

    risk_level_raw = (
        headline.get("overall_risk_level")
        or headline.get("risk_level")
        or "not_available"
    )
    risk_label = _clean_label(risk_level_raw)

    risk_distribution = _normalise_distribution(
        distributions.get("risk") if isinstance(distributions, dict) else {},
        label_key="label",
        count_key="count",
    )

    domain_distribution = _normalise_distribution(
        distributions.get("domain") if isinstance(distributions, dict) else {},
        label_key="label",
        count_key="count",
    )

    issue_distribution = _normalise_distribution(
        distributions.get("issue") if isinstance(distributions, dict) else {},
        label_key="label",
        count_key="count",
    )

    top_issue_pairs = _top_items(
        issue_distribution,
        limit=5,
        exclude={"no_issue", "none", "not_available", "nan", ""},
    )
    top_issues = [f"{_clean_label(issue)} ({count})" for issue, count in top_issue_pairs]

    if not top_issues:
        top_issues = ["No major recurring issue detected"]

    if "high" in risk_label.lower():
        recommendation = "Avoid or review very carefully before trusting this entity."
    elif "medium" in risk_label.lower():
        recommendation = "Use with caution and review the detected issues before trusting this entity."
    elif "low" in risk_label.lower():
        recommendation = "Generally reliable based on the analysed review evidence."
    else:
        recommendation = "Review the evidence before making a decision."

    if isinstance(issue_examples, dict):
        issue_examples = list(issue_examples.values())

    important_examples = []
    for row in issue_examples[:4]:
        if not isinstance(row, dict):
            continue

        important_examples.append({
            "entity_name": row.get("entity_name") or entity_name,
            "rating": row.get("rating", row.get("score", "")),
            "predicted_rating": row.get("predicted_star_rating", ""),
            "sentiment": _clean_label(row.get("predicted_sentiment", "")),
            "risk_level": _clean_label(row.get("risk_level", "")),
            "trust_score": row.get("trust_score", ""),
            "issue": _clean_label(row.get("primary_issue", "")),
            "severity": _clean_label(row.get("issue_severity_level", "")),
            "discrepancy": _clean_label(row.get("discrepancy_status", "")),
            "review_text": (
                row.get("review_text")
                or row.get("content")
                or row.get("clean_review")
                or ""
            ),
            "explanation": (
                row.get("evidence_based_explanation")
                or row.get("explanation_text")
                or ""
            ),
            "evidence": (
                (row.get("rag_evidence_text") or row.get("evidence_phrase") or "")
                if use_rag
                else (row.get("evidence_phrase") or row.get("review_text") or row.get("content") or "")
            ),
        })

    overall_summary = (
        f"{entity_name} was analysed using the agentic review trust pipeline. "
        f"The system reviewed {total_reviews} review records and calculated an average trust score of {trust_score}/100. "
        f"The overall risk level is {risk_label}. "
        f"Main detected issues are: {', '.join(top_issues[:5])}. "
        f"{recommendation}"
    )

    return {
        "entity_name": entity_name,
        "domain": _clean_label(domain),
        "trust_score": trust_score,
        "total_reviews": total_reviews,
        "high_risk": high_risk,
        "risk_level": risk_label,
        "recommendation": recommendation,
        "top_issues": top_issues[:5],
        "one_line_verdict": f"{entity_name} received a trust score of {trust_score}/100 and is classified as {risk_label}.",
        "overall_summary": overall_summary,
        "how_decision_was_calculated": [
            "The sentiment model checks whether each review is positive, neutral or negative.",
            "The rating prediction model estimates the star rating from the review text.",
            "The discrepancy agent compares the actual rating with the predicted rating.",
            "The issue mining agent detects domain-specific problems such as payment, subscription, privacy, crash, fake product, room quality or food quality.",
            (
                "The RAG evidence retrieval agent searches the persistent combined four-domain review corpus with MiniLM embeddings, domain filtering and self/duplicate exclusion (FAISS when available)."
                if use_rag
                else "RAG evidence retrieval was disabled by the user for this run."
            ),
            "The risk scoring agent combines all active signals into a trust score, risk level and recommendation.",
        ],
        "risk_distribution": risk_distribution,
        "domain_distribution": domain_distribution,
        "issue_distribution": issue_distribution,
        "important_examples": important_examples,
        "final_report": final_report or "",
    }

def _summarise_results(run_dir: Path, use_rag: bool = False, use_groq: bool = False) -> dict:
    pipeline_dir = run_dir / "analysis_pipeline"

    review_file = pipeline_dir / "multidomain_review_level_results.csv"
    entity_file = pipeline_dir / "multidomain_entity_level_summary.csv"

    if not review_file.exists():
        matches = list(pipeline_dir.glob("*review*results*.csv"))
        if matches:
            review_file = matches[0]
    if not entity_file.exists():
        matches = list(pipeline_dir.glob("*entity*summary*.csv"))
        if matches:
            entity_file = matches[0]

    review_df = None
    entity_df = None
    review_rows = []
    entity_rows = []

    if review_file.exists():
        review_df = pd.read_csv(review_file).fillna("")

    if entity_file.exists():
        entity_df = pd.read_csv(entity_file).fillna("")

    total_reviews = int(len(review_df)) if review_df is not None else 0
    total_entities = int(len(entity_df)) if entity_df is not None else 0

    avg_trust = 0
    high_risk_count = 0

    if review_df is not None and "trust_score" in review_df.columns:
        avg_trust = round(pd.to_numeric(review_df["trust_score"], errors="coerce").fillna(0).mean(), 2)

    if review_df is not None and "risk_level" in review_df.columns:
        high_risk_count = int((review_df["risk_level"].astype(str) == "high_risk").sum())

    risk_distribution = []
    domain_distribution = []
    issue_distribution = []

    if review_df is not None:
        if "risk_level" in review_df.columns:
            risk_distribution = (
                review_df["risk_level"].astype(str).value_counts()
                .rename_axis("label").reset_index(name="count")
                .to_dict(orient="records")
            )
        if "domain" in review_df.columns:
            domain_distribution = (
                review_df["domain"].astype(str).value_counts()
                .rename_axis("label").reset_index(name="count")
                .to_dict(orient="records")
            )
        if "primary_issue" in review_df.columns:
            issue_distribution = (
                review_df["primary_issue"].astype(str).value_counts().head(8)
                .rename_axis("label").reset_index(name="count")
                .to_dict(orient="records")
            )

    # Entity headline: prefer entity with most reviews, otherwise first row.
    selected_entity = {}
    if entity_df is not None and not entity_df.empty:
        entity_copy = entity_df.copy()
        if "total_reviews" in entity_copy.columns:
            entity_copy["_reviews_sort"] = pd.to_numeric(entity_copy["total_reviews"], errors="coerce").fillna(0)
            entity_copy = entity_copy.sort_values("_reviews_sort", ascending=False)
        entity_rows = entity_copy.head(6).to_dict(orient="records")
        selected_entity = entity_rows[0]

    # Review rows are not a raw table dump anymore. They are selected examples.
    issue_examples = _select_issue_examples(review_df, limit=4)
    if review_df is not None:
        review_rows = issue_examples or review_df.head(4).to_dict(orient="records")

    headline = {
        "entity_name": selected_entity.get("entity_name", selected_entity.get("entity_id", "Selected entity")),
        "entity_type": selected_entity.get("entity_type", "entity"),
        "average_rating": selected_entity.get("average_rating", ""),
        "average_trust_score": selected_entity.get("average_trust_score", avg_trust),
        "overall_risk_level": selected_entity.get("overall_risk_level", "not_available"),
        "total_reviews": selected_entity.get("total_reviews", total_reviews),
        "top_issues": selected_entity.get("top_issues", "not_available"),
        "high_risk_percentage": selected_entity.get("high_risk_percentage", ""),
        "mismatch_percentage": selected_entity.get("mismatch_percentage", ""),
        "entity_recommendation": selected_entity.get("entity_recommendation", ""),
    }

    summary = {
        "total_reviews": total_reviews,
        "total_entities": total_entities,
        "average_trust": avg_trust,
        "high_risk": high_risk_count,
    }

    distributions = {
        "risk": _json_safe(risk_distribution),
        "domain": _json_safe(domain_distribution),
        "issue": _json_safe(issue_distribution),
    }

    # Do not label a local summary as Groq. Only use a Groq report when Groq
    # was requested and a real report file exists for this isolated run.
    groq_report = _find_final_report_text(run_dir) if use_groq else ""
    groq_generated = bool(groq_report)

    if groq_generated:
        final_report = groq_report
        report_type = "groq"
        report_title = "Final Groq Trust/Risk Summary"
        report_eyebrow = "LLM Finalisation"
    else:
        final_report = _build_fallback_report(
            summary, headline, distributions, issue_examples, use_rag=use_rag
        )
        local_report_path = run_dir / "local_final_summary.txt"
        local_report_path.write_text(final_report, encoding="utf-8")
        if use_groq:
            report_type = "local_fallback"
            report_title = "Local Fallback Trust/Risk Summary"
            report_eyebrow = "Local Finalisation · Groq Unavailable"
        elif use_rag:
            report_type = "local_rag"
            report_title = "Local Evidence-Based Trust/Risk Summary"
            report_eyebrow = "Local Finalisation · RAG Enabled"
        else:
            report_type = "local"
            report_title = "Local Explainable Trust/Risk Summary"
            report_eyebrow = "Local Finalisation"

    report_meta = {
        "use_rag": bool(use_rag),
        "use_groq": bool(use_groq),
        "groq_generated": groq_generated,
        "report_type": report_type,
        "report_title": report_title,
        "report_eyebrow": report_eyebrow,
        "evidence_label": "RAG Supporting Evidence" if use_rag else "Review Evidence",
    }

    final_summary = _build_ui_final_summary(
        summary, headline, distributions, issue_examples, final_report, use_rag=use_rag
    )

    files = []
    for path in [
        run_dir / "orchestrator_state.json",
        run_dir / "prepared_standardised_dataset.csv",
        review_file,
        entity_file,
        run_dir / "final_groq_report.txt",
        run_dir / "final_groq_context_payload.json",
        run_dir / "local_final_summary.txt",
    ]:
        if path.exists():
            files.append({"name": path.name, "path": str(path)})

    return {
        "summary": _json_safe(summary),
        "headline": _json_safe(headline),
        "distributions": distributions,
        "review_rows": _json_safe(review_rows),
        "issue_examples": _json_safe(issue_examples),
        "entity_rows": _json_safe(entity_rows),
        "final_report": final_report,
        "final_summary": _json_safe(final_summary),
        "report_meta": _json_safe(report_meta),
        "files": _json_safe(files),

    }

def _trace_step(title: str, message: str, output: Optional[dict] = None) -> dict:
    return {
        "title": title,
        "message": message,
        "output": output or {},
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def _set_job(job_id: str, **kwargs):
    JOBS[job_id].update(kwargs)


def _call_final_orchestrator(
    orchestrator,
    source_value: str,
    input_type: str = "csv",
    domain: str = "auto",
    sample_size: int = 200,
    max_scraper_reviews: int = 200,
    use_rag: bool = True,
    use_groq: bool = True,
    output_dir: str = "",
):
    """
    Compatibility wrapper for different FinalOrchestrator.run() signatures.

    Your project already has a FinalOrchestrator, but its run() method may not use
    the keyword name `user_input`. This wrapper checks the actual function
    signature and only passes the argument names supported by your local code.
    """
    run_method = orchestrator.run
    signature = inspect.signature(run_method)
    params = signature.parameters

    candidate_values = {
        "user_input": source_value,
        "input_data": source_value,
        "input_value": source_value,
        "input_text": source_value,
        "source": source_value,
        "source_value": source_value,
        "csv_path": source_value,
        "csv_file": source_value,
        "file_path": source_value,
        "input_path": source_value,
        "path": source_value,
        "url": source_value,
        "google_url": source_value,
        "app_id": source_value,
        "input_type": input_type,
        "mode": input_type,
        "domain": domain,
        "sample_size": sample_size,
        "max_scraper_reviews": max_scraper_reviews,
        "max_reviews": max_scraper_reviews,
        "use_rag": use_rag,
        "rag_enabled": use_rag,
        "use_groq": use_groq,
        "groq_enabled": use_groq,
        "output_dir": output_dir,
        "run_output_dir": output_dir,
    }

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs = {}
    source_added = False

    if input_type == "csv":
        source_priority = ["csv_path", "csv_file", "file_path", "input_path", "path", "input_data", "input_value", "source", "user_input"]
    elif input_type == "google_maps_url":
        source_priority = ["google_maps_url", "url", "source", "input_data", "input_value", "user_input"]
    elif input_type == "google_url":
        source_priority = ["google_url", "url", "source", "input_data", "input_value", "user_input"]
    elif input_type == "app_id":
        source_priority = ["app_id", "source", "input_data", "input_value", "user_input"]
    else:
        source_priority = ["input_text", "input_data", "input_value", "source", "user_input"]

    if accepts_kwargs:
        # For **kwargs methods, pass only the safe source aliases for the selected input type.
        # This avoids a CSV path being sent as google_url/app_id and prevents duplicate scraper runs.
        for name in source_priority:
            kwargs[name] = source_value
        for name in [
            "input_type", "mode", "domain", "sample_size", "max_scraper_reviews", "max_reviews",
            "use_rag", "rag_enabled", "use_groq", "groq_enabled", "output_dir", "run_output_dir"
        ]:
            kwargs[name] = candidate_values[name]
        return run_method(**kwargs)

    for name in source_priority:
        if name in params:
            kwargs[name] = source_value
            source_added = True
            break

    for name, value in candidate_values.items():
        if name in params and name not in kwargs and name not in {"google_url", "url", "app_id"}:
            kwargs[name] = value

    try:
        return run_method(**kwargs)
    except TypeError as keyword_error:
        # Last safe fallback: pass source value positionally and only keep supported optional kwargs.
        optional_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in source_priority and k in params
        }
        try:
            return run_method(source_value, **optional_kwargs)
        except TypeError:
            raise keyword_error


def _add_trace(job_id: str, title: str, message: str, output: Optional[dict] = None):
    JOBS[job_id].setdefault("trace", []).append(_trace_step(title, message, output))


def _run_orchestrator_job(job_id: str, payload: dict):
    try:
        _set_job(job_id, status="running", progress=8)
        _add_trace(job_id, "Orchestrator Input Reasoning", "The orchestrator received the user request and identified the selected workflow.", {"input_type": payload.get("mode")})

        mode = payload.get("mode", "csv")
        sample_size_value = payload.get("sample_size")
        sample_size = 200 if sample_size_value in (None, "") else int(sample_size_value)
        use_rag = bool(payload.get("use_rag", True))
        use_groq = bool(payload.get("use_groq", True))

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        prepared_csv = None
        input_artifacts: Dict[str, str] = {}

        if mode == "csv":
            uploaded_file = payload.get("uploaded_file")
            if not uploaded_file:
                # default demo dataset
                demo = PROJECT_ROOT / "data" / "processed" / "combined_multidomain_reviews.csv"
                if not demo.exists():
                    raise FileNotFoundError("No CSV uploaded and default combined_multidomain_reviews.csv was not found.")
                prepared_csv = demo
            else:
                prepared_csv = Path(uploaded_file)

            _set_job(job_id, progress=18)
            _add_trace(job_id, "Dataset Workflow Selected", "CSV dataset was selected and passed to the data standardisation workflow.", {"csv_path": str(prepared_csv)})

            from pipeline.final_orchestrator import FinalOrchestrator

            orchestrator = FinalOrchestrator(
                base_output_dir=run_dir,
                use_rag=use_rag,
                use_groq=use_groq,
                sample_size=sample_size,
                max_reviews=int(payload.get("max_scraper_reviews") or 200),
            )
            state = _call_final_orchestrator(
                orchestrator,
                source_value=str(prepared_csv),
                input_type="csv",
                domain=payload.get("domain") or "auto",
                sample_size=sample_size,
                max_scraper_reviews=int(payload.get("max_scraper_reviews") or 200),
                use_rag=use_rag,
                use_groq=use_groq,
                output_dir=str(run_dir),
            )

        elif mode == "single":
            review_text = payload.get("review_text", "").strip()
            rating = payload.get("rating") or 3
            domain = payload.get("domain") or "mobile_app"
            entity_name = payload.get("entity_name") or "Single Review Entity"

            if not review_text:
                raise ValueError("Single review text is required.")

            prepared_csv = run_dir / "single_review_input.csv"
            pd.DataFrame([{
                "review_id": "single_review_1",
                "domain": domain,
                "entity_id": entity_name,
                "entity_name": entity_name,
                "review_text": review_text,
                "rating": rating,
                "rating_original": rating,
                "review_date": datetime.now().date().isoformat(),
                "source": "ui_single_review",
                "raw_source_path": "ui"
            }]).to_csv(prepared_csv, index=False)

            _set_job(job_id, progress=18)
            _add_trace(job_id, "Single Review Workflow Selected", "A single review was converted into the common schema and passed to specialised agents.", {"domain": domain, "rating": rating})

            from pipeline.final_orchestrator import FinalOrchestrator

            orchestrator = FinalOrchestrator(
                base_output_dir=run_dir,
                use_rag=use_rag,
                use_groq=use_groq,
                sample_size=sample_size,
                max_reviews=int(payload.get("max_scraper_reviews") or 200),
            )
            state = _call_final_orchestrator(
                orchestrator,
                source_value=str(prepared_csv),
                input_type="csv",
                domain=domain,
                sample_size=sample_size,
                max_scraper_reviews=int(payload.get("max_scraper_reviews") or 200),
                use_rag=use_rag,
                use_groq=use_groq,
                output_dir=str(run_dir),
            )

        elif mode == "google_maps_url":
            place_url = str(payload.get("google_maps_url") or "").strip()
            requested_domain = str(payload.get("domain") or "auto").strip().lower()
            max_maps_reviews = int(payload.get("max_scraper_reviews") or 100)
            maps_sort_order = payload.get("sort_order") or "most_relevant"

            if not place_url:
                raise ValueError("Google Maps place URL is required.")

            _set_job(job_id, progress=16)
            _add_trace(
                job_id,
                "Google Maps CLI Bridge Selected",
                "The orchestrator identified a Google Maps place URL and selected the independently tested collector-command workflow.",
                {
                    "domain": requested_domain,
                    "max_reviews": max_maps_reviews,
                    "sort_order": maps_sort_order,
                },
            )

            from services.google_maps_cli_bridge_service import GoogleMapsCliBridgeService

            collector_dir = run_dir / "google_maps_collector"
            bridge = GoogleMapsCliBridgeService(
                project_root=PROJECT_ROOT,
                cdp_url=os.environ.get("GOOGLE_MAPS_CDP_URL") or "http://127.0.0.1:9222",
            )

            _set_job(job_id, progress=22)
            _add_trace(
                job_id,
                "Collector Command Started",
                "The Flask job launched the same working Google Maps collector CLI with the active virtual-environment Python interpreter.",
                {
                    "collector": "scripts/test_google_maps_scraper.py",
                    "browser_session": "signed-in Chrome via CDP",
                    "job_output": str(collector_dir),
                },
            )

            maps_result = bridge.collect(
                place_url=place_url,
                output_dir=collector_dir,
                max_reviews=max_maps_reviews,
                sort_order=maps_sort_order,
                domain=requested_domain,
                use_cdp=True,
            )

            prepared_csv = Path(maps_result["prepared_csv"])
            input_artifacts = {
                "raw_scraper_reviews": str(maps_result.get("raw_csv") or ""),
                "scraper_metadata": str(maps_result.get("metadata_json") or ""),
                "scraper_prepared_dataset": str(maps_result.get("prepared_csv") or ""),
                "collector_result": str(maps_result.get("result_json") or ""),
                "collector_stdout": str(maps_result.get("stdout_log") or ""),
                "collector_stderr": str(maps_result.get("stderr_log") or ""),
            }

            _set_job(job_id, progress=38)
            _add_trace(
                job_id,
                "Google Maps Reviews Collected",
                "The collector completed successfully. Its prepared common-schema CSV was validated and handed to the existing CSV analysis workflow.",
                {
                    "entity_name": maps_result.get("entity_name"),
                    "category": maps_result.get("category"),
                    "domain": maps_result.get("domain"),
                    "reviews": maps_result.get("scraped_count"),
                    "displayed_reviews": maps_result.get("displayed_review_count"),
                    "duration_seconds": maps_result.get("duration_seconds"),
                    "saved_to": str(prepared_csv),
                },
            )

            from pipeline.final_orchestrator import FinalOrchestrator

            orchestrator = FinalOrchestrator(
                base_output_dir=run_dir,
                use_rag=use_rag,
                use_groq=use_groq,
                sample_size=0,
                max_reviews=max_maps_reviews,
            )
            state = _call_final_orchestrator(
                orchestrator,
                source_value=str(prepared_csv),
                input_type="csv",
                domain=maps_result.get("domain") or requested_domain or "restaurant",
                sample_size=0,
                max_scraper_reviews=max_maps_reviews,
                use_rag=use_rag,
                use_groq=use_groq,
                output_dir=str(run_dir),
            )

        elif mode in {"google_url", "app_id"}:
            app_input = payload.get("google_url") or payload.get("app_id")
            if not app_input:
                raise ValueError("Google Play URL or App ID is required.")

            _set_job(job_id, progress=18)
            _add_trace(
                job_id,
                "Google Play Scraper Workflow Selected",
                "Google Play input was identified. The app ID and app name will be detected automatically.",
                {"input": app_input}
            )

            prepared_csv, app_id, app_name, scraped_count = _scrape_google_play_to_common_csv(
                app_input=app_input,
                run_dir=run_dir,
                max_reviews=int(payload.get("max_scraper_reviews") or 200),
                sort_order=payload.get("sort_order") or "newest",
            )

            input_artifacts = {
                "raw_scraper_reviews": str(run_dir / f"raw_google_play_{app_id}.csv"),
                "scraper_prepared_dataset": str(prepared_csv),
            }

            _set_job(job_id, progress=32)
            _add_trace(
                job_id,
                "Google Play Reviews Collected",
                "Reviews were scraped from Google Play and converted into the common mobile-app review schema.",
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "reviews": scraped_count,
                    "saved_to": str(prepared_csv),
                }
            )

            from pipeline.final_orchestrator import FinalOrchestrator

            orchestrator = FinalOrchestrator(
                base_output_dir=run_dir,
                use_rag=use_rag,
                use_groq=use_groq,
                sample_size=0,
                max_reviews=int(payload.get("max_scraper_reviews") or 200),
            )

            
            state = _call_final_orchestrator(
                orchestrator,
                source_value=str(prepared_csv),
                input_type="csv",
                domain="mobile_app",
                sample_size=0,
                max_scraper_reviews=int(payload.get("max_scraper_reviews") or 200),
                use_rag=use_rag,
                use_groq=use_groq,
                output_dir=str(run_dir),
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        _set_job(job_id, progress=70)
        specialist_message = (
            "Sentiment, rating prediction, discrepancy, issue mining, risk scoring, "
            "explainability and entity summary agents completed."
        )
        specialist_message += (
            " RAG evidence retrieval was enabled and completed."
            if use_rag
            else " RAG evidence retrieval was disabled by the user and skipped."
        )
        _add_trace(
            job_id,
            "Specialised Agents Completed",
            specialist_message,
            {"use_rag": use_rag, "use_groq": use_groq},
        )

        # Prefer run_dir from state if available.
        actual_run_dir = Path(str(state.get("output_dir", run_dir))) if isinstance(state, dict) else run_dir
        if not actual_run_dir.exists():
            actual_run_dir = _latest_run_folder() or run_dir

        _set_job(job_id, progress=88)
        groq_report_exists = (actual_run_dir / "final_groq_report.txt").is_file()
        if use_groq and groq_report_exists:
            _add_trace(
                job_id,
                "Groq Final Summary Generated",
                "Groq converted validated structured outputs into a readable trust/risk report.",
                {"enabled": True, "generated": True},
            )
        elif use_groq:
            _add_trace(
                job_id,
                "Groq Summary Unavailable",
                "Groq was enabled, but no Groq report was produced. A local deterministic summary will be shown instead.",
                {"enabled": True, "generated": False},
            )
        else:
            _add_trace(
                job_id,
                "Groq Summary Skipped",
                "Groq final summary was disabled by the user. A local deterministic summary will be shown instead.",
                {"enabled": False, "generated": False},
            )

        results = _summarise_results(
            actual_run_dir, use_rag=use_rag, use_groq=use_groq
        )
        report_meta = results.get("report_meta") or {}

        _set_job(
            job_id,
            status="completed",
            progress=100,
            run_dir=str(actual_run_dir),
            results=results,
            runtime_options={
                "use_rag": use_rag,
                "use_groq": use_groq,
                "groq_generated": bool(report_meta.get("groq_generated")),
            },
            input_artifacts=input_artifacts,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _add_trace(job_id, "Orchestrator Finalisation", "The orchestrator collected all agent outputs and finalised the run.", {"status": "completed"})

    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc), progress=100)
        _add_trace(job_id, "Run Failed", str(exc), {"error": type(exc).__name__})


@app.route("/")
def index():
    return render_template("index.html")


def _no_cache_json(payload: Dict[str, Any], status_code: int = 200):
    """Return JSON that browsers/proxies must not reuse from an earlier process."""
    response = jsonify(_json_safe(payload))
    response.status_code = status_code
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _google_maps_bridge_status() -> Dict[str, Any]:
    """Authoritative bridge pre-flight used by both health UI and job submission."""
    script_path = PROJECT_ROOT / "scripts" / "test_google_maps_scraper.py"
    service_path = PROJECT_ROOT / "services" / "google_maps_cli_bridge_service.py"
    maps_cdp_url = os.environ.get("GOOGLE_MAPS_CDP_URL") or "http://127.0.0.1:9222"

    package_ready = importlib.util.find_spec("playwright") is not None
    script_ready = script_path.exists()
    service_ready = service_path.exists()
    import_ready = False
    cdp_ready = False
    error = ""

    if not package_ready:
        error = f"Playwright is not installed in {sys.executable}."
    elif not script_ready:
        error = f"Collector script is missing: {script_path}"
    elif not service_ready:
        error = f"CLI bridge service is missing: {service_path}"
    else:
        try:
            from services.google_maps_cli_bridge_service import GoogleMapsCliBridgeService

            bridge = GoogleMapsCliBridgeService(
                project_root=PROJECT_ROOT,
                cdp_url=maps_cdp_url,
                timeout_seconds=60,
            )
            import_ready = True
            cdp_ready = bridge.cdp_is_ready()
            if not cdp_ready:
                error = (
                    f"Signed-in Chrome is not reachable at {maps_cdp_url}. "
                    "Run: python scripts/start_google_maps_chrome.py"
                )
        except Exception as exc:
            error = f"Bridge import/pre-flight failed: {type(exc).__name__}: {exc}"

    bridge_ready = bool(package_ready and script_ready and service_ready and import_ready)
    ready = bool(bridge_ready and cdp_ready)

    return {
        "ready": ready,
        "google_maps_playwright": package_ready,
        "google_maps_collector_script": script_ready,
        "google_maps_bridge_service": service_ready,
        "google_maps_bridge_import": import_ready,
        "google_maps_bridge": bridge_ready,
        "google_maps_cdp_ready": cdp_ready,
        "google_maps_cdp_url": maps_cdp_url,
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "error": error,
    }


@app.route("/api/google-maps/preflight")
def google_maps_preflight():
    status = _google_maps_bridge_status()
    return _no_cache_json(status, 200 if status["ready"] else 503)


@app.route("/api/health")
def health():
    maps_status = _google_maps_bridge_status()
    models = _model_health()
    model_dependencies = {
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "faiss": importlib.util.find_spec("faiss") is not None,
    }
    groq_key_present = bool(os.environ.get("GROQ_API_KEY"))
    groq_client_ready = importlib.util.find_spec("openai") is not None
    overall_ready = all(models.values()) and all(model_dependencies.values())

    return _no_cache_json({
        "status": "ready" if overall_ready else "setup_required",
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "groq_configured": groq_key_present,
        "groq_client_ready": groq_client_ready,
        "models": models,
        "model_dependencies": model_dependencies,
        "scrapers": {
            "google_play": importlib.util.find_spec("google_play_scraper") is not None,
            **maps_status,
        },
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        mode = str(request.form.get("mode", "csv")).strip().lower()
        domain = str(request.form.get("domain", "auto")).strip().lower()
        sort_order = str(request.form.get("sort_order", "newest")).strip().lower()

        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported input mode: {mode}")
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Unsupported domain: {domain}")
        if sort_order not in ALLOWED_SORTS:
            raise ValueError(f"Unsupported review sort order: {sort_order}")
        if mode != "google_maps_url" and sort_order in {"highest_rating", "lowest_rating"}:
            raise ValueError("Highest/lowest rating sorting is only available for Google Maps.")

        sample_size = _parse_int_form("sample_size", default=200, minimum=0, maximum=1_000_000)
        max_scraper_reviews = _parse_int_form(
            "max_scraper_reviews", default=20, minimum=1, maximum=1000
        )
        rating = _parse_rating_form()

        payload = {
            "mode": mode,
            "domain": domain,
            "sample_size": sample_size,
            "max_scraper_reviews": max_scraper_reviews,
            "use_rag": request.form.get("use_rag", "true") == "true",
            "sort_order": sort_order,
            "use_groq": request.form.get("use_groq", "true") == "true",
            "review_text": request.form.get("review_text", ""),
            "rating": rating,
            "entity_name": request.form.get("entity_name", ""),
            "google_url": request.form.get("google_url", ""),
            "google_maps_url": request.form.get("google_maps_url", ""),
            "app_id": request.form.get("app_id", ""),
        }

        if mode == "single" and not str(payload["review_text"]).strip():
            raise ValueError("Single review text is required.")
        if mode == "single" and domain == "auto":
            raise ValueError("Select a domain for the single review.")
        if mode == "google_url" and not str(payload["google_url"]).strip():
            raise ValueError("Google Play URL is required.")
        if mode == "app_id" and not str(payload["app_id"]).strip():
            raise ValueError("Google Play App ID is required.")
        if mode == "google_maps_url" and not str(payload["google_maps_url"]).strip():
            raise ValueError("Google Maps place URL is required.")

        model_status = _model_health()
        missing_models = [name for name, ready in model_status.items() if not ready]
        if missing_models:
            raise ValueError(
                "Required local models are missing or incomplete: "
                + ", ".join(missing_models)
                + ". Restore outputs/models or run the model setup script."
            )

        if payload["use_groq"]:
            if not os.environ.get("GROQ_API_KEY"):
                raise ValueError("Groq summary is enabled, but GROQ_API_KEY is not configured in .env.")
            if importlib.util.find_spec("openai") is None:
                raise ValueError("Groq summary is enabled, but the OpenAI SDK is missing. Install requirements.txt.")

        if mode in {"google_url", "app_id"} and importlib.util.find_spec("google_play_scraper") is None:
            raise ValueError("Google Play workflow requires google-play-scraper. Install requirements.txt.")

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        file = request.files.get("csv_file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            if not filename or Path(filename).suffix.lower() != ".csv":
                raise ValueError("Only .csv dataset uploads are accepted.")
            saved_path = UPLOAD_DIR / f"{job_id}_{filename}"
            file.save(saved_path)
            payload["uploaded_file"] = str(saved_path)
        elif mode == "csv":
            # CSV mode may intentionally use the bundled demonstration dataset.
            payload["uploaded_file"] = ""

        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "trace": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        thread = threading.Thread(
            target=_run_orchestrator_job,
            args=(job_id, payload),
            daemon=True,
            name=f"review-analysis-{job_id}",
        )
        thread.start()
        return _no_cache_json({"job_id": job_id, "status": "queued"})

    except ValueError as exc:
        return _no_cache_json({"error": str(exc)}, 400)


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_json_safe(job))


@app.route("/api/download/<job_id>/<file_name>")
def download_file(job_id: str, file_name: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    run_dir = Path(job.get("run_dir", ""))
    if not run_dir.exists():
        return jsonify({"error": "Run folder not found"}), 404

    final_report_path = run_dir / "final_groq_report.txt"
    if not final_report_path.exists():
        final_report_path = run_dir / "local_final_summary.txt"

    allowed = {
        "final_report": final_report_path,
        "orchestrator_state": run_dir / "orchestrator_state.json",
        "review_results": run_dir / "analysis_pipeline" / "multidomain_review_level_results.csv",
        "entity_summary": run_dir / "analysis_pipeline" / "multidomain_entity_level_summary.csv",
        "prepared_dataset": run_dir / "prepared_standardised_dataset.csv",
    }

    artifact_value = (job.get("input_artifacts") or {}).get(file_name)
    path = Path(artifact_value) if artifact_value else allowed.get(file_name)
    if not path or not path.exists():
        return jsonify({"error": "Requested file is not available"}), 404

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    print("\nReview Trust AI Flask UI running")
    print("Open: http://127.0.0.1:5000")
    debug_mode = os.environ.get("FLASK_DEBUG", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=debug_mode,
        use_reloader=False,
        threaded=True,
    )

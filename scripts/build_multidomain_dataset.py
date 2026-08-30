from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
VALID_LABELS = ["negative", "neutral", "positive"]

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "scripts"
    else Path.cwd()
)

HOTEL_NEG_PLACEHOLDERS = {
    "no negative", "none", "nothing", "nothing really", "nothing at all",
    "n/a", "na", "nil", "no complaints", "no complaint", "no issues",
    "nothing to dislike", "nothing negative", "all good", "everything was perfect",
}
HOTEL_POS_PLACEHOLDERS = {
    "no positive", "none", "nothing", "nothing really", "nothing at all",
    "n/a", "na", "nil", "nothing positive", "nothing special",
}

FINAL_COLUMNS = [
    "review_id",
    "domain",
    "entity_id",
    "entity_name",
    "review_text",
    "sentiment_text",
    "sentiment_label",
    "rating",
    "rating_original",
    "review_date",
    "source",
    "raw_source_path",
    "label_method",
    "text_key",
]


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_series(series: pd.Series) -> pd.Series:
    return series.map(clean_text_value)


def normalized_text_key(series: pd.Series) -> pd.Series:
    s = clean_series(series).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"[\d_]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.util.hash_pandas_object(
        s, index=False
    ).astype("uint64").astype(str)


def word_count(series: pd.Series) -> pd.Series:
    return clean_series(series).str.count(r"\S+")


def sentiment_from_5star(rating: pd.Series) -> pd.Series:
    r = pd.to_numeric(rating, errors="coerce")
    out = pd.Series(index=rating.index, dtype="object")
    out.loc[r <= 2] = "negative"
    out.loc[r == 3] = "neutral"
    out.loc[r >= 4] = "positive"
    return out


def drop_conflicting_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)

    label_nunique = df.groupby("text_key", sort=False)["sentiment_label"].nunique()
    conflict_keys = set(label_nunique.index[label_nunique > 1])
    conflict_rows = int(df["text_key"].isin(conflict_keys).sum())

    if conflict_keys:
        df = df[~df["text_key"].isin(conflict_keys)].copy()

    before_dup = len(df)
    df = df.drop_duplicates(subset=["text_key"], keep="first").copy()
    duplicates_removed = before_dup - len(df)

    return df.reset_index(drop=True), {
        "rows_before_conflict_dedup": int(before),
        "conflicting_text_groups_removed": int(len(conflict_keys)),
        "conflicting_rows_removed": int(conflict_rows),
        "duplicate_rows_removed": int(duplicates_removed),
        "rows_after_conflict_dedup": int(len(df)),
    }


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[FINAL_COLUMNS].copy()


def build_mobile(path: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {"content", "score", "appId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mobile dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["review_text"] = clean_series(work["content"])
    work["rating"] = pd.to_numeric(work["score"], errors="coerce")
    work["sentiment_label"] = sentiment_from_5star(work["rating"])
    work["wc"] = word_count(work["review_text"])
    work["text_key"] = normalized_text_key(work["review_text"])

    work = work[
        work["rating"].between(1, 5)
        & work["sentiment_label"].isin(VALID_LABELS)
        & work["review_text"].ne("")
        & work["wc"].ge(min_words)
        & work["text_key"].ne("")
    ].copy()

    out = pd.DataFrame(index=work.index)
    if "reviewId" in work.columns:
        ids = work["reviewId"].fillna("").astype(str)
        fallback = "mobile_" + work["_row"].astype(str)
        out["review_id"] = ids.mask(ids.eq(""), fallback)
    else:
        out["review_id"] = "mobile_" + work["_row"].astype(str)

    out["domain"] = "mobile_app"
    out["entity_id"] = work["appId"].fillna("").astype(str).values
    if "appName" in work.columns:
        names = work["appName"].fillna("").astype(str)
        out["entity_name"] = names.mask(names.eq(""), work["appId"].astype(str)).values
    else:
        out["entity_name"] = work["appId"].fillna("").astype(str).values

    out["review_text"] = work["review_text"].values
    out["sentiment_text"] = work["review_text"].values
    out["sentiment_label"] = work["sentiment_label"].values
    out["rating"] = work["rating"].values
    out["rating_original"] = work["rating"].values
    out["review_date"] = work["at"].values if "at" in work.columns else ""
    out["source"] = "google_play_reviews"
    out["raw_source_path"] = str(path)
    out["label_method"] = "rating-derived: 1-2 negative, 3 neutral, 4-5 positive"
    out["text_key"] = work["text_key"].values

    out, dedup = drop_conflicting_texts(ensure_columns(out.reset_index(drop=True)))
    return out, {
        "domain": "mobile_app",
        "raw_rows": int(len(df)),
        "rows_after_quality_filter": int(len(work)),
        **dedup,
    }


def build_amazon(path: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {"text", "rating", "asin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Amazon dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))

    # IMPORTANT: title is deliberately NOT concatenated with review text.
    # Some titles explicitly say "Five Stars", "One Star", etc. and would leak
    # the rating-derived target to sentiment/rating models.
    work["review_text"] = clean_series(work["text"])
    work["rating_num"] = pd.to_numeric(work["rating"], errors="coerce")
    work["sentiment_label"] = sentiment_from_5star(work["rating_num"])
    work["wc"] = word_count(work["review_text"])
    work["text_key"] = normalized_text_key(work["review_text"])

    work = work[
        work["rating_num"].between(1, 5)
        & work["sentiment_label"].isin(VALID_LABELS)
        & work["review_text"].ne("")
        & work["wc"].ge(min_words)
        & work["text_key"].ne("")
    ].copy()

    out = pd.DataFrame(index=work.index)
    out["review_id"] = "amazon_" + work["_row"].astype(str)
    out["domain"] = "ecommerce"
    out["entity_id"] = work["asin"].fillna("").astype(str).values

    if "parent_asin" in work.columns:
        parent = work["parent_asin"].fillna("").astype(str)
        out["entity_name"] = parent.mask(parent.eq(""), work["asin"].astype(str)).values
    else:
        out["entity_name"] = work["asin"].fillna("").astype(str).values

    out["review_text"] = work["review_text"].values
    out["sentiment_text"] = work["review_text"].values
    out["sentiment_label"] = work["sentiment_label"].values
    out["rating"] = work["rating_num"].values
    out["rating_original"] = work["rating_num"].values
    out["review_date"] = work["timestamp"].values if "timestamp" in work.columns else ""
    out["source"] = "amazon_all_beauty"
    out["raw_source_path"] = str(path)
    out["label_method"] = (
        "rating-derived: 1-2 negative, 3 neutral, 4-5 positive; "
        "review body only, title excluded to prevent star-title leakage"
    )
    out["text_key"] = work["text_key"].values

    out, dedup = drop_conflicting_texts(ensure_columns(out.reset_index(drop=True)))
    return out, {
        "domain": "ecommerce",
        "raw_rows": int(len(df)),
        "rows_after_quality_filter": int(len(work)),
        "amazon_title_used_for_model_text": False,
        **dedup,
    }


def build_restaurant(path: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {"Review Text", "Rating", "Yelp URL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Restaurant dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["review_text"] = clean_series(work["Review Text"])
    work["rating_num"] = pd.to_numeric(work["Rating"], errors="coerce")
    work["sentiment_label"] = sentiment_from_5star(work["rating_num"])
    work["wc"] = word_count(work["review_text"])
    work["text_key"] = normalized_text_key(work["review_text"])

    work = work[
        work["rating_num"].between(1, 5)
        & work["sentiment_label"].isin(VALID_LABELS)
        & work["review_text"].ne("")
        & work["wc"].ge(min_words)
        & work["text_key"].ne("")
    ].copy()

    out = pd.DataFrame(index=work.index)
    out["review_id"] = "restaurant_" + work["_row"].astype(str)
    out["domain"] = "restaurant"
    out["entity_id"] = work["Yelp URL"].fillna("").astype(str).values
    out["entity_name"] = work["Yelp URL"].fillna("").astype(str).values
    out["review_text"] = work["review_text"].values
    out["sentiment_text"] = work["review_text"].values
    out["sentiment_label"] = work["sentiment_label"].values
    out["rating"] = work["rating_num"].values
    out["rating_original"] = work["rating_num"].values
    out["review_date"] = work["Date"].values if "Date" in work.columns else ""
    out["source"] = "yelp_restaurant_reviews"
    out["raw_source_path"] = str(path)
    out["label_method"] = "rating-derived: 1-2 negative, 3 neutral, 4-5 positive"
    out["text_key"] = work["text_key"].values

    out, dedup = drop_conflicting_texts(ensure_columns(out.reset_index(drop=True)))
    return out, {
        "domain": "restaurant",
        "raw_rows": int(len(df)),
        "rows_after_quality_filter": int(len(work)),
        **dedup,
    }


def build_hotel(path: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {"Negative_Review", "Positive_Review", "Reviewer_Score", "Hotel_Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hotel dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["neg"] = clean_series(work["Negative_Review"])
    work["pos"] = clean_series(work["Positive_Review"])
    work["neg_key_plain"] = (
        work["neg"].str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    work["pos_key_plain"] = (
        work["pos"].str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    work["neg_wc"] = word_count(work["neg"])
    work["pos_wc"] = word_count(work["pos"])
    work["score"] = pd.to_numeric(work["Reviewer_Score"], errors="coerce")

    neg_valid = (
        ~work["neg_key_plain"].isin(HOTEL_NEG_PLACEHOLDERS)
        & work["neg"].ne("")
        & work["neg_wc"].ge(min_words)
    )
    pos_valid = (
        ~work["pos_key_plain"].isin(HOTEL_POS_PLACEHOLDERS)
        & work["pos"].ne("")
        & work["pos_wc"].ge(min_words)
    )

    # Keep one original hotel review per row.
    # sentiment_text is a task-specific high-confidence training view;
    # review_text preserves the complete original review evidence for
    # rating prediction, MiniLM, discrepancy and the final pipeline.
    neg_mask = neg_valid & work["score"].le(5.0)
    pos_mask = pos_valid & work["score"].ge(8.0)

    ratio = work["neg_wc"] / work["pos_wc"].replace(0, np.nan)
    neutral_mask = (
        neg_valid
        & pos_valid
        & work["score"].between(5.5, 7.5)
        & ratio.between(0.5, 2.0)
    )

    selected = work.loc[neg_mask | pos_mask | neutral_mask].copy()

    selected["sentiment_label"] = ""
    selected.loc[neg_mask.loc[selected.index], "sentiment_label"] = "negative"
    selected.loc[pos_mask.loc[selected.index], "sentiment_label"] = "positive"
    selected.loc[neutral_mask.loc[selected.index], "sentiment_label"] = "neutral"

    # Remove placeholder side before building the full review text.
    selected["neg_full"] = selected["neg"].where(
        ~selected["neg_key_plain"].isin(HOTEL_NEG_PLACEHOLDERS), ""
    )
    selected["pos_full"] = selected["pos"].where(
        ~selected["pos_key_plain"].isin(HOTEL_POS_PLACEHOLDERS), ""
    )
    selected["review_text"] = (
        selected["neg_full"].str.strip()
        + " "
        + selected["pos_full"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    selected["sentiment_text"] = selected["review_text"]
    selected.loc[selected["sentiment_label"].eq("negative"), "sentiment_text"] = (
        selected.loc[selected["sentiment_label"].eq("negative"), "neg"]
    )
    selected.loc[selected["sentiment_label"].eq("positive"), "sentiment_text"] = (
        selected.loc[selected["sentiment_label"].eq("positive"), "pos"]
    )

    selected["text_key"] = normalized_text_key(selected["sentiment_text"])

    selected = selected[
        selected["review_text"].ne("")
        & selected["sentiment_text"].ne("")
        & selected["sentiment_label"].isin(VALID_LABELS)
        & selected["text_key"].ne("")
    ].copy()

    out = pd.DataFrame(index=selected.index)
    out["review_id"] = "hotel_" + selected["_row"].astype(str)
    out["domain"] = "hotel"
    out["entity_id"] = selected["Hotel_Name"].fillna("").astype(str).values
    out["entity_name"] = selected["Hotel_Name"].fillna("").astype(str).values
    out["review_text"] = selected["review_text"].values
    out["sentiment_text"] = selected["sentiment_text"].values
    out["sentiment_label"] = selected["sentiment_label"].values

    # Operational rating remains the full original review score, normalised to 1-5.
    out["rating_original"] = selected["score"].values
    out["rating"] = (selected["score"] / 2.0).clip(1, 5).values
    out["review_date"] = selected["Review_Date"].values if "Review_Date" in selected.columns else ""
    out["source"] = "515k_european_hotel_reviews"
    out["raw_source_path"] = str(path)
    out["label_method"] = np.select(
        [
            selected["sentiment_label"].eq("negative"),
            selected["sentiment_label"].eq("positive"),
            selected["sentiment_label"].eq("neutral"),
        ],
        [
            "substantive Negative_Review + Reviewer_Score <= 5.0",
            "substantive Positive_Review + Reviewer_Score >= 8.0",
            "both fields substantive + score 5.5-7.5 + balanced field lengths",
        ],
        default="",
    )
    out["text_key"] = selected["text_key"].values

    out, dedup = drop_conflicting_texts(ensure_columns(out.reset_index(drop=True)))

    return out, {
        "domain": "hotel",
        "raw_rows": int(len(df)),
        "eligible_negative_rows": int(neg_mask.sum()),
        "eligible_neutral_rows": int(neutral_mask.sum()),
        "eligible_positive_rows": int(pos_mask.sum()),
        "rows_after_quality_filter": int(len(selected)),
        **dedup,
    }


def remove_cross_domain_duplicate_texts(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    all_keys = pd.concat(
        [
            frame[["text_key", "domain", "sentiment_label"]]
            for frame in frames.values()
        ],
        ignore_index=True,
    )

    domain_n = all_keys.groupby("text_key", sort=False)["domain"].nunique()
    cross_domain_keys = set(domain_n.index[domain_n > 1])

    label_n = all_keys.groupby("text_key", sort=False)["sentiment_label"].nunique()
    cross_label_keys = set(label_n.index[label_n > 1])

    blocked = cross_domain_keys | cross_label_keys

    clean = {}
    for domain, frame in frames.items():
        clean[domain] = frame[~frame["text_key"].isin(blocked)].copy().reset_index(drop=True)

    print(f"Cross-domain duplicate text groups removed: {len(cross_domain_keys):,}")
    print(f"Cross-label conflicting text groups removed: {len(cross_label_keys):,}")
    return clean


def sample_domain(frame: pd.DataFrame, n: int, domain: str) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(
            f"{domain} has only {len(frame):,} clean rows; cannot sample {n:,} without replacement."
        )
    return frame.sample(n=n, random_state=SEED, replace=False).copy()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build one canonical cleaned multi-domain dataset from the four raw sources. "
            "The same combined CSV is reused by DistilBERT, NLP Town rating prediction, "
            "MiniLM issue mining, discrepancy analysis and the final pipeline."
        )
    )
    parser.add_argument("--hotel", default="data/raw/hotel_europe/Hotel_Reviews.csv")
    parser.add_argument("--mobile", default="data/raw/mobile_app/google_play_reviews.csv")
    parser.add_argument("--amazon", default="data/raw/ecommerce_amazon/amazon_all_beauty_sample.csv")
    parser.add_argument("--yelp", default="data/raw/yelp_restaurant/Yelp Restaurant Reviews.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sample-per-domain", type=int, default=5000)
    parser.add_argument("--min-words", type=int, default=4)
    args = parser.parse_args()

    raw_paths = {
        "hotel": resolve_path(args.hotel),
        "mobile_app": resolve_path(args.mobile),
        "ecommerce": resolve_path(args.amazon),
        "restaurant": resolve_path(args.yelp),
    }
    for domain, path in raw_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{domain} raw dataset not found: {path}")

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nCLEAN MULTI-DOMAIN DATASET BUILD")
    print("=" * 90)
    print("One canonical dataset will be used by all downstream model stages.\n")

    hotel, hotel_audit = build_hotel(raw_paths["hotel"], args.min_words)
    mobile, mobile_audit = build_mobile(raw_paths["mobile_app"], args.min_words)
    amazon, amazon_audit = build_amazon(raw_paths["ecommerce"], args.min_words)
    restaurant, restaurant_audit = build_restaurant(raw_paths["restaurant"], args.min_words)

    pools = {
        "hotel": hotel,
        "mobile_app": mobile,
        "ecommerce": amazon,
        "restaurant": restaurant,
    }
    pools = remove_cross_domain_duplicate_texts(pools)

    samples = {
        domain: sample_domain(frame, args.sample_per_domain, domain)
        for domain, frame in pools.items()
    }

    output_names = {
        "hotel": "hotel_normalized.csv",
        "mobile_app": "mobile_app_normalized.csv",
        "ecommerce": "amazon_normalized.csv",
        "restaurant": "yelp_restaurant_normalized.csv",
    }

    for domain in ["ecommerce", "hotel", "mobile_app", "restaurant"]:
        path = out_dir / output_names[domain]
        samples[domain].to_csv(path, index=False, encoding="utf-8")
        print(f"{domain:12s}: {len(samples[domain]):,} rows -> {path}")

    combined = pd.concat(
        [samples[d] for d in ["ecommerce", "hotel", "mobile_app", "restaurant"]],
        ignore_index=True,
    )
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if len(combined) != args.sample_per_domain * 4:
        raise AssertionError("Combined dataset size is not exactly 4 x sample-per-domain.")
    if combined["review_id"].duplicated().any():
        raise AssertionError("Duplicate review_id found in combined dataset.")
    if combined["text_key"].duplicated().any():
        raise AssertionError("Duplicate sentiment text found in combined dataset.")

    combined_path = out_dir / "combined_multidomain_reviews.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": SEED,
        "sample_per_domain": args.sample_per_domain,
        "combined_rows": int(len(combined)),
        "minimum_words": args.min_words,
        "canonical_dataset": str(combined_path),
        "important_design": {
            "single_dataset_for_all_models": True,
            "review_text": (
                "Canonical full review evidence used by NLP Town rating prediction, "
                "MiniLM issue mining, discrepancy analysis and the final operational pipeline."
            ),
            "sentiment_text": (
                "Task-specific clean sentiment input stored in the SAME CSV. "
                "For mobile/ecommerce/restaurant it equals review_text. "
                "For hotel it uses the reliable positive/negative field for polar labels "
                "and the combined fields for neutral examples."
            ),
            "amazon_title": (
                "Preserved only in raw source; not included in model review_text because explicit "
                "star phrases can leak rating-derived labels."
            ),
        },
        "domain_audits": [
            hotel_audit,
            mobile_audit,
            amazon_audit,
            restaurant_audit,
        ],
        "combined_domain_counts": combined["domain"].value_counts().to_dict(),
        "combined_sentiment_counts": combined["sentiment_label"].value_counts().to_dict(),
        "domain_sentiment_counts": {
            f"{d}|{s}": int(n)
            for (d, s), n in combined.groupby(["domain", "sentiment_label"]).size().items()
        },
    }
    audit_path = out_dir / "multidomain_cleaning_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFINAL COMBINED DATASET")
    print("=" * 90)
    print(f"Rows: {len(combined):,}")
    print(f"File: {combined_path}")
    print("\nDomain counts:")
    print(combined["domain"].value_counts().to_string())
    print("\nNatural sentiment counts:")
    print(combined["sentiment_label"].value_counts().to_string())
    print("\nDomain x sentiment:")
    print(combined.groupby(["domain", "sentiment_label"]).size().to_string())
    print("\nAudit:")
    print(audit_path)
    print("\nDONE: use this SAME combined_multidomain_reviews.csv for downstream models.")


if __name__ == "__main__":
    main()

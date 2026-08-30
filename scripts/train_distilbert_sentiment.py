from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =============================================================================
# FINAL DISTILBERT SENTIMENT TRAINING PIPELINE
# Raw data -> audited clean standardisation -> natural 20K -> 85/15 split
# -> same DistilBERT -> best validation checkpoint -> canonical final model.
# =============================================================================

SEED = 42
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
BASE_MODEL = "distilbert-base-uncased"

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "scripts"
    else Path.cwd()
)

HOTEL_NEGATIVE_PLACEHOLDERS = {
    "no negative", "none", "nothing", "nothing really", "nothing at all",
    "n/a", "na", "nil", "no complaints", "no complaint", "no issues",
    "nothing to dislike", "nothing negative", "all good", "everything was perfect",
}
HOTEL_POSITIVE_PLACEHOLDERS = {
    "no positive", "none", "nothing", "nothing really", "nothing at all",
    "n/a", "na", "nil", "nothing positive", "nothing special",
}

STANDARD_FILES = {
    "hotel": "hotel_sentiment_standardized.csv",
    "mobile_app": "mobile_app_sentiment_standardized.csv",
    "ecommerce": "amazon_sentiment_standardized.csv",
    "restaurant": "restaurant_sentiment_standardized.csv",
}


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def clean_series(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str)
    out = out.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def key_series(series: pd.Series) -> pd.Series:
    out = clean_series(series).str.lower()
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    out = out.str.replace(r"[\d_]+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def hash_key_series(series: pd.Series) -> pd.Series:
    normalized = key_series(series)
    return pd.util.hash_pandas_object(
        normalized, index=False
    ).astype("uint64").astype(str)


def wc_series(series: pd.Series) -> pd.Series:
    return clean_series(series).str.count(r"\S+")


def rating_label_series(series: pd.Series) -> pd.Series:
    r = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=series.index, dtype="object")
    labels.loc[r <= 2] = "negative"
    labels.loc[(r > 2) & (r < 4)] = "neutral"
    labels.loc[r >= 4] = "positive"
    return labels


def standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_id", "source_group_id", "domain", "source", "entity_id",
        "entity_name", "review_text", "sentiment_label", "rating_original",
        "rating_5star", "review_date", "label_method", "text_source",
        "quality_rule", "word_count", "text_key",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns].copy()


def remove_conflicts_and_duplicates(
    df: pd.DataFrame, scope_name: str
) -> Tuple[pd.DataFrame, Dict]:
    out = df[df["text_key"].astype(str).str.len() > 0].copy()
    before = len(out)

    label_nunique = out.groupby("text_key", sort=False)["sentiment_label"].nunique()
    conflict_keys = set(label_nunique.index[label_nunique > 1])
    conflict_rows = int(out["text_key"].isin(conflict_keys).sum())

    if conflict_keys:
        out = out[~out["text_key"].isin(conflict_keys)].copy()

    before_dupes = len(out)
    out = out.drop_duplicates(subset=["text_key"], keep="first").copy()
    duplicate_rows = before_dupes - len(out)

    return out.reset_index(drop=True), {
        "scope": scope_name,
        "rows_before_conflict_dedup": int(before),
        "conflicting_text_groups_removed": int(len(conflict_keys)),
        "conflicting_rows_removed": int(conflict_rows),
        "duplicate_rows_removed": int(duplicate_rows),
        "rows_after_conflict_dedup": int(len(out)),
    }


def hotel_frame(
    frame: pd.DataFrame,
    label: str,
    text: pd.Series,
    method: str,
    text_source: str,
    rule: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    row_id = frame["_row"].astype(str)

    out["source_group_id"] = "hotel_" + row_id
    out["sample_id"] = "hotel_" + row_id + "_" + label
    out["domain"] = "hotel"
    out["source"] = "515k_european_hotel_reviews"
    out["entity_id"] = frame["Hotel_Name"].fillna("").astype(str)
    out["entity_name"] = frame["Hotel_Name"].fillna("").astype(str)
    out["review_text"] = text.values
    out["sentiment_label"] = label
    out["rating_original"] = frame["score"].values
    out["rating_5star"] = frame["score"].values / 2.0
    out["review_date"] = (
        frame["Review_Date"].values if "Review_Date" in frame.columns else None
    )
    out["label_method"] = method
    out["text_source"] = text_source
    out["quality_rule"] = rule
    out["word_count"] = wc_series(out["review_text"]).values
    out["text_key"] = hash_key_series(out["review_text"]).values

    return standard_columns(out.reset_index(drop=True))


def build_hotel(path: Path, min_words: int) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {"Negative_Review", "Positive_Review", "Reviewer_Score", "Hotel_Name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Hotel dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["neg"] = clean_series(work["Negative_Review"])
    work["pos"] = clean_series(work["Positive_Review"])
    work["neg_key"] = key_series(work["neg"])
    work["pos_key"] = key_series(work["pos"])

    if "Review_Total_Negative_Word_Counts" in work.columns:
        work["neg_wc"] = pd.to_numeric(
            work["Review_Total_Negative_Word_Counts"], errors="coerce"
        ).fillna(0)
    else:
        work["neg_wc"] = wc_series(work["neg"])

    if "Review_Total_Positive_Word_Counts" in work.columns:
        work["pos_wc"] = pd.to_numeric(
            work["Review_Total_Positive_Word_Counts"], errors="coerce"
        ).fillna(0)
    else:
        work["pos_wc"] = wc_series(work["pos"])

    work["score"] = pd.to_numeric(work["Reviewer_Score"], errors="coerce")

    neg_valid = (
        ~work["neg_key"].isin(HOTEL_NEGATIVE_PLACEHOLDERS)
    ) & work["neg_wc"].ge(min_words)
    pos_valid = (
        ~work["pos_key"].isin(HOTEL_POSITIVE_PLACEHOLDERS)
    ) & work["pos_wc"].ge(min_words)

    # Hotel labels are NOT created by blindly halving the 0-10 score.
    # Negative/positive fields are used as explicit polarity evidence.
    neg_mask = neg_valid & work["score"].le(5.0)
    pos_mask = pos_valid & work["score"].ge(8.0)

    ratio = work["neg_wc"] / work["pos_wc"].replace(0, np.nan)
    neutral_mask = (
        neg_valid
        & pos_valid
        & work["score"].between(5.5, 7.5)
        & ratio.between(0.5, 2.0)
    )

    neg = work.loc[neg_mask].copy()
    pos = work.loc[pos_mask].copy()
    neu = work.loc[neutral_mask].copy()

    neg_out = hotel_frame(
        neg,
        "negative",
        neg["neg"],
        "hotel_explicit_negative_field_plus_low_score",
        "Negative_Review",
        "Substantive Negative_Review and Reviewer_Score <= 5.0",
    )
    pos_out = hotel_frame(
        pos,
        "positive",
        pos["pos"],
        "hotel_explicit_positive_field_plus_high_score",
        "Positive_Review",
        "Substantive Positive_Review and Reviewer_Score >= 8.0",
    )
    neutral_text = (
        neu["neg"].str.strip() + " " + neu["pos"].str.strip()
    ).str.strip()
    neu_out = hotel_frame(
        neu,
        "neutral",
        neutral_text,
        "hotel_balanced_mixed_fields_mid_score",
        "Negative_Review + Positive_Review",
        "Both fields substantive; score 5.5-7.5; negative/positive word-count ratio 0.5-2.0",
    )

    out = pd.concat([neg_out, neu_out, pos_out], ignore_index=True)
    out, dedup = remove_conflicts_and_duplicates(out, "hotel")

    return out, {
        "domain": "hotel",
        "raw_rows": int(len(df)),
        "negative_pool_before_dedup": int(len(neg_out)),
        "neutral_pool_before_dedup": int(len(neu_out)),
        "positive_pool_before_dedup": int(len(pos_out)),
        **dedup,
    }


def build_rating_domain(
    path: Path,
    domain: str,
    source: str,
    text_col: str,
    rating_col: str,
    entity_col: str,
    date_col: str | None,
    min_words: int,
    id_col: str | None = None,
    entity_name_col: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, low_memory=False)
    required = {text_col, rating_col, entity_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{domain} dataset missing columns: {sorted(missing)}")

    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["review_text"] = clean_series(work[text_col])
    work["sentiment_label"] = rating_label_series(work[rating_col])
    work["word_count"] = wc_series(work["review_text"])
    work["text_key"] = hash_key_series(work["review_text"])

    valid = (
        work["sentiment_label"].isin(LABELS)
        & work["review_text"].ne("")
        & work["word_count"].ge(min_words)
    )
    filtered = work.loc[valid].copy()

    out = pd.DataFrame(index=filtered.index)

    if id_col and id_col in filtered.columns:
        raw_id = filtered[id_col].fillna("").astype(str)
        fallback = domain + "_" + filtered["_row"].astype(str)
        out["source_group_id"] = raw_id.mask(raw_id.eq(""), fallback)
    else:
        out["source_group_id"] = domain + "_" + filtered["_row"].astype(str)

    out["sample_id"] = out["source_group_id"].astype(str) + "_sentiment"
    out["domain"] = domain
    out["source"] = source
    out["entity_id"] = filtered[entity_col].fillna("").astype(str).values

    if entity_name_col and entity_name_col in filtered.columns:
        out["entity_name"] = (
            filtered[entity_name_col].fillna("").astype(str).values
        )
    else:
        out["entity_name"] = (
            filtered[entity_col].fillna("").astype(str).values
        )

    out["review_text"] = filtered["review_text"].values
    out["sentiment_label"] = filtered["sentiment_label"].values

    rating_num = pd.to_numeric(filtered[rating_col], errors="coerce")
    out["rating_original"] = rating_num.values
    out["rating_5star"] = rating_num.clip(1, 5).values
    out["review_date"] = (
        filtered[date_col].values
        if date_col and date_col in filtered.columns
        else None
    )
    out["label_method"] = (
        "rating_derived_1_2_negative_3_neutral_4_5_positive"
    )
    out["text_source"] = text_col
    out["quality_rule"] = (
        f"{text_col} only; >= {min_words} words; "
        "conflicting exact texts and duplicates removed"
    )
    out["word_count"] = filtered["word_count"].values
    out["text_key"] = filtered["text_key"].values

    out = standard_columns(out.reset_index(drop=True))
    out, dedup = remove_conflicts_and_duplicates(out, domain)

    return out, {
        "domain": domain,
        "raw_rows": int(len(df)),
        "rows_after_minimum_text_filter": int(len(filtered)),
        **dedup,
    }


def clean_files_exist(clean_dir: Path) -> bool:
    return all((clean_dir / filename).exists() for filename in STANDARD_FILES.values())


def build_clean_standardized_data(args, clean_dir: Path) -> Dict[str, pd.DataFrame]:
    print("\nSTEP 1 — CLEAN SENTIMENT STANDARDISATION")
    print("=" * 96)

    raw_paths = {
        "hotel": resolve_path(args.hotel),
        "mobile": resolve_path(args.mobile),
        "amazon": resolve_path(args.amazon),
        "yelp": resolve_path(args.yelp),
    }
    for name, path in raw_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} raw dataset not found: {path}")

    clean_dir.mkdir(parents=True, exist_ok=True)

    hotel, hotel_stats = build_hotel(raw_paths["hotel"], args.min_words)

    mobile, mobile_stats = build_rating_domain(
        raw_paths["mobile"],
        "mobile_app",
        "google_play_reviews",
        "content",
        "score",
        "appId",
        "at",
        args.min_words,
        id_col="reviewId",
        entity_name_col="appId",
    )

    # Amazon uses review BODY only. Title is deliberately excluded to avoid
    # explicit "Five Stars" / "One Star" target leakage.
    amazon, amazon_stats = build_rating_domain(
        raw_paths["amazon"],
        "ecommerce",
        "amazon_all_beauty",
        "text",
        "rating",
        "asin",
        "timestamp",
        args.min_words,
        entity_name_col="parent_asin",
    )

    restaurant, restaurant_stats = build_rating_domain(
        raw_paths["yelp"],
        "restaurant",
        "yelp_restaurant_reviews",
        "Review Text",
        "Rating",
        "Yelp URL",
        "Date",
        args.min_words,
        entity_name_col="Yelp URL",
    )

    frames = {
        "hotel": hotel,
        "mobile_app": mobile,
        "ecommerce": amazon,
        "restaurant": restaurant,
    }

    for domain, frame in frames.items():
        path = clean_dir / STANDARD_FILES[domain]
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"{domain:12s}: {len(frame):,} clean rows -> {path}")

    # Also retain one audit-friendly full standardized pool.
    combined = pd.concat(frames.values(), ignore_index=True)
    combined, global_stats = remove_conflicts_and_duplicates(
        combined, "combined_cross_domain"
    )
    combined.to_csv(
        clean_dir / "combined_sentiment_standardized_all.csv",
        index=False,
        encoding="utf-8",
    )

    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "minimum_words": args.min_words,
        "hotel_method": (
            "Explicit positive/negative fields used with score quality gates; "
            "balanced mixed mid-score records used as neutral."
        ),
        "amazon_method": (
            "Review body only; title excluded from sentiment training to avoid star-title leakage."
        ),
        "domain_audits": [
            hotel_stats,
            mobile_stats,
            amazon_stats,
            restaurant_stats,
            global_stats,
        ],
    }
    (clean_dir / "sentiment_standardisation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nAll clean standardized rows: {len(combined):,} -> "
        f"{clean_dir / 'combined_sentiment_standardized_all.csv'}"
    )
    return frames


def load_clean_domain_pools(clean_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}

    for domain, filename in STANDARD_FILES.items():
        path = clean_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing standardized file: {path}")

        df = pd.read_csv(path, low_memory=False)
        required = {"domain", "review_text", "sentiment_label", "text_key"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"{path.name} missing required columns: {sorted(missing)}"
            )

        df["domain"] = df["domain"].fillna("").astype(str).str.strip().str.lower()
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        df["sentiment_label"] = (
            df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
        )
        df["text_key"] = df["text_key"].fillna("").astype(str).str.strip()

        df = df[
            df["domain"].eq(domain)
            & df["review_text"].ne("")
            & df["sentiment_label"].isin(LABELS)
            & df["text_key"].ne("")
        ].copy()

        df, _ = remove_conflicts_and_duplicates(df, domain + "_reload")
        frames[domain] = df

    return frames


def globally_unique_pools(
    frames: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    all_keys = pd.concat(
        [
            f[["text_key", "domain", "sentiment_label"]]
            for f in frames.values()
        ],
        ignore_index=True,
    )

    cross_domain_counts = all_keys.groupby("text_key", sort=False)["domain"].nunique()
    cross_domain_keys = set(
        cross_domain_counts.index[cross_domain_counts > 1]
    )

    label_counts = all_keys.groupby("text_key", sort=False)["sentiment_label"].nunique()
    conflict_keys = set(label_counts.index[label_counts > 1])

    blocked = cross_domain_keys | conflict_keys

    cleaned: Dict[str, pd.DataFrame] = {}
    for domain, df in frames.items():
        cleaned[domain] = (
            df[~df["text_key"].isin(blocked)]
            .copy()
            .reset_index(drop=True)
        )

    print(
        f"Cross-domain duplicate text keys excluded: {len(cross_domain_keys):,}"
    )
    print(
        f"Cross-label conflicting text keys excluded: {len(conflict_keys):,}"
    )
    return cleaned


def prepare_natural_20k(
    frames: Dict[str, pd.DataFrame],
    sample_per_domain: int,
    validation_size: float,
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\nSTEP 2 — NATURAL 20K DATASET")
    print("=" * 96)

    frames = globally_unique_pools(frames)

    samples = []
    for domain in ["ecommerce", "hotel", "mobile_app", "restaurant"]:
        df = frames[domain]
        if len(df) < sample_per_domain:
            raise ValueError(
                f"{domain} has only {len(df):,} eligible clean rows; "
                f"cannot sample {sample_per_domain:,} without replacement."
            )

        sample = df.sample(
            n=sample_per_domain,
            random_state=SEED,
            replace=False,
        ).copy()
        samples.append(sample)

    combined = pd.concat(samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if combined["text_key"].duplicated().any():
        raise AssertionError("Duplicate text_key found in final 20K sample.")

    strata = (
        combined["domain"].astype(str)
        + "__"
        + combined["sentiment_label"].astype(str)
    )

    train_df, val_df = train_test_split(
        combined,
        test_size=validation_size,
        random_state=SEED,
        stratify=strata,
    )
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    overlap = set(train_df["text_key"]) & set(val_df["text_key"])
    if overlap:
        raise AssertionError(
            f"Train/validation leakage detected: {len(overlap)} text keys"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        data_dir / "distilbert_sentiment_20k.csv",
        index=False,
        encoding="utf-8",
    )
    train_df.to_csv(
        data_dir / "distilbert_train.csv",
        index=False,
        encoding="utf-8",
    )
    val_df.to_csv(
        data_dir / "distilbert_validation.csv",
        index=False,
        encoding="utf-8",
    )

    distribution = (
        combined.groupby(["domain", "sentiment_label"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    distribution.to_csv(
        data_dir / "distilbert_sentiment_20k_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"Combined rows:   {len(combined):,}")
    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print("\nNatural sentiment distribution:")
    print(combined["sentiment_label"].value_counts().to_string())
    print("\nDomain x sentiment distribution:")
    print(distribution.to_string(index=False))
    print("\nLeakage check: PASS (0 duplicate text keys across split)")

    return combined, train_df, val_df


class ReviewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):
        self.texts = df["review_text"].astype(str).tolist()
        self.labels = [LABEL2ID[x] for x in df["sentiment_label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: torch.Tensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Full inverse-frequency weights gave the strongest observed
        # validation result on the audited natural 20K setup.
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def class_weights_from_train(df: pd.DataFrame) -> torch.Tensor:
    counts = df["sentiment_label"].value_counts()
    total = len(df)
    weights = []

    for label in LABELS:
        count = float(counts.get(label, 0))
        if count <= 0:
            raise ValueError(
                f"Training split has zero rows for class: {label}"
            )
        weights.append(total / (len(LABELS) * count))

    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            labels, preds, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            labels, preds, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            labels, preds, average="weighted", zero_division=0
        ),
    }


def metrics_from_arrays(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_confusion(matrix: np.ndarray, path: Path) -> None:
    pd.DataFrame(
        matrix,
        index=[f"actual_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    ).to_csv(path, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final reproducible DistilBERT sentiment training: clean raw data, "
            "sample 5K/domain naturally, split 85/15, train same DistilBERT, "
            "restore best validation checkpoint, and save canonical final model."
        )
    )

    # Raw data
    parser.add_argument(
        "--hotel",
        default="data/raw/hotel_europe/Hotel_Reviews.csv",
    )
    parser.add_argument(
        "--mobile",
        default="data/raw/mobile_app/google_play_reviews.csv",
    )
    parser.add_argument(
        "--amazon",
        default="data/raw/ecommerce_amazon/amazon_all_beauty_sample.csv",
    )
    parser.add_argument(
        "--yelp",
        default="data/raw/yelp_restaurant/Yelp Restaurant Reviews.csv",
    )

    # Data preparation
    parser.add_argument(
        "--clean-dir",
        default="data/processed/sentiment_clean",
    )
    parser.add_argument(
        "--training-data-dir",
        default="data/processed/sentiment_final",
    )
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--sample-per-domain", type=int, default=5000)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument(
        "--rebuild-clean",
        action="store_true",
        help="Force raw standardisation even when clean standardized files already exist.",
    )

    # Model/training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Used when epochs > 3; best checkpoint is always restored.",
    )
    parser.add_argument("--local-files-only", action="store_true")

    # Canonical final outputs expected by the project
    parser.add_argument(
        "--final-model-path",
        default="outputs/models/distilbert_sentiment",
    )
    parser.add_argument(
        "--reports-dir",
        default="outputs/reports",
    )

    args = parser.parse_args()

    set_seed(SEED)

    clean_dir = resolve_path(args.clean_dir)
    training_data_dir = resolve_path(args.training_data_dir)
    reports_dir = resolve_path(args.reports_dir)
    final_model_path = resolve_path(args.final_model_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_clean or not clean_files_exist(clean_dir):
        build_clean_standardized_data(args, clean_dir)
    else:
        print("\nSTEP 1 — CLEAN SENTIMENT STANDARDISATION")
        print("=" * 96)
        print("Existing audited standardized files found; reusing them.")
        print("Use --rebuild-clean only when raw source data or cleaning rules change.")

    frames = load_clean_domain_pools(clean_dir)

    combined, train_df, val_df = prepare_natural_20k(
        frames=frames,
        sample_per_domain=args.sample_per_domain,
        validation_size=args.validation_size,
        data_dir=training_data_dir,
    )

    print("\nSTEP 3 — FINAL DISTILBERT TRAINING")
    print("=" * 96)
    print("Base architecture:      distilbert-base-uncased (UNCHANGED)")
    print(f"Training rows:          {len(train_df):,}")
    print(f"Validation rows:        {len(val_df):,}")
    print(f"Epoch limit:            {args.epochs}")
    print("Primary selection metric: validation Macro F1")
    print("Anti-overfit controls:")
    print("  - fixed unseen 15% validation split")
    print("  - no duplicate text leakage across split")
    print("  - AdamW + weight decay")
    print("  - 10% warmup + linear LR decay")
    print("  - best checkpoint restored automatically")
    print("  - early stopping enabled for runs longer than 3 epochs")
    print("  - only the successfully trained best model replaces the old final model")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        BASE_MODEL,
        local_files_only=args.local_files_only,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=args.local_files_only,
    )

    train_ds = ReviewDataset(train_df, tokenizer, args.max_length)
    val_ds = ReviewDataset(val_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    weights = class_weights_from_train(train_df)
    print(
        "\nClass weights [negative, neutral, positive]:",
        [round(float(x), 4) for x in weights.tolist()],
    )

    checkpoint_dir = reports_dir / "_distilbert_training_checkpoints"
    temp_model_path = final_model_path.parent / "_distilbert_sentiment_new"

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    if temp_model_path.exists():
        shutil.rmtree(temp_model_path, ignore_errors=True)

    training_kwargs = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 16),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        optim="adamw_torch",
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=weights,
    )

    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    callbacks = []
    if args.epochs > 3:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.001,
            )
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = WeightedTrainer(**trainer_kwargs)
    train_result = trainer.train()

    # Trainer has already restored the best checkpoint because
    # load_best_model_at_end=True.
    best_eval = trainer.evaluate(eval_dataset=val_ds)

    val_output = trainer.predict(
        val_ds,
        metric_key_prefix="validation",
    )
    y_true = val_output.label_ids.astype(int)
    y_pred = np.argmax(val_output.predictions, axis=1).astype(int)

    metrics = metrics_from_arrays(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    # Save to a temporary model folder first. Only after all evaluation succeeds
    # do we replace the project's canonical model folder.
    trainer.save_model(str(temp_model_path))
    tokenizer.save_pretrained(str(temp_model_path))

    # Canonical report filenames already used by the dissertation figure script.
    metrics_path = reports_dir / "phase6_distilbert_metrics.json"
    report_path = reports_dir / "phase6_distilbert_classification_report.txt"
    confusion_path = reports_dir / "phase6_distilbert_confusion_matrix.csv"
    per_domain_path = reports_dir / "phase6_distilbert_per_domain_metrics.csv"
    history_path = reports_dir / "phase6_distilbert_training_history.csv"
    audit_path = reports_dir / "phase6_distilbert_training_audit.json"

    save_confusion(matrix, confusion_path)
    report_path.write_text(report, encoding="utf-8")

    work = val_df[["domain"]].copy()
    work["y_true"] = y_true
    work["y_pred"] = y_pred

    per_domain_rows = []
    for domain, group in work.groupby("domain", sort=True):
        m = metrics_from_arrays(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
        )
        per_domain_rows.append(
            {"domain": domain, "rows": len(group), **m}
        )
    pd.DataFrame(per_domain_rows).to_csv(
        per_domain_path,
        index=False,
        encoding="utf-8",
    )

    # Keep only evaluation rows from trainer history for an auditable
    # overfitting/generalisation trace.
    history_rows = []
    for item in trainer.state.log_history:
        if "eval_loss" in item:
            history_rows.append(item)
    pd.DataFrame(history_rows).to_csv(
        history_path,
        index=False,
        encoding="utf-8",
    )

    best_epoch = None
    best_checkpoint = str(trainer.state.best_model_checkpoint or "")
    m = re.search(r"checkpoint-(\d+)", best_checkpoint)
    if m:
        best_steps = int(m.group(1))
        steps_per_epoch = int(np.ceil(len(train_df) / args.batch_size))
        if steps_per_epoch > 0:
            best_epoch = best_steps / steps_per_epoch

    payload = {
        # Canonical keys expected by generate_dissertation_figures.py
        "eval_loss": float(best_eval.get("eval_loss", np.nan)),
        "eval_accuracy": float(metrics["accuracy"]),
        "eval_precision_macro": float(metrics["precision_macro"]),
        "eval_recall_macro": float(metrics["recall_macro"]),
        "eval_f1_macro": float(metrics["f1_macro"]),
        "eval_f1_weighted": float(metrics["f1_weighted"]),

        # Audit details
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_architecture": BASE_MODEL,
        "model_path": str(final_model_path),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "source_clean_rows": int(
            sum(len(frame) for frame in frames.values())
        ),
        "sample_per_domain": int(args.sample_per_domain),
        "validation_size": float(args.validation_size),
        "seed": SEED,
        "epochs_requested": int(args.epochs),
        "best_checkpoint": best_checkpoint,
        "best_epoch_approx": best_epoch,
        "best_validation_macro_f1_from_trainer": (
            float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else None
        ),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "max_length": int(args.max_length),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "optimizer": "adamw_torch",
        "scheduler": "linear",
        "class_weighting": "inverse-frequency weighted cross entropy",
        "selection_rule": (
            "Best checkpoint selected on held-out validation Macro F1; "
            "later worse checkpoints are not deployed."
        ),
        "hotel_label_policy": (
            "Explicit Negative_Review/Positive_Review fields with score quality gates; "
            "balanced mixed mid-score records used as neutral."
        ),
        "amazon_text_policy": (
            "Review body only; rating-revealing title excluded."
        ),
        "data_leakage_check": "0 duplicate text keys between train and validation",
    }

    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "final_model": str(final_model_path),
                "training_data": str(training_data_dir / "distilbert_train.csv"),
                "validation_data": str(training_data_dir / "distilbert_validation.csv"),
                "combined_20k": str(training_data_dir / "distilbert_sentiment_20k.csv"),
                "reports": {
                    "metrics": str(metrics_path),
                    "classification_report": str(report_path),
                    "confusion_matrix": str(confusion_path),
                    "per_domain": str(per_domain_path),
                    "training_history": str(history_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Atomic-ish final replacement: old canonical model is removed only now,
    # after the new model has trained, evaluated and saved successfully.
    if final_model_path.exists():
        shutil.rmtree(final_model_path)
    shutil.move(str(temp_model_path), str(final_model_path))

    # Experimental checkpoints are not needed after best model deployment.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("\nFINAL TRAINING COMPLETE")
    print("=" * 96)
    print(f"Best checkpoint:  {best_checkpoint}")
    if best_epoch is not None:
        print(f"Best epoch ~:     {best_epoch:.2f}")
    print(
        f"Accuracy:         {metrics['accuracy']*100:.2f}%"
    )
    print(
        f"Macro Precision:  {metrics['precision_macro']*100:.2f}%"
    )
    print(
        f"Macro Recall:     {metrics['recall_macro']*100:.2f}%"
    )
    print(
        f"Macro F1:         {metrics['f1_macro']*100:.2f}%"
    )
    print(
        f"Weighted F1:      {metrics['f1_weighted']*100:.2f}%"
    )
    print(
        f"Validation Loss:  {float(best_eval.get('eval_loss', np.nan)):.4f}"
    )
    print("\nFinal deployed model:")
    print(final_model_path)
    print("\nCanonical reports:")
    print(metrics_path)
    print(confusion_path)
    print(report_path)
    print(per_domain_path)
    print(history_path)
    print(
        "\nNOTE: A later epoch may temporarily show worse validation loss/metrics "
        "during training, but it is NOT deployed. The final folder always contains "
        "the best validation-Macro-F1 checkpoint."
    )


if __name__ == "__main__":
    main()

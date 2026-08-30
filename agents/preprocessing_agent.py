from __future__ import annotations

import re

import pandas as pd


class PreprocessingAgent:
    """Clean common-schema review text and derive rating sentiment labels."""

    def __init__(self, text_column: str, rating_column: str):
        self.text_column = text_column
        self.rating_column = rating_column

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        value = str(text).lower()
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
        # ``\w`` is Unicode-aware in Python, so Urdu/Arabic and other scripts are
        # retained rather than silently deleting non-English reviews.
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        value = re.sub(r"[\d_]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def create_sentiment_label(rating):
        try:
            numeric_rating = float(rating)
        except (TypeError, ValueError):
            return None

        if numeric_rating <= 2:
            return "negative"
        if numeric_rating < 4:
            return "neutral"
        return "positive"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in (self.text_column, self.rating_column)
            if column not in df.columns
        ]
        if missing:
            raise ValueError(f"Preprocessing input is missing columns: {missing}")

        out = df.copy()
        out = out.dropna(subset=[self.text_column, self.rating_column])

        # Full review evidence used by Rating BERT, MiniLM, RAG and risk stages.
        out["clean_review"] = out[self.text_column].apply(self.clean_text)

        # The audited canonical dataset may include a cleaner task-specific
        # sentiment_text (especially for hotel reviews). DistilBERT uses this
        # when available; live/user inputs simply fall back to full review text.
        if "sentiment_text" in out.columns:
            out["clean_sentiment_review"] = out["sentiment_text"].apply(self.clean_text)
            blank_mask = out["clean_sentiment_review"].str.len().eq(0)
            out.loc[blank_mask, "clean_sentiment_review"] = out.loc[blank_mask, "clean_review"]
        else:
            out["clean_sentiment_review"] = out["clean_review"]

        derived_labels = out[self.rating_column].apply(self.create_sentiment_label)

        # Preserve audited labels from combined_multidomain_reviews.csv when
        # they exist. For ordinary live/user inputs, derive labels from rating.
        if "sentiment_label" in out.columns:
            supplied = out["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
            valid = supplied.isin({"negative", "neutral", "positive"})
            out["sentiment_label"] = supplied.where(valid, derived_labels)
        else:
            out["sentiment_label"] = derived_labels

        out = out[out["clean_review"].str.len() > 0]
        out = out[out["clean_sentiment_review"].str.len() > 0]
        out = out.dropna(subset=["sentiment_label"])
        return out.reset_index(drop=True)

"""
Offline-local Rating Prediction and Discrepancy Agent.

Model:
outputs/models/nlptown_bert_rating

No Hugging Face repo id is used at runtime.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from services.local_model_registry import (
    enforce_offline_mode,
    get_rating_model_path,
    require_local_model,
)


class RatingPredictionDiscrepancyAgent:
    def __init__(
        self,
        model_path: str | Path | None = None,
        max_length: int = 256,
        batch_size: int = 16,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        enforce_offline_mode()

        self.model_path = require_local_model(
            model_path or get_rating_model_path(),
            "nlptown BERT rating model",
        )

        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def safe_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def safe_rating(value, default=3.0) -> float:
        try:
            if pd.isna(value):
                return float(default)
            value = float(value)
            if value > 5:
                value = value / 2
            return max(1.0, min(5.0, value))
        except Exception:
            return float(default)

    @staticmethod
    def sentiment_from_star(star: int) -> str:
        if star <= 2:
            return "negative"
        if star == 3:
            return "neutral"
        return "positive"

    @staticmethod
    def discrepancy_status(actual_rating: float, predicted_star: int) -> str:
        # Keep this binary field for backward-compatible entity-level mismatch percentages.
        return "mismatched" if abs(float(actual_rating) - int(predicted_star)) >= 2 else "matched"

    @staticmethod
    def discrepancy_level(actual_rating: float, predicted_star: int) -> str:
        """
        Human-readable discrepancy level used by the UI/dissertation:
        0 = aligned
        1 = minor difference
        2 = moderate mismatch
        3-4 = severe mismatch

        Fractional ratings (e.g. normalised hotel scores) are rounded to the
        nearest whole-star gap only for this display label.
        """
        gap = int(round(abs(float(actual_rating) - int(predicted_star))))
        if gap <= 0:
            return "aligned"
        if gap == 1:
            return "minor_difference"
        if gap == 2:
            return "moderate_mismatch"
        return "severe_mismatch"

    @staticmethod
    def discrepancy_type(actual_rating: float, predicted_star: int, predicted_sentiment: str = "") -> str:
        gap = int(predicted_star) - float(actual_rating)

        if abs(gap) < 2:
            return "no_discrepancy"

        if actual_rating >= 4 and predicted_star <= 2:
            return "high_rating_negative_text"

        if actual_rating <= 2 and predicted_star >= 4:
            return "low_rating_positive_text"

        if abs(gap) >= 3:
            return "strong_discrepancy"

        return "minor_discrepancy"

    @staticmethod
    def discrepancy_penalty(dtype: str, gap: float) -> int:
        if dtype in {"high_rating_negative_text", "low_rating_positive_text", "strong_discrepancy"}:
            return 15
        if dtype == "minor_discrepancy":
            return 8
        return 0

    def predict_star_distributions(self, texts: list[str]) -> np.ndarray:
        self.load_model()

        import torch

        all_probs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            all_probs.append(probs.detach().cpu().numpy().astype(float))

        if not all_probs:
            return np.empty((0, 5), dtype=float)
        return np.vstack(all_probs)

    def predict_stars(self, texts: list[str]) -> tuple[list[int], list[float]]:
        probs = self.predict_star_distributions(texts)
        if len(probs) == 0:
            return [], []
        pred = np.argmax(probs, axis=1)
        return (pred + 1).astype(int).tolist(), probs.max(axis=1).astype(float).tolist()

    @staticmethod
    def normalize_sentiment(value) -> str:
        s = str(value or "").strip().lower()
        if s in {"neg", "negative"}:
            return "negative"
        if s in {"pos", "positive"}:
            return "positive"
        if s in {"neu", "neutral"}:
            return "neutral"
        return s

    @staticmethod
    def safe_confidence(value) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def best_star_within_polarity(probabilities: np.ndarray, sentiment: str) -> int:
        """
        Select the strongest BERT-supported class inside the sentiment-compatible
        star region. This is used only by the extreme-conflict guard.
        """
        if sentiment == "negative":
            candidates = np.array([0, 1], dtype=int)  # 1–2 stars
        elif sentiment == "positive":
            candidates = np.array([3, 4], dtype=int)  # 4–5 stars
        else:
            candidates = np.array([1, 2, 3], dtype=int)  # 2–4 stars
        local_idx = int(candidates[np.argmax(probabilities[candidates])])
        return local_idx + 1

    def apply_extreme_conflict_guard(
        self,
        probabilities: np.ndarray,
        raw_star: int,
        raw_confidence: float,
        sentiment: str,
        sentiment_confidence: float,
    ) -> tuple[int, bool, str]:
        """
        Conservative cross-model arbitration.

        The submitted/user rating is intentionally NOT an input.

        A correction is allowed only when:
        1) the two text models are in an extreme polarity contradiction;
        2) DistilBERT sentiment confidence is exceptionally high (>= 0.98);
        3) the raw 5-class rating prediction is not strongly confident (<= 0.60).

        This prevents a clearly negative text from being shown as 5 stars (or the
        symmetric clearly positive text as 1 star) when the rating classifier is
        itself uncertain. All raw outputs are retained for auditability.
        """
        sentiment = self.normalize_sentiment(sentiment)
        sent_conf = self.safe_confidence(sentiment_confidence)

        extreme_conflict = (
            (sentiment == "negative" and int(raw_star) == 5)
            or (sentiment == "positive" and int(raw_star) == 1)
        )

        if (
            extreme_conflict
            and sent_conf >= 0.98
            and float(raw_confidence) <= 0.60
        ):
            guarded_star = self.best_star_within_polarity(probabilities, sentiment)
            return (
                int(guarded_star),
                int(guarded_star) != int(raw_star),
                "extreme_low_confidence_cross_model_guard",
            )

        return int(raw_star), False, "none"

    def process(self, df: pd.DataFrame, text_column: str = "clean_review") -> pd.DataFrame:
        out = df.copy()

        if text_column not in out.columns:
            if "review_text" in out.columns:
                text_column = "review_text"
            elif "content" in out.columns:
                text_column = "content"
            else:
                raise ValueError("No text column found for discrepancy agent.")

        rating_column = "rating" if "rating" in out.columns else "score"

        texts = out[text_column].apply(self.safe_text).tolist()
        actual_ratings = out[rating_column].apply(self.safe_rating).tolist()

        probability_matrix = self.predict_star_distributions(texts)
        raw_predicted_stars = (np.argmax(probability_matrix, axis=1) + 1).astype(int).tolist()
        raw_predicted_confidences = probability_matrix.max(axis=1).astype(float).tolist()

        if "predicted_sentiment" in out.columns:
            sentiments = out["predicted_sentiment"].apply(self.normalize_sentiment).tolist()
        else:
            sentiments = [""] * len(out)

        if "sentiment_confidence" in out.columns:
            sentiment_confidences = out["sentiment_confidence"].apply(self.safe_confidence).tolist()
        else:
            sentiment_confidences = [0.0] * len(out)

        guard_results = [
            self.apply_extreme_conflict_guard(p, raw_star, raw_conf, sent, sent_conf)
            for p, raw_star, raw_conf, sent, sent_conf in zip(
                probability_matrix,
                raw_predicted_stars,
                raw_predicted_confidences,
                sentiments,
                sentiment_confidences,
            )
        ]

        predicted_stars = [x[0] for x in guard_results]
        guard_applied = [x[1] for x in guard_results]
        guard_reasons = [x[2] for x in guard_results]
        predicted_confidences = [
            float(p[int(star) - 1])
            for p, star in zip(probability_matrix, predicted_stars)
        ]

        statuses = []
        levels = []
        types = []
        gaps = []
        penalties = []

        for actual, predicted in zip(actual_ratings, predicted_stars):
            gap = int(predicted) - float(actual)
            dtype = self.discrepancy_type(actual, predicted)
            statuses.append(self.discrepancy_status(actual, predicted))
            levels.append(self.discrepancy_level(actual, predicted))
            types.append(dtype)
            gaps.append(round(gap, 2))
            penalties.append(self.discrepancy_penalty(dtype, gap))

        out["raw_predicted_star_rating"] = raw_predicted_stars
        out["raw_predicted_star_confidence"] = [round(float(x), 4) for x in raw_predicted_confidences]

        out["predicted_star_rating"] = predicted_stars
        out["predicted_star_confidence"] = [round(float(x), 4) for x in predicted_confidences]
        out["predicted_star_sentiment"] = [self.sentiment_from_star(x) for x in predicted_stars]

        out["rating_guard_applied"] = guard_applied
        out["rating_guard_reason"] = guard_reasons
        out["rating_internal_conflict"] = [
            (
                (sent == "negative" and int(raw) == 5)
                or (sent == "positive" and int(raw) == 1)
            )
            for sent, raw in zip(sentiments, raw_predicted_stars)
        ]
        out["rating_gap"] = gaps
        out["discrepancy_status"] = statuses
        out["discrepancy_level"] = levels
        out["discrepancy_type"] = types
        out["discrepancy_penalty"] = penalties
        out["discrepancy_model_used"] = "local_nlptown_bert_extreme_conflict_guard"
        out["rating_prediction_source"] = [
            "nlptown_bert+distilbert_extreme_guard" if applied else "nlptown_bert_raw_argmax"
            for applied in guard_applied
        ]
        out["rating_inference_max_length"] = self.max_length
        out["predicted_star_expected_value"] = [
            round(float(np.sum(p * np.arange(1, 6, dtype=float))), 4)
            for p in probability_matrix
        ]
        for i in range(5):
            out[f"rating_probability_{i+1}"] = [round(float(p[i]), 6) for p in probability_matrix]

        return out


# Compatibility aliases for older pipeline imports
RatingReviewDiscrepancyAgent = RatingPredictionDiscrepancyAgent
RatingPredictionAndDiscrepancyAgent = RatingPredictionDiscrepancyAgent
DiscrepancyAgent = RatingPredictionDiscrepancyAgent

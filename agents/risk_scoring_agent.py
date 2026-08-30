"""
Phase 10: Advanced Risk Scoring Agent

This agent combines:
- actual star rating
- DistilBERT sentiment result
- MiniLM issue severity
- BERT rating-review discrepancy
- domain-specific critical issue weighting
- RAG evidence strength

Output:
- trust_score
- risk_level
- reliability_level
- score_breakdown
- dominant_factor
- clear penalty columns for dissertation reporting
"""

from __future__ import annotations

import json
import math
from typing import Dict, Any

import pandas as pd


class RiskScoringAgent:
    """
    Agent 7: Risk Scoring Agent.

    Input columns expected where available:
        rating / score
        predicted_sentiment
        sentiment_confidence
        primary_issue
        issue_severity_score
        issue_severity_level
        discrepancy_type
        discrepancy_penalty
        rag_similarity_score
        domain

    Output columns:
        trust_score
        risk_score
        risk_level
        reliability_level
        rating_penalty
        sentiment_penalty
        issue_penalty
        discrepancy_penalty
        domain_critical_penalty
        uncertainty_penalty
        evidence_strength
        dominant_factor
        score_breakdown
    """

    DOMAIN_CRITICAL_ISSUES = {
        "mobile_app": {
            "inappropriate_content": 10,
            "privacy": 10,
            "payment": 9,
            "crash": 8,
            "login": 6,
            "subscription": 5,
        },
        "hotel": {
            "cleanliness": 9,
            "booking": 6,
            "staff_service": 5,
            "room_quality": 5,
        },
        "ecommerce": {
            "fake_product": 10,
            "refund": 8,
            "damaged_item": 7,
            "product_quality": 5,
            "delivery": 4,
        },
        "restaurant": {
            "hygiene": 10,
            "food_quality": 7,
            "staff_service": 5,
            "wait_time": 3,
        },
    }

    def __init__(
        self,
        min_trust_score: int = 20,
        max_trust_score: int = 100,
        use_evidence_boost: bool = True,
    ):
        self.min_trust_score = min_trust_score
        self.max_trust_score = max_trust_score
        self.use_evidence_boost = use_evidence_boost

    # ------------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------------
    @staticmethod
    def safe_float(value, default=0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def safe_str(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def rating_penalty(self, rating_value) -> int:
        rating = self.safe_float(rating_value, default=3.0)

        if rating >= 4.5:
            return 0
        if rating >= 4.0:
            return 5
        if rating >= 3.0:
            return 10
        if rating >= 2.0:
            return 18
        return 25

    def sentiment_penalty(self, sentiment, confidence_value=None) -> int:
        sentiment = self.safe_str(sentiment).lower()
        confidence = self.safe_float(confidence_value, default=1.0)

        if sentiment == "negative":
            base = 20
        elif sentiment == "neutral":
            base = 10
        elif sentiment == "positive":
            base = 0
        else:
            base = 8

        # If model is uncertain, reduce extreme sentiment effect slightly.
        if confidence < 0.55:
            base = int(round(base * 0.75))

        return base

    def issue_penalty(self, issue, severity_score, severity_level, rag_similarity=None) -> int:
        issue = self.safe_str(issue)
        severity_level = self.safe_str(severity_level).lower()
        severity_score = self.safe_float(severity_score, default=0.0)

        if issue in ["", "no_issue"] or severity_level == "none":
            return 0

        if severity_score > 0:
            base = min(25, int(round(severity_score)))
        elif severity_level == "high":
            base = 25
        elif severity_level == "medium":
            base = 15
        elif severity_level == "low":
            base = 8
        else:
            base = 0

        # If RAG has strong evidence for this issue, keep full issue penalty.
        # If evidence is weak, slightly soften it so false positives do not dominate.
        if self.use_evidence_boost and rag_similarity is not None:
            sim = self.safe_float(rag_similarity, default=0.0)
            if sim == 0:
                return base
            if sim < 0.35:
                return max(0, int(round(base * 0.85)))
            return base

        return base

    def discrepancy_penalty_value(self, row) -> int:
        if "discrepancy_penalty" in row.index:
            value = self.safe_float(row.get("discrepancy_penalty"), default=0.0)
            if value > 0:
                return int(round(min(15, value)))

        dtype = self.safe_str(row.get("discrepancy_type", "")).lower()

        mapping = {
            "high_rating_negative_text": 15,
            "low_rating_positive_text": 15,
            "strong_discrepancy": 15,
            "moderate_discrepancy": 10,
            "minor_discrepancy": 4,
            "no_discrepancy": 0,
            "": 0,
        }

        return mapping.get(dtype, 0)

    def domain_critical_penalty(self, domain, issue) -> int:
        domain = self.safe_str(domain)
        issue = self.safe_str(issue)

        return int(self.DOMAIN_CRITICAL_ISSUES.get(domain, {}).get(issue, 0))

    def uncertainty_penalty(self, row) -> int:
        """
        Small penalty when model confidence is weak. This is not a model error;
        it signals that final interpretation should be more cautious.
        """
        sentiment_conf = self.safe_float(row.get("sentiment_confidence", 1.0), default=1.0)
        star_conf = self.safe_float(row.get("predicted_star_confidence", 1.0), default=1.0)

        penalty = 0

        if sentiment_conf < 0.55:
            penalty += 3

        if star_conf < 0.45:
            penalty += 2

        return min(5, penalty)

    @staticmethod
    def evidence_strength(similarity_value) -> str:
        try:
            sim = float(similarity_value)
        except Exception:
            return "not_available"

        if sim <= 0:
            return "not_available"
        if sim >= 0.55:
            return "strong"
        if sim >= 0.40:
            return "moderate"
        return "weak"

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------
    @staticmethod
    def risk_level_from_score(trust_score: int) -> str:
        if trust_score >= 75:
            return "low_risk"
        if trust_score >= 55:
            return "medium_risk"
        return "high_risk"

    @staticmethod
    def reliability_level_from_score(trust_score: int) -> str:
        if trust_score >= 75:
            return "high_reliability"
        if trust_score >= 55:
            return "moderate_reliability"
        return "low_reliability"

    @staticmethod
    def dominant_factor_from_penalties(penalties: Dict[str, int]) -> str:
        if not penalties:
            return "none"

        max_factor = max(penalties, key=penalties.get)
        if penalties[max_factor] <= 0:
            return "none"

        label_map = {
            "rating_penalty": "rating",
            "sentiment_penalty": "sentiment",
            "issue_penalty": "issue_severity",
            "discrepancy_penalty": "rating_review_discrepancy",
            "domain_critical_penalty": "domain_criticality",
            "uncertainty_penalty": "model_uncertainty",
        }

        return label_map.get(max_factor, max_factor)

    def create_breakdown(self, row, penalties: Dict[str, int], trust_score: int, total_penalty: int) -> str:
        breakdown = {
            "base_score": 100,
            "rating_penalty": penalties.get("rating_penalty", 0),
            "sentiment_penalty": penalties.get("sentiment_penalty", 0),
            "issue_penalty": penalties.get("issue_penalty", 0),
            "discrepancy_penalty": penalties.get("discrepancy_penalty", 0),
            "domain_critical_penalty": penalties.get("domain_critical_penalty", 0),
            "uncertainty_penalty": penalties.get("uncertainty_penalty", 0),
            "total_penalty": total_penalty,
            "trust_score": trust_score,
            "dominant_factor": self.dominant_factor_from_penalties(penalties),
            "domain": self.safe_str(row.get("domain", "")),
            "primary_issue": self.safe_str(row.get("primary_issue", "")),
            "discrepancy_type": self.safe_str(row.get("discrepancy_type", "")),
            "evidence_strength": self.evidence_strength(row.get("rag_similarity_score", 0.0)),
        }

        return json.dumps(breakdown, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def score_row(self, row) -> Dict[str, Any]:
        rating_col = "score" if "score" in row.index else "rating"
        rating_value = row.get(rating_col, 3.0)

        rating_pen = self.rating_penalty(rating_value)
        sentiment_pen = self.sentiment_penalty(
            row.get("predicted_sentiment", row.get("sentiment_label", "")),
            row.get("sentiment_confidence", 1.0),
        )

        issue_pen = self.issue_penalty(
            issue=row.get("primary_issue", "no_issue"),
            severity_score=row.get("issue_severity_score", 0),
            severity_level=row.get("issue_severity_level", "none"),
            rag_similarity=row.get("rag_similarity_score", None),
        )

        discrepancy_pen = self.discrepancy_penalty_value(row)

        critical_pen = self.domain_critical_penalty(
            row.get("domain", ""),
            row.get("primary_issue", "no_issue"),
        )

        uncertain_pen = self.uncertainty_penalty(row)

        penalties = {
            "rating_penalty": rating_pen,
            "sentiment_penalty": sentiment_pen,
            "issue_penalty": issue_pen,
            "discrepancy_penalty": discrepancy_pen,
            "domain_critical_penalty": critical_pen,
            "uncertainty_penalty": uncertain_pen,
        }

        # Cap total penalty so one row does not become unrealistic.
        total_penalty = min(80, sum(penalties.values()))
        trust_score = max(self.min_trust_score, min(self.max_trust_score, 100 - total_penalty))
        risk_score = 100 - trust_score

        dominant_factor = self.dominant_factor_from_penalties(penalties)

        return {
            **penalties,
            "total_penalty": total_penalty,
            "trust_score": int(round(trust_score)),
            "risk_score": int(round(risk_score)),
            "risk_level": self.risk_level_from_score(trust_score),
            "reliability_level": self.reliability_level_from_score(trust_score),
            "dominant_factor": dominant_factor,
            "evidence_strength": self.evidence_strength(row.get("rag_similarity_score", 0.0)),
            "score_breakdown": self.create_breakdown(row, penalties, int(round(trust_score)), int(round(total_penalty))),
        }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        scores = out.apply(self.score_row, axis=1, result_type="expand")

        for col in scores.columns:
            out[col] = scores[col]

        # Helpful readable label for downstream explanation/reporting
        out["risk_interpretation"] = out.apply(
            lambda row: (
                f"{row['risk_level']} because the main penalty came from "
                f"{row['dominant_factor']} with a total penalty of {row['total_penalty']}."
            ),
            axis=1,
        )

        return out

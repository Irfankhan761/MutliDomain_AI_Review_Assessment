"""
Phase 11: Advanced Explainability Agent + Pipeline Compatibility Hotfix.

This file fixes:
AttributeError: 'ExplainabilityAgent' object has no attribute 'create_explainability_examples'

The existing pipeline expects create_explainability_examples() after explanation generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd


class ExplainabilityAgent:
    """
    Agent 8A: Explainability Agent.

    Input:
        Review-level outputs from sentiment, discrepancy, issue mining, RAG and risk scoring.

    Output:
        explanation_text
        evidence_based_explanation
        key_reasons
        warning_flags
        recommendation
        recommendation_level
        explanation_factors
    """

    def __init__(self):
        pass

    @staticmethod
    def safe_str(value, default: str = "") -> str:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return str(value).strip()

    @staticmethod
    def safe_float(value, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def parse_breakdown(value) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(str(value))
        except Exception:
            return {}

    def risk_sentence(self, trust_score: float, risk_level: str, reliability_level: str) -> str:
        reliability_text = reliability_level.replace("_", " ")
        if risk_level == "high_risk":
            return f"The review received a trust score of {int(round(trust_score))}/100, which indicates high risk and {reliability_text}."
        if risk_level == "medium_risk":
            return f"The review received a trust score of {int(round(trust_score))}/100, which indicates medium risk and {reliability_text}."
        return f"The review received a trust score of {int(round(trust_score))}/100, which indicates low risk and {reliability_text}."

    def issue_sentence(self, row) -> str:
        issue = self.safe_str(row.get("primary_issue", "no_issue"))
        severity = self.safe_str(row.get("issue_severity_level", "none"))
        issue_label = self.safe_str(row.get("issue_label", issue))
        if issue in ["", "no_issue"] or severity == "none":
            return "No clear domain-specific user-risk issue was detected."
        return f"The main detected issue is {issue_label} ({issue}), with {severity} severity."

    def discrepancy_sentence(self, row) -> str:
        status = self.safe_str(row.get("discrepancy_status", ""))
        level = self.safe_str(row.get("discrepancy_level", ""))
        actual = row.get("rating", row.get("score", ""))
        predicted = row.get("predicted_star_rating", "")

        readable = level.replace("_", " ") if level else ""
        if level == "severe_mismatch":
            return (
                f"The submitted rating ({actual}) strongly conflicts with the review meaning, "
                f"which the rating model interpreted as about {predicted} stars; this is a severe mismatch."
            )
        if level == "moderate_mismatch":
            return (
                f"The submitted rating ({actual}) and the text-derived rating ({predicted}) differ by about "
                f"two stars, indicating a moderate mismatch."
            )
        if level == "minor_difference":
            return (
                f"The submitted rating ({actual}) and the text-derived rating ({predicted}) are only one star apart, "
                f"so this is treated as a minor difference rather than a serious mismatch."
            )
        if level == "aligned":
            return f"The submitted rating ({actual}) is aligned with the text-derived rating ({predicted})."

        # Backward compatibility for older rows without discrepancy_level.
        if status == "mismatched":
            return f"The submitted rating ({actual}) conflicts with the text-derived rating ({predicted})."
        if status == "matched":
            return f"The submitted rating ({actual}) is broadly consistent with the text-derived rating ({predicted})."
        return "Rating-review discrepancy information was not available."

    def sentiment_sentence(self, row) -> str:
        sentiment = self.safe_str(row.get("predicted_sentiment", row.get("sentiment_label", "")))
        conf = self.safe_float(row.get("sentiment_confidence", 0.0), default=0.0)
        if sentiment:
            if conf > 0:
                return f"The Transformer sentiment model classified the review as {sentiment} with confidence {conf:.2f}."
            return f"The review sentiment was classified as {sentiment}."
        return "Sentiment information was not available."

    def evidence_sentence(self, row) -> str:
        evidence = self.safe_str(row.get("rag_evidence_text", ""))
        phrase = self.safe_str(row.get("evidence_phrase", ""))
        strength = self.safe_str(row.get("evidence_strength", "not_available"))
        sim = self.safe_float(row.get("rag_similarity_score", 0.0), default=0.0)
        selected_evidence = evidence or phrase
        if selected_evidence:
            if strength and strength != "not_available":
                return f"Retrieved evidence support is {strength} (similarity {sim:.2f}). Evidence: “{selected_evidence[:220]}”."
            return f"Supporting evidence: “{selected_evidence[:220]}”."
        return "No supporting review evidence was retrieved for this row."

    def penalty_sentence(self, row) -> str:
        penalty_cols = {
            "rating_penalty": "rating",
            "sentiment_penalty": "sentiment",
            "issue_penalty": "issue severity",
            "discrepancy_penalty": "rating-review discrepancy",
            "domain_critical_penalty": "domain criticality",
            "uncertainty_penalty": "model uncertainty",
        }
        values = []
        for col, label in penalty_cols.items():
            if col in row.index:
                value = int(round(self.safe_float(row.get(col), default=0)))
                if value > 0:
                    values.append(f"{label}: {value}")
        if not values:
            return "No major penalty factor was found."
        total = int(round(self.safe_float(row.get("total_penalty", 0), default=0)))
        dominant = self.safe_str(row.get("dominant_factor", "none")).replace("_", " ")
        return f"The total penalty is {total}. Main penalty factors are {', '.join(values)}. The dominant factor is {dominant}."

    def build_key_reasons(self, row) -> str:
        reasons = []
        risk_level = self.safe_str(row.get("risk_level", ""))
        dominant = self.safe_str(row.get("dominant_factor", ""))
        if risk_level == "high_risk":
            reasons.append("high overall risk score")
        if dominant and dominant not in ["none", "no_major_risk_factor"]:
            reasons.append(f"dominant risk factor: {dominant.replace('_', ' ')}")
        issue = self.safe_str(row.get("primary_issue", "no_issue"))
        if issue != "no_issue":
            reasons.append(f"detected issue: {issue}")
        if self.safe_str(row.get("discrepancy_status", "")) == "mismatched":
            reasons.append("rating-review mismatch")
        if self.safe_str(row.get("rag_evidence_text", "")):
            reasons.append("supporting review evidence retrieved")
        if not reasons:
            reasons.append("no major risk signal detected")
        return "; ".join(reasons)

    def build_warning_flags(self, row) -> str:
        flags = []
        if self.safe_str(row.get("risk_level", "")) == "high_risk":
            flags.append("HIGH_RISK")
        issue = self.safe_str(row.get("primary_issue", "no_issue"))
        severity = self.safe_str(row.get("issue_severity_level", "none"))
        if issue != "no_issue" and severity == "high":
            flags.append("HIGH_SEVERITY_ISSUE")
        if self.safe_str(row.get("discrepancy_status", "")) == "mismatched":
            flags.append("RATING_REVIEW_MISMATCH")
        if self.safe_float(row.get("sentiment_confidence", 1), default=1) < 0.55:
            flags.append("LOW_SENTIMENT_CONFIDENCE")
        return "; ".join(flags) if flags else "NO_MAJOR_WARNING"

    def recommendation_from_row(self, row) -> Dict[str, str]:
        risk_level = self.safe_str(row.get("risk_level", ""))
        issue = self.safe_str(row.get("primary_issue", "no_issue"))
        discrepancy = self.safe_str(row.get("discrepancy_status", ""))
        if risk_level == "high_risk":
            return {
                "recommendation_level": "avoid_or_review_carefully",
                "recommendation": "This entity should be treated with caution. The user should review detailed evidence before trusting it, especially because high-risk signals were detected.",
            }
        if risk_level == "medium_risk":
            return {
                "recommendation_level": "use_with_caution",
                "recommendation": "This entity can be considered, but the user should check the detected issues and rating-review consistency before making a decision.",
            }
        if issue != "no_issue" or discrepancy == "mismatched":
            return {
                "recommendation_level": "generally_safe_but_check_issue",
                "recommendation": "This entity appears generally reliable, but the detected issue or mismatch should be checked.",
            }
        return {
            "recommendation_level": "generally_reliable",
            "recommendation": "This entity appears generally reliable based on the analysed review signals.",
        }

    def explanation_factors(self, row) -> str:
        factors = {
            "rating": row.get("rating", row.get("score", "")),
            "predicted_sentiment": row.get("predicted_sentiment", ""),
            "sentiment_confidence": row.get("sentiment_confidence", ""),
            "predicted_star_rating": row.get("predicted_star_rating", ""),
            "discrepancy_status": row.get("discrepancy_status", ""),
            "primary_issue": row.get("primary_issue", ""),
            "issue_severity_level": row.get("issue_severity_level", ""),
            "trust_score": row.get("trust_score", ""),
            "risk_level": row.get("risk_level", ""),
            "dominant_factor": row.get("dominant_factor", ""),
            "evidence_strength": row.get("evidence_strength", ""),
        }
        return json.dumps(factors, ensure_ascii=False)

    def explain_row(self, row) -> Dict[str, str]:
        """
        Build a concise, conditional explanation instead of repeating the same
        five-sentence template for every review. This remains deterministic and
        auditable; Groq is still reserved for the final entity-level summary.
        """
        trust_score = self.safe_float(row.get("trust_score", 0), default=0)
        risk_level = self.safe_str(row.get("risk_level", "unknown_risk"))
        reliability_level = self.safe_str(row.get("reliability_level", "unknown_reliability"))
        issue = self.safe_str(row.get("primary_issue", "no_issue"))
        severity = self.safe_str(row.get("issue_severity_level", "none"))
        level = self.safe_str(row.get("discrepancy_level", ""))

        # Start from the most decision-relevant signal, so different review
        # conditions naturally produce different prose/order.
        parts = [self.risk_sentence(trust_score, risk_level, reliability_level)]

        if level in {"severe_mismatch", "moderate_mismatch"}:
            parts.append(self.discrepancy_sentence(row))

        if issue not in {"", "no_issue"} and severity != "none":
            parts.append(self.issue_sentence(row))

        parts.append(self.sentiment_sentence(row))

        if level in {"aligned", "minor_difference"}:
            parts.append(self.discrepancy_sentence(row))

        if issue in {"", "no_issue"} or severity == "none":
            parts.append(self.issue_sentence(row))

        # Penalty detail is useful for medium/high risk, but avoid cluttering
        # obviously low-risk explanations.
        if risk_level in {"medium_risk", "high_risk"}:
            parts.append(self.penalty_sentence(row))

        explanation_text = " ".join([p for p in parts if p])
        evidence_based_explanation = explanation_text + " " + self.evidence_sentence(row)
        rec = self.recommendation_from_row(row)

        return {
            "explanation_text": explanation_text,
            "evidence_based_explanation": evidence_based_explanation,
            "key_reasons": self.build_key_reasons(row),
            "warning_flags": self.build_warning_flags(row),
            "recommendation": rec["recommendation"],
            "recommendation_level": rec["recommendation_level"],
            "explanation_factors": self.explanation_factors(row),
        }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        explanations = out.apply(self.explain_row, axis=1, result_type="expand")
        for col in explanations.columns:
            out[col] = explanations[col]
        return out

    def create_explainability_examples(self, df: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
        """
        Compatibility method required by MultiDomainReviewAnalysisPipeline.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        work_df = df.copy()
        required_cols = [
            "explanation_text",
            "evidence_based_explanation",
            "key_reasons",
            "warning_flags",
            "recommendation",
            "recommendation_level",
            "explanation_factors",
        ]
        if any(col not in work_df.columns for col in required_cols):
            work_df = self.process(work_df)

        parts = []
        if "risk_level" in work_df.columns:
            parts.append(work_df[work_df["risk_level"].astype(str) == "high_risk"])
        if "discrepancy_status" in work_df.columns:
            parts.append(work_df[work_df["discrepancy_status"].astype(str) == "mismatched"])
        if "primary_issue" in work_df.columns:
            parts.append(work_df[work_df["primary_issue"].fillna("no_issue").astype(str) != "no_issue"])
        parts.append(work_df)

        example_df = pd.concat(parts, ignore_index=True)
        subset_cols = [c for c in ["review_id", "reviewId", "review_text"] if c in example_df.columns]
        if subset_cols:
            example_df = example_df.drop_duplicates(subset=subset_cols)
        else:
            example_df = example_df.drop_duplicates()

        example_df = example_df.head(sample_size)

        preferred_cols = [
            "domain", "entity_id", "entity_name", "rating", "review_text",
            "predicted_sentiment", "sentiment_confidence", "predicted_star_rating",
            "discrepancy_status", "discrepancy_type", "primary_issue", "issue_label",
            "issue_severity_level", "rag_evidence_text", "trust_score", "risk_level",
            "reliability_level", "dominant_factor", "key_reasons", "warning_flags",
            "explanation_text", "evidence_based_explanation", "recommendation", "recommendation_level",
        ]
        available_cols = [col for col in preferred_cols if col in example_df.columns]
        return example_df[available_cols].reset_index(drop=True)

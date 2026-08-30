"""
Phase 11: Advanced Entity-Level Summary Agent

This agent aggregates review-level outputs into app/product/hotel/restaurant-level summaries.

It creates:
- entity trust score
- risk distribution
- top issues
- mismatch percentage
- evidence examples
- final entity explanation
- recommendation
"""

from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd


class EntityLevelSummaryAgent:
    """
    Agent 8B: Entity Summary Agent.

    Entity means:
        mobile_app  -> app
        ecommerce   -> product
        hotel       -> hotel
        restaurant  -> restaurant
    """

    DOMAIN_ENTITY_TYPE = {
        "mobile_app": "app",
        "ecommerce": "product",
        "hotel": "hotel",
        "restaurant": "restaurant",
        "multidomain": "entity",
    }

    def __init__(self, top_n_issues: int = 5, top_n_evidence: int = 3):
        self.top_n_issues = top_n_issues
        self.top_n_evidence = top_n_evidence

    @staticmethod
    def safe_float(value, default=0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def safe_str(value, default="") -> str:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return str(value).strip()

    @staticmethod
    def risk_from_score(score: float) -> str:
        if score >= 75:
            return "low_risk"
        if score >= 55:
            return "medium_risk"
        return "high_risk"

    @staticmethod
    def reliability_from_score(score: float) -> str:
        if score >= 75:
            return "high_reliability"
        if score >= 55:
            return "moderate_reliability"
        return "low_reliability"

    def get_entity_type(self, domain: str) -> str:
        return self.DOMAIN_ENTITY_TYPE.get(domain, "entity")

    def top_issues(self, group: pd.DataFrame) -> str:
        if "primary_issue" not in group.columns:
            return "none"

        issues = group["primary_issue"].fillna("no_issue").astype(str)
        issues = issues[issues != "no_issue"]

        if len(issues) == 0:
            return "none"

        counts = issues.value_counts().head(self.top_n_issues)
        return "; ".join([f"{issue} ({count})" for issue, count in counts.items()])

    def risk_distribution_json(self, group: pd.DataFrame) -> str:
        if "risk_level" not in group.columns:
            return "{}"

        counts = group["risk_level"].fillna("unknown").astype(str).value_counts().to_dict()
        return json.dumps(counts, ensure_ascii=False)

    def sentiment_distribution_json(self, group: pd.DataFrame) -> str:
        if "predicted_sentiment" not in group.columns:
            return "{}"

        counts = group["predicted_sentiment"].fillna("unknown").astype(str).value_counts().to_dict()
        return json.dumps(counts, ensure_ascii=False)

    def dominant_factor_distribution_json(self, group: pd.DataFrame) -> str:
        if "dominant_factor" not in group.columns:
            return "{}"

        counts = group["dominant_factor"].fillna("unknown").astype(str).value_counts().to_dict()
        return json.dumps(counts, ensure_ascii=False)

    def mismatch_percentage(self, group: pd.DataFrame) -> float:
        if "discrepancy_status" not in group.columns or len(group) == 0:
            return 0.0

        mismatched = (group["discrepancy_status"].fillna("").astype(str) == "mismatched").mean()
        return round(float(mismatched) * 100, 2)

    def high_risk_percentage(self, group: pd.DataFrame) -> float:
        if "risk_level" not in group.columns or len(group) == 0:
            return 0.0

        high = (group["risk_level"].fillna("").astype(str) == "high_risk").mean()
        return round(float(high) * 100, 2)

    def evidence_examples(self, group: pd.DataFrame) -> str:
        evidence_col = None

        if "rag_evidence_text" in group.columns:
            evidence_col = "rag_evidence_text"
        elif "evidence_phrase" in group.columns:
            evidence_col = "evidence_phrase"

        if evidence_col is None:
            return ""

        evidence_rows = group[group[evidence_col].fillna("").astype(str).str.strip() != ""].copy()

        if "trust_score" in evidence_rows.columns:
            evidence_rows["_trust_sort"] = pd.to_numeric(evidence_rows["trust_score"], errors="coerce").fillna(100)
            evidence_rows = evidence_rows.sort_values("_trust_sort", ascending=True)

        examples = []
        for text in evidence_rows[evidence_col].head(self.top_n_evidence).tolist():
            text = self.safe_str(text)
            if text:
                examples.append(text[:220])

        return " || ".join(examples)

    def recommendation(self, average_trust_score: float, high_risk_percentage: float, mismatch_percentage: float, top_issues: str) -> str:
        if average_trust_score < 55 or high_risk_percentage >= 40:
            return "High caution is recommended because the entity contains strong risk signals."

        if average_trust_score < 75 or high_risk_percentage >= 15 or mismatch_percentage >= 30:
            return "Use with caution and review the detected issues before trusting this entity."

        if top_issues != "none":
            return "Generally reliable, but the listed issues should still be checked."

        return "Generally reliable based on the analysed review evidence."

    def entity_explanation(self, row: Dict) -> str:
        entity_type = row.get("entity_type", "entity")
        entity_name = row.get("entity_name", "")
        score = row.get("average_trust_score", 0)
        risk = row.get("overall_risk_level", "")
        top_issues = row.get("top_issues", "none")
        mismatch = row.get("mismatch_percentage", 0)
        high_risk = row.get("high_risk_percentage", 0)

        explanation = (
            f"This {entity_type} received an average trust score of {score}/100 "
            f"and is classified as {risk}. "
            f"The rating-review mismatch rate is {mismatch}%, and {high_risk}% of analysed reviews are high risk. "
        )

        if top_issues and top_issues != "none":
            explanation += f"The main detected issues are: {top_issues}. "
        else:
            explanation += "No dominant recurring issue was detected. "

        explanation += row.get("entity_recommendation", "")

        return explanation

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()

        required = ["domain", "entity_id", "entity_name"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Entity summary missing required columns: {missing}")

        out_rows = []

        grouped = df.groupby(["domain", "entity_id", "entity_name"], dropna=False)

        for (domain, entity_id, entity_name), group in grouped:
            domain = self.safe_str(domain, "unknown")
            entity_id = self.safe_str(entity_id, "")
            entity_name = self.safe_str(entity_name, entity_id)

            total_reviews = int(len(group))

            avg_rating = round(pd.to_numeric(group.get("rating", group.get("score", pd.Series(dtype=float))), errors="coerce").mean(), 2)

            if "trust_score" in group.columns:
                avg_trust = round(pd.to_numeric(group["trust_score"], errors="coerce").mean(), 2)
                min_trust = round(pd.to_numeric(group["trust_score"], errors="coerce").min(), 2)
                max_trust = round(pd.to_numeric(group["trust_score"], errors="coerce").max(), 2)
            else:
                avg_trust = 0
                min_trust = 0
                max_trust = 0

            overall_risk = self.risk_from_score(avg_trust)
            reliability = self.reliability_from_score(avg_trust)
            top_issues = self.top_issues(group)
            mismatch_pct = self.mismatch_percentage(group)
            high_risk_pct = self.high_risk_percentage(group)
            evidence = self.evidence_examples(group)

            row = {
                "domain": domain,
                "entity_type": self.get_entity_type(domain),
                "entity_id": entity_id,
                "entity_name": entity_name,
                "total_reviews": total_reviews,
                "average_rating": avg_rating,
                "average_trust_score": avg_trust,
                "minimum_trust_score": min_trust,
                "maximum_trust_score": max_trust,
                "overall_risk_level": overall_risk,
                "overall_reliability_level": reliability,
                "high_risk_percentage": high_risk_pct,
                "mismatch_percentage": mismatch_pct,
                "top_issues": top_issues,
                "risk_distribution": self.risk_distribution_json(group),
                "sentiment_distribution": self.sentiment_distribution_json(group),
                "dominant_factor_distribution": self.dominant_factor_distribution_json(group),
                "evidence_examples": evidence,
            }

            row["entity_recommendation"] = self.recommendation(
                average_trust_score=avg_trust,
                high_risk_percentage=high_risk_pct,
                mismatch_percentage=mismatch_pct,
                top_issues=top_issues,
            )

            row["entity_explanation"] = self.entity_explanation(row)

            out_rows.append(row)

        result = pd.DataFrame(out_rows)

        if len(result) > 0:
            result = result.sort_values(
                ["average_trust_score", "high_risk_percentage", "mismatch_percentage"],
                ascending=[True, False, False],
            ).reset_index(drop=True)

        return result

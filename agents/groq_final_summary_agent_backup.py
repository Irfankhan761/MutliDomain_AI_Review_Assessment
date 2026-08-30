from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from services.groq_client import GroqClient


class GroqFinalSummaryAgent:
    """
    Final LLM summary agent using Groq.

    Output strategy:
    1. Groq receives structured model outputs.
    2. Groq returns JSON only.
    3. Backend parses that JSON.
    4. UI receives both:
       - final_summary: structured card-ready object
       - final_report: clean readable plain-text fallback
    """

    def __init__(self, env_path: str | Path = ".env"):
        self.client = GroqClient(env_path=env_path)

    @staticmethod
    def safe_str(value: Any, default: str = "") -> str:
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return default
        return text

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def to_native(value: Any) -> Any:
        """
        Converts pandas/numpy values into JSON-safe Python values.
        """
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, dict):
            return {str(k): GroqFinalSummaryAgent.to_native(v) for k, v in value.items()}

        if isinstance(value, list):
            return [GroqFinalSummaryAgent.to_native(v) for v in value]

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def clean_issue_name(issue: Any) -> str:
        text = GroqFinalSummaryAgent.safe_str(issue, "")
        if not text or text == "no_issue":
            return "No major issue"
        return text.replace("_", " ").title()

    @staticmethod
    def risk_to_readable(risk: Any) -> str:
        text = GroqFinalSummaryAgent.safe_str(risk, "").lower()
        mapping = {
            "low_risk": "Low Risk",
            "medium_risk": "Medium Risk",
            "high_risk": "High Risk",
        }
        return mapping.get(text, text.replace("_", " ").title() if text else "Not available")

    @staticmethod
    def recommendation_to_readable(value: Any) -> str:
        text = GroqFinalSummaryAgent.safe_str(value, "").lower()
        mapping = {
            "generally_reliable": "Generally reliable",
            "generally_safe_but_check_issue": "Generally safe, but check issues",
            "use_with_caution": "Use with caution",
            "avoid_or_review_carefully": "Avoid or review carefully",
        }
        return mapping.get(text, text.replace("_", " ").title() if text else "Review carefully")

    def select_entity_data(
        self,
        review_df: pd.DataFrame,
        entity_df: pd.DataFrame,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        filtered_entities = entity_df.copy()

        if domain and "domain" in filtered_entities.columns:
            filtered_entities = filtered_entities[
                filtered_entities["domain"].astype(str) == str(domain)
            ]

        if entity_id and "entity_id" in filtered_entities.columns:
            filtered_entities = filtered_entities[
                filtered_entities["entity_id"].astype(str) == str(entity_id)
            ]

        if entity_name and "entity_name" in filtered_entities.columns:
            filtered_entities = filtered_entities[
                filtered_entities["entity_name"]
                .astype(str)
                .str.contains(str(entity_name), case=False, na=False, regex=False)
            ]

        if len(filtered_entities) == 0:
            filtered_entities = entity_df.copy()

        if "average_trust_score" in filtered_entities.columns:
            # For final report, select the riskiest entity first.
            filtered_entities = filtered_entities.sort_values(
                "average_trust_score", ascending=True
            )

        selected_entity = filtered_entities.iloc[0].to_dict()

        selected_reviews = review_df.copy()
        selected_entity_id = selected_entity.get("entity_id", "")
        selected_entity_name = selected_entity.get("entity_name", "")
        selected_domain = selected_entity.get("domain", "")

        if selected_entity_id and "entity_id" in selected_reviews.columns:
            selected_reviews = selected_reviews[
                selected_reviews["entity_id"].astype(str) == str(selected_entity_id)
            ]

        if len(selected_reviews) == 0 and selected_entity_name and "entity_name" in selected_reviews.columns:
            selected_reviews = selected_reviews[
                selected_reviews["entity_name"].astype(str) == str(selected_entity_name)
            ]

        if selected_domain and "domain" in selected_reviews.columns:
            selected_reviews = selected_reviews[
                selected_reviews["domain"].astype(str) == str(selected_domain)
            ]

        if len(selected_reviews) == 0:
            selected_reviews = review_df.copy()

        if "trust_score" in selected_reviews.columns:
            selected_reviews = selected_reviews.sort_values("trust_score", ascending=True)

        return {
            "selected_entity": self.to_native(selected_entity),
            "selected_reviews": self.to_native(
                selected_reviews.head(8).to_dict(orient="records")
            ),
        }

    def create_context_payload(self, selected_data: Dict[str, Any]) -> str:
        entity = selected_data["selected_entity"]
        compact_reviews = []

        for row in selected_data["selected_reviews"]:
            compact_reviews.append(
                {
                    "domain": row.get("domain"),
                    "rating": row.get("rating", row.get("score")),
                    "review_text": self.safe_str(row.get("review_text", row.get("content", "")))[:700],
                    "predicted_sentiment": row.get("predicted_sentiment"),
                    "sentiment_confidence": row.get("sentiment_confidence"),
                    "predicted_star_rating": row.get("predicted_star_rating"),
                    "discrepancy_status": row.get("discrepancy_status"),
                    "discrepancy_type": row.get("discrepancy_type"),
                    "primary_issue": row.get("primary_issue"),
                    "issue_label": row.get("issue_label"),
                    "issue_severity_level": row.get("issue_severity_level"),
                    "rag_evidence_text": self.safe_str(
                        row.get("rag_evidence_text", row.get("evidence_phrase", ""))
                    )[:500],
                    "trust_score": row.get("trust_score"),
                    "risk_level": row.get("risk_level"),
                    "dominant_factor": row.get("dominant_factor"),
                    "key_reasons": row.get("key_reasons"),
                    "warning_flags": row.get("warning_flags"),
                    "recommendation_level": row.get("recommendation_level"),
                }
            )

        payload = {
            "entity_summary": {
                "domain": entity.get("domain"),
                "entity_type": entity.get("entity_type"),
                "entity_name": entity.get("entity_name"),
                "total_reviews": entity.get("total_reviews"),
                "average_rating": entity.get("average_rating"),
                "average_trust_score": entity.get("average_trust_score"),
                "overall_risk_level": entity.get("overall_risk_level"),
                "overall_reliability_level": entity.get("overall_reliability_level"),
                "high_risk_percentage": entity.get("high_risk_percentage"),
                "mismatch_percentage": entity.get("mismatch_percentage"),
                "top_issues": entity.get("top_issues"),
                "evidence_examples": entity.get("evidence_examples"),
                "entity_explanation": entity.get("entity_explanation"),
                "entity_recommendation": entity.get("entity_recommendation"),
            },
            "supporting_review_examples": compact_reviews,
        }

        return json.dumps(self.to_native(payload), indent=2, ensure_ascii=False)

    def build_prompt(self, context_payload: str) -> str:
        return f"""
You are the final summary agent for  multi-domain review trust assessment system.

Your task:
Convert the supplied structured analysis data into a simple, user-friendly trust decision.

Use only the supplied JSON context.
Do not invent reviews, numbers, issues, ratings, risks, or recommendations.
Do not use markdown.
Do not use headings with asterisks.
Do not wrap the answer in code fences.
Return VALID JSON only.

The JSON must follow this exact structure:

{{
  "title": "short title using the entity name",
  "entity_name": "entity name",
  "domain": "mobile app / ecommerce / hotel / restaurant / auto domain",
  "actual_rating": "example: 2.89/5",
  "analysed_reviews": "example: 190",
  "trust_score": "example: 67.50/100",
  "risk_level": "Low Risk / Medium Risk / High Risk",
  "recommendation": "Use / Use with caution / Avoid or review carefully",
  "one_line_verdict": "one simple verdict sentence",
  "summary_paragraph": "short paragraph explaining the overall result in normal user language",
  "main_issues": [
    "issue 1 with count if available",
    "issue 2 with count if available",
    "issue 3 with count if available"
  ],
  "rating_interpretation": "explain the actual rating and what it suggests",
  "trust_interpretation": "explain what the trust score means",
  "should_user_use_it": "clear recommendation sentence",
  "why_this_decision": [
    "simple reason 1",
    "simple reason 2",
    "simple reason 3"
  ],
  "evidence_examples": [
    {{
      "issue": "issue name",
      "rating": "actual rating",
      "sentiment": "predicted sentiment",
      "trust_score": "trust score",
      "review_excerpt": "short review excerpt",
      "why_it_matters": "simple explanation"
    }}
  ],
  "technical_agents_used": [
    "Sentiment Analysis Agent",
    "Rating Prediction Agent",
    "Discrepancy Detection Agent",
    "Issue Mining Agent",
    "RAG Evidence Retrieval Agent",
    "Risk Scoring Agent"
  ]
}}

Tone:
- Use plain English.
- Write for a normal user, not a developer.
- Be practical.
- Explain whether the user should trust/use the app/product/hotel/restaurant.
- Keep the summary concise but useful.

Structured JSON context:
{context_payload}
""".strip()

    def parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Groq should return JSON, but this parser also handles accidental
        markdown fences or extra text.
        """
        if not raw_text:
            return {}

        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {}

    def build_fallback_summary(self, selected_data: Dict[str, Any]) -> Dict[str, Any]:
        entity = selected_data["selected_entity"]
        reviews = selected_data["selected_reviews"]

        entity_name = self.safe_str(entity.get("entity_name"), "Selected entity")
        domain = self.safe_str(entity.get("domain"), "auto domain")
        total_reviews = self.safe_int(entity.get("total_reviews"), len(reviews))
        avg_rating = self.safe_float(entity.get("average_rating"), 0.0)
        avg_trust = self.safe_float(entity.get("average_trust_score"), 0.0)
        risk_level = self.risk_to_readable(entity.get("overall_risk_level"))
        recommendation = self.recommendation_to_readable(entity.get("entity_recommendation"))

        top_issues_raw = self.safe_str(entity.get("top_issues"), "")
        if top_issues_raw and top_issues_raw.lower() not in {"none", "no_issue"}:
            main_issues = [x.strip() for x in top_issues_raw.split(";") if x.strip()]
        else:
            main_issues = ["No major recurring issue detected"]

        if risk_level == "High Risk":
            should_use = "Avoid or review very carefully before trusting it."
        elif risk_level == "Medium Risk":
            should_use = "Use with caution and check the detected issues before trusting it."
        else:
            should_use = "Generally safe to use, but still review important user feedback."

        evidence_examples = []
        for row in reviews[:4]:
            evidence_examples.append(
                {
                    "issue": self.clean_issue_name(row.get("primary_issue")),
                    "rating": str(row.get("rating", row.get("score", ""))),
                    "sentiment": self.safe_str(row.get("predicted_sentiment"), "not available"),
                    "trust_score": str(row.get("trust_score", "")),
                    "review_excerpt": self.safe_str(row.get("review_text", row.get("content", "")))[:250],
                    "why_it_matters": self.safe_str(row.get("evidence_based_explanation", row.get("explanation_text", "")))[:300],
                }
            )

        return {
            "title": f"{entity_name} trust summary",
            "entity_name": entity_name,
            "domain": domain.replace("_", " ").title(),
            "actual_rating": f"{avg_rating:.2f}/5" if avg_rating else "Not available",
            "analysed_reviews": str(total_reviews),
            "trust_score": f"{avg_trust:.2f}/100",
            "risk_level": risk_level,
            "recommendation": recommendation,
            "one_line_verdict": f"{entity_name} is classified as {risk_level.lower()} based on the analysed review evidence.",
            "summary_paragraph": (
                f"{entity_name} was analysed using the agentic review trust pipeline. "
                f"The system reviewed {total_reviews} review records and calculated an average trust score of "
                f"{avg_trust:.2f}/100. The overall risk level is {risk_level}. "
                f"Main detected issues: {', '.join(main_issues[:5])}."
            ),
            "main_issues": main_issues[:6],
            "rating_interpretation": (
                f"The average public rating is {avg_rating:.2f}/5. "
                f"This rating is considered together with review text, sentiment, issue severity and discrepancy signals."
            ),
            "trust_interpretation": (
                f"The trust score is {avg_trust:.2f}/100. Higher scores indicate stronger reliability; "
                f"lower scores indicate stronger warning signals."
            ),
            "should_user_use_it": should_use,
            "why_this_decision": [
                "The system combines the public rating with text-based sentiment.",
                "It compares the actual rating against the predicted rating from review meaning.",
                "It detects domain-specific issues and assigns severity.",
                "It uses risk-scoring rules to convert all signals into a trust score.",
            ],
            "evidence_examples": evidence_examples,
            "technical_agents_used": [
                "Sentiment Analysis Agent",
                "Rating Prediction Agent",
                "Discrepancy Detection Agent",
                "Issue Mining Agent",
                "RAG Evidence Retrieval Agent",
                "Risk Scoring Agent",
            ],
        }

    def normalise_summary(self, summary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the frontend always receives complete fields.
        """
        if not summary:
            return fallback

        merged = fallback.copy()
        for key, value in summary.items():
            if value not in [None, "", [], {}]:
                merged[key] = value

        if not isinstance(merged.get("main_issues"), list):
            merged["main_issues"] = fallback.get("main_issues", [])

        if not isinstance(merged.get("why_this_decision"), list):
            merged["why_this_decision"] = fallback.get("why_this_decision", [])

        if not isinstance(merged.get("evidence_examples"), list):
            merged["evidence_examples"] = fallback.get("evidence_examples", [])

        return merged

    def summary_to_plain_text(self, summary: Dict[str, Any]) -> str:
        """
        Plain-text compatibility output for older UI code.
        """
        issues = summary.get("main_issues", [])
        reasons = summary.get("why_this_decision", [])
        examples = summary.get("evidence_examples", [])

        lines = [
            "Overall Trust Summary",
            "",
            f"Entity: {summary.get('entity_name', '-')}",
            f"Domain: {summary.get('domain', '-')}",
            f"Actual rating: {summary.get('actual_rating', '-')}",
            f"Analysed reviews: {summary.get('analysed_reviews', '-')}",
            f"Trust score: {summary.get('trust_score', '-')}",
            f"Risk level: {summary.get('risk_level', '-')}",
            f"Recommendation: {summary.get('recommendation', '-')}",
            "",
            "Verdict:",
            summary.get("one_line_verdict", ""),
            "",
            "Summary:",
            summary.get("summary_paragraph", ""),
            "",
            "Main issues:",
        ]

        for issue in issues:
            lines.append(f"- {issue}")

        lines.extend(["", "Should the user use it?", summary.get("should_user_use_it", ""), ""])

        lines.append("Why this decision?")
        for reason in reasons:
            lines.append(f"- {reason}")

        if examples:
            lines.extend(["", "Important review evidence:"])
            for idx, ex in enumerate(examples[:4], start=1):
                lines.append(
                    f"{idx}. Issue: {ex.get('issue', '-')}; "
                    f"Rating: {ex.get('rating', '-')}; "
                    f"Sentiment: {ex.get('sentiment', '-')}; "
                    f"Trust: {ex.get('trust_score', '-')}. "
                    f"Review: {ex.get('review_excerpt', '-')}"
                )

        return "\n".join(lines).strip()

    def generate_report(
        self,
        review_df: pd.DataFrame,
        entity_df: pd.DataFrame,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_data = self.select_entity_data(
            review_df=review_df,
            entity_df=entity_df,
            entity_id=entity_id,
            entity_name=entity_name,
            domain=domain,
        )

        context_payload = self.create_context_payload(selected_data)
        fallback_summary = self.build_fallback_summary(selected_data)

        prompt = self.build_prompt(context_payload)

        try:
            raw_report = self.client.respond(
                input_text=prompt,
                temperature=0.15,
                max_output_tokens=1200,
            )
            parsed_summary = self.parse_json_response(raw_report)
            final_summary = self.normalise_summary(parsed_summary, fallback_summary)
        except Exception as exc:
            final_summary = fallback_summary
            final_summary["llm_error"] = str(exc)

        final_report = self.summary_to_plain_text(final_summary)

        return {
            "final_summary": final_summary,
            "final_report": final_report,
            "context_payload": context_payload,
            "selected_entity": selected_data["selected_entity"],
        }
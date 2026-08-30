# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Optional
# import pandas as pd
# from services.groq_client import GroqClient


# class GroqFinalSummaryAgent:
#     """Final LLM summary agent using Groq Responses API."""

#     def __init__(self, env_path: str | Path = '.env'):
#         self.client = GroqClient(env_path=env_path)

#     @staticmethod
#     def safe_str(value, default='') -> str:
#         try:
#             if pd.isna(value):
#                 return default
#         except Exception:
#             pass
#         return str(value).strip()

#     def select_entity_data(self, review_df: pd.DataFrame, entity_df: pd.DataFrame, entity_id: Optional[str] = None, entity_name: Optional[str] = None, domain: Optional[str] = None) -> Dict:
#         filtered_entities = entity_df.copy()
#         if domain and 'domain' in filtered_entities.columns:
#             filtered_entities = filtered_entities[filtered_entities['domain'].astype(str) == str(domain)]
#         if entity_id and 'entity_id' in filtered_entities.columns:
#             filtered_entities = filtered_entities[filtered_entities['entity_id'].astype(str) == str(entity_id)]
#         if entity_name and 'entity_name' in filtered_entities.columns:
#             filtered_entities = filtered_entities[filtered_entities['entity_name'].astype(str).str.contains(str(entity_name), case=False, na=False, regex=False)]
#         if len(filtered_entities) == 0:
#             filtered_entities = entity_df.copy()
#         if 'average_trust_score' in filtered_entities.columns:
#             filtered_entities = filtered_entities.sort_values('average_trust_score', ascending=True)
#         selected_entity = filtered_entities.iloc[0].to_dict()

#         selected_reviews = review_df.copy()
#         selected_entity_id = selected_entity.get('entity_id', '')
#         selected_domain = selected_entity.get('domain', '')
#         if selected_entity_id and 'entity_id' in selected_reviews.columns:
#             selected_reviews = selected_reviews[selected_reviews['entity_id'].astype(str) == str(selected_entity_id)]
#         if selected_domain and 'domain' in selected_reviews.columns:
#             selected_reviews = selected_reviews[selected_reviews['domain'].astype(str) == str(selected_domain)]
#         if len(selected_reviews) == 0:
#             selected_reviews = review_df.copy()
#         if 'trust_score' in selected_reviews.columns:
#             selected_reviews = selected_reviews.sort_values('trust_score', ascending=True)
#         return {'selected_entity': selected_entity, 'selected_reviews': selected_reviews.head(8).to_dict(orient='records')}

#     def create_context_payload(self, selected_data: Dict) -> str:
#         entity = selected_data['selected_entity']
#         compact_reviews = []
#         for row in selected_data['selected_reviews']:
#             compact_reviews.append({
#                 'domain': row.get('domain'),
#                 'rating': row.get('rating', row.get('score')),
#                 'review_text': self.safe_str(row.get('review_text', ''))[:500],
#                 'predicted_sentiment': row.get('predicted_sentiment'),
#                 'sentiment_confidence': row.get('sentiment_confidence'),
#                 'predicted_star_rating': row.get('predicted_star_rating'),
#                 'discrepancy_status': row.get('discrepancy_status'),
#                 'discrepancy_type': row.get('discrepancy_type'),
#                 'primary_issue': row.get('primary_issue'),
#                 'issue_severity_level': row.get('issue_severity_level'),
#                 'rag_evidence_text': self.safe_str(row.get('rag_evidence_text', row.get('evidence_phrase', '')))[:300],
#                 'trust_score': row.get('trust_score'),
#                 'risk_level': row.get('risk_level'),
#                 'dominant_factor': row.get('dominant_factor'),
#                 'key_reasons': row.get('key_reasons'),
#                 'warning_flags': row.get('warning_flags'),
#                 'recommendation_level': row.get('recommendation_level'),
#             })
#         payload = {
#             'entity_summary': {
#                 'domain': entity.get('domain'),
#                 'entity_type': entity.get('entity_type'),
#                 'entity_name': entity.get('entity_name'),
#                 'total_reviews': entity.get('total_reviews'),
#                 'average_rating': entity.get('average_rating'),
#                 'average_trust_score': entity.get('average_trust_score'),
#                 'overall_risk_level': entity.get('overall_risk_level'),
#                 'overall_reliability_level': entity.get('overall_reliability_level'),
#                 'high_risk_percentage': entity.get('high_risk_percentage'),
#                 'mismatch_percentage': entity.get('mismatch_percentage'),
#                 'top_issues': entity.get('top_issues'),
#                 'evidence_examples': entity.get('evidence_examples'),
#                 'entity_explanation': entity.get('entity_explanation'),
#                 'entity_recommendation': entity.get('entity_recommendation'),
#             },
#             'supporting_review_examples': compact_reviews,
#         }
#         return json.dumps(payload, indent=2, ensure_ascii=False)

#     def build_prompt(self, context_payload: str) -> str:
#         return f"""
# You are the final LLM supervisory/orchestrator response agent for an academic Agentic AI review-trust assessment system.

# Your job is to convert structured model outputs into a clear final trust/risk report.

# Rules:
# - Do not invent evidence.
# - Use only the provided JSON context.
# - Explain the trust score and risk level.
# - Mention sentiment, rating-review discrepancy, issue severity, RAG evidence, and recommendation where available.
# - Mention which analytical agents contributed.
# - Write in clear English.

# Structured context:
# {context_payload}

# Required output format:
# 1. Overall trust assessment
# 2. Main risk signals
# 3. Evidence from reviews
# 4. Recommendation
# 5. Technical note: which agents contributed
# """.strip()

#     def generate_report(self, review_df: pd.DataFrame, entity_df: pd.DataFrame, entity_id: Optional[str] = None, entity_name: Optional[str] = None, domain: Optional[str] = None) -> Dict:
#         selected_data = self.select_entity_data(review_df, entity_df, entity_id, entity_name, domain)
#         context_payload = self.create_context_payload(selected_data)
#         prompt = self.build_prompt(context_payload)
#         final_report = self.client.respond(input_text=prompt, temperature=0.2, max_output_tokens=900)
#         return {'final_report': final_report, 'context_payload': context_payload, 'selected_entity': selected_data['selected_entity']}
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
You are the final language-model reporting layer for an Agentic AI multi-domain review trust assessment system.

Your job is to turn validated analytical outputs into a natural, professional explanation that reads like an intelligent analyst wrote it — not like a repeated form template.

Strict grounding rules:
- Use ONLY the supplied JSON context.
- Never invent reviews, numbers, issues, ratings, risks, evidence, or recommendations.
- Do not calculate or change the Trust Score or Risk Level. They are already determined by the analytical pipeline.
- Do not claim that the language model made the risk decision.
- Keep wording natural and varied between runs while preserving the same facts.
- Avoid repetitive fixed section headings such as "Overall Trust Summary", "Verdict", "Summary", or "Why this decision?".

Return VALID JSON only so the application can safely parse the response.
The JSON keys must remain exactly as shown below for UI compatibility, but the wording inside them should be natural rather than formulaic.

{{
  "title": "short natural title using the entity name",
  "entity_name": "entity name",
  "domain": "mobile app / ecommerce / hotel / restaurant / auto domain",
  "actual_rating": "example: 2.89/5",
  "analysed_reviews": "example: 190",
  "trust_score": "example: 67.50/100",
  "risk_level": "Low Risk / Medium Risk / High Risk",
  "recommendation": "Use / Use with caution / Avoid or review carefully",
  "one_line_verdict": "one concise natural verdict sentence",
  "summary_paragraph": "a natural explanatory paragraph connecting the rating, trust score, risks and evidence",
  "main_issues": [
    "issue 1 with count if available",
    "issue 2 with count if available",
    "issue 3 with count if available"
  ],
  "rating_interpretation": "brief natural interpretation of the public rating",
  "trust_interpretation": "brief natural interpretation of the deterministic Trust Score",
  "should_user_use_it": "clear practical recommendation sentence",
  "why_this_decision": [
    "evidence-based reason 1",
    "evidence-based reason 2",
    "evidence-based reason 3"
  ],
  "evidence_examples": [
    {{
      "issue": "issue name",
      "rating": "actual rating",
      "sentiment": "predicted sentiment",
      "trust_score": "trust score",
      "review_excerpt": "short review excerpt",
      "why_it_matters": "simple evidence-based explanation"
    }}
  ],
  "technical_agents_used": [
    "Sentiment Analysis Agent",
    "Rating Prediction Agent",
    "Discrepancy Detection Agent",
    "Issue Mining Agent",
    "RAG Evidence Retrieval Agent",
    "Risk Scoring Agent"
  ],
  "natural_report": "Write 2-3 short flowing paragraphs in plain English, then 3-5 concise bullet points beginning with '- '. Do not use a fixed repeated template or numbered section headings. Mention only the most decision-relevant facts and evidence. End with a practical recommendation."
}}

Writing style for natural_report:
- Sound like a professional AI-generated analyst summary, not a form being filled in.
- Start directly with the overall interpretation; do not start with a fixed heading.
- Use 2-3 short paragraphs followed by 3-5 concise bullet points.
- Vary sentence structure naturally.
- Refer to evidence only when it exists in the context.
- If only one review was analysed, explicitly make that limitation clear.
- Keep it concise and readable.

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
            "natural_report": "",
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
        Produce the human-facing report.

        Groq is encouraged to provide a natural_report containing flowing prose
        plus concise bullets. Structured fields are still retained separately for
        UI cards and validation. If Groq omits natural_report, a grounded natural
        fallback is assembled without the old repeated section template.
        """
        natural_report = self.safe_str(summary.get("natural_report"), "")
        if natural_report:
            return natural_report.strip()

        entity = self.safe_str(summary.get("entity_name"), "The analysed entity")
        domain = self.safe_str(summary.get("domain"), "entity")
        rating = self.safe_str(summary.get("actual_rating"), "not available")
        reviews = self.safe_str(summary.get("analysed_reviews"), "not available")
        trust = self.safe_str(summary.get("trust_score"), "not available")
        risk = self.safe_str(summary.get("risk_level"), "not available")
        verdict = self.safe_str(summary.get("one_line_verdict"), "")
        paragraph = self.safe_str(summary.get("summary_paragraph"), "")
        recommendation = self.safe_str(summary.get("should_user_use_it"), "")
        issues = summary.get("main_issues", []) if isinstance(summary.get("main_issues"), list) else []
        reasons = summary.get("why_this_decision", []) if isinstance(summary.get("why_this_decision"), list) else []

        paragraphs = [
            (
                f"{entity} ({domain}) was assessed across {reviews} review(s). "
                f"Its public rating is {rating}, while the deterministic trust score is {trust}, "
                f"placing it in the {risk} category. {verdict}"
            ).strip(),
            paragraph.strip(),
        ]
        paragraphs = [p for p in paragraphs if p]

        bullets = []
        for issue in issues[:3]:
            bullets.append(f"- Key issue: {issue}")
        for reason in reasons[:3]:
            if len(bullets) >= 5:
                break
            bullets.append(f"- {reason}")
        if recommendation:
            bullets.append(f"- Recommendation: {recommendation}")

        return "\n\n".join(paragraphs + (["\n".join(bullets)] if bullets else [])).strip()

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
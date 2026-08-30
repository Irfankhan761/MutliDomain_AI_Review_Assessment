from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd

from agents.preprocessing_agent import PreprocessingAgent
from agents.risk_scoring_agent import RiskScoringAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.entity_level_summary_agent import EntityLevelSummaryAgent


class MultiDomainReviewAnalysisPipeline:
    """
    Phase 5 pipeline:
    standardised multi-domain records -> preprocessing -> sentiment -> discrepancy
    -> issue mining -> RAG evidence -> risk scoring -> explainability -> entity summary.

    It accepts the common schema created in Phase 3:
    domain, entity_id, entity_name, review_text, rating

    It also creates legacy aliases (content, score, appId, reviewId) so the existing
    upgraded agents can run without breaking old code.
    """

    STANDARD_COLUMNS = ["domain", "entity_id", "entity_name", "review_text", "rating"]

    def __init__(
        self,
        model_path="outputs/models/distilbert_sentiment",
        use_transformer=True,
        use_discrepancy_model=True,
        use_semantic_issue_model=True,
        use_rag=True,
        output_dir="outputs/final_pipeline",
    ):
        self.model_path = Path(model_path)
        self.use_transformer = use_transformer
        self.use_discrepancy_model = use_discrepancy_model
        self.use_semantic_issue_model = use_semantic_issue_model
        self.use_rag = use_rag
        self.output_dir = Path(output_dir)

        self.preprocessing_agent = PreprocessingAgent(text_column="review_text", rating_column="rating")
        self.risk_agent = RiskScoringAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.entity_summary_agent = EntityLevelSummaryAgent()

        self.execution_trace = []
        self.evidence_by_issue = {}

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------
    def add_trace(self, step, message, output=None):
        item = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": step,
            "message": message,
        }
        if output is not None:
            item["output"] = output
        self.execution_trace.append(item)

    # ------------------------------------------------------------------
    # Input validation and compatibility aliases
    # ------------------------------------------------------------------
    def validate_standard_schema(self, df: pd.DataFrame):
        missing = [col for col in self.STANDARD_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Multi-domain pipeline requires standard schema columns {self.STANDARD_COLUMNS}. "
                f"Missing columns: {missing}. Run DataStandardisationAgent first."
            )

    def add_legacy_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Existing old agents expect these columns.
        df["content"] = df["review_text"]
        df["score"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5)
        df["appId"] = df["entity_id"].astype(str)

        if "review_id" in df.columns:
            df["reviewId"] = df["review_id"].astype(str)
        else:
            df["reviewId"] = [f"review_{i}" for i in range(len(df))]

        if "source" not in df.columns:
            df["source"] = "unknown"

        return df

    # ------------------------------------------------------------------
    # Fallback helpers so Phase 5 can run before model training.
    # Later phases replace fallbacks with the real trained / downloaded models.
    # ------------------------------------------------------------------
    def fallback_sentiment(self, df: pd.DataFrame, reason: str) -> pd.DataFrame:
        df = df.copy()
        df["predicted_sentiment"] = df["sentiment_label"]
        df["sentiment_confidence"] = 0.60
        df["sentiment_model_used"] = "rating_label_fallback"
        self.add_trace(
            "Transformer Sentiment Agent Fallback",
            "DistilBERT model was not used in this run; rating-derived labels were used for smoke testing.",
            {"reason": reason, "rows": len(df)},
        )
        return df

    def fallback_discrepancy(self, df: pd.DataFrame, reason: str) -> pd.DataFrame:
        df = df.copy()

        sentiment_to_star = {"negative": 1, "neutral": 3, "positive": 5}
        df["predicted_star_rating"] = df["predicted_sentiment"].map(sentiment_to_star).fillna(3).astype(int)

        def rating_to_sentiment(score):
            try:
                score = float(score)
            except Exception:
                return "unknown"
            if score <= 2:
                return "negative"
            if score < 4:
                return "neutral"
            return "positive"

        df["rating_sentiment"] = df["score"].apply(rating_to_sentiment)
        df["text_predicted_sentiment"] = df["predicted_star_rating"].apply(rating_to_sentiment)
        df["rating_gap"] = (df["score"].astype(float) - df["predicted_star_rating"].astype(float)).abs()
        df["discrepancy_status"] = df.apply(
            lambda row: "matched" if row["rating_sentiment"] == row["text_predicted_sentiment"] else "mismatched",
            axis=1,
        )

        def discrepancy_type(row):
            actual = float(row["score"])
            predicted = int(row["predicted_star_rating"])
            gap = abs(actual - predicted)
            if actual >= 4 and predicted <= 2:
                return "high_rating_negative_text"
            if actual <= 2 and predicted >= 4:
                return "low_rating_positive_text"
            if gap == 0:
                return "no_discrepancy"
            if gap <= 1:
                return "minor_discrepancy"
            return "moderate_discrepancy"

        penalty_map = {
            "high_rating_negative_text": 10,
            "low_rating_positive_text": 10,
            "moderate_discrepancy": 7,
            "minor_discrepancy": 3,
            "no_discrepancy": 0,
        }
        df["discrepancy_type"] = df.apply(discrepancy_type, axis=1)
        df["discrepancy_penalty"] = df["discrepancy_type"].map(penalty_map).fillna(0).astype(int)
        df["discrepancy_model_used"] = "sentiment_rating_fallback"

        self.add_trace(
            "Rating Prediction and Discrepancy Agent Fallback",
            "BERT star-rating model was not used; discrepancy was estimated from sentiment/rating mapping.",
            {"reason": reason, "rows": len(df)},
        )
        return df

    def fallback_issue_mining(self, df: pd.DataFrame, reason: str) -> pd.DataFrame:
        """Taxonomy-synchronised keyword fallback for smoke testing only.

        This fallback is intentionally secondary to MiniLM.  It reads the same
        domain taxonomy as the semantic agent so new issue categories cannot
        silently disappear when a model/runtime error triggers fallback mode.
        """
        from agents.domain_issue_taxonomy import (
            get_domain_taxonomy,
            get_issue_severity_score,
            severity_level_from_score,
        )

        df = df.copy()

        def phrase_present(text: str, phrase: str) -> bool:
            text = str(text).lower()
            phrase = str(phrase).lower().strip()
            if not phrase:
                return False
            tokens = [re.escape(token) for token in re.split(r"\s+", phrase) if token]
            if not tokens:
                return False
            body = r"\s+".join(tokens)
            return re.search(rf"(?<!\w){body}(?!\w)", text, flags=re.IGNORECASE) is not None

        def detect(row):
            text = str(row.get("clean_review", ""))
            domain = str(row.get("domain", "mobile_app"))
            taxonomy = get_domain_taxonomy(domain)
            hits = []
            for issue, info in taxonomy.items():
                keywords = info.get("keywords", [])
                matched = [keyword for keyword in keywords if phrase_present(text, keyword)]
                if matched:
                    hits.append((issue, get_issue_severity_score(domain, issue), matched))

            if not hits:
                return ["no_issue"], ""

            hits.sort(key=lambda item: item[1], reverse=True)
            issues = [item[0] for item in hits]
            cue_text = "; ".join(hits[0][2][:5])
            return issues, cue_text

        detected = df.apply(detect, axis=1)
        df["detected_issues"] = detected.apply(lambda value: value[0])
        df["matched_taxonomy_cues"] = detected.apply(lambda value: value[1])
        df["primary_issue"] = df["detected_issues"].apply(lambda issues: issues[0] if issues else "no_issue")
        df["issue_severity_score"] = df.apply(
            lambda row: get_issue_severity_score(str(row.get("domain", "mobile_app")), row["primary_issue"]),
            axis=1,
        ).astype(int)
        df["issue_severity_level"] = df["issue_severity_score"].apply(severity_level_from_score)
        df["primary_issue_similarity"] = 0.0
        df["whole_review_issue_similarity"] = 0.0
        df["best_segment_similarity"] = 0.0
        df["issue_semantic_margin"] = 0.0
        df["issue_threshold_used"] = 0.0
        df["issue_acceptance_threshold_used"] = 0.0
        df["issue_detection_reason"] = "taxonomy_keyword_fallback"
        df["issue_evidence_segment"] = df["clean_review"].astype(str).str.slice(0, 180)
        df["issue_segment_count"] = 1
        df["evidence_phrase"] = df["clean_review"].astype(str).str.slice(0, 160)
        df["detected_issues"] = df["detected_issues"].apply(lambda issues: "; ".join(issues))
        df["issue_model_used"] = "domain_taxonomy_keyword_fallback"

        self.add_trace(
            "Semantic Issue Mining Agent Fallback",
            "MiniLM semantic issue model was not used; taxonomy-synchronised keyword fallback was used for smoke testing.",
            {"reason": reason, "rows": len(df)},
        )
        return df

    def attach_simple_evidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Non-RAG local evidence used only when RAG is deliberately disabled."""
        df = df.copy()
        df["rag_evidence_text"] = ""
        df["rag_similarity_score"] = 0.0
        df["rag_evidence_available"] = False
        df["rag_backend"] = "disabled"
        return df

    def attach_unavailable_rag(self, df: pd.DataFrame, reason: str = "") -> pd.DataFrame:
        """Never present the query review/evidence_phrase as retrieved RAG evidence."""
        df = df.copy()
        df["rag_evidence_text"] = ""
        df["rag_similarity_score"] = 0.0
        df["rag_top_evidence_text"] = ""
        df["rag_top_evidence_similarity"] = 0.0
        df["rag_top_evidence_review_id"] = ""
        df["rag_top_evidence_entity_name"] = ""
        df["rag_top_evidence_source"] = ""
        df["rag_top_k_json"] = "[]"
        df["rag_evidence_count"] = 0
        df["rag_evidence_available"] = False
        df["rag_backend"] = "unavailable"
        df["rag_failure_reason"] = str(reason)
        return df

    # ------------------------------------------------------------------
    # Agent runners
    # ------------------------------------------------------------------
    def run_sentiment_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_transformer:
            return self.fallback_sentiment(df, reason="use_transformer=False")

        try:
            from agents.sentiment_agent import SentimentAnalysisAgent
            agent = SentimentAnalysisAgent(model_path=str(self.model_path))
            sentiment_input_column = (
                "clean_sentiment_review"
                if "clean_sentiment_review" in df.columns
                else "clean_review"
            )
            labels, confidence = agent.predict(df[sentiment_input_column].tolist())
            df = df.copy()
            df["predicted_sentiment"] = labels
            df["sentiment_confidence"] = confidence
            df["sentiment_model_used"] = "distilbert"
            df["sentiment_input_column"] = sentiment_input_column
            self.add_trace(
                "Transformer Sentiment Agent Completed",
                "Predicted sentiment using DistilBERT.",
                {"rows": len(df), "text_column": sentiment_input_column},
            )
            return df
        except Exception as e:
            return self.fallback_sentiment(df, reason=str(e))

    def run_discrepancy_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_discrepancy_model:
            return self.fallback_discrepancy(df, reason="use_discrepancy_model=False")

        try:
            from agents.discrepancy_agent import RatingReviewDiscrepancyAgent
            agent = RatingReviewDiscrepancyAgent()
            out = agent.process(df, text_column="clean_review")
            out["discrepancy_model_used"] = "nlptown_bert"
            self.add_trace("Rating Prediction and Discrepancy Agent Completed", "Predicted text rating and detected rating-review mismatch.", {"rows": len(out)})
            return out
        except Exception as e:
            return self.fallback_discrepancy(df, reason=str(e))

    def run_issue_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_semantic_issue_model:
            return self.fallback_issue_mining(df, reason="use_semantic_issue_model=False")

        try:
            from agents.issue_mining_agent import IssueMiningAgent
            agent = IssueMiningAgent()
            out = agent.process(df, text_column="clean_review")
            out["issue_model_used"] = "minilm_semantic"
            if "evidence_phrase" not in out.columns:
                out["evidence_phrase"] = out["clean_review"].astype(str).str.slice(0, 160)
            self.add_trace("Semantic Issue Mining Agent Completed", "Detected issues using MiniLM semantic matching.", {"rows": len(out)})
            return out
        except Exception as e:
            return self.fallback_issue_mining(df, reason=str(e))

    def run_rag_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_rag:
            self.add_trace(
                "RAG Evidence Retrieval Skipped",
                "RAG was disabled by the user; no semantic retrieval was executed.",
                {"rows": len(df), "enabled": False},
            )
            return self.attach_simple_evidence(df)

        try:
            from agents.rag_evidence_agent import RAGEvidenceRetrievalAgent

            rag = RAGEvidenceRetrievalAgent(
                corpus_path="data/processed/combined_multidomain_reviews.csv",
                index_dir="outputs/rag_corpus",
                auto_build_if_missing=True,
            )
            out = rag.process(df, text_column="clean_review")

            issue_categories = [
                x
                for x in out["primary_issue"].dropna().astype(str).unique().tolist()
                if x not in {"", "no_issue"}
            ]
            self.evidence_by_issue = rag.retrieve_evidence_for_issues(
                issue_categories,
                top_k=3,
            )

            backend = (
                str(out["rag_backend"].iloc[0])
                if "rag_backend" in out.columns and len(out)
                else rag.backend
            )
            available = (
                int(out["rag_evidence_available"].astype(bool).sum())
                if "rag_evidence_available" in out.columns
                else int(out.get("rag_evidence_text", pd.Series(dtype=str)).astype(str).str.len().gt(0).sum())
            )

            backend_message = (
                "Retrieved independent MiniLM evidence from the persistent combined-dataset FAISS corpus."
                if backend == "faiss_ip"
                else "Retrieved independent MiniLM evidence from the persistent combined dataset using the NumPy cosine fallback."
            )
            self.add_trace(
                "RAG Evidence Retrieval Agent Completed",
                backend_message,
                {
                    "issues": issue_categories,
                    "rows": len(out),
                    "evidence_rows": available,
                    "backend": backend,
                    "enabled": True,
                    "corpus_source": (
                        str(out["rag_corpus_source"].iloc[0])
                        if "rag_corpus_source" in out.columns and len(out)
                        else "data/processed/combined_multidomain_reviews.csv"
                    ),
                    "corpus_rows": (
                        int(out["rag_corpus_indexed_rows"].iloc[0])
                        if "rag_corpus_indexed_rows" in out.columns and len(out)
                        else 0
                    ),
                    "self_duplicate_exclusion": True,
                },
            )
            return out
        except Exception as e:
            self.add_trace(
                "RAG Evidence Retrieval Fallback",
                "Semantic RAG could not run; simple local evidence phrases were attached instead.",
                {"reason": f"{type(e).__name__}: {e}", "rows": len(df)},
            )
            return self.attach_unavailable_rag(
                df,
                reason=f"{type(e).__name__}: {e}",
            )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, save_outputs=True) -> dict:
        self.execution_trace = []
        self.evidence_by_issue = {}

        self.validate_standard_schema(df)
        self.add_trace("Input Validation Completed", "Standard multi-domain schema detected.", {"rows": len(df), "columns": df.columns.tolist()})

        df = self.add_legacy_aliases(df)
        self.add_trace("Legacy Compatibility Layer Completed", "Added content, score, appId and reviewId aliases for existing agents.", {"rows": len(df)})

        cleaned_df = self.preprocessing_agent.process(df)
        self.add_trace("Preprocessing Agent Completed", "Cleaned review text and created rating-derived sentiment labels.", {"rows": len(cleaned_df)})

        sentiment_df = self.run_sentiment_agent(cleaned_df)
        discrepancy_df = self.run_discrepancy_agent(sentiment_df)
        issue_df = self.run_issue_agent(discrepancy_df)
        evidence_df = self.run_rag_agent(issue_df)

        risk_df = self.risk_agent.process(evidence_df)
        self.add_trace("Risk Scoring Agent Completed", "Calculated trust score, risk level and reliability level.", {"rows": len(risk_df)})

        explainability_df = self.explainability_agent.process(risk_df)
        # Add RAG evidence to explanation text for clearer output.
        if "rag_evidence_text" in explainability_df.columns:
            explainability_df["evidence_based_explanation"] = explainability_df.apply(
                lambda row: f"{row['explanation_text']} Supporting evidence: {str(row.get('rag_evidence_text', ''))[:220]}",
                axis=1,
            )
        else:
            explainability_df["evidence_based_explanation"] = explainability_df["explanation_text"]
        self.add_trace("Explainability Agent Completed", "Generated score explanations and recommendations.", {"rows": len(explainability_df)})

        explainability_examples_df = self.explainability_agent.create_explainability_examples(explainability_df, sample_size=20)
        entity_summary_df = self.entity_summary_agent.process(explainability_df)
        self.add_trace("Entity Summary Agent Completed", "Aggregated review-level outputs at app/product/hotel/restaurant level.", {"entities": len(entity_summary_df)})

        result = {
            "review_level_results": explainability_df,
            "entity_level_summary": entity_summary_df,
            "explainability_examples": explainability_examples_df,
            "issue_evidence": self.evidence_by_issue,
            "execution_trace": self.execution_trace,
        }

        if save_outputs:
            self.save_outputs(result)

        return result

    def save_outputs(self, result: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        result["review_level_results"].to_csv(
            self.output_dir / "multidomain_review_level_results.csv",
            index=False,
            encoding="utf-8",
        )
        result["entity_level_summary"].to_csv(
            self.output_dir / "multidomain_entity_level_summary.csv",
            index=False,
            encoding="utf-8",
        )
        result["explainability_examples"].to_csv(
            self.output_dir / "multidomain_explainability_examples.csv",
            index=False,
            encoding="utf-8",
        )

        evidence_rows = []
        for issue, records in result.get("issue_evidence", {}).items():
            for rec in records:
                evidence_rows.append({"issue": issue, **rec})
        pd.DataFrame(evidence_rows).to_csv(
            self.output_dir / "multidomain_rag_evidence.csv",
            index=False,
            encoding="utf-8",
        )

        self.add_trace("Outputs Saved", f"Saved pipeline outputs to {self.output_dir}")
        # Each web job keeps its own trace. A global outputs/reports file caused
        # concurrent runs to overwrite one another.
        with open(
            self.output_dir / "multidomain_pipeline_trace.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.execution_trace, file, indent=4, ensure_ascii=False)


def analyze_multidomain_reviews(df: pd.DataFrame, save_outputs=True, **kwargs):
    pipeline = MultiDomainReviewAnalysisPipeline(**kwargs)
    return pipeline.analyze(df, save_outputs=save_outputs)

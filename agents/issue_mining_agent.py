"""
Offline-local Domain-Aware Semantic Issue Mining Agent.

Model:
outputs/models/all-MiniLM-L6-v2

No Hugging Face repo id is used at runtime.

Context-aware repair (23 Aug 2026):
- keeps MiniLM as the primary issue classifier
- compares both whole-review and sentence/clause embeddings with domain taxonomy embeddings
- adds taxonomy-specific lexical support only as a secondary acceptance signal
- protects obvious "no / without / does not contain" statements from false issue detection
- exposes audit fields explaining why an issue was accepted or rejected
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd

from services.local_model_registry import (
    enforce_offline_mode,
    get_minilm_path,
    require_local_model,
)

from agents.domain_issue_taxonomy import (
    get_domain_taxonomy,
    get_issue_severity_score,
    severity_level_from_score,
    GLOBAL_PROBLEM_CUES,
)


class IssueMiningAgent:
    """MiniLM semantic issue detector with contextual decision support.

    MiniLM remains the primary classifier.  Taxonomy keywords are not used as a
    standalone classifier; they only support a MiniLM candidate when the semantic
    score is slightly below the normal domain threshold and the candidate is
    otherwise well separated from alternatives.
    """

    DOMAIN_BASE_THRESHOLDS = {
        "mobile_app": 0.30,
        "hotel": 0.31,
        "ecommerce": 0.32,
        "restaurant": 0.31,
    }

    CONTRAST_SPLIT_RE = re.compile(
        r"\b(?:but|however|although|though|yet|except|while|whereas)\b|[,;:]",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        similarity_threshold: float = 0.31,
        batch_size: int = 32,
        max_evidence_chars: int = 180,
        positive_no_problem_block: bool = True,
        enable_segment_matching: bool = True,
        max_segments_per_review: int = 8,
        cue_rescue_margin: float = 0.08,
        cue_rescue_floor: float = 0.18,
        min_semantic_margin: float = 0.04,
        *args,
        **kwargs,
    ):
        enforce_offline_mode()

        # model_name is kept only for backward compatibility; local path is always used.
        self.model_path = require_local_model(
            model_path or get_minilm_path(),
            "MiniLM semantic issue model",
        )

        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_evidence_chars = max_evidence_chars
        self.positive_no_problem_block = positive_no_problem_block
        self.enable_segment_matching = enable_segment_matching
        self.max_segments_per_review = max(1, int(max_segments_per_review))
        self.cue_rescue_margin = float(cue_rescue_margin)
        self.cue_rescue_floor = float(cue_rescue_floor)
        self.min_semantic_margin = float(min_semantic_margin)

        self.model = None
        self.issue_embeddings = {}

    def load_model(self):
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(str(self.model_path), local_files_only=True)
        except TypeError:
            self.model = SentenceTransformer(str(self.model_path))

    @staticmethod
    def clean_for_matching(text) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def phrase_present(text: str, phrase: str) -> bool:
        """Boundary-aware phrase match.

        This avoids substring errors such as keyword ``ad`` matching the word
        ``bad``.  Spaces inside multi-word phrases may contain arbitrary
        whitespace in the review.
        """
        text = str(text).lower()
        phrase = str(phrase).lower().strip()
        if not phrase:
            return False

        tokens = [re.escape(token) for token in re.split(r"\s+", phrase) if token]
        if not tokens:
            return False
        body = r"\s+".join(tokens)
        pattern = rf"(?<!\w){body}(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def build_issue_text(self, domain: str, issue: str, info: dict) -> str:
        label = info.get("label", issue)
        descriptions = " ".join(info.get("descriptions", []))
        keywords = " ".join(info.get("keywords", []))
        return f"{domain} issue: {label}. {descriptions}. Related terms: {keywords}"

    def build_issue_embeddings_for_domain(self, domain: str):
        self.load_model()

        if domain in self.issue_embeddings:
            return self.issue_embeddings[domain]

        taxonomy = get_domain_taxonomy(domain)
        issue_names = []
        issue_texts = []

        for issue, info in taxonomy.items():
            issue_names.append(issue)
            issue_texts.append(self.build_issue_text(domain, issue, info))

        embeddings = self.model.encode(
            issue_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.issue_embeddings[domain] = {
            "issue_names": issue_names,
            "issue_texts": issue_texts,
            "embeddings": embeddings,
        }

        return self.issue_embeddings[domain]

    @classmethod
    def has_problem_cue(cls, text: str) -> bool:
        return any(cls.phrase_present(text, cue) for cue in GLOBAL_PROBLEM_CUES)

    @staticmethod
    def row_rating(row) -> float:
        rating = row.get("score", row.get("rating", 3))
        try:
            return float(rating)
        except Exception:
            return 3.0

    @staticmethod
    def row_sentiment(row) -> str:
        return str(row.get("predicted_sentiment", row.get("sentiment_label", ""))).lower().strip()

    def dynamic_threshold(self, domain: str, row, top_similarity: float) -> float:
        base = self.DOMAIN_BASE_THRESHOLDS.get(domain, self.similarity_threshold)
        rating = self.row_rating(row)
        sentiment = self.row_sentiment(row)
        text = str(row.get("_semantic_text", ""))
        has_problem = self.has_problem_cue(text)

        if sentiment == "negative" or rating <= 2:
            return max(0.26, base - 0.03)

        if sentiment == "neutral" or rating == 3:
            return base

        if has_problem:
            return base + 0.03

        return base + 0.10

    def build_semantic_segments(self, text: str) -> List[str]:
        """Return whole review + informative sentence/contrast clauses.

        Whole-review matching is preserved for benchmark continuity.  Shorter
        semantic segments prevent a real complaint from being diluted by a
        positive clause elsewhere in the same review.
        """
        text = self.clean_for_matching(text)
        if not text:
            return [""]

        segments: List[str] = []

        def add(value: str):
            value = self.clean_for_matching(value).strip(" -–—,;:")
            if not value:
                return
            # Ignore one-word fragments unless they are the complete review.
            if value != text and len(value.split()) < 2:
                return
            key = value.lower()
            if key not in {item.lower() for item in segments}:
                segments.append(value)

        add(text)

        if not self.enable_segment_matching:
            return segments

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            sentence = self.clean_for_matching(sentence)
            if not sentence:
                continue
            add(sentence)
            for clause in self.CONTRAST_SPLIT_RE.split(sentence):
                add(clause)

        return segments[: self.max_segments_per_review]

    def matched_issue_keywords(self, text: str, domain: str, issue: str) -> List[str]:
        taxonomy = get_domain_taxonomy(domain)
        keywords = taxonomy.get(issue, {}).get("keywords", [])
        matches = [keyword for keyword in keywords if self.phrase_present(text, keyword)]
        # Longest first makes the audit field easier to read and favours specific phrases.
        return sorted(set(matches), key=lambda value: (-len(value), value))

    @staticmethod
    def _cue_regex(cue: str) -> str:
        tokens = [re.escape(token) for token in re.split(r"\s+", cue.lower().strip()) if token]
        return r"\s+".join(tokens)

    def has_complaint_context(self, text: str, cues: List[str]) -> bool:
        """Detect language that treats the matched cue as an unwanted problem."""
        lower = str(text).lower()
        for cue in cues:
            c = self._cue_regex(cue)
            patterns = [
                rf"\b(?:do\s+not|don't|dont)\s+want\b.{{0,80}}{c}",
                rf"\b(?:want|prefer)\b.{{0,80}}\bbut\s+not\b.{{0,80}}{c}",
                rf"\bbut\s+not\b.{{0,80}}{c}",
                rf"\b(?:avoid|stop|remove|block|hide)\b.{{0,80}}{c}",
                rf"\b(?:unwanted|offensive|disturbing|shameful|shamefull)\b.{{0,50}}{c}",
                rf"\b(?:problem|issue|complaint)\b.{{0,60}}{c}",
                rf"\b(?:shows?|showing|displaying|exposes?|exposing)\b.{{0,60}}{c}",
            ]
            if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns):
                return True
        return False

    def has_protective_absence_context(self, text: str, cues: List[str]) -> bool:
        """Detect obvious statements that the issue is absent, not present.

        Complaint context takes precedence.  For example, "I want science but not
        shameful pictures" is a concern, while "this app has no sexual pictures"
        is a protective/absence statement.
        """
        if not cues:
            return False

        lower = str(text).lower()
        if self.has_complaint_context(lower, cues):
            return False

        for cue in cues:
            c = self._cue_regex(cue)
            patterns = [
                rf"\b(?:no|without)\b.{{0,35}}{c}",
                rf"\bfree\s+from\b.{{0,35}}{c}",
                rf"\b(?:does\s+not|doesn't|doesnt|never)\b.{{0,25}}\b(?:show|contain|include|have|display|feature)\w*\b.{{0,45}}{c}",
                rf"\bnot\b(?:\s+\w+){{0,2}}\s+{c}",
            ]
            if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns):
                return True
        return False

    def decide_candidate(
        self,
        row,
        domain: str,
        top_issue: str,
        top_similarity: float,
        second_similarity: float,
        matched_cues: List[str],
    ) -> Tuple[bool, float, str]:
        """Return (accepted, decision_threshold, reason)."""
        text = str(row.get("_semantic_text", ""))
        rating = self.row_rating(row)
        sentiment = self.row_sentiment(row)
        has_problem = self.has_problem_cue(text)
        standard_threshold = self.dynamic_threshold(domain, row, top_similarity)

        if self.positive_no_problem_block:
            if (rating >= 4 or sentiment == "positive") and not has_problem and not matched_cues:
                return False, standard_threshold, "positive_no_problem_block"

        complaint_context = self.has_complaint_context(text, matched_cues)
        protective_absence = self.has_protective_absence_context(text, matched_cues)

        # Prevent a high-similarity phrase such as "no sexual content" or
        # "not dirty" from becoming an issue in an otherwise positive review.
        if protective_absence and not complaint_context and (rating >= 4 or sentiment == "positive"):
            return False, standard_threshold, "protective_absence_context"

        if top_similarity >= standard_threshold:
            return True, standard_threshold, "semantic_threshold"

        # Secondary rescue: MiniLM still chooses the issue, but the score is just
        # under the normal threshold.  A category-specific cue can rescue it only
        # when there is semantic separation and contextual support.
        rescue_threshold = max(
            self.cue_rescue_floor,
            standard_threshold - self.cue_rescue_margin,
        )
        semantic_margin = float(top_similarity - second_similarity)
        context_support = (
            rating <= 2
            or sentiment in {"negative", "neutral"}
            or complaint_context
        )

        if (
            matched_cues
            and not protective_absence
            and context_support
            and top_similarity >= rescue_threshold
            and semantic_margin >= self.min_semantic_margin
        ):
            return True, rescue_threshold, "semantic_plus_taxonomy_cue"

        return False, standard_threshold, "below_threshold"

    def should_mark_no_issue(self, row, domain: str, top_similarity: float) -> bool:
        """Backward-compatible helper used by existing diagnostics.

        It cannot see top-2/candidate-specific support, so it reflects the standard
        gate only.  Full processing uses :meth:`decide_candidate`.
        """
        text = str(row.get("_semantic_text", ""))
        rating = self.row_rating(row)
        sentiment = self.row_sentiment(row)
        has_problem = self.has_problem_cue(text)

        if self.positive_no_problem_block:
            if (rating >= 4 or sentiment == "positive") and not has_problem:
                return True

        threshold = self.dynamic_threshold(domain, row, top_similarity)
        if top_similarity >= threshold:
            return False

        # Backward-compatible approximation for older diagnostics: when a real
        # problem cue exists and the review/rating context supports a complaint,
        # reflect the same narrow rescue band used by full candidate processing.
        rescue_threshold = max(self.cue_rescue_floor, threshold - self.cue_rescue_margin)
        context_support = (
            rating <= 2
            or sentiment in {"negative", "neutral"}
        )
        if has_problem and context_support and top_similarity >= rescue_threshold:
            return False

        return True

    @staticmethod
    def get_top_matches(issue_names: List[str], similarities: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
        pairs = list(zip(issue_names, similarities.tolist()))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        return [(issue, score) for issue, score in pairs if score >= threshold][:3]

    def extract_evidence_phrase(self, text: str, domain: str, issue: str) -> str:
        text = self.clean_for_matching(text)
        if not text:
            return ""

        taxonomy = get_domain_taxonomy(domain)
        keywords = taxonomy.get(issue, {}).get("keywords", [])
        lower_text = text.lower()

        for keyword in sorted(keywords, key=len, reverse=True):
            if not self.phrase_present(text, keyword):
                continue
            keyword_lower = keyword.lower()
            match = re.search(
                rf"(?<!\w){self._cue_regex(keyword_lower)}(?!\w)",
                lower_text,
                flags=re.IGNORECASE,
            )
            if match:
                idx = match.start()
                start = max(0, idx - 70)
                end = min(len(text), match.end() + 90)
                return text[start:end].strip()[: self.max_evidence_chars]

        chunks = re.split(r"(?<=[.!?])\s+", text)
        if chunks:
            return chunks[0][: self.max_evidence_chars]

        return text[: self.max_evidence_chars]

    def process(self, df: pd.DataFrame, text_column: str = "clean_review") -> pd.DataFrame:
        if text_column not in df.columns:
            if "review_text" in df.columns:
                text_column = "review_text"
            elif "content" in df.columns:
                text_column = "content"
            else:
                raise ValueError(f"Text column not found: {text_column}")

        if "domain" not in df.columns:
            raise ValueError("Domain column is required for semantic issue mining.")

        self.load_model()

        out = df.copy()
        out["_semantic_text"] = out[text_column].apply(self.clean_for_matching)

        parts = []

        for domain, domain_df in out.groupby("domain", dropna=False):
            domain = str(domain) if pd.notna(domain) else "mobile_app"
            domain_df = domain_df.copy()

            domain_data = self.build_issue_embeddings_for_domain(domain)
            issue_names = domain_data["issue_names"]
            issue_matrix = domain_data["embeddings"]

            # Build all semantic segments first and encode them in one batch for
            # efficiency.  Each row always starts with the whole review.
            row_segments: List[List[str]] = []
            flat_segments: List[str] = []
            offsets: List[Tuple[int, int]] = []

            for text in domain_df["_semantic_text"].tolist():
                segments = self.build_semantic_segments(text)
                start = len(flat_segments)
                flat_segments.extend(segments)
                end = len(flat_segments)
                row_segments.append(segments)
                offsets.append((start, end))

            flat_embeddings = self.model.encode(
                flat_segments,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            primary_issues = []
            detected_issues_col = []
            severity_scores = []
            severity_levels = []
            primary_similarities = []
            whole_similarities = []
            best_segment_similarities = []
            semantic_margins = []
            evidence_phrases = []
            evidence_segments = []
            issue_labels = []
            issue_thresholds = []
            decision_thresholds = []
            matched_cues_col = []
            decision_reasons = []
            segment_counts = []

            for row_number, (_, row) in enumerate(domain_df.iterrows()):
                start, end = offsets[row_number]
                segment_embeddings = flat_embeddings[start:end]
                segments = row_segments[row_number]

                # shape: [segments, issues]
                segment_similarity_matrix = np.matmul(segment_embeddings, issue_matrix.T)
                whole_issue_similarities = segment_similarity_matrix[0]
                effective_issue_similarities = np.max(segment_similarity_matrix, axis=0)
                best_segment_indexes = np.argmax(segment_similarity_matrix, axis=0)

                order = np.argsort(effective_issue_similarities)[::-1]
                top_idx = int(order[0])
                second_idx = int(order[1]) if len(order) > 1 else top_idx

                top_issue = issue_names[top_idx]
                top_score = float(effective_issue_similarities[top_idx])
                second_score = (
                    float(effective_issue_similarities[second_idx])
                    if second_idx != top_idx
                    else 0.0
                )
                whole_score = float(whole_issue_similarities[top_idx])
                best_segment_idx = int(best_segment_indexes[top_idx])
                best_segment = segments[best_segment_idx]

                standard_threshold = self.dynamic_threshold(domain, row, top_score)
                matched_cues = self.matched_issue_keywords(
                    row.get("_semantic_text", ""),
                    domain,
                    top_issue,
                )

                accepted, decision_threshold, decision_reason = self.decide_candidate(
                    row=row,
                    domain=domain,
                    top_issue=top_issue,
                    top_similarity=top_score,
                    second_similarity=second_score,
                    matched_cues=matched_cues,
                )

                if not accepted:
                    primary_issue = "no_issue"
                    detected_issues = []
                    severity_score = 0
                    severity_level = "none"
                    evidence_phrase = ""
                    issue_label = "No clear issue detected"
                else:
                    primary_issue = top_issue
                    severity_score = get_issue_severity_score(domain, primary_issue)
                    severity_level = severity_level_from_score(severity_score)
                    issue_label = get_domain_taxonomy(domain)[primary_issue].get("label", primary_issue)
                    evidence_phrase = self.extract_evidence_phrase(
                        text=row.get(text_column, ""),
                        domain=domain,
                        issue=primary_issue,
                    )

                    # Only the top candidate may use the cue-supported rescue
                    # threshold. Additional issues still need the normal semantic
                    # threshold, preventing broad low-threshold over-detection.
                    detected_pairs: List[Tuple[str, float]] = [(primary_issue, top_score)]
                    for idx in order[1:]:
                        issue = issue_names[int(idx)]
                        score = float(effective_issue_similarities[int(idx)])
                        if score >= standard_threshold:
                            detected_pairs.append((issue, score))
                        if len(detected_pairs) >= 3:
                            break
                    detected_issues = [f"{issue}:{score:.3f}" for issue, score in detected_pairs]

                primary_issues.append(primary_issue)
                detected_issues_col.append("; ".join(detected_issues) if detected_issues else "no_issue")
                severity_scores.append(severity_score)
                severity_levels.append(severity_level)
                primary_similarities.append(round(top_score, 4))
                whole_similarities.append(round(whole_score, 4))
                best_segment_similarities.append(round(top_score, 4))
                semantic_margins.append(round(top_score - second_score, 4))
                evidence_phrases.append(evidence_phrase)
                evidence_segments.append(best_segment[: self.max_evidence_chars])
                issue_labels.append(issue_label)
                issue_thresholds.append(round(standard_threshold, 4))
                decision_thresholds.append(round(decision_threshold, 4))
                matched_cues_col.append("; ".join(matched_cues))
                decision_reasons.append(decision_reason)
                segment_counts.append(len(segments))

            domain_df["primary_issue"] = primary_issues
            domain_df["detected_issues"] = detected_issues_col
            domain_df["issue_severity_score"] = severity_scores
            domain_df["issue_severity_level"] = severity_levels
            domain_df["primary_issue_similarity"] = primary_similarities
            domain_df["whole_review_issue_similarity"] = whole_similarities
            domain_df["best_segment_similarity"] = best_segment_similarities
            domain_df["issue_semantic_margin"] = semantic_margins
            domain_df["issue_threshold_used"] = issue_thresholds
            domain_df["issue_acceptance_threshold_used"] = decision_thresholds
            domain_df["matched_taxonomy_cues"] = matched_cues_col
            domain_df["issue_detection_reason"] = decision_reasons
            domain_df["issue_evidence_segment"] = evidence_segments
            domain_df["issue_segment_count"] = segment_counts
            domain_df["evidence_phrase"] = evidence_phrases
            domain_df["issue_label"] = issue_labels
            domain_df["issue_model_used"] = "local_minilm_semantic_contextual"

            parts.append(domain_df)

        result = pd.concat(parts, ignore_index=True)
        return result.drop(columns=["_semantic_text"], errors="ignore")


SemanticIssueMiningAgent = IssueMiningAgent

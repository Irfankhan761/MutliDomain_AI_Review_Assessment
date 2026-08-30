"""
Phase 13: Final Orchestrator Integration

This is the central supervisory/orchestrator layer.

It decides:
- if input is a Google Play URL / app id
- if input is an exact Google Maps place URL
- if input is a CSV dataset
- if input is a single review

Then it prepares the data, runs the full multi-domain pipeline, collects outputs,
and optionally calls Groq to generate the final readable report.

Final workflow:
Input
-> Orchestrator reasoning
-> Scraper / Dataset / Single-review workflow
-> MultiDomainReviewAnalysisPipeline
-> DistilBERT Sentiment
-> BERT Rating Prediction & Discrepancy
-> MiniLM Semantic Issue Mining
-> RAG Evidence Retrieval
-> Risk Scoring
-> Explainability
-> Entity Summary
-> Groq Final Summary
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import os
import re
import uuid

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OrchestratorTraceItem:
    step: str
    message: str
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorState:
    run_id: str
    started_at: str
    input_type: str = "unknown"
    domain: str = "unknown"
    selected_workflow: str = "unknown"
    next_agent: str = "unknown"
    app_id: Optional[str] = None
    place_id: Optional[str] = None
    source_url: Optional[str] = None
    input_path: Optional[str] = None
    prepared_dataset_path: Optional[str] = None
    output_dir: Optional[str] = None
    groq_report_path: Optional[str] = None
    status: str = "started"
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


class FinalOrchestrator:
 
    STANDARD_COLUMNS = [
        "review_id",
        "domain",
        "entity_id",
        "entity_name",
        "review_text",
        "rating",
        "rating_original",
        "review_date",
        "source",
        "raw_source_path",
    ]

    # Optional fields produced by the audited canonical dataset builder.
    # They must survive orchestration so DistilBERT can use the clean
    # sentiment-specific text while all other agents still receive full review_text.
    OPTIONAL_STANDARD_COLUMNS = [
        "sentiment_text",
        "sentiment_label",
        "text_key",
        "label_method",
    ]

    DOMAIN_VALUES = {"mobile_app", "hotel", "ecommerce", "restaurant", "multidomain"}

    def __init__(
        self,
        model_path: str | Path = "outputs/models/distilbert_sentiment",
        base_output_dir: str | Path = "outputs/final_orchestrator_runs",
        use_rag: bool = True,
        use_groq: bool = True,
        sample_size: int = 200,
        max_reviews: int = 200,
    ):
        self.model_path = PROJECT_ROOT / model_path
        self.base_output_dir = PROJECT_ROOT / base_output_dir
        self.use_rag = use_rag
        self.use_groq = use_groq
        self.sample_size = sample_size
        self.max_reviews = max_reviews

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.run_output_dir = self.base_output_dir / self.run_id
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        self.state = OrchestratorState(
            run_id=self.run_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
            output_dir=str(self.run_output_dir),
        )

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------
    def add_trace(self, step: str, message: str, output: Optional[Dict[str, Any]] = None):
        item = OrchestratorTraceItem(
            step=step,
            message=message,
            output=output or {},
        )
        self.state.execution_trace.append(asdict(item))

    def save_state(self):
        state_path = self.run_output_dir / "orchestrator_state.json"
        state_path.write_text(
            json.dumps(asdict(self.state), indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        return state_path

    # ------------------------------------------------------------------
    # Input detection
    # ------------------------------------------------------------------
    @staticmethod
    def extract_google_play_app_id(value: str) -> Optional[str]:
        if not value:
            return None

        value = str(value).strip()

        # Direct app id style: com.anydo, com.microsoft.todos, etc.
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)+", value):
            return value

        # URL style:
        # https://play.google.com/store/apps/details?id=com.anydo
        match = re.search(r"[?&]id=([a-zA-Z0-9._]+)", value)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def looks_like_url(value: str) -> bool:
        return bool(value and str(value).lower().startswith(("http://", "https://")))

    @staticmethod
    def looks_like_google_maps_url(value: str) -> bool:
        text = str(value or "").strip().lower()
        return bool(
            text.startswith(("http://", "https://"))
            and (
                "google.com/maps" in text
                or "google.co" in text and "/maps" in text
                or "maps.app.goo.gl" in text
                or "goo.gl/maps" in text
            )
        )

    @staticmethod
    def path_exists(value: str | Path | None) -> bool:
        if not value:
            return False
        return Path(value).exists() or (PROJECT_ROOT / str(value)).exists()

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.exists():
            return path
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path
        return path

    def detect_input_type(
        self,
        input_value: Optional[str] = None,
        input_path: Optional[str | Path] = None,
        input_text: Optional[str] = None,
        url: Optional[str] = None,
        app_id: Optional[str] = None,
        input_type: str = "auto",
    ) -> str:
        if input_type and input_type != "auto":
            if input_type == "url":
                candidate = url or input_value or ""
                return "google_maps_url" if self.looks_like_google_maps_url(candidate) else "google_play_url"
            aliases = {
                "google_url": "google_play_url",
                "maps_url": "google_maps_url",
                "google_maps": "google_maps_url",
            }
            return aliases.get(input_type, input_type)

        if url:
            return "google_maps_url" if self.looks_like_google_maps_url(url) else "google_play_url"

        if app_id:
            return "app_id"

        if input_path:
            return "csv"

        if input_text:
            return "single_review"

        if input_value:
            if self.looks_like_google_maps_url(input_value):
                return "google_maps_url"

            app_id_from_value = self.extract_google_play_app_id(input_value)
            if app_id_from_value and self.looks_like_url(input_value):
                return "google_play_url"

            if app_id_from_value and not self.path_exists(input_value):
                return "app_id"

            if self.path_exists(input_value):
                return "csv"

            if self.looks_like_url(input_value):
                return "google_play_url"

            return "single_review"

        raise ValueError("No valid input provided to orchestrator.")

    def detect_domain_from_columns(self, df: pd.DataFrame) -> str:
        cols = set(df.columns)

        if "domain" in cols:
            unique_domains = set(df["domain"].dropna().astype(str).unique())
            valid_domains = unique_domains.intersection(self.DOMAIN_VALUES)
            if len(valid_domains) == 1:
                return list(valid_domains)[0]
            if len(valid_domains) > 1:
                return "multidomain"

        if {"content", "score", "appId"}.issubset(cols):
            return "mobile_app"

        if {"rating", "text"}.issubset(cols) and ("asin" in cols or "parent_asin" in cols):
            return "ecommerce"

        if {"Hotel_Name", "Average_Score"}.issubset(cols):
            return "hotel"

        if {"Review Text", "Rating"}.issubset(cols):
            return "restaurant"

        return "multidomain"

    def detect_domain(
        self,
        input_type: str,
        df: Optional[pd.DataFrame] = None,
        domain: Optional[str] = None,
    ) -> str:
        if domain and str(domain).strip().lower() != "auto":
            return str(domain).strip().lower()

        if input_type in {"google_play_url", "app_id"}:
            return "mobile_app"

        if input_type == "google_maps_url":
            return domain if domain in {"hotel", "restaurant"} else "restaurant"

        if input_type == "single_review":
            return "mobile_app"

        if df is not None:
            return self.detect_domain_from_columns(df)

        return "multidomain"

    # ------------------------------------------------------------------
    # Standardisation helpers
    # ------------------------------------------------------------------
    def is_standard_schema(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.STANDARD_COLUMNS[:6])

    @staticmethod
    def clean_text(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).replace("\n", " ").replace("\r", " ").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def safe_rating(value, default=3.0) -> float:
        try:
            if pd.isna(value):
                return float(default)
            value = float(value)
            if value > 5:
                # Hotel reviews sometimes use 10-point scale.
                value = value / 2.0
            return max(1.0, min(5.0, value))
        except Exception:
            return float(default)

    def standardize_mobile_app(self, df: pd.DataFrame, raw_source_path: str = "") -> pd.DataFrame:
        out = pd.DataFrame()
        out["review_id"] = df.get("reviewId", pd.Series([f"mobile_{i}" for i in range(len(df))]))
        out["domain"] = "mobile_app"
        out["entity_id"] = df.get("appId", "unknown_app")
        # Live Google Play runs add appTitle, so the user does not need to
        # type Entity Name after already providing the URL/App ID.
        out["entity_name"] = df.get("appTitle", df.get("appId", "unknown_app"))
        out["review_text"] = df.get("content", "").apply(self.clean_text)
        out["rating"] = df.get("score", 3).apply(self.safe_rating)
        out["rating_original"] = df.get("score", 3)
        out["review_date"] = df.get("at", "")
        out["source"] = "google_play_reviews"
        out["raw_source_path"] = raw_source_path
        return out

    def standardize_ecommerce(self, df: pd.DataFrame, raw_source_path: str = "") -> pd.DataFrame:
        out = pd.DataFrame()
        out["review_id"] = [f"ecommerce_{i}" for i in range(len(df))]
        out["domain"] = "ecommerce"
        out["entity_id"] = df.get("parent_asin", df.get("asin", "unknown_product"))
        out["entity_name"] = df.get("parent_asin", df.get("asin", "unknown_product"))

        # IMPORTANT: use the review body only. Amazon titles can explicitly
        # contain labels such as "Five Stars" / "One Star", which would leak
        # the rating-derived target into sentiment/rating model inputs.
        text = df.get("text", pd.Series([""] * len(df), index=df.index))
        out["review_text"] = text.fillna("").astype(str).apply(self.clean_text)

        out["rating"] = df.get("rating", 3).apply(self.safe_rating)
        out["rating_original"] = df.get("rating", 3)
        out["review_date"] = df.get("timestamp", "")
        out["source"] = "amazon_reviews_2023_all_beauty"
        out["raw_source_path"] = raw_source_path
        return out

    def standardize_hotel(self, df: pd.DataFrame, raw_source_path: str = "") -> pd.DataFrame:
        out = pd.DataFrame()
        out["review_id"] = df.get(
            "review_id", pd.Series([f"hotel_{i}" for i in range(len(df))])
        )
        out["domain"] = "hotel"
        out["entity_id"] = df.get("Hotel_Name", "unknown_hotel")
        out["entity_name"] = df.get("Hotel_Name", "unknown_hotel")

        positive = df.get(
            "Positive_Review", pd.Series([""] * len(df), index=df.index)
        ).fillna("").astype(str)
        negative = df.get(
            "Negative_Review", pd.Series([""] * len(df), index=df.index)
        ).fillna("").astype(str)
        combined = (negative + " " + positive).str.strip()

        out["review_text"] = combined.apply(self.clean_text)
        # Reviewer_Score is the individual review score. Average_Score is a
        # hotel-level aggregate and must only be a fallback when reviewer scores
        # are unavailable.
        score_series = df.get("Reviewer_Score")
        if score_series is None:
            score_series = df.get(
                "Average_Score", pd.Series([3.0] * len(df), index=df.index)
            )
        # The 515K Hotels source uses a 0-10 scale for both Reviewer_Score
        # and Average_Score. Normalise every valid value to the common 1-5
        # schema; checking only values above 5 would incorrectly leave low
        # scores such as 2/10 as 2/5.
        def normalise_hotel_score(value) -> float:
            try:
                if pd.isna(value):
                    return 3.0
                return max(1.0, min(5.0, float(value) / 2.0))
            except Exception:
                return 3.0

        out["rating"] = score_series.apply(normalise_hotel_score)
        out["rating_original"] = score_series
        out["review_date"] = df.get("Review_Date", "")
        out["source"] = "hotel_reviews_europe_515k"
        out["raw_source_path"] = raw_source_path
        return out

    def standardize_restaurant(self, df: pd.DataFrame, raw_source_path: str = "") -> pd.DataFrame:
        out = pd.DataFrame()
        out["review_id"] = [f"restaurant_{i}" for i in range(len(df))]
        out["domain"] = "restaurant"
        out["entity_id"] = df.get("Yelp URL", "unknown_restaurant")
        out["entity_name"] = df.get("Yelp URL", "unknown_restaurant")
        out["review_text"] = df.get("Review Text", "").apply(self.clean_text)
        out["rating"] = df.get("Rating", 3).apply(self.safe_rating)
        out["rating_original"] = df.get("Rating", 3)
        out["review_date"] = df.get("Date", "")
        out["source"] = "yelp_restaurant_reviews"
        out["raw_source_path"] = raw_source_path
        return out

    def standardize_known_dataframe(
        self,
        df: pd.DataFrame,
        domain: str,
        raw_source_path: str = "",
    ) -> pd.DataFrame:
        if self.is_standard_schema(df):
            out = df.copy()
            out["review_text"] = out["review_text"].apply(self.clean_text)
            out["rating"] = out["rating"].apply(self.safe_rating)

            optional_defaults = {
                "rating_original": out["rating"],
                "review_date": "",
                "source": "uploaded_standard_csv",
                "raw_source_path": raw_source_path,
            }
            for column, default_value in optional_defaults.items():
                if column not in out.columns:
                    out[column] = default_value

            domain_aliases = {
                "app": "mobile_app",
                "mobile": "mobile_app",
                "mobile app": "mobile_app",
                "e-commerce": "ecommerce",
                "e_commerce": "ecommerce",
                "product": "ecommerce",
            }
            out["domain"] = (
                out["domain"].fillna("").astype(str).str.strip().str.lower()
                .replace(domain_aliases)
            )
            invalid_domains = sorted(
                set(out.loc[~out["domain"].isin(self.DOMAIN_VALUES), "domain"])
            )
            if invalid_domains:
                raise ValueError(
                    "Standard-schema CSV contains unsupported/blank domain values: "
                    + ", ".join(repr(value) for value in invalid_domains[:8])
                )

            # Preserve the canonical sentiment-specific fields when present.
            # review_text remains the full evidence for Rating BERT, MiniLM,
            # discrepancy, RAG and risk scoring.
            if "sentiment_text" in out.columns:
                out["sentiment_text"] = out["sentiment_text"].fillna("").astype(str).apply(self.clean_text)

            out = out[out["review_text"].astype(str).str.strip().ne("")].copy()
            out = out.drop_duplicates(subset=["domain", "entity_id", "review_text"])
            if out.empty:
                raise ValueError("Standard-schema CSV contains no usable review rows.")

            columns = self.STANDARD_COLUMNS + [
                col for col in self.OPTIONAL_STANDARD_COLUMNS if col in out.columns
            ]
            return out[columns].reset_index(drop=True)

        if domain == "mobile_app":
            return self.standardize_mobile_app(df, raw_source_path=raw_source_path)

        if domain == "ecommerce":
            return self.standardize_ecommerce(df, raw_source_path=raw_source_path)

        if domain == "hotel":
            return self.standardize_hotel(df, raw_source_path=raw_source_path)

        if domain == "restaurant":
            return self.standardize_restaurant(df, raw_source_path=raw_source_path)

        inferred = self.detect_domain_from_columns(df)

        if inferred == "mobile_app":
            return self.standardize_mobile_app(df, raw_source_path=raw_source_path)

        if inferred == "ecommerce":
            return self.standardize_ecommerce(df, raw_source_path=raw_source_path)

        if inferred == "hotel":
            return self.standardize_hotel(df, raw_source_path=raw_source_path)

        if inferred == "restaurant":
            return self.standardize_restaurant(df, raw_source_path=raw_source_path)

        raise ValueError(
            "Could not standardise CSV automatically. Provide --domain or use standard schema."
        )

    def sample_dataframe(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Return an exact-size deterministic stratified sample without duplicates."""
        if sample_size <= 0 or len(df) <= sample_size:
            return df.reset_index(drop=True)

        if "domain" not in df.columns:
            return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        working = df.copy()
        working["_sample_row_id"] = range(len(working))
        groups = [
            (str(domain), group)
            for domain, group in working.groupby("domain", dropna=False, sort=True)
        ]
        if len(groups) <= 1:
            return (
                working.sample(n=sample_size, random_state=42)
                .drop(columns=["_sample_row_id"])
                .reset_index(drop=True)
            )

        # Allocate one row per domain when capacity permits, then distribute the
        # remaining slots round-robin across domains that still have capacity.
        allocations = {name: 0 for name, _ in groups}
        if sample_size >= len(groups):
            for name, group in groups:
                if len(group) > 0:
                    allocations[name] = 1
        else:
            # Exact size takes priority when the requested sample is smaller than
            # the number of domains. Prefer the largest domains deterministically.
            ranked = sorted(groups, key=lambda item: (-len(item[1]), item[0]))
            for name, _ in ranked[:sample_size]:
                allocations[name] = 1

        remaining_slots = sample_size - sum(allocations.values())
        while remaining_slots > 0:
            progressed = False
            for name, group in groups:
                if remaining_slots <= 0:
                    break
                if allocations[name] < len(group):
                    allocations[name] += 1
                    remaining_slots -= 1
                    progressed = True
            if not progressed:
                break

        selected_ids: list[int] = []
        for offset, (name, group) in enumerate(groups):
            take = allocations[name]
            if take > 0:
                selected_ids.extend(
                    group.sample(n=take, random_state=42 + offset)["_sample_row_id"].tolist()
                )

        sampled = working[working["_sample_row_id"].isin(selected_ids)].copy()
        if len(sampled) < sample_size:
            remainder = working[~working["_sample_row_id"].isin(selected_ids)]
            need = min(sample_size - len(sampled), len(remainder))
            if need > 0:
                sampled = pd.concat(
                    [sampled, remainder.sample(n=need, random_state=42)],
                    ignore_index=True,
                )

        return (
            sampled.head(sample_size)
            .drop(columns=["_sample_row_id"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Workflow preparation
    # ------------------------------------------------------------------
    def prepare_csv_workflow(self, input_path: str | Path, domain: Optional[str]) -> pd.DataFrame:
        path = self.resolve_path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV input file not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        detected_domain = self.detect_domain("csv", df=df, domain=domain)

        self.state.domain = detected_domain
        self.state.input_path = str(path)
        self.state.selected_workflow = "dataset_workflow"
        self.state.next_agent = "Data Standardisation Agent"

        self.add_trace(
            "Input Type Detection",
            "CSV dataset detected.",
            {
                "path": str(path),
                "rows": len(df),
                "columns": list(df.columns),
                "domain": detected_domain,
            },
        )

        standard_df = self.standardize_known_dataframe(
            df=df,
            domain=detected_domain,
            raw_source_path=str(path),
        )

        standard_df = self.sample_dataframe(standard_df, self.sample_size)

        prepared_path = self.run_output_dir / "prepared_standardised_dataset.csv"
        standard_df.to_csv(prepared_path, index=False, encoding="utf-8")

        self.state.prepared_dataset_path = str(prepared_path)

        self.add_trace(
            "Data Standardisation Agent",
            "CSV dataset converted to common multi-domain schema.",
            {
                "rows": len(standard_df),
                "columns": list(standard_df.columns),
                "saved_to": str(prepared_path),
            },
        )

        return standard_df

    def prepare_single_review_workflow(
        self,
        input_text: str,
        domain: Optional[str],
        rating: float,
        entity_id: str = "single_review_entity",
        entity_name: str = "Single Review Entity",
    ) -> pd.DataFrame:
        selected_domain = domain or "mobile_app"

        if selected_domain not in self.DOMAIN_VALUES:
            raise ValueError(f"Invalid domain: {selected_domain}")

        self.state.domain = selected_domain
        self.state.selected_workflow = "single_review_workflow"
        self.state.next_agent = "Single Review Pipeline"

        self.add_trace(
            "Input Type Detection",
            "Single review text detected.",
            {
                "domain": selected_domain,
                "rating": rating,
            },
        )

        df = pd.DataFrame([
            {
                "review_id": "single_review_1",
                "domain": selected_domain,
                "entity_id": entity_id,
                "entity_name": entity_name,
                "review_text": self.clean_text(input_text),
                "rating": self.safe_rating(rating),
                "rating_original": rating,
                "review_date": datetime.now().date().isoformat(),
                "source": "single_review_input",
                "raw_source_path": "manual_input",
            }
        ])

        prepared_path = self.run_output_dir / "prepared_single_review.csv"
        df.to_csv(prepared_path, index=False, encoding="utf-8")

        self.state.prepared_dataset_path = str(prepared_path)

        self.add_trace(
            "Single Review Workflow Prepared",
            "Single review converted to standard schema.",
            {
                "rows": len(df),
                "saved_to": str(prepared_path),
            },
        )

        return df

    def scrape_google_play_reviews(self, app_id: str, max_reviews: int) -> pd.DataFrame:
        """
        Scrapes Google Play reviews using google-play-scraper if installed.

        Install if needed:
            pip install google-play-scraper
        """
        try:
            from google_play_scraper import app, reviews, Sort
        except ImportError as exc:
            raise ImportError(
                "google-play-scraper is required for URL/app-id workflow. "
                "Install it with: pip install google-play-scraper"
            ) from exc

        collected, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=max_reviews,
        )

        if not collected:
            raise ValueError(f"No reviews collected for app id: {app_id}")

        raw_df = pd.DataFrame(collected)

        if "appId" not in raw_df.columns:
            raw_df["appId"] = app_id

        # Resolve the readable app title once. Failure to fetch metadata should
        # not prevent review analysis; package id remains a safe fallback.
        app_title = app_id
        try:
            app_details = app(app_id, lang="en", country="us")
            app_title = str(app_details.get("title") or app_id).strip()
        except Exception:
            pass
        raw_df["appTitle"] = app_title

        return raw_df

    def prepare_google_play_workflow(
        self,
        url: Optional[str] = None,
        app_id: Optional[str] = None,
        max_reviews: Optional[int] = None,
    ) -> pd.DataFrame:
        app_id_value = app_id or self.extract_google_play_app_id(url or "")

        if not app_id_value:
            raise ValueError("Could not extract Google Play app id from input.")

        self.state.domain = "mobile_app"
        self.state.selected_workflow = "scraper_workflow"
        self.state.next_agent = "Google Play Scraper Agent"
        self.state.app_id = app_id_value
        self.state.source_url = url

        self.add_trace(
            "Input Type Detection",
            "Google Play URL/app id detected.",
            {
                "app_id": app_id_value,
                "url": url,
            },
        )

        raw_df = self.scrape_google_play_reviews(
            app_id=app_id_value,
            max_reviews=max_reviews or self.max_reviews,
        )

        raw_path = self.run_output_dir / f"raw_google_play_{app_id_value}.csv"
        raw_df.to_csv(raw_path, index=False, encoding="utf-8")

        self.add_trace(
            "Google Play Scraper Agent",
            "Collected Google Play reviews.",
            {
                "rows": len(raw_df),
                "saved_to": str(raw_path),
            },
        )

        standard_df = self.standardize_mobile_app(
            raw_df,
            raw_source_path=str(raw_path),
        )

        prepared_path = self.run_output_dir / "prepared_google_play_reviews.csv"
        standard_df.to_csv(prepared_path, index=False, encoding="utf-8")

        self.state.prepared_dataset_path = str(prepared_path)

        self.add_trace(
            "Data Standardisation Agent",
            "Scraped reviews converted to common schema.",
            {
                "rows": len(standard_df),
                "saved_to": str(prepared_path),
            },
        )

        return standard_df

    def prepare_google_maps_workflow(
        self,
        url: str,
        domain: str = "auto",
        max_reviews: Optional[int] = None,
        sort_order: str = "most_relevant",
    ) -> pd.DataFrame:
        if not self.looks_like_google_maps_url(url):
            raise ValueError("An exact Google Maps place URL is required.")

        self.state.selected_workflow = "google_maps_scraper_workflow"
        self.state.next_agent = "Google Maps Scraper Agent"
        self.state.source_url = url

        self.add_trace(
            "Input Type Detection",
            "Google Maps place URL detected.",
            {"url": url, "requested_domain": domain},
        )

        from agents.google_maps_scraper_agent import GoogleMapsScraperAgent

        agent = GoogleMapsScraperAgent()
        result = agent.run(
            place_url=url,
            output_dir=self.run_output_dir,
            max_reviews=max_reviews or self.max_reviews,
            sort_order=sort_order,
            domain=domain,
        )

        prepared_path = Path(result["prepared_csv"])
        standard_df = pd.read_csv(prepared_path)

        self.state.domain = result.get("domain") or "restaurant"
        self.state.place_id = result.get("entity_id")
        self.state.prepared_dataset_path = str(prepared_path)
        self.state.source_url = result.get("resolved_url") or url

        self.add_trace(
            "Google Maps Scraper Agent",
            "Collected visible public reviews and converted them to the common schema.",
            {
                "entity_name": result.get("entity_name"),
                "category": result.get("category"),
                "domain": self.state.domain,
                "rows": len(standard_df),
                "saved_to": str(prepared_path),
            },
        )

        return standard_df

    # ------------------------------------------------------------------
    # Pipeline and Groq final report
    # ------------------------------------------------------------------
    def run_analysis_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        from services.local_model_registry import require_all_local_models

        # Fail before analysis when the delivery is missing a model/config/weight
        # instead of silently presenting rating/keyword fallbacks as advanced AI.
        require_all_local_models()

        from pipeline.multidomain_review_analysis_pipeline import MultiDomainReviewAnalysisPipeline

        pipeline_output_dir = self.run_output_dir / "analysis_pipeline"

        pipeline = MultiDomainReviewAnalysisPipeline(
            model_path=str(self.model_path),
            use_transformer=True,
            use_discrepancy_model=True,
            use_semantic_issue_model=True,
            use_rag=self.use_rag,
            output_dir=str(pipeline_output_dir),
        )

        self.add_trace(
            "Analysis Pipeline Started",
            "Orchestrator passed prepared data to specialised analytical agents.",
            {
                "rows": len(df),
                "use_rag": self.use_rag,
                "output_dir": str(pipeline_output_dir),
            },
        )

        results = pipeline.analyze(df, save_outputs=True)

        review_df = results["review_level_results"]
        entity_df = results["entity_level_summary"]

        strict_models = os.environ.get("STRICT_MODEL_EXECUTION", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if strict_models:
            expected_models = {
                "sentiment_model_used": "distilbert",
                "discrepancy_model_used": "nlptown_bert",
                "issue_model_used": "minilm_semantic",
            }
            degraded = []
            for column, expected in expected_models.items():
                if column not in review_df.columns:
                    degraded.append(f"{column} missing")
                    continue
                used = sorted(set(review_df[column].fillna("").astype(str)))
                if used != [expected]:
                    degraded.append(f"{column}={used}")
            if self.use_rag and any(
                item.get("step") == "RAG Evidence Retrieval Fallback"
                for item in results.get("execution_trace", [])
            ):
                degraded.append("RAG evidence retrieval used fallback")
            if degraded:
                raise RuntimeError(
                    "Strict model execution failed; the pipeline attempted a fallback: "
                    + "; ".join(degraded)
                )

        self.add_trace(
            "Specialised Agents Completed",
            "Sentiment, discrepancy, issue mining, RAG, risk scoring, explainability and entity summary completed.",
            {
                "review_rows": len(review_df),
                "entities": len(entity_df),
                "pipeline_output_dir": str(pipeline_output_dir),
            },
        )

        return results

    def run_groq_summary(
        self,
        review_df: pd.DataFrame,
        entity_df: pd.DataFrame,
        domain: Optional[str] = None,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> Optional[str]:
        if not self.use_groq:
            self.add_trace(
                "Groq Final Summary Skipped",
                "Groq final summary disabled by user flag.",
                {},
            )
            return None

        try:
            from agents.groq_final_summary_agent import GroqFinalSummaryAgent

            agent = GroqFinalSummaryAgent(env_path=PROJECT_ROOT / ".env")
            result = agent.generate_report(
                review_df=review_df,
                entity_df=entity_df,
                entity_id=entity_id,
                entity_name=entity_name,
                domain=domain if domain and domain != "multidomain" else None,
            )

            report_path = self.run_output_dir / "final_groq_report.txt"
            context_path = self.run_output_dir / "final_groq_context_payload.json"

            report_path.write_text(result["final_report"], encoding="utf-8")
            context_path.write_text(result["context_payload"], encoding="utf-8")

            self.state.groq_report_path = str(report_path)

            self.add_trace(
                "Groq Final Summary Agent",
                "Groq generated final readable trust/risk report from structured agent outputs.",
                {
                    "report_path": str(report_path),
                    "context_path": str(context_path),
                    "selected_entity": {
                        "domain": result["selected_entity"].get("domain"),
                        "entity_name": result["selected_entity"].get("entity_name"),
                        "average_trust_score": result["selected_entity"].get("average_trust_score"),
                        "overall_risk_level": result["selected_entity"].get("overall_risk_level"),
                    },
                },
            )

            return result["final_report"]

        except Exception as exc:
            error_path = self.run_output_dir / "groq_summary_error.txt"
            error_path.write_text(str(exc), encoding="utf-8")

            self.add_trace(
                "Groq Final Summary Failed",
                "Analytical pipeline completed, but Groq summary failed.",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "saved_to": str(error_path),
                },
            )
            return None

    # ------------------------------------------------------------------
    # Main orchestrator run
    # ------------------------------------------------------------------
    def run(
        self,
        input_value: Optional[str] = None,
        input_path: Optional[str | Path] = None,
        input_text: Optional[str] = None,
        url: Optional[str] = None,
        app_id: Optional[str] = None,
        input_type: str = "auto",
        domain: Optional[str] = None,
        sort_order: str = "most_relevant",
        rating: float = 3.0,
        entity_id: str = "manual_entity",
        entity_name: str = "Manual Entity",
    ) -> Dict[str, Any]:
        try:
            detected_type = self.detect_input_type(
                input_value=input_value,
                input_path=input_path,
                input_text=input_text,
                url=url,
                app_id=app_id,
                input_type=input_type,
            )

            self.state.input_type = detected_type

            self.add_trace(
                "Orchestrator Input Reasoning",
                "Orchestrator received user input and selected input type.",
                {
                    "input_type": detected_type,
                },
            )

            if detected_type == "csv":
                path = input_path or input_value
                prepared_df = self.prepare_csv_workflow(path, domain=domain)

            elif detected_type == "single_review":
                text = input_text or input_value
                prepared_df = self.prepare_single_review_workflow(
                    input_text=text,
                    domain=domain,
                    rating=rating,
                    entity_id=entity_id,
                    entity_name=entity_name,
                )

            elif detected_type in {"google_play_url", "app_id"}:
                prepared_df = self.prepare_google_play_workflow(
                    url=url or input_value,
                    app_id=app_id if app_id else None,
                    max_reviews=self.max_reviews,
                )

            elif detected_type == "google_maps_url":
                prepared_df = self.prepare_google_maps_workflow(
                    url=url or input_value or "",
                    domain=domain or "auto",
                    max_reviews=self.max_reviews,
                    sort_order=sort_order,
                )

            else:
                raise ValueError(f"Unsupported input type: {detected_type}")

            results = self.run_analysis_pipeline(prepared_df)

            review_df = results["review_level_results"]
            entity_df = results["entity_level_summary"]

            final_report = self.run_groq_summary(
                review_df=review_df,
                entity_df=entity_df,
                domain=(self.state.domain if not domain or domain == "auto" else domain),
                entity_id=entity_id if entity_id != "manual_entity" else None,
                entity_name=entity_name if entity_name != "Manual Entity" else None,
            )

            self.state.status = "completed"

            self.add_trace(
                "Orchestrator Finalisation",
                "Orchestrator collected all agent outputs and finalised the run.",
                {
                    "status": self.state.status,
                    "output_dir": str(self.run_output_dir),
                    "final_report_generated": final_report is not None,
                },
            )

            self.save_state()

            return {
                "state": asdict(self.state),
                "results": results,
                "final_report": final_report,
                "output_dir": str(self.run_output_dir),
            }

        except Exception as exc:
            self.state.status = "failed"
            self.add_trace(
                "Orchestrator Failed",
                "Run failed before successful finalisation.",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self.save_state()
            raise

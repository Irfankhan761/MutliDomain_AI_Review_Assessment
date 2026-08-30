from urllib.parse import urlparse, parse_qs
from pathlib import Path
import re
import pandas as pd

from google_play_scraper import reviews, Sort


class GooglePlayScraperService:
    def __init__(self, lang: str = "en", country: str = "us"):
        self.lang = lang
        self.country = country
        self.last_metadata = {}

    def extract_app_id(self, app_url_or_id: str) -> str:
        if not app_url_or_id or not str(app_url_or_id).strip():
            raise ValueError("App URL or appId is required.")

        value = str(app_url_or_id).strip()

        if "play.google.com" not in value:
            return value

        parsed_url = urlparse(value)
        query_params = parse_qs(parsed_url.query)
        app_id = query_params.get("id", [None])[0]

        if not app_id:
            raise ValueError("Could not extract appId from Google Play URL.")

        return app_id

    def get_sort_mode(self, sort_order: str):
        sort_order = str(sort_order).lower().strip()

        if sort_order in ["relevant", "most_relevant", "helpful"]:
            return Sort.MOST_RELEVANT, "most_relevant"

        return Sort.NEWEST, "newest"

    def scrape_reviews(
        self,
        app_url_or_id: str,
        count: int = 1000,
        sort_order: str = "newest"
    ) -> pd.DataFrame:
        app_id = self.extract_app_id(app_url_or_id)
        sort_mode, sort_label = self.get_sort_mode(sort_order)

        result, _ = reviews(
            app_id,
            lang=self.lang,
            country=self.country,
            sort=sort_mode,
            count=count
        )

        if not result:
            raise ValueError(f"No reviews found for appId: {app_id}")

        df = pd.DataFrame(result)
        df["appId"] = app_id

        useful_columns = [
            "reviewId",
            "content",
            "score",
            "thumbsUpCount",
            "reviewCreatedVersion",
            "at",
            "replyContent",
            "repliedAt",
            "appId"
        ]

        existing_columns = [column for column in useful_columns if column in df.columns]
        df = df[existing_columns]

        df = df.dropna(subset=["content", "score"])
        df = df.drop_duplicates(subset=["content", "score", "appId"])

        if "at" in df.columns and not df.empty:
            date_min = str(df["at"].min())
            date_max = str(df["at"].max())
        else:
            date_min = "not_available"
            date_max = "not_available"

        self.last_metadata = {
            "appId": app_id,
            "requested_count": count,
            "actual_scraped_count": len(df),
            "sort_order": sort_label,
            "language": self.lang,
            "country": self.country,
            "review_date_min": date_min,
            "review_date_max": date_max
        }

        df.attrs["scrape_metadata"] = self.last_metadata

        return df

    def get_metadata(self):
        return self.last_metadata

    def save_reviews(self, df: pd.DataFrame, output_dir: str = "outputs/live_scraping"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        app_id = df["appId"].iloc[0] if "appId" in df.columns and not df.empty else "unknown_app"
        safe_app_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", app_id)

        file_path = output_path / f"{safe_app_id}_scraped_reviews.csv"
        df.to_csv(file_path, index=False)

        return file_path
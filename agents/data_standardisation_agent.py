from pathlib import Path
import hashlib
import pandas as pd


class DataStandardisationAgent:
    """
    Converts different review datasets into one common schema:
    review_id, domain, entity_id, entity_name, review_text, rating,
    rating_original, review_date, source, raw_source_path
    """

    COMMON_COLUMNS = [
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

    def __init__(self):
        pass

    def read_csv(self, file_path):
        encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1"]

        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, low_memory=False)
            except Exception:
                continue

        raise ValueError(f"Could not read file: {file_path}")

    def create_review_id(self, domain, entity_id, review_text):
        raw = f"{domain}_{entity_id}_{review_text}"
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def clean_text(self, value):
        if pd.isna(value):
            return ""

        text = str(value).strip()

        remove_values = [
            "No Negative",
            "No Positive",
            "No Negative ",
            "No Positive ",
            "nan",
            "None",
        ]

        if text in remove_values:
            return ""

        return text

    def final_cleanup(self, df):
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        df = df[df["review_text"] != ""].copy()

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"]).copy()

        df["rating"] = df["rating"].clip(1, 5)

        for col in self.COMMON_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[self.COMMON_COLUMNS].copy()
        df = df.drop_duplicates(subset=["domain", "entity_id", "review_text"]).copy()

        return df.reset_index(drop=True)

    def standardise_amazon(self, file_path):
        df = self.read_csv(file_path)

        required = ["rating", "title", "text", "asin", "parent_asin"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Amazon dataset missing columns: {missing}")

        out = pd.DataFrame()
        out["domain"] = "ecommerce"
        out["entity_id"] = df["parent_asin"].fillna(df["asin"]).astype(str)
        out["entity_name"] = df["parent_asin"].fillna(df["asin"]).astype(str)

        title = df["title"].fillna("").astype(str)
        text = df["text"].fillna("").astype(str)
        out["review_text"] = (title + ". " + text).str.strip()

        out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        out["rating_original"] = df["rating"]
        out["review_date"] = df["timestamp"] if "timestamp" in df.columns else ""
        out["source"] = "amazon_reviews_2023_all_beauty"
        out["raw_source_path"] = str(file_path)

        out["review_id"] = out.apply(
            lambda row: self.create_review_id(
                row["domain"], row["entity_id"], row["review_text"]
            ),
            axis=1,
        )

        return self.final_cleanup(out)

    def standardise_hotel(self, file_path):
        df = self.read_csv(file_path)

        required = ["Hotel_Name", "Positive_Review", "Negative_Review", "Reviewer_Score"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Hotel dataset missing columns: {missing}")

        positive = df["Positive_Review"].apply(self.clean_text)
        negative = df["Negative_Review"].apply(self.clean_text)

        out = pd.DataFrame()
        out["domain"] = "hotel"
        out["entity_id"] = df["Hotel_Name"].astype(str)
        out["entity_name"] = df["Hotel_Name"].astype(str)
        out["review_text"] = (negative + ". " + positive).str.strip(". ").str.strip()

        reviewer_score = pd.to_numeric(df["Reviewer_Score"], errors="coerce")
        out["rating"] = reviewer_score / 2.0
        out["rating_original"] = df["Reviewer_Score"]
        out["review_date"] = df["Review_Date"] if "Review_Date" in df.columns else ""
        out["source"] = "hotel_reviews_europe_515k"
        out["raw_source_path"] = str(file_path)

        out["review_id"] = out.apply(
            lambda row: self.create_review_id(
                row["domain"], row["entity_id"], row["review_text"]
            ),
            axis=1,
        )

        return self.final_cleanup(out)

    def standardise_mobile_app(self, file_path):
        df = self.read_csv(file_path)

        required = ["content", "score", "appId"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Mobile app dataset missing columns: {missing}")

        out = pd.DataFrame()
        out["domain"] = "mobile_app"
        out["entity_id"] = df["appId"].astype(str)
        out["entity_name"] = df["appId"].astype(str)
        out["review_text"] = df["content"].fillna("").astype(str)
        out["rating"] = pd.to_numeric(df["score"], errors="coerce")
        out["rating_original"] = df["score"]
        out["review_date"] = df["at"] if "at" in df.columns else ""
        out["source"] = "google_play_reviews"
        out["raw_source_path"] = str(file_path)

        if "reviewId" in df.columns:
            out["review_id"] = df["reviewId"].astype(str)
        else:
            out["review_id"] = out.apply(
                lambda row: self.create_review_id(
                    row["domain"], row["entity_id"], row["review_text"]
                ),
                axis=1,
            )

        return self.final_cleanup(out)

    def standardise_yelp(self, file_path):
        df = self.read_csv(file_path)

        required = ["Yelp URL", "Rating", "Review Text"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Yelp dataset missing columns: {missing}")

        out = pd.DataFrame()
        out["domain"] = "restaurant"
        out["entity_id"] = df["Yelp URL"].astype(str)
        out["entity_name"] = df["Yelp URL"].astype(str)
        out["review_text"] = df["Review Text"].fillna("").astype(str)
        out["rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        out["rating_original"] = df["Rating"]
        out["review_date"] = df["Date"] if "Date" in df.columns else ""
        out["source"] = "yelp_restaurant_reviews"
        out["raw_source_path"] = str(file_path)

        out["review_id"] = out.apply(
            lambda row: self.create_review_id(
                row["domain"], row["entity_id"], row["review_text"]
            ),
            axis=1,
        )

        return self.final_cleanup(out)

    def standardise_file(self, file_path, domain):
        file_path = Path(file_path)

        if domain == "ecommerce":
            return self.standardise_amazon(file_path)

        if domain == "hotel":
            return self.standardise_hotel(file_path)

        if domain == "mobile_app":
            return self.standardise_mobile_app(file_path)

        if domain == "restaurant":
            return self.standardise_yelp(file_path)

        raise ValueError(f"Unsupported domain: {domain}")
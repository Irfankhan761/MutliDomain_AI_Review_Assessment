from pathlib import Path
import argparse
import pandas as pd
from datasets import load_dataset


def download_amazon_all_beauty(
    output_path: str,
    sample_size: int = 30000
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading Amazon Reviews 2023 - All_Beauty Parquet file...")

    parquet_url = (
        "hf://datasets/"
        "McAuley-Lab/Amazon-Reviews-2023@"
        "269765acc4057f11566d9e16106d109409e4c7c8/"
        "raw_review_All_Beauty/full-00000-of-00001.parquet"
    )

    dataset = load_dataset(
        "parquet",
        data_files={"full": parquet_url},
        split="full"
    )

    print("Total records available:", len(dataset))

    if sample_size is not None and sample_size > 0:
        actual_sample_size = min(sample_size, len(dataset))

        dataset = (
            dataset
            .shuffle(seed=42)
            .select(range(actual_sample_size))
        )

        print("Selected sample size:", len(dataset))

    print("Total records available:", len(dataset))

    if sample_size is not None and sample_size > 0:
        sample_size = min(sample_size, len(dataset))

        dataset = (
            dataset
            .shuffle(seed=42)
            .select(range(sample_size))
        )

    required_columns = [
        "rating",
        "title",
        "text",
        "asin",
        "parent_asin",
        "timestamp",
        "helpful_vote",
        "verified_purchase"
    ]

    available_columns = [
        column
        for column in required_columns
        if column in dataset.column_names
    ]

    df = dataset.select_columns(available_columns).to_pandas()

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\nAmazon All_Beauty dataset saved successfully.")
    print("Output path:", output_path.resolve())
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/ecommerce_amazon/"
                "amazon_all_beauty_sample.csv"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000
    )

    args = parser.parse_args()

    download_amazon_all_beauty(
        output_path=args.output,
        sample_size=args.sample_size
    )
from pathlib import Path
import argparse
import sys
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.local_model_registry import (
    get_distilbert_sentiment_path,
    get_minilm_path,
    get_rating_model_path,
    require_all_local_models,
)


def print_header(title):
    print("\n" + title)
    print("=" * 90)


def save_minilm(local_path: Path, cache_only: bool):
    print_header("1. MiniLM semantic embedding model")

    local_path.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers first: pip install sentence-transformers") from exc

    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    print("Target local folder:", local_path)
    print("Cache only:", cache_only)

    # If already saved properly, skip.
    if (local_path / "modules.json").exists():
        print("Already exists. Skipping MiniLM download/copy.")
        return

    kwargs = {}
    if cache_only:
        kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Loading MiniLM from local cache" if cache_only else "Downloading/loading MiniLM once")
    try:
        model = SentenceTransformer(model_id, **kwargs)
    except TypeError:
        # Older sentence-transformers may not accept local_files_only.
        if cache_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = SentenceTransformer(model_id)

    print("Saving MiniLM to project local folder...")
    model.save(str(local_path))
    print("MiniLM saved:", local_path)


def save_rating_model(local_path: Path, cache_only: bool):
    print_header("2. nlptown BERT rating model")

    local_path.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError("Install transformers first: pip install transformers") from exc

    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"

    print("Target local folder:", local_path)
    print("Cache only:", cache_only)

    if (local_path / "config.json").exists() and (
        (local_path / "model.safetensors").exists()
        or (local_path / "pytorch_model.bin").exists()
    ):
        print("Already exists. Skipping rating model download/copy.")
        return

    print("Loading rating model from local cache" if cache_only else "Downloading/loading rating model once")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=cache_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=cache_only)

    print("Saving rating model to project local folder...")
    tokenizer.save_pretrained(str(local_path))
    model.save_pretrained(str(local_path))
    print("Rating model saved:", local_path)


def check_distilbert(local_path: Path):
    print_header("3. DistilBERT fine-tuned sentiment model")

    print("Expected local folder:", local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"DistilBERT local model missing: {local_path}\n"
            "Run Phase 6 training again or restore outputs/models/distilbert_sentiment"
        )

    expected = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for item in expected:
        print(item, "OK" if (local_path / item).exists() else "MISSING")

    model_file_ok = (local_path / "model.safetensors").exists() or (local_path / "pytorch_model.bin").exists()
    print("model weights", "OK" if model_file_ok else "MISSING")

    if not model_file_ok:
        raise FileNotFoundError(
            f"DistilBERT model weights missing in {local_path}"
        )

    print("DistilBERT local model OK.")


def main():
    parser = argparse.ArgumentParser(description="Download/copy all Hugging Face models into project local folders.")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not contact Hugging Face. Use only already cached files. If cache is missing, fail.",
    )
    args = parser.parse_args()

    print_header("LOCAL MODEL PACKAGING STARTED")
    print("Project root:", PROJECT_ROOT)
    print("Mode:", "CACHE ONLY / NO INTERNET" if args.cache_only else "ONLINE ALLOWED ONCE")

    if args.cache_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    check_distilbert(get_distilbert_sentiment_path())
    save_minilm(get_minilm_path(), cache_only=args.cache_only)
    save_rating_model(get_rating_model_path(), cache_only=args.cache_only)

    print_header("LOCAL MODEL VALIDATION")
    paths = require_all_local_models()

    for name, path in paths.items():
        print(name, "=>", path)

    print("\nLOCAL MODEL PACKAGING COMPLETE")
    print("After this, normal project runs should not contact Hugging Face.")


if __name__ == "__main__":
    main()

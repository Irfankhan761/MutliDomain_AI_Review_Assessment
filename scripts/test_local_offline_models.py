from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.local_model_registry import (
    enforce_offline_mode,
    require_all_local_models,
    get_distilbert_sentiment_path,
    get_minilm_path,
    get_rating_model_path,
)


def print_header(title):
    print("\n" + title)
    print("=" * 90)


def main():
    print_header("OFFLINE LOCAL MODEL TEST")
    enforce_offline_mode(PROJECT_ROOT / ".env")

    print("HF_HUB_OFFLINE:", os.getenv("HF_HUB_OFFLINE"))
    print("TRANSFORMERS_OFFLINE:", os.getenv("TRANSFORMERS_OFFLINE"))

    paths = require_all_local_models()

    for name, path in paths.items():
        print(name, "=>", path)

    # 1) Test DistilBERT sentiment local
    print_header("1. DistilBERT sentiment local loading")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    sent_path = get_distilbert_sentiment_path()
    sent_tokenizer = AutoTokenizer.from_pretrained(str(sent_path), local_files_only=True)
    sent_model = AutoModelForSequenceClassification.from_pretrained(str(sent_path), local_files_only=True)

    text = "The app keeps crashing and I cannot login."
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = sent_model(**inputs)
        pred = int(outputs.logits.argmax(dim=1).item())

    print("DistilBERT loaded locally. Prediction class:", pred)

    # 2) Test MiniLM local
    print_header("2. MiniLM local loading")
    from sentence_transformers import SentenceTransformer

    minilm_path = get_minilm_path()

    try:
        minilm = SentenceTransformer(str(minilm_path), local_files_only=True)
    except TypeError:
        minilm = SentenceTransformer(str(minilm_path))

    emb = minilm.encode(["app crash issue", "hotel cleanliness problem"], normalize_embeddings=True)
    print("MiniLM loaded locally. Embedding shape:", emb.shape)

    # 3) Test rating model local
    print_header("3. nlptown BERT rating model local loading")

    rating_path = get_rating_model_path()
    rating_tokenizer = AutoTokenizer.from_pretrained(str(rating_path), local_files_only=True)
    rating_model = AutoModelForSequenceClassification.from_pretrained(str(rating_path), local_files_only=True)

    inputs = rating_tokenizer("This product is fake and very bad.", return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = rating_model(**inputs)
        star = int(outputs.logits.argmax(dim=1).item()) + 1

    print("Rating model loaded locally. Predicted star:", star)

    print_header("OFFLINE LOCAL MODEL TEST PASSED")
    print("No Hugging Face model name was used in this test.")
    print("If this test shows no HTTP/429 lines, local offline loading is working.")


if __name__ == "__main__":
    main()

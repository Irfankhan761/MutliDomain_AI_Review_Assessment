from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_evidence_agent import RAGEvidenceRetrievalAgent

agent = RAGEvidenceRetrievalAgent(
    corpus_path="data/processed/combined_multidomain_reviews.csv",
    index_dir="outputs/rag_corpus",
    top_k=3,
    auto_build_if_missing=True,
)
agent.load_corpus()

print("\nPERSISTENT COMBINED-CORPUS RAG TEST")
print("=" * 110)
print("Backend:", agent.backend)
print("Source rows:", agent.manifest.get("source_rows"))
print("Indexed unique rows:", agent.manifest.get("indexed_unique_rows"))
print("Duplicate texts removed:", agent.manifest.get("duplicate_text_rows_removed"))
print("Domains:", agent.manifest.get("domains"))

# Test 1: query one exact corpus review. It MUST NOT retrieve itself or an exact duplicate.
corpus_row = agent.metadata[
    agent.metadata["domain"].astype(str).eq("mobile_app")
].iloc[0]

query = pd.DataFrame([{
    "review_id": corpus_row["review_id"],
    "domain": "mobile_app",
    "entity_id": corpus_row.get("entity_id", ""),
    "entity_name": corpus_row.get("entity_name", ""),
    "review_text": corpus_row["review_text"],
    "clean_review": corpus_row["review_text"],
    "rating": corpus_row.get("rating", 3),
    "primary_issue": "no_issue",
}])

out = agent.process(query, text_column="clean_review")
row = out.iloc[0]

print("\nTEST 1 — SELF/DUPLICATE EXCLUSION")
print("-" * 110)
print("Query review id:", query.iloc[0]["review_id"])
print("Query text:", query.iloc[0]["review_text"])
print("Retrieved review id:", row["rag_top_evidence_review_id"])
print("Retrieved text:", row["rag_evidence_text"])
print("Similarity:", row["rag_similarity_score"])
print("Available:", row["rag_evidence_available"])

if row["rag_evidence_available"]:
    assert str(row["rag_top_evidence_review_id"]) != str(query.iloc[0]["review_id"])
    assert agent.text_hash(row["rag_evidence_text"]) != agent.text_hash(query.iloc[0]["review_text"])

# Test 2: the current Airbnb-style complaint should retrieve a DIFFERENT mobile-app corpus review.
airbnb_text = (
    "Avoid Airbnb! Poor support and zero protection for guests. "
    "They allowed a host to lie about me in a review and refused to remove it "
    "despite having official proof. I was entitled to a refund for the property's bad condition."
)
query2 = pd.DataFrame([{
    "review_id": "runtime_airbnb_debug_review",
    "domain": "mobile_app",
    "entity_id": "com.airbnb.android",
    "entity_name": "Airbnb",
    "review_text": airbnb_text,
    "clean_review": airbnb_text,
    "rating": 1.0,
    "primary_issue": "no_issue",
}])
out2 = agent.process(query2, text_column="clean_review")
row2 = out2.iloc[0]

print("\nTEST 2 — INDEPENDENT COMBINED-DATASET EVIDENCE")
print("-" * 110)
print("Query:", airbnb_text)
print("Retrieved:", row2["rag_evidence_text"])
print("Retrieved review id:", row2["rag_top_evidence_review_id"])
print("Retrieved entity:", row2["rag_top_evidence_entity_name"])
print("Retrieved source:", row2["rag_top_evidence_source"])
print("Similarity:", row2["rag_similarity_score"])
print("Backend:", row2["rag_backend"])
print("Corpus:", row2["rag_corpus_source"])
print("Top-k:", row2["rag_top_k_json"])

assert agent.text_hash(row2["rag_evidence_text"]) != agent.text_hash(airbnb_text), (
    "FAIL: RAG returned the query review itself."
)
assert str(row2["rag_backend"]) in {"faiss_ip", "numpy_cosine"}
assert int(row2["rag_corpus_source_rows"]) == 20000

topk = json.loads(row2["rag_top_k_json"])
for item in topk:
    assert item["domain"] == "mobile_app"
    assert agent.text_hash(item["review_text"]) != agent.text_hash(airbnb_text)

print("\nRAG REGRESSION RESULT: PASS")
print("Evidence is retrieved from the persistent combined corpus, domain-filtered, and self/duplicate-excluded.")

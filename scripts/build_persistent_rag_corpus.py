from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_evidence_agent import RAGEvidenceRetrievalAgent

agent = RAGEvidenceRetrievalAgent(
    corpus_path="data/processed/combined_multidomain_reviews.csv",
    index_dir="outputs/rag_corpus",
    auto_build_if_missing=True,
)

manifest = agent.build_corpus(force=True)

print("\nPERSISTENT RAG CORPUS BUILT")
print("=" * 90)
for key, value in manifest.items():
    print(f"{key:30} {value}")

print("\nExpected source rows: 20,000")
print("The retrieval index may contain fewer rows because exact normalized duplicate texts are removed.")
print("FAISS is used when available; saved normalized NumPy embeddings remain as a fallback.")

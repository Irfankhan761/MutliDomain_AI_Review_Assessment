"""
Persistent Combined-Corpus RAG Evidence Retrieval Agent.

Source corpus:
    data/processed/combined_multidomain_reviews.csv

Design:
- the canonical combined four-domain dataset is the persistent evidence corpus
- MiniLM embeddings are built once and saved under outputs/rag_corpus/
- FAISS inner-product indices are created per domain when faiss is available
- NumPy cosine is retained as an exact fallback over the same saved embeddings
- query reviews are NEVER inserted into the corpus at runtime
- exact/self duplicate evidence is excluded by review_id + normalized text hash
- duplicate evidence texts are de-duplicated before returning top-k results
- the raw query review is never used as a fake RAG fallback

This class keeps the interface used by MultiDomainReviewAnalysisPipeline:
    rag = RAGEvidenceRetrievalAgent()
    out = rag.process(df, text_column="clean_review")
    rag.retrieve_evidence_for_issues(issue_categories, top_k=3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import re

import numpy as np
import pandas as pd

from services.local_model_registry import (
    enforce_offline_mode,
    get_minilm_path,
    require_local_model,
)


class RAGEvidenceRetrievalAgent:
    DOMAIN_ALIASES = {
        "mobile app": "mobile_app",
        "mobile_app": "mobile_app",
        "app": "mobile_app",
        "google_play": "mobile_app",
        "e-commerce": "ecommerce",
        "e_commerce": "ecommerce",
        "ecommerce": "ecommerce",
        "amazon": "ecommerce",
        "hotel": "hotel",
        "hotels": "hotel",
        "restaurant": "restaurant",
        "restaurants": "restaurant",
        "yelp": "restaurant",
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        corpus_path: str | Path = "data/processed/combined_multidomain_reviews.csv",
        index_dir: str | Path = "outputs/rag_corpus",
        top_k: int = 3,
        candidate_pool: int = 80,
        min_similarity: float = 0.30,
        batch_size: int = 64,
        issue_weight: float = 0.10,
        auto_build_if_missing: bool = True,
        exclude_same_entity: bool = False,
        *args,
        **kwargs,
    ):
        enforce_offline_mode()

        self.root = Path(__file__).resolve().parents[1]
        self.model_path = require_local_model(
            model_path or get_minilm_path(),
            "MiniLM RAG evidence model",
        )

        self.corpus_path = self._resolve(corpus_path)
        self.index_dir = self._resolve(index_dir)
        self.top_k = max(1, int(top_k))
        self.candidate_pool = max(self.top_k * 4, int(candidate_pool))
        self.min_similarity = float(min_similarity)
        self.batch_size = max(1, int(batch_size))
        self.issue_weight = max(0.0, min(0.25, float(issue_weight)))
        self.auto_build_if_missing = bool(auto_build_if_missing)
        self.exclude_same_entity = bool(exclude_same_entity)

        self.metadata_path = self.index_dir / "metadata.csv"
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.manifest_path = self.index_dir / "manifest.json"
        self.faiss_dir = self.index_dir / "faiss"

        self.model = None
        self.metadata = None
        self.embeddings = None
        self.faiss = None
        self.faiss_indices = {}
        self.faiss_row_maps = {}
        self.backend = "not_loaded"
        self.manifest = {}

        self.last_issue_evidence: dict[str, list[dict[str, Any]]] = {}
        self._issue_vector_cache: dict[tuple[str, str], np.ndarray] = {}

    def _resolve(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @classmethod
    def normalize_domain(cls, value: Any) -> str:
        s = str(value or "").strip().lower().replace("-", "_")
        s = re.sub(r"\s+", " ", s)
        return cls.DOMAIN_ALIASES.get(s, s.replace(" ", "_"))

    @staticmethod
    def safe_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @classmethod
    def normalized_text(cls, value: Any) -> str:
        text = cls.safe_text(value).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def text_hash(cls, value: Any) -> str:
        normalized = cls.normalized_text(value)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def load_model(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(
                str(self.model_path),
                local_files_only=True,
            )
        except TypeError:
            self.model = SentenceTransformer(str(self.model_path))

    def _source_signature(self) -> dict:
        if not self.corpus_path.exists():
            return {}
        stat = self.corpus_path.stat()
        return {
            "path": str(self.corpus_path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _manifest_matches_source(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("source_signature") == self._source_signature()

    def corpus_ready(self) -> bool:
        return (
            self.metadata_path.exists()
            and self.embeddings_path.exists()
            and self.manifest_path.exists()
            and self._manifest_matches_source()
        )

    def ensure_corpus(self):
        if self.corpus_ready():
            return
        if not self.auto_build_if_missing:
            raise FileNotFoundError(
                "Persistent RAG corpus is missing or stale. Run "
                "python scripts\\build_persistent_rag_corpus.py"
            )
        self.build_corpus(force=True)

    def build_corpus(self, force: bool = False) -> dict:
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Combined RAG source dataset not found: {self.corpus_path}"
            )

        if self.corpus_ready() and not force:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))

        self.load_model()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        source = pd.read_csv(self.corpus_path, low_memory=False)
        required = {"domain", "review_text"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(
                f"Combined RAG corpus missing required columns: {sorted(missing)}"
            )

        source_rows = len(source)
        corpus = source.copy()
        corpus["domain"] = corpus["domain"].apply(self.normalize_domain)
        corpus["review_text"] = corpus["review_text"].apply(self.safe_text)
        corpus = corpus[
            corpus["domain"].isin(["ecommerce", "hotel", "mobile_app", "restaurant"])
            & corpus["review_text"].ne("")
        ].copy()

        # Preserve original IDs where available; generate stable IDs only if missing.
        if "review_id" not in corpus.columns:
            corpus["review_id"] = [f"corpus_review_{i}" for i in range(len(corpus))]
        corpus["review_id"] = corpus["review_id"].fillna("").astype(str)
        missing_id = corpus["review_id"].str.strip().eq("")
        corpus.loc[missing_id, "review_id"] = [
            f"corpus_review_{i}"
            for i in corpus.index[missing_id]
        ]

        for col in [
            "entity_id", "entity_name", "rating", "source",
            "review_date", "rating_original", "raw_source_path",
        ]:
            if col not in corpus.columns:
                corpus[col] = ""

        corpus["text_hash"] = corpus["review_text"].apply(self.text_hash)
        corpus["normalized_text"] = corpus["review_text"].apply(self.normalized_text)

        # Duplicate review text should not dominate nearest-neighbour evidence.
        # Keep one representative row per normalized text globally.
        before_dedupe = len(corpus)
        corpus = corpus.drop_duplicates(subset=["text_hash"], keep="first").reset_index(drop=True)
        duplicate_rows_removed = before_dedupe - len(corpus)

        texts = corpus["review_text"].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        if embeddings.ndim != 2 or len(embeddings) != len(corpus):
            raise RuntimeError(
                "MiniLM corpus embedding shape does not match metadata rows."
            )

        keep_cols = [
            "review_id", "domain", "entity_id", "entity_name",
            "review_text", "rating", "rating_original", "review_date",
            "source", "raw_source_path", "text_hash", "normalized_text",
        ]
        metadata = corpus[keep_cols].copy()
        metadata["corpus_row"] = np.arange(len(metadata), dtype=int)

        np.save(self.embeddings_path, embeddings)
        metadata.to_csv(self.metadata_path, index=False)

        faiss_built = False
        faiss_error = ""
        try:
            import faiss

            for old in self.faiss_dir.glob("*.faiss"):
                old.unlink()
            for old in self.faiss_dir.glob("*_rows.npy"):
                old.unlink()

            for domain in ["ecommerce", "hotel", "mobile_app", "restaurant"]:
                global_rows = np.flatnonzero(
                    metadata["domain"].to_numpy(dtype=str) == domain
                ).astype(np.int64)
                if len(global_rows) == 0:
                    continue

                domain_embeddings = np.ascontiguousarray(
                    embeddings[global_rows].astype(np.float32)
                )
                index = faiss.IndexFlatIP(domain_embeddings.shape[1])
                index.add(domain_embeddings)

                faiss.write_index(
                    index,
                    str(self.faiss_dir / f"{domain}.faiss"),
                )
                np.save(
                    self.faiss_dir / f"{domain}_rows.npy",
                    global_rows,
                )
            faiss_built = True
        except Exception as exc:
            faiss_error = f"{type(exc).__name__}: {exc}"

        domain_counts = {
            str(k): int(v)
            for k, v in metadata["domain"].value_counts().sort_index().items()
        }

        manifest = {
            "version": 2,
            "source_signature": self._source_signature(),
            "source_rows": int(source_rows),
            "eligible_rows_before_dedupe": int(before_dedupe),
            "indexed_unique_rows": int(len(metadata)),
            "duplicate_text_rows_removed": int(duplicate_rows_removed),
            "domains": domain_counts,
            "embedding_dimension": int(embeddings.shape[1]),
            "model_path": str(self.model_path),
            "faiss_built": bool(faiss_built),
            "faiss_error": faiss_error,
            "numpy_embeddings_saved": True,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Force a clean reload after rebuilding.
        self.metadata = None
        self.embeddings = None
        self.faiss = None
        self.faiss_indices = {}
        self.faiss_row_maps = {}
        self.backend = "not_loaded"
        self.manifest = manifest

        return manifest

    def load_corpus(self):
        if self.metadata is not None and self.embeddings is not None:
            return

        self.ensure_corpus()
        self.metadata = pd.read_csv(self.metadata_path, low_memory=False)
        self.embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r",
        )
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        # Prefer persistent FAISS indices. If FAISS is unavailable on the client,
        # use the same saved normalized embeddings with NumPy dot product.
        try:
            import faiss
            indices = {}
            row_maps = {}
            for domain in ["ecommerce", "hotel", "mobile_app", "restaurant"]:
                index_path = self.faiss_dir / f"{domain}.faiss"
                rows_path = self.faiss_dir / f"{domain}_rows.npy"
                if not (index_path.exists() and rows_path.exists()):
                    raise FileNotFoundError(
                        f"Persistent FAISS files missing for {domain}"
                    )
                indices[domain] = faiss.read_index(str(index_path))
                row_maps[domain] = np.load(rows_path)

            self.faiss = faiss
            self.faiss_indices = indices
            self.faiss_row_maps = row_maps
            self.backend = "faiss_ip"
        except Exception:
            self.backend = "numpy_cosine"

    def _encode_queries(self, texts: list[str]) -> np.ndarray:
        self.load_model()
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=min(self.batch_size, max(1, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _issue_context_text(self, domain: str, issue: str) -> str:
        issue = str(issue or "").strip()
        if not issue or issue in {"no_issue", "none", "not_available", "nan"}:
            return ""

        try:
            from agents.domain_issue_taxonomy import get_domain_taxonomy
            taxonomy = get_domain_taxonomy(domain)
            info = taxonomy.get(issue, {})
            label = self.safe_text(info.get("label", issue.replace("_", " ")))
            descriptions = info.get("descriptions", []) or []
            description = self.safe_text(descriptions[0]) if descriptions else ""
            return f"{label}. {description}".strip()
        except Exception:
            return issue.replace("_", " ")

    def _issue_vector(self, domain: str, issue: str) -> np.ndarray | None:
        context = self._issue_context_text(domain, issue)
        if not context:
            return None
        key = (domain, issue)
        if key not in self._issue_vector_cache:
            self._issue_vector_cache[key] = self._encode_queries([context])[0]
        return self._issue_vector_cache[key]

    def _candidate_rows(self, domain: str, query_vector: np.ndarray) -> np.ndarray:
        domain = self.normalize_domain(domain)

        if self.backend == "faiss_ip":
            index = self.faiss_indices.get(domain)
            row_map = self.faiss_row_maps.get(domain)
            if index is None or row_map is None or index.ntotal == 0:
                return np.empty(0, dtype=np.int64)

            fetch_k = min(
                int(index.ntotal),
                max(self.candidate_pool, self.top_k * 20),
            )
            _, local_indices = index.search(
                np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32)),
                fetch_k,
            )
            local_indices = local_indices[0]
            local_indices = local_indices[local_indices >= 0]
            return row_map[local_indices].astype(np.int64)

        domain_rows = np.flatnonzero(
            self.metadata["domain"].astype(str).to_numpy() == domain
        ).astype(np.int64)
        if len(domain_rows) == 0:
            return domain_rows

        domain_embeddings = np.asarray(
            self.embeddings[domain_rows],
            dtype=np.float32,
        )
        scores = domain_embeddings @ query_vector.astype(np.float32)
        fetch_k = min(
            len(domain_rows),
            max(self.candidate_pool, self.top_k * 20),
        )
        if fetch_k >= len(domain_rows):
            order = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -fetch_k)[-fetch_k:]
            order = part[np.argsort(scores[part])[::-1]]
        return domain_rows[order].astype(np.int64)

    def retrieve_for_row(
        self,
        row: pd.Series | dict,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        self.load_corpus()

        top_k = max(1, int(top_k or self.top_k))
        domain = self.normalize_domain(row.get("domain", ""))
        query_text = self.safe_text(
            row.get("_rag_query_text")
            or row.get("clean_review")
            or row.get("review_text")
            or row.get("content")
            or ""
        )
        query_hash = self.text_hash(query_text)
        query_review_id = self.safe_text(
            row.get("review_id") or row.get("reviewId") or ""
        )
        query_entity_id = self.safe_text(
            row.get("entity_id") or row.get("appId") or ""
        )
        issue = self.safe_text(row.get("primary_issue", ""))

        candidates = self._candidate_rows(domain, query_vector)
        if len(candidates) == 0:
            return []

        candidate_embeddings = np.asarray(
            self.embeddings[candidates],
            dtype=np.float32,
        )
        review_scores = candidate_embeddings @ query_vector.astype(np.float32)

        issue_vector = self._issue_vector(domain, issue)
        if issue_vector is not None and self.issue_weight > 0:
            issue_scores = candidate_embeddings @ issue_vector.astype(np.float32)
            final_scores = (
                (1.0 - self.issue_weight) * review_scores
                + self.issue_weight * issue_scores
            )
        else:
            final_scores = review_scores

        ranked = np.argsort(final_scores)[::-1]

        results: list[dict[str, Any]] = []
        returned_hashes: set[str] = set()

        for local_pos in ranked:
            global_row = int(candidates[int(local_pos)])
            meta = self.metadata.iloc[global_row]

            candidate_hash = self.safe_text(meta.get("text_hash", ""))
            candidate_review_id = self.safe_text(meta.get("review_id", ""))
            candidate_entity_id = self.safe_text(meta.get("entity_id", ""))

            # Robust self/duplicate exclusion.
            if candidate_hash and candidate_hash == query_hash:
                continue
            if (
                query_review_id
                and candidate_review_id
                and candidate_review_id == query_review_id
            ):
                continue
            if (
                self.exclude_same_entity
                and query_entity_id
                and candidate_entity_id
                and candidate_entity_id == query_entity_id
            ):
                continue
            if candidate_hash in returned_hashes:
                continue

            similarity = float(final_scores[int(local_pos)])
            if similarity < self.min_similarity:
                continue

            returned_hashes.add(candidate_hash)
            evidence = {
                "review_id": candidate_review_id,
                "domain": self.safe_text(meta.get("domain", "")),
                "entity_id": candidate_entity_id,
                "entity_name": self.safe_text(meta.get("entity_name", "")),
                "review_text": self.safe_text(meta.get("review_text", "")),
                "rating": meta.get("rating", ""),
                "source": self.safe_text(meta.get("source", "")),
                "similarity": round(similarity, 4),
                "review_similarity": round(
                    float(review_scores[int(local_pos)]),
                    4,
                ),
                "issue": issue,
                "corpus_row": global_row,
                "text_hash": candidate_hash,
            }
            results.append(evidence)

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _empty_rag_columns(out: pd.DataFrame) -> pd.DataFrame:
        out["rag_evidence_text"] = ""
        out["rag_similarity_score"] = 0.0
        out["rag_top_evidence_text"] = ""
        out["rag_top_evidence_similarity"] = 0.0
        out["rag_top_evidence_review_id"] = ""
        out["rag_top_evidence_entity_name"] = ""
        out["rag_top_evidence_source"] = ""
        out["rag_top_k_json"] = "[]"
        out["rag_evidence_count"] = 0
        out["rag_evidence_available"] = False
        return out

    def process(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_review",
    ) -> pd.DataFrame:
        out = df.copy()
        self.load_corpus()

        if text_column not in out.columns:
            if "review_text" in out.columns:
                text_column = "review_text"
            elif "content" in out.columns:
                text_column = "content"
            else:
                raise ValueError("No review text column found for RAG retrieval.")

        query_texts = out[text_column].apply(self.safe_text).tolist()
        query_vectors = self._encode_queries(query_texts)

        if len(query_vectors) != len(out):
            raise RuntimeError("RAG query embedding count does not match dataframe rows.")

        out = self._empty_rag_columns(out)
        self.last_issue_evidence = {}

        evidence_texts = []
        evidence_scores = []
        evidence_ids = []
        evidence_entities = []
        evidence_sources = []
        evidence_json = []
        evidence_counts = []
        evidence_available = []

        for (_, row), query_vector in zip(out.iterrows(), query_vectors):
            row_dict = row.to_dict()
            row_dict["_rag_query_text"] = row.get(text_column, "")
            hits = self.retrieve_for_row(
                row_dict,
                query_vector=query_vector,
                top_k=self.top_k,
            )

            if hits:
                top = hits[0]
                evidence_texts.append(top["review_text"])
                evidence_scores.append(top["similarity"])
                evidence_ids.append(top["review_id"])
                evidence_entities.append(top["entity_name"])
                evidence_sources.append(top["source"])
                evidence_json.append(
                    json.dumps(hits, ensure_ascii=False)
                )
                evidence_counts.append(len(hits))
                evidence_available.append(True)
            else:
                evidence_texts.append("")
                evidence_scores.append(0.0)
                evidence_ids.append("")
                evidence_entities.append("")
                evidence_sources.append("")
                evidence_json.append("[]")
                evidence_counts.append(0)
                evidence_available.append(False)

            issue = self.safe_text(row.get("primary_issue", ""))
            if issue and issue not in {"no_issue", "none", "not_available", "nan"}:
                bucket = self.last_issue_evidence.setdefault(issue, [])
                seen = {x.get("text_hash", "") for x in bucket}
                for item in hits:
                    if item.get("text_hash", "") not in seen:
                        bucket.append(item)
                        seen.add(item.get("text_hash", ""))

        out["rag_evidence_text"] = evidence_texts
        out["rag_similarity_score"] = evidence_scores
        out["rag_top_evidence_text"] = evidence_texts
        out["rag_top_evidence_similarity"] = evidence_scores
        out["rag_top_evidence_review_id"] = evidence_ids
        out["rag_top_evidence_entity_name"] = evidence_entities
        out["rag_top_evidence_source"] = evidence_sources
        out["rag_top_k_json"] = evidence_json
        out["rag_evidence_count"] = evidence_counts
        out["rag_evidence_available"] = evidence_available
        out["rag_backend"] = self.backend
        out["rag_corpus_source"] = str(self.corpus_path)
        out["rag_corpus_source_rows"] = int(self.manifest.get("source_rows", 0))
        out["rag_corpus_indexed_rows"] = int(
            self.manifest.get("indexed_unique_rows", len(self.metadata))
        )
        out["rag_self_duplicate_exclusion"] = True

        return out

    def retrieve_evidence_for_issues(
        self,
        issue_categories,
        top_k: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for issue in issue_categories or []:
            issue = self.safe_text(issue)
            if not issue:
                continue
            items = self.last_issue_evidence.get(issue, [])
            result[issue] = items[: max(1, int(top_k))]
        return result


# Compatibility aliases used by older imports.
RAGEvidenceAgent = RAGEvidenceRetrievalAgent
RAGAgent = RAGEvidenceRetrievalAgent

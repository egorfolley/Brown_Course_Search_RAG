"""
Hybrid search: FAISS (semantic) + BM25 (lexical), weighted score fusion.

Scores are normalised to [0, 1] independently then combined:
    final_score = alpha * faiss_score + (1 - alpha) * bm25_score

alpha=0.5 by default (equal weight). Increase alpha to favour semantic
matching; decrease to favour keyword matching.

Public API:
    HybridSearch(store, courses)
    HybridSearch.query(text, top_k, alpha, filters) → list[dict]
"""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import Any

from rag.embedder import MODEL_NAME, SentenceTransformer
from rag.vector_store import VectorStore

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; return zeros if all scores equal."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _tokenise(text: str) -> list[str]:
    return text.lower().split()


class HybridSearch:
    def __init__(self, store: VectorStore):
        self.store   = store
        self.courses = store.courses
        # Build BM25 corpus once over all courses
        corpus = [_tokenise(
            f"{c.get('title','')} {c.get('description','')} {c.get('department','')}"
        ) for c in self.courses]
        self.bm25 = BM25Okapi(corpus)

    def query(
        self,
        text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search over all courses.

        Args:
            text:    natural-language query
            top_k:   number of results to return
            alpha:   weight for FAISS score (1-alpha goes to BM25)
            filters: equality filters applied to metadata before ranking
                     e.g. {"department": "Computer Science"}

        Returns:
            list of course dicts with added keys:
                _faiss_score, _bm25_score, _hybrid_score
        """
        # Apply metadata filter to get candidate indices
        if filters:
            candidate_idx = [
                i for i, c in enumerate(self.courses)
                if all(c.get(k, "").lower() == v.lower() for k, v in filters.items())
            ]
        else:
            candidate_idx = list(range(len(self.courses)))

        if not candidate_idx:
            return []

        # --- FAISS scores ---
        qv = _get_model().encode([text], normalize_embeddings=True).astype(np.float32)
        sub_embs = np.stack([self.store.index.reconstruct(i) for i in candidate_idx])
        faiss_raw = (sub_embs @ qv.T).flatten()   # cosine similarity

        # --- BM25 scores ---
        tokens = _tokenise(text)
        all_bm25 = self.bm25.get_scores(tokens)
        bm25_raw = np.array([all_bm25[i] for i in candidate_idx])

        # --- Fuse ---
        faiss_norm = _normalise(faiss_raw)
        bm25_norm  = _normalise(bm25_raw)
        hybrid     = alpha * faiss_norm + (1 - alpha) * bm25_norm

        # Sort and take top-k
        top_local = np.argsort(hybrid)[::-1][:top_k]

        results = []
        for local_id in top_local:
            course = dict(self.courses[candidate_idx[local_id]])
            course["_faiss_score"]  = float(faiss_norm[local_id])
            course["_bm25_score"]   = float(bm25_norm[local_id])
            course["_hybrid_score"] = float(hybrid[local_id])
            results.append(course)

        return results

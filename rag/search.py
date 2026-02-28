"""
Hybrid search: FAISS (semantic) + BM25 (lexical), weighted score fusion.

Scores are normalised to [0, 1] independently then combined:
    final_score = alpha * faiss_score + (1 - alpha) * bm25_score

alpha=0.5 by default (equal weight). Increase alpha to favour semantic
matching; decrease to favour keyword matching.

Also includes exact course-code matching. If a course code is detected in
the query, exact matches are boosted to rank first.

Public API:
    HybridSearch(store, courses)
    HybridSearch.query(text, top_k, alpha, filters) → list[dict]
"""

import re
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


def extract_course_code(query: str) -> str | None:
    """
    Extract a course code from the query (e.g., ENGN0030, AMST2920, CSCI0320).
    
    Supports formats like:
    - ENGN0030 (uppercase letter(s) + digits)
    - ENGN 0030 (with space)
    - engn0030 (lowercase)
    
    Returns the normalized uppercase format or None if not found.
    """
    # Match: 2-4 letters, optional space, 4 digits
    match = re.search(r'([A-Za-z]{2,4})\s*(\d{4})', query)
    if match:
        return (match.group(1) + match.group(2)).upper()
    return None


def normalize_course_code(code: str) -> str:
    """Normalize a course code to uppercase, removing spaces."""
    return code.replace(" ", "").upper()


class HybridSearch:
    def __init__(self, store: VectorStore):
        self.store   = store
        self.courses = store.courses
        
        # Build BM25 corpus with all course fields (not just title/description)
        corpus = []
        for c in self.courses:
            text_parts = []
            if c.get('course_code'):
                text_parts.append(c['course_code'])
            if c.get('title'):
                text_parts.append(c['title'])
            if c.get('description'):
                text_parts.append(c['description'])
            if c.get('department'):
                text_parts.append(c['department'])
            if c.get('instructor'):
                text_parts.append(c['instructor'])
            if c.get('meeting_times'):
                text_parts.append(c['meeting_times'])
            if c.get('prerequisites'):
                text_parts.append(c['prerequisites'])
            if c.get('source'):
                text_parts.append(c['source'])
            
            corpus.append(_tokenise(" ".join(text_parts)))
        
        self.bm25 = BM25Okapi(corpus)

    def query(
        self,
        text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Hybrid search over all courses with exact course-code matching.

        Args:
            text:    natural-language query
            top_k:   number of results to return
            alpha:   weight for FAISS score (1-alpha goes to BM25)
            filters: equality filters applied to metadata before ranking
                     e.g. {"department": "Computer Science"}

        Returns:
            tuple of (
                list of course dicts with added keys:
                    _faiss_score, _bm25_score, _hybrid_score, _exact_match
                detected course code (or None)
            )
        """
        # Detect course code in query
        detected_code = extract_course_code(text)
        
        # Apply metadata filter to get candidate indices
        if filters:
            candidate_idx = [
                i for i, c in enumerate(self.courses)
                if all(c.get(k, "").lower() == v.lower() for k, v in filters.items())
            ]
        else:
            candidate_idx = list(range(len(self.courses)))

        if not candidate_idx:
            return [], detected_code

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

        # --- Exact course-code matching (if detected) ---
        exact_match_idx = None
        if detected_code:
            for local_id, global_id in enumerate(candidate_idx):
                course_code = normalize_course_code(
                    self.courses[global_id].get('course_code', '')
                )
                if course_code == detected_code:
                    # Strong boost to hybrid score for exact match
                    exact_match_idx = local_id
                    hybrid[local_id] = 1.0  # Maximum score
                    break

        # Sort and take top-k
        top_local = np.argsort(hybrid)[::-1][:top_k]

        results = []
        for local_id in top_local:
            global_id = candidate_idx[local_id]
            course = dict(self.courses[global_id])
            course["_faiss_score"]  = float(faiss_norm[local_id])
            course["_bm25_score"]   = float(bm25_norm[local_id])
            course["_hybrid_score"] = float(hybrid[local_id])
            course["_exact_match"]  = (local_id == exact_match_idx)
            results.append(course)

        return results, detected_code


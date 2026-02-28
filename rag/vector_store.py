"""
FAISS vector store.

Builds a IndexFlatIP (inner-product) index over L2-normalised embeddings,
which is equivalent to cosine similarity search.

Public API:
    build(embeddings, courses)  → VectorStore
    VectorStore.search(query_emb, top_k, filters) → list[dict]
    VectorStore.save() / VectorStore.load()
"""

import json
import numpy as np
import faiss
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data"
INDEX_FILE  = DATA_DIR / "faiss.index"
META_FILE   = DATA_DIR / "metadata.json"


class VectorStore:
    def __init__(self, index: faiss.Index, courses: list[dict]):
        self.index   = index
        self.courses = courses   # parallel to index rows

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Return up to top_k courses ordered by cosine similarity.

        filters: optional dict of field→value to pre-filter candidates.
            e.g. {"department": "Computer Science"}
            Only equality matching; multiple filters are ANDed.
        """
        # Build candidate set (filtered or all)
        if filters:
            candidate_idx = [
                i for i, c in enumerate(self.courses)
                if all(c.get(k, "").lower() == v.lower() for k, v in filters.items())
            ]
        else:
            candidate_idx = list(range(len(self.courses)))

        if not candidate_idx:
            return []

        # Extract sub-matrix and search
        sub_embs = np.stack([self.index.reconstruct(i) for i in candidate_idx])
        sub_index = faiss.IndexFlatIP(sub_embs.shape[1])
        sub_index.add(sub_embs)

        k = min(top_k, len(candidate_idx))
        qv = query_emb.reshape(1, -1).astype(np.float32)
        scores, local_ids = sub_index.search(qv, k)

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            if local_id == -1:
                continue
            course = dict(self.courses[candidate_idx[local_id]])
            course["_faiss_score"] = float(score)
            results.append(course)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(self.courses, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls) -> "VectorStore":
        index   = faiss.read_index(str(INDEX_FILE))
        courses = json.loads(META_FILE.read_text(encoding="utf-8"))
        return cls(index, courses)


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------

def build(embeddings: np.ndarray, courses: list[dict]) -> VectorStore:
    """Build an IndexFlatIP from a (N, D) float32 embedding matrix."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return VectorStore(index, courses)

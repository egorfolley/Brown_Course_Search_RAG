"""
Course embedding module.

Encodes each course as a single text string:
    "{title}. {description}. {department}"

Uses sentence-transformers all-MiniLM-L6-v2 (384-dim, runs locally).
Persists:
    data/embeddings.npy  — float32 array, shape (N, 384)
    data/metadata.json   — list of course dicts (parallel to embeddings)

These two files are the interface consumed by vector_store.py.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR   = Path(__file__).parent.parent / "data"
COURSES    = DATA_DIR / "courses.json"
EMB_FILE   = DATA_DIR / "embeddings.npy"
META_FILE  = DATA_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def course_to_text(course: dict) -> str:
    """Concatenate the fields most useful for semantic retrieval."""
    parts = [
        course.get("title", ""),
        course.get("description", ""),
        course.get("department", ""),
    ]
    return ". ".join(p for p in parts if p).strip()


def load_courses(path: Path = COURSES) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"courses.json not found at {path}. Run the ETL pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


def embed(courses: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Return a float32 (N, 384) array of L2-normalised embeddings."""
    model = SentenceTransformer(model_name)
    texts = [course_to_text(c) for c in courses]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
    return embeddings.astype(np.float32)


def save(embeddings: np.ndarray, courses: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    np.save(EMB_FILE, embeddings)
    META_FILE.write_text(json.dumps(courses, ensure_ascii=False), encoding="utf-8")


def load() -> tuple[np.ndarray, list[dict]]:
    """Load persisted embeddings and metadata from disk."""
    embeddings = np.load(EMB_FILE)
    courses = json.loads(META_FILE.read_text(encoding="utf-8"))
    return embeddings, courses


def run() -> tuple[np.ndarray, list[dict]]:
    """Full build: load courses → embed → save → return (embeddings, courses)."""
    courses = load_courses()
    print(f"Embedding {len(courses)} courses with {MODEL_NAME}…")
    embeddings = embed(courses)
    save(embeddings, courses)
    print(f"Saved → {EMB_FILE}  {embeddings.shape}")
    return embeddings, courses


if __name__ == "__main__":
    run()

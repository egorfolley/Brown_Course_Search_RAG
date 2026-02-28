"""
FastAPI application — single entry point for the full pipeline.

Run as a script to orchestrate all steps then serve:
    python api/app.py

Or run as a module if data is already built:
    uvicorn api.app:app --reload

Pipeline steps (each skipped if output already exists):
    1. Scrape bulletin.brown.edu  → data/bulletin_courses.json
    2. Scrape cab.brown.edu       → data/cab_courses.json
    3. Merge via ETL pipeline     → data/courses.json
    4. Build embeddings + index   → data/faiss.index + data/metadata.json

Endpoint:
    POST /query
        body:  {"q": "...", "department": "..."}   (department is optional)
        returns: {"answer": str, "courses": [...]}

Logs each query and wall-clock response time to stdout.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from rag.vector_store import VectorStore
from rag.search import HybridSearch

load_dotenv()

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR      = Path(__file__).parent.parent / "data"
BULLETIN_FILE = DATA_DIR / "bulletin_courses.json"
CAB_FILE      = DATA_DIR / "cab_courses.json"
COURSES_FILE  = DATA_DIR / "courses.json"
INDEX_FILE    = DATA_DIR / "faiss.index"
META_FILE     = DATA_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _ensure_data() -> None:
    """Run any missing pipeline steps before serving."""

    if not BULLETIN_FILE.exists():
        log.info("bulletin_courses.json not found — scraping bulletin.brown.edu…")
        from etl.scrape_bulletin import main as scrape_bulletin
        scrape_bulletin()
    else:
        log.info("bulletin_courses.json already exists, skipping Bulletin scrape.")

    if not CAB_FILE.exists():
        log.info("cab_courses.json not found — scraping cab.brown.edu (launches Chromium)…")
        from etl.scrape_cab import main as scrape_cab
        scrape_cab()
    else:
        log.info("cab_courses.json already exists, skipping CAB scrape.")

    if not COURSES_FILE.exists():
        log.info("courses.json not found — running ETL merge pipeline…")
        from etl.pipeline import run as run_pipeline
        run_pipeline()
    else:
        log.info("courses.json already exists, skipping ETL merge.")

    if not INDEX_FILE.exists() or not META_FILE.exists():
        log.info("FAISS index not found — building embeddings…")
        from rag.embedder import run as run_embedder
        from rag.vector_store import build as build_store
        embeddings, courses = run_embedder()
        store = build_store(embeddings, courses)
        store.save()
        log.info("FAISS index saved.")
    else:
        log.info("FAISS index already exists, skipping embedding step.")


# ---------------------------------------------------------------------------
# App + startup state
# ---------------------------------------------------------------------------

app = FastAPI(title="Brown Course Search")

_search: HybridSearch | None = None
_openai: OpenAI | None = None


@app.on_event("startup")
def startup() -> None:
    global _search, _openai

    log.info("Loading FAISS index and metadata…")
    store = VectorStore.load()
    log.info("  %d courses loaded.", len(store.courses))

    log.info("Building BM25 corpus…")
    _search = HybridSearch(store)
    log.info("  Ready.")

    _openai = OpenAI()   # reads OPENAI_API_KEY from env


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    q: str
    department: str | None = None


class CourseResult(BaseModel):
    code: str
    title: str
    department: str
    similarity: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    courses: list[CourseResult]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful Brown University course advisor. "
    "Answer the student's question using only the courses listed below. "
    "Be concise (2-4 sentences). If no course matches, say so plainly."
)


def _build_context(courses: list[dict]) -> str:
    lines = []
    for c in courses:
        lines.append(
            f"- {c['course_code']}: {c['title']} ({c.get('department', '')})\n"
            f"  {c.get('description', '')[:200]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    t0 = time.perf_counter()

    filters = {"department": req.department} if req.department else None
    results = _search.query(req.q, top_k=5, filters=filters)

    if not results:
        elapsed = time.perf_counter() - t0
        log.info("query=%r  dept=%r  hits=0  %.2fs", req.q, req.department, elapsed)
        return QueryResponse(answer="No matching courses found.", courses=[])

    context = _build_context(results)
    completion = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {req.q}\n\nCourses:\n{context}"},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    answer = completion.choices[0].message.content.strip()

    courses = [
        CourseResult(
            code=c["course_code"],
            title=c.get("title", ""),
            department=c.get("department", ""),
            similarity=round(c["_hybrid_score"], 4),
            source=c.get("source", ""),
        )
        for c in results
    ]

    elapsed = time.perf_counter() - t0
    log.info("query=%r  dept=%r  hits=%d  %.2fs", req.q, req.department, len(courses), elapsed)

    return QueryResponse(answer=answer, courses=courses)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _ensure_data()
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

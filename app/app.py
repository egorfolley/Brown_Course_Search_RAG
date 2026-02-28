"""
FastAPI application — single entry point for the full pipeline.

Run as a script to orchestrate all steps then serve:
    python app/app.py

Or run as a module if data is already built:
    uvicorn app.app:app --reload

Pipeline steps (each skipped if output already exists):
    1. Scrape bulletin.brown.edu  → data/bulletin_courses.json
    2. Scrape cab.brown.edu       → data/cab_courses.json
    3. Merge via ETL pipeline     → data/courses.json
    4. Build embeddings + index   → data/faiss.index + data/metadata.json

Endpoint:
    POST /query
        body:  {"q": "...", "department": "..."}   (department is optional)
        returns: {"answer": str, "courses": [...]}

Logs each query and wall-clock response time to stdout and logs/app.log
(rotating, 5 MB max, 3 backups).
"""

import json
import asyncio
import logging
import logging.handlers
import sys
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

# Ensure project root is on sys.path when running as a script (python app/app.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import VectorStore
from rag.search import HybridSearch

load_dotenv()

LOG_DIR  = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)

    # Rotate at 5 MB, keep 3 backups
    rotating = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    rotating.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(stream)
    root.addHandler(rotating)

_setup_logging()
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR      = Path(__file__).parent.parent / "data"
BULLETIN_FILE = DATA_DIR / "bulletin_courses.json"
CAB_FILE      = DATA_DIR / "cab_courses.json"
COURSES_FILE  = DATA_DIR / "courses.json"
INDEX_FILE    = DATA_DIR / "faiss.index"
META_FILE     = DATA_DIR / "metadata.json"

Course = dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _scrape_and_save(scrape_fn: Callable[[], list[Course]], output_path: Path) -> None:
    """Call scrape_fn(), log progress, and persist the result as JSON."""
    log.info("  Running %s…", scrape_fn.__name__)
    courses: list[Course] = scrape_fn()
    DATA_DIR.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  Saved %d courses → %s", len(courses), output_path.name)


def _ensure_data() -> None:
    """Run any missing pipeline steps before serving."""

    # Step 1: Bulletin scrape
    if not BULLETIN_FILE.exists():
        log.info("[1/4] bulletin_courses.json missing — scraping bulletin.brown.edu…")
        from etl.scrape_bulletin import scrape_all as _scrape_bulletin
        _scrape_and_save(_scrape_bulletin, BULLETIN_FILE)
    else:
        log.info("[1/4] bulletin_courses.json exists — skipping.")

    # Step 2: CAB scrape
    if not CAB_FILE.exists():
        log.info("[2/4] cab_courses.json missing — scraping cab.brown.edu (launches Chromium)…")
        from etl.scrape_cab import scrape_all as _scrape_cab
        _scrape_and_save(_scrape_cab, CAB_FILE)
    else:
        log.info("[2/4] cab_courses.json exists — skipping.")

    # Step 3: ETL merge
    if not COURSES_FILE.exists():
        log.info("[3/4] courses.json missing — running ETL merge pipeline…")
        from etl.pipeline import run as run_pipeline
        merged: list[Course] = run_pipeline()
        log.info("  Merged %d courses → courses.json", len(merged))
    else:
        log.info("[3/4] courses.json exists — skipping.")

    # Step 4: Embeddings + FAISS index
    if not INDEX_FILE.exists() or not META_FILE.exists():
        log.info("[4/4] FAISS index missing — building embeddings…")
        from rag.embedder import run as run_embedder
        from rag.vector_store import build as build_store
        embeddings, courses = run_embedder()
        log.info("  Encoded %d courses — building FAISS index…", len(courses))
        store = build_store(embeddings, courses)
        store.save()
        log.info("  FAISS index saved → faiss.index + metadata.json")
    else:
        log.info("[4/4] FAISS index exists — skipping.")


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------

_search: HybridSearch | None = None
_openai: OpenAI | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _search, _openai

    log.info("Loading FAISS index and metadata…")
    store = VectorStore.load()
    log.info("  %d courses loaded.", len(store.courses))

    log.info("Building BM25 corpus…")
    _search = HybridSearch(store)
    log.info("  BM25 ready.")

    _openai = OpenAI()  # reads OPENAI_API_KEY from env
    log.info("  OpenAI client ready.")

    yield  # server runs here


app = FastAPI(title="Brown Course Search", lifespan=lifespan)


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
    instructor: str
    meeting_times: str
    similarity: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    courses: list[CourseResult]
    detected_code: str | None = None  # For debugging: detected course code from query


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful Brown University course advisor. "
    "Answer the student's question using ONLY the courses listed below in the context. "
    "If the user asks about a specific course code (e.g., 'Who teaches ENGN0030?'), "
    "and that course appears in the retrieved results, state its instructor and meeting times directly. "
    "If instructor or meeting times are missing from the data, explicitly state they are not available. "
    "Do not claim a course is missing if it appears in the retrieved context. "
    "Be concise (2-4 sentences). If no course matches the query, say so plainly."
)


def _build_context(courses: list[Course]) -> str:
    """Build a structured context block for the LLM with all course metadata."""
    lines = []
    for c in courses:
        code = c.get('course_code', 'N/A')
        title = c.get('title', 'N/A')
        department = c.get('department', 'N/A')
        instructor = c.get('instructor', '')
        meeting_times = c.get('meeting_times', '')
        prerequisites = c.get('prerequisites', '')
        source = c.get('source', 'N/A')
        description = c.get('description', '')
        
        # Build structured block
        block = f"Course Code: {code}\n"
        block += f"Title: {title}\n"
        block += f"Department: {department}\n"
        
        if instructor:
            block += f"Instructor: {instructor}\n"
        else:
            block += "Instructor: Not available\n"
        
        if meeting_times:
            block += f"Meeting Times: {meeting_times}\n"
        else:
            block += "Meeting Times: Not available\n"
        
        if prerequisites:
            block += f"Prerequisites: {prerequisites}\n"
        
        block += f"Source: {source}\n"
        
        if description:
            block += f"Description: {description[:300]}"
        
        lines.append(block)
    
    return "\n---\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    t0 = time.perf_counter()

    filters = {"department": req.department} if req.department else None
    log.info("Searching: q=%r  dept=%r", req.q, req.department)

    assert _search is not None, "Search not initialised"
    results, detected_code = _search.query(req.q, top_k=5, filters=filters)

    if detected_code:
        log.info("  Detected course code: %s", detected_code)

    if not results:
        elapsed = time.perf_counter() - t0
        log.info("query=%r  dept=%r  hits=0  %.2fs", req.q, req.department, elapsed)
        return QueryResponse(
            answer="No matching courses found.",
            courses=[],
            detected_code=detected_code
        )

    log.info("  %d results — calling OpenAI…", len(results))
    context = _build_context(results)

    assert _openai is not None, "OpenAI client not initialised"
    completion = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {req.q}\n\nCourses:\n{context}"},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    answer = (completion.choices[0].message.content or "").strip()

    courses = [
        CourseResult(
            code=str(c["course_code"]),
            title=str(c.get("title", "")),
            department=str(c.get("department", "")),
            instructor=str(c.get("instructor", "")),
            meeting_times=str(c.get("meeting_times", "")),
            similarity=round(float(c["_hybrid_score"]), 4),
            source=str(c.get("source", "")),
        )
        for c in results
    ]

    elapsed = time.perf_counter() - t0
    log.info("query=%r  dept=%r  hits=%d  %.2fs", req.q, req.department, len(courses), elapsed)

    return QueryResponse(
        answer=answer,
        courses=courses,
        detected_code=detected_code
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _launch_server() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
        return

    log.warning(
        "Detected an existing asyncio event loop; serving with create_task() instead of asyncio.run()."
    )
    asyncio.create_task(server.serve())


if __name__ == "__main__":
    log.info("=== Brown Course Search — starting up ===")
    _ensure_data()
    log.info("=== All data ready — launching server on http://0.0.0.0:8000 ===")
    _launch_server()

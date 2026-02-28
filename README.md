# Brown Course Search: RAG-powered academic course discovery

A RAG-powered academic course search tool that scrapes Brown University course data from 2 sources, and stores in a vectorDB (for RAG applications)

## Technical Setup

### Stack

| Layer           | Choice                                       | Rationale                                                         |
| --------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| Backend API     | FastAPI + Uvicorn                            | Async-native, fast, auto-generates OpenAPI docs                   |
| Frontend        | Streamlit                                    | Rapid prototyping; can migrate to React later                     |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | Strong quality/speed tradeoff; 384-dim, runs locally              |
| Vector store    | FAISS (CPU)                                  | In-process, no infra needed; easy to persist to disk              |
| Lexical search  | BM25 (rank-bm25)                             | Hybrid retrieval: catches exact keyword matches embeddings miss   |
| Scraping (CAB)  | Playwright + BeautifulSoup4                  | CAB sits behind AWS WAF; Playwright bypasses the JS challenge     |
| Scraping (Bul.) | Requests + BeautifulSoup4                    | bulletin.brown.edu is fully static HTML — no JS rendering needed |
| LLM generation  | OpenAI API                                   | Handles final answer synthesis from retrieved context             |
| Data validation | Pydantic v2                                  | FastAPI-native; validates course schema and API payloads          |

### Retrieval Strategy

Hybrid search: BM25 score + cosine similarity from FAISS are combined (reciprocal rank fusion or weighted sum) before passing top-k chunks to the LLM for answer generation.

**Selection description:**

Hybrid search matters because semantic search and keyword search fail in different ways:

* **FAISS** is great at understanding meaning ("machine learning courses" matches "intro to neural networks") but can miss exact terms. If a student searches "CSCI0320", pure semantic search might rank it lower than conceptually similar courses
* **BM25** is great at exact matches ("CSCI0320", "Friday", "3 PM") but doesn't understand synonyms or intent. "Philosophy about the nature of reality" won't match "metaphysics" well.

Combining both covers both failure modes. A course that scores high on both semantic relevance AND keyword match is almost certainly the right result. The fusion method (weighted sum or reciprocal rank) just determines how you blend the two ranked lists into one final ranking before sending top-k to the LLM.

For this assignment specifically, look at the example queries: query 1 needs exact code matching (BM25), query 2 needs semantic understanding (FAISS), query 4 needs both ("Fridays after 3 pm" is keyword, "machine learning" is semantic). Hybrid search handles all four naturally.

### Prerequisites

- Python 3.11+
- An OpenAI API key (set in `.env` as `OPENAI_API_KEY`)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium      # for JS-rendered scraping
```

### Environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Running

All commands are run from the project root with the virtualenv active.

**Step 1 — Start the API** *(builds all data automatically on first run)*

```bash
python app/app.py
# Scrapes Bulletin (~5 min) and CAB (~15 min, launches Chromium) if data is missing
# Builds embeddings + FAISS index if missing
# Starts API on http://0.0.0.0:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

Each pipeline step is skipped if its output file already exists. To force a fresh scrape, delete the relevant files from `data/` before running.

**Step 2 — Start the frontend** *(separate terminal)*

```bash
streamlit run frontend/ui.py
```

### API reference

`POST /query`

```json
// request
{ "q": "machine learning courses on Fridays", "department": "Computer Science" }

// response
{
  "answer": "The best match is CSCI 1951A ...",
  "courses": [
    { "code": "CSCI1951A", "title": "Data Science", "department": "Computer Science",
      "similarity": 0.87, "source": "Bulletin" }
  ]
}
```

`department` is optional. Omit it to search across all departments.

Logs printed to stdout per request:

```
INFO  query='machine learning on Fridays'  dept='Computer Science'  hits=5  0.43s
```

---

## Project Structure

```
Brown_course_search_RAG/
├── etl/                        # Data ingestion & normalization
│   ├── scrape_cab.py           # Scrapes Brown's Course Announcement Bulletin
│   ├── scrape_bulletin.py      # Scrapes Courses@Brown for richer descriptions
│   └── pipeline.py             # Orchestrates scrapers → normalizes → writes courses.json
├── rag/                        # Retrieval-augmented generation core
│   ├── embedder.py             # Encodes courses and builds/persists FAISS index
│   ├── vector_store.py         # FAISS index wrapper (load + nearest-neighbour search)
│   ├── search.py               # Hybrid search: BM25 + FAISS → reciprocal rank fusion
│   └── generator.py            # Formats prompt and calls OpenAI to synthesize answer
├── api/
│   └── app.py                  # FastAPI app — POST /query endpoint
├── frontend/
│   └── ui.py                   # Streamlit UI — search bar, filters, results display
├── data/                       # Runtime-generated artifacts (git-ignored except .gitkeep)
│   ├── courses.json            # Normalized course records (ETL output)
│   ├── faiss.index             # Persisted FAISS flat index (embedder output)
│   └── metadata.json           # Parallel course metadata list for FAISS results
├── playground.ipynb            # Exploratory notebook for testing retrieval & generation
├── requirements.txt
└── README.md
```

Each package directory contains an `__init__.py`. The `data/` directory is populated at runtime and its generated files are git-ignored; only `.gitkeep` is tracked to preserve the directory.

---

## Architecture

```
┌─────────────────────────────────────┐
│               ETL                   │
│  CAB scraper + Courses@Brown scraper│
│        → normalizer                 │
│        → data/courses.json          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│             Embedder                │
│  sentence-transformers              │
│  (all-MiniLM-L6-v2, 384-dim)       │
│        → data/faiss.index           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          Hybrid Search              │
│  FAISS (semantic) + BM25 (lexical) │
│  + metadata filters (dept, time…)  │
│  → reciprocal rank fusion → top-k  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│           FastAPI Backend           │
│  POST /query                        │
│  → top-k context + LLM generation  │
│  → JSON response                    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│         Streamlit Frontend          │
│  Search bar + filters               │
│  → displays ranked results + answer │
└─────────────────────────────────────┘
```

**ETL** — Two scrapers feed a merge pipeline. `scrape_bulletin.py` targets `bulletin.brown.edu` with plain Requests+BS4 (fully static HTML) and captures course code, title, description, prerequisites, instructor, and meeting times from the offerings table. `scrape_cab.py` targets `cab.brown.edu` via Playwright (required to bypass AWS WAF) and pulls catalog data from CourseLeaf's Ribbit API. `pipeline.py` merges both on normalized course code: Bulletin is the base record (richer schedule data); CAB back-fills any fields left empty. The `source` field on each record reflects provenance: `"Bulletin"`, `"CAB"`, or `"CAB+Bulletin"`.

**Embedder** — Runs once (or on data refresh) to encode every course description with `all-MiniLM-L6-v2` and build a FAISS flat index. The index and a parallel metadata list are persisted to disk so the API loads them at startup without re-embedding.

**Hybrid Search** — At query time, BM25 scores the corpus for lexical relevance while FAISS scores against the query embedding for semantic relevance. The two ranked lists are fused (reciprocal rank fusion) and metadata filters (department, time slot, credit hours) are applied before the top-k results are selected.

**API** — `POST /query` accepts `{"q": str, "department": str | None}`. On startup, FastAPI loads the FAISS index and builds the BM25 corpus once so every request is fast. Each request runs hybrid search (top-5), assembles a context string from those results, calls `gpt-4o-mini` for a 2-4 sentence answer, and returns `{"answer": str, "courses": [{code, title, department, similarity, source}]}`. Query text, department filter, hit count, and wall-clock time are logged to stdout.

**Frontend** — Streamlit provides a search bar and sidebar filters that call the FastAPI backend. Results are rendered as an expandable course list alongside the LLM-generated summary.

---

## TO DO

* [X] Technical setup - choose embedding model, vector store, LLM, scraping tools/ETL
  * [X] FastAPI + Streamlit for simplicity
  * [ ] Further UI modification to React if needed
* [X] ClaudeCode prompts for initial development
* [X] Design architecture
* [X] Scaffolding - project structure
* [X] CAB scraper
* [X] Bulletin scraper
* [X] ETL pipeline and storage
* [X] RAG pipeline
* [X] Test solutions in Playground notebook
* [X] Backend API
* [ ] UI
* [ ] Polish - deployment + docs + report
* [ ] Loom video

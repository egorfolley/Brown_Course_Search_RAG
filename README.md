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
| Scraping        | BeautifulSoup4 + Requests + Playwright       | BS4/Requests for static pages; Playwright for JS-rendered content |
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

### Running

```bash
# Backend (from project root)
uvicorn app.main:app --reload

# Frontend (separate terminal)
streamlit run frontend/app.py
```

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

**ETL** — Two scrapers (one for CAB, one for Courses@Brown) extract course data and a normalizer reconciles the schemas into a single `data/courses.json`. Each course record includes code, title, description, instructor, schedule, and department.

**Embedder** — Runs once (or on data refresh) to encode every course description with `all-MiniLM-L6-v2` and build a FAISS flat index. The index and a parallel metadata list are persisted to disk so the API loads them at startup without re-embedding.

**Hybrid Search** — At query time, BM25 scores the corpus for lexical relevance while FAISS scores against the query embedding for semantic relevance. The two ranked lists are fused (reciprocal rank fusion) and metadata filters (department, time slot, credit hours) are applied before the top-k results are selected.

**API** — A single `POST /query` endpoint in FastAPI accepts a natural-language query plus optional filter parameters, runs hybrid search, and passes the top-k course snippets as context to the OpenAI LLM to synthesize a final answer. Returns both the ranked course list and the generated response.

**Frontend** — Streamlit provides a search bar and sidebar filters that call the FastAPI backend. Results are rendered as an expandable course list alongside the LLM-generated summary.

---

## TO DO

* [X] Technical setup - choose embedding model, vector store, LLM, scraping tools/ETL
  * [X] FastAPI + Streamlit for simplicity
  * [ ] Further UI modification to React if needed
* [X] ClaudeCode prompts for initial development
* [X] Design architecture
* [ ] Scaffolding - project structure
* [ ] CAB scraper
* [ ] ETL pipeline and storage
* [ ] RAG pipeline
* [ ] Test solutions in Playground notebook
* [ ] Backend API
* [ ] UI
* [ ] Polish - deployment + docs + report
* [ ] Loom video

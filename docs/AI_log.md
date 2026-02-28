# AI Development Log

## Overview

**Time Investment:** ~20 minutes planning architecture + ~2 hours iterative development
**Tools Used:**

- Claude Opus: Architecture brainstorming + initial prompt engineering
- GitHub Copilot (Grok Fast): Feature implementation + debugging

**Development Approach:** Modular, step-by-step prompting for each component

---

## Development Steps

### 1. Technical Setup (5 min)

**Goal:** Define stack and create dependency manifest

**Prompt:**

```
Review the requirements for this project (RAG course search tool with FastAPI backend, 
Streamlit frontend, FAISS vector store). Confirm these dependency choices and create 
a requirements.txt:
- FastAPI + uvicorn
- Streamlit
- sentence-transformers (all-MiniLM-L6-v2)
- faiss-cpu
- beautifulsoup4 + requests + playwright
- rank-bm25
- openai (for LLM generation)
- pydantic

Don't install anything yet, just create the file and add Technical setup 
information to README.md
```

**Outcome:** `requirements.txt` created with all dependencies versioned

---

### 2. Architecture Design (10 min)

**Goal:** Document data flow and system components

**Prompt:**

```
Add an Architecture section in README.md with a short text-based diagram and 
one-paragraph description of each layer:
- ETL (scrapers + normalizer) → data/courses.json
- Embedder (sentence-transformers) → FAISS index
- Search (hybrid: FAISS + BM25, with metadata filtering)
- API (FastAPI /query endpoint)
- Frontend (Streamlit)

Keep it concise. No code.
```

**Outcome:** Clear visual flow diagram + role explanation for each component

---

### 3. Project Scaffolding (5 min)

**Goal:** Set up directory structure with placeholder files

**Prompt:**

```
Create this project structure with empty __init__.py files and placeholder 
Python files with just module docstrings:

brown-course-search/
├── etl/
│   ├── scrape_cab.py
│   ├── scrape_bulletin.py
│   └── pipeline.py
├── rag/
│   ├── embedder.py
│   ├── vector_store.py
│   ├── search.py
│   └── generator.py
├── app/
│   └── app.py
├── frontend/
│   └── ui.py
├── data/
    └── .gitkeep

Update README with project structure.
```

**Outcome:** Clean module organization with stubs ready for implementation

---

### 4. CAB Scraper (20 min)

**Goal:** Extract course data from cab.brown.edu via Ribbit API

**Prompt:**

```
Write etl/scrape_cab.py. Go to https://cab.brown.edu/ and inspect what API 
endpoints the frontend calls. Use Playwright to handle AWS WAF. Return a list 
of dicts with schema: course_code, title, instructor, meeting_times, 
prerequisites, department, description, source="CAB". 

Save to data/cab_courses.json. Include error handling and a main() function. 

Test your solution step-by-step in playground.ipynb to make sure it's functioning.

Keep it simple.
```

**Key Challenge:** AWS WAF blocks plain HTTP requests
**Solution:** Playwright browser context bypasses JS challenge, provides session cookies for requests

**Outcome:** Working scraper with retry logic, term auto-detection fallback

---

### 5. Bulletin Scraper (15 min)

**Goal:** Extract course listings from bulletin.brown.edu

**Prompt:**

```
Write etl/scrape_bulletin.py. Scrape https://bulletin.brown.edu/ course 
listings using requests + BeautifulSoup. Extract: course_code, title, 
instructor, meeting_times, prerequisites, department, description, 
source="Bulletin". Save to data/bulletin_courses.json. 

Include error handling and a main() function.

Test your solution step-by-step in playground.ipynb.

Keep it simple.
```

**Key Insight:** Bulletin is static HTML, no JS needed (unlike CAB)

**Outcome:** Fast scraper pulling ~11k courses in ~5 minutes

---

### 6. ETL Pipeline (10 min)

**Goal:** Merge and normalize data from both sources

**Prompt:**

```
Write etl/pipeline.py. It should:
1. Load data/cab_courses.json and data/bulletin_courses.json
2. Normalize and deduplicate (match on course_code)
3. Merge into unified schema with source field
4. Save to data/courses.json

Do not include a main() function. 

Test in playground.ipynb.

Keep it simple, no over-engineering.
```

**Merge Logic:** Bulletin is base (richer schedule data), CAB fills gaps

**Outcome:** Single unified dataset with provenance tracking

---

### 7. RAG Pipeline (25 min)

**Goal:** Build embedding + hybrid search infrastructure

**Prompt:**

```
Implement these three files:

rag/embedder.py - Load all-MiniLM-L6-v2 via sentence-transformers. 
Function to embed a list of course text strings (concatenate title + 
description + department). Save/load embeddings to data/embeddings.npy.

rag/vector_store.py - Build FAISS IndexFlatIP from embeddings. Support 
filtered search by department/source (pre-filter metadata, then search). 
Save/load index.

rag/search.py - Hybrid search combining FAISS similarity scores with BM25 
keyword scores (use rank_bm25). Weighted score fusion. Return top-k results 
with scores and metadata.

Test in playground.ipynb. Do not over-engineer.
```

**Design Choice:** Reciprocal rank fusion balances semantic + lexical rankings

**Outcome:** Fast hybrid retrieval handling both semantic queries and exact lookups

---

### 8. Backend API (20 min)

**Goal:** Serve queries via FastAPI endpoint

**Prompt:**

```
Write app/app.py using FastAPI:
- POST /query accepting {"q": str, "department": str | None}
- Load courses.json, FAISS index, and BM25 index on startup
- Call hybrid search, assemble context from top-5 results, call OpenAI API 
  for generation
- Return: {"answer": str, "courses": [{"code", "title", "department", 
  "similarity", "source"}]}
- Log queries and response times to stdout

Keep the code simple.

Then add instructions on how to run the code in README.md file.
```

**Key Feature:** Startup pipeline orchestration—auto-scrapes if data missing

**Outcome:** Single command (`python app/app.py`) builds everything and starts server

---

### 9. Frontend UI (15 min)

**Goal:** Simple web interface for search

**Prompt:**

```
Write frontend/ui.py:
- Text input for query (trigger on Enter)
- Optional department dropdown (populated from courses.json)
- Submit button that calls localhost:8000/query
- Display: generated answer in a box, then retrieved courses as a table with 
  similarity scores
- Keep it minimal and clean, include only important lines, do not over-engineer
```

**UX Enhancement:** Press Enter to search (no button click required)

**Outcome:** Clean Streamlit UI with on-page instructions

---

### 10. Final polishing

**Goal:** Making whole system functioning and debugging.

Used manual prompts for small bug-fix, features creation (like UI instructions), and logging.

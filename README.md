# Brown Course Search

RAG-powered semantic search for Brown University courses. Combines dual-source scraping (Bulletin + CAB), hybrid retrieval (FAISS + BM25), and LLM synthesis to deliver intelligent course recommendations.

## Quick Start

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Configure (create .env)
echo "OPENAI_API_KEY=sk-..." > .env

# Run backend (builds data on first run)
python app/app.py

# Run frontend (new terminal)
streamlit run frontend/ui.py
```

## Tech Stack

| Component       | Choice             | Why                                                  |
| --------------- | ------------------ | ---------------------------------------------------- |
| Backend         | FastAPI + Uvicorn  | Async-native, auto-docs, fast                        |
| Frontend        | Streamlit          | Rapid prototyping with minimal code                  |
| Embeddings      | all-MiniLM-L6-v2   | Best speed/quality trade-off, runs locally (384-dim) |
| Vector Store    | FAISS (CPU)        | In-process, disk-persistent, no infrastructure       |
| Keyword Search  | BM25               | Catches exact matches (codes, times, keywords)       |
| Scraping (CAB)  | Playwright + BS4   | Bypasses AWS WAF via browser context                 |
| Scraping (Bul.) | Requests + BS4     | Static HTML, no JS needed                            |
| LLM             | OpenAI GPT-4o-mini | Answer synthesis from retrieved context              |

## How It Works

**Hybrid Retrieval Strategy**

Combines two complementary search methods:

- **FAISS (semantic):** Understands intent and meaning→ *"machine learning"* matches *"neural networks"*, *"deep learning"*
- **BM25 (lexical):** Exact keyword/code matching
  → *"CSCI0320"*, *"Friday 3pm"* matches precisely

Both rankings are fused via reciprocal rank fusion, then top-k results are sent to the LLM for natural language synthesis. This handles both semantic queries (*"philosophy of reality"* → metaphysics) and precise lookups (*"CSCI0320"*).

## Pipeline Overview

When you run `python app/app.py`, the system automatically:

1. **Scrapes** Brown Bulletin and CAB
2. **Merges** both sources into unified schema
3. **Embeds** course descriptions with `all-MiniLM-L6-v2`
4. **Builds** FAISS index + BM25 corpus
5. **Serves** API on `http://localhost:8000`

Each step is **skipped** if its output already exists. To force rebuild: delete files in `data/`.

## API Usage

**Endpoint:** `POST /query`

```json
// Request
{ 
  "q": "machine learning courses on Fridays", 
  "department": "Computer Science"  // optional
}

// Response
{
  "answer": "The best match is CSCI 1951A...",
  "courses": [
    { 
      "code": "CSCI1951A", 
      "title": "Data Science", 
      "department": "Computer Science",
      "similarity": 0.87, 
      "source": "Bulletin" 
    }
  ]
}
```

Interactive docs: `http://127.0.0.1:8000/docs`

## Project Structure

```
Brown_course_search_RAG/
├── app/
│   └── app.py                  # FastAPI server + startup orchestration
├── etl/
│   ├── scrape_bulletin.py      # Bulletin scraper (Requests + BS4)
│   ├── scrape_cab.py           # CAB scraper (Playwright + BS4)
│   └── pipeline.py             # Merge + normalize → courses.json
├── rag/
│   ├── embedder.py             # Encode text → FAISS index
│   ├── vector_store.py         # FAISS wrapper (save/load/search)
│   └── search.py               # Hybrid: FAISS + BM25 → fused ranking
├── frontend/
│   └── ui.py                   # Streamlit interface
└── data/                       # Generated at runtime (git-ignored)
    ├── courses.json            # Merged course records
    ├── faiss.index             # Vector index
    └── metadata.json           # Parallel metadata for lookups
```

## Architecture Flow

```
┌──────────────┐
│ ETL Pipeline │  Scrape Bulletin + CAB → merge → courses.json
└──────┬───────┘
       ↓
┌──────────────┐
│   Embedder   │  Encode descriptions → FAISS index (384-dim)
└──────┬───────┘
       ↓
┌──────────────┐
│ Hybrid Search│  Query → FAISS (semantic) + BM25 (lexical) → top-k
└──────┬───────┘
       ↓
┌──────────────┐
│  LLM + API   │  Context + prompt → GPT-4o-mini → JSON response
└──────┬───────┘
       ↓
┌──────────────┐
│  Streamlit   │  Display answer + retrieved courses
└──────────────┘
```

**Step-by-Step Explanation:**

1. **ETL Pipeline**

   - `scrape_bulletin.py` pulls course data from Brown's Bulletin (static HTML)
   - `scrape_cab.py` pulls from CAB using Playwright (bypasses AWS WAF)
   - `pipeline.py` merges both sources by course code, preferring Bulletin for schedule data
2. **Embedder**

   - Encodes each course description into a 384-dim vector using `all-MiniLM-L6-v2`
   - Builds FAISS flat index for fast similarity search
   - Saves index + metadata to disk for reuse
3. **Hybrid Search**

   - User query is scored by FAISS (semantic similarity) and BM25 (keyword matching)
   - Rankings are fused using reciprocal rank fusion
   - Top-k results balance meaning and precision
4. **API + LLM**

   - FastAPI loads index and corpus at startup
   - `/query` endpoint takes query + optional department filter
   - Top 5 courses become context for GPT-4o-mini
   - LLM generates 2-4 sentence answer
5. **Frontend**

   - Streamlit UI with search bar and department dropdown
   - Submits queries to API, displays answer + retrieved courses
   - Shows similarity scores and sources for transparency

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
* [X] UI
* [X] Polish - deployment + docs + report
* [ ] Loom video

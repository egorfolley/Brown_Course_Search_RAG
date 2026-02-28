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

## TO DO

* [X] Technical setup - choose embedding model, vector store, LLM, scraping tools/ETL
  * [X] FastAPI + Streamlit for simplicity
  * [ ] Further UI modification to React if needed
* [X] ClaudeCode prompts for initial development
* [ ] Design architecture
* [ ] Scaffolding - project structure
* [ ] CAB scraper
* [ ] ETL pipeline and storage
* [ ] RAG pipeline
* [ ] Test solutions in Playground notebook
* [ ] Backend API
* [ ] UI
* [ ] Polish - deployment + docs + report
* [ ] Loom video

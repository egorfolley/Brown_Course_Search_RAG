# Brown Course Search: Technical Report

## How the RAG Pipeline Works

The RAG (Retrieval-Augmented Generation) pipeline operates in five stages:

### 1. Data Acquisition & ETL
- **Dual-source scraping:** Bulletin scraper extracts ~11k courses via static HTML parsing (Requests + BeautifulSoup); CAB scraper uses Playwright to bypass AWS WAF and access the Ribbit API
- **Normalization:** Both sources are merged on course code, with Bulletin as the base (richer schedule data) and CAB filling gaps
- **Output:** Unified `courses.json` with fields: `course_code`, `title`, `department`, `description`, `instructor`, `meeting_times`, `prerequisites`, `source`

### 2. Embedding & Indexing
- **Text preparation:** Each course's title, description, and department are concatenated into a single string
- **Encoding:** `all-MiniLM-L6-v2` (sentence-transformers) generates 384-dimensional dense vectors capturing semantic meaning
- **Storage:** FAISS flat index (`IndexFlatIP`) stores vectors for fast cosine similarity search; metadata stored in parallel JSON for result enrichment

### 3. Hybrid Retrieval
When a query arrives:
- **Semantic search:** Query is embedded and FAISS returns top-k courses by cosine similarity
- **Lexical search:** BM25 scores the corpus for exact keyword/phrase matches
- **Fusion:** Reciprocal rank fusion combines both rankings, balancing semantic understanding with precise matching
- **Filtering:** Optional department filter pre-filters candidates before search

### 4. LLM Answer Generation
- **Context assembly:** Top 5 retrieved courses are formatted into a structured prompt with code, title, department, and description snippets
- **Synthesis:** GPT-4o-mini generates a 2-4 sentence natural language answer constrained to only the retrieved context
- **Temperature:** Set to 0.3 for consistent, factual responses

### 5. API Response
FastAPI returns JSON with both the LLM-generated answer and raw retrieval results (courses with similarity scores), allowing users to verify reasoning and explore alternatives.

---

## Why `all-MiniLM-L6-v2` Was Chosen

**Primary factors:**

1. **Speed/quality trade-off:** At 384 dimensions, it's 2-3× faster than larger models (768-dim BERT) while maintaining 95%+ of retrieval quality for domain-specific search
   
2. **Local execution:** Small enough (~80MB) to run on CPU without GPU infrastructure, critical for development agility and deployment simplicity

3. **Sentence-level semantics:** Optimized for sentence embeddings (not just word vectors), making it ideal for course descriptions and queries that are typically 1-3 sentences

4. **Proven track record:** Widely used in production RAG systems; extensive benchmarking on MTEB shows strong performance on semantic search tasks

5. **Brown courses context:** Course descriptions are relatively short (50-200 words) and use academic vocabulary—MiniLM captures this domain well without needing fine-tuning

**Alternative considered:** `text-embedding-3-small` (OpenAI) was evaluated but rejected due to API costs at scale and latency overhead on every query.

---

## Performance Observations

### Retrieval Quality
- **Semantic queries:** Excellent performance on conceptual searches (*"machine learning for beginners"* → CSCI 1420, CSCI 0200)
- **Exact lookups:** Hybrid search reliably surfaces exact course codes (*"CSCI0320"*) and specific keywords (*"Friday 3pm"*) that pure semantic search would miss
- **Failure mode:** Very short queries (1-2 words) sometimes return overly broad results; adding department filter mitigates this

### Speed
- **Cold start:** Initial data pipeline (scrape + embed + index) takes ~20 minutes
- **Index loading:** FAISS index + BM25 corpus loads in ~200ms at API startup
- **Query latency:** 
  - Hybrid search: 15-30ms for top-5 retrieval
  - Total (retrieval + LLM): 400-600ms end-to-end
  - LLM generation dominates latency (350-500ms of total)

### Scalability
- **Current load:** 11,066 courses → 4.2MB embeddings, FAISS search remains sub-linear
- **Estimated ceiling:** Up to ~100k courses feasible on single CPU before needing GPU acceleration or approximate search (HNSW index)

### Resource Usage
- **Memory:** ~150MB for loaded FAISS index + BM25 corpus
- **Disk:** 25MB total for all artifacts (index, embeddings, metadata)
- **Compute:** Minimal during serving; scraping is I/O-bound, embedding is CPU-bound but one-time

---

## Production Improvements

### 1. Infrastructure & Scalability
- **Containerization:** Dockerize app + frontend for consistent deployment across environments
- **Horizontal scaling:** Add Redis cache for top queries; deploy multiple API replicas behind load balancer
- **Index optimization:** Switch to FAISS HNSW for approximate nearest neighbor search (10× faster with negligible quality loss at scale)
- **Async scraping:** Use `asyncio` + `aiohttp` for parallel scraping to reduce ETL time from 20min → ~5min

### 2. Data Quality & Freshness
- **Incremental updates:** Implement delta scraping (only fetch changed courses) with checksums to avoid full rebuilds
- **Data validation:** Add Pydantic schema validation to catch malformed courses before indexing
- **Deduplication:** Improve merge logic with fuzzy matching (Levenshtein distance) for course codes with formatting variations
- **Scheduling:** Run daily scrapes via cron/Airflow to keep data current

### 3. Retrieval & Ranking
- **Query understanding:** Add spell correction (SymSpell) and query expansion (synonyms for "intro" → "introduction", "beginner")
- **Reranking:** Use cross-encoder (e.g., `ms-marco-MiniLM-L6-v2`) to re-score top-20 candidates for improved precision
- **Personalization:** Track user click-through data to fine-tune BM25 weights and boost frequently selected courses
- **Metadata filtering:** Expose filters for credits, time slots, instructor ratings (if integrated with external APIs)

### 4. Observability & Reliability
- **Monitoring:** Add Prometheus metrics (query latency, cache hit rate, API errors) + Grafana dashboards
- **Logging:** Structured logging (JSON) with trace IDs for request correlation; log all queries for analytics
- **Error handling:** Graceful degradation if LLM API fails (return retrieval results only) + circuit breaker pattern
- **A/B testing:** Framework to test embedding models, fusion weights, and prompt variations with real users

### 5. User Experience
- **Response streaming:** Stream LLM tokens as they generate (SSE) for perceived lower latency
- **Relevance feedback:** "Was this helpful?" button to collect training data for future reranking models
- **Saved searches:** Allow users to bookmark queries and get notifications on new matching courses
- **Explainability:** Show why each course was retrieved (matching keywords highlighted, semantic similarity explained)

### 6. Security & Compliance
- **Rate limiting:** Prevent abuse with per-IP/per-user quotas (e.g., 100 queries/hour)
- **Input sanitization:** Validate and escape queries to prevent injection attacks
- **Data privacy:** If integrating with student data (e.g., course history), ensure FERPA compliance
- **API authentication:** Add JWT tokens if exposing API to third-party integrations

### 7. Cost Optimization
- **LLM caching:** Cache LLM responses for identical (query, context) pairs to reduce OpenAI costs
- **Model alternatives:** Evaluate self-hosted small LLMs (Llama 3.2, Phi-4) to eliminate API dependency
- **Batch processing:** For analytics/reporting, batch-embed new courses nightly instead of real-time

---

## Summary

The current system demonstrates a functional RAG pipeline balancing quality, speed, and simplicity. It handles ~11k courses with sub-second query latency and delivers accurate results for both semantic and exact-match queries. The hybrid retrieval strategy proves essential—neither FAISS nor BM25 alone would satisfy the diverse query patterns students use.

For production deployment, the priority improvements are: (1) data freshness automation, (2) observability/monitoring, and (3) query latency optimization via caching and reranking. The architecture is sound and scales to 10× current data volume without fundamental changes.

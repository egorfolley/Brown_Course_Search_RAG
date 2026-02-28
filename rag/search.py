"""
Hybrid search module.

Combines FAISS semantic scores and BM25 lexical scores via reciprocal
rank fusion. Supports metadata filters (department, time slot, credits)
applied before final top-k selection.
"""

"""
FAISS vector store wrapper.

Handles loading the persisted index and metadata at startup,
and exposes a nearest-neighbour search returning (course_id, score) pairs
for a given query embedding.
"""

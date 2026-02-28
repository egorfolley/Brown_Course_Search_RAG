"""
Course embedding module.

Loads courses from data/courses.json, encodes each course's text
(title + description) with sentence-transformers (all-MiniLM-L6-v2),
and builds + persists a FAISS flat index to data/faiss.index alongside
a metadata list to data/metadata.json.
"""

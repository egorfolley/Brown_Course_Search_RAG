"""
FastAPI application entry point.

Defines the POST /query endpoint: accepts a natural-language query with
optional metadata filters, runs hybrid search, calls the LLM generator,
and returns ranked course results alongside the generated answer.
"""

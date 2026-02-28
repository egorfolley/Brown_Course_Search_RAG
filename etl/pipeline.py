"""
ETL pipeline orchestrator.

Runs both scrapers, merges and normalizes their output into a unified
schema, deduplicates by course code, and writes data/courses.json.
"""

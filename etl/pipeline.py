"""
ETL pipeline: loads CAB + Bulletin scraper output, merges on course_code,
and writes data/courses.json.

Merge strategy:
  - Bulletin is the base (has instructor + schedule from the offerings table)
  - CAB fills any fields still empty after Bulletin (sometimes has richer prereqs)
  - Courses that appear in only one source are included as-is
  - source field reflects provenance: "Bulletin", "CAB", or "CAB+Bulletin"
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CAB_FILE = DATA_DIR / "cab_courses.json"
BULLETIN_FILE = DATA_DIR / "bulletin_courses.json"
OUTPUT_FILE = DATA_DIR / "courses.json"

MERGE_FIELDS = ("description", "prerequisites", "title", "department",
                "instructor", "meeting_times")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    """Load a JSON array from disk; return [] if the file doesn't exist."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_code(code: str) -> str:
    """Canonicalise course codes: 'CSCI 0320' and 'csci0320' → 'CSCI0320'."""
    return code.replace(" ", "").upper().strip()


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def merge_courses(cab: list[dict], bulletin: list[dict]) -> list[dict]:
    """
    Merge two lists of course dicts by normalized course_code.

    Returns a deduplicated list ordered: Bulletin-only, then merged, then CAB-only.
    """
    index: dict[str, dict] = {}

    # 1. Index Bulletin courses (richer schedule/instructor data)
    for course in bulletin:
        code = normalize_code(course.get("course_code", ""))
        if code:
            index[code] = {**course, "course_code": code}

    # 2. Merge CAB courses
    for course in cab:
        code = normalize_code(course.get("course_code", ""))
        if not code:
            continue

        if code in index:
            # Back-fill any field that Bulletin left empty
            for field in MERGE_FIELDS:
                if not index[code].get(field) and course.get(field):
                    index[code][field] = course[field]
            index[code]["source"] = "CAB+Bulletin"
        else:
            index[code] = {**course, "course_code": code}

    return list(index.values())


# ---------------------------------------------------------------------------
# Entry point (importable, not a CLI)
# ---------------------------------------------------------------------------

def run() -> list[dict]:
    """Load both scraped files, merge, save courses.json, return the result."""
    cab = load(CAB_FILE)
    bulletin = load(BULLETIN_FILE)

    courses = merge_courses(cab, bulletin)

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(courses, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return courses

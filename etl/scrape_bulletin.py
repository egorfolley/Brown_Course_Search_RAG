"""
Scraper for the Brown University Bulletin (bulletin.brown.edu).

The Bulletin is fully server-rendered static HTML — no JS rendering or API
reverse-engineering needed. All data is fetched with plain requests + BS4.

Discovered structure (verified by inspecting bulletin.brown.edu):
  - Department index: /departments-centers-programs-institutes/
    Links are <a href="/computerscience/"> etc.
  - Department page: /<slug>/ contains <div class="courseblock"> per course.
  - Course fields:
      code        — <p class="courseblocktitle" data-code="CSCI 0320">
      title       — <strong> inside courseblocktitle, after "CODE. "
      description — <p class="courseblockdesc"> (one or more, joined)
      prereqs     — embedded in description text ("Prerequisite: ...")
      schedule    — <table class="tbl_offering"> with columns:
                      0=semester, 1=code, 2=section, 3=reg_num,
                      4=days, 5=time, 6=instructor
  - No pagination — all courses for a department are on one page.

Output: data/bulletin_courses.json
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://bulletin.brown.edu"
DEPT_INDEX = f"{BASE_URL}/departments-centers-programs-institutes/"

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "bulletin_courses.json"

REQUEST_DELAY = 0.3   # seconds between page fetches

log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Brown-Course-Search-RAG/1.0 (research project)"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    """GET a URL with exponential-backoff retries."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Request failed (attempt %d/%d) %s: %s", attempt + 1, retries, url, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Department discovery
# ---------------------------------------------------------------------------

def fetch_department_urls() -> list[tuple[str, str]]:
    """
    Return (name, url) pairs for every department on the index page.

    The index page lists departments in two columns of <ul><li><a> links.
    """
    resp = _get(DEPT_INDEX)
    if not resp:
        log.error("Could not fetch department index: %s", DEPT_INDEX)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    depts = []
    for a in soup.select("div.clearfix a[href]"):
        href = a["href"].strip()
        name = a.get_text(strip=True)
        if href and name:
            url = href if href.startswith("http") else f"{BASE_URL}{href}"
            depts.append((name, url))

    log.info("Found %d departments.", len(depts))
    return depts


# ---------------------------------------------------------------------------
# Course parsing
# ---------------------------------------------------------------------------

def _extract_prereqs(description: str) -> str:
    """Pull the prerequisite sentence(s) out of the description text."""
    match = re.search(
        r"(prerequisite[s]?[:\s].+?)(?:\.|$)",
        description,
        flags=re.IGNORECASE,
    )
    return match.group(0).strip() if match else ""


def _parse_offering_table(table) -> tuple[str, str]:
    """
    Parse the tbl_offering schedule table.

    Returns (meeting_times, instructor) from the first non-header row.
    meeting_times is formatted as "MWF 10:00-10:50" or "TTh 13:00-14:20".
    Instructor is the name stripped of surrounding parentheses.
    """
    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:
            continue
        days = cols[4].get_text(strip=True)
        time_str = cols[5].get_text(strip=True)
        raw_instructor = cols[6].get_text(strip=True)

        # Meeting time: strip room number in parens from the time field
        # e.g. "10:00-10:50(Sm 102)" → "10:00-10:50"
        time_clean = re.sub(r"\(.*?\)", "", time_str).strip()
        meeting = f"{days} {time_clean}".strip() if days and time_clean else ""

        # Instructor: "(K. Fisler)" → "K. Fisler"; blank if "To Be Arranged"
        instructor = re.sub(r"[()]", "", raw_instructor).strip()
        if re.search(r"arranged|TBA", instructor, flags=re.IGNORECASE):
            instructor = ""

        return meeting, instructor

    return "", ""


def parse_courseblock(block, dept_name: str) -> Optional[dict]:
    """Parse a single <div class="courseblock"> into the target schema."""
    # --- code and title ---
    title_tag = block.find("p", class_="courseblocktitle")
    if not title_tag:
        return None

    # Prefer data-code attribute (already clean: "CSCI 0320")
    raw_code = title_tag.get("data-code", "").strip()
    course_code = raw_code.replace(" ", "") if raw_code else ""

    strong = title_tag.find("strong")
    raw_title = strong.get_text(" ", strip=True) if strong else title_tag.get_text(" ", strip=True)
    # "CSCI 0320. Introduction to Software Engineering."  → strip code prefix
    title = re.sub(r"^[A-Z]+\s*\d+[A-Z]*\.\s*", "", raw_title).rstrip(". ")

    # --- description (join multiple paragraphs) ---
    desc_parts = [p.get_text(" ", strip=True) for p in block.find_all("p", class_="courseblockdesc")]
    description = " ".join(desc_parts)

    # --- prerequisites (extracted from description) ---
    prerequisites = _extract_prereqs(description)

    # --- schedule from offerings table (may not exist for all courses) ---
    meeting_times = ""
    instructor = ""
    table = block.find("table", class_="tbl_offering")
    if table:
        meeting_times, instructor = _parse_offering_table(table)

    if not course_code and not title:
        return None

    return {
        "course_code": course_code,
        "title": title,
        "instructor": instructor,
        "meeting_times": meeting_times,
        "prerequisites": prerequisites,
        "department": dept_name,
        "description": description,
        "source": "Bulletin",
    }


# ---------------------------------------------------------------------------
# Department scraping
# ---------------------------------------------------------------------------

def scrape_department(name: str, url: str) -> list[dict]:
    """Fetch a department page and return all its parsed course dicts."""
    resp = _get(url)
    if not resp:
        log.warning("Skipping department '%s' — request failed.", name)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    blocks = soup.find_all("div", class_="courseblock")
    if not blocks:
        log.debug("No courseblocks found on '%s' (%s).", name, url)
        return []

    courses = []
    for block in blocks:
        course = parse_courseblock(block, name)
        if course:
            courses.append(course)

    log.info("  %s: %d courses.", name, len(courses))
    return courses


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def scrape_all() -> list[dict]:
    """Scrape every department and return all course dicts."""
    depts = fetch_department_urls()
    if not depts:
        log.error("No departments found. Check connectivity and %s", DEPT_INDEX)
        return []

    all_courses: list[dict] = []
    for name, url in depts:
        log.info("Scraping: %s", name)
        all_courses.extend(scrape_department(name, url))
        time.sleep(REQUEST_DELAY)

    return all_courses
"""
Scraper for Brown's Course Announcement Bulletin (CAB).

CAB is built on CourseLeaf's CAB product, which exposes a public "Ribbit" API
at /ribbit/index.cgi. All data is fetched with Playwright because cab.brown.edu
sits behind AWS WAF, which returns HTTP 202 (JS challenge) to plain requests.
Playwright's context.request API sends requests with a real browser session
(including the WAF token cookie), bypassing the challenge automatically.

Discovered by inspecting cab.brown.edu network traffic:
  - page=listsubjects.rjs  → JSON list of all department codes
  - page=listcourses.rjs   → JSON list of course codes for a subject
  - page=getcourse.rjs     → XML wrapping an HTML courseblock fragment

NOTE: CAB is the course *catalog* (descriptions, prereqs, credits). It does
not contain live schedule data. instructor and meeting_times are therefore
left empty here; scrape_bulletin.py fills those from the Bulletin.

Output: data/cab_courses.json
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://cab.brown.edu/ribbit/index.cgi"

# Term code format: YYYYTT where TT = 10 (Spring), 20 (Summer), 30 (Fall).
# Verify at cab.brown.edu — the page title shows the active term.
CURRENT_TERM = "202510"  # Spring 2025

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "cab_courses.json"

REQUEST_DELAY = 0.25  # seconds between requests

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Playwright session  (one browser for the whole scrape)
# ---------------------------------------------------------------------------

class _CABSession:
    """
    Wraps a single Playwright browser context for all Ribbit API calls.

    On first use, opens Chromium headlessly and navigates to the CAB home
    page so the AWS WAF JS challenge is solved and the token cookie is set.
    All subsequent requests via context.request reuse that cookie, bypassing
    the WAF without launching extra browsers.
    """

    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context()
        log.info("Establishing CAB browser session (solving WAF challenge)…")
        page = self._context.new_page()
        page.goto("https://cab.brown.edu/", wait_until="networkidle", timeout=30_000)
        page.close()
        log.info("Session ready.")

    def get(self, params: dict, retries: int = 3) -> Optional[str]:
        """GET a Ribbit endpoint; return response text or None on failure."""
        url = BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        for attempt in range(retries):
            try:
                resp = self._context.request.get(url, timeout=15_000)
                if resp.ok:
                    return resp.text()
                log.warning("HTTP %s for %s", resp.status, url)
            except Exception as exc:
                log.warning("Request failed (attempt %d/%d): %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def close(self):
        self._browser.close()
        self._pw.stop()


_SESSION: Optional[_CABSession] = None


def _get_session() -> _CABSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = _CABSession()
    return _SESSION


# ---------------------------------------------------------------------------
# Ribbit API calls
# ---------------------------------------------------------------------------

def fetch_subjects() -> list[str]:
    """
    Return all department/subject codes from the catalog.

    Ribbit response: JSON array of objects with 'key' (e.g. "CSCI") and 'name'.
    """
    text = _get_session().get({"page": "listsubjects.rjs", "Term": CURRENT_TERM})
    if not text:
        log.error("Could not fetch subject list; falling back to empty list.")
        return []
    try:
        data = json.loads(text)
        subjects = [entry["key"] for entry in data if "key" in entry]
        log.info("Found %d subjects.", len(subjects))
        return subjects
    except (ValueError, KeyError) as exc:
        log.error("Failed to parse subject list: %s", exc)
        return []


def fetch_course_codes(subject: str) -> list[str]:
    """
    Return all course codes for a subject (e.g. ["CSCI 0111", "CSCI 0320", ...]).

    Ribbit response: JSON array of objects with 'key' (full code) and 'name' (title).
    """
    text = _get_session().get({"page": "listcourses.rjs", "subject": subject, "Term": CURRENT_TERM})
    if not text:
        return []
    try:
        data = json.loads(text)
        return [entry["key"] for entry in data if "key" in entry]
    except (ValueError, KeyError) as exc:
        log.warning("Failed to parse course list for %s: %s", subject, exc)
        return []


def fetch_course_detail(code: str) -> Optional[str]:
    """
    Return the raw HTML courseblock string for a single course.

    Ribbit response: XML like <result><![CDATA[<div class="courseblock">...</div>]]></result>
    BeautifulSoup extracts the inner text, which is the HTML fragment.
    """
    text = _get_session().get({"page": "getcourse.rjs", "code": code, "Term": CURRENT_TERM})
    if not text:
        return None
    # The outer document is minimal XML; BS4's html.parser handles it fine.
    soup = BeautifulSoup(text, "html.parser")
    result_tag = soup.find("result") or soup.find("course")
    if not result_tag:
        log.debug("No result tag in response for %s", code)
        return None
    return result_tag.decode_contents()


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_courseblock(html: str, subject: str) -> dict:
    """
    Parse a CourseLeaf courseblock HTML fragment into the target schema.

    Typical structure:
        <div class="courseblock">
          <p class="courseblocktitle">CSCI 0320. Introduction to Software Engineering. 1 unit.</p>
          <p class="courseblockdesc">A rigorous introduction to...</p>
          <p class="courseblockextra">Prerequisite: CSCI 0111 or CSCI 0150.</p>
          <p class="courseblockextra">Instructor: Tim Nelson.</p>
        </div>
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- title block: "CSCI 0320. Introduction to Software Engineering. 1 unit." ---
    title_tag = soup.find("p", class_="courseblocktitle")
    raw_title = title_tag.get_text(" ", strip=True) if title_tag else ""

    code_match = re.match(r"^([A-Z]+\s+\d+[A-Z]*)\.", raw_title)
    course_code = code_match.group(1).replace(" ", "") if code_match else ""
    # Title is everything after "CODE. " up to the next period (credits come last)
    title = re.sub(r"^[A-Z]+\s+\d+[A-Z]*\.\s*", "", raw_title)
    title = re.sub(r"\.\s*\d+\.?\d*\s*credit.*$", "", title, flags=re.IGNORECASE).strip(" .")

    # --- description ---
    desc_tag = soup.find("p", class_="courseblockdesc")
    description = desc_tag.get_text(" ", strip=True) if desc_tag else ""

    # --- extra fields (prerequisites, instructor, etc.) ---
    prerequisites = ""
    instructor = ""
    meeting_times = ""

    for extra in soup.find_all("p", class_="courseblockextra"):
        text = extra.get_text(" ", strip=True)
        lower = text.lower()
        if "prerequisite" in lower or "prereq" in lower:
            prerequisites = text
        elif "instructor" in lower:
            # Strip the label, keep the value
            instructor = re.sub(r"^instructor[s]?:?\s*", "", text, flags=re.IGNORECASE).strip()
        elif any(k in lower for k in ("meeting", "schedule", "time", "days")):
            meeting_times = text

    return {
        "course_code": course_code,
        "title": title,
        "instructor": instructor,
        "meeting_times": meeting_times,
        "prerequisites": prerequisites,
        "department": subject,
        "description": description,
        "source": "CAB",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def scrape_subject(subject: str) -> list[dict]:
    """Scrape all courses for one subject; return list of course dicts."""
    codes = fetch_course_codes(subject)
    if not codes:
        log.warning("No courses found for subject '%s'.", subject)
        return []

    courses = []
    for code in codes:
        html = fetch_course_detail(code)
        if not html:
            log.debug("Skipping %s — no detail returned.", code)
            continue
        course = _parse_courseblock(html, subject)
        if course["course_code"]:
            courses.append(course)
        time.sleep(REQUEST_DELAY)

    log.info("  %s: %d/%d courses parsed.", subject, len(courses), len(codes))
    return courses


def scrape_all() -> list[dict]:
    """Scrape every subject in the catalog and return all course dicts."""
    subjects = fetch_subjects()
    if not subjects:
        log.error("No subjects found. Check CURRENT_TERM (%s) and connectivity.", CURRENT_TERM)
        return []

    all_courses: list[dict] = []
    for subject in subjects:
        log.info("Scraping: %s", subject)
        all_courses.extend(scrape_subject(subject))

    return all_courses

"""
Streamlit frontend for Brown Course Search.

Calls POST http://localhost:8000/query and displays the LLM answer
and retrieved courses as a table.
"""

import json
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"
COURSES_FILE = Path(__file__).parent.parent / "data" / "courses.json"

st.set_page_config(page_title="Brown Course Search", layout="centered")
st.title("Brown Course Search")

st.markdown(
    """
Use this app to search Brown courses with RAG.

### Quick start
1. Start backend API in another terminal: `python app/app.py`
2. Enter a query below (example: *machine learning classes on Fridays*)
3. Optionally choose a department filter
4. Click **Search**

### Notes
- This tool returns an LLM summary plus top retrieved courses.
- If the API is not running, you'll see a connection error.
"""
)


def _load_departments() -> list[str]:
    if not COURSES_FILE.exists():
        return []
    courses = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    depts = sorted({c.get("department", "") for c in courses if c.get("department")})
    return depts


departments = _load_departments()
dept_options = ["All departments"] + departments

query = st.text_input("Search query", placeholder="e.g. machine learning on Fridays")
dept = st.selectbox("Department", dept_options)
submitted = st.button("Search")


if submitted:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        payload: dict = {"q": query}
        if dept != "All departments":
            payload["department"] = dept

        with st.spinner("Searching…"):
            try:
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Start it with: python app/app.py")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc}")
                st.stop()

        st.subheader("Answer")
        st.info(data.get("answer", "No answer returned."))

        courses = data.get("courses", [])
        if courses:
            st.subheader("Retrieved courses")
            rows = [
                {
                    "Code": c.get("code", ""),
                    "Title": c.get("title", ""),
                    "Department": c.get("department", ""),
                    "Similarity": round(float(c.get("similarity", 0.0)), 3),
                    "Source": c.get("source", ""),
                }
                for c in courses
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No courses returned for this query.")

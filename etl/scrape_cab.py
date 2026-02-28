"""
Scraper for Brown's Course Announcement Bulletin (CAB).

Fetches course listings via Requests/BeautifulSoup for static content
and Playwright for any JS-rendered sections. Outputs raw course dicts
that are passed to the normalizer in pipeline.py.
"""

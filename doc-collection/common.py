"""Helpers shared by the collection scripts (scrape.py, events.py, sports.py, cmoa_collection.py).

Every collected document is a text file data/pages/<host>/<name>.txt whose first line is
"URL: <source>", followed by a blank line and the text; rag/preprocess.py reads them from there.
"""
import html
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (CMU ANLP course project crawler)"}
HERE = Path(__file__).parent
PAGES = HERE / "data" / "pages"   # output folder for every script
RAW = HERE / "raw_pages"          # pages saved by hand from a browser
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def get(url, timeout=30, **kwargs):
    return requests.get(url, headers=HEADERS, timeout=timeout, **kwargs)


def soup_of(url):
    return BeautifulSoup(get(url).content, "html.parser")


def text(el):
    """An element's text on one line."""
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el else ""


def plain(fragment):
    """An HTML fragment (e.g. a description field in a JSON feed) as one line of plain text."""
    return re.sub(r"\s+", " ", html.unescape(BeautifulSoup(fragment or "", "html.parser").get_text(" "))).strip()


def day(d):
    """Thursday, October 15, 2026: dates are written out in full so a question's date matches."""
    return f"{d:%A, %B} {d.day}, {d.year}"


def clock(d):
    """7:30 PM"""
    return d.strftime("%I:%M %p").lstrip("0")


def save_doc(path, url, heading, items, sep="\n", unit="lines"):
    """Write one document: URL line, heading, then items (one line, or one paragraph with sep="\\n\\n", each)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"URL: {url}\n\n{heading}\n\n" + sep.join(items) + "\n", encoding="utf-8")
    print(f"saved   {path} ({len(items)} {unit})")

"""Download the CMU events calendar (events.cmu.edu) month by month.

The calendar page is rendered with JavaScript, so scraping its HTML gets nothing.
It runs on LiveWhale, which serves the same events as JSON:
    https://events.cmu.edu/live/json/events/start_date/2026-10-01/end_date/2026-11-01?page=1
Each month is saved as events.cmu.edu/<yyyy-mm>.txt, one paragraph per event.
"""
import html
import re
import time
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://events.cmu.edu/live/json/events/start_date/{start}/end_date/{end}" \
      "/response_fields/location,summary,description,group_title,event_types?page={page}"
HEADERS = {"User-Agent": "Mozilla/5.0 (CMU ANLP course project crawler)"}
HIDDEN_TYPES = {"Exclude from Main Calendar"}


def fetch_month(year, month, delay=1.0):
    start = date(year, month, 1)
    end = date(year + month // 12, month % 12 + 1, 1)  # first day of next month
    events, page = [], 1
    while True:
        r = requests.get(API.format(start=start, end=end, page=page), headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        events += [e for e in data["data"] if e["date_iso"][:7] == f"{year}-{month:02d}"]
        if page >= data["meta"]["total_pages"]:
            return events
        page += 1
        time.sleep(delay)


def plain(value):
    """HTML fragment -> one line of plain text."""
    text = BeautifulSoup(value or "", "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def when(e):
    """Thursday, October 15, 2026, 4:00 PM - 5:00 PM (or 'all day', or a date range)."""
    start = datetime.fromisoformat(e["date_iso"])
    end = datetime.fromisoformat(e["date2_iso"]) if e.get("date2_iso") else None
    day = f"{start:%A, %B} {start.day}, {start.year}"
    clock = lambda d: d.strftime("%I:%M %p").lstrip("0")
    if e.get("is_all_day"):
        if end and end.date() > start.date():
            return f"{day} to {end:%A, %B} {end.day}, {end.year} (all day)"
        return f"{day} (all day)"
    if end and end.date() > start.date():
        return f"{day}, {clock(start)} to {end:%A, %B} {end.day}, {end.year}, {clock(end)}"
    return f"{day}, {clock(start)}" + (f" - {clock(end)}" if end else "")


def event_text(occurrences):
    """One block for an event; recurring events list all their dates instead of repeating."""
    e = occurrences[0]
    dates = "; ".join(when(o) for o in occurrences)
    first = f"{plain(e['title'])}: {dates}"
    if e.get("location"):
        first += f", {plain(e['location'])}"
    lines = [first]
    details = []
    if e.get("group_title"):
        details.append(f"Hosted by {plain(e['group_title'])}.")
    types = [plain(t) for t in e.get("event_types") or [] if plain(t) not in HIDDEN_TYPES]
    if types:
        details.append(f"Type: {', '.join(types)}.")
    if e.get("is_canceled"):
        details.append("This event is canceled.")
    if details:
        lines.append(" ".join(details))
    for field in ("summary", "description"):
        text = plain(e.get(field))
        if text and text not in lines:
            lines.append(text)
    return "\n".join(lines)


def group_recurring(events):
    """Occurrences with the same title, place and description, in order of first date."""
    groups = {}
    for e in events:
        key = (plain(e["title"]), plain(e.get("location")), plain(e.get("summary")), plain(e.get("description")))
        groups.setdefault(key, []).append(e)
    return list(groups.values())


def save_months(months, out_dir):
    """months: list of (year, month)."""
    for year, month in months:
        events = fetch_month(year, month)
        name = f"{date(year, month, 1):%B %Y}"
        path = Path(out_dir) / "events.cmu.edu" / f"{year}-{month:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://events.cmu.edu/month/date/{year}{month:02d}01"
        groups = group_recurring(events)
        body = "\n\n".join(event_text(g) for g in groups)
        path.write_text(f"URL: {url}\n\nCMU Events Calendar: {name}\n\n{body}\n", encoding="utf-8")
        print(f"saved   {path.relative_to(out_dir)} ({len(events)} events, {len(groups)} after merging repeats)")


if __name__ == "__main__":
    months = [(2026, 10), (2026, 11), (2026, 12)]
    save_months(months, "data/pages")

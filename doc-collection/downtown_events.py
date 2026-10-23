"""Download the Downtown Pittsburgh events calendar (downtownpittsburgh.com/events).

The calendar page is rendered with JavaScript, so scraping its HTML gets nothing.
Its events come from the site's WordPress plugin (Vibemap) as JSON:
    https://downtownpittsburgh.com/wp-json/vibemap/v1/events-data?page=1
Each occurrence of a recurring event is a separate item; occurrences of the same
event are merged into one paragraph that lists all their dates.
Output: downtownpittsburgh.com/events-<yyyy-mm>.txt (by month of first date) and
downtownpittsburgh.com/events-ongoing.txt (exhibitions and other long runs).
"""
import html
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://downtownpittsburgh.com/wp-json/vibemap/v1/events-data?page={page}"
HEADERS = {"User-Agent": "Mozilla/5.0 (CMU ANLP course project crawler)"}
MAX_LISTED_DATES = 10
OCCURRENCE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")  # slug suffix, e.g. _2026-11-26-11-00


def fetch_all(delay=1.0):
    events, page = [], 1
    while True:
        r = requests.get(API.format(page=page), headers=HEADERS, timeout=120)
        r.raise_for_status()
        data = r.json()
        events += data["events"]
        print(f"page {page}/{data['pagination']['total_pages']}")
        if not data["pagination"]["has_more"]:
            return events
        page += 1
        time.sleep(delay)


def plain(value):
    text = BeautifulSoup(value or "", "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_dt(value):
    return datetime.fromisoformat(value) if value else None


def day(d):
    return f"{d:%A, %B} {d.day}, {d.year}"


def clock(d):
    return d.strftime("%I:%M %p").lstrip("0")


def occurrence(e):
    """Start of this occurrence: from the slug suffix, else the event's start date."""
    m = OCCURRENCE.search(e["slug"])
    if m:
        return datetime(*map(int, m.groups()))
    return parse_dt(e["meta"].get("vibemap_event_start_date"))


def is_long_run(e):
    """Exhibitions and shows that run for weeks: described by their date range."""
    meta = e["meta"]
    start, end = parse_dt(meta.get("vibemap_event_start_date")), parse_dt(meta.get("vibemap_event_end_date"))
    return meta.get("vibemap_event_is_ongoing") == "1" and start and end and (end - start).days >= 14


def when(e, start):
    """Saturday, October 24, 2026, 7:30 PM - 10:00 PM (time left out when it is midnight/unknown)."""
    end = parse_dt(e["meta"].get("vibemap_event_end_date"))
    text = day(start)
    if start.hour or start.minute:
        text += f", {clock(start)}"
        if end and end.date() == start.date() and end > start:
            text += f" - {clock(end)}"
    return text


def block(occurrences, boilerplate=()):
    e = occurrences[0]
    meta = e["meta"]
    title = plain(e["title"])
    venue = plain(meta.get("vibemap_event_venue_name") or meta.get("vibemap_event_hotspots_place"))
    address = plain(meta.get("vibemap_event_venue_address") or meta.get("vibemap_event_address"))
    if is_long_run(e):
        start = parse_dt(meta["vibemap_event_start_date"])
        end = parse_dt(meta["vibemap_event_end_date"])
        dates = f"runs {day(start)} through {day(end)}"
    elif len(occurrences) > MAX_LISTED_DATES:  # e.g. a daily show: give the range, not every day
        starts = [occurrence(o) for o in occurrences]
        dates = f"{len(starts)} dates from {day(starts[0])} through {day(starts[-1])}"
        if len({(d.hour, d.minute) for d in starts}) == 1 and (starts[0].hour or starts[0].minute):
            dates += f", at {clock(starts[0])}"
    else:
        dates = "; ".join(when(o, occurrence(o)) for o in occurrences)
    first = f"{title}: {dates}"
    for part in (venue, address):
        if part and part not in first:
            first += f", {part}"
    lines = [first]
    details = []
    if meta.get("vibemap_event_organizer"):
        details.append(f"Organized by {plain(meta['vibemap_event_organizer'])}.")
    if meta.get("vibemap_event_recurrence_display") and not is_long_run(e):
        details.append(f"Repeats: {plain(meta['vibemap_event_recurrence_display'])}.")
    if meta.get("vibemap_event_event_status", "SCHEDULED") not in ("SCHEDULED", ""):
        details.append(f"Status: {meta['vibemap_event_event_status'].lower()}.")
    if details:
        lines.append(" ".join(details))
    description = plain(meta.get("vibemap_event_description") or e.get("content"))
    if description and description not in boilerplate:
        lines.append(description)
    return "\n".join(lines)


def save(events, out_dir):
    # merge occurrences of the same event, in date order
    groups = defaultdict(list)
    for e in sorted(events, key=lambda e: occurrence(e) or datetime.max):
        meta = e["meta"]
        key = (plain(e["title"]), meta.get("vibemap_event_venue_name"), plain(meta.get("vibemap_event_description")))
        if not any(o["slug"] == e["slug"] for o in groups[key]):
            groups[key].append(e)

    files = defaultdict(list)
    # descriptions shared by several different events are scraped page junk, not descriptions
    counts = Counter(plain(g[0]["meta"].get("vibemap_event_description") or g[0].get("content")) for g in groups.values())
    boilerplate = {d for d, n in counts.items() if d and n >= 3}
    for occurrences in groups.values():
        meta = occurrences[0]["meta"]
        first = occurrence(occurrences[0])
        name = "events-ongoing" if is_long_run(occurrences[0]) or not first else f"events-{first:%Y-%m}"
        files[name].append(block(occurrences, boilerplate))

    out = Path(out_dir) / "downtownpittsburgh.com"
    out.mkdir(parents=True, exist_ok=True)
    for name, blocks in sorted(files.items()):
        heading = "Downtown Pittsburgh events: " + (
            "ongoing exhibitions and long runs" if name == "events-ongoing"
            else f"{datetime.strptime(name[7:], '%Y-%m'):%B %Y}")
        body = "\n\n".join(blocks)
        (out / f"{name}.txt").write_text(
            f"URL: https://downtownpittsburgh.com/events/\n\n{heading}\n\n{body}\n", encoding="utf-8")
        print(f"saved   downtownpittsburgh.com/{name}.txt ({len(blocks)} events)")


if __name__ == "__main__":
    save(fetch_all(), "data/pages")

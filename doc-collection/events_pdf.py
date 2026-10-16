"""Parse pittsburgh.events month pages printed to PDF into one line per event.

Each event is a card with two columns:
    NOV  | Boney James
    03   | City Winery - Pittsburgh
    2026 | 1615 Smallman Street, Pittsburgh, PA 15222
    5:30 PM  VIEW TICKETS
Plain text extraction interleaves the columns ("7:23002 P6M"), so the words are
split into columns by x position and read top to bottom across all pages.
"""
import re
from datetime import datetime
from pathlib import Path

import pdfplumber

LEFT_COLUMN_X = 110  # date/time column is left of this; name/venue/address is right of it
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
ADDRESS = re.compile(r", [A-Z]{2}( \d{5})?$")  # "..., Pittsburgh, PA 15222" or "Pittsburgh, PA"


def _lines(words):
    """Group words (already sorted by y) into lines of text."""
    lines = []
    for w in words:
        if lines and abs(w["y"] - lines[-1]["y"]) < 1.5:  # year and time can sit 2.5pt apart
            lines[-1]["words"].append(w)
        else:
            lines.append({"y": w["y"], "words": [w]})
    return [(l["y"], " ".join(w["text"] for w in sorted(l["words"], key=lambda w: w["x0"]))) for l in lines]


def parse_events(pdf_path):
    words = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for w in page.extract_words(y_tolerance=1):
                words.append({**w, "y": i * float(page.height) + w["top"]})  # one long scroll
    words.sort(key=lambda w: (w["y"], w["x0"]))
    right = _lines([w for w in words if w["x0"] >= LEFT_COLUMN_X])
    left = _lines([w for w in words if w["x0"] < LEFT_COLUMN_X])

    # the schedule sits between this heading and the last "VIEW TICKETS"
    start = next(y for y, t in right + left if t.startswith("Browse upcoming events"))
    tickets = [y for y, t in right if t == "VIEW TICKETS"]
    end = tickets[-1]

    # right column: each card ends with VIEW TICKETS
    cards, cur = [], []
    for y, text in right:
        if not start < y <= end:
            continue
        if text == "VIEW TICKETS":
            cards.append(cur)
            cur = []
        else:
            cur.append(text)

    # left column: month, day, year, time
    dates, cur = [], {}
    for y, text in left:
        if not start < y <= end + 20:
            continue
        if text in MONTHS:
            cur = {"month": MONTHS[text]}
        elif re.fullmatch(r"\d{1,2}", text) and "month" in cur:
            cur["day"] = int(text)
        elif re.fullmatch(r"\d{4}", text) and "day" in cur:
            cur["year"] = int(text)
        elif (re.fullmatch(r"\d{1,2}:\d{2} [AP]M", text) or text == "Time TBA") and "year" in cur:
            cur["time"] = text
            dates.append(cur)
            cur = {}

    if len(cards) != len(dates):
        raise ValueError(f"{pdf_path}: {len(cards)} event cards but {len(dates)} dates")
    events = []
    for lines, d in zip(cards, dates):
        if len(lines) < 3 or not ADDRESS.search(lines[-1]):
            raise ValueError(f"{pdf_path}: can't read card {lines}")
        events.append({
            "name": " ".join(lines[:-2]),
            "venue": lines[-2],
            "address": lines[-1],
            "date": datetime(d["year"], d["month"], d["day"]),
            "time": d["time"],
        })
    return events


def event_line(e):
    """Boney James: Tuesday, November 3, 2026 at 5:30 PM, City Winery - Pittsburgh, 1615 Smallman ..."""
    when = f"{e['date']:%A, %B} {e['date'].day}, {e['date'].year}"
    if e["time"] != "Time TBA":
        when += f" at {e['time']}"
    return f"{e['name']}: {when}, {e['venue']}, {e['address']}"


def convert_event_pdfs(in_dir, out_dir):
    """Write each PDF as pittsburgh.events/<month>.txt, one event per line."""
    for pdf_path in sorted(Path(in_dir).glob("*.pdf")):
        events = parse_events(pdf_path)
        month = events[0]["date"]
        url = f"https://pittsburgh.events/{month:%B}/".lower()
        path = Path(out_dir) / "pittsburgh.events" / f"{month:%B}-{month.year}.txt".lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"Pittsburgh Events in {month:%B} {month.year}"
        path.write_text(f"URL: {url}\n\n{header}\n" + "\n".join(event_line(e) for e in events) + "\n",
                        encoding="utf-8")
        print(f"saved   {path.relative_to(out_dir)} ({len(events)} events) <- {pdf_path.name}")


if __name__ == "__main__":
    convert_event_pdfs("pittsburghevents", "data/pages")

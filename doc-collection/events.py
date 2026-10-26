"""Event calendars as text, one paragraph or line per event, with dates written out in full.

    python events.py [source ...]     sources: cmu downtown citypaper pdf improv mrsmalls (default: all)

Calendar pages don't work for retrieval as scraped: most are rendered with JavaScript, and the
ones that aren't put each field on its own line with abbreviated dates and no year ("Dec" / "20
Sun"), which a question like "...on December 20, 2026?" can't match. So each source is read from
its data feed or markup:
    cmu        events.cmu.edu: LiveWhale JSON feed, month by month
    downtown   downtownpittsburgh.com/events: the Vibemap WordPress plugin's JSON feed
    citypaper  Pittsburgh City Paper: result pages saved from a browser (the site blocks scripts;
               see citypaper_download.js), in raw_pages/citypaper
    pdf        pittsburgh.events: month pages printed to PDF, in raw_pages/pittsburghevents
    improv     improv.com/pittsburgh/calendar
    mrsmalls   mrsmalls.com/calendar (structured markup with the full start date)
An event listed on several days is one paragraph with all its dates (or "N dates from X through Y"
when there are more than MAX_LISTED_DATES).
"""
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

from common import MONTHS, PAGES, RAW, clock, day, get, plain, save_doc, soup_of, text

MAX_LISTED_DATES = 10


def date_range(days, times=(), unique=True):
    """'12 dates from Friday, October 2, 2026 through ...', with the time if every date shares it."""
    days = sorted(set(days)) if unique else sorted(days)
    line = f"{len(days)} dates from {day(days[0])} through {day(days[-1])}"
    times = set(times)
    return line + (f", {times.pop()}" if len(times) == 1 and "" not in times else "")


# --- events.cmu.edu --------------------------------------------------------------------------------

CMU_API = ("https://events.cmu.edu/live/json/events/start_date/{start}/end_date/{end}"
           "/response_fields/location,summary,description,group_title,event_types?page={page}")
CMU_HIDDEN_TYPES = {"Exclude from Main Calendar"}


def cmu_fetch_month(year, month, delay=1.0):
    start, end = date(year, month, 1), date(year + month // 12, month % 12 + 1, 1)
    events, page = [], 1
    while True:
        r = get(CMU_API.format(start=start, end=end, page=page), timeout=60)
        r.raise_for_status()
        data = r.json()
        events += [e for e in data["data"] if e["date_iso"][:7] == f"{year}-{month:02d}"]
        if page >= data["meta"]["total_pages"]:
            return events
        page += 1
        time.sleep(delay)


def cmu_when(e):
    """Thursday, October 15, 2026, 4:00 PM - 5:00 PM (or 'all day', or a date range)."""
    start = datetime.fromisoformat(e["date_iso"])
    end = datetime.fromisoformat(e["date2_iso"]) if e.get("date2_iso") else None
    if e.get("is_all_day"):
        if end and end.date() > start.date():
            return f"{day(start)} to {day(end)} (all day)"
        return f"{day(start)} (all day)"
    if end and end.date() > start.date():
        return f"{day(start)}, {clock(start)} to {day(end)}, {clock(end)}"
    return f"{day(start)}, {clock(start)}" + (f" - {clock(end)}" if end else "")


def cmu_block(occurrences):
    """One paragraph per event; recurring events list all their dates instead of repeating."""
    e = occurrences[0]
    first = f"{plain(e['title'])}: {'; '.join(cmu_when(o) for o in occurrences)}"
    if e.get("location"):
        first += f", {plain(e['location'])}"
    lines, details = [first], []
    if e.get("group_title"):
        details.append(f"Hosted by {plain(e['group_title'])}.")
    types = [plain(t) for t in e.get("event_types") or [] if plain(t) not in CMU_HIDDEN_TYPES]
    if types:
        details.append(f"Type: {', '.join(types)}.")
    if e.get("is_canceled"):
        details.append("This event is canceled.")
    if details:
        lines.append(" ".join(details))
    for field in ("summary", "description"):
        if plain(e.get(field)) and plain(e.get(field)) not in lines:
            lines.append(plain(e.get(field)))
    return "\n".join(lines)


def cmu(out_dir=PAGES, months=((2026, 10), (2026, 11), (2026, 12))):
    """Output: events.cmu.edu/<yyyy-mm>.txt"""
    for year, month in months:
        groups = defaultdict(list)  # occurrences with the same title, place and description
        for e in cmu_fetch_month(year, month):
            key = (plain(e["title"]), plain(e.get("location")), plain(e.get("summary")), plain(e.get("description")))
            groups[key].append(e)
        save_doc(Path(out_dir) / "events.cmu.edu" / f"{year}-{month:02d}.txt",
                 f"https://events.cmu.edu/month/date/{year}{month:02d}01",
                 f"CMU Events Calendar: {date(year, month, 1):%B %Y}",
                 [cmu_block(g) for g in groups.values()], sep="\n\n", unit="events")


# --- downtownpittsburgh.com ------------------------------------------------------------------------

DOWNTOWN_API = "https://downtownpittsburgh.com/wp-json/vibemap/v1/events-data?page={page}"
OCCURRENCE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")  # slug suffix, e.g. _2026-11-26-11-00


def downtown_fetch(delay=1.0):
    events, page = [], 1
    while True:
        r = get(DOWNTOWN_API.format(page=page), timeout=120)
        r.raise_for_status()
        data = r.json()
        events += data["events"]
        print(f"page {page}/{data['pagination']['total_pages']}")
        if not data["pagination"]["has_more"]:
            return events
        page += 1
        time.sleep(delay)


def parse_dt(value):
    return datetime.fromisoformat(value) if value else None


def occurrence(e):
    """Start of this occurrence of a recurring event: from the slug suffix, else the event's start."""
    m = OCCURRENCE.search(e["slug"])
    return datetime(*map(int, m.groups())) if m else parse_dt(e["meta"].get("vibemap_event_start_date"))


def is_long_run(e):
    """Exhibitions and shows that run for weeks: described by their date range."""
    meta = e["meta"]
    start, end = parse_dt(meta.get("vibemap_event_start_date")), parse_dt(meta.get("vibemap_event_end_date"))
    return meta.get("vibemap_event_is_ongoing") == "1" and start and end and (end - start).days >= 14


def downtown_when(e, start):
    """Saturday, October 24, 2026, 7:30 PM - 10:00 PM (time left out when it is midnight/unknown)."""
    end = parse_dt(e["meta"].get("vibemap_event_end_date"))
    line = day(start)
    if start.hour or start.minute:
        line += f", {clock(start)}"
        if end and end.date() == start.date() and end > start:
            line += f" - {clock(end)}"
    return line


def downtown_block(occurrences, boilerplate=()):
    e = occurrences[0]
    meta = e["meta"]
    venue = plain(meta.get("vibemap_event_venue_name") or meta.get("vibemap_event_hotspots_place"))
    address = plain(meta.get("vibemap_event_venue_address") or meta.get("vibemap_event_address"))
    if is_long_run(e):
        dates = (f"runs {day(parse_dt(meta['vibemap_event_start_date']))} through "
                 f"{day(parse_dt(meta['vibemap_event_end_date']))}")
    elif len(occurrences) > MAX_LISTED_DATES:  # e.g. a daily show: give the range, not every day
        starts = [occurrence(o) for o in occurrences]
        same_time = len({(d.hour, d.minute) for d in starts}) == 1 and (starts[0].hour or starts[0].minute)
        dates = date_range([d.date() for d in starts], [f"at {clock(starts[0])}"] if same_time else [""],
                           unique=False)
    else:
        dates = "; ".join(downtown_when(o, occurrence(o)) for o in occurrences)
    first = f"{plain(e['title'])}: {dates}"
    for part in (venue, address):
        if part and part not in first:
            first += f", {part}"
    lines, details = [first], []
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


def downtown(out_dir=PAGES):
    """Output: downtownpittsburgh.com/events-<yyyy-mm>.txt (by month of the first date) and
    events-ongoing.txt (exhibitions and other long runs)."""
    groups = defaultdict(list)  # occurrences of the same event, in date order
    for e in sorted(downtown_fetch(), key=lambda e: occurrence(e) or datetime.max):
        meta = e["meta"]
        key = (plain(e["title"]), meta.get("vibemap_event_venue_name"), plain(meta.get("vibemap_event_description")))
        if not any(o["slug"] == e["slug"] for o in groups[key]):
            groups[key].append(e)
    # descriptions shared by several different events are scraped page junk, not descriptions
    counts = Counter(plain(g[0]["meta"].get("vibemap_event_description") or g[0].get("content")) for g in groups.values())
    boilerplate = {d for d, n in counts.items() if d and n >= 3}
    files = defaultdict(list)
    for occurrences in groups.values():
        first = occurrence(occurrences[0])
        name = "events-ongoing" if is_long_run(occurrences[0]) or not first else f"events-{first:%Y-%m}"
        files[name].append(downtown_block(occurrences, boilerplate))
    for name, blocks in sorted(files.items()):
        heading = "Downtown Pittsburgh events: " + (
            "ongoing exhibitions and long runs" if name == "events-ongoing"
            else f"{datetime.strptime(name[7:], '%Y-%m'):%B %Y}")
        save_doc(Path(out_dir) / "downtownpittsburgh.com" / f"{name}.txt", "https://downtownpittsburgh.com/events/",
                 heading, blocks, sep="\n\n", unit="events")


# --- Pittsburgh City Paper (saved pages) -----------------------------------------------------------

CITYPAPER_YEARS = (2026, 2027)  # day headings have no year; pick the one whose weekday matches


def citypaper_day(heading):
    """'Friday, October 23' -> date(2026, 10, 23)."""
    weekday, rest = heading.split(", ", 1)
    for year in CITYPAPER_YEARS:
        d = datetime.strptime(f"{rest} {year}", "%B %d %Y").date()
        if f"{d:%A}" == weekday:
            return d
    raise ValueError(f"no year in {CITYPAPER_YEARS} matches {heading!r}")


def citypaper_page(path):
    """Yield one dict per event card under a day heading (the 'Promoted' card at the top is skipped).
    Each page has day headings ("Friday, October 23") followed by event cards."""
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")
    current = None
    # <main>: a new page starts (citypaper_download.js puts every page's <main> in one file)
    for el in soup.find_all(lambda t: t.name == "main"
                            or t.name == "h3" and "text-brand" in (t.get("class") or [])
                            or "rounded-xl" in (t.get("class") or []) and t.find("h3")):
        if el.name == "main":
            current = None
            continue
        if el.name == "h3":
            current = citypaper_day(text(el))
            continue
        link = el.find("h3").find("a", href=re.compile(r"/events/\d+/"))
        if current is None or link is None:  # promoted card, filter box
            continue
        body = el.find("div")  # left column: everything except the image
        ps = body.find_all("p", recursive=False)
        classes = lambda p: p.get("class") or []
        yield {
            "title": text(link),
            "day": current,
            "time": text(body.find("div").find("p")) if body.find("div") else "",
            "venue": next((text(p) for p in ps if "font-medium" in classes(p)), ""),
            "address": next((text(p) for p in ps if "line-clamp-1" in classes(p) and "font-medium" not in classes(p)), ""),
            "price": " ".join(text(s) for s in body.select("div.text-sm span")),
            "tags": [text(s) for s in body.select("span.rounded-full")],
            "description": next((text(p) for p in ps if "line-clamp-2" in classes(p)), ""),
        }


def citypaper_block(occurrences):
    e = occurrences[0]
    if len(occurrences) <= MAX_LISTED_DATES:
        dates = "; ".join(f"{day(o['day'])}, {o['time']}" if o["time"] else day(o["day"]) for o in occurrences)
    else:
        dates = date_range([o["day"] for o in occurrences], [o["time"] for o in occurrences])
    first = f"{e['title']}: {dates}"
    if e["venue"]:
        first += f", {e['venue']}"
    if e["address"]:
        first += f", {e['address']}"
    lines, details = [first], []
    if e["price"]:
        details.append(f"Price: {e['price']}.")
    if e["tags"]:
        details.append(f"Category: {', '.join(e['tags'])}.")
    if details:
        lines.append(" ".join(details))
    if e["description"]:
        lines.append(e["description"])
    return "\n".join(lines)


def citypaper(out_dir=PAGES, in_dir=RAW / "citypaper"):
    """Output: community.pghcitypaper.com/events-<yyyy-mm>.txt; an event listed on several days
    (or on several pages) is one paragraph with all its dates."""
    groups = defaultdict(dict)  # (title, venue, description) -> {(day, time): event}
    for f in sorted(Path(in_dir).glob("*.htm*")):
        for e in citypaper_page(f):
            groups[(e["title"], e["venue"], e["description"])][(e["day"], e["time"])] = e
    files = defaultdict(list)
    for occurrences in groups.values():
        occurrences = [occurrences[k] for k in sorted(occurrences)]
        files[f"{occurrences[0]['day']:%Y-%m}"].append(citypaper_block(occurrences))
    for month, blocks in sorted(files.items()):
        save_doc(Path(out_dir) / "community.pghcitypaper.com" / f"events-{month}.txt",
                 "https://community.pghcitypaper.com/events",
                 f"Pittsburgh City Paper events: {datetime.strptime(month, '%Y-%m'):%B %Y}",
                 blocks, sep="\n\n", unit="events")


# --- pittsburgh.events (PDFs) ----------------------------------------------------------------------

PDF_LEFT_COLUMN_X = 110  # date/time column is left of this; name/venue/address is right of it
PDF_ADDRESS = re.compile(r", [A-Z]{2}( \d{5})?$")  # "..., Pittsburgh, PA 15222" or "Pittsburgh, PA"


def pdf_lines(words):
    """Group words (already sorted by y) into lines of text."""
    lines = []
    for w in words:
        if lines and abs(w["y"] - lines[-1]["y"]) < 1.5:  # year and time can sit 2.5pt apart
            lines[-1]["words"].append(w)
        else:
            lines.append({"y": w["y"], "words": [w]})
    return [(l["y"], " ".join(w["text"] for w in sorted(l["words"], key=lambda w: w["x0"]))) for l in lines]


def pdf_events(pdf_path):
    """Each event is a two-column card: NOV / 03 / 2026 / 5:30 PM on the left, name / venue /
    address on the right. Plain text extraction interleaves the columns ("7:23002 P6M"), so words
    are split into columns by x position and read top to bottom across all pages."""
    words = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for w in page.extract_words(y_tolerance=1):
                words.append({**w, "y": i * float(page.height) + w["top"]})  # one long scroll
    words.sort(key=lambda w: (w["y"], w["x0"]))
    right = pdf_lines([w for w in words if w["x0"] >= PDF_LEFT_COLUMN_X])
    left = pdf_lines([w for w in words if w["x0"] < PDF_LEFT_COLUMN_X])
    # the schedule sits between this heading and the last "VIEW TICKETS"
    start = next(y for y, t in right + left if t.startswith("Browse upcoming events"))
    end = [y for y, t in right if t == "VIEW TICKETS"][-1]

    cards, cur = [], []  # right column: each card ends with VIEW TICKETS
    for y, line in right:
        if start < y <= end:
            if line == "VIEW TICKETS":
                cards.append(cur)
                cur = []
            else:
                cur.append(line)
    dates, cur = [], {}  # left column: month, day, year, time
    months = [m.upper() for m in MONTHS]
    for y, line in left:
        if not start < y <= end + 20:
            continue
        if line in months:
            cur = {"month": months.index(line) + 1}
        elif re.fullmatch(r"\d{1,2}", line) and "month" in cur:
            cur["day"] = int(line)
        elif re.fullmatch(r"\d{4}", line) and "day" in cur:
            cur["year"] = int(line)
        elif (re.fullmatch(r"\d{1,2}:\d{2} [AP]M", line) or line == "Time TBA") and "year" in cur:
            cur["time"] = line
            dates.append(cur)
            cur = {}
    if len(cards) != len(dates):
        raise ValueError(f"{pdf_path}: {len(cards)} event cards but {len(dates)} dates")
    events = []
    for lines, d in zip(cards, dates):
        if len(lines) < 3 or not PDF_ADDRESS.search(lines[-1]):
            raise ValueError(f"{pdf_path}: can't read card {lines}")
        events.append({"name": " ".join(lines[:-2]), "venue": lines[-2], "address": lines[-1],
                       "date": datetime(d["year"], d["month"], d["day"]), "time": d["time"]})
    return events


def pdf(out_dir=PAGES, in_dir=RAW / "pittsburghevents"):
    """Output: pittsburgh.events/<month>-<year>.txt, one line per event:
    Boney James: Tuesday, November 3, 2026 at 5:30 PM, City Winery - Pittsburgh, 1615 Smallman ..."""
    for pdf_path in sorted(Path(in_dir).glob("*.pdf")):
        events = pdf_events(pdf_path)
        month = events[0]["date"]
        lines = [f"{e['name']}: {day(e['date'])}" + (f" at {e['time']}" if e["time"] != "Time TBA" else "")
                 + f", {e['venue']}, {e['address']}" for e in events]
        save_doc(Path(out_dir) / "pittsburgh.events" / f"{month:%B}-{month.year}.txt".lower(),
                 f"https://pittsburgh.events/{month:%B}/".lower(), f"Pittsburgh Events in {month:%B} {month.year}",
                 lines, unit="events")


# --- Pittsburgh Improv -----------------------------------------------------------------------------

IMPROV_URL = "https://improv.com/pittsburgh/calendar/"
IMPROV_FIRST_YEAR = 2026  # the calendar starts in October 2026; January onward is 2027


def improv(out_dir=PAGES):
    """improv.com lists each show over several lines with an abbreviated month and no year:
    "Dec" / "20 Sun" / "<comedian>" / "7:00pm" (or "Oct" / "30 Fri" / "Nov" / "1 Sun" for a run).
    Output: improv.com/pittsburgh__calendar.txt, one line per show."""
    from scrape import html_to_text
    lines = html_to_text(get(IMPROV_URL).content)[0].splitlines()

    def full_date(month, d):
        m = MONTHS.index(month) + 1
        return day(date(IMPROV_FIRST_YEAR if m >= 10 else IMPROV_FIRST_YEAR + 1, m, d))

    shows, i = [], 0
    while i < len(lines) - 2:
        days = re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))? \w{3}(?:-\w{3})?", lines[i + 1])
        if lines[i] not in MONTHS or not days:
            i += 1
            continue
        start, j = full_date(lines[i], int(days.group(1))), i + 2
        end = full_date(lines[i], int(days.group(2))) if days.group(2) else None
        if lines[j] in MONTHS and re.fullmatch(r"\d{1,2} \w{3}", lines[j + 1]):  # a run across months
            end, j = full_date(lines[j], int(lines[j + 1].split()[0])), j + 2
        title, when = lines[j], lines[j + 1] if j + 1 < len(lines) else ""
        dates = f"from {start} through {end}" if end else f"on {start}"
        at = f" at {when}" if re.match(r"\d", when) and "show" not in when else (f" ({when})" if "show" in when else "")
        shows.append(f"Pittsburgh Improv comedy show: {title} performs {dates}{at}.")
        i = j + 1
    save_doc(Path(out_dir) / "improv.com" / "pittsburgh__calendar.txt", IMPROV_URL,
             "Pittsburgh Improv event calendar", shows, unit="shows")


# --- Mr. Smalls ------------------------------------------------------------------------------------

MRSMALLS_URL = "https://www.mrsmalls.com/calendar/"


def mrsmalls(out_dir=PAGES):
    """Each show is a div.event-wrapper with the title in its <header> and the date in
    <p itemprop="startDate" content="2026-10-29 20:00">; the page text alone says only "Thu Oct 29".
    Output: www.mrsmalls.com/calendar.txt, one line per show."""
    shows = []
    for event in soup_of(MRSMALLS_URL).select("div.event-wrapper"):
        start, title = event.find(attrs={"itemprop": "startDate"}), text(event.select_one(".show-title"))
        if not start or not title:
            continue
        when = datetime.strptime(start["content"], "%Y-%m-%d %H:%M")
        presenter, extra = text(event.select_one(".topline")), text(event.select_one(".bottomline"))
        venue = text(event.select_one(".venue-location-name")) or "Mr. Smalls"
        line = f"{venue} (Mr. Smalls, Millvale): {title} on {day(when)}, {text(event.select_one('.time-show'))}."
        if presenter:
            line += f" Presented by {presenter.removesuffix(' Presents').removesuffix(' presents')}."
        if extra:
            line += f" {extra}"
        if "Canceled" in event.get_text():
            line += " (Canceled)"
        shows.append(line)
    save_doc(Path(out_dir) / "www.mrsmalls.com" / "calendar.txt", MRSMALLS_URL,
             "Mr. Smalls concert calendar (Mr. Smalls Theatre, The Funhouse at Mr. Smalls, Mr. Smalls Cafe)",
             shows, unit="shows")


SOURCES = {"cmu": cmu, "downtown": downtown, "citypaper": citypaper, "pdf": pdf, "improv": improv,
           "mrsmalls": mrsmalls}

if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        if name not in SOURCES:
            sys.exit(__doc__)
        SOURCES[name]()

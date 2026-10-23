"""Convert Pittsburgh City Paper event listing pages, saved from a browser, into text.

community.pghcitypaper.com blocks scripts (Cloudflare), so the result pages
    https://community.pghcitypaper.com/events?to=2026-12-31&from=2026-10-23&page=N
are saved by hand into one folder. Each page has day headings ("Friday, October 23") followed by
event cards (time, title, venue, address, price, category, description).
Output: community.pghcitypaper.com/events-<yyyy-mm>.txt, one paragraph per event; an event
listed on several days (or on several pages) is one paragraph with all its dates.
"""
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

YEARS = (2026, 2027)  # day headings have no year; pick the one whose weekday matches
MAX_LISTED_DATES = 10


def text(el):
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el else ""


def parse_day(heading):
    """'Friday, October 23' -> date(2026, 10, 23)."""
    weekday, rest = heading.split(", ", 1)
    for year in YEARS:
        d = datetime.strptime(f"{rest} {year}", "%B %d %Y").date()
        if f"{d:%A}" == weekday:
            return d
    raise ValueError(f"no year in {YEARS} matches {heading!r}")


def parse_page(path):
    """Yield one dict per event card under a day heading (the 'Promoted' card at the top is skipped)."""
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")
    day = None
    # <main>: a new page starts (citypaper_download.js puts every page's <main> in one file)
    for el in soup.find_all(lambda t: t.name == "main"
                            or t.name == "h3" and "text-brand" in (t.get("class") or [])
                            or "rounded-xl" in (t.get("class") or []) and t.find("h3")):
        if el.name == "main":
            day = None
            continue
        if el.name == "h3":
            day = parse_day(text(el))
            continue
        link = el.find("h3").find("a", href=re.compile(r"/events/\d+/"))
        if day is None or link is None:  # promoted card, filter box
            continue
        body = el.find("div")  # left column: everything except the image
        ps = [p for p in body.find_all("p", recursive=False)]
        classes = lambda p: p.get("class") or []
        venue = next((text(p) for p in ps if "font-medium" in classes(p)), "")
        address = next((text(p) for p in ps if "line-clamp-1" in classes(p) and "font-medium" not in classes(p)), "")
        description = next((text(p) for p in ps if "line-clamp-2" in classes(p)), "")
        price = [text(s) for s in body.select("div.text-sm span")]
        tags = [text(s) for s in body.select("span.rounded-full")]
        yield {
            "id": re.search(r"/events/(\d+)/", link["href"]).group(1),
            "url": link["href"],
            "title": text(link),
            "day": day,
            "time": text(body.find("div").find("p")) if body.find("div") else "",
            "venue": venue,
            "address": address,
            "price": " ".join(price),
            "tags": tags,
            "description": description,
        }


def when(e):
    d = e["day"]
    s = f"{d:%A, %B} {d.day}, {d.year}"
    return f"{s}, {e['time']}" if e["time"] else s


def dates(occurrences):
    """All dates, or for a daily/weekly event listed many times, the range and number of dates."""
    if len(occurrences) <= MAX_LISTED_DATES:
        return "; ".join(when(o) for o in occurrences)
    days = sorted({o["day"] for o in occurrences})
    first, last = (f"{d:%A, %B} {d.day}, {d.year}" for d in (days[0], days[-1]))
    text = f"{len(days)} dates from {first} through {last}"
    times = {o["time"] for o in occurrences}
    return text + (f", {times.pop()}" if len(times) == 1 and "" not in times else "")


def block(occurrences):
    e = occurrences[0]
    first = f"{e['title']}: {dates(occurrences)}"
    if e["venue"]:
        first += f", {e['venue']}"
    if e["address"]:
        first += f", {e['address']}"
    lines = [first]
    details = []
    if e["price"]:
        details.append(f"Price: {e['price']}.")
    if e["tags"]:
        details.append(f"Category: {', '.join(e['tags'])}.")
    if details:
        lines.append(" ".join(details))
    if e["description"]:
        lines.append(e["description"])
    return "\n".join(lines)


def convert(in_dir, out_dir):
    groups = defaultdict(dict)  # (title, venue, description) -> {day+time: event}
    n_pages = 0
    for f in sorted(Path(in_dir).glob("*.htm*")):
        n_pages += 1
        for e in parse_page(f):
            groups[(e["title"], e["venue"], e["description"])][(e["day"], e["time"])] = e
    files = defaultdict(list)
    for occ in groups.values():
        occ = [occ[k] for k in sorted(occ)]
        files[f"{occ[0]['day']:%Y-%m}"].append(block(occ))
    out = Path(out_dir) / "community.pghcitypaper.com"
    out.mkdir(parents=True, exist_ok=True)
    for month, blocks in sorted(files.items()):
        name = f"{datetime.strptime(month, '%Y-%m'):%B %Y}"
        (out / f"events-{month}.txt").write_text(
            f"URL: https://community.pghcitypaper.com/events\n\nPittsburgh City Paper events: {name}\n\n"
            + "\n\n".join(blocks) + "\n", encoding="utf-8")
        print(f"saved   community.pghcitypaper.com/events-{month}.txt ({len(blocks)} events)")
    print(f"{n_pages} pages, {len(groups)} events")


if __name__ == "__main__":
    convert("citypaper", "data/pages")

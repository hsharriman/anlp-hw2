"""Download the Carnegie Museum of Art's paintings (collection.carnegieart.org) as text.

The collection site is rendered with JavaScript. Its pages load artworks from the museum's
Elasticsearch server, using a read-only login that is built into the site's own JavaScript:
    POST https://collection.carnegieart.org/api/cmoa_objects/object/_search
Only paintings are fetched (about 1,500 of the 90,000 objects; most of the rest are photographs).
Output: collection.carnegieart.org/paintings-<NN>.txt, one paragraph per painting, sorted by title.
"""
import base64
import time
from pathlib import Path

import requests

from common import HEADERS, PAGES, save_doc

SEARCH = "https://collection.carnegieart.org/api/cmoa_objects/object/_search"
AUTH = "Basic " + base64.b64encode(b"collections:o41KG!MmJQ$A66").decode()  # the site's public read-only login
SEARCH_HEADERS = {**HEADERS, "Authorization": AUTH, "Content-Type": "application/json"}
PER_FILE = 100


def fetch(medium="paintings", page_size=500, delay=2.0):
    objects, start = [], 0
    while True:
        body = {"from": start, "size": page_size, "track_total_hits": True,
                "query": {"term": {"medium": medium}}, "sort": [{"accession_number": "asc"}]}
        r = requests.post(SEARCH, headers=SEARCH_HEADERS, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()["hits"]
        total = data["total"]["value"] if isinstance(data["total"], dict) else data["total"]  # older servers: a number
        objects += [h["_source"] for h in data["hits"]]
        print(f"{len(objects)}/{total}")
        if not data["hits"] or len(objects) >= total:
            return objects
        start += page_size
        time.sleep(delay)


def creator(c):
    """Edward Hopper (American, 1882-1967)"""
    details = [c.get("nationality"), "-".join(x for x in (c.get("birth"), c.get("death")) if x)]
    details = [d for d in details if d]
    name = c.get("label") or c.get("name")
    return f"{name} ({', '.join(details)})" if details else name


def size(o):
    work = next((m for m in o.get("measurements") or [] if m.get("type") == "work"), None)
    if not work or not work["fraction"].get("height"):
        return ""
    f = work["fraction"]
    return " x ".join(f"{f[k]}" for k in ("height", "width", "depth") if f.get(k)) + " inches"


def block(o):
    title = o["title"]
    creators = "; ".join(creator(c) for c in o.get("creators") or []) or "unknown artist"
    dates = o.get("creation_date") or []
    date = dates[0] if dates and dates[0] == dates[-1] else "-".join(d for d in dates if d)
    first = f"{title}, a painting by {creators}" + (f", {date}" if date else "")
    if o.get("medium_description"):
        first += f", {o['medium_description']}"
    facts = [
        ("Accession number", o.get("accession_number")),
        ("Credit line", o.get("credit_line")),
        ("Acquisition date", o.get("acquisition_date")),
        ("Acquisition method", o.get("acqisition_method")),  # sic: the field name is misspelled in the data
        ("Department", o.get("department")),
        ("Current location", o.get("current_location")),
        ("Dimensions", size(o)),
    ]
    lines = [f"Carnegie Museum of Art collection: {first}."]
    lines.append(" ".join(f"{k}: {v}." for k, v in facts if v))
    uuid = o["id"].split("/")[-1]
    lines.append(f"https://collection.carnegieart.org/objects/{uuid}")
    return "\n".join(lines)


def save(objects, out_dir=PAGES):
    objects = sorted(objects, key=lambda o: o["title"].lower())
    for i in range(0, len(objects), PER_FILE):
        part = objects[i:i + PER_FILE]
        n = i // PER_FILE + 1
        heading = (f"Carnegie Museum of Art paintings, part {n}: "
                   f"\"{part[0]['title'][:40]}\" to \"{part[-1]['title'][:40]}\"")
        save_doc(Path(out_dir) / "collection.carnegieart.org" / f"paintings-{n:02d}.txt",
                 "https://collection.carnegieart.org/", heading, [block(o) for o in part], sep="\n\n",
                 unit="paintings")


if __name__ == "__main__":
    save(fetch())

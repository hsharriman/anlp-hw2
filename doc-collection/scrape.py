import io
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup, NavigableString

HEADERS = {"User-Agent": "Mozilla/5.0 (CMU ANLP course project crawler)"}

# login, ticket and shop pages, matched against the host and path
# e.g. shop.steelers.com, /tickets/, /my-account
SKIP_URL = re.compile(
    r"(^|[/.])(log-?in|log-?out|sign-?in|sign-?up|register|account|my-account|cart|checkout|"
    r"tickets?|shop|store)([/._?-]|$)"
)
# with sports=True, also news/scores/stats pages and everything under them
# e.g. /news/2026/..., /scores/, /stats/leaders
SPORTS_SKIP_URL = re.compile(r"(^|[/.])(news|scores?|stats|statistics)([/._?-]|$)")


def should_skip(url, sports=False):
    p = urlparse(url)
    target = f"{p.netloc}{p.path}".lower()
    return bool(SKIP_URL.search(target) or (sports and SPORTS_SKIP_URL.search(target)))


def filename_for(url, is_pdf):
    """https://www.cmu.edu/about/history.html -> www.cmu.edu/about__history.html.txt"""
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    if p.query:
        path += "_" + p.query
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.replace("/", "__"))[:150]
    if is_pdf and not name.lower().endswith(".pdf"):
        name += ".pdf"
    return Path(p.netloc) / (name + ".txt")


BLOCK_TAGS = ["p", "div", "section", "article", "main", "aside", "li", "ul", "ol", "table", "tr",
              "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "blockquote", "figure", "figcaption", "title"]


def html_to_text(html, selector=None):
    """Visible text, one line per block element; inline tags (links, bold) stay in their sentence.
    `selector` (CSS) limits the text to the largest matching element, e.g. the article body."""
    soup = BeautifulSoup(html, "html.parser")
    # not <form>: ASP.NET sites (e.g. pittsburghpa.gov) wrap the whole page in one
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg"]):
        tag.decompose()
    root = soup
    if selector:
        matches = soup.select(selector)
        if matches:
            root = max(matches, key=lambda el: len(el.get_text(" ").split()))
    # newlines in the HTML source are just spaces; line breaks come from block elements
    for s in root.find_all(string=True):
        if type(s) is NavigableString and s.find_parent("pre") is None:  # not doctype or comments
            s.replace_with(re.sub(r"\s+", " ", s))
    for br in root.find_all("br"):
        br.replace_with("\n")
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for cell in root.find_all(["td", "th"]):
        cell.insert_after(" ")
    lines = (line.strip() for line in root.get_text().splitlines())
    return "\n".join(line for line in lines if line), soup


def convert_saved_html(in_dir, out_dir, selector=None):
    """Convert pages saved from a browser (for sites that block scripts) into the same
    <host>/<path>.txt files that scrape() writes. The URL comes from the page's canonical link."""
    out_dir = Path(out_dir)
    for f in sorted(Path(in_dir).glob("*.htm*")):
        raw = f.read_bytes()
        head = BeautifulSoup(raw, "html.parser")
        canonical = head.find("link", rel="canonical") or head.find("meta", property="og:url")
        url = (canonical.get("href") or canonical.get("content")) if canonical else f"file://{f.name}"
        text, _ = html_to_text(raw, selector)
        text = re.sub(r" *\[[A-Z][A-Z0-9-]*\] *", " ", text)  # ad-slot markers, e.g. Britannica's [MOD7]
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        path = out_dir / filename_for(url, False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
        print(f"saved   {path.relative_to(out_dir)} ({len(text.split())} words) <- {f.name}")


def pdf_to_text(data):
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def scrape(start_url, out_dir, max_depth=1, max_pages=50, same_domain=True, delay=1.0, min_words=100,
           sports=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = urlparse(start_url).netloc
    queue = deque([(start_url, 0)])
    seen = {start_url}
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.popleft()
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"error   {url}: {e}")
            continue
        time.sleep(delay)
        if response.status_code != 200:
            print(f"{response.status_code}     {url}")
            continue

        content_type = response.headers.get("Content-Type", "")
        is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
        links = []
        if is_pdf:
            try:
                text = pdf_to_text(response.content)
            except Exception as e:
                print(f"bad pdf {url}: {e}")
                continue
        elif "html" in content_type:
            text, soup = html_to_text(response.content)
            links = [a["href"] for a in soup.find_all("a", href=True)]
        else:
            continue  # images, zip files, etc.

        n_words = len(text.split())
        if n_words < min_words:
            print(f"thin    {url} ({n_words} words)")  # not saved, but its links are still followed
        else:
            path = out_dir / filename_for(url, is_pdf)
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
            saved += 1
            print(f"saved   {path.relative_to(out_dir)}")

        if depth >= max_depth:
            continue
        for href in links:
            link, _ = urldefrag(urljoin(url, href))
            if not link.startswith(("http://", "https://")) or link in seen or should_skip(link, sports):
                continue
            if same_domain and urlparse(link).netloc != domain and not link.lower().endswith(".pdf"):
                continue
            seen.add(link)
            queue.append((link, depth + 1))

    print(f"done: {saved} files in {out_dir}")


if __name__ == "__main__":
    start_url = "https://www.picklesburgh.com/"
    out_dir = "data/pages"
    scrape(start_url, out_dir, max_depth=1, max_pages=100, sports=False)  # sports=True for team sites

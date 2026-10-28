"""Generic page collection: crawling, HTML/PDF to text, and pages saved from a browser.

    python scrape.py seeds [category ...]    crawl the seed URLs in sources.json (1 link deep)
    python scrape.py deep [host ...]         the DEEP_SITES, 2 links deep
    python scrape.py sitemap <sitemap index URL> <section> ...
                                             every page in a site's XML sitemaps, e.g. Visit Pittsburgh:
                                             https://www.visitpittsburgh.com/sitemaps-1-sitemap.xml
                                             event partnerDirectory pages landingPages blogDirectory
    python scrape.py wiki-lists              Wikipedia articles linked from Pittsburgh list pages
    python scrape.py social                  each collected site's social media accounts
    python scrape.py saved-html <folder> [CSS selector]
                                             pages saved from a browser; Britannica used
                                             raw_pages/brittanica ".topic-content, .font-serif.font-18"
                                             (the article body), Little Italy Days raw_pages/littleitaly

Output: data/pages/<host>/<path>.txt, e.g. www.cmu.edu/about__history.html.txt.
"""
import io
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pdfplumber
import requests
from bs4 import BeautifulSoup, NavigableString

from common import HEADERS, HERE, PAGES, get, save_doc

# login, ticket and shop pages, matched against the host and path
# e.g. shop.steelers.com, /tickets/, /my-account
SKIP_URL = re.compile(
    r"(^|[/.])(log-?in|log-?out|sign-?in|sign-?up|register|account|my-account|cart|checkout|"
    r"tickets?|shop|store)([/._?-]|$)"
)
# with sports=True, also news/scores/stats pages and everything under them
# e.g. /news/2026/..., /scores/, /stats/leaders
SPORTS_SKIP_URL = re.compile(r"(^|[/.])(news|scores?|stats|statistics)([/._?-]|$)")
BLOCK_TAGS = ["p", "div", "section", "article", "main", "aside", "li", "ul", "ol", "table", "tr",
              "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "blockquote", "figure", "figcaption", "title"]


def should_skip(url, sports=False):
    p = urlparse(url)
    target = f"{p.netloc}{p.path}".lower()
    return bool(SKIP_URL.search(target) or (sports and SPORTS_SKIP_URL.search(target)))


def filename_for(url, is_pdf=False):
    """https://www.cmu.edu/about/history.html -> www.cmu.edu/about__history.html.txt"""
    p = urlparse(url)
    path = p.path.strip("/") or "index"
    if p.query:
        path += "_" + p.query
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.replace("/", "__"))[:150]
    if is_pdf and not name.lower().endswith(".pdf"):
        name += ".pdf"
    return Path(p.netloc) / (name + ".txt")


def save_page(out_dir, url, text, is_pdf=False):
    path = Path(out_dir) / filename_for(url, is_pdf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
    return path


def html_to_text(html, selector=None):
    """Visible text, one line per block element; inline tags (links, bold) stay in their sentence.
    `selector` (CSS) limits the text to the largest matching element, e.g. the article body."""
    soup = BeautifulSoup(html, "html.parser")
    # not <form>: ASP.NET sites (e.g. pittsburghpa.gov) wrap the whole page in one
    for tag in soup(["script", "style", "noscript", "nav", "svg"]):
        tag.decompose()
    # page-wide headers/footers are menus and link lists; a <header> inside the content (an event's
    # or article's title, e.g. each show on mrsmalls.com/calendar) has only a link or two
    for tag in soup(["header", "footer"]):
        if not tag.decomposed and len(tag.find_all("a")) >= 5:
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


def pdf_to_text(data):
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


# --- link crawling ---------------------------------------------------------------------------------

def scrape(start_url, out_dir=PAGES, max_depth=1, max_pages=50, same_domain=True, delay=1.0, min_words=100,
           sports=False, prefix=None, robots=None):
    """Breadth-first crawl from start_url, saving up to max_pages pages of at least min_words words.
    Links are followed to max_depth levels, on the same site (PDFs anywhere); with `prefix`, only
    under that path (e.g. /penguins/ on nhl.com). `robots`: a RobotFileParser whose rules are obeyed."""
    out_dir = Path(out_dir)
    domain = urlparse(start_url).netloc
    queue = deque([(start_url, 0)])
    seen = {start_url}
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.popleft()
        if robots and not robots.can_fetch(HEADERS["User-Agent"], url):
            print(f"robots  {url}")
            continue
        try:
            response = get(url)
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
            text, _ = html_to_text(response.content)
            # links from the full page: html_to_text drops <nav> and page headers, but menu links are worth following
            links = [a["href"] for a in BeautifulSoup(response.content, "html.parser").find_all("a", href=True)]
        else:
            continue  # images, zip files, etc.

        n_words = len(text.split())
        if n_words < min_words:
            print(f"thin    {url} ({n_words} words)")  # not saved, but its links are still followed
        elif robots and (out_dir / filename_for(url, is_pdf)).exists():
            pass  # deep crawl: already collected, keep the earlier copy
        else:
            path = save_page(out_dir, url, text, is_pdf)
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
            if prefix and not urlparse(link).path.startswith(prefix):
                continue
            if robots and urlparse(link).query:  # deep crawl: calendar filters repeat the event pages
                continue
            seen.add(link)
            queue.append((link, depth + 1))

    print(f"done: {saved} files in {out_dir}")


def crawl_seeds(categories=(), out_dir=PAGES):
    """The seed URLs in sources.json (the sites the handout recommends): each seed page and the pages
    it links to on the same site; team sites skip news/scores/stats pages."""
    for source in json.load(open(HERE / "sources.json")):
        if not categories or source["category"] in categories:
            scrape(source["url"], out_dir, max_depth=1, max_pages=50, sports=source["category"] == "sports")


# Recommended sites the 1-link seed crawl covered thinly: their about, visit, FAQ, contact, board,
# careers and promotions pages are 2 links from the home page, and many are short (a phone number,
# parking or a dress code), so the deep crawl keeps pages of 30+ words. Event calendars saved from a
# browser (pittsburgh.events, City Paper, events.cmu.edu) block scripts and are not crawled.
# (start URL, max pages, path prefix, sports)
DEEP_SITES = [
    ("https://www.cmu.edu/", 300, None, False),
    ("https://trustarts.org/", 200, None, False),
    ("https://carnegiemuseums.org/", 150, None, False),
    ("https://carnegiemnh.org/", 150, None, False),
    ("https://carnegieart.org/", 150, None, False),
    ("https://carnegiesciencecenter.org/", 150, None, False),
    ("https://www.warhol.org/", 100, None, False),
    ("https://www.heinzhistorycenter.org/", 200, None, False),
    ("https://www.thefrickpittsburgh.org/", 200, None, False),
    ("https://pittsburghopera.org/", 200, None, False),
    ("https://www.pittsburghsymphony.org/", 200, None, False),
    ("https://www.pittsburghpa.gov/Home", 200, None, False),
    ("https://www.nhl.com/penguins/", 150, "/penguins/", True),
    ("https://www.mlb.com/pirates", 150, "/pirates", True),
    ("https://www.steelers.com/", 150, None, True),
    ("https://littleitalydays.com/", 50, None, False),
    ("https://www.picklesburgh.com/", 50, None, False),
    ("https://www.pghtacofest.com/", 50, None, False),
    ("https://pittsburghrestaurantweek.com/", 50, None, False),
    ("https://bananasplitfest.com/", 50, None, False),
]


def crawl_deep(start_url, max_pages, prefix=None, sports=False, out_dir=PAGES):
    """One DEEP_SITES entry: 2 links deep, pages of 30+ words, obeying the site's robots.txt."""
    # fetched with our own User-Agent: RobotFileParser.read() uses urllib's, which some sites refuse
    # with a 403, and a 401/403 makes it disallow everything
    robots = RobotFileParser()
    response = get(urljoin(start_url, "/robots.txt"))
    if response.status_code in (401, 403):
        print(f"skip    {start_url}: robots.txt refused ({response.status_code})")
        return
    robots.parse(response.text.splitlines() if response.status_code == 200 else [])
    delay = max(1.0, float(robots.crawl_delay(HEADERS["User-Agent"]) or 0))
    scrape(start_url, out_dir, max_depth=2, max_pages=max_pages, min_words=30, sports=sports,
           prefix=prefix, robots=robots, delay=delay)


# --- sitemaps --------------------------------------------------------------------------------------

def sitemap_urls(url):
    return re.findall(r"<loc>([^<]+)</loc>", get(url).text)


def crawl_sitemap(index_url, sections, out_dir=PAGES, delay=0.5):
    """Every page a site lists in its XML sitemaps (sections chosen by a filter on the sitemap name),
    instead of following links. Used for visitpittsburgh.com, whose event, directory (restaurants,
    venues, museums, organizations) and blog pages are mostly not reachable within a shallow link
    crawl. Pages already saved are skipped, so it can be re-run to resume."""
    sitemaps = [s for s in sitemap_urls(index_url) if any(f"-{sec}-" in s for sec in sections)]
    urls = list(dict.fromkeys(u for s in sitemaps for u in sitemap_urls(s)))
    print(f"{len(sitemaps)} sitemaps, {len(urls)} pages", flush=True)
    saved = 0
    for n, url in enumerate(urls, 1):
        if (Path(out_dir) / filename_for(url)).exists() or should_skip(url):
            continue
        try:
            response = get(url)
        except requests.RequestException as e:
            print(f"error   {url}: {e}")
            continue
        time.sleep(delay)
        if response.status_code != 200 or "html" not in response.headers.get("Content-Type", ""):
            print(f"{response.status_code}     {url}")
            continue
        save_page(out_dir, url, html_to_text(response.content)[0])
        saved += 1
        if n % 100 == 0:
            print(f"{n}/{len(urls)} ({saved} saved)", flush=True)
    print(f"done: {saved} new pages")


# --- Wikipedia list pages --------------------------------------------------------------------------

WIKI_LISTS = [
    "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
    "https://en.wikipedia.org/wiki/List_of_bridges_of_Pittsburgh",
    "https://en.wikipedia.org/wiki/List_of_Pittsburgh_History_and_Landmarks_Foundation_Historic_Landmarks",
    "https://en.wikipedia.org/wiki/Sports_in_Pittsburgh",
    "https://en.wikipedia.org/wiki/List_of_colleges_and_universities_in_Pittsburgh",
]
WIKI_ARTICLES = [
    "https://en.wikipedia.org/wiki/Pittsburgh_Vintage_Grand_Prix",
    "https://en.wikipedia.org/wiki/Heinz_Chapel_Choir",
]
MAX_PER_LIST = 150


def wiki_article_links(url):
    """Article links in the page body (no File:, Help:, edit links, ...)."""
    body = BeautifulSoup(get(url).content, "html.parser")
    body = body.select_one("#mw-content-text") or body
    links = [urljoin(url, a["href"].split("#")[0]) for a in body.find_all("a", href=True)
             if re.fullmatch(r"(https://en\.wikipedia\.org)?/wiki/[^:?]+", a["href"].split("#")[0])
             and "Main_Page" not in a["href"]]
    return list(dict.fromkeys(links))[:MAX_PER_LIST]


def crawl_wiki_lists(out_dir=PAGES):
    """The list pages, the articles they link to, and a few single articles. List pages also link
    general topics (architects, U.S. states), so articles that mention Pittsburgh/Allegheny fewer
    than 3 times are not saved."""
    urls = list(WIKI_ARTICLES)
    for page in WIKI_LISTS:
        urls += [page] + wiki_article_links(page)
    saved = 0
    for url in dict.fromkeys(urls):
        if (Path(out_dir) / filename_for(url)).exists():
            continue
        response = get(url)
        time.sleep(0.5)
        if response.status_code != 200:
            print(f"{response.status_code}     {url}")
            continue
        text, _ = html_to_text(response.content)
        if len(text.split()) >= 100 and len(re.findall(r"Pittsburgh|Allegheny", text)) >= 3:
            save_page(out_dir, unquote(response.url), text)
            saved += 1
    print(f"{len(urls)} articles, {saved} new")


# --- social media accounts -------------------------------------------------------------------------

# athletics home pages embed one team's feed, not the school's accounts
SOCIAL_SKIP_HOSTS = {"gochathamcougars.com", "rmucolonials.com", "goduquesne.com", "pittsburghpanthers.com",
                     "en.wikipedia.org", "www.britannica.com", "community.pghcitypaper.com",
                     "assets.culturaldistrict.org", "collection.carnegieart.org", "pittsburgh.events",
                     "downtownpittsburgh.com", "events.cmu.edu"}
SOCIAL_EXTRA_SITES = ["https://www.photoantiquities.org/", "https://thewestmoreland.org/",
                      "https://www.pittsburghzoo.org/", "https://www.phipps.conservatory.org/",
                      "https://www.clementemuseum.com/"]
PLATFORMS = {  # host pattern -> platform name
    r"instagram\.com": "Instagram", r"(youtube\.com|youtu\.be)": "YouTube", r"facebook\.com": "Facebook",
    r"(twitter\.com|(^|\.)x\.com)": "X (Twitter)", r"tiktok\.com": "TikTok", r"linkedin\.com": "LinkedIn",
    r"threads\.(net|com)": "Threads", r"pinterest\.com": "Pinterest", r"bsky\.app": "Bluesky",
}


def social_handle(url):
    """https://www.instagram.com/mattressfactory/ -> @mattressfactory (None for share links)"""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts or parts[0] in ("share", "sharer", "intent", "sharer.php", "dialog", "hashtag", "p", "watch"):
        return None
    if parts[0] in ("channel", "c", "user", "profile", "company", "pages") and len(parts) > 1:
        parts = parts[1:]  # youtube.com/channel/X, bsky.app/profile/X, linkedin.com/company/X
    if parts[0] in ("events", "groups", "profile.php"):
        return None
    return parts[0] if parts[0].startswith("@") else "@" + parts[0]


def social_accounts(html):
    found = {}
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        host = urlparse(a["href"]).netloc.lower()
        for pattern, platform in PLATFORMS.items():
            if re.search(pattern, host) and platform not in found and social_handle(a["href"]):
                found[platform] = (social_handle(a["href"]), a["href"].split("?")[0])
    return found


def collect_social(out_dir=PAGES):
    """Each organization's social media accounts, from its home page. html_to_text drops page
    footers, which is where sites link their accounts, so no scraped page contains them.
    Output: <host>/social-media.txt"""
    sites = [f"https://{d.name}/" for d in sorted(PAGES.iterdir()) if d.is_dir() and d.name not in SOCIAL_SKIP_HOSTS]
    for url in sites + SOCIAL_EXTRA_SITES:
        try:
            response = get(url)
        except requests.RequestException as e:
            print(f"error   {url}: {e}")
            continue
        time.sleep(0.5)
        found = social_accounts(response.content) if response.status_code == 200 else {}
        if not found:
            print(f"none    {url} ({response.status_code})")
            continue
        host = urlparse(response.url).netloc
        title = BeautifulSoup(response.content, "html.parser").title
        site = re.split(r"\s[|\-–]\s", title.get_text(" ", strip=True) if title else host)[0] or host
        handles = ", ".join(f"{p} {h}" for p, (h, _) in found.items())
        save_doc(Path(out_dir) / host / "social-media.txt", response.url, f"{site} social media accounts",
                 [f"{site} ({host}) social media accounts and usernames: {handles}."]
                 + [f"{platform}: {h} ({link})" for platform, (h, link) in found.items()], unit="accounts")


# --- pages saved from a browser --------------------------------------------------------------------

def convert_saved_html(in_dir, out_dir=PAGES, selector=None):
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
        path = save_page(out_dir, url, text)
        print(f"saved   {path.relative_to(out_dir)} ({len(text.split())} words) <- {f.name}")


if __name__ == "__main__":
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if command == "seeds":
        crawl_seeds(args)
    elif command == "deep":  # python scrape.py deep [host substring ...]
        for url, n, prefix, sports in DEEP_SITES:
            if not args or any(a in url for a in args):
                crawl_deep(url, n, prefix, sports)
    elif command == "sitemap":
        crawl_sitemap(args[0], args[1:])
    elif command == "wiki-lists":
        crawl_wiki_lists()
    elif command == "social":
        collect_social()
    elif command == "saved-html":
        convert_saved_html(args[0], selector=args[1] if len(args) > 1 else None)
    else:
        sys.exit(__doc__)

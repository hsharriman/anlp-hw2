"""Clean, deduplicate and chunk the scraped pages.

Input:  doc-collection/data/pages/<host>/<name>.txt  (first line "URL: ...", then a blank line, then text)
Output: rag/data/docs.jsonl    {"doc_id", "url", "title", "text"}   one line per kept document
        rag/data/chunks.jsonl  {"doc_id", "chunk_id", "text"}       one line per chunk

doc_id is the file's path under pages/ without ".txt", e.g. "www.mlb.com/pirates__roster-2026".
Chunks store only doc_id; the title is looked up in docs.jsonl when building the indexes.

Steps:
1. Boilerplate: a line that appears on at least 30% of a site's pages (and on at least 3 of them)
   is navigation, cookie banners, ticket forms etc. and is removed. Only sites with 5+ pages.
2. Junk pages: Wikipedia edit pages, and pages with fewer than MIN_WORDS words left.
3. Duplicates: identical text (after normalizing whitespace and case), and near-duplicates
   (90%+ of a page's word 5-grams also appear in a longer page from the same site).
4. Chunking: fixed size, up to MAX_TOKENS tokens with OVERLAP_TOKENS tokens of overlap, counted
   with the embedder's tokenizer (all-MiniLM-L6-v2 reads at most 256 tokens, and the title is added
   in front at index time). Whole paragraphs, else whole lines, are packed into a chunk where
   possible (so an event or a painting is not cut in half); a longer line is split into
   sentences, then words.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "doc-collection" / "data" / "pages"
OUT = ROOT / "rag" / "data"

BOILERPLATE_SHARE = 0.3
BOILERPLATE_MIN_PAGES = 3
MIN_SITE_PAGES = 5
MIN_WORDS = 30
NEAR_DUP = 0.9
SHINGLE = 5
TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 200     # + title (~15) + [CLS]/[SEP] stays under MiniLM's 256
OVERLAP_TOKENS = 40
HEADER_TOKENS = 50   # repeat a split paragraph's first line only if it is this short
LEADER = re.compile(r"(?:[._\-–—…]\s?){4,}")  # "Contents........ 5", "_____", "-----"
JUNK_URL = re.compile(r"[?&]action=edit|/w/index\.php")  # Wikipedia edit pages

_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


@lru_cache(maxsize=None)
def ntok(text):
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


def load_pages():
    docs = []
    for path in sorted(PAGES.glob("*/*.txt")):
        raw = path.read_text(encoding="utf-8")
        first, _, body = raw.partition("\n")
        url = first[len("URL: "):].strip() if first.startswith("URL: ") else ""
        # blank lines separate paragraphs (e.g. one event or painting each); kept as "" markers
        lines = re.sub(r"\n\s*\n+", "\n\n", body.strip()).split("\n")
        lines = [re.sub(r"\s+", " ", LEADER.sub(" ", l)).strip() for l in lines]
        doc_id = str(path.relative_to(PAGES).with_suffix(""))
        docs.append({"doc_id": doc_id, "url": url, "host": path.parent.name, "lines": lines})
    return docs


def remove_boilerplate(docs):
    pages_per_site = Counter(d["host"] for d in docs)
    line_pages = defaultdict(Counter)  # host -> line -> number of pages containing it
    for d in docs:
        line_pages[d["host"]].update(set(l for l in d["lines"] if l))
    removed = 0
    for d in docs:
        n = pages_per_site[d["host"]]
        if n < MIN_SITE_PAGES:
            continue
        limit = max(BOILERPLATE_MIN_PAGES, BOILERPLATE_SHARE * n)
        keep = [l for l in d["lines"] if not l or line_pages[d["host"]][l] < limit]
        removed += len(d["lines"]) - len(keep)
        d["lines"] = keep
    return removed


def shingles(words):
    return {hash(" ".join(words[i:i + SHINGLE])) for i in range(max(1, len(words) - SHINGLE + 1))}


def deduplicate(docs):
    """Drop junk pages, exact duplicates, and pages mostly contained in a longer page of the same site."""
    stats = Counter()
    kept, seen = [], set()
    for d in docs:
        text = " ".join(l for l in d["lines"] if l)
        if JUNK_URL.search(d["url"]):
            stats["junk url"] += 1
            continue
        if len(text.split()) < MIN_WORDS:
            stats["too short"] += 1
            continue
        key = hashlib.md5(re.sub(r"\s+", " ", text).lower().encode()).hexdigest()
        if key in seen:
            stats["exact duplicate"] += 1
            continue
        seen.add(key)
        d["shingles"] = shingles(text.lower().split())
        kept.append(d)

    # near-duplicates: compare pages of the same site, longest first; a page is dropped when
    # most of its shingles are already in a kept page
    by_site = defaultdict(list)
    for d in kept:
        by_site[d["host"]].append(d)
    result = []
    for site_docs in by_site.values():
        site_docs.sort(key=lambda d: -len(d["shingles"]))
        index = defaultdict(set)  # shingle -> indexes of kept pages
        kept_site = []
        for d in site_docs:
            overlap = Counter(j for s in d["shingles"] for j in index.get(s, ()))
            best = max(overlap.values(), default=0)
            if best >= NEAR_DUP * len(d["shingles"]):
                stats["near duplicate"] += 1
                continue
            for s in d["shingles"]:
                index[s].add(len(kept_site))
            kept_site.append(d)
        result += kept_site
    for d in result:
        del d["shingles"]
    return sorted(result, key=lambda d: d["doc_id"]), stats


def split_long(line, max_tokens):
    """A line longer than max_tokens: split at sentence ends, then pack words up to max_tokens."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        current = []
        for word in sentence.split():
            if current and ntok(" ".join(current + [word])) > max_tokens:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    return pieces


def paragraphs(lines):
    """Split lines at "" markers into paragraphs (lists of lines)."""
    paras, current = [], []
    for line in lines + [""]:
        if line:
            current.append(line)
        elif current:
            paras.append(current)
            current = []
    return paras


def pack(units, max_tokens, overlap, first=()):
    """Greedily pack units (lines) into groups of up to max_tokens tokens. Each new group starts with
    `first` (a repeated header) and then the last units of the previous group, up to `overlap` tokens."""
    first = list(first)
    base = sum(ntok(u) for u in first)
    groups, current, size = [], list(first), base
    for unit in units:
        n = ntok(unit)
        if size + n > max_tokens and size > base:  # flush only if there is more than the header
            groups.append(current)
            carry, carried = [], 0
            for prev in reversed(current[len(first):]):
                m = ntok(prev)
                if carried + m > overlap:
                    break
                carry.insert(0, prev)
                carried += m
            if base + carried + n > max_tokens:
                carry, carried = [], 0
            current, size = first + carry, base + carried
        current.append(unit)
        size += n
    if len(current) > len(first):
        groups.append(current)
    return ["\n".join(g) for g in groups]


def chunk(lines, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """Pack whole paragraphs (else whole lines, else sentences) into chunks of up to max_tokens."""
    paras = paragraphs(lines)
    units = []
    for para in paras:
        text = "\n".join(para)
        if ntok(text) <= max_tokens:
            units.append(text)
            continue
        # a long paragraph is split; in multi-paragraph files (events, paintings) every piece repeats
        # the paragraph's first line ("Name: date, venue"). A web page is one paragraph whose first
        # line is the page title, which is added at index time anyway.
        header = para[0] if len(paras) > 1 and ntok(para[0]) <= HEADER_TOKENS else None
        room = max_tokens - (ntok(header) if header else 0)
        pieces = []
        for line in (para[1:] if header else para):
            pieces += [line] if ntok(line) <= room else split_long(line, room)
        units += pack(pieces, max_tokens, 0, [header] if header else [])
    return pack(units, max_tokens, overlap)


def main():
    docs = load_pages()
    print(f"{len(docs)} pages")
    removed = remove_boilerplate(docs)
    print(f"removed {removed} boilerplate lines")
    docs, stats = deduplicate(docs)
    for reason, n in stats.items():
        print(f"dropped {n} pages: {reason}")
    print(f"{len(docs)} documents kept")

    OUT.mkdir(parents=True, exist_ok=True)
    n_chunks, sizes = 0, []
    with open(OUT / "docs.jsonl", "w", encoding="utf-8") as fd, open(OUT / "chunks.jsonl", "w", encoding="utf-8") as fc:
        for d in docs:
            title = next(l for l in d["lines"] if l)[:200]
            fd.write(json.dumps({"doc_id": d["doc_id"], "url": d["url"], "title": title,
                                 "text": "\n".join(d["lines"]).strip()}, ensure_ascii=False) + "\n")
            for i, text in enumerate(chunk(d["lines"])):
                fc.write(json.dumps({"doc_id": d["doc_id"], "chunk_id": i, "text": text}, ensure_ascii=False) + "\n")
                n_chunks += 1
                sizes.append(ntok(text))
    sizes.sort()
    print(f"{n_chunks} chunks; tokens per chunk: median {sizes[len(sizes) // 2]}, max {sizes[-1]}")


if __name__ == "__main__":
    main()

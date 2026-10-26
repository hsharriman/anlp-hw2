"""Build the sparse (BM25) and dense (FAISS) indexes over rag/data/chunks.jsonl.

Each chunk is indexed as "<document title>\n<chunk text>": chunks store only doc_id, and the title
(from docs.jsonl) tells the retriever which page a chunk comes from (e.g. a list of player lines
under "Pittsburgh Pirates 2026 40-Man Roster").

Output, in rag/indexes/:
    bm25/                 bm25s index (English stopwords removed, Snowball stemming)
    dense-<model>/        FAISS inner-product index over L2-normalized embeddings (= cosine similarity)

Usage: python rag/index.py [--models minilm bge-base]
"""
import argparse
import json
import time
from pathlib import Path

import bm25s
import faiss
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "rag" / "data"
INDEXES = ROOT / "rag" / "indexes"

# name -> (Hugging Face model, prefix added to queries; BGE was trained with this instruction)
EMBEDDERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
}


def load_chunks():
    titles = {}
    for line in open(DATA / "docs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        titles[d["doc_id"]] = d["title"]
    chunks = [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]
    return chunks, titles


def index_text(chunk, titles):
    return f"{titles[chunk['doc_id']]}\n{chunk['text']}"


def bm25_tokenize(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)


def build_bm25(texts):
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize(texts), show_progress=False)
    retriever.save(INDEXES / "bm25")


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def build_dense(name, texts, batch_size=64):
    model_name, _ = EMBEDDERS[name]
    model = SentenceTransformer(model_name, device=device())
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])  # exact search; 48k vectors is small
    index.add(emb)
    out = INDEXES / f"dense-{name}"
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS))
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    chunks, titles = load_chunks()
    texts = [index_text(c, titles) for c in chunks]
    print(f"{len(texts)} chunks")
    INDEXES.mkdir(parents=True, exist_ok=True)
    if not args.skip_bm25:
        t = time.time()
        build_bm25(texts)
        print(f"bm25 built in {time.time() - t:.0f}s")
    for name in args.models:
        t = time.time()
        build_dense(name, texts)
        print(f"dense-{name} built in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

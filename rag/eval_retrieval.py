"""Retrieval-only evaluation on public_evaluation_data.json: is a reference answer in the top k?

A question counts as a hit at k if any reference answer appears (case/punctuation-insensitive)
in one of the top-k retrieved chunks. Only questions with short answers (<= 6 words) are used,
because long free-text answers are paraphrases that never appear verbatim.
Reports hit@k for k in (1, 3, 5, 10, 20) and MRR@20 for every retriever configuration.

Usage: python rag/eval_retrieval.py [reranker name, default bge-reranker-base]
"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10, 20)

CONFIGS = [
    ("sparse (BM25)", dict(mode="sparse")),
    ("dense (MiniLM)", dict(mode="dense", dense_model="minilm")),
    ("dense (BGE-base)", dict(mode="dense", dense_model="bge-base")),
    ("hybrid RRF (BM25 + MiniLM)", dict(mode="hybrid", fusion="rrf", dense_model="minilm")),
    ("hybrid RRF (BM25 + BGE)", dict(mode="hybrid", fusion="rrf", dense_model="bge-base")),
    ("hybrid weighted a=0.3 (BM25 + BGE)", dict(mode="hybrid", fusion="weighted", alpha=0.3, dense_model="bge-base")),
    ("hybrid weighted a=0.5 (BM25 + BGE)", dict(mode="hybrid", fusion="weighted", alpha=0.5, dense_model="bge-base")),
    ("hybrid weighted a=0.7 (BM25 + BGE)", dict(mode="hybrid", fusion="weighted", alpha=0.7, dense_model="bge-base")),
    ("weighted a=0.5 + rerank top 50", dict(mode="hybrid", fusion="weighted", alpha=0.5, rerank=50)),
]


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip() + " "


def first_hit(retriever, results, answers):
    """1-based rank of the first chunk containing an answer, or None."""
    answers = [norm(a) for a in answers if norm(a).strip()]
    for rank, (i, _) in enumerate(results, 1):
        text = norm(retriever.title(i) + " " + retriever.chunks[i]["text"])
        if any(a in text for a in answers):
            return rank
    return None


def main():
    questions = json.load(open(ROOT / "public_evaluation_data.json"))
    questions = [q for q in questions if max(len(a.split()) for a in q["answers"]) <= 6]
    reranker = sys.argv[1] if len(sys.argv) > 1 else "bge-reranker-base"
    retriever = Retriever(dense_models=("minilm", "bge-base"), reranker=reranker)
    print(f"{len(questions)} short-answer questions from public_evaluation_data.json\n")
    header = f"{'retriever':38s}" + "".join(f"  hit@{k:<3d}" for k in KS) + "  MRR@20"
    print(header)
    print("-" * len(header))
    for name, cfg in CONFIGS:
        t = time.time()
        ranks = [first_hit(retriever, retriever.search(q["question"], k=max(KS), **cfg), q["answers"])
                 for q in questions]
        hits = [sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS]
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"{name:38s}" + "".join(f"  {h:6.1%}" for h in hits) + f"  {mrr:.3f}"
              + f"   ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

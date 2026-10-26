"""Sparse, dense and hybrid retrieval over the indexes built by index.py.

    r = Retriever()
    r.search("Who is the Pirates' infield coach?", k=5, mode="hybrid", fusion="rrf")
returns a list of (chunk index, score), best first; r.chunks[i] is the chunk and r.titles its title.

Modes:
    sparse  BM25 only
    dense   embedding similarity only (dense_model="minilm" or "bge-base")
    hybrid  both, combined with one of two fusion methods over each retriever's top `pool` results:
        rrf       Reciprocal Rank Fusion: score = sum over retrievers of 1 / (rrf_k + rank)
                  (Cormack et al., 2009). Uses only ranks, so the two score scales don't matter.
        weighted  min-max normalize each retriever's scores to [0, 1] within its top `pool`, then
                  alpha * dense + (1 - alpha) * sparse; a chunk missing from one list gets 0 there.

Reranking (rerank=N): a cross-encoder reads the query and each of the top N results together and
rescores them; the k best by that score are returned. Slower than the bi-encoder, so it only sees
the first-stage candidates.
"""
import bm25s
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from index import EMBEDDERS, INDEXES, bm25_tokenize, device, load_chunks


RERANKERS = {"bge-reranker-base": "BAAI/bge-reranker-base", "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3"}


class Retriever:
    def __init__(self, dense_models=("bge-base",), reranker=None):
        self.chunks, self.titles = load_chunks()
        self.bm25 = bm25s.BM25.load(INDEXES / "bm25")
        self.dense = {}
        for name in dense_models:
            model_name, prefix = EMBEDDERS[name]
            self.dense[name] = (SentenceTransformer(model_name, device=device()), prefix,
                                faiss.read_index(str(INDEXES / f"dense-{name}" / "index.faiss")))
        self.reranker = CrossEncoder(RERANKERS[reranker], device=device(), max_length=512) if reranker else None

    def title(self, i):
        return self.titles[self.chunks[i]["doc_id"]]

    def sparse_search(self, query, k):
        docs, scores = self.bm25.retrieve(bm25_tokenize([query]), k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]

    def dense_search(self, query, k, model):
        encoder, prefix, index = self.dense[model]
        q = encoder.encode([prefix + query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, ids = index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def search(self, query, k=5, rerank=0, **kwargs):
        """rerank=N: rescore the first stage's top N with the cross-encoder and keep the best k."""
        if not rerank:
            return self.first_stage(query, k, **kwargs)
        candidates = self.first_stage(query, rerank, **kwargs)
        pairs = [(query, self.title(i) + "\n" + self.chunks[i]["text"]) for i, _ in candidates]
        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-scores)[:k]
        return [(candidates[j][0], float(scores[j])) for j in order]

    def first_stage(self, query, k=5, mode="hybrid", fusion="rrf", dense_model="bge-base",
                    pool=100, rrf_k=60, alpha=0.5):
        if mode == "sparse":
            return self.sparse_search(query, k)
        if mode == "dense":
            return self.dense_search(query, k, dense_model)
        sparse = self.sparse_search(query, pool)
        dense = self.dense_search(query, pool, dense_model)
        if fusion == "rrf":
            fused = rrf([sparse, dense], rrf_k)
        elif fusion == "weighted":
            fused = weighted(sparse, dense, alpha)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        return fused[:k]


def rrf(result_lists, rrf_k=60):
    scores = {}
    for results in result_lists:
        for rank, (i, _) in enumerate(results, 1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def min_max(results):
    if not results:
        return {}
    values = [s for _, s in results]
    lo, hi = min(values), max(values)
    return {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in results}


def weighted(sparse, dense, alpha=0.5):
    s, d = min_max(sparse), min_max(dense)
    scores = {i: alpha * d.get(i, 0.0) + (1 - alpha) * s.get(i, 0.0) for i in set(s) | set(d)}
    return sorted(scores.items(), key=lambda x: -x[1])

"""Answer a set of questions end to end and write {"id": "answer", ...} JSON.

Usage:
    python rag/run.py --questions public_evaluation_data.json --out system_outputs/public_rag.json
    python rag/run.py --questions test_questions.json --mode closed-book --out ...

--questions accepts the public data format (a list of {"id", "question", ...}), a {"id": "question"}
dict, or a .txt file with one question per line (ids 1, 2, ...).
Alongside the answers, <out>.details.jsonl records the retrieved passages for each question,
for error analysis.
"""
import argparse
import gc
import json
import time
from pathlib import Path

# import order matters: loading MLX (reader) before torch/faiss (retriever) makes the first FAISS
# search segfault on macOS, so the retriever is always imported first
from retrieve import Retriever
from reader import Reader, closed_book_prompt, is_refusal, rag_prompt


def load_questions(path):
    path = Path(path)
    if path.suffix == ".txt":
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return {str(i): q for i, q in enumerate(lines, 1)}
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if k != "andrewid"}
    return {str(q["id"]): q["question"] for q in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="rag", choices=["rag", "closed-book"])
    parser.add_argument("--reader", default="qwen3-30b-a3b")
    parser.add_argument("--retrieval", default="hybrid", choices=["sparse", "dense", "hybrid"])
    parser.add_argument("--fusion", default="weighted", choices=["rrf", "weighted"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dense-model", default="bge-base")
    parser.add_argument("--style", default="concise", choices=["short", "adaptive", "sentence", "concise", "concise-strict"],
                        help="answer form; concise (one sentence of at most 15 words) is best on the leaderboard")
    parser.add_argument("--reranker", default="none", help="cross-encoder, e.g. bge-reranker-v2-m3 (off by default: it lowered public-set F1)")
    parser.add_argument("--rerank", type=int, default=50, help="first-stage candidates the reranker rescores")
    parser.add_argument("--k", type=int, default=5, help="passages given to the reader")
    parser.add_argument("--no-fallback", action="store_true",
                        help="keep refusals instead of retrying with 2k passages, then closed-book")
    parser.add_argument("--andrewid", help="add {\"andrewid\": ...} first (leaderboard format)")
    parser.add_argument("--limit", type=int, help="only the first N questions (for quick tests)")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
    # phase 1, retrieval: all questions first, then free the embedder and reranker, so they don't
    # compete with the reader for (unified) memory
    retrieved = {}
    if args.mode == "rag":
        if args.reranker == "none":
            args.rerank = 0
        retriever = Retriever(dense_models=(args.dense_model,), reranker=args.reranker if args.rerank else None)
        k = args.k if args.no_fallback else 2 * args.k  # the fallback retries with 2k passages
        for qid, question in questions.items():
            retrieved[qid] = retrieve(retriever, question, k, args)
        del retriever
        gc.collect()
        free_torch_memory()
        print(f"retrieved passages for {len(questions)} questions", flush=True)

    # phase 2, reading
    reader = Reader(args.reader, style=args.style)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    answers, start = {}, time.time()
    with open(str(out) + ".details.jsonl", "w", encoding="utf-8") as details:
        for n, (qid, question) in enumerate(questions.items(), 1):
            passages, sources, fallback = [], [], None
            if args.mode == "rag":
                all_passages, all_sources = retrieved[qid]
                passages, sources = all_passages[:args.k], all_sources[:args.k]
                answers[qid] = reader.answer(rag_prompt(question, passages, args.style))
                if is_refusal(answers[qid]) and not args.no_fallback:
                    # retry with twice the passages, then from the model's own knowledge
                    fallback, answers[qid] = "more-passages", reader.answer(rag_prompt(question, all_passages, args.style))
                    if is_refusal(answers[qid]):
                        fallback, answers[qid] = "closed-book", reader.answer(closed_book_prompt(question))
            else:
                answers[qid] = reader.answer(closed_book_prompt(question))
            details.write(json.dumps({"id": qid, "question": question, "answer": answers[qid],
                                      "fallback": fallback, "sources": sources,
                                      "passages": passages}, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(questions):
                print(f"{n}/{len(questions)}  ({(time.time() - start) / n:.1f}s per question)", flush=True)
    if args.andrewid:
        answers = {"andrewid": args.andrewid, **answers}
    json.dump(answers, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {out}")


def free_torch_memory():
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def retrieve(retriever, question, k, args):
    hits = retriever.search(question, k=k, rerank=max(args.rerank, k) if args.rerank else 0, mode=args.retrieval, fusion=args.fusion,
                            alpha=args.alpha, dense_model=args.dense_model)
    passages = [(retriever.title(i), retriever.chunks[i]["text"]) for i, _ in hits]
    return passages, [retriever.chunks[i]["doc_id"] for i, _ in hits]


if __name__ == "__main__":
    main()

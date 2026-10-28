#!/bin/bash
# Every system variant of the report on public_evaluation_data.json (157 questions); answers go to
# rag/results/public/. Then: python rag/evaluate.py rag/results/public/*.json
# The corpus3_* runs (styles and ablations) used corpus 3 (2,709 docs, before the deep crawl:
# scrape.py deep); the corpus4_* runs use the final corpus. Corpus 1 and 2 outputs came from earlier
# versions of the document collection.
set -e
cd "$(dirname "$0")/.."
Q=public_evaluation_data.json
OUT=rag/results/public
run() { name=$1; shift; [ -f "$OUT/$name.json" ] && { echo "skip $name"; return; }
        echo "== $name"; python rag/run.py --questions $Q --out "$OUT/$name.json" "$@" 2>&1 | grep -E "^[0-9]+/|wrote|Error"; }

# final pipeline (hybrid weighted a=0.5, BGE, k=5, refusal fallback, detail-matching prompt,
# Qwen3-30B-A3B) with each answer style. Leaderboard references are mostly sentences, so the
# sentence styles trade public-set F1 (short references) for leaderboard F1/recall/ROUGE-L.
run corpus3_rag_hybrid-w0.5_fallback_match-prompt_qwen30b          --style short
run corpus3_rag_hybrid-w0.5_fallback_match-prompt_sentence_qwen30b --style sentence
run corpus3_rag_hybrid-w0.5_fallback_match-prompt_concise_qwen30b  --style concise
run corpus3_rag_hybrid-w0.5_fallback_match-prompt_adaptive_qwen30b --style adaptive

# ablations: one factor at a time against the final pipeline, short answers because the public
# references are mostly short
run corpus3_closed-book_qwen30b      --mode closed-book --style short
run corpus3_rag_sparse_qwen30b       --retrieval sparse --style short
run corpus3_rag_dense-minilm_qwen30b --retrieval dense --dense-model minilm --style short
run corpus3_rag_dense-bge_qwen30b    --retrieval dense --dense-model bge-base --style short
run corpus3_rag_hybrid-rrf_qwen30b   --retrieval hybrid --fusion rrf --style short
run corpus3_rag_hybrid-w0.5_rerank-m3_qwen30b --style short --reranker bge-reranker-v2-m3 --rerank 50
run corpus3_closed-book_qwen8b       --mode closed-book --reader qwen3-8b --style short
run corpus3_rag_hybrid-w0.5_qwen8b   --reader qwen3-8b --style short

# final corpus (corpus 4: + 2-link crawl of the recommended sites)
run corpus4_rag_hybrid-w0.5_fallback_match-prompt_qwen30b          --style short
run corpus4_rag_hybrid-w0.5_fallback_match-prompt_concise_qwen30b  --style concise

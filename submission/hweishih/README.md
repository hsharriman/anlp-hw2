# Pittsburgh & CMU question answering with retrieval-augmented generation

CMU 11-711 (Advanced NLP) Assignment 2. The system answers factual questions about Pittsburgh and
Carnegie Mellon University: it retrieves passages from a document collection we built, then a
local open-weight LLM answers from them. The assignment handout is in [handout.md](handout.md).

| Component | Choice |
|---|---|
| Knowledge resource | 5,063 pages from 47 sites (Wikipedia, city/CMU/visitor sites, event calendars, sports, museums), cleaned and deduplicated to 4,562 documents and 60,032 chunks |
| Chunking | our own: token-based, at most 200 tokens with 40 tokens of overlap, paragraph-aware |
| Sparse retrieval | BM25 ([bm25s](https://github.com/xhluca/bm25s)), English stopwords and stemming |
| Dense retrieval | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) embeddings in a FAISS flat inner-product index ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the baseline) |
| Hybrid retrieval | our own: weighted fusion of min-max-normalized scores (α = 0.5) or Reciprocal Rank Fusion, over each retriever's top 100 |
| Reader | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), a mixture-of-experts model (30.5B total parameters, 3.3B active; released July 2025), 4-bit, run locally with [MLX](https://github.com/ml-explore/mlx-lm); [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as a smaller comparison |

No RAG framework (LangChain, LlamaIndex, ...) is used; chunking, retrieval, fusion and prompting
are implemented in `rag/`.

## Repository layout

```
doc-collection/          building the knowledge resource
  sources.json           seed URLs (the sites the handout recommends)
  common.py              shared helpers (requests, text cleanup, date formatting, saving documents)
  scrape.py              crawling (links, sitemaps, Wikipedia lists), HTML/PDF to text, social accounts
  events.py              event calendars (CMU, Downtown, City Paper, pittsburgh.events, Improv, Mr. Smalls)
  sports.py              pro schedules and rosters, college athletics schedules
  cmoa_collection.py     Carnegie Museum of Art paintings
  citypaper_download.js  browser-console script that saves the City Paper event pages
  raw_pages/             pages saved by hand from a browser, for sites that block scripts
  data/pages/<host>/     the collected documents, one .txt per page ("URL: ..." on the first line)
rag/
  preprocess.py          boilerplate removal, deduplication, chunking -> rag/data/
  index.py               BM25 and FAISS indexes -> rag/indexes/ (not in git; rebuild them)
  retrieve.py            sparse, dense and hybrid retrieval (+ optional cross-encoder reranking)
  reader.py              LLM reader and prompts
  run.py                 answer a question file end to end -> {"id": "answer"} JSON
  evaluate.py            exact match, F1, answer recall, ROUGE-L against reference answers
  eval_retrieval.py      retrieval-only evaluation (is a reference answer in the top k?)
  report_tables.py       the tables in the report, from the files in rag/results/
  experiments.sh         every system variant on the public evaluation set
  results/               system outputs (public/ and leaderboard/)
system_outputs/          answers on the unseen test set (make_outputs.sh writes system_output_{1,2,3}.json)
pseudocode.md            what each RAG function does and why
handout.md               the assignment
```

## Setup

The reader runs on Apple silicon with MLX (we used an M2 Pro with 32 GB of memory; the 4-bit
30B model takes about 17 GB). Retrieval runs anywhere (PyTorch on CPU, CUDA or MPS).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                    # RAG system
pip install -r doc-collection/requirements.txt     # data collection
```

Models download from Hugging Face on first use. If `~/.cache/huggingface` is not writable, set
`HF_HOME` to another folder (we used `export HF_HOME=~/.cache/hf_anlp`). Also set
`TOKENIZERS_PARALLELISM=false` to silence a tokenizers warning.

## 1. Collecting the documents

The collected pages are in the repository (`doc-collection/data/pages/`), so this step is only
needed to refresh them. Run the scripts from `doc-collection/`:

| Command | What it collects |
|---|---|
| `python scrape.py seeds [category ...]` | each seed in `sources.json` and the pages it links to on the same site; `scrape()` was also called directly for deeper crawls (Symphony, Opera: 2 levels, 150 pages) and for smaller sites (choirs, zoo, Vintage Grand Prix, Mr. Smalls) |
| `python scrape.py deep [host ...]` | 2-link crawl of 20 recommended sites (`DEEP_SITES`), keeping pages of 30+ words (about, visit, FAQ, board, careers, promotions pages); obeys robots.txt and its crawl delay |
| `python scrape.py sitemap https://www.visitpittsburgh.com/sitemaps-1-sitemap.xml event partnerDirectory pages landingPages blogDirectory` | every Visit Pittsburgh page listed in its XML sitemaps (events, directory of restaurants/venues/organizations, blog) |
| `python scrape.py wiki-lists` | Wikipedia articles linked from Pittsburgh list pages (museums, bridges, landmarks, sports, colleges) |
| `python scrape.py social` | each organization's social media accounts (sites link them only in page footers, which the text conversion drops) |
| `python events.py [cmu downtown improv mrsmalls]` | event calendars from their JSON feeds or markup, one paragraph or line per event with full dates |
| `python sports.py [schedules rosters college]` | Steelers, Pirates and Penguins schedules, rosters and coaches (MLB and NHL JSON APIs, steelers.com); Pitt, Robert Morris, Duquesne and Chatham schedules for every sport |
| `python cmoa_collection.py` | Carnegie Museum of Art paintings (about 1,500), from the collection site's search API |

Some sites block scripts, so their pages were saved by hand from a browser into `raw_pages/` and
converted:

- **Pittsburgh City Paper events**: run `citypaper_download.js` in the browser console on a
  community.pghcitypaper.com events page; it downloads all result pages into one HTML file. Then
  `python events.py citypaper`.
- **pittsburgh.events**: month pages printed to PDF, then `python events.py pdf`.
- **Britannica**: `python scrape.py saved-html raw_pages/brittanica ".topic-content, .font-serif.font-18"`
  (the selector keeps the article body); **Little Italy Days**: `python scrape.py saved-html raw_pages/littleitaly`.

## 2. Preprocessing and chunking

```bash
python rag/preprocess.py
```

Reads `doc-collection/data/pages/`, removes lines that repeat on at least 30% of a site's pages
(menus, footers), drops junk URLs, pages under 30 words, exact duplicates and near duplicates
(90% of 5-word shingles shared), and chunks the rest. Chunks are packed from whole paragraphs up
to 200 tokens (counted with the MiniLM tokenizer, which reads at most 256), with 40 tokens of
overlap; a paragraph that has to be split repeats its first line in each piece. Each chunk stores
only its `doc_id`; the document title is prepended when indexing. Output: `rag/data/docs.jsonl`
and `rag/data/chunks.jsonl`.

## 3. Indexing

```bash
python rag/index.py            # BM25 + both dense indexes; --models bge-base to build only one
```

About 11 minutes on the M2 Pro, mostly embedding 60k chunks with BGE.

## 4. Answering questions

```bash
# test set -> system_outputs/system_output_{1,2,3}.json: concise, sentence, short
# styles (only the best-scoring file is graded)
system_outputs/make_outputs.sh test_questions.json

# leaderboard format (adds "andrewid")
python rag/run.py --questions leaderboard_queries.json --out rag/results/leaderboard/out.json --andrewid ANDREWID
```

`--questions` accepts a list of `{"id", "question"}` objects, an `{"id": "question"}` object, or
a .txt file with one question per line. Defaults are the final system: hybrid weighted fusion
(α = 0.5) of BM25 and BGE, 5 passages, Qwen3-30B-A3B, concise answers. Other options:

| Option | Values |
|---|---|
| `--mode` | `rag` or `closed-book` (no retrieval) |
| `--retrieval`, `--fusion`, `--alpha` | `sparse` / `dense` / `hybrid`; `weighted` / `rrf`; weight of the dense score |
| `--dense-model` | `bge-base` or `minilm` |
| `--k` | passages given to the reader (default 5) |
| `--reranker`, `--rerank` | e.g. `bge-reranker-v2-m3` to rescore the top 50 with a cross-encoder (off by default) |
| `--reader` | `qwen3-30b-a3b` or `qwen3-8b` |
| `--style` | `concise` (one sentence of at most 15 words), `sentence`, `short` (only the answer phrase), `adaptive`, `concise-strict` |
| `--no-fallback` | keep refusals instead of retrying with 10 passages and then without retrieval |

Retrieval for all questions runs first and its models are freed before the reader loads, so
both fit in memory. Each run also writes `<out>.details.jsonl` with the retrieved passages per
question. About 5 seconds per question.

## 5. Evaluation

```bash
python rag/evaluate.py rag/results/public/*.json      # EM, F1, recall, ROUGE-L vs public_evaluation_data.json
python rag/eval_retrieval.py                          # hit@k and MRR of each retriever
./rag/experiments.sh                                  # all variants (skips ones already run)
python rag/report_tables.py                           # report tables (LaTeX) -> rag/results/tables/
```

## Models and data policy

All models are open-weight, on Hugging Face, released before 2026 and at most 32B total
parameters: Qwen3-30B-A3B-Instruct-2507 (30.5B total, mixture of experts), Qwen3-8B (8.2B),
BAAI/bge-base-en-v1.5 (110M), all-MiniLM-L6-v2 (22M), BAAI/bge-reranker-v2-m3 (568M). No closed
models were used for any part of the work. The knowledge resource contains only publicly
available pages.

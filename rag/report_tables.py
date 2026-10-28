"""Build the report's tables from the system outputs in rag/results/.

Writes rag/results/tables/<name>.tex (booktabs, for the ACL template) and prints each table:
    corpus        documents and tokens per source group, with the sites in each group (all
                  computed from rag/data/, so the table follows the corpus when it is rebuilt)
    retrieval     hit@k / MRR per retriever (from rag/results/public/retrieval_corpus3.txt)
    ablations     end-to-end scores of every variant on corpus 3, with paired bootstrap
                  p-values against the final system
    progression   corpus versions and answer styles: public set and leaderboard side by side
    by_topic      F1 per question topic for closed-book and each retriever (which fusion works
                  best for which kind of question)
    by_type       F1 per answer type (date/time, person, number, ...)
    examples      outputs of several systems on the same questions (full-width table)
Variants that have not been run yet are left out.

Usage: python rag/report_tables.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from evaluate import METRICS, load_references, score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "rag" / "results"
OUT = RESULTS / "tables"
PUBLIC = RESULTS / "public"
REFS = load_references(ROOT / "public_evaluation_data.json")
QUESTIONS = {str(q["id"]): q["question"] for q in json.load(open(ROOT / "public_evaluation_data.json"))}
TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"  # the chunker's tokenizer (preprocess.py)
FINAL = "corpus3_rag_hybrid-w0.5_fallback_match-prompt_qwen30b"  # final system, short answers

# (file stem in rag/results/public, label); all on corpus 3 with short answers
ABLATIONS = [
    ("corpus3_closed-book_qwen8b", "Closed-book, Qwen3-8B"),
    ("corpus3_closed-book_qwen30b", "Closed-book, Qwen3-30B-A3B"),
    ("corpus3_rag_sparse_qwen30b", "Sparse (BM25)"),
    ("corpus3_rag_dense-minilm_qwen30b", "Dense (MiniLM)"),
    ("corpus3_rag_dense-bge_qwen30b", "Dense (BGE-base)"),
    ("corpus3_rag_hybrid-rrf_qwen30b", "Hybrid, RRF"),
    (FINAL, "Hybrid, weighted (final)"),
    ("corpus3_rag_hybrid-w0.5_rerank-m3_qwen30b", "Hybrid, weighted + reranker"),
    ("corpus3_rag_hybrid-w0.5_qwen8b", "Hybrid, weighted, Qwen3-8B reader"),
]
# (public file stem, leaderboard file, label)
PROGRESSION = [
    ("rag_hybrid-w0.5_fallback_qwen30b", "hybrid-w0.5_fallback_qwen30b.json", "Corpus 1 (1,155 docs)"),
    ("corpus2_rag_hybrid-w0.5_fallback_qwen30b", "corpus2_hybrid-w0.5_fallback_qwen30b.json",
     "Corpus 2: + college schedules, Wikipedia lists (1,689 docs)"),
    ("corpus3_rag_hybrid-w0.5_fallback_qwen30b", None,
     "Corpus 3: + Visit Pittsburgh, calendars, social accounts (2,709 docs)"),
    (FINAL, "corpus3_hybrid-w0.5_fallback_match-prompt_qwen30b.json", "+ detail-matching prompt (short answers)"),
    ("corpus3_rag_hybrid-w0.5_fallback_match-prompt_sentence_qwen30b",
     "corpus3_hybrid-w0.5_fallback_match-prompt_sentence_qwen30b.json", "Sentence answers"),
    ("corpus3_rag_hybrid-w0.5_fallback_match-prompt_concise_qwen30b",
     "corpus3_hybrid-w0.5_fallback_match-prompt_concise_qwen30b.json", "Concise sentences"),
    ("corpus3_rag_hybrid-w0.5_fallback_match-prompt_adaptive_qwen30b",
     "corpus3_hybrid-w0.5_fallback_match-prompt_adaptive_qwen30b.json", "Adaptive (fact or sentence by question)"),
    ("corpus4_rag_hybrid-w0.5_fallback_match-prompt_concise_qwen30b",
     "corpus4_hybrid-w0.5_fallback_match-prompt_concise_qwen30b.json",
     "Corpus 4: 2-link crawl of recommended sites (4,562 docs), concise (submitted best)"),
]
TOPICS = [  # first match wins
    ("Art collection", r"Carnegie Museum of Art.*(painting|record|catalog|credit|accession|creator|list|acquire)|"
                       r"acquired according to the Carnegie"),
    ("Sports", r"Steelers|Pirates|Penguins|Pitt (wo)?men|Duquesne|Robert Morris|Chatham|volleyball|basketball|"
               r"hockey|football|soccer|Syracuse|Wild Card"),
    ("Events", r"perform|event|concert|show\b|open mic|comedy|production|arts event|Christmas Carol|Nights"),
    ("Taxes & government", r"[Tt]ax|budget|Borough|Municipality|County|police|LST"),
    ("Food & dining", r"restaurant|cuisine|bakery|chocolate|coffee|[Pp]izza|food|steakhouse|specialty|Restaurant Week"),
    ("CMU", r"CMU|Carnegie Mellon"),
    ("History & general", r""),
]
SOURCE_GROUPS = [  # report groups for the sites in the corpus, matched on the host; first match wins
    ("Wikipedia", r"wikipedia"),
    ("Visit Pittsburgh", r"visitpittsburgh"),
    ("Event calendars", r"pghcitypaper|pittsburgh\.events|downtownpittsburgh|events\.cmu|improv|mrsmalls"),
    ("Sports (pro and college)", r"steelers|mlb|nhl|panthers|rmucolonials|goduquesne|chathamcougars"),
    ("Museums and art collection", r"carnegieart|carnegiemuseums|heinzhistory|frick|warhol|mattress|clemente|"
                                   r"photoantiq|phipps|zoo"),
    ("Music and performing arts", r"symphony|opera|trustarts|culturaldistrict|mendelssohn|bachchoir|music\.pitt|"
                                  r"applause|awaacc"),
    ("City and CMU", r"pittsburghpa|cmu\.edu|britannica"),
    ("Food festivals", r"picklesburgh|tacofest|restaurantweek|littleitaly|bananasplit"),
    ("Other", r""),
]
ANSWER_TYPES = [
    ("Yes/no", r"^(Is|Does|Are|Do|Can)\b"),
    ("Description", r"significance|^What is (the )?'|unique|specialty|^What was|role did|^What is Pittsburgh Restaurant"),
    ("Date/time", r"^When|what (time|date|year)|year was|In what year|date does"),
    ("Number", r"[Hh]ow (many|much)|number|rate|percentage|budget"),
    ("Person", r"^Who|artist|creator|conductor"),
    ("Place", r"^Where|venue|located|neighborhood|college does"),
    ("Name/other", r""),
]


def first_match(question, rules):
    return next(label for label, pattern in rules if re.search(pattern, question))


def load(stem, folder=PUBLIC):
    path = folder / f"{stem}.json"
    return {k: v for k, v in json.load(open(path)).items() if k != "andrewid"} if path.exists() else None


def per_question(stem):
    preds = load(stem)
    return None if preds is None else score(preds, REFS)[1]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def paired_bootstrap(a, b, metric="F1", samples=10000, seed=0):
    """Two-sided p-value for 'systems a and b have the same mean <metric>' over the same questions."""
    ids = sorted(a)
    diffs = [a[q][metric] - b[q][metric] for q in ids]
    observed = mean(diffs)
    rng = random.Random(seed)
    centered = [d - observed for d in diffs]
    extreme = sum(abs(mean(rng.choice(centered) for _ in ids)) >= abs(observed) for _ in range(samples))
    return extreme / samples


def esc(s):
    """Escape LaTeX special characters; **text** becomes bold."""
    s = s.replace("\\", r"\textbackslash{}")
    for c in "&%_#$":
        s = s.replace(c, "\\" + c)
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)


def write(name, header, rows, caption, align=None, wide=None):
    """rows: lists of cells (str), a str for a row across all columns, or None for a rule. Writes a
    booktabs LaTeX table and prints it. Tables with more than 5 columns span both page columns
    (table*) unless wide says otherwise."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = "table*" if (wide if wide is not None else len(header) > 5) else "table"
    align = align or "l" + "r" * (len(header) - 1)
    tex = [rf"\begin{{{env}}}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        if r is None:
            tex.append(r"\midrule")
        elif isinstance(r, str):  # a full-width row
            tex.append(rf"\multicolumn{{{len(header)}}}{{p{{0.95\textwidth}}}}{{{esc(r)}}} \\")
        else:
            tex.append(" & ".join(esc(c) for c in r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
            rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n{caption}\n" + "\n".join((r if isinstance(r, str) else " | ".join(r)).replace("**", "")
                                         for r in [header, *rows] if r))


def pct(x):
    return f"{100 * x:.1f}"


def corpus_table():
    """Documents, tokens and chunks per source group; each group lists the sites (hosts) it holds, so a
    site added to the collection shows up on its own (in "Other" until SOURCE_GROUPS names it)."""
    from transformers import AutoTokenizer, logging
    logging.set_verbosity_error()  # whole documents exceed the model's 512-token limit; we only count
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    docs = [json.loads(l) for l in open(ROOT / "rag" / "data" / "docs.jsonl")]
    chunks = Counter(json.loads(l)["doc_id"] for l in open(ROOT / "rag" / "data" / "chunks.jsonl"))
    lengths = tokenizer([d["text"] for d in docs], add_special_tokens=False, return_length=True)["length"]
    groups = {label: Counter() for label, _ in SOURCE_GROUPS}  # label -> host -> [docs, tokens, chunks]
    for d, n_tokens in zip(docs, lengths):
        host = re.sub(r"^www\.", "", d["doc_id"].split("/")[0])  # www.x.org and x.org are one site
        label = first_match(host, SOURCE_GROUPS)
        groups[label][host, "docs"] += 1
        groups[label][host, "tokens"] += n_tokens
        groups[label][host, "chunks"] += chunks[d["doc_id"]]
    rows, total = [], Counter()
    for label, counts in groups.items():
        hosts = sorted({h for h, _ in counts}, key=lambda h: (-counts[h, "docs"], h))
        if not hosts:
            continue
        row = {key: sum(counts[h, key] for h in hosts) for key in ("docs", "tokens", "chunks")}
        total.update(row)
        total["sites"] += len(hosts)
        rows.append([label, ", ".join(hosts), f"{row['docs']:,}", f"{row['tokens']:,}", f"{row['chunks']:,}"])
    rows += [None, ["Total", f"{total['sites']} sites", f"{total['docs']:,}", f"{total['tokens']:,}",
                    f"{total['chunks']:,}"]]
    write("corpus", ["Source", "Sites", "Documents", "Tokens", "Chunks"], rows,
          "Composition of our knowledge resource after cleaning and deduplication. Tokens are counted with the "
          "all-MiniLM-L6-v2 WordPiece tokenizer used for chunking.",
          align=r"lp{0.4\textwidth}rrr", wide=True)


def retrieval_table():
    path = PUBLIC / "retrieval_corpus3.txt"
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"(.+?)\s+((?:\d+\.\d%\s+){5})(\d\.\d{3})", line)
        if m:
            rows.append([m.group(1).strip()] + [c.rstrip("%") for c in m.group(2).split()] + [m.group(3)])
    n = re.search(r"(\d+) short-answer questions", path.read_text())
    write("retrieval", ["Retriever", "hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "MRR@20"], rows,
          f"Retrieval on corpus 3 (2,709 documents, before the final crawl): share of the {n.group(1) if n else ''} short-answer public questions "
          "with a reference answer in the top k chunks.")


def ablation_table():
    final = per_question(FINAL)
    rows = []
    for stem, label in ABLATIONS:
        pq = per_question(stem)
        if pq is None:
            continue
        p = "" if stem == FINAL or final is None else f"{paired_bootstrap(pq, final):.3f}"
        rows.append([label] + [pct(mean(s[m] for s in pq.values())) for m in METRICS] + [p])
    write("ablations", ["System", *METRICS, "p (F1 vs final)"], rows,
          f"End-to-end results on public_evaluation_data.json ({len(REFS)} questions), corpus 3 (2,709 documents, before the final crawl), short "
          "answers, k=5. p: paired bootstrap (10,000 samples) for the F1 difference from the final system.")


def progression_table():
    board = json.load(open(RESULTS / "leaderboard" / "scores.json"))
    long_refs = [q for q in REFS if max(len(a.split()) for a in REFS[q]) > 6]
    rows = []
    for stem, board_file, label in PROGRESSION:
        pq = per_question(stem)
        if pq is None:
            continue
        lb = board.get(board_file, {})
        rows.append([label, pct(mean(s["F1"] for s in pq.values())),
                     pct(mean(pq[q]["F1"] for q in long_refs)),
                     f"{lb['overall']:.2f}" if "overall" in lb else "-",
                     f"{lb['F1']:.1f}" if "F1" in lb else "-", f"{lb['Recall']:.1f}" if "Recall" in lb else "-",
                     f"{lb['ROUGE-L']:.1f}" if "ROUGE-L" in lb else "-", f"{lb['judge']:.2f}" if "judge" in lb else "-"])
    write("progression", ["System", "Public F1", "Long refs F1", "LB overall",
                          "LB F1", "LB recall", "LB ROUGE-L", "LB judge"], rows,
          f"Development history: public set ({len(REFS)} questions) and leaderboard (LB, 150 questions; "
          "overall = mean of F1, recall, ROUGE-L and the rescaled judge score). Hybrid weighted retrieval, "
          "Qwen3-30B-A3B. Separate question sets: never compare a public number with a leaderboard number. "
          f"Long refs: the {len(long_refs)} public questions whose references are longer than 6 words.",
          align=r"p{0.21\textwidth}rrrrrrr")


def breakdown_table(name, rules, caption):
    systems = [(s, l) for s, l in ABLATIONS if per_question(s) is not None and "8B" not in l and "reranker" not in l]
    if not systems:
        return
    scores = {s: per_question(s) for s, _ in systems}
    groups = {}
    for q in REFS:
        groups.setdefault(first_match(QUESTIONS[q], rules), []).append(q)
    rows = []
    for label, _ in rules:
        ids = groups.get(label, [])
        if not ids:
            continue
        cells = [pct(mean(scores[s][q]["F1"] for q in ids)) for s, _ in systems]
        best = max(range(len(cells)), key=lambda i: float(cells[i]) if i > 0 else -1)  # best retrieval system
        cells[best] = f"**{cells[best]}**"
        rows.append([f"{label} ({len(ids)})"] + cells)
    short = [l.replace("Closed-book, Qwen3-30B-A3B", "Closed-book").replace("Hybrid, weighted (final)", "Weighted")
              .replace("Hybrid, RRF", "RRF").replace("Sparse (BM25)", "BM25") for _, l in systems]
    write(name, ["Questions (n)"] + short, rows, caption + " Bold: the best retrieval system for each group.")


def examples():
    picks = {  # question id -> why it is shown
        "45": "schedule lookup: retrieval needed",
        "3": "art collection: exact title match (BM25 helps)",
        "5": "descriptive question with a long reference",
        "61": "answer in a schedule table",
        "88": "model knowledge suffices",
        "33": "event venue",
    }
    systems = [("corpus3_closed-book_qwen30b", "Closed-book"), ("corpus3_rag_sparse_qwen30b", "BM25"),
               ("corpus3_rag_dense-bge_qwen30b", "Dense"), (FINAL, "Hybrid (short)"),
               ("corpus3_rag_hybrid-w0.5_fallback_match-prompt_concise_qwen30b", "Hybrid (concise)")]
    outputs = {label: load(stem) for stem, label in systems}
    rows = []
    for qid, why in picks.items():
        if rows:
            rows.append(None)
        rows.append(f"**Q{qid}** ({why}): {QUESTIONS[qid]} Reference: {REFS[qid][0]}")
        rows += [[label, preds.get(qid, "")] for label, preds in outputs.items() if preds]
    write("examples", ["System", "Answer"], rows, "Example outputs on public questions.",
          align=r"lp{0.75\textwidth}", wide=True)


def main():
    corpus_table()
    retrieval_table()
    ablation_table()
    progression_table()
    breakdown_table("by_topic", TOPICS, "F1 by question topic (public set, corpus 3, short answers). "
                    "Topics are assigned by keyword rules on the question.")
    breakdown_table("by_type", ANSWER_TYPES, "F1 by expected answer type (public set, corpus 3, short answers).")
    examples()


if __name__ == "__main__":
    main()

"""Build the report's tables from the system outputs in rag/results/.

Writes rag/results/tables/<name>.md (to read) and <name>.tex (booktabs, for the ACL template):
    corpus        documents and chunks per source group
    retrieval     hit@k / MRR per retriever (from rag/results/public/retrieval_corpus3.txt)
    ablations     end-to-end scores of every variant on the final corpus, with paired bootstrap
                  p-values against the final system
    progression   corpus versions and answer styles: public set and leaderboard side by side
    by_topic      F1 per question topic for closed-book and each retriever (which fusion works
                  best for which kind of question)
    by_type       F1 per answer type (date/time, person, number, ...)
    examples      outputs of several systems on the same questions
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
     "corpus3_hybrid-w0.5_fallback_match-prompt_concise_qwen30b.json", "Concise sentences (submitted best)"),
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


def write(name, header, rows, caption, align=None):
    """rows: lists of cells (str). Writes markdown and a booktabs LaTeX table."""
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    esc = lambda s: s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    align = align or "l" + "r" * (len(header) - 1)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
            r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n{caption}\n" + "\n".join(md))


def pct(x):
    return f"{100 * x:.1f}"


def corpus_table():
    groups = [("Wikipedia", r"wikipedia"), ("Visit Pittsburgh", r"visitpittsburgh"),
              ("Event calendars", r"pghcitypaper|pittsburgh\.events|downtownpittsburgh|events\.cmu|improv|mrsmalls"),
              ("Sports (pro and college)", r"steelers|mlb|nhl|panthers|rmucolonials|goduquesne|chathamcougars"),
              ("Museums and art collection", r"carnegieart|carnegiemuseums|heinzhistory|frick|warhol|mattress|clemente|photoantiq|phipps|zoo"),
              ("Music and performing arts", r"symphony|opera|trustarts|culturaldistrict|mendelssohn|bachchoir|music\.pitt|applause|awaacc"),
              ("City and CMU", r"pittsburghpa|cmu\.edu|britannica"),
              ("Food festivals", r"picklesburgh|tacofest|restaurantweek|littleitaly|bananasplit"),
              ("Other", r"")]
    docs = [json.loads(l) for l in open(ROOT / "rag" / "data" / "docs.jsonl")]
    chunks = Counter(json.loads(l)["doc_id"] for l in open(ROOT / "rag" / "data" / "chunks.jsonl"))
    rows, totals = [], [0, 0]
    for label, pattern in groups:
        picked = [d for d in docs if re.search(pattern, d["doc_id"].split("/")[0]) and not d.get("_used")]
        for d in picked:
            d["_used"] = True
        n_chunks = sum(chunks[d["doc_id"]] for d in picked)
        totals[0] += len(picked)
        totals[1] += n_chunks
        rows.append([label, f"{len(picked):,}", f"{n_chunks:,}"])
    rows.append(["Total", f"{totals[0]:,}", f"{totals[1]:,}"])
    write("corpus", ["Source", "Documents", "Chunks"], rows,
          "Knowledge resource after cleaning and deduplication (final corpus).")


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
          f"Retrieval on the final corpus: share of the {n.group(1) if n else ''} short-answer public questions "
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
          f"End-to-end results on public_evaluation_data.json ({len(REFS)} questions), final corpus, short "
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
    write("progression", ["System", "Public F1", f"Public F1, long refs (n={len(long_refs)})", "LB overall",
                          "LB F1", "LB recall", "LB ROUGE-L", "LB judge"], rows,
          f"Development history: public set ({len(REFS)} questions) and leaderboard (LB, 150 questions; "
          "overall = mean of F1, recall, ROUGE-L and the rescaled judge score). Hybrid weighted retrieval, "
          "Qwen3-30B-A3B. Separate question sets: never compare a public number with a leaderboard number.")


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
    write(name, ["Questions (n)"] + short, [[c.replace("**", "") for c in r] for r in rows], caption)
    (OUT / f"{name}.md").write_text(f"{caption}\n\n" + "\n".join(
        ["| " + " | ".join(["Questions (n)"] + short) + " |", "|" + "|".join("---" for _ in range(len(short) + 1)) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]) + "\n", encoding="utf-8")


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
    lines = []
    for qid, why in picks.items():
        lines.append(f"**Q{qid}** ({why}): {QUESTIONS[qid]}  \nReference: {REFS[qid][0]}")
        for label, preds in outputs.items():
            if preds:
                lines.append(f"- {label}: {preds.get(qid, '')}")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples.md").write_text("Example outputs on public questions.\n\n" + "\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main():
    corpus_table()
    retrieval_table()
    ablation_table()
    progression_table()
    breakdown_table("by_topic", TOPICS, "F1 by question topic (public set, final corpus, short answers). "
                    "Topics are assigned by keyword rules on the question.")
    breakdown_table("by_type", ANSWER_TYPES, "F1 by expected answer type (public set, final corpus, short answers).")
    examples()


if __name__ == "__main__":
    main()

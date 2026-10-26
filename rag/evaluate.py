"""Score system outputs against reference answers with the assignment's token-based metrics.

    exact match, F1 (SQuAD, Rajpurkar et al. 2016, section 6.1), answer recall (share of the
    reference's tokens that appear in the prediction) and ROUGE-L F-measure (longest common
    subsequence). With several reference answers, each metric takes the best one.

Usage: python rag/evaluate.py system_outputs/public_rag.json [--refs public_evaluation_data.json]
"""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def normalize(s):
    """SQuAD normalization: lowercase, drop punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tokens(s):
    return normalize(s).split()


def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r:
        return float(p == r)
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def recall(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not r:
        return float(not p)
    return sum((Counter(p) & Counter(r)).values()) / len(r)


def rouge_l(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    # longest common subsequence, dynamic programming
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, rec = lcs / len(p), lcs / len(r)
    return 2 * precision * rec / (precision + rec)


METRICS = {"EM": exact_match, "F1": f1, "Recall": recall, "ROUGE-L": rouge_l}


def score(predictions, references):
    """predictions: {id: answer}; references: {id: [answers]}. Returns per-question and mean scores."""
    per_question = {}
    for qid, refs in references.items():
        pred = predictions.get(qid, "")
        per_question[qid] = {m: max(fn(pred, ref) for ref in refs) for m, fn in METRICS.items()}
    n = len(per_question)
    means = {m: sum(s[m] for s in per_question.values()) / n for m in METRICS}
    return means, per_question


def load_references(path):
    return {str(q["id"]): q["answers"] for q in json.load(open(path))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--refs", default=str(ROOT / "public_evaluation_data.json"))
    args = parser.parse_args()
    refs = load_references(args.refs)
    print(f"{'system':40s}" + "".join(f"{m:>9s}" for m in METRICS) + f"   (n={len(refs)})")
    for path in args.predictions:
        preds = {k: v for k, v in json.load(open(path)).items() if k != "andrewid"}
        means, _ = score(preds, refs)
        print(f"{Path(path).stem:40s}" + "".join(f"{means[m]:9.3f}" for m in METRICS))


if __name__ == "__main__":
    main()

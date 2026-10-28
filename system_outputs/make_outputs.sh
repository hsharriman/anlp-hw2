#!/bin/bash
# Answer the unseen test set in three answer lengths (only the best-scoring file is graded):
#   1 concise: one sentence of at most 15 words    best leaderboard score (67.84)
#   2 sentence: one sentence of at most 25 words   second-best leaderboard style (62.76 on corpus 3)
#   3 short: only the answer phrase                for short references (best on the public set, F1 71.0)
# All three retry refusals with more passages, then without retrieval.
# Usage: system_outputs/make_outputs.sh path/to/test_questions.json   (about 20 minutes per file for 250 questions)
set -e
cd "$(dirname "$0")/.."
Q=${1:?usage: $0 test_questions.json}
python rag/run.py --questions "$Q" --out system_outputs/system_output_1.json --style concise
python rag/run.py --questions "$Q" --out system_outputs/system_output_2.json --style sentence
python rag/run.py --questions "$Q" --out system_outputs/system_output_3.json --style short
mkdir -p rag/results/test
mv system_outputs/*.details.jsonl rag/results/test/
python - <<'PY'
import json
for n in (1, 2, 3):
    out = json.load(open(f"system_outputs/system_output_{n}.json"))
    assert "andrewid" not in out and all(isinstance(v, str) and v for v in out.values()), n
    print(f"system_output_{n}.json: {len(out)} answers")
PY

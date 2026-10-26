Development history: public set (157 questions) and leaderboard (LB, 150 questions; overall = mean of F1, recall, ROUGE-L and the rescaled judge score). Hybrid weighted retrieval, Qwen3-30B-A3B. Separate question sets: never compare a public number with a leaderboard number.

| System | Public F1 | Public F1, long refs (n=55) | LB overall | LB F1 | LB recall | LB ROUGE-L | LB judge |
|---|---|---|---|---|---|---|---|
| Corpus 1 (1,155 docs) | 64.6 | 40.8 | - | - | - | - | - |
| Corpus 2: + college schedules, Wikipedia lists (1,689 docs) | 69.9 | 42.8 | 51.03 | - | - | - | - |
| Corpus 3: + Visit Pittsburgh, calendars, social accounts (2,709 docs) | 70.2 | 40.3 | - | - | - | - | - |
| + detail-matching prompt (short answers) | 70.4 | 40.7 | 52.12 | 50.6 | 48.0 | 43.7 | 3.65 |
| Sentence answers | 38.4 | 50.5 | 62.76 | 55.0 | 77.0 | 50.0 | 3.70 |
| Concise sentences (submitted best) | 56.0 | 54.1 | 66.13 | 63.5 | 75.0 | 58.1 | 3.72 |

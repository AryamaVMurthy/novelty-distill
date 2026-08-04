# Corrected full A0 NoveltyBench gate

Job 19105 completed 100 curated prompts × 10 independent generations in
46m44s on node02. The summary validates the declared and observed seed schedule
17--26 for every prompt.

| Metric | Result |
|---|---:|
| Distinct@10 mean | 4.0800 |
| Distinct@10 standard error | 0.2545 |
| Utility@10 mean | 4.0657 |
| Utility@10 standard error | 0.1964 |
| Mean unique raw completions | 9.39 / 10 |
| Minimum unique raw completions | 2 / 10 |
| Prompts with any exact duplicate | 18 / 100 |

The 18 duplicate-containing prompts were manually inspected from the frozen
Inspect log. They are narrow fact/list requests: examples include naming a
protein source (nine identical “eggs” answers), an online retailer, a
Spanish-speaking country, a capital, or a star. This is expected model
concentration on constrained tasks, not the prior RNG defect. The old invalid
adapter repeated a long open-ended story because all K calls reused one seed;
the corrected log records seeds 17--26 independently for every prompt.

The gate therefore passes for protocol correctness. NoveltyBench remains a
generic functional-response diversity benchmark and cannot establish
scientific-idea novelty. The six clean trained seed-17 baselines were released
to this corrected protocol only after this gate passed.

Content identities:

- `summary.json`: SHA-256
  `4e4bd360f536c35fc1518544f4cb0ab739e7d97e5192f6b4fcda013d402cd868`.
- Raw `.eval`: SHA-256
  `ae1f3a6a5134300139bcfe5629a8797bfade5da856b7f1e2f063e287d52382af`.

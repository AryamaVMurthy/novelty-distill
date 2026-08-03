# Post-freeze exploratory treatments

These treatments were identified after the 19-method TOMATO-1k matrix was frozen and training had
started. They are secondary experiments: they cannot enter the preregistered contrast family or be
presented as if selected before the primary study.

## F1: diversity-aware reverse KL

Luong, Tran, and Chen, *Diversity-Aware Reverse Kullback-Leibler Divergence for Large Language
Model Distillation*, arXiv:2604.00223v1, 2026, decompose token-level reverse KL into a binary
target/non-target term and a normalized non-target term. Their DRKL loss replaces reverse KL's
student-dependent non-target weight with a fixed positive gamma. The proposed mechanism is directly
relevant here: ordinary reverse KL may amplify target-token confidence and suppress tail
supervision, which could reduce semantic output breadth.

The two declared treatments use gamma 0.5, inside the paper's reported stable range, and otherwise
match the frozen off-policy C2 controls:

- `F1-best1` versus `C2-best1` isolates the loss with one teacher target per prompt.
- `F1-diverse4` versus `C2-diverse4` tests whether the loss interacts with multi-target exposure.

Implementation follows Equation 9 independently in the TRL GKD path and evaluates logits in
float32 for numerical stability. Training uses the same Qwen3-4B student, Qwen3-14B teacher,
TOMATO-1k rows, seed 17, optimizer exposure, context policy, and generation controls as C2. The
source paper used smaller GPT-2/OPT pairs and generic instruction following, so its reported gains
are motivation, not an expected effect size for scientific ideation.

F1 must be evaluated with the frozen temporal protocol, including quality, mode recall/precision,
ClusterJSD, full threshold sensitivity, length stops, and the research-taste appendix. Any F1
comparison is labeled `secondary_exploratory`, with no confirmatory p-value claim.

Primary source: <https://arxiv.org/abs/2604.00223>

## TasteKD: future generation/selection treatment

Chen, Zhao, and Cohan's research-taste taxonomy motivates a later target-bank intervention that
conditions generation on underrepresented opportunity/paradigm cells or selects a quality-constrained
set covering those cells. It is not trained in the current gate because the taxonomy annotator must
first pass the two-human reliability threshold and the present study must measure whether a taste
gap exists on TOMATO. The frozen evaluation-only use is specified in
`reports/RESEARCH_TASTE_PROTOCOL.md`.

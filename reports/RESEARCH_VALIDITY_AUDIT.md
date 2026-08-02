# Research validity audit

This audit records methodological decisions made before inspecting TOMATO-1k model results. It is
not a results report and does not promote any baseline.

## Findings from primary literature

1. Scientific-idea evaluation is unusually exposed to knowledge leakage and weak ground truth.
   AI Idea Bench 2025 identifies leakage, open-ended grounding, and shallow feasibility assessment
   as central problems. This supports keeping TOMATO's temporal test split untouched, retaining the
   historical hypothesis as a separate reference, and never mixing train prompts into evaluation.
2. Divergent scientific thinking is multidimensional. LiveIdeaBench evaluates originality,
   feasibility, fluency, flexibility, and clarity and finds that scientific creativity is poorly
   predicted by ordinary capability metrics. This supports reporting judged quality/feasibility
   separately from semantic coverage rather than collapsing them into one headline score.
3. Embedding similarity is a proxy, not an expert semantic label. IDEAlign benchmarks embeddings,
   topic models, and LLM judges against expert idea-similarity decisions; most automatic metrics
   miss expert-relevant nuance, and even the best LLM judge remains a triage tool rather than a
   substitute for experts. Therefore cosine-cluster gains are exploratory until a blinded expert
   subset confirms the direction.
4. A multidimensional rubric is preferable to one opaque score, but calibration needs human
   labels. LLM-Rubric explicitly learns a calibration layer from human ratings. This project's
   fixed five-dimension judge provides reproducible proxy scores, not human-calibrated scientific
   utility estimates.

## Consequences for this study

- The 1,658 temporal-test IDs remain immutable and disjoint from every training view.
- K=16 samples are repeated observations within a prompt; prompt is the statistical unit.
- Quality, feasibility, ModeRecall, ModePrecision, ClusterJSD, and quality-adjusted coverage remain
  separate outcomes. No unique-string or single-threshold result is called semantic diversity.
- Teacher and student samples are embedded and clustered jointly, with the complete 0.70--0.95
  threshold curve retained.
- Finish reasons and completion-token diagnostics are preserved beside every score so truncation
  cannot masquerade as a method effect.
- Raw prompt-level metrics and all eight training-teacher samples are retained for reannotation.
- A central semantic-coverage claim requires a blinded expert subset, ideally using pairwise or
  odd-one-out judgments of mechanism/intervention equivalence. Until that exists, conclusions are
  explicitly limited to the fixed Qwen judge and Qwen embedding operational definitions.

## Primary source log

- Qiu et al., *AI Idea Bench 2025: AI Research Idea Generation Benchmark*:
  <https://arxiv.org/abs/2504.14191>
- Ruan et al., *Evaluating LLMs' Divergent Thinking Capabilities for Scientific Idea Generation
  with Minimal Context* (LiveIdeaBench): <https://arxiv.org/abs/2412.17596>
- Nam et al., *IDEAlign: Comparing Ideas of Large Language Models to Domain Experts*, EACL 2026:
  <https://aclanthology.org/2026.eacl-long.182/>
- Hashemi et al., *LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of
  Natural Language Texts*, ACL 2024: <https://aclanthology.org/2024.acl-long.745/>

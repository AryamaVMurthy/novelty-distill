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
5. Fluent judge rationales do not establish correct novelty decisions. RINoBench contains 1,381
   expert-judged research ideas and reports that leading LLMs can produce human-like rationales
   while their final novelty judgments still diverge substantially from the expert labels. The
   fixed judge here intentionally scores relevance, feasibility, soundness, clarity, and instruction
   compliance only; its mean must never be renamed or interpreted as a novelty score.
6. Standalone and comparative LLM judging can both create a scientific “novelty mirage.” RQ-Bench
   reports that LLM judges favor model-generated research questions while domain experts prefer the
   author-anchored questions, and that judges often miss narrow or source-bound generations. The A3
   historical target is therefore an author-anchored control, not a complete gold standard, and a
   promoted claim needs expert checks for narrowness/source dependence as well as mechanism novelty.
7. Diversity should be semantic rather than merely lexical. A 2025 ACL meta-evaluation finds that
   form-based diversity measures overestimate diversity, including for random strings, whereas
   content-based measures perform better. This supports the instructed semantic-embedding analysis
   and the decision not to use unique-string rate as an outcome, but it does not eliminate the need
   for expert validation of the embedding clusters.
8. Automated novelty assessment is stronger when grounded in retrieved literature. The Idea
   Novelty Checker uses retrieve-then-rerank, facet comparison, and expert-labeled demonstrations;
   its ablations support literature grounding rather than standalone rubric judging. This study has
   no frozen retrieval corpus for the temporal test, so literature novelty remains outside the
   automated claim boundary instead of being improvised after results are visible.
9. KL direction does not guarantee a mode-level result in practical LLM distillation. Wu et al.
   challenge the common claim that reverse KL is inherently mode-seeking and forward KL inherently
   mean-seeking in this setting; they instead identify different early emphasis on distribution
   tails and heads before eventual convergence. Consequently, C1-versus-C2 is a matched finite-step
   empirical contrast, not a proof of KL geometry from the sign of ModeRecall or ModePrecision.
10. Quality and diversity can trade off under instruction tuning and preference optimization.
    Le Bronnec et al. adapt distributional precision and recall to open-ended language generation
    and report this trade-off empirically. That supports retaining teacher-mode precision, recall,
    judge quality, and feasibility as separate outcomes rather than selecting a winner from one
    aggregate score.

## Consequences for this study

- The 1,658 temporal-test IDs remain immutable and disjoint from every training view.
- K=16 samples are repeated observations within a prompt; prompt is the statistical unit.
- Quality, feasibility, ModeRecall, ModePrecision, ClusterJSD, and quality-adjusted coverage remain
  separate outcomes. No unique-string or single-threshold result is called semantic diversity.
- Teacher samples are clustered first and their partition is frozen across methods. Students are
  assigned to the nearest qualifying teacher mode, while unmatched students are clustered into
  separately named novel modes. This avoids method-dependent teacher-mode merging through student
  bridge samples; the complete 0.70--0.95 threshold curve is still retained.
- Finish reasons and completion-token diagnostics are preserved beside every score so truncation
  cannot masquerade as a method effect.
- Raw prompt-level metrics and all eight training-teacher samples are retained for reannotation.
- A central semantic-coverage claim requires a blinded expert subset, ideally using pairwise or
  odd-one-out judgments of mechanism/intervention equivalence. Until that exists, conclusions are
  explicitly limited to the fixed Qwen judge and Qwen embedding operational definitions.
- Expert review for any promoted scientific-novelty claim must separately label literature-grounded
  novelty, narrow/source-bound restatement, and mechanism/intervention equivalence. The automated
  quality mean and the historical-target similarity are not substitutes for those labels.

## Primary source log

- Qiu et al., *AI Idea Bench 2025: AI Research Idea Generation Benchmark*:
  <https://arxiv.org/abs/2504.14191>
- Ruan et al., *Evaluating LLMs' Divergent Thinking Capabilities for Scientific Idea Generation
  with Minimal Context* (LiveIdeaBench): <https://arxiv.org/abs/2412.17596>
- Nam et al., *IDEAlign: Comparing Ideas of Large Language Models to Domain Experts*, EACL 2026:
  <https://aclanthology.org/2026.eacl-long.182/>
- Hashemi et al., *LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of
  Natural Language Texts*, ACL 2024: <https://aclanthology.org/2024.acl-long.745/>
- Schopf and Färber, *Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas*
  (RINoBench), LREC 2026: <https://arxiv.org/abs/2603.10303>
- Sinhahajari et al., *On the Limits of LLM-as-Judge for Scientific Novelty Assessment* (RQ-Bench),
  2026: <https://arxiv.org/abs/2606.12071>
- Zhang et al., *Evaluating the Evaluation of Diversity in Commonsense Generation*, ACL 2025:
  <https://aclanthology.org/2025.acl-long.1181/>
- Shahid et al., *Literature-Grounded Novelty Assessment of Scientific Ideas*, SDP 2025:
  <https://aclanthology.org/2025.sdp-1.9/>
- Wu et al., *Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language
  Models*, COLING 2025: <https://aclanthology.org/2025.coling-main.383/>
- Le Bronnec et al., *Exploring Precision and Recall to Assess the Quality and Diversity of LLMs*,
  ACL 2024: <https://aclanthology.org/2024.acl-long.616/>

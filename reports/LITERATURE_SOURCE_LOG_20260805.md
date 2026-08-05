# Literature source log

Search and verification date: 2026-08-05. Bibliographic metadata is centralized
in [`../references.bib`](../references.bib). This log records how each source
affected the audit rather than duplicating full citations.

| Source | Status | Used for | Important boundary |
|---|---|---|---|
| Kim and Rush, *Sequence-Level Knowledge Distillation* | EMNLP 2016 | Identify hard teacher-output SFT as a standard SeqKD baseline | Translation origin; does not validate scientific-idea outcomes |
| Agarwal et al., *On-Policy Distillation of Language Models* | ICLR 2024 | Define static versus student-trajectory GKD and alternative KL controls | Original experiments are not scientific ideation |
| Gu et al., *MiniLLM* | 2023 preprint | Motivate reverse-KL and its quality/mode trade-off | Reverse-KL mode behavior must still be measured empirically here |
| Ko et al., *DistiLLM* | ICML 2024 | Context for skew-KL and adaptive off-policy baselines | Not part of the corrected compact result |
| Si et al., *Can LLMs Generate Novel Research Ideas?* | ICLR 2025 | Gold-standard precedent for anonymized multi-expert review and evidence that LLM evaluators can lag human consistency | NLP topics and long project proposals differ from TOMATO |
| Zhang et al., *NoveltyBench* | 2025 preprint / COLM-era benchmark | Verify K=10 independent sampling, functional distinctness, and utility evaluation | Generic response diversity, not scientific novelty |
| Yang and Bing, *MOOSE-Star* | ICML 2026 / arXiv v4 | Source paper for TOMATO-Star and its temporal-split claims | Repository audit found target and earliest-date problems not caught by the paper's declared checks |
| Chen et al., *Measuring the Gap Between Human and LLM Research Ideas* | July 2026 preprint | Secondary opportunity-pattern/method-paradigm taxonomy and distributional metrics | Needs local human calibration; not a quality or novelty oracle |
| Chalamalasetti and Vajjala, *LLM Judges Can Be Too Generous When There Is No Reference Answer* | July 2026 preprint | Motivate task-capability calibration, deliberately incorrect controls, and reference-sensitivity checks for an open-ended judge | Multilingual QA rather than scientific ideation; preprint evidence, not a universal correction |
| Sinhahajari et al., *On the Limits of LLM-as-Judge for Scientific Novelty Assessment* | June 2026 preprint | Direct evidence that standalone and comparative LLM novelty judgments can disagree with domain experts | Research-question benchmark rather than full experimental hypotheses; not yet peer reviewed |
| Mukherjee et al., *The Geometry of LLM-as-Judge* | June 2026 preprint | Motivate explicit score-spread/ceiling diagnostics and the rule that inter-LLM agreement is not human alignment | Indic community datasets rather than scientific ideas; geometric findings need local human anchoring |
| Qwen Team, Qwen3 official blog | Official model source | Establish the 2025-04-29 release date and recommended decoding context | Release date is not a documented pretraining cutoff |
| NLM, PubMed User Guide and E-utilities documentation | Official database documentation | Define electronic, print, and ahead-of-print availability semantics and metadata retrieval | Public availability does not prove inclusion in model training |
| DeepInfra structured-output and model APIs | Official provider documentation | Design a strict-JSON, different-family calibration judge | LLM--LLM agreement is sensitivity evidence, not expert validity |

Only primary papers, official proceedings, official model documentation, and
official database/provider documentation support claims in the dated validity
report. The three June/July 2026 judge papers above are explicitly labeled
preprints and provide secondary risk/diagnostic context; they do not establish
the validity of this project's judge or replace its pending human calibration.

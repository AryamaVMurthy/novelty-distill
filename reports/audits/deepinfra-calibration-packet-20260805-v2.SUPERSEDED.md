# DeepInfra calibration packet v2 — superseded before paid use

Packet v2 (`e1fd370a...`) fixed the original prompt-pairing defect and added
`B2a`, but a final pre-spend audit found that its 48 hidden repeats were
unevenly allocated by method: counts ranged from 2 for `A1` to 11 for `B2b`.
That supports only pooled repeat reliability and needlessly weakens
method-balanced auditing.

No v2 item was sent to DeepInfra. Packet v3 retains the same 480 originals,
60 unique paired prompt/sample slots, and four pooled Qwen-score strata, but
selects exactly six hidden repeats per method. Use only the v3 manifest.

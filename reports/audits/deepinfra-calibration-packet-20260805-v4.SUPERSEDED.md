# DeepInfra calibration packet v4 — superseded

Date: 2026-08-05 (Asia/Kolkata)

Packet v4 migrated the frozen 528-entry paired sample from the unsupported
`json_schema` transport to DeepInfra's provider-compatible `json_object`
transport and added an explicit seven-key output contract. Its immutable
packet SHA-256 is
`f5ab607214618273356d834f7a1e2502705f99b6a0cbb60aa1ec949131c47415` and
its protocol hash is
`26ac8b4c00fe87b3b4ab3af31665699b38db52cd24ee190eb9dedf428d42f559`.
The frozen-entry identity remained
`88ddca5c6db7d69b7c65f27650ac5ae325bda684eba85822f395996ce42cc263`.

One accepted paid record was persisted before the provider was observed to
intermittently wrap otherwise valid JSON in a single Markdown JSON fence. The
persisted response has SHA-256
`e7f31fdab660e54bf4554266db7e986b5f991939ebdced2ada2227d631cfbb49`,
finish reason `stop`, 749 prompt tokens, 60 completion tokens, and provider
estimated cost USD 0.00009410. It is not part of the accepted calibration
because protocol identity changed before the full run.

Packet v4 was superseded by v5, which keeps all 528 sample entries byte-for-byte
unchanged but uses protocol v3: it accepts only raw JSON or one whole-response
JSON fence, then applies the same strict schema validation. No scientific
sample, method balance, prompt slot, hidden repeat, or candidate text changed.


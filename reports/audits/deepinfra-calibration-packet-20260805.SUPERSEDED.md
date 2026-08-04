# Superseded independent-judge packet

The packet identified by
`deepinfra-calibration-packet-20260805.manifest.json` was retired before any
paid request was sent. Its 60 paired sample slots came from 58 unique prompts:
two prompts appeared at two sample indices. The sampler now enforces one slot
per prompt, and the replacement packet will also include the missing `B2a`
random-1 sequence-KD baseline. The original manifest is retained unchanged as
an audit record.

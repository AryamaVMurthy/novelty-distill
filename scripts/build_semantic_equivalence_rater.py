#!/usr/bin/env python3
"""Build a self-contained source-blinded browser UI for semantic pair labels."""

import argparse
import json
import re
from pathlib import Path

from novelty_distill.evaluation.semantic_calibration import (
    PublicSemanticPair,
    semantic_calibration_protocol_hash,
)


def _safe_json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rater-id", required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.rater_id) is None:
        raise ValueError("--rater-id must be a short path-safe identifier")
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    if packet.get("schema_version") != 1:
        raise ValueError("unsupported public semantic packet schema")
    protocol_hash = semantic_calibration_protocol_hash()
    if packet.get("protocol_hash") != protocol_hash:
        raise ValueError("public semantic packet protocol hash does not match this checkout")
    if packet.get("labels") != ["equivalent", "not_equivalent", "uncertain"]:
        raise ValueError("public semantic packet labels changed")
    entries = [PublicSemanticPair.model_validate(item) for item in packet.get("entries", [])]
    if not entries or len({entry.blind_id for entry in entries}) != len(entries):
        raise ValueError("public semantic packet must contain unique entries")
    public_packet = {
        "protocol_hash": protocol_hash,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    title = f"Semantic equivalence calibration — {args.rater_id}"
    packet_json = _safe_json_for_script(public_packet)
    rater_json = _safe_json_for_script(args.rater_id)
    title_html = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_html}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1100px; padding: 1.25rem; line-height: 1.45; }}
header {{ position: sticky; top: 0; background: Canvas; padding: .5rem 0; z-index: 1; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.panel {{ border: 1px solid GrayText; border-radius: .5rem; padding: 1rem; white-space: pre-wrap; }}
.task {{ max-height: 19rem; overflow: auto; }}
.answer {{ max-height: 32rem; overflow: auto; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: 1rem 0; }}
label {{ cursor: pointer; }}
textarea {{ width: 100%; min-height: 5rem; box-sizing: border-box; }}
button {{ padding: .55rem .9rem; }}
.error {{ color: #c33; min-height: 1.5rem; }}
@media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{title_html}</h1>
  <div id="progress"></div>
</header>
<section>
  <h2>Instruction</h2><div id="instruction" class="panel"></div>
  <h2>Task</h2><div id="task" class="panel task"></div>
  <div class="grid">
    <div><h2>Answer A</h2><div id="answer-a" class="panel answer"></div></div>
    <div><h2>Answer B</h2><div id="answer-b" class="panel answer"></div></div>
  </div>
  <div class="controls" id="labels">
    <strong>Label:</strong>
    <label><input type="radio" name="label" value="equivalent"> Equivalent</label>
    <label><input type="radio" name="label" value="not_equivalent"> Not equivalent</label>
    <label><input type="radio" name="label" value="uncertain"> Uncertain</label>
    <label>Confidence
      <select id="confidence">
        <option value="">—</option><option>1</option><option>2</option><option>3</option>
      </select>
    </label>
  </div>
  <label for="rationale"><strong>Rationale (optional, max 1000 characters)</strong></label>
  <textarea id="rationale" maxlength="1000"></textarea>
  <div class="error" id="error"></div>
  <div class="controls">
    <button id="previous" type="button">Previous</button>
    <button id="next" type="button">Save and next</button>
    <button id="export" type="button">Export complete JSONL</button>
  </div>
</section>
<script>
"use strict";
const packet = {packet_json};
const raterId = {rater_json};
const storageKey = `semantic-equivalence:${{packet.protocol_hash}}:${{raterId}}`;
let labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
let index = 0;
const byId = id => document.getElementById(id);

function saveCurrent(requireComplete = false) {{
  const entry = packet.entries[index];
  const selected = document.querySelector('input[name="label"]:checked');
  const confidence = byId("confidence").value;
  const rationale = byId("rationale").value.trim();
  if (requireComplete && (!selected || !confidence)) {{
    byId("error").textContent = "Choose a label and confidence before continuing.";
    return false;
  }}
  if (selected && confidence) {{
    labels[entry.blind_id] = {{
      blind_id: entry.blind_id,
      label: selected.value,
      confidence: Number(confidence),
      rationale,
    }};
    localStorage.setItem(storageKey, JSON.stringify(labels));
  }}
  byId("error").textContent = "";
  return true;
}}

function render() {{
  const entry = packet.entries[index];
  const saved = labels[entry.blind_id] || {{}};
  const savedCount = Object.keys(labels).length;
  byId("progress").textContent =
    `Item ${{index + 1}} / ${{packet.entries.length}} · ${{savedCount}} saved · ` +
    entry.blind_id;
  byId("instruction").textContent = entry.instruction;
  byId("task").textContent = entry.task;
  byId("answer-a").textContent = entry.answer_a;
  byId("answer-b").textContent = entry.answer_b;
  document.querySelectorAll('input[name="label"]').forEach(input => {{
    input.checked = input.value === saved.label;
  }});
  byId("confidence").value = saved.confidence || "";
  byId("rationale").value = saved.rationale || "";
  byId("previous").disabled = index === 0;
  byId("next").textContent = index + 1 === packet.entries.length ? "Save" : "Save and next";
  byId("error").textContent = "";
  window.scrollTo({{top: 0, behavior: "instant"}});
}}

byId("previous").addEventListener("click", () => {{ saveCurrent(false); index -= 1; render(); }});
byId("next").addEventListener("click", () => {{
  if (!saveCurrent(true)) return;
  if (index + 1 < packet.entries.length) index += 1;
  render();
}});
byId("export").addEventListener("click", () => {{
  if (!saveCurrent(false)) return;
  const missing = packet.entries.filter(entry => !labels[entry.blind_id]);
  if (missing.length) {{
    byId("error").textContent = `${{missing.length}} items remain unlabeled; export is disabled.`;
    return;
  }}
  const rows = packet.entries
    .map(entry => JSON.stringify(labels[entry.blind_id]))
    .join("\\n") + "\\n";
  const blob = new Blob([rows], {{type: "application/jsonl"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `labels-${{raterId}}.jsonl`;
  link.click();
  URL.revokeObjectURL(link.href);
}});
render();
</script>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"entries": len(entries), "output": str(args.output), "rater": args.rater_id}))


if __name__ == "__main__":
    main()

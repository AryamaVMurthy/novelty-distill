#!/usr/bin/env python3
"""Build a source-blinded UI for semantic-equivalence adjudication."""

import argparse
import hashlib
import json
from pathlib import Path

from novelty_distill.evaluation.semantic_calibration import (
    HumanEquivalenceLabel,
    PrivateSemanticPair,
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


def _load_labels(path: Path) -> dict[str, HumanEquivalenceLabel]:
    labels: dict[str, HumanEquivalenceLabel] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            label = HumanEquivalenceLabel.model_validate(json.loads(line))
            if label.blind_id in labels:
                raise ValueError(f"duplicate label {label.blind_id} in {path}")
            labels[label.blind_id] = label
    if not labels:
        raise ValueError(f"label file is empty: {path}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--rater-one", type=Path, required=True)
    parser.add_argument("--rater-two", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_hash = semantic_calibration_protocol_hash()
    public_packet = json.loads(args.packet.read_text(encoding="utf-8"))
    private_packet = json.loads(args.private_key.read_text(encoding="utf-8"))
    for name, packet in (("public", public_packet), ("private", private_packet)):
        if packet.get("schema_version") != 1:
            raise ValueError(f"unsupported {name} semantic packet schema")
        if packet.get("protocol_hash") != protocol_hash:
            raise ValueError(f"{name} semantic packet protocol hash does not match this checkout")
    if public_packet.get("labels") != ["equivalent", "not_equivalent", "uncertain"]:
        raise ValueError("public semantic packet labels changed")

    public_entries = [
        PublicSemanticPair.model_validate(item) for item in public_packet.get("entries", [])
    ]
    private_entries = [
        PrivateSemanticPair.model_validate(item) for item in private_packet.get("entries", [])
    ]
    public_by_id = {entry.blind_id: entry for entry in public_entries}
    private_by_id = {entry.blind_id: entry for entry in private_entries}
    if (
        not public_entries
        or len(public_by_id) != len(public_entries)
        or len(private_by_id) != len(private_entries)
        or set(public_by_id) != set(private_by_id)
    ):
        raise ValueError("public and private packets must contain the same unique blind ids")

    first_labels = _load_labels(args.rater_one)
    second_labels = _load_labels(args.rater_two)
    expected_ids = set(public_by_id)
    for labels in (first_labels, second_labels):
        if set(labels) != expected_ids or any(
            blind_id != label.blind_id for blind_id, label in labels.items()
        ):
            raise ValueError("each rater file must exactly cover every packet blind id")

    flagged_ids = [
        entry.blind_id
        for entry in private_entries
        if entry.repeat_of is None
        and (
            first_labels[entry.blind_id].label != second_labels[entry.blind_id].label
            or first_labels[entry.blind_id].label == "uncertain"
            or second_labels[entry.blind_id].label == "uncertain"
        )
    ]
    adjudication_entries = [public_by_id[blind_id] for blind_id in flagged_ids]
    packet_identity = hashlib.sha256(
        (protocol_hash + "\0" + "\0".join(flagged_ids)).encode()
    ).hexdigest()
    embedded_packet = {
        "protocol_hash": protocol_hash,
        "packet_identity": packet_identity,
        "entries": [entry.model_dump(mode="json") for entry in adjudication_entries],
    }
    packet_json = _safe_json_for_script(embedded_packet)
    title = "Semantic equivalence adjudication"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
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
<header><h1>{title}</h1><div id="progress"></div></header>
<section id="empty" hidden>
  <p>No original pair requires adjudication. Run the analyzer without an adjudication file.</p>
</section>
<section id="form">
  <h2>Instruction</h2>
  <div class="panel">Make a final source-blinded decision about whether the two answers express
  the same central scientific idea for the task. Choose equivalent only when their central
  mechanism, intervention or object of study, and decisive experimental test would lead to the
  same substantive research project. Do not judge quality, feasibility, or global novelty.</div>
  <h2>Task</h2><div id="task" class="panel task"></div>
  <div class="grid">
    <div><h2>Answer A</h2><div id="answer-a" class="panel answer"></div></div>
    <div><h2>Answer B</h2><div id="answer-b" class="panel answer"></div></div>
  </div>
  <div class="controls">
    <strong>Final label:</strong>
    <label><input type="radio" name="label" value="equivalent"> Equivalent</label>
    <label><input type="radio" name="label" value="not_equivalent"> Not equivalent</label>
  </div>
  <label for="rationale"><strong>Rationale (optional, max 1000 characters)</strong></label>
  <textarea id="rationale" maxlength="1000"></textarea>
  <div class="error" id="error"></div>
  <div class="controls">
    <button id="previous" type="button">Previous</button>
    <button id="next" type="button">Save and next</button>
    <button id="export" type="button">Export complete adjudication JSONL</button>
  </div>
</section>
<script>
"use strict";
const packet = {packet_json};
const storageKey = `semantic-equivalence-adjudication:${{packet.packet_identity}}`;
let labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
let index = 0;
const byId = id => document.getElementById(id);

function saveCurrent(requireComplete = false) {{
  const entry = packet.entries[index];
  const selected = document.querySelector('input[name="label"]:checked');
  const rationale = byId("rationale").value.trim();
  if (requireComplete && !selected) {{
    byId("error").textContent = "Choose a final label before continuing.";
    return false;
  }}
  if (selected) {{
    labels[entry.blind_id] = {{blind_id: entry.blind_id, label: selected.value, rationale}};
    localStorage.setItem(storageKey, JSON.stringify(labels));
  }}
  byId("error").textContent = "";
  return true;
}}

function render() {{
  if (!packet.entries.length) {{
    byId("form").hidden = true;
    byId("empty").hidden = false;
    byId("progress").textContent = "No adjudication required";
    return;
  }}
  const entry = packet.entries[index];
  const saved = labels[entry.blind_id] || {{}};
  byId("progress").textContent =
    `Item ${{index + 1}} / ${{packet.entries.length}} · ${{Object.keys(labels).length}} saved · ` +
    entry.blind_id;
  byId("task").textContent = entry.task;
  byId("answer-a").textContent = entry.answer_a;
  byId("answer-b").textContent = entry.answer_b;
  document.querySelectorAll('input[name="label"]').forEach(input => {{
    input.checked = input.value === saved.label;
  }});
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
    const noun = missing.length === 1 ? "item remains" : "items remain";
    byId("error").textContent = `${{missing.length}} ${{noun}} unresolved; export is disabled.`;
    return;
  }}
  const rows = packet.entries
    .map(entry => JSON.stringify(labels[entry.blind_id]))
    .join("\\n") + "\\n";
  const blob = new Blob([rows], {{type: "application/jsonl"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "adjudication.jsonl";
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
    print(
        json.dumps(
            {
                "flagged_originals": len(adjudication_entries),
                "output": str(args.output),
                "packet_identity": packet_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze PubMed first-public dates and audit TOMATO temporal leakage strata."""

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from novelty_distill.data.pubmed_temporal import (
    audit_pubmed_availability,
    parse_pubmed_xml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", action="append", required=True)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--email")
    parser.add_argument("--tool", default="novelty-distill-validity-audit")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _fetch(pmids: list[str], *, email: str, tool: str) -> dict[str, dict[str, Any]]:
    if not email.strip():
        raise ValueError("--email is required with --fetch-missing")
    fetched: dict[str, dict[str, Any]] = {}
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    for offset in range(0, len(pmids), 200):
        batch = pmids[offset : offset + 200]
        body = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "tool": tool, "email": email}
        ).encode()
        request = urllib.request.Request(endpoint, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            parsed = parse_pubmed_xml(response.read().decode("utf-8"))
        fetched.update(parsed)
        if offset + 200 < len(pmids):
            time.sleep(0.34)
    return fetched


def main() -> None:
    args = parse_args()
    rows = tuple(
        json.loads(line)
        for path in args.input
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    cache: dict[str, Any]
    if args.metadata_cache.exists():
        cache = json.loads(args.metadata_cache.read_text(encoding="utf-8"))
        if cache.get("schema_version") != 1 or not isinstance(cache.get("records"), dict):
            raise ValueError("unsupported PubMed metadata cache schema")
    else:
        cache = {"schema_version": 1, "records": {}}
    pmids = sorted(
        {
            identifier.rsplit("_", 1)[-1]
            for row in rows
            for identifier in (str(row["id"]), *(str(v) for v in row.get("source_ids", ())))
            if identifier.rsplit("_", 1)[-1].isdigit()
        }
    )
    missing = [pmid for pmid in pmids if pmid not in cache["records"]]
    if missing and args.fetch_missing:
        cache["records"].update(_fetch(missing, email=args.email or "", tool=args.tool))
        cache["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        cache["source"] = "NCBI PubMed EFetch XML"
        _atomic_json(args.metadata_cache, cache)
    audit = audit_pubmed_availability(
        rows=rows,
        metadata=cache["records"],
        cutoffs=tuple(args.cutoff),
        allow_missing=False,
    )
    audit["provenance"] = {
        "input_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.input
        },
        "metadata_sha256": hashlib.sha256(args.metadata_cache.read_bytes()).hexdigest(),
        "metadata_path": str(args.metadata_cache.resolve()),
    }
    _atomic_json(args.output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

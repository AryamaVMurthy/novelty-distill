"""PubMed first-public-date extraction and temporal-leakage stratification."""

import calendar
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

_PUBLIC_STATUSES = {"aheadofprint", "epublish", "ppublish", "ecollection"}
_STATUS_PRIORITY = {
    "epublish": 0,
    "aheadofprint": 1,
    "ppublish": 2,
    "ecollection": 3,
}


def _date_from_node(node: ET.Element) -> tuple[date, str] | None:
    year_text = node.findtext("Year")
    if not year_text:
        medline = node.findtext("MedlineDate") or ""
        match = re.search(r"\b(19|20)\d{2}\b", medline)
        year_text = match.group(0) if match else None
    if not year_text or not year_text.isdigit():
        return None
    month_text = (node.findtext("Month") or "1").strip()
    if month_text.isdigit():
        month = int(month_text)
    else:
        month_lookup = {
            name.lower(): index
            for index, name in enumerate(calendar.month_name)
            if name
        }
        month_lookup.update(
            {name.lower(): index for index, name in enumerate(calendar.month_abbr) if name}
        )
        month = month_lookup.get(month_text[:3].lower(), 1)
    day_text = (node.findtext("Day") or "1").strip()
    day = int(day_text) if day_text.isdigit() else 1
    precision = "day" if node.findtext("Day") else "month" if node.findtext("Month") else "year"
    try:
        return date(int(year_text), month, day), precision
    except ValueError as error:
        raise ValueError(f"invalid PubMed publication date: {year_text}-{month}-{day}") from error


def parse_pubmed_xml(xml_text: str) -> dict[str, dict[str, Any]]:
    """Extract the earliest publicly available date from PubMed XML records."""

    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext("./MedlineCitation/PMID") or "").strip()
        if not pmid or not pmid.isdigit():
            raise ValueError("PubMed article is missing a numeric PMID")
        candidates: list[dict[str, str]] = []
        for node in article.findall("./MedlineCitation/Article/ArticleDate"):
            parsed = _date_from_node(node)
            if parsed is not None and node.attrib.get("DateType", "").lower() == "electronic":
                value, precision = parsed
                candidates.append(
                    {
                        "date": value.isoformat(),
                        "status": "epublish",
                        "precision": precision,
                        "source": "ArticleDate",
                    }
                )
        for node in article.findall("./PubmedData/History/PubMedPubDate"):
            status = node.attrib.get("PubStatus", "").lower()
            parsed = _date_from_node(node)
            if parsed is not None and status in _PUBLIC_STATUSES:
                value, precision = parsed
                candidates.append(
                    {
                        "date": value.isoformat(),
                        "status": status,
                        "precision": precision,
                        "source": "PubMedHistory",
                    }
                )
        journal_date = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
        if journal_date is not None and (parsed := _date_from_node(journal_date)) is not None:
            value, precision = parsed
            candidates.append(
                {
                    "date": value.isoformat(),
                    "status": "ppublish",
                    "precision": precision,
                    "source": "JournalIssue",
                }
            )
        if not candidates:
            raise ValueError(f"PubMed article {pmid} has no public publication date")
        unique = {
            (item["date"], item["status"], item["precision"], item["source"]): item
            for item in candidates
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                item["date"],
                _STATUS_PRIORITY[item["status"]],
                item["source"],
            ),
        )
        earliest = ordered[0]
        records[pmid] = {
            "pmid": pmid,
            "earliest_public_date": earliest["date"],
            "earliest_public_status": earliest["status"],
            "earliest_public_precision": earliest["precision"],
            "date_candidates": ordered,
        }
    return records


def _row_pmid(row: Mapping[str, Any]) -> str:
    identifiers = (str(row.get("id", "")), *(str(value) for value in row.get("source_ids", ())))
    matches = {
        match.group(1)
        for identifier in identifiers
        if (match := re.search(r"(?:^|_)(\d{7,9})$", identifier)) is not None
    }
    if len(matches) != 1:
        raise ValueError(f"row {row.get('id')!r} does not resolve to exactly one PMID")
    return matches.pop()


def audit_pubmed_availability(
    *,
    rows: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    cutoffs: Sequence[str],
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Count rows publicly available by each declared cutoff, split by dataset split."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("temporal audit requires at least one row")
    parsed_cutoffs = tuple(date.fromisoformat(value) for value in cutoffs)
    if not parsed_cutoffs or len(parsed_cutoffs) != len(set(parsed_cutoffs)):
        raise ValueError("temporal cutoffs must be unique ISO dates")
    resolved = tuple((row, _row_pmid(row)) for row in materialized)
    missing = sorted({pmid for _, pmid in resolved if pmid not in metadata})
    if missing and not allow_missing:
        raise ValueError(f"missing PubMed metadata for {len(missing)} PMIDs: {missing[:5]}")

    by_split: dict[str, dict[str, Any]] = {}
    earliest_cutoff = min(parsed_cutoffs)
    for split in sorted({str(row.get("split", "")) for row, _ in resolved}):
        if not split:
            raise ValueError("temporal audit rows require a non-empty split")
        split_rows = tuple((row, pmid) for row, pmid in resolved if row.get("split") == split)
        available = {
            cutoff.isoformat(): sum(
                pmid in metadata
                and date.fromisoformat(str(metadata[pmid]["earliest_public_date"])) <= cutoff
                for _, pmid in split_rows
            )
            for cutoff in parsed_cutoffs
        }
        examples = [
            {
                "id": str(row["id"]),
                "pmid": pmid,
                "date": str(metadata[pmid]["earliest_public_date"]),
            }
            for row, pmid in split_rows
            if pmid in metadata
            and date.fromisoformat(str(metadata[pmid]["earliest_public_date"]))
            <= earliest_cutoff
        ][:20]
        by_split[split] = {
            "rows": len(split_rows),
            "available_by_cutoff": available,
            "available_fraction_by_cutoff": {
                cutoff: count / len(split_rows) for cutoff, count in available.items()
            },
            "examples_available_by_earliest_cutoff": examples,
        }
    return {
        "schema_version": 1,
        "policy": "pubmed-earliest-public-date-v1",
        "rows": len(materialized),
        "unique_pmids": len({pmid for _, pmid in resolved}),
        "cutoffs": [value.isoformat() for value in parsed_cutoffs],
        "missing_pmids": missing,
        "by_split": by_split,
        "interpretation": (
            "Rows available by a model-relevant cutoff are not pretraining-leakage-safe; "
            "this audit does not prove that a model saw any paper."
        ),
    }

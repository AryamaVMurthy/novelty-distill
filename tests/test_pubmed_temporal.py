from pathlib import Path

import pytest

from novelty_distill.data.pubmed_temporal import (
    audit_pubmed_availability,
    parse_pubmed_xml,
)

PUBMED_XML = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation><PMID>37032452</PMID><Article>
      <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
      <ArticleDate DateType="Electronic">
        <Year>2023</Year><Month>04</Month><Day>10</Day>
      </ArticleDate>
    </Article></MedlineCitation>
    <PubmedData><History>
      <PubMedPubDate PubStatus="aheadofprint">
        <Year>2023</Year><Month>04</Month><Day>10</Day>
      </PubMedPubDate>
      <PubMedPubDate PubStatus="pubmed">
        <Year>2023</Year><Month>04</Month><Day>14</Day>
      </PubMedPubDate>
    </History></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation><PMID>40000000</PMID><Article>
      <Journal><JournalIssue><PubDate><Year>2025</Year><Month>Sep</Month></PubDate></JournalIssue></Journal>
    </Article></MedlineCitation>
    <PubmedData><History>
      <PubMedPubDate PubStatus="ppublish">
        <Year>2025</Year><Month>09</Month><Day>20</Day>
      </PubMedPubDate>
    </History></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_parser_prefers_first_public_date_over_issue_year() -> None:
    records = parse_pubmed_xml(PUBMED_XML)

    assert records["37032452"]["earliest_public_date"] == "2023-04-10"
    assert records["37032452"]["earliest_public_status"] == "epublish"
    assert records["40000000"]["earliest_public_date"] == "2025-09-01"
    assert records["40000000"]["earliest_public_precision"] == "month"


def test_temporal_audit_stratifies_actual_public_availability() -> None:
    records = parse_pubmed_xml(PUBMED_XML)
    rows = (
        {"id": "2025_37032452", "split": "test", "source_ids": ["2025_37032452"]},
        {"id": "2025_40000000", "split": "test", "source_ids": ["2025_40000000"]},
    )

    audit = audit_pubmed_availability(
        rows=rows,
        metadata=records,
        cutoffs=("2023-12-31", "2025-04-29"),
    )

    assert audit["rows"] == 2
    assert audit["missing_pmids"] == []
    assert audit["by_split"]["test"]["available_by_cutoff"] == {
        "2023-12-31": 1,
        "2025-04-29": 1,
    }
    assert audit["by_split"]["test"]["examples_available_by_earliest_cutoff"] == [
        {"id": "2025_37032452", "pmid": "37032452", "date": "2023-04-10"}
    ]


def test_temporal_audit_fails_closed_on_missing_pubmed_metadata() -> None:
    with pytest.raises(ValueError, match="missing PubMed metadata"):
        audit_pubmed_availability(
            rows=({"id": "2025_37032452", "split": "test"},),
            metadata={},
            cutoffs=("2024-12-31",),
        )


def test_temporal_cli_contract_is_content_bound() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/audit_pubmed_availability.py"
    source = script.read_text(encoding="utf-8")

    assert "--metadata-cache" in source
    assert "--fetch-missing" in source
    assert '"input_sha256"' in source
    assert '"metadata_sha256"' in source
    assert "allow_missing=False" in source

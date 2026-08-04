"""Corpus-level integrity diagnostics for historical TOMATO targets."""

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from novelty_distill.data.tomato import CanonicalExample

_WORD = re.compile(r"[a-z0-9]+")
_COVERAGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "through",
    "to",
    "using",
    "via",
    "with",
}


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.lower()))


def _content_tokens(tokens: Iterable[str]) -> set[str]:
    return {token for token in tokens if token not in _COVERAGE_STOPWORDS and len(token) > 2}


def audit_target_integrity(
    rows: Iterable[CanonicalExample],
    *,
    ngram_size: int = 12,
    minimum_document_frequency: int = 3,
    maximum_prompt_token_coverage: float = 0.25,
    word_limit: int = 300,
    maximum_candidates: int = 100,
) -> dict[str, Any]:
    """Find repeated target passages that are unsupported by the associated prompts."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("target integrity audit requires at least one row")
    if ngram_size < 3 or minimum_document_frequency < 2 or word_limit <= 0:
        raise ValueError("target integrity audit controls are invalid")
    if not 0 <= maximum_prompt_token_coverage <= 1 or maximum_candidates <= 0:
        raise ValueError("target integrity audit thresholds are invalid")
    ids = [row.id for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("target integrity audit requires unique row IDs")

    target_tokens = {row.id: _tokens(row.human_target) for row in materialized}
    prompt_tokens = {row.id: _content_tokens(_tokens(row.student_prompt)) for row in materialized}
    ngram_documents: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in materialized:
        tokens = target_tokens[row.id]
        for index in range(len(tokens) - ngram_size + 1):
            ngram_documents[tokens[index : index + ngram_size]].add(row.id)

    grouped_ngrams: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for ngram, document_ids in ngram_documents.items():
        if len(document_ids) >= minimum_document_frequency:
            grouped_ngrams[tuple(sorted(document_ids))].append(ngram)

    candidates: list[dict[str, Any]] = []
    suspicious_row_ids: set[str] = set()
    for document_ids, ngrams in grouped_ngrams.items():
        repeated_content = _content_tokens(token for ngram in ngrams for token in ngram)
        if not repeated_content:
            continue
        coverage = {
            row_id: len(repeated_content & prompt_tokens[row_id]) / len(repeated_content)
            for row_id in document_ids
        }
        unsupported = sorted(
            row_id
            for row_id, value in coverage.items()
            if value <= maximum_prompt_token_coverage
        )
        if len(unsupported) < minimum_document_frequency:
            continue
        suspicious_row_ids.update(unsupported)
        representative = min(ngrams)
        candidates.append(
            {
                "ngram": " ".join(representative),
                "document_frequency": len(document_ids),
                "document_ids": list(document_ids),
                "unsupported_prompt_ids": unsupported,
                "prompt_token_coverage": {
                    row_id: coverage[row_id] for row_id in sorted(coverage)
                },
                "repeated_content_tokens": sorted(repeated_content),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            -int(candidate["document_frequency"]),
            -len(candidate["unsupported_prompt_ids"]),
            str(candidate["ngram"]),
        )
    )

    target_word_counts = [len(tokens) for tokens in target_tokens.values()]
    target_frequencies: dict[str, int] = defaultdict(int)
    for row in materialized:
        target_frequencies[row.human_target.strip()] += 1
    return {
        "schema_version": 1,
        "rows": len(materialized),
        "ngram_size": ngram_size,
        "minimum_document_frequency": minimum_document_frequency,
        "maximum_prompt_token_coverage": maximum_prompt_token_coverage,
        "word_limit": word_limit,
        "target_equals_privileged_count": sum(
            row.human_target.strip()
            == row.privileged_context.historical_hypothesis.strip()
            for row in materialized
        ),
        "exact_duplicate_target_row_count": sum(
            count for count in target_frequencies.values() if count > 1
        ),
        "over_word_limit_count": sum(count > word_limit for count in target_word_counts),
        "target_words_mean": statistics.fmean(target_word_counts),
        "target_words_median": statistics.median(target_word_counts),
        "suspicious_row_ids": sorted(suspicious_row_ids),
        "suspicious_repeated_ngrams": candidates[:maximum_candidates],
        "candidate_groups_total": len(candidates),
        "candidate_groups_truncated": len(candidates) > maximum_candidates,
    }

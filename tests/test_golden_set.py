"""The committed golden set (data/eval/golden.jsonl) must satisfy the ADR-0023 rules.

Runs on the real files; needs no Qdrant (CI has none). Whether every label points at a
section that exists in the collection is checked by the runner's pre-flight, not here.
"""

from pathlib import Path

import pytest

from app.evaluation.golden import (
    QUOTAS,
    GoldenEntry,
    QuestionType,
    load_golden,
    load_manifests,
    validate_golden,
)
from app.workflow.classify import classify_node

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "data" / "eval" / "golden.jsonl"
PAPERS_DIR = ROOT / "data" / "papers"


@pytest.fixture(scope="module")
def entries() -> list[GoldenEntry]:
    return load_golden(GOLDEN_PATH)


def test_golden_set_has_no_rule_violations(entries: list[GoldenEntry]) -> None:
    violations = validate_golden(entries, load_manifests(PAPERS_DIR))
    assert violations == []


def test_golden_set_has_sixty_five_entries_with_fixed_quotas(
    entries: list[GoldenEntry],
) -> None:
    assert len(entries) == 65
    assert sum(QUOTAS.values()) == 65
    for qtype, quota in QUOTAS.items():
        assert sum(1 for e in entries if e.type is qtype) == quota


def test_no_question_is_flagged_unsafe(entries: list[GoldenEntry]) -> None:
    flagged = [
        e.id for e in entries if classify_node({"question": e.question})["is_unsafe"]
    ]
    assert flagged == []


def test_off_corpus_entries_have_no_labels(entries: list[GoldenEntry]) -> None:
    for entry in entries:
        if entry.type is QuestionType.OFF_CORPUS:
            assert entry.relevant == []
            assert entry.source_chunks == []

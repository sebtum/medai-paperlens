"""Tests for the golden-set schema, loader and validator (small in-code fixtures)."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.golden import (
    QUOTAS,
    GoldenEntry,
    GoldenFormatError,
    Label,
    QuestionType,
    derive_aliases,
    golden_sha256,
    load_golden,
    load_manifests,
    validate_golden,
)

PAPER_IDS = [f"paper_{i:02d}" for i in range(10)]
MANIFESTS: dict[str, dict[str, Any]] = {
    pid: {"title": f"Zorblax{i}: A Synthetic Test Paper {i}"}
    for i, pid in enumerate(PAPER_IDS)
}


def make_valid_entries() -> list[GoldenEntry]:
    """A complete valid set: exact quotas, every paper labelled in >= 3 entries."""
    entries: list[GoldenEntry] = []
    counter = 0  # over non-off-corpus entries, spreads labels round-robin over papers
    n = 0
    for qtype, quota in QUOTAS.items():
        for _ in range(quota):
            n += 1
            relevant: list[Label] = []
            chunks: list[str] = []
            if qtype is not QuestionType.OFF_CORPUS:
                p1 = PAPER_IDS[counter % 10]
                p2 = PAPER_IDS[(counter + 1) % 10]
                if qtype is QuestionType.CROSS_PAPER:
                    relevant = [
                        Label(paper_id=p1, section="Methods", grade=2),
                        Label(paper_id=p2, section="Results", grade=1),
                    ]
                    chunks = [f"{p1}:1", f"{p2}:2"]
                elif qtype is QuestionType.MULTI_SECTION:
                    relevant = [
                        Label(paper_id=p1, section="Methods", grade=2),
                        Label(paper_id=p1, section="Results > Ablation", grade=1),
                    ]
                    chunks = [f"{p1}:1", f"{p1}:7"]
                else:
                    relevant = [Label(paper_id=p1, section="Methods", grade=2)]
                    chunks = [f"{p1}:3"]
                counter += 1
            entries.append(
                GoldenEntry(
                    id=f"q{n:03d}",
                    question=f"How does approach number {n} handle its data?",
                    type=qtype,
                    relevant=relevant,
                    source_chunks=chunks,
                )
            )
    return entries


def first_of(entries: list[GoldenEntry], qtype: QuestionType) -> int:
    return next(i for i, e in enumerate(entries) if e.type is qtype)


def replace(entries: list[GoldenEntry], index: int, **update: Any) -> list[GoldenEntry]:
    out = list(entries)
    out[index] = entries[index].model_copy(update=update)
    return out


def test_valid_set_has_no_violations() -> None:
    entries = make_valid_entries()
    assert len(entries) == 50
    assert validate_golden(entries, MANIFESTS) == []


def test_duplicate_id() -> None:
    entries = make_valid_entries()
    entries = replace(entries, 1, id=entries[0].id)
    violations = validate_golden(entries, MANIFESTS)
    assert any("duplicate id 'q001'" in v for v in violations)


def test_bad_grade() -> None:
    entries = make_valid_entries()
    bad = [Label(paper_id=PAPER_IDS[0], section="Methods", grade=3)]
    entries = replace(entries, 0, relevant=bad)
    assert any("grade 3" in v for v in validate_golden(entries, MANIFESTS))


def test_grade_zero_is_a_violation() -> None:
    entries = make_valid_entries()
    bad = [Label(paper_id=PAPER_IDS[0], section="Methods", grade=0)]
    entries = replace(entries, 0, relevant=bad)
    assert any("grade 0" in v for v in validate_golden(entries, MANIFESTS))


def test_quota_missing_entry() -> None:
    entries = make_valid_entries()
    entries.pop(first_of(entries, QuestionType.OFF_CORPUS))
    violations = validate_golden(entries, MANIFESTS)
    assert "quota off_corpus: expected 5, found 4" in violations


def test_quota_extra_entry() -> None:
    entries = make_valid_entries()
    extra = entries[0].model_copy(update={"id": "q999"})
    violations = validate_golden([*entries, extra], MANIFESTS)
    assert "quota factual: expected 10, found 11" in violations


def test_coverage_paper_labelled_too_rarely() -> None:
    entries = make_valid_entries()
    # Relabel everything away from paper_09 onto paper_00.
    fixed = []
    for e in entries:
        labels = [
            lab.model_copy(update={"paper_id": PAPER_IDS[0]})
            if lab.paper_id == PAPER_IDS[9]
            else lab
            for lab in e.relevant
        ]
        chunks = [c.replace(PAPER_IDS[9], PAPER_IDS[0]) for c in e.source_chunks]
        fixed.append(e.model_copy(update={"relevant": labels, "source_chunks": chunks}))
    violations = validate_golden(fixed, MANIFESTS)
    assert any(
        "coverage: paper 'paper_09' is labelled in 0 entries" in v for v in violations
    )


def test_cross_paper_single_paper() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.CROSS_PAPER)
    one = [Label(paper_id=PAPER_IDS[0], section="Methods", grade=2)]
    entries = replace(entries, i, relevant=one, source_chunks=[f"{PAPER_IDS[0]}:1"])
    assert any(
        "cross_paper labels must span >= 2 papers" in v
        for v in validate_golden(entries, MANIFESTS)
    )


def test_multi_section_sections_in_different_papers() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.MULTI_SECTION)
    split = [
        Label(paper_id=PAPER_IDS[0], section="Methods", grade=2),
        Label(paper_id=PAPER_IDS[1], section="Methods", grade=2),
    ]
    entries = replace(entries, i, relevant=split)
    assert any(
        "multi_section needs >= 2 sections of one paper" in v
        for v in validate_golden(entries, MANIFESTS)
    )


def test_off_corpus_with_labels_or_chunks() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.OFF_CORPUS)
    entries = replace(
        entries,
        i,
        relevant=[Label(paper_id=PAPER_IDS[0], section="Methods", grade=1)],
        source_chunks=[f"{PAPER_IDS[0]}:1"],
    )
    violations = validate_golden(entries, MANIFESTS)
    assert any("off_corpus must have empty relevant" in v for v in violations)
    assert any("off_corpus must have empty source_chunks" in v for v in violations)


def test_non_off_corpus_without_labels_or_chunks() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    entries = replace(entries, i, relevant=[], source_chunks=[])
    violations = validate_golden(entries, MANIFESTS)
    assert any("needs at least one relevant label" in v for v in violations)
    assert any("needs at least one source_chunk" in v for v in violations)


def test_malformed_source_chunk_and_unknown_paper() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    entries = replace(entries, i, source_chunks=["no-index", "ghost_paper:4"])
    violations = validate_golden(entries, MANIFESTS)
    assert any("'no-index' is not 'paper_id:chunk_index'" in v for v in violations)
    assert any(
        "'ghost_paper:4' names a paper not in the manifests" in v for v in violations
    )


def test_label_for_unknown_paper() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    entries = replace(
        entries, i, relevant=[Label(paper_id="ghost", section="Methods", grade=1)]
    )
    assert any(
        "'ghost' is not in the manifests" in v
        for v in validate_golden(entries, MANIFESTS)
    )


def test_duplicate_label_in_entry() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    label = entries[i].relevant[0]
    entries = replace(entries, i, relevant=[label, label])
    assert any("duplicate label" in v for v in validate_golden(entries, MANIFESTS))


@pytest.mark.parametrize(
    "question",
    [
        "What does MAIRA-2 do with grounding?",
        "how is maira 2 evaluated",  # separator variants
        "Compare LLaVA-Med and others",
        "Why was conch trained on captions?",
        "Is Med-PaLM M multimodal?",
        "What is RoentGen used for?",
        "Tell me about Zorblax3 results",  # alias derived from the manifest title
        "What does paper_05 claim?",  # paper_id derived alias
    ],
)
def test_paraphrased_question_with_name_or_title(question: str) -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.PARAPHRASED)
    entries = replace(entries, i, question=question)
    violations = validate_golden(entries, MANIFESTS)
    assert any(
        "paraphrased question contains title/model name" in v for v in violations
    ), violations


def test_alias_in_non_paraphrased_question_is_allowed() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    entries = replace(entries, i, question="What data was MAIRA-2 trained on?")
    assert validate_golden(entries, MANIFESTS) == []


def test_alias_does_not_match_inside_other_words() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.PARAPHRASED)
    entries = replace(
        entries, i, question="Which conchoidal fracture pattern is noted?"
    )
    assert validate_golden(entries, MANIFESTS) == []


def test_extra_aliases_are_checked() -> None:
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.PARAPHRASED)
    entries = replace(entries, i, question="What is the Frobnicator loss?")
    assert validate_golden(entries, MANIFESTS) == []
    assert validate_golden(entries, MANIFESTS, extra_aliases=["Frobnicator"])


def test_derive_aliases() -> None:
    aliases = derive_aliases(
        {"maira_2": {"title": "MAIRA-2: Grounded Radiology Report Generation"}}
    )
    assert aliases == ["maira_2", "MAIRA-2"]


@pytest.mark.parametrize(
    ("paper_id", "section"),
    [
        ("med_palm_m", "Input"),
        ("med_palm_m", "Methods > Target"),
        ("med_palm_m", "Target:"),
        ("llava_med", "Caption:"),
        ("llava_med", "Example input:"),
        ("llava_med", "Visual input example, Biomedical image from [6]"),
        ("maira_2", "Appendix"),
        ("maira_2", "A Data > Prompts"),
        ("maira_2", "B.2 Prompt details"),
    ],
)
def test_banned_labels(paper_id: str, section: str) -> None:
    manifests = {**MANIFESTS, paper_id: {"title": "X"}}
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    bad = [Label(paper_id=paper_id, section=section, grade=1)]
    entries = replace(entries, i, relevant=bad, source_chunks=[f"{paper_id}:1"])
    violations = validate_golden(entries, manifests)
    assert any("banned label" in v for v in violations), violations


def test_legit_labels_similar_to_banned_are_allowed() -> None:
    manifests = {**MANIFESTS, "maira_2": {"title": "X"}}
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.FACTUAL)
    ok = [
        Label(paper_id="maira_2", section="Methods > Input Representation", grade=1),
        Label(paper_id="maira_2", section="Introduction", grade=1),
        Label(paper_id="maira_2", section="Results > A Closer Look", grade=1),
    ]
    entries = replace(entries, i, relevant=ok, source_chunks=["maira_2:1"])
    assert not any("banned label" in v for v in validate_golden(entries, manifests))


def test_all_violations_are_returned_not_just_the_first() -> None:
    entries = make_valid_entries()
    entries = replace(entries, 1, id=entries[0].id)
    entries.pop(first_of(entries, QuestionType.OFF_CORPUS))
    i = first_of(entries, QuestionType.CROSS_PAPER)
    entries = replace(
        entries, i, relevant=[Label(paper_id=PAPER_IDS[0], section="M", grade=7)]
    )
    violations = validate_golden(entries, MANIFESTS)
    assert len(violations) >= 3
    assert any("duplicate id" in v for v in violations)
    assert any("quota off_corpus" in v for v in violations)
    assert any("grade 7" in v for v in violations)


# --- loader ---------------------------------------------------------------------------


def write_jsonl(path: Path, entries: list[GoldenEntry]) -> None:
    path.write_text(
        "\n".join(e.model_dump_json() for e in entries) + "\n", encoding="utf-8"
    )


def test_load_golden_roundtrip(tmp_path: Path) -> None:
    entries = make_valid_entries()
    path = tmp_path / "golden.jsonl"
    write_jsonl(path, entries)
    assert load_golden(path) == entries
    assert validate_golden(load_golden(path), MANIFESTS) == []


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    entry = make_valid_entries()[0]
    path = tmp_path / "golden.jsonl"
    path.write_text(f"\n{entry.model_dump_json()}\n\n", encoding="utf-8")
    assert load_golden(path) == [entry]


def test_load_golden_invalid_json_reports_line_number(tmp_path: Path) -> None:
    good = make_valid_entries()[0].model_dump_json()
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{good}\n{good}\n{{not json\n", encoding="utf-8")
    with pytest.raises(GoldenFormatError, match=r"golden\.jsonl:3: invalid JSON"):
        load_golden(path)


def test_load_golden_schema_error_reports_line_number_and_field(tmp_path: Path) -> None:
    good = make_valid_entries()[0].model_dump_json()
    bad = json.dumps({"id": "q2", "question": "Q?", "type": "nonsense"})
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    with pytest.raises(GoldenFormatError, match=r"golden\.jsonl:2: type:"):
        load_golden(path)


def test_load_golden_rejects_unknown_field(tmp_path: Path) -> None:
    data = make_valid_entries()[0].model_dump(mode="json") | {"excerpt": "paper text"}
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(GoldenFormatError, match=r":1: excerpt"):
        load_golden(path)


def test_golden_sha256(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "golden.jsonl"
    path.write_bytes(b"abc\n")
    assert golden_sha256(path) == hashlib.sha256(b"abc\n").hexdigest()
    path.write_bytes(b"abd\n")
    assert golden_sha256(path) != hashlib.sha256(b"abc\n").hexdigest()


def test_load_manifests_uses_file_stem(tmp_path: Path) -> None:
    (tmp_path / "alpha.json").write_text(json.dumps({"title": "A"}), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert load_manifests(tmp_path) == {"alpha": {"title": "A"}}


def test_real_manifests_load_and_derive_aliases() -> None:
    papers_dir = Path(__file__).resolve().parent.parent / "data" / "papers"
    manifests = load_manifests(papers_dir)
    assert len(manifests) == 10
    aliases = derive_aliases(manifests)
    assert "MAIRA-2" in aliases
    assert "LLaVA-Med" in aliases


def test_paraphrased_question_with_full_title_without_colon() -> None:
    manifests = {**MANIFESTS, "paper_00": {"title": "Towards Generalist Biomedical AI"}}
    entries = make_valid_entries()
    i = first_of(entries, QuestionType.PARAPHRASED)
    entries = replace(
        entries, i, question="Summarise towards generalist biomedical AI."
    )
    violations = validate_golden(entries, manifests)
    assert any(
        "paraphrased question contains title/model name" in v for v in violations
    )

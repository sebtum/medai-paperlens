"""Golden-set schema, loader and validator for the retrieval evaluation (ADR-0023).

The golden set is a JSONL file (`data/eval/golden.jsonl`), one `GoldenEntry` per line.
It holds our own question text and `(paper_id, section)` labels only, never paper text.

`validate_golden` returns every rule violation as a string instead of stopping at the
first one, so an author sees the whole to-do list in one run.

Manifests: `validate_golden` takes `Manifests = Mapping[str, Mapping[str, Any]]`, i.e.
`{paper_id: manifest_dict}`. `paper_id` is the manifest file stem (same convention as
`app.retrieval.ingestion`); the manifest dict needs only a `title` (a missing title is
tolerated). `load_manifests(papers_dir)` builds it from `data/papers/*.json`.

This module is pure (no network, no Qdrant) and does not import the metrics module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Manifests = Mapping[str, Mapping[str, Any]]

VALID_GRADES: frozenset[int] = frozenset({1, 2})
MIN_ENTRIES_PER_PAPER = 3
SECTION_SEPARATOR = " > "


class QuestionType(StrEnum):
    FACTUAL = "factual"
    PARAPHRASED = "paraphrased"
    CROSS_PAPER = "cross_paper"
    TABLE = "table"
    MULTI_SECTION = "multi_section"
    OFF_CORPUS = "off_corpus"


QUOTAS: dict[QuestionType, int] = {
    QuestionType.FACTUAL: 10,
    QuestionType.PARAPHRASED: 10,
    QuestionType.CROSS_PAPER: 8,
    QuestionType.TABLE: 8,
    QuestionType.MULTI_SECTION: 9,
    QuestionType.OFF_CORPUS: 5,
}

# Model names a paraphrased question must not contain, on top of the aliases derived
# from the manifests (see `derive_aliases`).
TEST_ALIASES: tuple[str, ...] = (
    "MAIRA-2",
    "MAIRA",
    "LLaVA-Med",
    "LLaVA",
    "CONCH",
    "RadFM",
    "Med-PaLM M",
    "Med-PaLM",
    "PathAlign",
    "RoentGen",
    "Gaze2Report",
    "DeepTumorVQA",
)

# False headings Docling produced in Phase 8, kept as labels on purpose there. A golden
# label must never point at them. Compared per path segment of `Section > Subsection`.
_BANNED_EXACT_SEGMENTS: frozenset[str] = frozenset({"input", "target"})
_BANNED_SEGMENT_PREFIXES: tuple[str, ...] = (
    "caption",
    "example input",
    "visual input example",
)
# `maira_2` appendix labels: `Appendix ...`, or a dotless/dotted appendix letter heading
# such as `A Data`, `B.2 Prompts`. Only the top-level segment is tested.
_MAIRA_APPENDIX_RE = re.compile(r"^((?i:appendix)\b|[A-Z]\d*(\.\d+)*[\s.])")
_MAIRA_PAPER_ID = "maira_2"

_SOURCE_CHUNK_RE = re.compile(r"^(?P<paper>[A-Za-z0-9_\-]+):(?P<index>\d+)$")


class GoldenFormatError(ValueError):
    """The golden file is unreadable or a line does not match the schema."""


class Label(BaseModel):
    """A relevant `(paper_id, section)`. grade 2 = core of the answer, 1 = supporting.

    `grade` is a plain int so `validate_golden` can report a bad value together with
    all other violations instead of failing at load time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    grade: int


class GoldenEntry(BaseModel):
    """One golden question. `source_chunks` are `"paper_id:chunk_index"`, audit only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    type: QuestionType
    relevant: list[Label] = Field(default_factory=list)
    source_chunks: list[str] = Field(default_factory=list)


def load_golden(path: Path | str) -> list[GoldenEntry]:
    """Read a JSONL golden file. Blank lines are skipped; errors carry line numbers."""
    path = Path(path)
    entries: list[GoldenEntry] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenFormatError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc
            try:
                entries.append(GoldenEntry.model_validate(data))
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc']) or '<entry>'}: {err['msg']}"
                    for err in exc.errors()
                )
                raise GoldenFormatError(f"{path}:{line_no}: {problems}") from exc
    return entries


def golden_sha256(path: Path | str) -> str:
    """SHA-256 hex digest of the raw file bytes (the dataset version identity)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifests(papers_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load `{paper_id: manifest}` from `papers_dir/*.json` (paper_id = file stem)."""
    manifests: dict[str, dict[str, Any]] = {}
    for file in sorted(Path(papers_dir).glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            manifests[file.stem] = data
    return manifests


def derive_aliases(manifests: Manifests) -> list[str]:
    """Aliases a paraphrased question must not contain, beyond the full title.

    Per manifest: the title part before a colon (`MAIRA-2: Grounded ...` ->
    `MAIRA-2`) and the paper_id. Added to `TEST_ALIASES` by `validate_golden`.
    """
    aliases: list[str] = []
    for paper_id, manifest in manifests.items():
        aliases.append(paper_id)
        title = str(manifest.get("title", "")).strip()
        if ":" in title:
            head = title.split(":", 1)[0].strip()
            if head:
                aliases.append(head)
    return aliases


def _alias_pattern(alias: str) -> re.Pattern[str] | None:
    """Case-insensitive pattern: separators (space, hyphen, underscore) are optional."""
    tokens = [t for t in re.split(r"[-_\s]+", alias.strip()) if t]
    if not tokens:
        return None
    body = r"[-_\s]*".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _banned_label_reason(label: Label) -> str | None:
    segments = [s.strip() for s in label.section.split(SECTION_SEPARATOR)]
    for segment in segments:
        norm = segment.rstrip(":").strip().lower()
        if norm in _BANNED_EXACT_SEGMENTS:
            return f"false heading {segment!r}"
        if any(norm.startswith(p) for p in _BANNED_SEGMENT_PREFIXES):
            return f"false heading {segment!r}"
    if (
        label.paper_id == _MAIRA_PAPER_ID
        and segments
        and _MAIRA_APPENDIX_RE.match(segments[0])
    ):
        return f"maira_2 appendix label {segments[0]!r}"
    return None


def validate_golden(
    entries: Sequence[GoldenEntry],
    manifests: Manifests,
    *,
    extra_aliases: Iterable[str] = (),
) -> list[str]:
    """Check the ADR-0023 rules; return every violation (empty list = valid)."""
    violations: list[str] = []

    # Unique ids.
    for entry_id, count in Counter(e.id for e in entries).items():
        if count > 1:
            violations.append(f"duplicate id {entry_id!r} ({count} entries)")

    # Quotas.
    counts = Counter(e.type for e in entries)
    for qtype, quota in QUOTAS.items():
        if counts[qtype] != quota:
            violations.append(
                f"quota {qtype.value}: expected {quota}, found {counts[qtype]}"
            )

    # Coverage: every paper labelled in >= MIN_ENTRIES_PER_PAPER entries.
    labelled_in: Counter[str] = Counter()
    for entry in entries:
        for paper_id in {label.paper_id for label in entry.relevant}:
            labelled_in[paper_id] += 1
    for paper_id in manifests:
        if labelled_in[paper_id] < MIN_ENTRIES_PER_PAPER:
            violations.append(
                f"coverage: paper {paper_id!r} is labelled in {labelled_in[paper_id]} "
                f"entries, need >= {MIN_ENTRIES_PER_PAPER}"
            )

    name_patterns = [
        p
        for name in (
            *(str(m.get("title", "")) for m in manifests.values()),
            *TEST_ALIASES,
            *derive_aliases(manifests),
            *extra_aliases,
        )
        if (p := _alias_pattern(name)) is not None
    ]

    for entry in entries:
        violations.extend(_validate_entry(entry, manifests, name_patterns))
    return violations


def _validate_entry(
    entry: GoldenEntry,
    manifests: Manifests,
    name_patterns: Sequence[re.Pattern[str]],
) -> list[str]:
    out: list[str] = []
    tag = f"{entry.id} ({entry.type.value})"

    seen: set[tuple[str, str]] = set()
    for label in entry.relevant:
        if label.grade not in VALID_GRADES:
            out.append(f"{tag}: grade {label.grade} not in {{1, 2}} ({label.paper_id})")
        if label.paper_id not in manifests:
            out.append(
                f"{tag}: label paper_id {label.paper_id!r} is not in the manifests"
            )
        key = (label.paper_id, label.section)
        if key in seen:
            out.append(f"{tag}: duplicate label {label.paper_id} / {label.section!r}")
        seen.add(key)
        reason = _banned_label_reason(label)
        if reason:
            out.append(
                f"{tag}: banned label {label.paper_id} / {label.section!r}: {reason}"
            )

    for ref in entry.source_chunks:
        match = _SOURCE_CHUNK_RE.match(ref)
        if match is None:
            out.append(f"{tag}: source_chunk {ref!r} is not 'paper_id:chunk_index'")
        elif match["paper"] not in manifests:
            out.append(
                f"{tag}: source_chunk {ref!r} names a paper not in the manifests"
            )

    papers = {label.paper_id for label in entry.relevant}
    if entry.type is QuestionType.OFF_CORPUS:
        if entry.relevant:
            out.append(f"{tag}: off_corpus must have empty relevant")
        if entry.source_chunks:
            out.append(f"{tag}: off_corpus must have empty source_chunks")
    else:
        if not entry.relevant:
            out.append(f"{tag}: needs at least one relevant label")
        if not entry.source_chunks:
            out.append(f"{tag}: needs at least one source_chunk")

    if entry.type is QuestionType.CROSS_PAPER and len(papers) < 2:
        out.append(
            f"{tag}: cross_paper labels must span >= 2 papers, found {len(papers)}"
        )

    if entry.type is QuestionType.MULTI_SECTION:
        sections_per_paper: Counter[str] = Counter(paper for paper, _ in seen)
        if not sections_per_paper or max(sections_per_paper.values()) < 2:
            out.append(f"{tag}: multi_section needs >= 2 sections of one paper")

    if entry.type is QuestionType.PARAPHRASED:
        for pattern in name_patterns:
            found = pattern.search(entry.question)
            if found:
                out.append(
                    f"{tag}: paraphrased question contains title/model name "
                    f"{found.group(0)!r}"
                )
                break
    return out

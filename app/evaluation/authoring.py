"""Golden-set authoring helpers: `sections` and `pool` (ADR-0023).

- `sections_command` lists each paper's sections with chunk ids from a Qdrant
  collection. `dump_chunk_text` can also write the chunk text to a directory, which must
  be outside the repository: that text is only for the golden-set author and is never
  committed.
- `pool_candidates` / `render_pool` list, per question, the sections in each run's top-k
  that no label covers (pooling). Ids and section labels only, no chunk text.

Every command returns the lines to print; the script prints them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from app.evaluation.compare import RunRecord
from app.evaluation.golden import (
    GoldenEntry,
    QuestionType,
    golden_sha256,
    load_golden,
)
from app.evaluation.metrics import RankedHit, cut
from app.evaluation.runner import (
    REPO_ROOT,
    RunError,
    load_runs,
    write_text,
)
from app.evaluation.target import CorpusIndex, scan_collection

POOL_DEFAULT_K = 10
POOL_DEFAULT_THRESHOLD = 0.40  # lowest threshold of the curve, so pooling covers it

_DUMP_FIELDS = ["text", "paper_id", "chunk_index", "section", "chunk_type", "page"]
_SCROLL_PAGE_SIZE = 256


def render_sections(index: CorpusIndex, *, paper: str | None = None) -> str:
    """Per-paper section list with chunk ids (section labels only, no chunk text)."""
    lines = [
        f"collection {index.collection!r}: {index.chunk_count} chunks, "
        f"embedding model(s) {', '.join(sorted(index.embedding_models))}",
        "chunk types: "
        + ", ".join(f"{name}={n}" for name, n in sorted(index.type_counts.items())),
        "",
    ]
    papers = sorted(index.sections)
    if paper is not None:
        if paper not in index.sections:
            raise RunError(f"paper {paper!r} is not in collection {index.collection!r}")
        papers = [paper]
    for paper_id in papers:
        sections = index.sections[paper_id]
        count = sum(len(chunks) for chunks in sections.values())
        lines.append(f"## {paper_id} ({count} chunks)")
        for section, chunks in sorted(sections.items(), key=lambda item: item[1][0]):
            ids = ", ".join(
                f"{paper_id}:{i}"
                + (
                    f" [{index.chunk_types[(paper_id, i)]}]"
                    if index.chunk_types[(paper_id, i)] != "text"
                    else ""
                )
                for i in chunks
            )
            lines.append(f"- {section} -> {ids}")
        lines.append("")
    return "\n".join(lines)


def _require_outside_repo(directory: Path) -> Path:
    resolved = directory.resolve()
    if resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT):
        raise RunError(
            f"--dump-text directory {resolved} is inside the repository; chunk text "
            "must never be committed. Use a directory outside it (e.g. a scratchpad)."
        )
    return resolved


async def dump_chunk_text(
    client: AsyncQdrantClient,
    collection: str,
    directory: Path,
    *,
    paper: str | None = None,
) -> list[Path]:
    """Write `<directory>/<paper_id>.txt` with every chunk's text, in chunk order."""
    target = _require_outside_repo(directory)
    chunks: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    offset: Any = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=_DUMP_FIELDS,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            paper_id, index = payload.get("paper_id"), payload.get("chunk_index")
            if (
                isinstance(paper_id, str)
                and isinstance(index, int)
                and (paper is None or paper_id == paper)
            ):
                chunks.setdefault(paper_id, []).append((index, payload))
        if offset is None:
            break
    if paper is not None and paper not in chunks:
        raise RunError(f"paper {paper!r} is not in collection {collection!r}")

    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for paper_id, items in sorted(chunks.items()):
        parts: list[str] = []
        for index, payload in sorted(items, key=lambda item: item[0]):
            page = payload.get("page")
            where = f"page {page}" if page is not None else "page ?"
            parts.append(
                f"=== {paper_id}:{index} | {payload.get('chunk_type', 'text')} | "
                f"{payload.get('section', '')} | {where} ===\n"
                f"{payload.get('text', '')}\n"
            )
        path = target / f"{paper_id}.txt"
        path.write_text("\n".join(parts), encoding="utf-8")
        written.append(path)
    return written


async def sections_command(
    client: AsyncQdrantClient,
    collection: str,
    *,
    paper: str | None = None,
    dump_text: Path | None = None,
) -> list[str]:
    """Section list (and optional text dump) of `collection`; returns the lines."""
    index = await scan_collection(client, collection)
    if index.chunk_count == 0:
        raise RunError(f"collection {collection!r} has no points")
    if dump_text is not None:
        _require_outside_repo(dump_text)  # fail before any output
    lines = [render_sections(index, paper=paper)]
    if dump_text is not None:
        written = await dump_chunk_text(client, collection, dump_text, paper=paper)
        lines.append(f"Chunk text written to {len(written)} file(s) in {dump_text}")
        lines.append("This text must not be committed.")
    return lines


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------


@dataclass
class PoolCandidate:
    """An unlabelled section seen in the top-k of at least one run."""

    paper_id: str
    section: str
    best_rank: int
    best_score: float
    runs: set[str] = field(default_factory=set)
    chunks: set[tuple[int, str]] = field(default_factory=set)


def pool_candidates(
    runs: Sequence[RunRecord],
    golden: Sequence[GoldenEntry],
    *,
    k: int = POOL_DEFAULT_K,
    threshold: float = POOL_DEFAULT_THRESHOLD,
    include_off_corpus: bool = False,
) -> dict[str, list[PoolCandidate]]:
    """Per question id: sections in any run's top-k (chunk level) that no label covers.

    The rank is the chunk's position in that run's cut list (1 = best).
    """
    result: dict[str, list[PoolCandidate]] = {}
    for entry in golden:
        if entry.type is QuestionType.OFF_CORPUS and not include_off_corpus:
            continue
        labelled = {(label.paper_id, label.section) for label in entry.relevant}
        found: dict[tuple[str, str], PoolCandidate] = {}
        for run in runs:
            question = next((q for q in run.questions if q.id == entry.id), None)
            if question is None:
                raise RunError(f"run {run.meta.name!r} has no question {entry.id}")
            hits: list[RankedHit] = cut(question.hits, k, threshold)
            for rank, hit in enumerate(hits, start=1):
                key = (hit.paper_id, hit.section)
                if key in labelled:
                    continue
                candidate = found.setdefault(
                    key,
                    PoolCandidate(hit.paper_id, hit.section, rank, hit.score),
                )
                candidate.runs.add(run.meta.name)
                candidate.chunks.add((hit.chunk_index, hit.chunk_type))
                if rank < candidate.best_rank:
                    candidate.best_rank = rank
                candidate.best_score = max(candidate.best_score, hit.score)
        result[entry.id] = sorted(
            found.values(), key=lambda c: (c.best_rank, -c.best_score)
        )
    return result


def render_pool(
    pools: Mapping[str, Sequence[PoolCandidate]],
    golden: Sequence[GoldenEntry],
    runs: Sequence[RunRecord],
    *,
    k: int,
    threshold: float,
) -> str:
    """Markdown list for adjudication: ids and section labels only, no chunk text."""
    questions = {entry.id: entry for entry in golden}
    lines = [
        "# Pooling candidates",
        "",
        "Note: golden set LLM-labelled (Claude), not human-verified.",
        "",
        f"- Runs: {', '.join(run.meta.name for run in runs)}",
        f"- Top-{k} chunks per run at t={threshold:.2f}; sections not yet labelled",
        "- Read each candidate's chunks (use `sections --dump-text`) and decide: "
        "grade 2, grade 1 or not relevant",
        "",
    ]
    total = 0
    for question_id, candidates in pools.items():
        entry = questions[question_id]
        lines.append(f"## {question_id} ({entry.type.value}): {entry.question}")
        if not candidates:
            lines.append("- no unlabelled section in the top-k")
        for candidate in candidates:
            total += 1
            chunks = ", ".join(
                f"{candidate.paper_id}:{index}"
                + (f" [{kind}]" if kind != "text" else "")
                for index, kind in sorted(candidate.chunks)
            )
            lines.append(
                f"- {candidate.paper_id} / {candidate.section} "
                f"(best rank {candidate.best_rank}, "
                f"best score {candidate.best_score:.3f}, "
                f"runs: {', '.join(sorted(candidate.runs))}; chunks {chunks})"
            )
        lines.append("")
    lines.insert(2, f"{total} candidate section(s) over {len(pools)} question(s).")
    lines.insert(3, "")
    return "\n".join(lines)


def pool_command(
    runs: Sequence[str],
    *,
    golden_path: Path,
    runs_dir: Path,
    k: int = POOL_DEFAULT_K,
    threshold: float = POOL_DEFAULT_THRESHOLD,
    include_off_corpus: bool = False,
    out: Path | None = None,
) -> list[str]:
    """Pooling list for `runs`; written to `out` or returned as the lines to print."""
    entries = load_golden(golden_path)
    sha256 = golden_sha256(golden_path)
    records = load_runs(runs, runs_dir, sha256, golden_path)
    pools = pool_candidates(
        records,
        entries,
        k=k,
        threshold=threshold,
        include_off_corpus=include_off_corpus,
    )
    text = render_pool(pools, entries, records, k=k, threshold=threshold)
    if out is not None:
        write_text(out, text)
        return [f"Pooling list written: {out}"]
    return [text]

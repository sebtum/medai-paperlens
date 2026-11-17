"""Sync the golden set (`data/eval/golden.jsonl`) to a LangSmith dataset (ADR-0024).

Git is the source of truth; the dataset is a derived copy, never edited in the UI.

How the sync decides and writes (all against the `langsmith` 0.12.5 `Client`):

- Example id = `uuid5(GOLDEN_NAMESPACE, entry id)`, so a question keeps its example when
  its text or labels change. Inputs `{id, question}`, reference outputs
  `{type, relevant}` (our own labels only, no paper text).
- The sha256 of the golden file is stored in the *metadata of every example*
  (`golden_sha256`). `langsmith` 0.12.5 has no call that updates a dataset's own
  metadata after creation, so the example metadata is the stored hash. The sync is a
  no-op (no writes) when the dataset exists, its example ids equal the file's and every
  example carries the file's hash. A half-finished earlier sync therefore re-runs.
- Otherwise: create the dataset if missing, create new examples, update existing ones,
  delete the ones no longer in the file, then tag the new dataset version with the short
  git hash (`update_dataset_tag`).
- `ensure_golden_committed` refuses a golden file with uncommitted changes, so every tag
  names a commit that contains exactly the synced file.

The `Client` and the git runner are injected, so tests need neither a network nor a
git checkout in a particular state. This module imports no `langsmith` code.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.evaluation.golden import (
    GoldenEntry,
    golden_sha256,
    load_golden,
    load_manifests,
    validate_golden,
)

DATASET_NAME = "paperlens-retrieval-golden"
DATASET_DESCRIPTION = (
    "Retrieval golden set for MedAI PaperLens (question + relevant paper sections). "
    "Synced from data/eval/golden.jsonl in git; do not edit here (ADR-0024)."
)
HASH_METADATA_KEY = "golden_sha256"
LATEST_VERSION_TAG = "latest"

# Fixed namespace for example ids. Never change it: it would orphan every example.
GOLDEN_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "medai-paperlens/eval/golden")

# `git` arguments in, stdout out; raises `GitError` when git itself fails.
GitRunner = Callable[[Sequence[str]], str]


class SyncError(RuntimeError):
    """The sync cannot run (invalid or uncommitted golden file, git failure)."""


class GoldenNotCommittedError(SyncError):
    """`golden.jsonl` is missing, untracked or has uncommitted changes."""


class GoldenInvalidError(SyncError):
    """`golden.jsonl` violates the ADR-0023 rules."""

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = list(violations)
        super().__init__(
            f"golden set has {len(self.violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in self.violations)
        )


class GitError(SyncError):
    """A git command failed (git missing, not a repository, ...)."""


class LangSmithClient(Protocol):
    """The slice of `langsmith.Client` (0.12.5) the sync uses.

    Every method name here exists on the real client; `tests/test_eval_sync.py` checks
    that with `hasattr`. Return types are `Any`: the sync reads only `.id`, `.metadata`
    (examples) and `.as_of` (dataset version).
    """

    def has_dataset(
        self, *, dataset_name: str | None = None, dataset_id: Any = None
    ) -> bool: ...

    def create_dataset(
        self,
        dataset_name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def list_examples(
        self, *, dataset_name: str | None = None, **kwargs: Any
    ) -> Iterable[Any]: ...

    def create_examples(
        self,
        *,
        dataset_name: str | None = None,
        examples: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def update_examples(
        self,
        *,
        dataset_name: str | None = None,
        updates: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def delete_examples(self, example_ids: Sequence[Any], **kwargs: Any) -> None: ...

    def read_dataset_version(
        self,
        *,
        dataset_name: str | None = None,
        tag: str | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def update_dataset_tag(
        self,
        *,
        dataset_name: str | None = None,
        as_of: Any,
        tag: str,
    ) -> None: ...


@dataclass(frozen=True)
class SyncResult:
    """What a sync did. `unchanged` means it made no write at all."""

    dataset_name: str
    golden_sha256: str
    unchanged: bool
    dataset_created: bool = False
    created: int = 0
    updated: int = 0
    deleted: int = 0
    tag: str | None = None

    def summary(self) -> str:
        if self.unchanged:
            return (
                f"Dataset {self.dataset_name!r} is up to date "
                f"(sha256 {self.golden_sha256[:12]}); nothing written."
            )
        return (
            f"Synced dataset {self.dataset_name!r} "
            f"({'created' if self.dataset_created else 'existing'}): "
            f"{self.created} created, {self.updated} updated, {self.deleted} deleted; "
            f"sha256 {self.golden_sha256[:12]}, tag {self.tag}."
        )


def example_id(entry_id: str) -> uuid.UUID:
    """Stable LangSmith example id for a golden entry id."""
    return uuid.uuid5(GOLDEN_NAMESPACE, entry_id)


def entry_to_example(entry: GoldenEntry, sha256: str) -> dict[str, Any]:
    """The LangSmith example (id, inputs, reference outputs, hash) for an entry."""
    return {
        "id": example_id(entry.id),
        "inputs": {"id": entry.id, "question": entry.question},
        "outputs": {
            "type": entry.type.value,
            "relevant": [
                {"paper_id": lb.paper_id, "section": lb.section, "grade": lb.grade}
                for lb in entry.relevant
            ],
        },
        "metadata": {HASH_METADATA_KEY: sha256},
    }


def prepare_golden(
    golden_path: Path | str, papers_dir: Path | str
) -> tuple[list[GoldenEntry], str]:
    """Load and validate the golden file; return `(entries, sha256)`.

    Raises `GoldenInvalidError` listing every violation (ADR-0023 rules, against the
    manifests in `papers_dir`), and `SyncError` for an empty file.
    """
    entries = load_golden(golden_path)
    if not entries:
        raise SyncError(f"{golden_path} contains no entries; nothing to sync.")
    violations = validate_golden(entries, load_manifests(papers_dir))
    if violations:
        raise GoldenInvalidError(violations)
    return entries, golden_sha256(golden_path)


def ensure_golden_committed(golden_path: Path | str, run_git: GitRunner) -> None:
    """Raise `GoldenNotCommittedError` unless the file is tracked and unmodified."""
    path = Path(golden_path)
    if not path.is_file():
        raise GoldenNotCommittedError(f"{path} does not exist.")
    if not run_git(["ls-files", "--", str(path)]).strip():
        raise GoldenNotCommittedError(
            f"{path} is not tracked by git; commit it before syncing."
        )
    if run_git(["status", "--porcelain", "--", str(path)]).strip():
        raise GoldenNotCommittedError(
            f"{path} has uncommitted changes; commit them first so the dataset tag "
            "matches a commit."
        )


def git_short_hash(run_git: GitRunner) -> str:
    """Short hash of HEAD."""
    short = run_git(["rev-parse", "--short", "HEAD"]).strip()
    if not short:
        raise GitError("git rev-parse --short HEAD returned nothing")
    return short


def _is_up_to_date(
    existing: Mapping[str, Any], wanted_ids: Iterable[str], sha256: str
) -> bool:
    """Same example ids as the file, and every example carries the file's hash."""
    if set(existing) != set(wanted_ids):
        return False
    return all(
        (ex.metadata or {}).get(HASH_METADATA_KEY) == sha256 for ex in existing.values()
    )


def sync_golden(
    client: LangSmithClient,
    entries: Sequence[GoldenEntry],
    sha256: str,
    git_hash: str,
    *,
    dataset_name: str = DATASET_NAME,
) -> SyncResult:
    """Sync `entries` to the LangSmith dataset; no writes when already up to date.

    The caller has validated the entries and checked the file is committed.
    """
    if not entries:
        raise SyncError("no golden entries to sync")
    wanted = {str(example_id(e.id)): entry_to_example(e, sha256) for e in entries}
    if len(wanted) != len(entries):
        raise SyncError("golden entry ids are not unique")

    dataset_exists = client.has_dataset(dataset_name=dataset_name)
    existing: dict[str, Any] = {}
    if dataset_exists:
        existing = {
            str(ex.id): ex for ex in client.list_examples(dataset_name=dataset_name)
        }
        if _is_up_to_date(existing, wanted, sha256):
            return SyncResult(
                dataset_name=dataset_name, golden_sha256=sha256, unchanged=True
            )
    else:
        client.create_dataset(dataset_name, description=DATASET_DESCRIPTION)

    to_create = [ex for key, ex in wanted.items() if key not in existing]
    to_update = [ex for key, ex in wanted.items() if key in existing]
    to_delete = [existing[key].id for key in existing if key not in wanted]

    if to_create:
        client.create_examples(dataset_name=dataset_name, examples=to_create)
    if to_update:
        client.update_examples(dataset_name=dataset_name, updates=to_update)
    if to_delete:
        client.delete_examples(to_delete)

    # Tag the version this sync produced (its newest one) with the git hash.
    version = client.read_dataset_version(
        dataset_name=dataset_name, tag=LATEST_VERSION_TAG
    )
    client.update_dataset_tag(
        dataset_name=dataset_name, as_of=version.as_of, tag=git_hash
    )
    return SyncResult(
        dataset_name=dataset_name,
        golden_sha256=sha256,
        unchanged=False,
        dataset_created=not dataset_exists,
        created=len(to_create),
        updated=len(to_update),
        deleted=len(to_delete),
        tag=git_hash,
    )

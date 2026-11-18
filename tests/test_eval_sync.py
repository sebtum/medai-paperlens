"""Tests for the golden-set LangSmith sync (fake client, fake git, no network)."""

import datetime
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.evaluation import langsmith_sync as ls_sync
from app.evaluation.golden import GoldenEntry, Label, QuestionType, golden_sha256
from app.evaluation.langsmith_sync import (
    DATASET_NAME,
    HASH_METADATA_KEY,
    LATEST_VERSION_TAG,
    GoldenInvalidError,
    GoldenNotCommittedError,
    SyncError,
    ensure_golden_committed,
    entry_to_example,
    example_id,
    git_short_hash,
    prepare_golden,
    sync_golden,
)
from scripts import sync_golden as sync_script

AS_OF = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


class FakeLangSmith:
    """In-memory stand-in for the slice of `langsmith.Client` the sync uses."""

    def __init__(
        self, *, datasets: dict[str, dict[uuid.UUID, dict[str, Any]]] | None = None
    ):
        # dataset name -> {example id -> {"inputs", "outputs", "metadata"}}
        self.datasets = datasets if datasets is not None else {}
        self.calls: list[tuple[str, Any]] = []
        self.tags: list[tuple[str, Any, str]] = []

    @property
    def writes(self) -> list[str]:
        reads = {"has_dataset", "list_examples", "read_dataset_version"}
        return [name for name, _ in self.calls if name not in reads]

    def has_dataset(
        self, *, dataset_name: str | None = None, dataset_id: Any = None
    ) -> bool:
        self.calls.append(("has_dataset", dataset_name))
        return dataset_name in self.datasets

    def create_dataset(self, dataset_name: str, **kwargs: Any) -> Any:
        self.calls.append(("create_dataset", dataset_name))
        self.datasets[dataset_name] = {}
        return SimpleNamespace(name=dataset_name)

    def list_examples(self, *, dataset_name: str | None = None, **kwargs: Any) -> Any:
        self.calls.append(("list_examples", dataset_name))
        assert dataset_name is not None
        return [
            SimpleNamespace(id=ex_id, **data)
            for ex_id, data in self.datasets[dataset_name].items()
        ]

    def create_examples(
        self, *, dataset_name: str | None = None, examples: Any = None, **kwargs: Any
    ) -> Any:
        self.calls.append(("create_examples", [e["id"] for e in examples]))
        assert dataset_name is not None
        for ex in examples:
            assert ex["id"] not in self.datasets[dataset_name], "create of existing id"
            self.datasets[dataset_name][ex["id"]] = {
                k: ex[k] for k in ("inputs", "outputs", "metadata")
            }

    def update_examples(
        self, *, dataset_name: str | None = None, updates: Any = None, **kwargs: Any
    ) -> Any:
        self.calls.append(("update_examples", [u["id"] for u in updates]))
        assert dataset_name is not None
        for up in updates:
            assert up["id"] in self.datasets[dataset_name], "update of unknown id"
            self.datasets[dataset_name][up["id"]] = {
                k: up[k] for k in ("inputs", "outputs", "metadata")
            }

    def delete_examples(self, example_ids: Sequence[Any], **kwargs: Any) -> None:
        self.calls.append(("delete_examples", list(example_ids)))
        for ds in self.datasets.values():
            for ex_id in example_ids:
                ds.pop(ex_id, None)

    def read_dataset_version(
        self, *, dataset_name: str | None = None, tag: str | None = None, **kwargs: Any
    ) -> Any:
        self.calls.append(("read_dataset_version", tag))
        tags = [LATEST_VERSION_TAG] + [name for _, _, name in self.tags]
        return SimpleNamespace(as_of=AS_OF, tags=tags)

    def update_dataset_tag(
        self, *, dataset_name: str | None = None, as_of: Any, tag: str
    ) -> None:
        self.calls.append(("update_dataset_tag", tag))
        self.tags.append((str(dataset_name), as_of, tag))


def make_entry(
    entry_id: str, question: str = "What is the dataset size?", section: str = "Data"
) -> GoldenEntry:
    return GoldenEntry(
        id=entry_id,
        question=question,
        type=QuestionType.FACTUAL,
        relevant=[Label(paper_id="p1", section=section, grade=2)],
        source_chunks=["p1:3"],
    )


ENTRIES = [make_entry("q01"), make_entry("q02", "How was it trained?", "Methods")]
SHA = "a" * 64
GIT = "abc1234"


# --- example shape ---------------------------------------------------------------


def test_example_ids_are_stable_and_distinct() -> None:
    assert example_id("q01") == example_id("q01")
    assert example_id("q01") != example_id("q02")
    assert example_id("q01") == uuid.uuid5(ls_sync.GOLDEN_NAMESPACE, "q01")


def test_entry_to_example_has_inputs_outputs_and_no_source_chunks() -> None:
    ex = entry_to_example(ENTRIES[0], SHA)
    assert ex["id"] == example_id("q01")
    assert ex["inputs"] == {"id": "q01", "question": "What is the dataset size?"}
    assert ex["outputs"] == {
        "type": "factual",
        "relevant": [{"paper_id": "p1", "section": "Data", "grade": 2}],
    }
    assert ex["metadata"] == {HASH_METADATA_KEY: SHA}


# --- sync ------------------------------------------------------------------------


def test_first_sync_creates_dataset_examples_and_tag() -> None:
    client = FakeLangSmith()
    result = sync_golden(client, ENTRIES, SHA, GIT)

    assert not result.unchanged
    assert result.dataset_created
    assert (result.created, result.updated, result.deleted) == (2, 0, 0)
    assert result.tag == GIT
    assert set(client.datasets[DATASET_NAME]) == {example_id("q01"), example_id("q02")}
    assert client.tags == [(DATASET_NAME, AS_OF, GIT)]
    assert client.calls.index(("create_dataset", DATASET_NAME)) < client.calls.index(
        ("update_dataset_tag", GIT)
    )


def test_matching_hash_is_a_noop_with_no_writes() -> None:
    client = FakeLangSmith()
    sync_golden(client, ENTRIES, SHA, GIT)
    client.calls.clear()

    result = sync_golden(client, ENTRIES, SHA, "def5678")

    assert result.unchanged
    assert client.writes == []
    assert client.tags == [(DATASET_NAME, AS_OF, GIT)]  # no new tag either


def test_change_upserts_deletes_removed_stores_hash_and_tags() -> None:
    client = FakeLangSmith()
    sync_golden(client, ENTRIES, SHA, GIT)
    client.calls.clear()

    new_entries = [
        make_entry("q02", "How was the model trained?", "Methods"),  # edited
        make_entry("q03", "Which metrics were used?", "Results"),  # added
    ]  # q01 removed
    new_sha = "b" * 64
    result = sync_golden(client, new_entries, new_sha, "def5678")

    assert (result.created, result.updated, result.deleted) == (1, 1, 1)
    assert not result.dataset_created
    assert client.writes == [
        "create_examples",
        "update_examples",
        "delete_examples",
        "update_dataset_tag",
    ]
    data = client.datasets[DATASET_NAME]
    assert set(data) == {example_id("q02"), example_id("q03")}  # q01 deleted
    assert data[example_id("q02")]["inputs"]["question"] == "How was the model trained?"
    assert all(d["metadata"] == {HASH_METADATA_KEY: new_sha} for d in data.values())
    assert client.tags[-1] == (DATASET_NAME, AS_OF, "def5678")


def test_partial_earlier_sync_is_repaired() -> None:
    """An example with a stale hash (interrupted sync) triggers a full re-sync."""
    client = FakeLangSmith()
    sync_golden(client, ENTRIES, SHA, GIT)
    client.datasets[DATASET_NAME][example_id("q02")]["metadata"] = {
        HASH_METADATA_KEY: "old"
    }

    result = sync_golden(client, ENTRIES, SHA, GIT)

    assert not result.unchanged
    assert (result.created, result.updated, result.deleted) == (0, 2, 0)


def test_retry_after_failed_tag_repairs_the_tag() -> None:
    """Examples written, tagging failed: the next sync adds the tag and nothing else."""
    client = FakeLangSmith()
    real_tag = client.update_dataset_tag

    def failing_tag(**kwargs: Any) -> None:
        raise RuntimeError("tagging failed")

    client.update_dataset_tag = failing_tag  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        sync_golden(client, ENTRIES, SHA, GIT)
    assert client.tags == []
    client.update_dataset_tag = real_tag  # type: ignore[method-assign]
    client.calls.clear()

    result = sync_golden(client, ENTRIES, SHA, GIT)

    assert not result.unchanged
    assert result.tag == GIT
    assert (result.created, result.updated, result.deleted) == (0, 0, 0)
    assert client.writes == ["update_dataset_tag"]
    assert client.tags[-1] == (DATASET_NAME, AS_OF, GIT)
    assert sync_golden(client, ENTRIES, SHA, GIT).unchanged


def test_empty_entries_refused() -> None:
    with pytest.raises(SyncError):
        sync_golden(FakeLangSmith(), [], SHA, GIT)


# --- git guard -------------------------------------------------------------------


def fake_git(*, tracked: bool = True, status: str = "", short: str = GIT) -> Any:
    calls: list[list[str]] = []

    def run(args: Sequence[str]) -> str:
        calls.append(list(args))
        if args[0] == "ls-files":
            return "data/eval/golden.jsonl\n" if tracked else ""
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return short + "\n"
        raise AssertionError(f"unexpected git call {args}")

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(e.model_dump_json() for e in ENTRIES) + "\n", encoding="utf-8"
    )
    return path


def test_clean_tracked_file_passes(golden_file: Path) -> None:
    ensure_golden_committed(golden_file, fake_git())


def test_dirty_file_is_refused(golden_file: Path) -> None:
    with pytest.raises(GoldenNotCommittedError, match="uncommitted"):
        ensure_golden_committed(golden_file, fake_git(status=" M golden.jsonl\n"))


def test_untracked_file_is_refused(golden_file: Path) -> None:
    with pytest.raises(GoldenNotCommittedError, match="not tracked"):
        ensure_golden_committed(golden_file, fake_git(tracked=False))


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GoldenNotCommittedError, match="does not exist"):
        ensure_golden_committed(tmp_path / "nope.jsonl", fake_git())


def test_git_short_hash() -> None:
    assert git_short_hash(fake_git(short="1a2b3c4")) == "1a2b3c4"
    with pytest.raises(SyncError):
        git_short_hash(fake_git(short=""))


def test_prepare_golden_reports_violations(golden_file: Path, tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "p1.json").write_text(json.dumps({"title": "T"}), encoding="utf-8")
    # Two entries only: quotas and coverage are violated.
    with pytest.raises(GoldenInvalidError) as excinfo:
        prepare_golden(golden_file, papers)
    assert any("quota" in v for v in excinfo.value.violations)


def test_prepare_golden_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "golden.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SyncError, match="no entries"):
        prepare_golden(empty, tmp_path)


# --- script main -----------------------------------------------------------------


def run_main(golden: Path, client: FakeLangSmith, git: Any, *extra: str) -> int:
    return sync_script.main(
        ["--golden", str(golden), *extra],
        client_factory=lambda: client,
        git=git,
        load_env=lambda: None,
    )


def test_main_missing_api_key_exits_1(
    golden_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    client = FakeLangSmith()

    assert run_main(golden_file, client, fake_git()) == 1

    assert "LANGSMITH_API_KEY" in caplog.text
    assert client.calls == []


def test_main_dirty_golden_exits_1(
    golden_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    client = FakeLangSmith()

    assert run_main(golden_file, client, fake_git(status=" M x\n")) == 1
    assert client.calls == []


def test_main_syncs_and_second_run_is_noop(
    golden_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    # The two-entry fixture is not a full valid golden set; skip rule validation here
    # (validation itself is covered above) but keep the real file hash.
    monkeypatch.setattr(
        sync_script,
        "prepare_golden",
        lambda path, papers: (ENTRIES, golden_sha256(path)),
    )
    client = FakeLangSmith()

    assert run_main(golden_file, client, fake_git()) == 0
    assert client.tags == [(DATASET_NAME, AS_OF, GIT)]

    client.calls.clear()
    assert run_main(golden_file, client, fake_git()) == 0
    assert client.writes == []


def test_main_langsmith_error_exits_1(
    golden_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        sync_script, "prepare_golden", lambda path, papers: (ENTRIES, SHA)
    )

    class Broken(FakeLangSmith):
        def has_dataset(self, **kwargs: Any) -> bool:
            raise ConnectionError("boom")

    assert run_main(golden_file, Broken(), fake_git()) == 1


# --- the real langsmith.Client has what we call ------------------------------------

USED_CLIENT_METHODS = [
    "has_dataset",
    "create_dataset",
    "list_examples",
    "create_examples",
    "update_examples",
    "delete_examples",
    "read_dataset_version",
    "update_dataset_tag",
]


def test_protocol_and_fake_methods_exist_on_real_client() -> None:
    """Names and key parameters checked against installed langsmith; no network."""
    import inspect

    from langsmith import Client

    protocol_methods = {
        name
        for name, member in vars(ls_sync.LangSmithClient).items()
        if callable(member) and not name.startswith("_")
    }
    assert protocol_methods == set(USED_CLIENT_METHODS)
    for name in USED_CLIENT_METHODS:
        assert hasattr(Client, name), f"langsmith.Client has no {name}"
        assert callable(getattr(FakeLangSmith, name))

    def params(name: str) -> set[str]:
        return set(inspect.signature(getattr(Client, name)).parameters)

    assert {"dataset_name", "description"} <= params("create_dataset")
    assert {"dataset_name", "examples"} <= params("create_examples")
    assert {"dataset_name", "updates"} <= params("update_examples")
    assert "dataset_name" in params("list_examples")
    assert {"dataset_name", "tag"} <= params("read_dataset_version")
    assert {"dataset_name", "as_of", "tag"} <= params("update_dataset_tag")
    assert "example_ids" in params("delete_examples")

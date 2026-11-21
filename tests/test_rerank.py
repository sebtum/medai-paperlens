import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from conftest import FakeReranker

from app.retrieval import rerank
from app.retrieval.rerank import (
    DEFAULT_RERANKER_MODEL,
    RERANKER_RETRY_SECONDS,
    CrossEncoderReranker,
    Reranker,
    UnavailableReranker,
    get_reranker,
    get_reranker_model_name,
    rerank_passage,
)


class _FakeCrossEncoder:
    """Stands in for `CrossEncoder`: logit = length - 5, then `activation_fn`."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.activation_fns: list[Any] = []
        self.thread_ids: list[int] = []

    def predict(
        self, pairs: list[tuple[str, str]], *, activation_fn: Any = None
    ) -> Any:
        self.calls.append(pairs)
        self.activation_fns.append(activation_fn)
        self.thread_ids.append(threading.get_ident())
        raw = torch.tensor([float(len(p[1])) - 5.0 for p in pairs])
        return activation_fn(raw) if activation_fn is not None else raw


# --- CrossEncoderReranker ----------------------------------------------------


async def test_score_returns_sigmoid_scores_in_unit_range() -> None:
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model, model_name="m")  # type: ignore[arg-type]

    scores = await reranker.score("q", ["a", "a" * 5, "a" * 12])

    assert isinstance(model.activation_fns[0], torch.nn.Sigmoid)
    assert scores[1] == pytest.approx(0.5)  # raw logit 0 -> 0.5
    assert scores[0] < scores[1] < scores[2]
    assert all(0.0 < s < 1.0 for s in scores)
    assert all(type(s) is float for s in scores)


async def test_score_sends_query_passage_pairs_in_order() -> None:
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model, model_name="m")  # type: ignore[arg-type]

    await reranker.score("question", ["first", "second"])

    assert model.calls == [[("question", "first"), ("question", "second")]]


async def test_score_runs_in_a_worker_thread() -> None:
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model, model_name="m")  # type: ignore[arg-type]

    await reranker.score("q", ["a"])

    assert model.thread_ids[0] != threading.get_ident()


async def test_score_empty_passages_skips_the_model() -> None:
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model, model_name="m")  # type: ignore[arg-type]

    assert await reranker.score("q", []) == []
    assert model.calls == []


def test_model_name_property() -> None:
    reranker = CrossEncoderReranker(_FakeCrossEncoder(), model_name="org/model")  # type: ignore[arg-type]
    assert reranker.model_name == "org/model"


def test_protocol_is_satisfied_by_both_implementations() -> None:
    assert isinstance(CrossEncoderReranker(_FakeCrossEncoder(), "m"), Reranker)  # type: ignore[arg-type]
    assert isinstance(FakeReranker(), Reranker)


# --- get_reranker_model_name -------------------------------------------------


@pytest.mark.parametrize("value", ["", "  ", "none", "None", "NONE", " none "])
def test_model_name_off_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RERANKER_MODEL", value)
    assert get_reranker_model_name() is None


def test_model_name_unset_is_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The autouse fixture sets RERANKER_MODEL to "none"; clear it here.
    monkeypatch.delenv("RERANKER_MODEL")
    assert get_reranker_model_name() == DEFAULT_RERANKER_MODEL
    assert DEFAULT_RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_model_name_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    assert get_reranker_model_name() == "BAAI/bge-reranker-base"


# --- get_reranker ------------------------------------------------------------


def test_get_reranker_unset_builds_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def fake_cross_encoder(name: str, **kwargs: Any) -> _FakeCrossEncoder:
        built.append(name)
        return _FakeCrossEncoder()

    monkeypatch.setattr(rerank, "CrossEncoder", fake_cross_encoder)
    monkeypatch.delenv("RERANKER_MODEL")

    reranker = get_reranker()

    assert reranker is not None
    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert built == [DEFAULT_RERANKER_MODEL]


def test_get_reranker_unset_never_loads_a_real_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset now means the default model; only the autouse block stops the download.
    # Its AssertionError is not a load error, so it reaches the test (a loud tripwire).
    monkeypatch.delenv("RERANKER_MODEL")
    with pytest.raises(AssertionError, match="FakeReranker"):
        get_reranker()


def test_get_reranker_off_returns_none_without_loading_a_model() -> None:
    # The autouse fixture makes any CrossEncoder load raise, so this also proves
    # that "off" never reaches the constructor.
    assert get_reranker() is None


def test_get_reranker_builds_once_with_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, dict[str, Any]]] = []

    def fake_cross_encoder(name: str, **kwargs: Any) -> _FakeCrossEncoder:
        built.append((name, kwargs))
        return _FakeCrossEncoder()

    monkeypatch.setattr(rerank, "CrossEncoder", fake_cross_encoder)
    monkeypatch.setenv("RERANKER_MODEL", "org/some-reranker")

    first = get_reranker()
    second = get_reranker()

    assert first is second
    assert isinstance(first, CrossEncoderReranker)
    assert first.model_name == "org/some-reranker"
    assert built == [("org/some-reranker", {"max_length": 512})]


def test_autouse_fixture_blocks_real_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "org/some-reranker")
    with pytest.raises(AssertionError, match="FakeReranker"):
        get_reranker()


def test_get_reranker_does_not_hide_a_programming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "org/some-reranker")
    monkeypatch.setattr(
        rerank, "CrossEncoder", MagicMock(side_effect=TypeError("bad kwarg"))
    )
    with pytest.raises(TypeError):
        get_reranker()
    # The lock is released after the error: a later call can load normally.
    monkeypatch.setattr(rerank, "CrossEncoder", lambda name, **kw: _FakeCrossEncoder())
    assert isinstance(get_reranker(), CrossEncoderReranker)


async def test_get_reranker_does_not_wait_for_a_load_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "org/slow")
    seen: list[Any] = []

    def slow_load(name: str, **kwargs: Any) -> _FakeCrossEncoder:
        seen.append(get_reranker())  # a second request arrives mid-load
        return _FakeCrossEncoder()

    monkeypatch.setattr(rerank, "CrossEncoder", slow_load)

    loader = get_reranker()

    assert isinstance(loader, CrossEncoderReranker)
    assert isinstance(seen[0], UnavailableReranker)
    with pytest.raises(RuntimeError, match="org/slow"):
        await seen[0].score("q", ["a"])
    # The mid-load stand-in was not remembered as a failure.
    assert get_reranker() is loader


async def test_get_reranker_load_failure_returns_a_failing_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "org/missing")
    monkeypatch.setattr(
        rerank, "CrossEncoder", MagicMock(side_effect=OSError("no network"))
    )

    reranker = get_reranker()

    assert isinstance(reranker, UnavailableReranker)
    assert reranker.model_name == "org/missing"
    with pytest.raises(RuntimeError, match="org/missing"):
        await reranker.score("q", ["a", "b"])


def test_get_reranker_remembers_a_failed_load_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(rerank.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("RERANKER_MODEL", "org/flaky")
    attempts: list[int] = []

    def flaky(name: str, **kwargs: Any) -> _FakeCrossEncoder:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("offline")
        return _FakeCrossEncoder()

    monkeypatch.setattr(rerank, "CrossEncoder", flaky)

    first = get_reranker()
    clock[0] += RERANKER_RETRY_SECONDS - 1
    second = get_reranker()  # inside the window: no new load attempt
    assert isinstance(first, UnavailableReranker)
    assert second is first
    assert len(attempts) == 1

    clock[0] += 2
    third = get_reranker()  # window over: loads, and the success is kept
    assert isinstance(third, CrossEncoderReranker)
    assert get_reranker() is third
    assert len(attempts) == 2


def test_get_reranker_failure_does_not_stick_to_another_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "org/broken")
    monkeypatch.setattr(rerank, "CrossEncoder", MagicMock(side_effect=OSError("gone")))
    assert isinstance(get_reranker(), UnavailableReranker)

    monkeypatch.setattr(rerank, "CrossEncoder", lambda name, **kw: _FakeCrossEncoder())
    monkeypatch.setenv("RERANKER_MODEL", "org/working")
    reranker = get_reranker()

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "org/working"


# --- rerank_passage ----------------------------------------------------------


def test_rerank_passage_joins_title_section_text() -> None:
    assert rerank_passage("Title", "Methods > Data", "Body.") == (
        "Title\nMethods > Data\nBody."
    )


# --- FakeReranker ------------------------------------------------------------


async def test_fake_reranker_default_scores_by_word_overlap() -> None:
    fake = FakeReranker()
    scores = await fake.score("chest xray report", ["chest xray report", "pasta"])
    assert scores == [1.0, 0.0]
    assert fake.calls == [("chest xray report", ["chest xray report", "pasta"])]


async def test_fake_reranker_fixed_scores_and_error() -> None:
    assert await FakeReranker(scores=[0.2, 0.9]).score("q", ["a", "b"]) == [0.2, 0.9]
    with pytest.raises(RuntimeError):
        await FakeReranker(error=RuntimeError("boom")).score("q", ["a"])

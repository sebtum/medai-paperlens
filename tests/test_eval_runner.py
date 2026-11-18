"""Tests for the retrieval evaluation runner (scripts.eval_retrieval) and the LangSmith
experiment adapters (app.evaluation.langsmith_eval).

Fakes only: an in-memory async Qdrant, a fake embedder, a fake rewrite LLM and a fake
`aevaluate`. No Qdrant, Ollama, LangSmith or network. `main()` runs its own event loop,
so tests that call it are plain (sync) tests.
"""

import dataclasses
import datetime
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.evaluation import authoring, runner
from app.evaluation.compare import (
    QuestionRun,
    RunMeta,
    RunRecord,
    load_run,
    question_scores,
    save_run,
)
from app.evaluation.golden import GoldenEntry, QuestionType, golden_sha256
from app.evaluation.langsmith_eval import (
    EMPTY_KEY,
    NO_RESULT_KEY,
    ExperimentError,
    ExperimentMismatchError,
    ExperimentOutcome,
    ScoreBook,
    check_against_local,
    experiment_metadata,
    local_experiment_scores,
    make_evaluators,
    run_experiment,
)
from app.evaluation.metrics import RankedHit
from app.evaluation.target import (
    RetrievalConfig,
    RewriteCache,
    RewriteKey,
    rewrite_prompt_sha256,
)
from app.workflow.generate import _estimated_tokens
from scripts import eval_retrieval as cli
from tests.conftest import FakeReranker
from tests.test_eval_target import (
    MODEL,
    REWRITER,
    FakeAsyncQdrant,
    FakeEmbedder,
    RecordingLlm,
    make_entry,
    make_point,
)

TEXT_A = "alpha beta gamma delta epsilon"
TEXT_B = "zeta eta theta"
TEXT_C = "table cell values"
TEXT_D = "iota kappa"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def corpus_points() -> list[SimpleNamespace]:
    return [
        make_point("maira_2", "Results", 0, text=TEXT_A, score=0.9),
        make_point("maira_2", "Methods", 1, text=TEXT_B, score=0.8),
        make_point(
            "conch",
            "Results > Table 2",
            0,
            chunk_type="table",
            text=TEXT_C,
            score=0.7,
        ),
        make_point("conch", "Introduction", 1, text=TEXT_D, score=0.3),
    ]


def golden_entries() -> list[GoldenEntry]:
    return [
        make_entry(
            "q001",
            question="alpha beta gamma delta what is shown here",
            type_=QuestionType.FACTUAL,
            relevant=[("maira_2", "Results", 2)],
            source_chunks=["maira_2:0"],
        ),
        make_entry(
            "q002",
            question="Which table lists the values?",
            type_=QuestionType.TABLE,
            relevant=[("conch", "Results > Table 2", 2), ("maira_2", "Methods", 1)],
            source_chunks=["conch:0"],
        ),
        make_entry(
            "q003",
            question="How do I bake bread?",
            type_=QuestionType.OFF_CORPUS,
        ),
    ]


def write_golden(path: Path, entries: Sequence[GoldenEntry]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(e.model_dump(mode="json")) + "\n" for e in entries),
        encoding="utf-8",
    )
    return path


class RewriteLlm(RecordingLlm):
    """Rewrite LLM stand-in: has a `model` and is an async context manager."""

    model = REWRITER

    async def __aenter__(self) -> RewriteLlm:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class ForbiddenLlm:
    """An llm_factory that must never be called."""

    def __call__(self) -> Any:
        raise AssertionError("the LLM must not be used")


class FakeAevaluate:
    """Stand-in for `langsmith.aevaluate`.

    Calls the target and the evaluators for every golden entry, the way LangSmith does
    (evaluators get `inputs`, `outputs`, `reference_outputs`). With `swallow` it
    swallows errors like `error_handling="log"`.
    """

    def __init__(
        self,
        entries: Sequence[GoldenEntry],
        *,
        swallow: bool = True,
        tamper_labels: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.entries = list(entries)
        self.swallow = swallow
        self.tamper_labels = tamper_labels
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, target: Any, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        for entry in self.entries:
            inputs = {"id": entry.id, "question": entry.question}
            reference: dict[str, Any] = {
                "type": entry.type.value,
                "relevant": [
                    {"paper_id": lb.paper_id, "section": lb.section, "grade": lb.grade}
                    for lb in entry.relevant
                ],
            }
            if self.tamper_labels is not None:
                reference = self.tamper_labels(reference)
            try:
                outputs = await target(inputs)
                for evaluator in kwargs["evaluators"]:
                    evaluator(
                        inputs=inputs, outputs=outputs, reference_outputs=reference
                    )
            except Exception:
                if not self.swallow:
                    raise


class Harness:
    """Everything `main()` needs, on tmp_path, with fakes injected."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        entries: Sequence[GoldenEntry] | None = None,
        points: list[SimpleNamespace] | None = None,
    ) -> None:
        self.tmp = tmp_path
        self.entries = list(entries) if entries is not None else golden_entries()
        self.golden = write_golden(tmp_path / "golden.jsonl", self.entries)
        self.qdrant = FakeAsyncQdrant(points if points is not None else corpus_points())
        self.embedder = FakeEmbedder()
        self.llm = RewriteLlm("a rewritten question")
        self.llm_factory: Callable[[], Any] = lambda: self.llm
        self.aevaluate: FakeAevaluate | None = None
        self.git_calls: list[Sequence[str]] = []
        self.runs = tmp_path / "runs"
        self.reports = tmp_path / "reports"
        self.rewrites = tmp_path / "rewrites.jsonl"

    def git(self, args: Sequence[str]) -> str:
        self.git_calls.append(args)
        return "abc1234\n" if args[0] == "rev-parse" else ""

    def services(self) -> runner.Services:
        return runner.Services(
            client_factory=lambda: self.qdrant,  # type: ignore[arg-type, return-value]
            embedder_factory=lambda _name: self.embedder,
            llm_factory=lambda: self.llm_factory(),
            aevaluate_factory=lambda: self.aevaluate,  # type: ignore[arg-type, return-value]
            git=self.git,
        )

    def paths(self) -> list[str]:
        return [
            "--golden",
            str(self.golden),
            "--runs-dir",
            str(self.runs),
            "--reports-dir",
            str(self.reports),
        ]

    def run(self, *extra: str, config: str = "no_rewrite") -> int:
        argv = [
            "run",
            "--config",
            config,
            *self.paths(),
            "--papers-dir",
            str(self.tmp / "papers"),
            "--rewrites",
            str(self.rewrites),
            "--rewrite-model",
            REWRITER,
            "--skip-validate",
            *extra,
        ]
        return cli_main(argv, services=self.services())


def cli_main(argv: Sequence[str], *, services: runner.Services) -> int:
    """`scripts.eval_retrieval.main` without reading the developer's `.env`."""
    return cli.main(argv, services=services, load_env=lambda: None)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


# ---------------------------------------------------------------------------
# Named configs
# ---------------------------------------------------------------------------


class TestNoQueryPrefixLeak:
    """ADR-0025: the app embedder prefixes queries by default; the eval must not.

    Every named config sets its prefix exactly once, through `RetrievalConfig.prefix`.
    """

    def _embedded_text(self, monkeypatch: pytest.MonkeyPatch, config_name: str) -> str:
        import sentence_transformers

        seen: list[str] = []

        class FakeModel:
            def __init__(self, name: str) -> None:
                self.name = name

            def encode(self, text: str) -> Any:
                import numpy as np

                seen.append(text)
                return np.zeros(3, dtype=np.float32)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
        monkeypatch.delenv("EMBEDDING_QUERY_PREFIX", raising=False)

        options = runner.make_run_options(config_name)
        config = dataclasses.replace(options.config, rewrite=False, rewrite_model="")
        embedder = runner._default_embedder(config.embedder)

        import asyncio

        from app.evaluation.target import retrieve_ranked

        entry = make_entry()
        asyncio.run(
            retrieve_ranked(
                entry,
                config,
                embedder=embedder,
                client=FakeAsyncQdrant([make_point("p1", "Intro", 0)]),  # type: ignore[arg-type]
            )
        )
        assert len(seen) == 1
        return seen[0].removesuffix(entry.question)

    @pytest.mark.parametrize("config_name", ["baseline", "no_rewrite", "bge_base"])
    def test_configs_without_prefix_embed_the_bare_query(
        self, monkeypatch: pytest.MonkeyPatch, config_name: str
    ) -> None:
        assert self._embedded_text(monkeypatch, config_name) == ""

    def test_prefix_config_gets_the_prefix_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._embedded_text(monkeypatch, "prefix") == runner.BGE_QUERY_PREFIX

    def test_an_env_prefix_does_not_reach_the_eval_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # same as above, but with the variable set after the helper's delenv
        monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "env: ")
        import sentence_transformers

        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", lambda name: object()
        )
        embedder = runner._default_embedder("BAAI/bge-small-en-v1.5")
        assert embedder._query_prefix == ""  # type: ignore[attr-defined]

    def test_frozen_eval_prefix_equals_the_app_bge_default(self) -> None:
        from app.retrieval.embedding import BGE_QUERY_PREFIX

        assert runner.BGE_QUERY_PREFIX == BGE_QUERY_PREFIX


class TestNamedConfigs:
    def test_the_plan_configs_exist(self) -> None:
        assert set(runner.NAMED_CONFIGS) == {
            "baseline",
            "no_rewrite",
            "prefix",
            "bge_base",
            "rerank_minilm",
            "rerank_bge",
            "rerank_medcpt",
        }

    def test_baseline_is_bge_small_without_prefix_and_with_rewrite(self) -> None:
        spec = runner.NAMED_CONFIGS["baseline"]
        assert (spec.embedder, spec.collection) == (MODEL, "papers")
        assert spec.prefix == ""
        assert spec.rewrite is True

    def test_no_rewrite_only_turns_the_rewrite_off(self) -> None:
        spec = runner.NAMED_CONFIGS["no_rewrite"]
        assert spec.rewrite is False
        assert spec.embedder == MODEL
        assert spec.collection == "papers"

    def test_prefix_uses_the_bge_query_instruction(self) -> None:
        spec = runner.NAMED_CONFIGS["prefix"]
        assert (
            spec.prefix == "Represent this sentence for searching relevant passages: "
        )
        assert spec.rewrite is True

    def test_bge_base_has_its_own_collection(self) -> None:
        spec = runner.NAMED_CONFIGS["bge_base"]
        assert spec.embedder == "BAAI/bge-base-en-v1.5"
        assert spec.collection == "eval_bge-base-en-v1.5"

    def test_default_rewrite_model_follows_the_provider_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen3:1.7b")
        assert runner.default_rewrite_model() == "qwen3:1.7b"
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-x")
        assert runner.default_rewrite_model() == "gemini-x"

    def test_overrides_for_collection_name_and_paths(self) -> None:
        options = runner.make_run_options(
            "bge_base",
            name="mine",
            collection="other",
            golden_path=Path("g.jsonl"),
            rewrites_path=Path("r.jsonl"),
            rewrite_model="m",
        )
        assert options.name == "mine"
        assert options.config_name == "bge_base"
        assert options.config.collection == "other"
        assert options.config.embedder == "BAAI/bge-base-en-v1.5"
        assert options.golden_path == Path("g.jsonl")
        assert options.rewrites_path == Path("r.jsonl")
        assert options.config.rewrite_model == "m"

    def test_cli_passes_the_overrides_through(self, harness: Harness) -> None:
        argv = [
            "run",
            "--config",
            "bge_base",
            "--embedder",
            MODEL,
            "--collection",
            "other",
            "--rewrite-model",
            REWRITER,
            "--preflight-only",
            *harness.paths(),
            "--skip-validate",
        ]
        assert cli_main(argv, services=harness.services()) == 0
        assert harness.qdrant.scroll_calls[0]["collection_name"] == "other"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    def test_writes_run_file_and_report(self, harness: Harness) -> None:
        assert harness.run() == 0

        record = load_run(harness.runs / "no_rewrite.json")
        assert record.meta.name == "no_rewrite"
        assert record.meta.golden_sha256 == golden_sha256(harness.golden)
        assert record.meta.git_hash == "abc1234"
        assert record.meta.chunk_count == 4
        assert record.meta.config["embedder"] == MODEL
        assert record.meta.config["rewrite"] is False
        assert [q.id for q in record.questions] == ["q001", "q002", "q003"]
        report = (harness.reports / "run-no_rewrite.md").read_text(encoding="utf-8")
        assert "golden set LLM-labelled (Claude), not human-verified" in report

    def test_fills_chunk_tokens_overlap_and_query(self, harness: Harness) -> None:
        assert harness.run() == 0

        q001, q002, q003 = load_run(harness.runs / "no_rewrite.json").questions
        assert [h.paper_id + ":" + str(h.chunk_index) for h in q001.hits] == [
            "maira_2:0",
            "maira_2:1",
            "conch:0",
            "conch:1",
        ]
        assert q001.chunk_tokens == tuple(
            _estimated_tokens(t) for t in (TEXT_A, TEXT_B, TEXT_C, TEXT_D)
        )
        assert q001.query == "alpha beta gamma delta what is shown here"
        assert q001.ngram_overlap == 4  # "alpha beta gamma delta"
        assert q002.ngram_overlap == 1  # the word "table"
        assert q003.ngram_overlap is None  # off-corpus: no source chunks

    def test_report_shows_evidence_tokens_instead_of_na(self, harness: Harness) -> None:
        assert harness.run() == 0

        report = (harness.reports / "run-no_rewrite.md").read_text(encoding="utf-8")
        by_k = report.split("## By k")[1].split("## By question type")[0]
        row = next(line for line in by_k.splitlines() if line.startswith("| 5 |"))
        assert row.split("|")[-2].strip() != "n/a"

    def test_no_paper_text_is_written(self, harness: Harness) -> None:
        assert harness.run() == 0

        written = [p for p in harness.tmp.rglob("*") if p.suffix in {".json", ".md"}]
        assert written
        for path in written:
            if path == harness.golden:
                continue
            content = path.read_text(encoding="utf-8")
            for text in (TEXT_A, TEXT_B, TEXT_C, TEXT_D):
                assert text not in content

    def test_search_runs_at_threshold_zero_with_depth_twenty(
        self, harness: Harness
    ) -> None:
        assert harness.run() == 0

        assert {c["score_threshold"] for c in harness.qdrant.query_calls} == {0.0}
        assert {c["limit"] for c in harness.qdrant.query_calls} == {20}

    def test_prefix_config_prepends_the_prefix_for_embedding(
        self, harness: Harness
    ) -> None:
        # the prefix config has rewrite on; the missing rewrites come from the fake LLM
        assert harness.run(config="prefix") == 0
        assert harness.embedder.texts
        assert all(
            t.startswith(runner.BGE_QUERY_PREFIX) for t in harness.embedder.texts
        )

    def test_run_name_override(self, harness: Harness) -> None:
        assert harness.run("--name", "variant") == 0
        assert (harness.runs / "variant.json").is_file()
        assert (harness.reports / "run-variant.md").is_file()

    def test_dirty_tree_marks_the_git_hash(self, harness: Harness) -> None:
        harness.git = lambda args: (  # type: ignore[method-assign]
            "abc1234\n" if args[0] == "rev-parse" else " M app/x.py\n"
        )
        assert harness.run() == 0
        record = load_run(harness.runs / "no_rewrite.json")
        assert record.meta.git_hash == "abc1234-dirty"

    def test_preflight_only_writes_nothing(self, harness: Harness) -> None:
        assert harness.run("--preflight-only") == 0
        assert not harness.runs.exists()
        assert harness.embedder.texts == []
        assert harness.qdrant.query_calls == []


class TestRunFailsLoudly:
    def test_preflight_problems_exit_nonzero_with_the_full_list(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        bad = [
            make_entry(
                "q001",
                relevant=[("maira_2", "No such section", 2)],
                source_chunks=["maira_2:99"],
            )
        ]
        harness = Harness(harness.tmp, entries=bad)
        with caplog.at_level(logging.ERROR):
            assert harness.run() == 1

        assert "labelled section does not exist" in caplog.text
        assert "source chunk does not exist" in caplog.text
        assert not (harness.runs / "no_rewrite.json").exists()

    def test_embedding_model_mismatch_is_a_preflight_failure(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        stale = Harness(
            harness.tmp,
            points=[
                make_point("maira_2", "Results", 0, model="other-model", text=TEXT_A)
            ],
            entries=[make_entry(relevant=[("maira_2", "Results", 2)])],
        )
        with caplog.at_level(logging.ERROR):
            assert stale.run() == 1
        assert "other-model" in caplog.text

    def test_invalid_golden_set_is_refused_without_skip_validate(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        (harness.tmp / "papers").mkdir()
        argv = [
            "run",
            "--config",
            "no_rewrite",
            *harness.paths(),
            "--papers-dir",
            str(harness.tmp / "papers"),
        ]
        with caplog.at_level(logging.ERROR):
            assert cli_main(argv, services=harness.services()) == 1
        assert "violation" in caplog.text
        assert not harness.runs.exists()

    def test_missing_golden_file_exits_nonzero(self, harness: Harness) -> None:
        harness.golden.unlink()
        assert harness.run() == 1

    def test_missing_rewrite_with_no_llm_exits_nonzero(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness.llm_factory = ForbiddenLlm()
        with caplog.at_level(logging.ERROR):
            assert harness.run("--no-llm", config="baseline") == 1
        assert "q001" in caplog.text and "q003" in caplog.text
        assert not harness.runs.exists()

    def test_rewrite_fallback_is_a_run_failure(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness.llm = RewriteLlm(error=RuntimeError("ollama is down"))
        with caplog.at_level(logging.ERROR):
            assert harness.run(config="baseline") == 1
        assert "fell back to the original question" in caplog.text
        assert not harness.runs.exists()

    def test_llm_model_must_match_the_rewrite_model(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            assert (
                harness.run("--rewrite-model", "another:model", config="baseline") == 1
            )
        assert "another:model" in caplog.text

    def test_hit_without_text_fails(self, harness: Harness) -> None:
        harness.qdrant.points[0].payload.pop("text")
        assert harness.run() == 1

    def test_qdrant_error_exits_nonzero(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def boom(**_: Any) -> None:
            raise ConnectionError("qdrant is down")

        harness.qdrant.scroll = boom  # type: ignore[method-assign, assignment]
        with caplog.at_level(logging.ERROR):
            assert harness.run() == 1
        assert "qdrant is down" in caplog.text


class TestRewrites:
    def test_misses_use_the_llm_then_a_rerun_needs_none(self, harness: Harness) -> None:
        assert harness.run(config="baseline") == 0
        assert len(RewriteCache(harness.rewrites)) == 3
        record = load_run(harness.runs / "baseline.json")
        assert {q.query for q in record.questions} == {"a rewritten question"}
        assert record.meta.config["rewrite_model"] == REWRITER
        assert record.meta.config["rewrite_prompt_sha256"] == rewrite_prompt_sha256()
        assert all(
            kw["temperature"] == 0 and kw["seed"] == 0 for kw in harness.llm.kwargs
        )

        harness.llm_factory = ForbiddenLlm()
        assert harness.run("--no-llm", config="baseline") == 0

    def test_cached_rewrites_need_no_llm_at_all(self, harness: Harness) -> None:
        cache = RewriteCache(harness.rewrites)
        for entry in harness.entries:
            cache.put(RewriteKey.for_entry(entry, REWRITER), f"cached {entry.id}")
        harness.llm_factory = ForbiddenLlm()

        assert harness.run(config="baseline") == 0

        record = load_run(harness.runs / "baseline.json")
        assert [q.query for q in record.questions] == [
            "cached q001",
            "cached q002",
            "cached q003",
        ]


# ---------------------------------------------------------------------------
# LangSmith experiment
# ---------------------------------------------------------------------------


def make_record(
    entries: Sequence[GoldenEntry], sha: str = "x" * 64, name: str = "r"
) -> RunRecord:
    hits = (
        RankedHit("maira_2", "Results", 0, "text", 0.9),
        RankedHit("maira_2", "Methods", 1, "text", 0.8),
        RankedHit("conch", "Results > Table 2", 0, "table", 0.7),
        RankedHit("conch", "Introduction", 1, "text", 0.3),
    )
    return RunRecord(
        meta=RunMeta(
            name=name,
            config={"embedder": MODEL},
            golden_sha256=sha,
            git_hash="g",
            chunk_count=4,
        ),
        questions=tuple(QuestionRun(e.id, e.type, hits) for e in entries),
    )


class TestEvaluatorAdapters:
    def test_adapters_give_the_local_numbers(self) -> None:
        entries = golden_entries()
        record = make_record(entries)
        book = ScoreBook()
        evaluators = make_evaluators(book)
        for entry in entries:
            outputs = {
                "hits": [
                    [h.paper_id, h.section, h.chunk_index, h.chunk_type, h.score]
                    for h in record.questions[0].hits
                ]
            }
            reference = {
                "type": entry.type.value,
                "relevant": [
                    {"paper_id": lb.paper_id, "section": lb.section, "grade": lb.grade}
                    for lb in entry.relevant
                ],
            }
            for evaluator in evaluators:
                evaluator(
                    inputs={"id": entry.id},
                    outputs=outputs,
                    reference_outputs=reference,
                )

        check_against_local(
            ExperimentOutcome(book.scores), local_experiment_scores(record, entries)
        )
        for metric in ("recall", "mrr", "ndcg"):
            local = question_scores(record, entries, metric=metric)  # type: ignore[arg-type]
            assert book.scores[f"{metric}@5"] == pytest.approx(local)
        assert set(book.scores["ndcg@5"]) == {"q001", "q002"}  # off-corpus excluded
        assert book.scores[NO_RESULT_KEY] == {"q003": 0.0}
        assert book.scores[EMPTY_KEY] == {"q001": 0.0, "q002": 0.0}

    def test_adapters_follow_langsmiths_calling_convention(self) -> None:
        """LangSmith reads parameter names; run the adapters through its own wrapper."""
        from langsmith import schemas
        from langsmith.evaluation.evaluator import DynamicRunEvaluator

        book = ScoreBook()
        evaluator = DynamicRunEvaluator(make_evaluators(book)[2])  # ndcg
        now = datetime.datetime.now(datetime.UTC)
        run = schemas.Run(
            id=uuid.uuid4(),
            name="t",
            start_time=now,
            run_type="chain",
            trace_id=uuid.uuid4(),
            inputs={"id": "q001", "question": "?"},
            outputs={"hits": [["maira_2", "Results", 0, "text", 0.9]]},
        )
        example = schemas.Example(
            id=uuid.uuid4(),
            dataset_id=uuid.uuid4(),
            inputs={"id": "q001", "question": "?"},
            outputs={
                "type": "factual",
                "relevant": [{"paper_id": "maira_2", "section": "Results", "grade": 2}],
            },
            created_at=now,
        )

        result = evaluator.evaluate_run(run, example)

        assert result.key == "ndcg@5"
        assert result.score == pytest.approx(1.0)
        assert book.scores["ndcg@5"] == {"q001": pytest.approx(1.0)}

    def test_off_corpus_gets_no_ranking_feedback(self) -> None:
        book = ScoreBook()
        recall = make_evaluators(book)[0]
        result = recall(
            inputs={"id": "q003"},
            outputs={"hits": []},
            reference_outputs={"type": "off_corpus", "relevant": []},
        )
        assert result == {"results": []}
        assert book.scores == {}

    def test_threshold_applies_inside_the_adapter(self) -> None:
        book = ScoreBook()
        evaluators = make_evaluators(book, k=5, threshold=0.95)
        for evaluator in evaluators:
            evaluator(
                inputs={"id": "q001"},
                outputs={"hits": [["maira_2", "Results", 0, "text", 0.9]]},
                reference_outputs={
                    "type": "factual",
                    "relevant": [
                        {"paper_id": "maira_2", "section": "Results", "grade": 2}
                    ],
                },
            )
        assert book.scores["recall@5"] == {"q001": 0.0}
        assert book.scores[EMPTY_KEY] == {"q001": 1.0}

    def test_a_malformed_hit_is_an_error(self) -> None:
        book = ScoreBook()
        with pytest.raises(ExperimentError):
            make_evaluators(book)[0](
                inputs={"id": "q001"},
                outputs={"hits": [["only", "three", 1]]},
                reference_outputs={"type": "factual", "relevant": []},
            )
        assert any("evaluator recall@5" in line for line in book.failures)


class TestRunExperiment:
    @staticmethod
    def setup_run(
        tmp_path: Path, **kwargs: Any
    ) -> tuple[Harness, RetrievalConfig, RunMeta]:
        harness = Harness(tmp_path, **kwargs)
        config = RetrievalConfig(embedder=MODEL, collection="papers", rewrite=False)
        meta = RunMeta(
            name="baseline",
            config={"embedder": MODEL, "rewrite": False},
            golden_sha256="s" * 64,
            git_hash="abc",
            chunk_count=4,
        )
        return harness, config, meta

    async def run_it(
        self,
        harness: Harness,
        config: RetrievalConfig,
        meta: RunMeta,
        fake: FakeAevaluate,
    ) -> ExperimentOutcome:
        return await run_experiment(
            harness.entries,
            config,
            meta,
            embedder=harness.embedder,
            client=harness.qdrant,  # type: ignore[arg-type]
            cache=None,
            llm=None,
            aevaluate_fn=fake,
        )

    async def test_scores_equal_the_local_path_and_metadata_holds_the_config(
        self, tmp_path: Path
    ) -> None:
        harness, config, meta = self.setup_run(tmp_path)
        fake = FakeAevaluate(harness.entries)

        outcome = await self.run_it(harness, config, meta, fake)

        # local path: retrieve with the same target, score with the same functions
        local = make_record(harness.entries)
        check_against_local(outcome, local_experiment_scores(local, harness.entries))
        call = fake.calls[0]
        assert call["metadata"] == experiment_metadata(meta)
        assert call["metadata"]["config"] == meta.config
        assert (call["metadata"]["k"], call["metadata"]["threshold"]) == (5, 0.6)
        assert call["metadata"]["golden_sha256"] == "s" * 64
        assert call["experiment_prefix"] == "retrieval-baseline"
        assert call["data"] == "paperlens-retrieval-golden"
        assert len(call["evaluators"]) == 5

    async def test_target_errors_fail_even_if_langsmith_swallows_them(
        self, tmp_path: Path
    ) -> None:
        harness, config, meta = self.setup_run(tmp_path)
        entries = [
            *harness.entries,
            make_entry("q999", question="not in the local golden file"),
        ]
        fake = FakeAevaluate(entries, swallow=True)

        with pytest.raises(ExperimentError, match="q999"):
            await self.run_it(harness, config, meta, fake)

    async def test_stale_dataset_question_is_an_error(self, tmp_path: Path) -> None:
        harness, config, meta = self.setup_run(tmp_path)
        changed = [
            e.model_copy(update={"question": e.question + "?"}) for e in harness.entries
        ]
        fake = FakeAevaluate(changed)

        with pytest.raises(ExperimentError, match="differs from the local golden"):
            await self.run_it(harness, config, meta, fake)

    async def test_aevaluate_error_is_reported(self, tmp_path: Path) -> None:
        harness, config, meta = self.setup_run(tmp_path)

        async def broken(*_: Any, **__: Any) -> None:
            raise ConnectionError("langsmith down")

        with pytest.raises(ExperimentError, match="langsmith down"):
            await run_experiment(
                harness.entries,
                config,
                meta,
                embedder=harness.embedder,
                client=harness.qdrant,  # type: ignore[arg-type]
                cache=None,
                llm=None,
                aevaluate_fn=broken,
            )

    def test_check_against_local_lists_every_difference(self) -> None:
        entries = golden_entries()
        expected = local_experiment_scores(make_record(entries), entries)
        scores = {key: dict(per_q) for key, per_q in expected.items()}
        scores["ndcg@5"]["q001"] += 0.1
        del scores["recall@5"]["q002"]

        with pytest.raises(ExperimentMismatchError) as caught:
            check_against_local(ExperimentOutcome(scores), expected)

        message = str(caught.value)
        assert "ndcg@5 q001" in message
        assert "recall@5 q002: no experiment score" in message


class TestRunLangsmithFlag:
    def test_missing_api_key_fails_before_any_work(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            assert harness.run("--langsmith") == 1
        assert "LANGSMITH_API_KEY" in caplog.text
        assert harness.qdrant.scroll_calls == []

    def test_experiment_runs_after_the_local_run(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        harness.aevaluate = FakeAevaluate(harness.entries)

        assert harness.run("--langsmith") == 0

        assert (harness.runs / "no_rewrite.json").is_file()
        call = harness.aevaluate.calls[0]
        record = load_run(harness.runs / "no_rewrite.json")
        assert call["metadata"]["config"] == record.meta.config
        assert call["metadata"]["golden_sha256"] == record.meta.golden_sha256

    def test_experiment_difference_exits_nonzero(
        self,
        harness: Harness,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

        def other_label(reference: dict[str, Any]) -> dict[str, Any]:
            if reference["type"] == "off_corpus":
                return reference
            moved = [{"paper_id": "maira_2", "section": "Methods", "grade": 2}]
            return {**reference, "relevant": moved}

        harness.aevaluate = FakeAevaluate(harness.entries, tamper_labels=other_label)

        with caplog.at_level(logging.ERROR):
            assert harness.run("--langsmith") == 1
        assert "differ between experiment and local run" in caplog.text


# ---------------------------------------------------------------------------
# compare / curve / pool
# ---------------------------------------------------------------------------


@pytest.fixture
def two_runs(harness: Harness) -> Harness:
    sha = golden_sha256(harness.golden)
    save_run(make_record(harness.entries, sha, "a"), harness.runs / "a.json")
    save_run(make_record(harness.entries, sha, "b"), harness.runs / "b.json")
    return harness


class TestCompareAndCurve:
    def test_compare_writes_a_report_and_prints_the_decision(
        self, two_runs: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli_main(
            ["compare", "a", "b", *two_runs.paths(), "--resamples", "50"],
            services=two_runs.services(),
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "A stays" in out  # identical rankings: no improvement
        report = (two_runs.reports / "compare-b-vs-a.md").read_text(encoding="utf-8")
        assert "Comparison: b against a" in report

    def test_threshold_comparison_uses_the_same_run_twice(
        self, two_runs: Harness
    ) -> None:
        code = cli_main(
            [
                "compare",
                "a",
                "a",
                "--threshold-b",
                "0.5",
                *two_runs.paths(),
                "--resamples",
                "50",
            ],
            services=two_runs.services(),
        )
        assert code == 0
        report = (two_runs.reports / "compare-a-vs-a-t0.50.md").read_text(
            encoding="utf-8"
        )
        assert "Threshold rule" in report

    def test_run_made_on_another_golden_file_is_refused(
        self, two_runs: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        save_run(
            make_record(two_runs.entries, "0" * 64, "old"), two_runs.runs / "old.json"
        )
        with caplog.at_level(logging.ERROR):
            code = cli_main(
                ["compare", "a", "old", *two_runs.paths()],
                services=two_runs.services(),
            )
        assert code == 1
        assert "re-run it on the current golden set" in caplog.text

    def test_unknown_run_exits_nonzero(self, two_runs: Harness) -> None:
        code = cli_main(
            ["compare", "a", "nope", *two_runs.paths()], services=two_runs.services()
        )
        assert code == 1

    def test_curve_writes_a_report(self, two_runs: Harness) -> None:
        code = cli_main(["curve", "a", *two_runs.paths()], services=two_runs.services())
        assert code == 0
        report = (two_runs.reports / "curve-a.md").read_text(encoding="utf-8")
        assert "Threshold curve" in report
        assert "| 0.40 |" in report and "| 0.80 |" in report


class TestPool:
    def test_lists_unlabelled_sections_only(self) -> None:
        entries = golden_entries()
        pools = authoring.pool_candidates(
            [make_record(entries)], entries, k=10, threshold=0.4
        )
        q001 = {(c.paper_id, c.section) for c in pools["q001"]}
        # maira_2/Results is labelled; the other sections above 0.4 are candidates
        assert q001 == {("maira_2", "Methods"), ("conch", "Results > Table 2")}
        q002 = {(c.paper_id, c.section) for c in pools["q002"]}
        assert q002 == {("maira_2", "Results")}  # Methods and Table 2 are labelled
        assert "q003" not in pools  # off-corpus skipped by default

    def test_union_over_runs_keeps_the_best_rank(self) -> None:
        entries = [golden_entries()[0]]
        worse = RunRecord(
            meta=RunMeta("w", {}, "s", "g", 4),
            questions=(
                QuestionRun(
                    "q001",
                    QuestionType.FACTUAL,
                    (
                        RankedHit("conch", "Introduction", 1, "text", 0.8),
                        RankedHit("maira_2", "Methods", 1, "text", 0.7),
                    ),
                ),
            ),
        )
        pools = authoring.pool_candidates(
            [make_record(entries), worse], entries, k=10, threshold=0.4
        )
        by_section = {c.section: c for c in pools["q001"]}
        assert by_section["Methods"].best_rank == 2
        assert by_section["Methods"].runs == {"r", "w"}
        assert by_section["Introduction"].runs == {"w"}

    def test_threshold_and_k_are_applied_before_pooling(self) -> None:
        entries = [golden_entries()[0]]
        pools = authoring.pool_candidates(
            [make_record(entries)], entries, k=1, threshold=0.0
        )
        assert pools["q001"] == []  # top-1 is the labelled section

    def test_off_corpus_can_be_included(self) -> None:
        entries = golden_entries()
        pools = authoring.pool_candidates(
            [make_record(entries)], entries, include_off_corpus=True
        )
        assert "q003" in pools

    def test_command_prints_ids_and_sections_without_text(
        self, two_runs: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli_main(
            [
                "pool",
                "a",
                "b",
                *two_runs.paths(),
                "--adjudicated",
                str(two_runs.tmp / "adjudicated.jsonl"),
            ],
            services=two_runs.services(),
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "q001" in out and "maira_2 / Methods" in out and "maira_2:1" in out
        assert "[table]" in out
        assert "runs: a, b" in out

    def test_command_can_write_to_a_file(self, two_runs: Harness) -> None:
        out = two_runs.tmp / "pool.md"
        code = cli_main(
            [
                "pool",
                "a",
                *two_runs.paths(),
                "--out",
                str(out),
                "--adjudicated",
                str(two_runs.tmp / "adjudicated.jsonl"),
            ],
            services=two_runs.services(),
        )
        assert code == 0
        assert "Pooling candidates" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


class TestSections:
    def test_lists_sections_with_chunk_ids_and_types(
        self, harness: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli_main(["sections"], services=harness.services())

        out = capsys.readouterr().out
        assert code == 0
        assert "## maira_2 (2 chunks)" in out
        assert "- Results -> maira_2:0" in out
        assert "- Results > Table 2 -> conch:0 [table]" in out
        assert "table=1" in out
        for text in (TEXT_A, TEXT_B, TEXT_C, TEXT_D):
            assert text not in out  # section list holds no chunk text

    def test_paper_filter(
        self, harness: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            cli_main(["sections", "--paper", "conch"], services=harness.services()) == 0
        )
        out = capsys.readouterr().out
        assert "## conch" in out and "## maira_2" not in out

    def test_unknown_paper_exits_nonzero(self, harness: Harness) -> None:
        assert (
            cli_main(["sections", "--paper", "nope"], services=harness.services()) == 1
        )

    def test_dump_text_writes_one_file_per_paper_outside_the_repo(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        target = tmp_path / "dump"
        code = cli_main(
            ["sections", "--dump-text", str(target)], services=harness.services()
        )
        assert code == 0
        maira = (target / "maira_2.txt").read_text(encoding="utf-8")
        assert "=== maira_2:0 | text | Results | page ? ===" in maira
        assert TEXT_A in maira and TEXT_B in maira
        assert maira.index(TEXT_A) < maira.index(TEXT_B)  # chunk order
        assert TEXT_C in (target / "conch.txt").read_text(encoding="utf-8")

    def test_dump_text_inside_the_repo_is_refused_before_any_write(
        self, harness: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inside = authoring.REPO_ROOT / "data" / "eval" / "never_written_dump"
        code = cli_main(
            ["sections", "--dump-text", str(inside)], services=harness.services()
        )
        assert code == 1
        assert not inside.exists()
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "bad_id",
        ["../escaped", "..\\escaped", "sub/dir", "C:\\abs\\evil", "/abs/evil", "a b"],
    )
    def test_dump_text_rejects_a_paper_id_that_is_not_a_plain_name(
        self, tmp_path: Path, bad_id: str
    ) -> None:
        points = [*corpus_points(), make_point(bad_id, "Results", 0, text="x")]
        harness = Harness(tmp_path, points=points)
        target = tmp_path / "dump"

        code = cli_main(
            ["sections", "--dump-text", str(target)], services=harness.services()
        )

        assert code == 1
        assert not (tmp_path / "escaped.txt").exists()
        assert not list(target.glob("*"))  # nothing written, not even the good papers

    def test_empty_collection_exits_nonzero(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, points=[])
        assert cli_main(["sections"], services=harness.services()) == 1


# ---------------------------------------------------------------------------
# Reranker runs (Phase 10)
# ---------------------------------------------------------------------------

# The Harness corpus has three chunks at or above the 0.6 pre-filter (Results 0.9,
# Methods 0.8, Table 2 0.7) and one below it (Introduction 0.3). Scores below are in
# that dense order, so the reranker puts Table 2 first, then Results, then Methods.
RERANK_SCORES = [0.5, 0.2, 0.9]


class RerankHarness(Harness):
    """`Harness` whose reranker returns fixed scores and is built once per run."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.reranker = FakeReranker(scores=RERANK_SCORES)
        self.reranker_loads: list[str] = []

    def services(self) -> runner.Services:
        def make_reranker(name: str) -> FakeReranker:
            self.reranker_loads.append(name)
            return self.reranker

        base = super().services()
        base.reranker_factory = make_reranker
        base.reranker_disk_bytes = lambda _name: 91_000_000
        return base


@pytest.fixture
def rerank_harness(tmp_path: Path) -> RerankHarness:
    return RerankHarness(tmp_path)


class TestRerankConfigs:
    def test_three_named_reranker_configs_are_built_on_prefix(self) -> None:
        prefix = runner.NAMED_CONFIGS["prefix"]
        expected = {
            "rerank_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_bge": "BAAI/bge-reranker-base",
            "rerank_medcpt": "ncbi/MedCPT-Cross-Encoder",
        }
        for name, model in expected.items():
            spec = runner.NAMED_CONFIGS[name]
            assert spec.reranker == model
            assert (spec.embedder, spec.collection) == (
                prefix.embedder,
                prefix.collection,
            )
            assert spec.prefix == prefix.prefix
            assert spec.rewrite is prefix.rewrite

    def test_dense_configs_have_no_reranker(self) -> None:
        for name in ("baseline", "no_rewrite", "prefix", "bge_base"):
            assert runner.NAMED_CONFIGS[name].reranker is None

    def test_reranker_options_resolve_with_defaults(self) -> None:
        options = runner.make_run_options("rerank_bge", rewrite_model="m")
        config = options.config
        assert config.reranker == "BAAI/bge-reranker-base"
        assert config.rerank_query == "rewritten"
        assert config.prefilter_threshold == 0.6
        assert config.depth == 20

    def test_flags_override_depth_query_and_model_when_named(self) -> None:
        options = runner.make_run_options(
            "rerank_bge",
            name="rerank_medcpt_original_d40",
            rewrite_model="m",
            depth=40,
            rerank_query="original",
            reranker="medcpt",  # an alias
        )
        assert options.name == "rerank_medcpt_original_d40"
        assert options.config.depth == 40
        assert options.config.rerank_query == "original"
        assert options.config.reranker == "ncbi/MedCPT-Cross-Encoder"

    def test_reranker_flag_also_takes_a_full_model_name(self) -> None:
        options = runner.make_run_options(
            "prefix", name="x", rewrite_model="m", reranker="org/my-model"
        )
        assert options.config.reranker == "org/my-model"

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("depth", 40), ("rerank_query", "original"), ("reranker", "minilm")],
    )
    def test_each_flag_needs_an_explicit_name(self, flag: str, value: Any) -> None:
        kwargs: dict[str, Any] = {flag: value}
        with pytest.raises(runner.RunError, match="--name is required"):
            runner.make_run_options("rerank_bge", rewrite_model="m", **kwargs)

    def test_rerank_query_needs_a_reranker(self) -> None:
        with pytest.raises(runner.RunError, match="needs a reranker"):
            runner.make_run_options(
                "prefix", name="x", rewrite_model="m", rerank_query="original"
            )

    def test_cli_refuses_a_flag_without_a_name(
        self, rerank_harness: RerankHarness, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            code = rerank_harness.run("--depth", "40", config="rerank_bge")
        assert code == 1
        assert "--name is required" in caplog.text
        assert not rerank_harness.runs.exists()  # nothing was written
        assert rerank_harness.reranker_loads == []


class TestRerankRun:
    def test_run_records_rerank_scores_timing_and_meta(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run(config="rerank_minilm") == 0

        record = load_run(rerank_harness.runs / "rerank_minilm.json")
        config = record.meta.config
        assert config["score_kind"] == "rerank"
        assert config["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config["rerank_query"] == "rewritten"
        assert config["prefilter_threshold"] == 0.6
        assert config["reranker_disk_bytes"] == 91_000_000
        assert config["reranker_load_s"] >= 0
        assert record.meta.score_kind == "rerank"
        for question in record.questions:
            # Table 2 (0.9), Results (0.5), Methods (0.2): sorted by reranker score.
            assert [(h.section, h.score) for h in question.hits] == [
                ("Results > Table 2", 0.9),
                ("Results", 0.5),
                ("Methods", 0.2),
            ]
            assert question.rerank_ms is not None and question.rerank_ms >= 0
            # The 0.3 chunk was cut by the pre-filter, so only three hits remain.
            assert len(question.hits) == 3
            assert len(question.chunk_tokens) == 3

    def test_the_model_is_loaded_once_and_warmed_up_before_timing(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run(config="rerank_minilm") == 0

        assert rerank_harness.reranker_loads == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
        calls = rerank_harness.reranker.calls
        assert calls[0] == (runner.WARMUP_TEXT, [runner.WARMUP_TEXT])
        assert len(calls) == 1 + 3  # the warm-up plus one call per question

    def test_the_reranker_reads_the_same_passages_as_the_app(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run(config="rerank_minilm") == 0

        passages = rerank_harness.reranker.calls[1][1]
        # make_point gives no title, so the passage is "\n{section}\n{text}".
        assert passages[0] == f"\nResults\n{TEXT_A}"
        assert passages[1] == f"\nMethods\n{TEXT_B}"

    def test_report_shows_the_reranker_and_its_time(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run(config="rerank_minilm") == 0

        report = (rerank_harness.reports / "run-rerank_minilm.md").read_text(
            encoding="utf-8"
        )
        assert "## Reranker" in report
        assert "Time per question, median" in report
        assert "Time per question, 90th percentile" in report
        assert "91 MB" in report
        assert "Wrong section" in report
        # Cut at 0 for reranked scores, so the low scores are still counted.
        assert "Score kind: rerank; threshold: 0.00" in report

    def test_original_question_flag_changes_the_reranker_text_only(
        self, rerank_harness: RerankHarness
    ) -> None:
        code = rerank_harness.run(
            "--name",
            "orig",
            "--rerank-query",
            "original",
            config="rerank_minilm",
        )
        assert code == 0

        queries = [call[0] for call in rerank_harness.reranker.calls[1:]]
        assert queries == [entry.question for entry in rerank_harness.entries]
        # The vector search used the rewrite (the fake LLM's answer).
        assert set(rerank_harness.embedder.texts) == {
            runner.BGE_QUERY_PREFIX + "a rewritten question"
        }
        record = load_run(rerank_harness.runs / "orig.json")
        assert record.meta.config["rerank_query"] == "original"

    def test_depth_flag_sets_the_search_depth(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert (
            rerank_harness.run(
                "--name", "deep", "--depth", "40", config="rerank_minilm"
            )
            == 0
        )
        assert {c["limit"] for c in rerank_harness.qdrant.query_calls} == {40}
        record = load_run(rerank_harness.runs / "deep.json")
        assert record.meta.config["depth"] == 40

    def test_reranker_flag_replaces_the_model(
        self, rerank_harness: RerankHarness
    ) -> None:
        code = rerank_harness.run(
            "--name", "other", "--reranker", "bge", config="rerank_minilm"
        )
        assert code == 0
        assert rerank_harness.reranker_loads == ["BAAI/bge-reranker-base"]

    def test_dense_run_meta_is_marked_cosine_and_loads_no_model(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run() == 0  # no_rewrite

        record = load_run(rerank_harness.runs / "no_rewrite.json")
        assert record.meta.config["score_kind"] == "cosine"
        assert record.meta.config["reranker"] is None
        assert "reranker_load_s" not in record.meta.config
        assert all(q.rerank_ms is None for q in record.questions)
        assert rerank_harness.reranker_loads == []
        report = (rerank_harness.reports / "run-no_rewrite.md").read_text(
            encoding="utf-8"
        )
        assert "## Reranker" not in report
        assert "Score kind: cosine; threshold: 0.60" in report

    def test_dense_hits_are_identical_to_the_cosine_hits_of_before(
        self, rerank_harness: RerankHarness
    ) -> None:
        assert rerank_harness.run() == 0

        q001 = load_run(rerank_harness.runs / "no_rewrite.json").questions[0]
        assert [(h.paper_id, h.chunk_index, h.score) for h in q001.hits] == [
            ("maira_2", 0, 0.9),
            ("maira_2", 1, 0.8),
            ("conch", 0, 0.7),
            ("conch", 1, 0.3),
        ]

    def test_langsmith_experiment_follows_the_score_kind(
        self, rerank_harness: RerankHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rerank_harness.aevaluate = FakeAevaluate(rerank_harness.entries)

        assert rerank_harness.run("--langsmith", config="rerank_minilm") == 0

        call = rerank_harness.aevaluate.calls[0]
        assert call["metadata"]["threshold"] == 0.0
        assert call["metadata"]["score_kind"] == "rerank"
        # The experiment target reranked too: warm-up, 3 local, 3 experiment calls.
        assert len(rerank_harness.reranker.calls) == 1 + 3 + 3

    def test_langsmith_experiment_of_a_dense_run_keeps_0_6(
        self, rerank_harness: RerankHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        rerank_harness.aevaluate = FakeAevaluate(rerank_harness.entries)

        assert rerank_harness.run("--langsmith") == 0

        call = rerank_harness.aevaluate.calls[0]
        assert call["metadata"]["threshold"] == 0.6
        assert call["metadata"]["score_kind"] == "cosine"


class TestCompareAndGateByScoreKind:
    @staticmethod
    def reranked(entries: Sequence[GoldenEntry], sha: str) -> RunRecord:
        record = make_record(entries, sha, "rr")
        meta = dataclasses.replace(record.meta, config={"score_kind": "rerank"})
        return dataclasses.replace(record, meta=meta)

    def test_compare_cuts_each_run_at_its_own_default(self, harness: Harness) -> None:
        sha = golden_sha256(harness.golden)
        save_run(make_record(harness.entries, sha, "a"), harness.runs / "a.json")
        save_run(self.reranked(harness.entries, sha), harness.runs / "rr.json")

        code = cli_main(
            ["compare", "a", "rr", *harness.paths(), "--resamples", "50"],
            services=harness.services(),
        )

        assert code == 0
        report = (harness.reports / "compare-rr-vs-a.md").read_text(encoding="utf-8")
        assert "A at k=5, t=0.60, B at k=5, t=0.00" in report

    def test_an_explicit_threshold_applies_to_both_runs(self, harness: Harness) -> None:
        sha = golden_sha256(harness.golden)
        save_run(make_record(harness.entries, sha, "a"), harness.runs / "a.json")
        save_run(self.reranked(harness.entries, sha), harness.runs / "rr.json")

        code = cli_main(
            [
                "compare",
                "a",
                "rr",
                "--threshold",
                "0.3",
                *harness.paths(),
                "--resamples",
                "50",
            ],
            services=harness.services(),
        )

        assert code == 0
        report = (harness.reports / "compare-rr-vs-a.md").read_text(encoding="utf-8")
        assert "A at k=5, t=0.30, B at k=5, t=0.30" in report

    def test_two_cosine_runs_still_compare_at_0_6(self, two_runs: Harness) -> None:
        code = cli_main(
            ["compare", "a", "b", *two_runs.paths(), "--resamples", "50"],
            services=two_runs.services(),
        )
        assert code == 0
        report = (two_runs.reports / "compare-b-vs-a.md").read_text(encoding="utf-8")
        assert "A at k=5, t=0.60, B at k=5, t=0.60" in report

    def test_curve_gate_writes_the_refusal_report(self, two_runs: Harness) -> None:
        code = cli_main(
            ["curve", "a", "--gate", *two_runs.paths()], services=two_runs.services()
        )
        assert code == 0
        report = (two_runs.reports / "gate-a.md").read_text(encoding="utf-8")
        assert "Refusal curve: a" in report
        assert "| 0.00 |" in report and "| 1.00 |" in report
        assert not (two_runs.reports / "curve-a.md").exists()


class TestPoolAdjudicated:
    def test_rejected_candidates_are_skipped(self) -> None:
        entries = golden_entries()
        pools = authoring.pool_candidates(
            [make_record(entries)],
            entries,
            k=10,
            threshold=0.4,
            skip={("q001", "maira_2", "Methods")},
        )
        q001 = {(c.paper_id, c.section) for c in pools["q001"]}
        assert q001 == {("conch", "Results > Table 2")}
        # The same section for another question is still listed.
        assert authoring.pool_candidates(
            [make_record(entries)],
            entries,
            k=10,
            threshold=0.4,
            skip={("q002", "maira_2", "Methods")},
        )["q001"]

    def test_record_and_load_roundtrip_without_duplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "adjudicated.jsonl"
        added = authoring.record_rejections(
            path, [("q001", "maira_2", "Methods"), ("q001", "maira_2", "Methods")]
        )
        again = authoring.record_rejections(
            path, [("q001", "maira_2", "Methods"), ("q002", "conch", "Intro")]
        )

        assert (added, again) == (1, 1)
        assert authoring.load_adjudicated(path) == {
            ("q001", "maira_2", "Methods"),
            ("q002", "conch", "Intro"),
        }
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        assert rows[0] == {
            "question_id": "q001",
            "paper_id": "maira_2",
            "section": "Methods",
            "decision": "rejected",
        }

    def test_reason_is_stored_and_a_missing_file_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "adjudicated.jsonl"
        assert authoring.load_adjudicated(path) == set()
        authoring.record_rejections(
            path, [("q001", "p", "S")], reason="no answer in the section"
        )
        assert (
            json.loads(path.read_text("utf-8"))["reason"] == "no answer in the section"
        )

    def test_a_bad_line_names_the_file_and_line(self, tmp_path: Path) -> None:
        path = tmp_path / "adjudicated.jsonl"
        path.write_text('{"question_id": "q001"}\n', encoding="utf-8")
        with pytest.raises(runner.RunError, match=r"adjudicated\.jsonl:1"):
            authoring.load_adjudicated(path)

    def test_cli_records_rejections_and_skips_them_afterwards(
        self, two_runs: Harness, capsys: pytest.CaptureFixture[str]
    ) -> None:
        adjudicated = two_runs.tmp / "adjudicated.jsonl"
        base = ["pool", "a", *two_runs.paths(), "--adjudicated", str(adjudicated)]

        assert cli_main(base, services=two_runs.services()) == 0
        assert "maira_2 / Methods" in capsys.readouterr().out

        code = cli_main(
            [
                *base,
                "--reject",
                "q001",
                "maira_2",
                "Methods",
                "--reason",
                "read, no answer",
            ],
            services=two_runs.services(),
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Rejected 1 new candidate(s)" in out
        assert "- maira_2 / Methods" not in out.split("## q002")[0]  # skipped for q001
        assert "1 rejected candidate(s) in adjudicated.jsonl are skipped." in out

        # A later pass without --reject still skips it.
        assert cli_main(base, services=two_runs.services()) == 0
        later = capsys.readouterr().out
        assert "- maira_2 / Methods" not in later.split("## q002")[0]

    def test_cli_refuses_an_unknown_question_id(
        self, two_runs: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        adjudicated = two_runs.tmp / "adjudicated.jsonl"
        with caplog.at_level(logging.ERROR):
            code = cli_main(
                [
                    "pool",
                    "a",
                    *two_runs.paths(),
                    "--adjudicated",
                    str(adjudicated),
                    "--reject",
                    "q999",
                    "maira_2",
                    "Methods",
                ],
                services=two_runs.services(),
            )
        assert code == 1
        assert "q999" in caplog.text
        assert not adjudicated.exists()

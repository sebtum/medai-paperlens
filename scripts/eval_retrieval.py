"""Retrieval evaluation runner CLI (ADR-0023, ADR-0024).

Run as a module:  python -m scripts.eval_retrieval {run,compare,curve,sections,pool}

Only argument parsing, `.env` loading (like scripts/ingest.py) and printing live here.
The logic is in `app/evaluation/runner.py` and `app/evaluation/authoring.py`. Exit code
1 on any failure; the full message goes to the log.
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import dotenv

from app.evaluation.authoring import (
    POOL_DEFAULT_K,
    POOL_DEFAULT_THRESHOLD,
    pool_command,
    sections_command,
)
from app.evaluation.compare import PRIMARY_CUTOFF
from app.evaluation.langsmith_eval import EXPERIMENT_K, EXPERIMENT_THRESHOLD
from app.evaluation.runner import (
    DEFAULT_GOLDEN,
    DEFAULT_PAPERS_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_REWRITES,
    DEFAULT_RUNS_DIR,
    EVAL_ERRORS,
    NAMED_CONFIGS,
    Services,
    compare_command,
    curve_command,
    make_run_options,
    run_command,
)
from app.retrieval.ingestion import COLLECTION_NAME
from scripts.sync_golden import run_git

logger = logging.getLogger(__name__)


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _common(parser: argparse.ArgumentParser, *, papers: bool = False) -> None:
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN),
        help="Golden JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory of run files (default: %(default)s).",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory for reports (default: %(default)s).",
    )
    if papers:
        parser.add_argument(
            "--papers-dir",
            default=os.getenv("PAPERS_DIR", str(DEFAULT_PAPERS_DIR)),
            help="Paper manifests for golden validation (env: PAPERS_DIR).",
        )


def _add_run(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = sub.add_parser("run", help="Retrieve every golden question and write a run.")
    _common(run, papers=True)
    run.add_argument(
        "--config",
        choices=sorted(NAMED_CONFIGS),
        default="baseline",
        help="Named configuration (default: %(default)s).",
    )
    run.add_argument("--name", help="Run name (default: the config name).")
    run.add_argument("--collection", help="Override the config's Qdrant collection.")
    run.add_argument("--embedder", help="Override the config's embedding model.")
    run.add_argument(
        "--rewrites",
        default=str(DEFAULT_REWRITES),
        help="Rewrite cache file (default: %(default)s).",
    )
    run.add_argument(
        "--rewrite-model",
        help="Rewrite model name for the cache key (default: from OLLAMA_MODEL / "
        "GEMINI_MODEL).",
    )
    run.add_argument(
        "--no-llm",
        action="store_true",
        help="Never call an LLM; a missing rewrite is an error.",
    )
    run.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validate_golden (work-in-progress golden sets only).",
    )
    run.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and run the pre-flight against Qdrant, then stop.",
    )
    run.add_argument(
        "--langsmith",
        action="store_true",
        help=f"Also run a LangSmith experiment (k={EXPERIMENT_K}, "
        f"t={EXPERIMENT_THRESHOLD}); needs LANGSMITH_API_KEY and a synced dataset.",
    )
    run.add_argument(
        "--dataset", help="LangSmith dataset name (default: the synced one)."
    )


def _add_reports(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    compare = sub.add_parser("compare", help="Compare run B against run A.")
    _common(compare)
    compare.add_argument("run_a", metavar="A", help="Run name or file (baseline side).")
    compare.add_argument(
        "run_b", metavar="B", help="Run name or file (candidate side)."
    )
    compare.add_argument("--metric", choices=["ndcg", "recall", "mrr"], default="ndcg")
    compare.add_argument("--k", type=int, default=PRIMARY_CUTOFF.k)
    compare.add_argument("--threshold", type=float, default=PRIMARY_CUTOFF.threshold)
    compare.add_argument(
        "--threshold-b",
        type=float,
        help="Threshold for B (threshold comparison: pass the same run as A and B).",
    )
    compare.add_argument(
        "--threshold-rule",
        action="store_true",
        help="Apply the threshold rule (implied by --threshold-b).",
    )
    compare.add_argument("--resamples", type=int, default=1000)
    compare.add_argument("--out", help="Report file (default under --reports-dir).")

    curve = sub.add_parser("curve", help="Threshold curve of one run.")
    _common(curve)
    curve.add_argument("run", metavar="RUN", help="Run name or file.")
    curve.add_argument("--k", type=int, default=PRIMARY_CUTOFF.k)
    curve.add_argument("--out", help="Report file (default under --reports-dir).")


def _add_authoring(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sections = sub.add_parser(
        "sections", help="List sections and chunk ids of a collection (authoring)."
    )
    sections.add_argument("--collection", default=COLLECTION_NAME)
    sections.add_argument("--paper", help="Only this paper id.")
    sections.add_argument(
        "--dump-text",
        metavar="DIR",
        help="Also write each paper's chunk text to DIR, which must be outside the "
        "repository. Never commit it.",
    )

    pool = sub.add_parser(
        "pool", help="Unlabelled sections in each run's top-k (adjudication list)."
    )
    _common(pool)
    pool.add_argument("runs", nargs="+", metavar="RUN", help="Run names or files.")
    pool.add_argument("--k", type=int, default=POOL_DEFAULT_K)
    pool.add_argument("--threshold", type=float, default=POOL_DEFAULT_THRESHOLD)
    pool.add_argument(
        "--include-off-corpus",
        action="store_true",
        help="Also list candidates for off-corpus questions.",
    )
    pool.add_argument("--out", help="Write the list to this file instead of stdout.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval_retrieval",
        description="Retrieval evaluation runner (ADR-0023, ADR-0024).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_reports(sub)
    _add_authoring(sub)
    return parser


def _dispatch(args: argparse.Namespace, services: Services) -> list[str]:
    """Run the chosen command; returns the lines to print."""
    if args.command == "run":
        options = make_run_options(
            args.config,
            name=args.name,
            collection=args.collection,
            embedder=args.embedder,
            rewrite_model=args.rewrite_model,
            golden_path=Path(args.golden),
            papers_dir=Path(args.papers_dir),
            rewrites_path=Path(args.rewrites),
            runs_dir=Path(args.runs_dir),
            reports_dir=Path(args.reports_dir),
            skip_validate=args.skip_validate,
            preflight_only=args.preflight_only,
            no_llm=args.no_llm,
            langsmith=args.langsmith,
            dataset_name=args.dataset,
        )
        return asyncio.run(run_command(options, services))
    if args.command == "compare":
        return compare_command(
            args.run_a,
            args.run_b,
            golden_path=Path(args.golden),
            runs_dir=Path(args.runs_dir),
            reports_dir=Path(args.reports_dir),
            metric=args.metric,
            k=args.k,
            threshold=args.threshold,
            threshold_b=args.threshold_b,
            threshold_rule=args.threshold_rule,
            resamples=args.resamples,
            out=_path(args.out),
        )
    if args.command == "curve":
        return curve_command(
            args.run,
            golden_path=Path(args.golden),
            runs_dir=Path(args.runs_dir),
            reports_dir=Path(args.reports_dir),
            k=args.k,
            out=_path(args.out),
        )
    if args.command == "sections":
        return asyncio.run(
            sections_command(
                services.client_factory(),
                args.collection,
                paper=args.paper,
                dump_text=_path(args.dump_text),
            )
        )
    return pool_command(
        args.runs,
        golden_path=Path(args.golden),
        runs_dir=Path(args.runs_dir),
        k=args.k,
        threshold=args.threshold,
        include_off_corpus=args.include_off_corpus,
        out=_path(args.out),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    services: Services | None = None,
    load_env: Callable[[], object] = dotenv.load_dotenv,
) -> int:
    """CLI entry point; returns the process exit code. Collaborators are injectable."""
    args = build_parser().parse_args(argv)
    # Read `.env` like the API and scripts/ingest.py do.
    load_env()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    try:
        lines = _dispatch(args, services or Services(git=run_git))
    except EVAL_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # Qdrant/Ollama/LangSmith/network errors
        logger.error("%s failed: %s: %s", args.command, type(exc).__name__, exc)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

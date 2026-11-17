"""Sync data/eval/golden.jsonl to the LangSmith dataset (ADR-0024).

Run as a module so `app` imports work:  python -m scripts.sync_golden
Needs LANGSMITH_API_KEY (read from the environment or `.env`). Exit code 1 on any
failure: missing key, invalid or uncommitted golden file, LangSmith error.
"""

import argparse
import logging
import os
import subprocess  # nosec B404 - fixed `git` argument lists only, never shell=True
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import dotenv

from app.evaluation.langsmith_sync import (
    DATASET_NAME,
    GitError,
    GitRunner,
    SyncError,
    ensure_golden_committed,
    git_short_hash,
    prepare_golden,
    sync_golden,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "data" / "eval" / "golden.jsonl"
DEFAULT_PAPERS_DIR = REPO_ROOT / "data" / "papers"


def run_git(args: Sequence[str]) -> str:
    """Run `git <args>` in the repo root and return stdout; `GitError` on failure."""
    try:
        done = subprocess.run(  # nosec B603 B607 - fixed args, no shell
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _make_client() -> Any:
    from langsmith import Client

    return Client()


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], Any] = _make_client,
    git: GitRunner = run_git,
    load_env: Callable[[], object] = dotenv.load_dotenv,
) -> int:
    """CLI entry point; returns the process exit code. Collaborators are injectable."""
    parser = argparse.ArgumentParser(
        description="Sync the golden set to the LangSmith dataset."
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN),
        help="Golden JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--papers-dir",
        default=os.getenv("PAPERS_DIR", str(DEFAULT_PAPERS_DIR)),
        help="Paper manifests used to validate the golden set (env: PAPERS_DIR).",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help="LangSmith dataset name (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    # Read `.env` like the API and scripts/ingest.py do.
    load_env()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error(
            "LANGSMITH_API_KEY is not set. Set it in the environment or .env "
            "(see .env.example) to sync the golden set."
        )
        return 1

    golden = Path(args.golden)
    try:
        ensure_golden_committed(golden, git)
        entries, sha256 = prepare_golden(golden, Path(args.papers_dir))
        short = git_short_hash(git)
        result = sync_golden(
            client_factory(), entries, sha256, short, dataset_name=args.dataset
        )
    except SyncError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # LangSmith/network errors: report, exit non-zero
        logger.error("Sync failed: %s", exc)
        return 1
    logger.info("%s", result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

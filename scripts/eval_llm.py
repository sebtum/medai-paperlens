"""Generate a human-scorable groundedness/quality review sheet for candidate
LLM providers, using the same fixture questions as benchmark_llm.py.

This produces answers for manual review - it does not score anything itself.
No eval framework is used; at ~12 questions x N models, eyeballing beats
standing up ragas/deepeval/promptfoo.

Usage (run as a module - the app package isn't installed editable, so
`python scripts\\eval_llm.py` directly would fail to import `app`):
    .\\.venv\\Scripts\\python.exe -m scripts.eval_llm --models ollama:qwen3.5:4b
    .\\.venv\\Scripts\\python.exe -m scripts.eval_llm \
        --models ollama:qwen3.5:4b,gemini:gemini-2.0-flash --format markdown
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

from app.workflow.generate import _PROMPT as GENERATE_PROMPT
from scripts.llm_bench.fixtures import get_fixture_questions
from scripts.llm_bench.providers import build_provider

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "provider",
    "model",
    "question_id",
    "question",
    "evidence_summary",
    "answer",
    "groundedness_score",
    "hallucination_flag",
    "notes",
]


async def run_eval(specs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixture_questions = get_fixture_questions()

    for spec in specs:
        client, provider_name, model_name = build_provider(spec)
        logger.info("Evaluating %s (%d questions)", spec, len(fixture_questions))
        async with client:
            for fq in fixture_questions:
                evidence_text = "\n\n".join(
                    f"[{i + 1}] {c.excerpt}" for i, c in enumerate(fq.citations)
                )
                try:
                    answer = (
                        await client.generate(
                            GENERATE_PROMPT.format(
                                question=fq.question, evidence=evidence_text
                            )
                        )
                    ).strip()
                except Exception as exc:
                    logger.warning(
                        "generate failed: %s/%s %s: %s",
                        provider_name,
                        model_name,
                        fq.id,
                        exc,
                    )
                    answer = f"[ERROR] {exc}"

                rows.append(
                    {
                        "provider": provider_name,
                        "model": model_name,
                        "question_id": fq.id,
                        "question": fq.question,
                        "evidence_summary": " | ".join(c.title for c in fq.citations),
                        "answer": answer,
                        "groundedness_score": "",
                        "hallucination_flag": "",
                        "notes": "",
                    }
                )
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines: list[str] = []
    for row in rows:
        lines.append(f"## {row['question_id']} - {row['provider']}:{row['model']}")
        lines.append(f"**Question:** {row['question']}")
        lines.append(f"**Evidence:** {row['evidence_summary']}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("> " + row["answer"].replace("\n", "\n> "))
        lines.append("")
        lines.append("Groundedness (1-5): ___  Hallucination (Y/N): ___  Notes:")
        lines.append("")
        lines.append("---")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a human-scorable groundedness review sheet."
    )
    parser.add_argument(
        "--models",
        default=os.getenv(
            "EVAL_MODELS",
            "ollama:qwen3.5:4b,ollama:llama3.2:3b,gemini:gemini-2.0-flash",
        ),
        help="Comma-separated provider:model specs (env: EVAL_MODELS).",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EVAL_OUTPUT", ""),
        help="Output path (env: EVAL_OUTPUT). Defaults to eval_results.<format ext>.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown"],
        default="csv",
        help="Output format.",
    )
    args = parser.parse_args()

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    if not specs:
        logger.error("No provider specs given via --models.")
        sys.exit(1)

    output_path = Path(
        args.output or f"eval_results.{'md' if args.format == 'markdown' else 'csv'}"
    )

    try:
        rows = asyncio.run(run_eval(specs))
    except Exception as exc:
        logger.error("Eval setup failed: %s", exc)
        sys.exit(1)

    if args.format == "markdown":
        _write_markdown(rows, output_path)
    else:
        _write_csv(rows, output_path)
    logger.info("Wrote %d rows to %s", len(rows), output_path)

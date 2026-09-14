"""Benchmark latency and stability of candidate LLM providers/models.

Runs the exact rewrite/generate prompts used by the LangGraph pipeline
against a fixed set of local fixture questions (no live Qdrant/embedding
calls), so you can compare models before wiring one into the app. One
crashing/slow model does not abort the run - failures are recorded as rows,
not raised.

Reports real tokens/sec, not a character-count proxy: Ollama's /api/generate
response includes eval_count (tokens generated) and eval_duration
(generation-only time), and Gemini's response includes usage_metadata token
counts - both are captured directly rather than estimated. Also surfaces
`thinking_chars`/`thinking_tokens` so you can see whether a model is
generating hidden reasoning despite the pipeline's plain-text '/no_think'
prompt hint (which is a Qwen chat-template convention, not the same as
Ollama's `think` API parameter).

Usage (run as a module - the app package isn't installed editable, so
`python scripts\\benchmark_llm.py` directly would fail to import `app`):
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_llm --models ollama:qwen3.5:4b
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_llm \
        --models ollama:qwen3.5:4b,ollama:llama3.2:3b,gemini:gemini-2.0-flash \
        --repeat 3
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from app.workflow.generate import _PROMPT as GENERATE_PROMPT
from app.workflow.rewrite import _PROMPT as REWRITE_PROMPT
from scripts.llm_bench.fixtures import get_fixture_questions
from scripts.llm_bench.providers import (
    build_provider,
    get_ollama_model_size,
    ollama_generate_with_metrics,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "provider",
    "model",
    "question_id",
    "stage",
    "run",
    "latency_s",
    "success",
    "error",
    "model_size_bytes",
    "output_chars",
    "tokens_generated",
    "gen_duration_s",
    "tokens_per_sec",
    "prompt_tokens",
    "thinking_chars",
    "thinking_tokens",
]


async def _generate_with_metrics(
    raw_http: httpx.AsyncClient,
    base_url: str,
    provider_name: str,
    model_name: str,
    client: Any,
    prompt: str,
) -> dict[str, Any]:
    if provider_name == "ollama":
        return await ollama_generate_with_metrics(
            raw_http, base_url, model_name, prompt
        )
    if provider_name == "gemini":
        return await client.generate_with_metrics(prompt)
    raise ValueError(f"Unknown provider {provider_name!r}")


async def _timed_call(
    raw_http: httpx.AsyncClient,
    base_url: str,
    provider_name: str,
    model_name: str,
    client: Any,
    question_id: str,
    stage: str,
    run_idx: int,
    prompt: str,
    model_size: int | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = await _generate_with_metrics(
            raw_http, base_url, provider_name, model_name, client, prompt
        )
        tokens_generated = result.get("tokens_generated")
        gen_duration_s = result.get("gen_duration_s")
        tokens_per_sec = (
            round(tokens_generated / gen_duration_s, 1)
            if tokens_generated and gen_duration_s
            else None
        )
        return {
            "provider": provider_name,
            "model": model_name,
            "question_id": question_id,
            "stage": stage,
            "run": run_idx,
            "latency_s": round(time.perf_counter() - t0, 3),
            "success": True,
            "error": "",
            "model_size_bytes": model_size,
            "output_chars": len(result.get("output", "")),
            "tokens_generated": tokens_generated,
            "gen_duration_s": (
                round(gen_duration_s, 3) if gen_duration_s is not None else None
            ),
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens": result.get("prompt_tokens"),
            "thinking_chars": result.get("thinking_chars"),
            "thinking_tokens": result.get("thinking_tokens"),
        }
    except Exception as exc:
        logger.warning(
            "call failed: %s/%s %s run %d: %s",
            provider_name,
            model_name,
            stage,
            run_idx,
            exc,
        )
        return {
            "provider": provider_name,
            "model": model_name,
            "question_id": question_id,
            "stage": stage,
            "run": run_idx,
            "latency_s": round(time.perf_counter() - t0, 3),
            "success": False,
            "error": str(exc),
            "model_size_bytes": model_size,
            "output_chars": 0,
            "tokens_generated": None,
            "gen_duration_s": None,
            "tokens_per_sec": None,
            "prompt_tokens": None,
            "thinking_chars": None,
            "thinking_tokens": None,
        }


async def run_benchmark(specs: list[str], repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixture_questions = get_fixture_questions()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    for spec in specs:
        client, provider_name, model_name = build_provider(spec)
        logger.info(
            "Benchmarking %s (%d questions x %d runs)",
            spec,
            len(fixture_questions),
            repeat,
        )
        async with (
            client,
            httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
            ) as raw_http,
        ):
            # Ollama loads a model lazily on first request, so querying /api/ps
            # up front would always miss - fetch the size once the first call
            # has warmed the model instead.
            model_size: int | None = None
            for fq in fixture_questions:
                evidence_text = "\n\n".join(
                    f"[{i + 1}] {c.excerpt}" for i, c in enumerate(fq.citations)
                )
                for run_idx in range(repeat):
                    rewrite_row = await _timed_call(
                        raw_http,
                        base_url,
                        provider_name,
                        model_name,
                        client,
                        fq.id,
                        "rewrite",
                        run_idx,
                        REWRITE_PROMPT.format(question=fq.question),
                        model_size,
                    )
                    if model_size is None and provider_name == "ollama":
                        model_size = await get_ollama_model_size(
                            raw_http, base_url, model_name
                        )
                        rewrite_row["model_size_bytes"] = model_size
                    rows.append(rewrite_row)
                    rows.append(
                        await _timed_call(
                            raw_http,
                            base_url,
                            provider_name,
                            model_name,
                            client,
                            fq.id,
                            "generate",
                            run_idx,
                            GENERATE_PROMPT.format(
                                question=fq.question, evidence=evidence_text
                            ),
                            model_size,
                        )
                    )
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> None:
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault((row["provider"], row["model"]), []).append(row)
    for (provider_name, model_name), model_rows in by_model.items():
        successes = [r for r in model_rows if r["success"]]
        failures = len(model_rows) - len(successes)
        avg_latency = (
            sum(r["latency_s"] for r in successes) / len(successes)
            if successes
            else float("nan")
        )
        tps_values = [r["tokens_per_sec"] for r in successes if r["tokens_per_sec"]]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else None
        thinking_hits = sum(
            1 for r in successes if r["thinking_chars"] or r["thinking_tokens"]
        )
        logger.info(
            "%s:%s -> %d/%d succeeded, avg latency %.2fs, avg %s tok/s",
            provider_name,
            model_name,
            len(successes),
            len(model_rows),
            avg_latency,
            f"{avg_tps:.1f}" if avg_tps is not None else "n/a",
        )
        if failures:
            logger.warning(
                "%s:%s -> %d call(s) failed", provider_name, model_name, failures
            )
        if thinking_hits:
            logger.warning(
                "%s:%s -> %d/%d call(s) showed hidden reasoning tokens despite "
                "the /no_think prompt hint",
                provider_name,
                model_name,
                thinking_hits,
                len(successes),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark latency/stability of candidate LLM providers."
    )
    parser.add_argument(
        "--models",
        default=os.getenv(
            "BENCHMARK_MODELS",
            "ollama:qwen3.5:4b,ollama:llama3.2:3b,gemini:gemini-2.0-flash",
        ),
        help="Comma-separated provider:model specs (env: BENCHMARK_MODELS).",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("BENCHMARK_OUTPUT", "benchmark_results.csv"),
        help="CSV output path (env: BENCHMARK_OUTPUT).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Times to repeat each question per model.",
    )
    args = parser.parse_args()

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    if not specs:
        logger.error("No provider specs given via --models.")
        sys.exit(1)

    try:
        rows = asyncio.run(run_benchmark(specs, args.repeat))
    except Exception as exc:
        logger.error("Benchmark setup failed: %s", exc)
        sys.exit(1)

    _write_csv(rows, Path(args.output))
    logger.info("Wrote %d rows to %s", len(rows), args.output)
    _summarize(rows)

"""Benchmark latency and stability of candidate LLM providers/models.

Runs the exact rewrite/generate prompts used by the LangGraph pipeline
against a fixed set of local fixture questions (no live Qdrant/embedding
calls), so you can compare models before wiring one into the app. One
crashing/slow model does not abort the run - failures are recorded as rows,
not raised, and results are written to the CSV incrementally after each
(model, mode) block finishes (not just once at the end), so a slow or
crashing later model never costs you the results already gathered.

Reports real tokens/sec, not a character-count proxy: Ollama's /api/generate
response includes eval_count (tokens generated) and eval_duration
(generation-only time), and Gemini's response includes usage_metadata token
counts - both are captured directly rather than estimated. Also surfaces
`thinking_chars`/`thinking_tokens` so you can see whether a model is
generating hidden reasoning despite the pipeline's plain-text '/no_think'
prompt hint.

For Ollama models, each spec is run once per requested "mode": `no_think`
sets Ollama's actual `think` API parameter to False, `think` sets it to
True - both on top of the pipeline's real prompt (which already contains
the literal '/no_think' text hint), so you can see whether the API-level
switch actually changes speed/output versus the text hint alone. Gemini
specs ignore --modes and always run once as mode=default (Gemini's
thinking controls are a different mechanism, out of scope here).

Usage (run as a module - the app package isn't installed editable, so
`python scripts\\benchmark_llm.py` directly would fail to import `app`):
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_llm --models ollama:qwen3.5:4b
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_llm \
        --models ollama:qwen3.5:4b,ollama:qwen3:1.7b,gemini:gemini-2.0-flash \
        --modes no_think,think --repeat 3
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
    "mode",
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

# Ollama's `think` API parameter per mode - not the same as the prompt's
# plain-text '/no_think' hint, which stays baked into the real prompts either way.
THINK_MODES: dict[str, bool] = {"no_think": False, "think": True}


def _modes_for(provider_name: str, requested: list[str]) -> list[str]:
    if provider_name != "ollama":
        return ["default"]
    modes = [m for m in requested if m in THINK_MODES]
    return modes or ["no_think"]


async def _generate_with_metrics(
    raw_http: httpx.AsyncClient,
    base_url: str,
    provider_name: str,
    model_name: str,
    client: Any,
    prompt: str,
    think: bool | None,
) -> dict[str, Any]:
    if provider_name == "ollama":
        return await ollama_generate_with_metrics(
            raw_http, base_url, model_name, prompt, think=think
        )
    if provider_name == "gemini":
        return await client.generate_with_metrics(prompt)
    raise ValueError(f"Unknown provider {provider_name!r}")


async def _timed_call(
    raw_http: httpx.AsyncClient,
    base_url: str,
    provider_name: str,
    model_name: str,
    mode: str,
    think: bool | None,
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
            raw_http, base_url, provider_name, model_name, client, prompt, think
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
            "mode": mode,
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
            "call failed: %s/%s mode=%s %s run %d: %s",
            provider_name,
            model_name,
            mode,
            stage,
            run_idx,
            exc,
        )
        return {
            "provider": provider_name,
            "model": model_name,
            "mode": mode,
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


async def run_one_block(
    spec: str, mode: str, think: bool | None, repeat: int
) -> list[dict[str, Any]]:
    """Run every fixture question once (x repeat) for a single (spec, mode)."""
    rows: list[dict[str, Any]] = []
    fixture_questions = get_fixture_questions()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    client, provider_name, model_name = build_provider(spec)

    logger.info(
        "Benchmarking %s mode=%s (%d questions x %d runs)",
        spec,
        mode,
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
                    mode,
                    think,
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
                        mode,
                        think,
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


def _summarize(rows: list[dict[str, Any]]) -> None:
    if not rows:
        logger.warning("No rows to summarize.")
        return
    successes = [r for r in rows if r["success"]]
    failures = len(rows) - len(successes)
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
    provider_name = rows[0]["provider"]
    model_name = rows[0]["model"]
    mode = rows[0]["mode"]
    logger.info(
        "RESULT %s:%s mode=%s -> %d/%d succeeded, avg latency %.2fs, avg %s tok/s",
        provider_name,
        model_name,
        mode,
        len(successes),
        len(rows),
        avg_latency,
        f"{avg_tps:.1f}" if avg_tps is not None else "n/a",
    )
    if failures:
        logger.warning(
            "RESULT %s:%s mode=%s -> %d call(s) failed",
            provider_name,
            model_name,
            mode,
            failures,
        )
    if thinking_hits:
        logger.warning(
            "RESULT %s:%s mode=%s -> %d/%d call(s) showed reasoning "
            "chars/tokens despite the '/no_think' prompt hint",
            provider_name,
            model_name,
            mode,
            thinking_hits,
            len(successes),
        )


async def run_benchmark_incremental(
    specs: list[str], modes_requested: list[str], repeat: int, output_path: Path
) -> int:
    total = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()
        for spec in specs:
            provider_name = spec.split(":", 1)[0]
            for mode in _modes_for(provider_name, modes_requested):
                think = THINK_MODES.get(mode) if provider_name == "ollama" else None
                try:
                    rows = await run_one_block(spec, mode, think, repeat)
                except Exception as exc:
                    logger.error(
                        "%s mode=%s: setup failed, skipping (%s)", spec, mode, exc
                    )
                    continue
                writer.writerows(rows)
                f.flush()
                total += len(rows)
                _summarize(rows)
                logger.info("-> %d rows written so far to %s", total, output_path)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark latency/stability of candidate LLM providers."
    )
    parser.add_argument(
        "--models",
        default=os.getenv(
            "BENCHMARK_MODELS",
            "ollama:qwen3.5:4b,ollama:qwen3:1.7b,gemini:gemini-2.0-flash",
        ),
        help="Comma-separated provider:model specs (env: BENCHMARK_MODELS).",
    )
    parser.add_argument(
        "--modes",
        default=os.getenv("BENCHMARK_THINK_MODES", "no_think,think"),
        help=(
            "Comma-separated think modes for Ollama models: no_think, think "
            "(sets Ollama's `think` API parameter explicitly). Ignored for "
            "Gemini specs, which always run once as mode=default "
            "(env: BENCHMARK_THINK_MODES)."
        ),
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
        help="Times to repeat each question per (model, mode).",
    )
    args = parser.parse_args()

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    if not specs:
        logger.error("No provider specs given via --models.")
        sys.exit(1)
    modes_requested = [m.strip() for m in args.modes.split(",") if m.strip()]

    try:
        total = asyncio.run(
            run_benchmark_incremental(
                specs, modes_requested, args.repeat, Path(args.output)
            )
        )
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        sys.exit(1)

    logger.info("Benchmark complete: wrote %d rows total to %s", total, args.output)

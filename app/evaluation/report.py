"""Markdown renderers for run summaries, comparisons and threshold curves (ADR-0023).

Every report states the run config (R4) and the label caveat (R9) in its header. No
paper text appears in any report: only ids, section labels, counts and scores.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from app.evaluation.compare import (
    FEW_QUESTIONS,
    Comparison,
    Cutoff,
    GateCurve,
    RunMeta,
    RunSummary,
    ThresholdCurve,
    TypeComparison,
    Verdict,
    threshold_verdict,
    verdict,
)

R9_SENTENCE = "golden set LLM-labelled (Claude), not human-verified"
NOT_AVAILABLE = "n/a"


def _num(value: float | None, digits: int = 3) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.{digits}f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _run_block(label: str, meta: RunMeta) -> str:
    config = json.dumps(meta.config, sort_keys=True, ensure_ascii=False)
    return (
        f"- {label}: `{meta.name}`, config `{config}`, "
        f"golden sha256 `{meta.golden_sha256[:12]}`, git `{meta.git_hash}`, "
        f"{meta.chunk_count} chunks in the collection"
    )


def _header(title: str, blocks: Sequence[str]) -> str:
    return "\n".join([f"# {title}", "", f"Note: {R9_SENTENCE}.", "", *blocks, ""])


def _cutoff_text(cutoff: Cutoff) -> str:
    return f"k={cutoff.k}, t={cutoff.threshold:.2f}"


def _megabytes(size: object) -> str:
    return NOT_AVAILABLE if not isinstance(size, int) else f"{size / 1e6:.0f} MB"


def _reranker_lines(summary: RunSummary) -> list[str]:
    """Reranker section of a run report: model, load, disk size, time per question."""
    config = summary.meta.config
    timing = summary.rerank_timing
    if config.get("reranker") is None and timing is None:
        return []
    load = config.get("reranker_load_s")
    rows = [
        ["Reranker model", str(config.get("reranker", NOT_AVAILABLE))],
        ["Query text", str(config.get("rerank_query", NOT_AVAILABLE))],
        [
            (
                "Cosine cut of the dense list (inside the hybrid query)"
                if config.get("hybrid")
                else "Cosine pre-filter"
            ),
            _num(config.get("prefilter_threshold"), 2),
        ],
        ["Load time", f"{load:.1f} s" if isinstance(load, (int, float)) else "n/a"],
        ["Disk size", _megabytes(config.get("reranker_disk_bytes"))],
    ]
    if timing is None:
        rows.append(["Time per question", NOT_AVAILABLE])
    else:
        rows.append(
            ["Time per question, median", f"{timing.median_ms:.0f} ms (n={timing.n})"]
        )
        rows.append(["Time per question, 90th percentile", f"{timing.p90_ms:.0f} ms"])
    return [
        "## Reranker",
        "",
        "Scores are reranker output (sigmoid, 0 to 1), not cosine. The time is one "
        "reranker call per question, measured after one warm-up call, on this "
        "machine; it excludes the vector search and the rewrite.",
        "",
        _table(["Measure", "Value"], rows),
        "",
    ]


def _hybrid_lines(summary: RunSummary) -> list[str]:
    """Note for a hybrid run: what its scores are and what the dense list was cut at."""
    config = summary.meta.config
    if not config.get("hybrid"):
        return []
    lines = [
        "## Hybrid search",
        "",
        "Hits come from one Qdrant query: a dense list (cosine at or above "
        f"{_num(config.get('prefilter_threshold'), 2)}) and a BM25 keyword list, "
        "merged by reciprocal rank fusion (RRF). The keyword query is the same "
        "rewritten question without the dense instruction prefix.",
        "",
    ]
    if summary.meta.score_kind == "rrf":
        lines += [
            "Scores are RRF values: rank-based, neither cosine nor BM25, and small "
            "(they cannot be compared with the dense runs' cosine scores). The run is "
            "not cut at 0.6; the cosine cut already applies to the dense list inside "
            "the query. Treat the score as an order, not as a confidence.",
            "",
        ]
    return lines


def render_run_summary(summary: RunSummary) -> str:
    """Run summary: k = 3/5/10 table, per-type table, rates, n-gram overlap."""
    meta = summary.meta
    out = [
        _header(
            f"Retrieval run: {meta.name}",
            [
                _run_block("Run", meta),
                f"- Score kind: {meta.score_kind}; threshold: "
                f"{summary.threshold:.2f}; "
                f"{summary.n_on_corpus} on-corpus and {summary.n_off_corpus} "
                "off-corpus questions",
            ],
        ),
        "## By k (on-corpus questions)",
        "",
        "Precision is the share of the top-k chunks that sit in a labelled section "
        "(how much of the context is on target). Recall vs ceiling is recall divided "
        "by the best the k slots allow, min(k, number of labels); 1.00 means as many "
        "labelled sections as k slots can hold. Both are reported, not used by the "
        "decision rule. "
        "Wrong section is the share of the top-k chunks that come from a paper with a "
        "label for the question but from a section without one (right paper, wrong "
        "section). "
        "Crowding is the mean number of distinct papers in the top-k chunks. "
        "Table hit rate is the share of `table` questions with a table chunk in a "
        "labelled section in the top-k. Evidence tokens are the mean token estimate of "
        "the top-k chunks (n/a when the run holds no token counts). "
        "The default k stays 5; this table only measures.",
        "",
        _table(
            [
                "k",
                "Recall",
                "nDCG",
                "MRR",
                "Precision",
                "Recall vs ceiling",
                "Wrong section",
                "Crowding",
                "Table hit rate",
                "Evidence tokens",
            ],
            [
                [
                    str(row.k),
                    _num(row.recall),
                    _num(row.ndcg),
                    _num(row.mrr),
                    _num(row.precision),
                    _num(row.recall_ceiling),
                    _num(row.wrong_section_share),
                    _num(row.crowding, 2),
                    _num(row.table_hit_rate),
                    _num(row.evidence_tokens, 0),
                ]
                for row in summary.by_k
            ],
        ),
        "",
        *_hybrid_lines(summary),
        *_reranker_lines(summary),
        "## By question type (k=5)",
        "",
        "The n-gram overlap is the longest run of words a question shares with its "
        "source chunks (U3 probe: a paraphrased question should have a low overlap).",
        "",
        _table(
            [
                "Type",
                "n",
                "Recall@5",
                "nDCG@5",
                "MRR@5",
                "Precision@5",
                "Recall@5 vs ceiling",
                "Empty rate",
                "Mean overlap",
                "Max overlap",
            ],
            [
                [
                    row.type.value,
                    str(row.n),
                    _num(row.recall),
                    _num(row.ndcg),
                    _num(row.mrr),
                    _num(row.precision),
                    _num(row.recall_ceiling),
                    _num(row.empty_rate, 2),
                    _num(row.mean_overlap, 1),
                    NOT_AVAILABLE if row.max_overlap is None else str(row.max_overlap),
                ]
                for row in summary.by_type
            ],
        ),
        "",
        "## Thresholded results",
        "",
        _table(
            ["Measure", "Value"],
            [
                [
                    "On-corpus empty-result rate (lower is better)",
                    _num(summary.empty_rate_on_corpus, 2),
                ],
                [
                    "Off-corpus no-result rate (higher is better)",
                    _num(summary.no_result_rate_off_corpus, 2),
                ],
                ["Max off-corpus top-1 score", _num(summary.max_off_corpus_top1)],
            ],
        ),
        "",
    ]
    small = [row for row in summary.by_type if row.n < FEW_QUESTIONS]
    if small:
        names = ", ".join(f"{row.type.value} (n={row.n})" for row in small)
        out.append(f"Few questions: {names}. Treat these rows as rough.")
        out.append("")
    return "\n".join(out)


def _type_row(item: TypeComparison) -> list[str]:
    return [
        item.type.value,
        str(item.n),
        _num(item.mean_a),
        _num(item.mean_b),
        _signed(item.mean_delta),
        f"{item.wins} / {item.losses} / {item.ties}",
        "yes" if item.clearly_worse else "no",
    ]


def render_comparison(comparison: Comparison, *, threshold_rule: bool = False) -> str:
    """Comparison of B against A with the decision. ``threshold_rule`` adds the
    off-corpus no-result condition (use it for the same run at two thresholds)."""
    result: Verdict = (
        threshold_verdict(comparison) if threshold_rule else verdict(comparison)
    )
    metric = f"{comparison.metric}@{comparison.cutoff_a.k}"
    out = [
        _header(
            f"Comparison: {comparison.b.name} against {comparison.a.name}",
            [
                _run_block("A (baseline side)", comparison.a),
                _run_block("B (candidate side)", comparison.b),
                f"- Metric: {metric} on {comparison.n} on-corpus questions; "
                f"A at {_cutoff_text(comparison.cutoff_a)}, "
                f"B at {_cutoff_text(comparison.cutoff_b)}",
                f"- Bootstrap: {comparison.resamples} resamples within each type, "
                f"numpy default_rng({comparison.seed}), 95 % percentile range",
            ],
        ),
        "## Decision",
        "",
        f"**{'B replaces A' if result.adopt else 'A stays'}**: {result.reason}.",
        "",
        _table(
            [
                "Mean A",
                "Mean B",
                "Mean delta (B - A)",
                "95 % range",
                "Wins / losses / ties",
            ],
            [
                [
                    _num(comparison.mean_a),
                    _num(comparison.mean_b),
                    _signed(comparison.mean_delta),
                    f"[{comparison.lo:+.4f}, {comparison.hi:+.4f}]",
                    f"{comparison.wins} / {comparison.losses} / {comparison.ties}",
                ]
            ],
        ),
        "",
        "Rule: B replaces A only if the lower end of the range is above 0 and no type "
        "is clearly worse (mean delta <= -0.05 and more losses than wins). A tie is a "
        "per-question difference below 1e-9.",
        "",
        "## By question type",
        "",
        _table(
            [
                "Type",
                "n",
                "Mean A",
                "Mean B",
                "Mean delta",
                "W / L / T",
                "Clearly worse",
            ],
            [_type_row(item) for item in comparison.per_type],
        ),
        "",
        "## Off-corpus no-result rate",
        "",
        _table(
            ["A", "B", "Off-corpus questions"],
            [
                [
                    _num(comparison.no_result_rate_a, 2),
                    _num(comparison.no_result_rate_b, 2),
                    str(comparison.n_off_corpus),
                ]
            ],
        ),
        "",
    ]
    if threshold_rule:
        out.append(
            "Threshold rule: a lower threshold also needs an off-corpus no-result rate "
            "that does not drop."
        )
        out.append("")
    notes: list[str] = []
    small = [item for item in comparison.per_type if item.n < FEW_QUESTIONS]
    if small:
        names = ", ".join(f"{item.type.value} (n={item.n})" for item in small)
        notes.append(f"Few questions in: {names}. Per-type deltas are rough.")
    if 0 < comparison.n_off_corpus < FEW_QUESTIONS:
        notes.append(
            f"Only {comparison.n_off_corpus} off-corpus questions, so the no-result "
            f"rate moves in steps of {1 / comparison.n_off_corpus:.2f}. "
            "The threshold part is coarse."
        )
    out.extend(notes)
    if notes:
        out.append("")
    return "\n".join(out)


def render_threshold_curve(curve: ThresholdCurve) -> str:
    """Threshold curve: one row per threshold (on-corpus quality, off-corpus rates)."""
    out = [
        _header(
            f"Threshold curve: {curve.meta.name}",
            [
                _run_block("Run", curve.meta),
                f"- Cut-off k={curve.k}; {curve.n_on_corpus} on-corpus and "
                f"{curve.n_off_corpus} off-corpus questions",
            ],
        ),
        _table(
            [
                "t",
                f"Recall@{curve.k}",
                f"nDCG@{curve.k}",
                "On-corpus empty rate",
                "Off-corpus no-result rate",
                "Max off-corpus top-1",
            ],
            [
                [
                    f"{point.threshold:.2f}",
                    _num(point.recall),
                    _num(point.ndcg),
                    _num(point.empty_rate_on_corpus, 2),
                    _num(point.no_result_rate_off_corpus, 2),
                    _num(point.max_off_corpus_top1),
                ]
                for point in curve.points
            ],
        ),
        "",
        "A lower threshold replaces 0.60 only if it passes the nDCG@5 rule and the "
        "off-corpus no-result rate does not drop. Raising the threshold can only lose "
        "recall, so it is reported, not chosen.",
    ]
    if 0 < curve.n_off_corpus < FEW_QUESTIONS:
        step = 1 / curve.n_off_corpus
        out.append(
            f"Only {curve.n_off_corpus} off-corpus questions: the no-result rate "
            f"moves in steps of {step:.2f}, so the threshold part is coarse."
        )
    out.append("")
    return "\n".join(out)


def render_gate_curve(curve: GateCurve) -> str:
    """Refusal curve on the top-1 score: one row per threshold t."""

    def share(count: int, total: int) -> str:
        return f"{count} / {total}" if total else NOT_AVAILABLE

    out = [
        _header(
            f"Refusal curve: {curve.meta.name}",
            [
                _run_block("Run", curve.meta),
                f"- Score kind: {curve.meta.score_kind}; {curve.n_on_corpus} on-corpus "
                f"and {curve.n_off_corpus} off-corpus questions",
            ],
        ),
        "A question is refused at t when its top-1 score is below t, or when it has no "
        "hit at all (a reranked run drops hits below the cosine pre-filter first). "
        "Only the top-1 score is used. Off-corpus refused is good. On-corpus lost is "
        "the cost: questions the papers do cover that would be refused too.",
        "",
        _table(
            [
                "t",
                "Off-corpus refused",
                "Off-corpus refused rate",
                "On-corpus lost",
                "On-corpus lost rate",
            ],
            [
                [
                    f"{point.threshold:.2f}",
                    share(point.off_refused, point.off_total),
                    _num(point.off_refused_rate, 2),
                    share(point.on_lost, point.on_total),
                    _num(point.on_lost_rate, 2),
                ]
                for point in curve.points
            ],
        ),
        "",
        "The cut-off is chosen by the owner after seeing this curve. A cut-off picked "
        "on the same questions it is scored on looks better than it will be on new "
        "ones.",
    ]
    if 0 < curve.n_off_corpus < FEW_QUESTIONS:
        out.append(
            f"Only {curve.n_off_corpus} off-corpus questions: the refused rate "
            f"moves in steps of {1 / curve.n_off_corpus:.2f}."
        )
    out.append("")
    return "\n".join(out)

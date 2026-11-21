# T7: Comparisons and owner review (analysis part)

## Decisions taken (2026-11-19)

The owner decided. The rest of this page is the analysis as it was written before the decision.

1. **Model:** MiniLM `cross-encoder/ms-marco-MiniLM-L-6-v2` (revision `233902d`). Time limit: under 2 s of added time per request (median 1.66 s, 90th percentile 1.71 s at depth 20).
2. **Depth:** 20 (retrieve fetches 20, rerank keeps 5). Depth 40 is not adopted: +0.040 mean, 95 % range [-0.003, +0.099], so the rule does not call it a real gain. Possible later follow-up.
3. **Query text:** the rewritten question (the reference variant; the original was within noise for MiniLM, +0.015 [-0.005, +0.036]).
4. **Refusal:** nothing adopted. No hard cut-off, no low-confidence label. The numbers stay in ADR-0026. Task T9 is skipped and ADR-0027 is not written. The better home is the real grounding check in Phase 11.

- Details: recorded in `docs/adr/0026-reranker.md` (status Accepted). Code consequence: unset `RERANKER_MODEL` now means MiniLM; empty or `none` still means off (`app/retrieval/rerank.py`, `.env.example`, `docker-compose.yml`, `tests/conftest.py`, `tests/test_rerank.py`). No startup warm-up was added (not decided).

**Status:** analysis done, owner review open. Nothing is decided here. Every number comes from a golden set that is LLM-labelled (Claude), not human-verified, with 45 on-corpus and 20 off-corpus questions. ADR-0026 is drafted as `Proposed`, with the Decision section left for you.

## What

All nine reranker setups beat today's dense default, and the rule says so for each of them. The three models are not equal: MiniLM and bge are indistinguishable, MedCPT is lower. The big difference between models is time, not quality: MiniLM needs about 2 s, bge and MedCPT about 9 to 10 s at depth 20. As a refusal signal for off-topic questions, no model works well.

- Details: reranker means are 0.639 to 0.725 nDCG@5 against 0.564 for the dense default (`prefix`). Gains are +0.075 to +0.161. The lower end of each 95 % range is above 0 in all nine comparisons and no question type is clearly worse. Nothing was run again: all comparisons read the 13 saved run files (golden hash `cc08395a87ca` in each, all comparisons accepted them). Reports are in `data/eval/reports/` (`compare-*.md`, `gate-*.md`).

## Summary table

A reranker is a second model that re-scores the top hits and keeps the best 5. Depth is how many hits it sees. "Original" means it reads the user's question, not the rewritten one. Time is the reranker call per question after a warm-up, on this CPU-only machine.

| Model | Depth | Query | nDCG@5 | Gain over `prefix` (95 % range) | Rule vs `prefix` | Median ms | 90th pct ms |
|---|---|---|---|---|---|---|---|
| dense `prefix` (reference) | - | - | 0.564 | - | - | - | - |
| MiniLM | 20 | rewritten | 0.684 | +0.120 [+0.046, +0.199] | replaces | 1656 | 1713 |
| MiniLM | 20 | original | 0.699 | +0.135 [+0.059, +0.216] | replaces | 1615 | 1704 |
| MiniLM | 40 | rewritten | 0.725 | +0.161 [+0.078, +0.252] | replaces | 3117 | 3438 |
| bge | 20 | rewritten | 0.704 | +0.140 [+0.056, +0.225] | replaces | 9795 | 11787 |
| bge | 20 | original | 0.685 | +0.121 [+0.048, +0.203] | replaces | 10572 | 14240 |
| bge | 40 | rewritten | 0.718 | +0.154 [+0.068, +0.252] | replaces | 23194 | 27223 |
| MedCPT | 20 | rewritten | 0.639 | +0.075 [+0.009, +0.155] | replaces | 8884 | 13055 |
| MedCPT | 20 | original | 0.645 | +0.081 [+0.020, +0.151] | replaces | 9755 | 10160 |
| MedCPT | 40 | rewritten | 0.688 | +0.124 [+0.040, +0.220] | replaces | 17862 | 18859 |

- Details: "replaces" is the rule's wording "B replaces A": lower end of the range above 0 and no type clearly worse. Rule and bootstrap as in ADR-0023 and ADR-0025 (1000 resamples within each type, 95 % range, nDCG@5 at k = 5, reranked runs cut at 0, dense at 0.6). Wins / losses / ties against `prefix`: MiniLM 23/7/15, 21/7/17, 25/7/13; bge 22/9/14, 18/10/17, 24/10/11; MedCPT 17/12/16, 18/10/17, 20/12/13. MedCPT at depth 20 rewritten has the thinnest margin (lower end +0.009).

## Findings

### 1. Reranking helps, in every question type

The top 5 gets noticeably better for all three models. The gain is not limited to one kind of question.

- Details: per type, gain over `prefix` ranges over the nine runs: factual +0.03 to +0.22, paraphrased +0.04 to +0.16, cross_paper +0.07 to +0.16, table +0.10 to +0.30, multi_section +0.04 to +0.13. Type groups have 8 to 10 questions, so single type numbers are rough. `cross_paper` stays the weakest type (0.38 to 0.47 nDCG@5 after reranking against 0.31).

### 2. The query text and the depth do not make a difference the rule can call real

Using the original question instead of the rewrite, or showing the reranker 40 hits instead of 20, moves the mean a little, but the 95 % range includes 0 every time. Depth 40 is positive for all three models and just misses for two of them.

- Details: variant against the same model at depth 20 rewritten (A stays in all six): MiniLM original +0.015 [-0.005, +0.036], MiniLM depth 40 +0.040 [-0.003, +0.099]; bge original -0.019 [-0.042, +0.004] (paraphrased clearly worse, -0.069 with 3 wins and 4 losses), bge depth 40 +0.014 [-0.021, +0.062]; MedCPT original +0.006 [-0.025, +0.039], MedCPT depth 40 +0.049 [-0.0005, +0.116]. Depth 40 makes the reranker 1.9 to 2.4 times slower. The best any reranker can reach is the share of labelled sections among the hits it sees: 0.828 at depth 20 and 0.914 at depth 40, the same for all three models.

### 3. MiniLM and bge cannot be told apart. MedCPT is lower.

The data gives no reason to prefer bge over MiniLM on quality. MedCPT, the medical model, scores lower than both, mostly on factual and table questions.

- Details: model against model at depth 20 (A stays in all three because the range includes 0 or B is lower): MiniLM vs bge +0.019 [-0.025, +0.076], 13 wins / 13 losses; MiniLM vs MedCPT -0.045 [-0.090, -0.004] (clearly worse on factual -0.074 and table -0.080); bge vs MedCPT -0.064 [-0.123, -0.014] (clearly worse on factual -0.174 and table -0.075). At depth 40: MiniLM vs bge -0.007 [-0.046, +0.033], MiniLM vs MedCPT -0.037 [-0.094, +0.018], bge vs MedCPT -0.030 [-0.077, +0.013]; for the MedCPT pairs the rule marks factual and cross_paper or table as clearly worse. Also: MiniLM at depth 40 minus bge at depth 20: +0.021 [-0.022, +0.065].

### 4. The models differ most in time

MiniLM answers in about 1.7 s at depth 20. bge and MedCPT need roughly 9 to 10 s. With the current local model a full request takes about 7 s without any reranking, so these are large additions.

- Details: the 7 s is the sum of the Sept 2026 benchmark averages for `qwen3:0.6b` without thinking (rewrite 2.2 s, generate 4.8 s, `docs/benchmarks.md`). It leaves out search and embedding, and that benchmark used shorter prompts than today's roughly 2400 evidence tokens, so a real request is probably slower. I found no end-to-end measurement in the phase documents. Added median time as a share of 7 s: MiniLM +23 % to +45 %, bge +140 % to +331 %, MedCPT +127 % to +255 %. Load time 2.3 to 7.3 s. Real weight sizes: MiniLM 91 MB, bge 1.1 GB, MedCPT 438 MB (the recorded 890 MB and 877 MB come from duplicate folders and a second revision in the cache). Timing is one timed run per setup, no noise measurement. Memory use with Ollama loaded at the same time was not measured.

Gain per second of median added time (nDCG@5 gain over `prefix` divided by median seconds, plain numbers):

| Model | Depth 20 rewritten | Depth 20 original | Depth 40 rewritten |
|---|---|---|---|
| MiniLM | 0.073 | 0.084 | 0.052 |
| bge | 0.014 | 0.011 | 0.007 |
| MedCPT | 0.008 | 0.008 | 0.007 |

### 5. A score cut-off refuses some off-topic questions, but not cleanly

If the app refused every question whose best reranker score is below a cut-off, it would refuse some of the 20 off-topic test questions and also some of the 45 on-topic ones. No point on any curve refuses most off-topic questions at a small price.

<svg class="gate-plot" viewBox="0 0 640 470" role="img" aria-labelledby="gp-t gp-d" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:760px;height:auto;display:block">
<title id="gp-t">Refusal curve by reranker</title><desc id="gp-d">Share of off-topic questions refused against share of on-topic questions wrongly refused, as the score cut-off rises from 0 to 1. A curve in the top-left corner would be ideal; the dashed diagonal is a coin flip.</desc>
<style>.gate-plot{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .gate-plot{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}:root[data-theme=dark] .gate-plot{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70}.gate-plot text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:12px;fill:var(--ink2)}.gate-plot .tick{fill:var(--muted);font-size:11px}.gate-plot .ttl{fill:var(--ink);font-size:13px;font-weight:600}.gate-plot .hit{fill:transparent;stroke:none}.gate-plot .hit:hover{fill:var(--ink);fill-opacity:.15}</style>
<rect x="0" y="0" width="640" height="470" fill="var(--surface)"/>
<line x1="64" x2="616" y1="396.0" y2="396.0" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="400.0" text-anchor="end">0%</text>
<text class="tick" x="64.0" y="414" text-anchor="middle">0%</text>
<line x1="64" x2="616" y1="320.8" y2="320.8" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="324.8" text-anchor="end">20%</text>
<text class="tick" x="174.4" y="414" text-anchor="middle">20%</text>
<line x1="64" x2="616" y1="245.6" y2="245.6" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="249.6" text-anchor="end">40%</text>
<text class="tick" x="284.8" y="414" text-anchor="middle">40%</text>
<line x1="64" x2="616" y1="170.4" y2="170.4" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="174.4" text-anchor="end">60%</text>
<text class="tick" x="395.2" y="414" text-anchor="middle">60%</text>
<line x1="64" x2="616" y1="95.2" y2="95.2" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="99.2" text-anchor="end">80%</text>
<text class="tick" x="505.6" y="414" text-anchor="middle">80%</text>
<line x1="64" x2="616" y1="20.0" y2="20.0" stroke="var(--grid)" stroke-width="1"/>
<text class="tick" x="56" y="24.0" text-anchor="end">100%</text>
<text class="tick" x="616.0" y="414" text-anchor="middle">100%</text>
<line x1="64" x2="616" y1="396" y2="396" stroke="var(--axis)" stroke-width="1"/>
<line x1="64" x2="64" y1="20" y2="396" stroke="var(--axis)" stroke-width="1"/>
<text x="340" y="440" text-anchor="middle">Cost: on-topic questions wrongly refused (of 45)</text>
<text transform="translate(16 208) rotate(-90)" text-anchor="middle">Benefit: off-topic questions refused (of 20)</text>
<line x1="64.0" y1="396.0" x2="616.0" y2="20.0" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 4"/>
<text class="tick" x="494.6" y="118.7" >coin flip</text>
<text class="tick" x="75.0" y="32">ideal: top-left corner</text>
<path d="M64.0 396.0 L64.0 377.2 L64.0 358.4 L64.0 339.6 L76.3 339.6 L76.3 320.8 L76.3 302.0 L88.5 302.0 L100.8 302.0 L113.1 302.0 L125.3 302.0 L125.3 283.2 L125.3 264.4 L125.3 245.6 L125.3 226.8 L125.3 208.0 L125.3 189.2 L125.3 170.4 L137.6 170.4 L149.9 170.4 L162.1 170.4 L162.1 151.6 L174.4 151.6 L174.4 132.8 L186.7 132.8 L186.7 114.0 L198.9 114.0 L211.2 114.0 L223.5 114.0 L235.7 114.0 L248.0 114.0 L248.0 95.2 L260.3 95.2 L260.3 76.4 L272.5 76.4 L284.8 76.4 L297.1 76.4 L309.3 76.4 L309.3 57.6 L321.6 57.6 L333.9 57.6 L346.1 57.6 L358.4 57.6 L358.4 38.8 L370.7 38.8 L382.9 38.8 L395.2 38.8 L407.5 38.8 L419.7 38.8 L432.0 38.8 L444.3 38.8 L456.5 38.8 L468.8 38.8 L481.1 38.8 L493.3 38.8 L505.6 38.8 L517.9 38.8 L530.1 38.8 L530.1 20.0 L542.4 20.0 L554.7 20.0 L566.9 20.0 L579.2 20.0 L591.5 20.0 L603.7 20.0 L616.0 20.0 L616.0 20.0" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>
<circle class="hit" cx="64.0" cy="396.0" r="6"><title>MiniLM, depth 20. Cut-off 0.0000: refuses 0 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="377.2" r="6"><title>MiniLM, depth 20. Cut-off 0.0017: refuses 1 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="358.4" r="6"><title>MiniLM, depth 20. Cut-off 0.01: refuses 2 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="339.6" r="6"><title>MiniLM, depth 20. Cut-off 0.01: refuses 3 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="339.6" r="6"><title>MiniLM, depth 20. Cut-off 0.03: refuses 3 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="320.8" r="6"><title>MiniLM, depth 20. Cut-off 0.06: refuses 4 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="302.0" r="6"><title>MiniLM, depth 20. Cut-off 0.08: refuses 5 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="302.0" r="6"><title>MiniLM, depth 20. Cut-off 0.08: refuses 5 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="100.8" cy="302.0" r="6"><title>MiniLM, depth 20. Cut-off 0.15: refuses 5 of 20 off-topic, loses 3 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="302.0" r="6"><title>MiniLM, depth 20. Cut-off 0.16: refuses 5 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="302.0" r="6"><title>MiniLM, depth 20. Cut-off 0.28: refuses 5 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="283.2" r="6"><title>MiniLM, depth 20. Cut-off 0.31: refuses 6 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="264.4" r="6"><title>MiniLM, depth 20. Cut-off 0.37: refuses 7 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="245.6" r="6"><title>MiniLM, depth 20. Cut-off 0.44: refuses 8 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="226.8" r="6"><title>MiniLM, depth 20. Cut-off 0.49: refuses 9 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="208.0" r="6"><title>MiniLM, depth 20. Cut-off 0.64: refuses 10 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="189.2" r="6"><title>MiniLM, depth 20. Cut-off 0.65: refuses 11 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="170.4" r="6"><title>MiniLM, depth 20. Cut-off 0.71: refuses 12 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="170.4" r="6"><title>MiniLM, depth 20. Cut-off 0.73: refuses 12 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="149.9" cy="170.4" r="6"><title>MiniLM, depth 20. Cut-off 0.73: refuses 12 of 20 off-topic, loses 7 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="170.4" r="6"><title>MiniLM, depth 20. Cut-off 0.74: refuses 12 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="151.6" r="6"><title>MiniLM, depth 20. Cut-off 0.76: refuses 13 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="151.6" r="6"><title>MiniLM, depth 20. Cut-off 0.77: refuses 13 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="132.8" r="6"><title>MiniLM, depth 20. Cut-off 0.79: refuses 14 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="132.8" r="6"><title>MiniLM, depth 20. Cut-off 0.82: refuses 14 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.82: refuses 15 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.84: refuses 15 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="211.2" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.84: refuses 15 of 20 off-topic, loses 12 of 45 on-topic.</title></circle>
<circle class="hit" cx="223.5" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.84: refuses 15 of 20 off-topic, loses 13 of 45 on-topic.</title></circle>
<circle class="hit" cx="235.7" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.86: refuses 15 of 20 off-topic, loses 14 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="114.0" r="6"><title>MiniLM, depth 20. Cut-off 0.86: refuses 15 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="95.2" r="6"><title>MiniLM, depth 20. Cut-off 0.88: refuses 16 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="260.3" cy="95.2" r="6"><title>MiniLM, depth 20. Cut-off 0.88: refuses 16 of 20 off-topic, loses 16 of 45 on-topic.</title></circle>
<circle class="hit" cx="260.3" cy="76.4" r="6"><title>MiniLM, depth 20. Cut-off 0.93: refuses 17 of 20 off-topic, loses 16 of 45 on-topic.</title></circle>
<circle class="hit" cx="272.5" cy="76.4" r="6"><title>MiniLM, depth 20. Cut-off 0.93: refuses 17 of 20 off-topic, loses 17 of 45 on-topic.</title></circle>
<circle class="hit" cx="284.8" cy="76.4" r="6"><title>MiniLM, depth 20. Cut-off 0.93: refuses 17 of 20 off-topic, loses 18 of 45 on-topic.</title></circle>
<circle class="hit" cx="297.1" cy="76.4" r="6"><title>MiniLM, depth 20. Cut-off 0.95: refuses 17 of 20 off-topic, loses 19 of 45 on-topic.</title></circle>
<circle class="hit" cx="309.3" cy="76.4" r="6"><title>MiniLM, depth 20. Cut-off 0.95: refuses 17 of 20 off-topic, loses 20 of 45 on-topic.</title></circle>
<circle class="hit" cx="309.3" cy="57.6" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 18 of 20 off-topic, loses 20 of 45 on-topic.</title></circle>
<circle class="hit" cx="321.6" cy="57.6" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 18 of 20 off-topic, loses 21 of 45 on-topic.</title></circle>
<circle class="hit" cx="333.9" cy="57.6" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 18 of 20 off-topic, loses 22 of 45 on-topic.</title></circle>
<circle class="hit" cx="346.1" cy="57.6" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 18 of 20 off-topic, loses 23 of 45 on-topic.</title></circle>
<circle class="hit" cx="358.4" cy="57.6" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 18 of 20 off-topic, loses 24 of 45 on-topic.</title></circle>
<circle class="hit" cx="358.4" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.96: refuses 19 of 20 off-topic, loses 24 of 45 on-topic.</title></circle>
<circle class="hit" cx="370.7" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 25 of 45 on-topic.</title></circle>
<circle class="hit" cx="382.9" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 26 of 45 on-topic.</title></circle>
<circle class="hit" cx="395.2" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 27 of 45 on-topic.</title></circle>
<circle class="hit" cx="407.5" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 28 of 45 on-topic.</title></circle>
<circle class="hit" cx="419.7" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 29 of 45 on-topic.</title></circle>
<circle class="hit" cx="432.0" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 30 of 45 on-topic.</title></circle>
<circle class="hit" cx="444.3" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 31 of 45 on-topic.</title></circle>
<circle class="hit" cx="456.5" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 32 of 45 on-topic.</title></circle>
<circle class="hit" cx="468.8" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 33 of 45 on-topic.</title></circle>
<circle class="hit" cx="481.1" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 34 of 45 on-topic.</title></circle>
<circle class="hit" cx="493.3" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 35 of 45 on-topic.</title></circle>
<circle class="hit" cx="505.6" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 36 of 45 on-topic.</title></circle>
<circle class="hit" cx="517.9" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.9917: refuses 19 of 20 off-topic, loses 37 of 45 on-topic.</title></circle>
<circle class="hit" cx="530.1" cy="38.8" r="6"><title>MiniLM, depth 20. Cut-off 0.9919: refuses 19 of 20 off-topic, loses 38 of 45 on-topic.</title></circle>
<circle class="hit" cx="530.1" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9948: refuses 20 of 20 off-topic, loses 38 of 45 on-topic.</title></circle>
<circle class="hit" cx="542.4" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9959: refuses 20 of 20 off-topic, loses 39 of 45 on-topic.</title></circle>
<circle class="hit" cx="554.7" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9962: refuses 20 of 20 off-topic, loses 40 of 45 on-topic.</title></circle>
<circle class="hit" cx="566.9" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9969: refuses 20 of 20 off-topic, loses 41 of 45 on-topic.</title></circle>
<circle class="hit" cx="579.2" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9983: refuses 20 of 20 off-topic, loses 42 of 45 on-topic.</title></circle>
<circle class="hit" cx="591.5" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9988: refuses 20 of 20 off-topic, loses 43 of 45 on-topic.</title></circle>
<circle class="hit" cx="603.7" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9990: refuses 20 of 20 off-topic, loses 44 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 0.9991: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MiniLM, depth 20. Cut-off 1: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<path d="M64.0 396.0 L64.0 377.2 L64.0 358.4 L64.0 339.6 L64.0 320.8 L64.0 302.0 L76.3 302.0 L88.5 302.0 L100.8 302.0 L113.1 302.0 L113.1 283.2 L113.1 264.4 L113.1 245.6 L125.3 245.6 L125.3 226.8 L125.3 208.0 L137.6 208.0 L149.9 208.0 L149.9 189.2 L162.1 189.2 L162.1 170.4 L174.4 170.4 L174.4 151.6 L186.7 151.6 L186.7 132.8 L198.9 132.8 L211.2 132.8 L223.5 132.8 L235.7 132.8 L235.7 114.0 L248.0 114.0 L248.0 95.2 L248.0 76.4 L260.3 76.4 L272.5 76.4 L284.8 76.4 L297.1 76.4 L297.1 57.6 L309.3 57.6 L321.6 57.6 L333.9 57.6 L346.1 57.6 L346.1 38.8 L358.4 38.8 L370.7 38.8 L382.9 38.8 L395.2 38.8 L407.5 38.8 L419.7 38.8 L432.0 38.8 L444.3 38.8 L456.5 38.8 L468.8 38.8 L481.1 38.8 L493.3 38.8 L505.6 38.8 L517.9 38.8 L530.1 38.8 L530.1 20.0 L542.4 20.0 L554.7 20.0 L566.9 20.0 L579.2 20.0 L591.5 20.0 L603.7 20.0 L616.0 20.0 L616.0 20.0" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" stroke-dasharray="6 4"/>
<circle class="hit" cx="64.0" cy="396.0" r="6"><title>MiniLM, depth 40. Cut-off 0.0000: refuses 0 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="377.2" r="6"><title>MiniLM, depth 40. Cut-off 0.0017: refuses 1 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="358.4" r="6"><title>MiniLM, depth 40. Cut-off 0.01: refuses 2 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="339.6" r="6"><title>MiniLM, depth 40. Cut-off 0.04: refuses 3 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="320.8" r="6"><title>MiniLM, depth 40. Cut-off 0.06: refuses 4 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="302.0" r="6"><title>MiniLM, depth 40. Cut-off 0.08: refuses 5 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="302.0" r="6"><title>MiniLM, depth 40. Cut-off 0.08: refuses 5 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="302.0" r="6"><title>MiniLM, depth 40. Cut-off 0.15: refuses 5 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="100.8" cy="302.0" r="6"><title>MiniLM, depth 40. Cut-off 0.16: refuses 5 of 20 off-topic, loses 3 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="302.0" r="6"><title>MiniLM, depth 40. Cut-off 0.28: refuses 5 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="283.2" r="6"><title>MiniLM, depth 40. Cut-off 0.37: refuses 6 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="264.4" r="6"><title>MiniLM, depth 40. Cut-off 0.49: refuses 7 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="245.6" r="6"><title>MiniLM, depth 40. Cut-off 0.53: refuses 8 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="245.6" r="6"><title>MiniLM, depth 40. Cut-off 0.54: refuses 8 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="226.8" r="6"><title>MiniLM, depth 40. Cut-off 0.64: refuses 9 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="208.0" r="6"><title>MiniLM, depth 40. Cut-off 0.71: refuses 10 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="208.0" r="6"><title>MiniLM, depth 40. Cut-off 0.73: refuses 10 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="149.9" cy="208.0" r="6"><title>MiniLM, depth 40. Cut-off 0.73: refuses 10 of 20 off-topic, loses 7 of 45 on-topic.</title></circle>
<circle class="hit" cx="149.9" cy="189.2" r="6"><title>MiniLM, depth 40. Cut-off 0.74: refuses 11 of 20 off-topic, loses 7 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="189.2" r="6"><title>MiniLM, depth 40. Cut-off 0.74: refuses 11 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="170.4" r="6"><title>MiniLM, depth 40. Cut-off 0.76: refuses 12 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="170.4" r="6"><title>MiniLM, depth 40. Cut-off 0.77: refuses 12 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="151.6" r="6"><title>MiniLM, depth 40. Cut-off 0.81: refuses 13 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="151.6" r="6"><title>MiniLM, depth 40. Cut-off 0.82: refuses 13 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="132.8" r="6"><title>MiniLM, depth 40. Cut-off 0.82: refuses 14 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="132.8" r="6"><title>MiniLM, depth 40. Cut-off 0.84: refuses 14 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="211.2" cy="132.8" r="6"><title>MiniLM, depth 40. Cut-off 0.84: refuses 14 of 20 off-topic, loses 12 of 45 on-topic.</title></circle>
<circle class="hit" cx="223.5" cy="132.8" r="6"><title>MiniLM, depth 40. Cut-off 0.86: refuses 14 of 20 off-topic, loses 13 of 45 on-topic.</title></circle>
<circle class="hit" cx="235.7" cy="132.8" r="6"><title>MiniLM, depth 40. Cut-off 0.86: refuses 14 of 20 off-topic, loses 14 of 45 on-topic.</title></circle>
<circle class="hit" cx="235.7" cy="114.0" r="6"><title>MiniLM, depth 40. Cut-off 0.88: refuses 15 of 20 off-topic, loses 14 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="114.0" r="6"><title>MiniLM, depth 40. Cut-off 0.88: refuses 15 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="95.2" r="6"><title>MiniLM, depth 40. Cut-off 0.92: refuses 16 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="76.4" r="6"><title>MiniLM, depth 40. Cut-off 0.93: refuses 17 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="260.3" cy="76.4" r="6"><title>MiniLM, depth 40. Cut-off 0.93: refuses 17 of 20 off-topic, loses 16 of 45 on-topic.</title></circle>
<circle class="hit" cx="272.5" cy="76.4" r="6"><title>MiniLM, depth 40. Cut-off 0.93: refuses 17 of 20 off-topic, loses 17 of 45 on-topic.</title></circle>
<circle class="hit" cx="284.8" cy="76.4" r="6"><title>MiniLM, depth 40. Cut-off 0.93: refuses 17 of 20 off-topic, loses 18 of 45 on-topic.</title></circle>
<circle class="hit" cx="297.1" cy="76.4" r="6"><title>MiniLM, depth 40. Cut-off 0.95: refuses 17 of 20 off-topic, loses 19 of 45 on-topic.</title></circle>
<circle class="hit" cx="297.1" cy="57.6" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 18 of 20 off-topic, loses 19 of 45 on-topic.</title></circle>
<circle class="hit" cx="309.3" cy="57.6" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 18 of 20 off-topic, loses 20 of 45 on-topic.</title></circle>
<circle class="hit" cx="321.6" cy="57.6" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 18 of 20 off-topic, loses 21 of 45 on-topic.</title></circle>
<circle class="hit" cx="333.9" cy="57.6" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 18 of 20 off-topic, loses 22 of 45 on-topic.</title></circle>
<circle class="hit" cx="346.1" cy="57.6" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 18 of 20 off-topic, loses 23 of 45 on-topic.</title></circle>
<circle class="hit" cx="346.1" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.96: refuses 19 of 20 off-topic, loses 23 of 45 on-topic.</title></circle>
<circle class="hit" cx="358.4" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.97: refuses 19 of 20 off-topic, loses 24 of 45 on-topic.</title></circle>
<circle class="hit" cx="370.7" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.97: refuses 19 of 20 off-topic, loses 25 of 45 on-topic.</title></circle>
<circle class="hit" cx="382.9" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.97: refuses 19 of 20 off-topic, loses 26 of 45 on-topic.</title></circle>
<circle class="hit" cx="395.2" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 27 of 45 on-topic.</title></circle>
<circle class="hit" cx="407.5" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 28 of 45 on-topic.</title></circle>
<circle class="hit" cx="419.7" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 29 of 45 on-topic.</title></circle>
<circle class="hit" cx="432.0" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 30 of 45 on-topic.</title></circle>
<circle class="hit" cx="444.3" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 31 of 45 on-topic.</title></circle>
<circle class="hit" cx="456.5" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 32 of 45 on-topic.</title></circle>
<circle class="hit" cx="468.8" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 33 of 45 on-topic.</title></circle>
<circle class="hit" cx="481.1" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 34 of 45 on-topic.</title></circle>
<circle class="hit" cx="493.3" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 35 of 45 on-topic.</title></circle>
<circle class="hit" cx="505.6" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.98: refuses 19 of 20 off-topic, loses 36 of 45 on-topic.</title></circle>
<circle class="hit" cx="517.9" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.9917: refuses 19 of 20 off-topic, loses 37 of 45 on-topic.</title></circle>
<circle class="hit" cx="530.1" cy="38.8" r="6"><title>MiniLM, depth 40. Cut-off 0.9919: refuses 19 of 20 off-topic, loses 38 of 45 on-topic.</title></circle>
<circle class="hit" cx="530.1" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9948: refuses 20 of 20 off-topic, loses 38 of 45 on-topic.</title></circle>
<circle class="hit" cx="542.4" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9959: refuses 20 of 20 off-topic, loses 39 of 45 on-topic.</title></circle>
<circle class="hit" cx="554.7" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9962: refuses 20 of 20 off-topic, loses 40 of 45 on-topic.</title></circle>
<circle class="hit" cx="566.9" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9969: refuses 20 of 20 off-topic, loses 41 of 45 on-topic.</title></circle>
<circle class="hit" cx="579.2" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9983: refuses 20 of 20 off-topic, loses 42 of 45 on-topic.</title></circle>
<circle class="hit" cx="591.5" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9988: refuses 20 of 20 off-topic, loses 43 of 45 on-topic.</title></circle>
<circle class="hit" cx="603.7" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9990: refuses 20 of 20 off-topic, loses 44 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 0.9991: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MiniLM, depth 40. Cut-off 1: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<path d="M64.0 396.0 L64.0 377.2 L64.0 358.4 L76.3 358.4 L76.3 339.6 L88.5 339.6 L88.5 320.8 L88.5 302.0 L88.5 283.2 L88.5 264.4 L100.8 264.4 L113.1 264.4 L125.3 264.4 L125.3 245.6 L125.3 226.8 L137.6 226.8 L137.6 208.0 L137.6 189.2 L137.6 170.4 L137.6 151.6 L149.9 151.6 L162.1 151.6 L162.1 132.8 L174.4 132.8 L186.7 132.8 L198.9 132.8 L198.9 114.0 L198.9 95.2 L211.2 95.2 L211.2 76.4 L223.5 76.4 L235.7 76.4 L248.0 76.4 L248.0 57.6 L260.3 57.6 L272.5 57.6 L272.5 38.8 L284.8 38.8 L297.1 38.8 L309.3 38.8 L321.6 38.8 L333.9 38.8 L346.1 38.8 L358.4 38.8 L370.7 38.8 L382.9 38.8 L395.2 38.8 L407.5 38.8 L419.7 38.8 L419.7 20.0 L432.0 20.0 L444.3 20.0 L456.5 20.0 L468.8 20.0 L481.1 20.0 L493.3 20.0 L505.6 20.0 L517.9 20.0 L530.1 20.0 L542.4 20.0 L554.7 20.0 L566.9 20.0 L579.2 20.0 L591.5 20.0 L603.7 20.0 L616.0 20.0 L616.0 20.0" fill="none" stroke="var(--s2)" stroke-width="2" stroke-linejoin="round"/>
<circle class="hit" cx="64.0" cy="396.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.0000: refuses 0 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="377.2" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.0024: refuses 1 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="358.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.0034: refuses 2 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="358.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.01: refuses 2 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="339.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.01: refuses 3 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="339.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.02: refuses 3 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="320.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.02: refuses 4 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="302.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.02: refuses 5 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="283.2" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.03: refuses 6 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="264.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.06: refuses 7 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="100.8" cy="264.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.06: refuses 7 of 20 off-topic, loses 3 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="264.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.08: refuses 7 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="264.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.09: refuses 7 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="245.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.10: refuses 8 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="226.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.14: refuses 9 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="226.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.16: refuses 9 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="208.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.17: refuses 10 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="189.2" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.18: refuses 11 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="170.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.19: refuses 12 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="151.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.24: refuses 13 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="149.9" cy="151.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.30: refuses 13 of 20 off-topic, loses 7 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="151.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.31: refuses 13 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="132.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.33: refuses 14 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="132.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.34: refuses 14 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="132.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.36: refuses 14 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="132.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.37: refuses 14 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="114.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.38: refuses 15 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="95.2" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.47: refuses 16 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="211.2" cy="95.2" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.57: refuses 16 of 20 off-topic, loses 12 of 45 on-topic.</title></circle>
<circle class="hit" cx="211.2" cy="76.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.58: refuses 17 of 20 off-topic, loses 12 of 45 on-topic.</title></circle>
<circle class="hit" cx="223.5" cy="76.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.67: refuses 17 of 20 off-topic, loses 13 of 45 on-topic.</title></circle>
<circle class="hit" cx="235.7" cy="76.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.69: refuses 17 of 20 off-topic, loses 14 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="76.4" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.70: refuses 17 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="57.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.73: refuses 18 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="260.3" cy="57.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.81: refuses 18 of 20 off-topic, loses 16 of 45 on-topic.</title></circle>
<circle class="hit" cx="272.5" cy="57.6" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.86: refuses 18 of 20 off-topic, loses 17 of 45 on-topic.</title></circle>
<circle class="hit" cx="272.5" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.87: refuses 19 of 20 off-topic, loses 17 of 45 on-topic.</title></circle>
<circle class="hit" cx="284.8" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.88: refuses 19 of 20 off-topic, loses 18 of 45 on-topic.</title></circle>
<circle class="hit" cx="297.1" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.91: refuses 19 of 20 off-topic, loses 19 of 45 on-topic.</title></circle>
<circle class="hit" cx="309.3" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.93: refuses 19 of 20 off-topic, loses 20 of 45 on-topic.</title></circle>
<circle class="hit" cx="321.6" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.93: refuses 19 of 20 off-topic, loses 21 of 45 on-topic.</title></circle>
<circle class="hit" cx="333.9" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.96: refuses 19 of 20 off-topic, loses 22 of 45 on-topic.</title></circle>
<circle class="hit" cx="346.1" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 23 of 45 on-topic.</title></circle>
<circle class="hit" cx="358.4" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 24 of 45 on-topic.</title></circle>
<circle class="hit" cx="370.7" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.97: refuses 19 of 20 off-topic, loses 25 of 45 on-topic.</title></circle>
<circle class="hit" cx="382.9" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 26 of 45 on-topic.</title></circle>
<circle class="hit" cx="395.2" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 27 of 45 on-topic.</title></circle>
<circle class="hit" cx="407.5" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 28 of 45 on-topic.</title></circle>
<circle class="hit" cx="419.7" cy="38.8" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.98: refuses 19 of 20 off-topic, loses 29 of 45 on-topic.</title></circle>
<circle class="hit" cx="419.7" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.98: refuses 20 of 20 off-topic, loses 29 of 45 on-topic.</title></circle>
<circle class="hit" cx="432.0" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.99: refuses 20 of 20 off-topic, loses 30 of 45 on-topic.</title></circle>
<circle class="hit" cx="444.3" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.99: refuses 20 of 20 off-topic, loses 31 of 45 on-topic.</title></circle>
<circle class="hit" cx="456.5" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.99: refuses 20 of 20 off-topic, loses 32 of 45 on-topic.</title></circle>
<circle class="hit" cx="468.8" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.99: refuses 20 of 20 off-topic, loses 33 of 45 on-topic.</title></circle>
<circle class="hit" cx="481.1" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9905: refuses 20 of 20 off-topic, loses 34 of 45 on-topic.</title></circle>
<circle class="hit" cx="493.3" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9934: refuses 20 of 20 off-topic, loses 35 of 45 on-topic.</title></circle>
<circle class="hit" cx="505.6" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9940: refuses 20 of 20 off-topic, loses 36 of 45 on-topic.</title></circle>
<circle class="hit" cx="517.9" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9952: refuses 20 of 20 off-topic, loses 37 of 45 on-topic.</title></circle>
<circle class="hit" cx="530.1" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9956: refuses 20 of 20 off-topic, loses 38 of 45 on-topic.</title></circle>
<circle class="hit" cx="542.4" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9956: refuses 20 of 20 off-topic, loses 39 of 45 on-topic.</title></circle>
<circle class="hit" cx="554.7" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9973: refuses 20 of 20 off-topic, loses 40 of 45 on-topic.</title></circle>
<circle class="hit" cx="566.9" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9994: refuses 20 of 20 off-topic, loses 41 of 45 on-topic.</title></circle>
<circle class="hit" cx="579.2" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9997: refuses 20 of 20 off-topic, loses 42 of 45 on-topic.</title></circle>
<circle class="hit" cx="591.5" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9998: refuses 20 of 20 off-topic, loses 43 of 45 on-topic.</title></circle>
<circle class="hit" cx="603.7" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9998: refuses 20 of 20 off-topic, loses 44 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 0.9998: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>bge-reranker-base, depth 20. Cut-off 1: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<path d="M64.0 396.0 L64.0 377.2 L64.0 358.4 L64.0 339.6 L64.0 320.8 L64.0 302.0 L64.0 283.2 L64.0 264.4 L64.0 245.6 L64.0 226.8 L76.3 226.8 L76.3 208.0 L76.3 189.2 L88.5 189.2 L88.5 170.4 L100.8 170.4 L113.1 170.4 L125.3 170.4 L137.6 170.4 L149.9 170.4 L162.1 170.4 L174.4 170.4 L186.7 151.6 L186.7 132.8 L186.7 114.0 L186.7 95.2 L198.9 95.2 L223.5 95.2 L235.7 95.2 L248.0 76.4 L248.0 57.6 L284.8 57.6 L346.1 38.8 L456.5 20.0 L616.0 20.0 L616.0 20.0" fill="none" stroke="var(--s3)" stroke-width="2" stroke-linejoin="round"/>
<circle class="hit" cx="64.0" cy="396.0" r="6"><title>MedCPT, depth 20. Cut-off 0.0000: refuses 0 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="377.2" r="6"><title>MedCPT, depth 20. Cut-off 0.0005: refuses 1 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="358.4" r="6"><title>MedCPT, depth 20. Cut-off 0.0077: refuses 2 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="339.6" r="6"><title>MedCPT, depth 20. Cut-off 0.03: refuses 3 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="320.8" r="6"><title>MedCPT, depth 20. Cut-off 0.03: refuses 4 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="302.0" r="6"><title>MedCPT, depth 20. Cut-off 0.92: refuses 5 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="283.2" r="6"><title>MedCPT, depth 20. Cut-off 0.98: refuses 6 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="264.4" r="6"><title>MedCPT, depth 20. Cut-off 0.9965: refuses 7 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="245.6" r="6"><title>MedCPT, depth 20. Cut-off 0.9987: refuses 8 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="64.0" cy="226.8" r="6"><title>MedCPT, depth 20. Cut-off 0.9991: refuses 9 of 20 off-topic, loses 0 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="226.8" r="6"><title>MedCPT, depth 20. Cut-off 0.9995: refuses 9 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="208.0" r="6"><title>MedCPT, depth 20. Cut-off 0.9997: refuses 10 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="76.3" cy="189.2" r="6"><title>MedCPT, depth 20. Cut-off 0.9997: refuses 11 of 20 off-topic, loses 1 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="189.2" r="6"><title>MedCPT, depth 20. Cut-off 0.9999: refuses 11 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="88.5" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 0.9999: refuses 12 of 20 off-topic, loses 2 of 45 on-topic.</title></circle>
<circle class="hit" cx="100.8" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 0.9999: refuses 12 of 20 off-topic, loses 3 of 45 on-topic.</title></circle>
<circle class="hit" cx="113.1" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 4 of 45 on-topic.</title></circle>
<circle class="hit" cx="125.3" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 5 of 45 on-topic.</title></circle>
<circle class="hit" cx="137.6" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 6 of 45 on-topic.</title></circle>
<circle class="hit" cx="149.9" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 7 of 45 on-topic.</title></circle>
<circle class="hit" cx="162.1" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 8 of 45 on-topic.</title></circle>
<circle class="hit" cx="174.4" cy="170.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 12 of 20 off-topic, loses 9 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="151.6" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 13 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="132.8" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 14 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="114.0" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 15 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="186.7" cy="95.2" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 16 of 20 off-topic, loses 10 of 45 on-topic.</title></circle>
<circle class="hit" cx="198.9" cy="95.2" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 16 of 20 off-topic, loses 11 of 45 on-topic.</title></circle>
<circle class="hit" cx="223.5" cy="95.2" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 16 of 20 off-topic, loses 13 of 45 on-topic.</title></circle>
<circle class="hit" cx="235.7" cy="95.2" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 16 of 20 off-topic, loses 14 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="76.4" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 17 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="248.0" cy="57.6" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 18 of 20 off-topic, loses 15 of 45 on-topic.</title></circle>
<circle class="hit" cx="284.8" cy="57.6" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 18 of 20 off-topic, loses 18 of 45 on-topic.</title></circle>
<circle class="hit" cx="346.1" cy="38.8" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 19 of 20 off-topic, loses 23 of 45 on-topic.</title></circle>
<circle class="hit" cx="456.5" cy="20.0" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 20 of 20 off-topic, loses 32 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<circle class="hit" cx="616.0" cy="20.0" r="6"><title>MedCPT, depth 20. Cut-off 1: refuses 20 of 20 off-topic, loses 45 of 45 on-topic.</title></circle>
<line x1="373.1" x2="395.1" y1="245.6" y2="245.6" stroke="var(--s1)" stroke-width="2"/>
<text x="403.1" y="249.6">MiniLM, depth 20</text>
<line x1="373.1" x2="395.1" y1="267.6" y2="267.6" stroke="var(--s1)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="403.1" y="271.6">MiniLM, depth 40</text>
<line x1="373.1" x2="395.1" y1="289.6" y2="289.6" stroke="var(--s2)" stroke-width="2"/>
<text x="403.1" y="293.6">bge-reranker-base, depth 20</text>
<line x1="373.1" x2="395.1" y1="311.6" y2="311.6" stroke="var(--s3)" stroke-width="2"/>
<text x="403.1" y="315.6">MedCPT, depth 20</text>
</svg>

The plot shows the trade-off. Each point is one cut-off. Higher means more off-topic questions refused. Further right means more on-topic questions lost. A curve near the top-left corner would be good. Hover a point to see its cut-off and counts.

- Details: counts are off-corpus refused (of 20) and on-corpus lost (of 45) at cut-off t. MiniLM: t 0.10 gives 5 and 2, t 0.50 gives 9 and 5, t 0.80 gives 14 and 9, t 0.90 gives 16 and 16. bge: t 0.10 gives 8 and 5, t 0.30 gives 13 and 7, t 0.50 gives 16 and 11, t 0.70 gives 17 and 15. MiniLM depth 40: t 0.10 gives 5 and 1, t 0.50 gives 7 and 4, t 0.80 gives 12 and 9. "Most off-corpus refused" (15 or more of 20) costs 13 to 16 of 45 on-corpus questions for MiniLM and 11 or more for bge. The chance that an off-corpus question scores below an on-corpus one (area under the curve; 1.00 = perfect) is 0.80 MiniLM, 0.84 bge, 0.87 MedCPT, 0.80 MiniLM at depth 40.
- Details: the 5 old off-topic questions are easy (MiniLM refuses all 5 at t = 0.9, bge all 5 at t = 0.5). The 15 near-miss questions are the hard part: at the same cut-offs 11 of 15 are refused. Near misses sit close to real papers, so the model scores them high.
- Details: the pre-filter caveat from T5 (questions with no hit above 0.6 cosine count as refused at every cut-off) did not matter: 0 of 65 questions are empty after the pre-filter in all nine runs.

### 6. MedCPT's score is squeezed against 1, but a cut-off still exists in the fourth decimal

MedCPT gives almost every question a score above 0.999. A normal cut-off (0.5, 0.9) does nothing. The curve step in the T5 report (0.05) hides that finer cut-offs do separate somewhat. They would be very fragile.

- Details: every on-corpus top-1 score is at least 0.9995. Only 4 of 20 off-corpus questions score below 0.04 and 14 of 20 score above 0.99. Finer cut-offs (not on the 0.05 grid): t 0.9995 gives 9 refused and 1 lost; t 0.9998 gives 11 and 1; t 0.9999 gives 11 and 2; t 0.99995 gives 12 and 3; t 0.99999 gives 12 and 8. Depth 40 is almost the same (8, 10, 11, 12, 12 refused; 1, 1, 2, 3, 7 lost). A usable cut-off would sit between 0.9995 and 0.99995 (raw score about 7.6 to 9.9), where model revision, rounding or other chunk text could move it. So: MiniLM and bge have a wide but weak cut-off range, MedCPT has a narrow range that is the best of the three in this set (about 11 of 20 refused with 2 of 45 lost) and also the most fragile.

### 7. These refusal numbers are optimistic

A cut-off picked from this curve is picked on the same 20 off-topic questions it is judged on. A new set would likely do worse. With only 20 questions, one question moves a rate by 5 points.

- Details: 95 % Wilson ranges on the counts above: 5/20 refused is [0.11, 0.47]; 14/20 is [0.48, 0.85]; 11/20 is [0.34, 0.74]; 2/45 lost is [0.01, 0.15]; 8/45 is [0.09, 0.31]. A repeated half split (cut-off chosen on half the questions with at most 10 % on-corpus loss, then scored on the other half, 2000 splits) refused 0.35 (MiniLM), 0.40 (bge), 0.52 (MedCPT), 0.34 (MiniLM depth 40) of the held-out off-corpus questions, against 0.43 to 0.61 on the half where it was chosen. On-corpus loss stayed at 0.04 to 0.11. Best in-sample points with at most 2 of 45 lost: MiniLM 5 of 20 (t 0.08), bge 7 of 20 (t 0.06), MedCPT 12 of 20 (t 0.9999), MiniLM depth 40 5 of 20 (t 0.08). With at most 9 of 45 lost: MiniLM 14, bge 14, MedCPT 12 to 16, MiniLM depth 40 13.

### 8. The label bias in favour of reranking is small, but points one way

The nine labels added in T6 came from reranked runs, so they can only help reranked runs. I measured the size of this effect. It is small, but it makes the thinnest results a bit less sure.

- Details: gain over `prefix` with the 150 labels from before the T6 pass against all 159: MiniLM 0.122 vs 0.120, bge 0.139 vs 0.140, MedCPT 0.070 vs 0.075; depth 40 MiniLM 0.159 vs 0.161, bge 0.148 vs 0.154, MedCPT 0.117 vs 0.124. Dense `prefix` 0.562 vs 0.564. The pass moved reranker gains by -0.002 to +0.007. The Phase 9 labels came from dense pools, which pushes the other way (in favour of dense). The one verdict near the edge is MedCPT depth 20 rewritten (lower end +0.009, and the pass moved its gain by +0.005). I did not re-run the bootstrap on the older labels, because `compare` refuses a golden file with another hash.

## Decisions the owner must make

Order matters: set the time limit first, then the model, depth and query text follow from it. Refusal is separate.

### Decision 1: the time limit

How long may the reranker take per request on your machine? Everything else depends on this number. Today's request takes about 7 s without reranking (estimate, see finding 4). "No reranker" is also an option.

| Option | What it means in practice | What the numbers say |
|---|---|---|
| Under 2 s | Only MiniLM at depth 20 fits. A request is about 23 % slower. | MiniLM +0.120 to +0.135 over dense, rule says replaces. |
| Under 5 s | MiniLM at depth 20 or 40. | MiniLM at depth 40 is the highest mean of all nine (0.725). |
| Under 15 s | Also bge and MedCPT at depth 20. A request is 2.3 to 2.5 times slower. | Neither adds a real quality gain over MiniLM (bge +0.019 [-0.025, +0.076]). |
| Under 30 s | Also bge and MedCPT at depth 40. | bge depth 40 vs MiniLM depth 40: -0.007 [-0.046, +0.033]. |
| No reranker | App stays as it is. | All nine setups pass the rule against dense. |

- Details: limit judged on the 90th percentile of the reranker time (the median gives the same sets).

Which run has the highest nDCG@5 within each example limit, and what the rule says about it (against `prefix`; the rule has no pairwise "best of several", so the second column shows whether any other eligible run is clearly different):

| Limit | Eligible runs | Highest mean nDCG@5 | Rule vs `prefix` | Any eligible run clearly better? |
|---|---|---|---|---|
| under 2 s | MiniLM 20 rewritten, MiniLM 20 original | MiniLM 20 original, 0.699 | replaces | no (original vs rewritten +0.015 [-0.005, +0.036]) |
| under 5 s | + MiniLM 40 | MiniLM 40, 0.725 | replaces | no (vs 20 rewritten +0.040 [-0.003, +0.099]; vs 20 original +0.026 [-0.025, +0.088]) |
| under 15 s | + bge 20, bge 20 original, MedCPT 20, MedCPT 20 original | MiniLM 40, 0.725 | replaces | no (MiniLM 40 minus bge 20: +0.021 [-0.022, +0.065]; MedCPT lower) |
| under 30 s | + bge 40, MedCPT 40 | MiniLM 40, 0.725 | replaces | no (bge 40 minus MiniLM 40: -0.007 [-0.046, +0.033]; MedCPT 40 minus MiniLM 40: -0.037 [-0.094, +0.018]) |

- Details: in the rule's wording, a "winner within the limit" is the run with the highest mean among the eligible ones whose comparison against `prefix` gives "B replaces A". All nine do. Between the eligible runs the rule never names a clearly better one, so the highest mean is a point estimate, not a proven winner. Pairwise comparisons for the "not clearly better" column were run for the pairs listed here and in findings 2 and 3.

### Decision 2: the model

Which of the three. Once you set a time limit, the models that fit are fixed (decision 1). MedCPT is the only domain-specific model but it is lower in quality here and its score is hard to use for refusal.

| Option | What it means in practice | What the numbers say |
|---|---|---|
| MiniLM (91 MB, about 1.7 s at depth 20) | Small download, small memory, smallest added time. General-purpose training data. | 0.684 to 0.725 nDCG@5, not distinguishable from bge. |
| bge-reranker-base (1.1 GB, about 10 s at depth 20) | Large download, large added time. | 0.704 to 0.718, not distinguishable from MiniLM. |
| MedCPT (438 MB, about 9 s at depth 20) | Medical (PubMed) training data. Public domain per its card. | 0.639 to 0.688, lower than MiniLM and bge at depth 20 (range fully below 0), clearly worse on factual and table. |

### Decision 3: the depth

How many hits the reranker sees, 20 or 40. Deeper costs about double the time and gives a slightly higher mean, which the rule does not call real.

| Option | What it means in practice | What the numbers say |
|---|---|---|
| 20 | Faster. The best any reranker can reach is 0.828 recall. | The reference variant in all comparisons. |
| 40 | About 2 times slower. The best reachable recall is 0.914. | Mean +0.014 to +0.049 higher than depth 20, ranges include 0 (MiniLM -0.003, MedCPT -0.0005, bge -0.021). |

### Decision 4: the query text

Whether the reranker reads the rewritten or the original question. This does not change time.

| Option | What it means in practice | What the numbers say |
|---|---|---|
| Rewritten | The reranker depends on the rewrite step and its small model. | Reference variant. |
| Original | The reranker reads exactly what the user typed. | Within noise for MiniLM (+0.015) and MedCPT (+0.006). bge is lower on paraphrased questions (-0.069 clearly worse). The golden questions are clean sentences, so messy real queries are not tested. |

### Decision 5: the refusal option

What the app does about off-topic questions, if anything. This changes what users see, so it is yours. A hard cut-off or a label needs a value, and that value would be chosen on the same 20 questions it is judged on.

| Option | What it means in practice | What the numbers say |
|---|---|---|
| Hard cut-off | Below the chosen top-1 score the app answers "No relevant papers found." Some on-topic questions would be refused too (see finding 5). Needs the API contract change (ADR-0027, task T9). | Best in-sample point with 2 of 45 lost: 5 to 12 of 20 refused depending on the model; held-out about 0.34 to 0.52 refused. MedCPT is the best on this set and the most fragile. |
| Low-confidence label | Same signal, looser cut-off. The app still answers and the UI adds "the papers may not cover this". No API contract change. | A looser cut-off costs fewer on-topic questions labelled by mistake, but still labels only some off-topic ones. |
| Nothing | Numbers stay in ADR-0026. The app answers every question as today. | Dense threshold is inert (ADR-0025). No cut-off on any model works cleanly on this set. |

Also open from T6 (not part of this task):
- q044 is not labelled because its heading is one of the banned Phase 8 labels.
- q037 has a label that rests on a chunking artefact.
- q064 was judged off-corpus; an owner check is invited.

## Not checked

- No human check of the golden set. The same author wrote the questions and the labels.
- No repeated timing runs, no timing with Ollama running at the same time, no memory measurement.
- No end-to-end request time with the real evidence prompt (7 s is an estimate from older benchmark averages).
- Refusal cut-offs are not tested on new off-topic questions. Only the half-split estimate above was done.
- Pooling only labelled sections that the nine reranked runs put in their top 5.
- No hybrid search yet (T10), no answer-quality check (Phase 11).
- Not changed: code, run files, golden set, plan file. Nothing is committed.

## Files

- Created: `docs/adr/0026-reranker.md` (status `Proposed`, Decision pending), this page.
- Created in `data/eval/reports/`: 23 comparison reports and 4 gate-curve reports (`gate-rerank_minilm.md`, `gate-rerank_bge.md`, `gate-rerank_medcpt.md`, `gate-rerank_minilm_d40.md`), plus the existing run reports.
- Details: comparisons run: each of the nine reranker runs against `prefix` (9); the six same-model variants (3 original, 3 depth 40); the three model pairs at depth 20 and at depth 40 (6); `rerank_bge` vs `rerank_minilm_d40`; `rerank_minilm_orig` vs `rerank_minilm_d40`. File names are `compare-<B>-vs-<A>.md`.

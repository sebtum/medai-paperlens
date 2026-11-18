# T11: Docs

**Status:** done. Docs only, no code or tests changed. Nothing committed. I found no code contradiction I could not fix with docs. Two small gaps are listed in the open points. The golden set is stated as LLM-labelled (Claude), not human-verified, everywhere it is mentioned.

| Check | Result |
|---|---|
| Files edited | `README.md`, `CLAUDE.md`, `docs/roadmap.md`, `docs/architecture.md`, ADR-0022, ADR-0023, ADR-0024 |
| ADR-0025 | Not touched |
| `2-plan.md` | Not touched (orchestrator ticks T11) |
| Commands documented | `--help` run for `eval_retrieval` (all five subcommands), `sync_golden`, `ingest`. No command that writes or calls a service was run |
| `pytest tests/ -q` | 442 passed (unchanged) |
| `ruff check .` | All checks passed |
| `.env` | Never read or printed |

## README.md

- New section "Retrieval evaluation" after "Full-text corpus": the five `eval_retrieval` subcommands, `sync_golden`, `ingest --collection`, the table of the four configs, and where runs, reports, rewrites and the changelog live.
- LangSmith setup in 4 steps (key in `.env`, `LANGSMITH_ENDPOINT` for the EU region, run the two commands, only questions and scores are sent). It says the local run needs none of it.
- New `EMBEDDING_QUERY_PREFIX` bullet with the default rule. The golden set is stated as LLM-labelled, not human-verified, in bold.
- Fixed a contradiction in the Goals bullet: it said "hand-labeled golden question set". It now says LLM-labelled; the judge calibration is marked "(planned)".

**Not checked:** the claim that `LANGSMITH_ENDPOINT` is the right variable for the EU host. It is the LangSmith SDK's standard variable and the T9c report says the key works on the EU host, but nothing in the code or docs set it before now.

## CLAUDE.md

- Commands: `ingest --collection`, the five eval subcommands and `sync_golden`, in the same Windows module form (no `uv run`).
- Architecture: heading is now "Phase 9 complete". New "Evaluation layer" block with one line per `app/evaluation/` module, both scripts and `data/eval/`. Updated the `embedding.py` line (`query_prefix`, `get_query_prefix`, `EMBEDDING_QUERY_PREFIX`), the `search.py` line (`collection_name`, `search_hits`, threshold "measured inert and kept") and the `ingest.py` line (`--collection`).
- Build order paragraph: Phase 9 executed, Verify and PR pending, next is Phase 10. Env var list: added `EMBEDDING_QUERY_PREFIX` next to `EMBEDDING_MODEL`, and optional `LANGSMITH_ENDPOINT`. No existing rule was removed or weakened.

**Not checked:** `LANGSMITH_ENDPOINT` is added to the list because README now mentions it. It is not in `.env.example` (see open point 2).

## docs/roadmap.md

- Row 9 is marked done with ADR-0023, ADR-0024, ADR-0025 and the wording "Claude-authored and LLM-labelled, not human-verified", plus a note that it replaced the Gemini-drafted plan.
- New section "Phase 9 findings that feed Phase 10", labelled as input candidates, not decisions, owner decides scope in Discuss. It lists: inert 0.6 threshold and a better refusal signal; the `Rewritten question:` leak and the weak 0.6B rewriter; the paraphrased gap (0.224); `cross_paper` weakest; the k = 10 token cost. Row 10 links to it.
- Updated stale text: the Gemini-as-drafter sentence, and the three open questions (bge-base settled, domain embedder not tested and still open, rewriting measured).

**Not checked:** there was no "Phase 9 next" text in the roadmap to update.

## docs/architecture.md

- New "Evaluation Layer" subsection under Core Components: purpose, a small flow, code location, one collection per embedder, git as source of truth with LangSmith as an opt-in synced copy, cached rewrites, the prefix handling, the limits.
- Retrieval Layer: one line about the query prefix. Env var example gained `EMBEDDING_MODEL` and a commented `EMBEDDING_QUERY_PREFIX`.
- The request-flow diagram is untouched.

**Not checked:** the file has older stale text (for example "Streamlit UI [planned]", the `/query`-only endpoint list, an outdated citation example). I did not fix it; it is not Phase 9.

## ADR-0023, ADR-0024, ADR-0022

- ADR-0023: new "Changes during Execute" section: golden set as built (142 labels, verify pass counts), pooling over all 383 candidates with 8 labels added, v0 to v1 numbers (150 labels, hash prefix `4a4733e5fa76`, Recall@5 0.510 to 0.509, nDCG@5 0.521 to 0.525), the `maira_2` banned-label rule that matches nothing, the report wording, a pointer to ADR-0025.
- ADR-0024: new "Changes during Execute" section: code in `app/evaluation/`, `numpy` declared, hash on each example and the `validate_golden` gate, run-file fields (`query`, `ngram_overlap`, `chunk_tokens`), cache semantics and `--no-llm`, truncation check (0 mismatches), the eval's empty provider prefix, `CHANGELOG.md`, the `.gitattributes` LF pin, sync of 50 examples to the EU host, the one duplicate experiment, and what was never run for real.
- ADR-0022: one parenthetical "Amended by ADR-0025" on the "no bge prefix yet" line. Nothing rewritten. History in all three was kept.

**Not checked:** I did not re-verify the numbers against the run files. They are copied from the T9a, T9b, T9c and T10 reports and from ADR-0025.

## Open points for the owner

| # | What is decided | Options in practice | Recommendation |
|---|---|---|---|
| 1 | The T10 recommendation to fix the `Rewritten question:` leak before Phase 10 is now written into the roadmap findings as a candidate | A: keep as a finding, decide in the Phase 10 Discuss. B: do it now as a small fix | A. The owner decides Phase 10 scope |
| 2 | `LANGSMITH_ENDPOINT` is now in README and CLAUDE.md but not in `.env.example` (I was told docs only) | A: add a commented `# LANGSMITH_ENDPOINT=` line to `.env.example`. B: leave it | A, one line, when you commit |
| 3 | `docs/architecture.md` has older stale text unrelated to Phase 9 | A: leave. B: a separate small docs pass | A for this PR |
| 4 | Phase status wording: "Phase 9 executed, Verify and PR pending" (CLAUDE.md) versus the heading "Phase 9 complete" you asked for | A: keep both as written. B: change the heading after Verify | A |
| 5 | Roadmap rows 7, 8 and 9 use a check mark symbol in the phase column (existing style) | A: keep for consistency. B: replace with the word "done" | A |

## Files

| File | State |
|---|---|
| `README.md` | changed |
| `CLAUDE.md` | changed |
| `docs/roadmap.md` | changed |
| `docs/architecture.md` | changed |
| `docs/adr/0022-default-embedder-bge-small.md` | changed (one line) |
| `docs/adr/0023-golden-set-methodology.md` | changed |
| `docs/adr/0024-eval-dataset-versioning-and-runner.md` | changed |
| `docs/phases/09-retrieval-evaluation/3-execute-T11.md` | new (this file) |

# Phase 9: Retrieval evaluation — Verify

**Result: Phase 9 passes verification, after four fixes.** My own verify and a second, independent review by Codex both found the main result sound. Codex found four real defects in the evaluation tooling; all four are fixed and tested. The golden set is still LLM-labelled (written by Claude), not human-verified.

Verified on branch `phase-9-retrieval-evaluation`, base commit 6bc09d1, on 2026-11-18, with the uncommitted working-tree changes included.

## What is in the working tree but not committed

These are the follow-on changes made after the last commit, and they are part of what was verified.

- Two new metrics, precision and recall-versus-ceiling (the best recall that k slots allow), in `metrics.py`, `compare.py`, `report.py`, with tests.
  - Details: reported only, not used by the decision rule. 451 tests pass with them.
- Docs: CLAUDE.md metric list, ADR-0025 pointer, roadmap section with the ranked options for a better off-corpus refusal signal.
- The diagnosis note `3-execute-diagnosis.md` (analysis only, no code).
- Not part of the phase, keep out of the PR: `main.py`, `plan_llm_model_decision.md`, `plan_phase4.md`, `plan_phase5.md`, `plan_phase6.md`.

## Acceptance criteria

Every task was checked against its criteria in `2-plan.md`. Tests were re-run, and the numbers in ADR-0025 were reproduced from the stored runs.

| Task | What it delivers | Result | Evidence |
|---|---|---|---|
| T1 | Metric functions (Recall, MRR, nDCG, crowding, table hit) | Pass | `tests/test_eval_metrics.py`, hand-computed fixtures |
| T2 | Golden-set schema, loader, validation rules | Pass | `tests/test_eval_golden.py` |
| T3 | `collection_name` in search and ingest, scored hits | Pass | `tests/test_retrieval.py`; `papers` collection untouched by eval collection |
| T4 | LangSmith dataset sync | Pass | `tests/test_eval_sync.py` (fake client); dataset synced in T9 |
| T5 | Retrieval target, rewrite cache, label check against Qdrant | Pass | `tests/test_eval_target.py`; live label check passes on 595 chunks |
| T6 | Bootstrap, comparison, reports | Pass | `tests/test_eval_compare.py`; `compare baseline prefix` re-run gives +0.0410, range [+0.0123, +0.0737], same as ADR-0025 |
| T7 | Runner CLI, LangSmith adapters | Pass | `tests/test_eval_runner.py`; live `run --config prefix --no-llm` against Qdrant |
| T8 | Golden set v0 | Pass | `tests/test_golden_set.py` (4 passed) |
| T9 | Runs, pooling, golden set v1 | Pass | 8 run files and 16 reports in `data/eval/`; `CHANGELOG.md` logs the 8 added labels |
| T10 | Comparisons and ADR-0025 | Pass | ADR numbers match the stored runs; prefix adopted, the rest kept |
| T11 | Docs | Pass | README, CLAUDE.md, roadmap, architecture.md, ADR-0022 to 0025 and the ADR index |

- Details, T7: the live re-run of `prefix` reproduced all 50 question records identical to the committed `data/eval/runs/prefix.json`. Only the `git_hash` field differed (`6bc09d1-dirty`). I restored the file afterwards so the tree stays clean.
- Details, T10: the live API prefix is now `'Represent this sentence for searching relevant passages: '` with model `BAAI/bge-small-en-v1.5`, read from inside the rebuilt API container.

## Quality gates

- [x] pytest: 464 passed (451 before the Codex fixes, 13 new tests)
- [x] ruff: all checks passed
- [x] mypy: no issues in 42 source files
- [x] bandit: no issues (0 low, 0 medium, 0 high)
- [x] pip-audit: no known vulnerabilities (run against the lock file export, as in Phase 8)

## Demo path

- [x] `docker compose up --build -d` rebuilt the api and ui images and finished with exit 0. All four services are up and healthy.
  - Details: the api image that was running before was from 2026-11-11, older than the prefix commit, and had no `get_query_prefix`. This is why the rebuild mattered. After it, the api reports the prefix default.
- [x] Research question about MAIRA-2: grounded, confidence 0.872, 5 citations, each with paper and section.
- [x] Personal-advice question ("Should I take ibuprofen for my symptoms?"): refused, `route: unsafe_medical_advice`, `grounded: false`.
- [x] Off-topic question (pasta recipe): "No relevant papers found.", no citations.
- [x] `POST /query/stream`: `step_start` and `step_end` for each of 5 steps, one `citations`, 196 `token` events, one `final`.
- [x] UI health endpoint answers `ok`; `/model/status` shows the model is warm.
- Not re-checked this time: the UI in a real browser and a Ctrl-C stop. Neither changed in Phase 9 (the UI is untouched, and streaming was verified in Phase 8).

## Docs

- [x] ADR-0023, 0024 and 0025 exist and are in the index (0025 marked as amending 0022 and 0012)
- [x] README: how to run the eval, the sync, and the "LLM-labelled, not human-verified" warning
- [x] CLAUDE.md: evaluation layer, commands, phase status
- [x] Roadmap row 9 and the "Phase 9 findings that feed Phase 10" section
- [x] `docs/architecture.md` has an Evaluation Layer section

## Second review by Codex, and the fixes

An independent review by Codex (model `gpt-6-astra`, effort high) ran over the whole phase. It re-derived the numbers itself: all 180 nDCG@5 scores, the three comparisons, the threshold curve and the 16 report headers match, and the prefix gain is +0.0406 (range +0.0123 to +0.0737). It changed no files. It found four medium defects, all in the evaluation tooling and none in the app. The full record is in `.workflow/phase9.md` (untracked).

| # | What was wrong | Fix | Test |
|---|---|---|---|
| 1 | The banned-label rule for `maira_2` matched no real heading, so a bad label such as `Additional qualitative examples > Current Study` passed validation. | `golden.py` now also bans an explicit list of 13 real false headings, matched on the whole section label. No label in the real golden set is on the list, so the file and its hash did not change. | `test_banned_labels` (4 real headings), `test_real_maira_labels_used_by_the_golden_set_are_not_banned` |
| 2 | A `factual` question could carry labels from several papers, although ADR-0023 says one paper. | New validator rule. All 10 real factual entries already complied. | `test_factual_rejects_labels_from_two_papers` |
| 3 | If tagging failed after the examples were updated, a retry reported "unchanged" and never added the tag. | The sync now checks that the newest dataset version has a git-hash tag and writes only the tag if it is missing. | `test_retry_after_failed_tag_repairs_the_tag` |
| 4 | A `paper_id` such as `../escaped` in the stored data could make the text dump write outside its folder. | `authoring.py` accepts only plain names (letters, digits, `_`, `-`), refuses before any write, and also checks the resolved path. | `test_dump_text_rejects_a_paper_id_that_is_not_a_plain_name` (6 bad ids) |

- Details: ADR-0023 and ADR-0024 were updated to say what changed. Run after the fixes: 464 tests pass, ruff, mypy (42 files), bandit and pip-audit are clean.
- Details, Codex's blocked checks: four tests in `tests/test_query.py` need Hugging Face access, which its sandbox blocks. They pass here. `pip-audit` and the live Qdrant checks also need network; I ran those myself.
- Details, not done: Codex suggested exact expected values in the tests for the two new metrics. This is an optional hardening, not a defect.

## Findings and follow-ups

1. **Run reports are stale (do before the PR).** The 8 `run-*.md` files were written before the Precision and Recall-versus-ceiling columns existed, so none show them. The stored runs are fine. Regenerating needs Qdrant, which is running.
   - Details: `run --config <name> --no-llm` rewrites the run file and report. I tested it for `prefix`: the run data is identical, only `git_hash` changes. I did not regenerate the reports, to avoid unplanned churn in 16 committed files. Suggest regenerating all four v1 runs in one commit.
2. **Safety filter gap, not caused by Phase 9.** "Should I stop taking my medication?" is not refused. The phrase list has "should i take" but not "should i stop". It only came back as not grounded because retrieval found nothing. This is an old gap from Phase 4 and 7.
   - Details: the filter is a fixed list of phrases (`app/workflow/classify.py`). Phase 11 or a small separate fix could add a few patterns. A real safety evaluation is the better answer.
3. The golden set is LLM-labelled and has not had a human check. The 10-entry spot-check list is in `3-execute-T8.md` and is still open (also named in ADR-0025).
4. Threshold 0.6 is inert: off-corpus questions are not refused by any score cut-off. The ranked options are in the roadmap. This feeds the Phase 10 Discuss.
5. The diagnosis note suggests two cheap checks (collapse chunks into sections before the top 5, and read about 20 "right paper, wrong section" hits). Not done; they fit the start of the Phase 10 Discuss.
6. Branch `phase-9-retrieval-evaluation` is pushed to `origin` with the Phase 9 commit; the PR is the next step.
7. Keep the five untracked root files (`main.py`, `plan_*.md`) and `.workflow/` out of the PR.

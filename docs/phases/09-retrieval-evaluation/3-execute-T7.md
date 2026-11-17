# T7: The runner (command line)

**Status:** done and tested on fakes. It has not run against live services yet, except `sections`, which the orchestrating session ran against the real Qdrant.

| | |
|---|---|
| Tests | 58 new, 424 in total, all pass |
| Style, types, security | `ruff`, `mypy` (42 files), `bandit` all clean |
| Files | `scripts/eval_retrieval.py` (284 lines, thin), `app/evaluation/runner.py`, `app/evaluation/authoring.py`, `app/evaluation/langsmith_eval.py`, `tests/test_eval_runner.py` |
| Changed existing files | none |

## What T7 is

One command that ties the earlier tasks together: `.\.venv\Scripts\python.exe -m scripts.eval_retrieval`. It has five subcommands. Each is a section below. Any failure exits with code 1 and prints the full message.

The logic lives in `app/evaluation/` so `mypy` and `bandit` check it. The script only reads arguments, loads `.env`, prints and sets the exit code.

## Subcommand 1: `run`

**What it is:** asks every test question to the search and writes down what came back.

**Example:** `run --config baseline`

**Steps:**

| Step | What happens |
|---|---|
| 1. Validate | Checks the question file against the T2 rules. `--skip-validate` allows a half-finished file. |
| 2. Pre-flight | Looks through the whole database once. Stops with one list of problems (missing section, table question without a table, wrong embedding model). `--preflight-only` stops here. |
| 3. Rewrites | Reads each rewrite from the notebook file. If one is missing, it asks the AI. If the AI fails, the run stops. `--no-llm` lists all missing rewrites and stops. |
| 4. Search | Top 20 results per question, no similarity cut-off, one search per question. |
| 5. Per question extras | Reads the chunk text once to record the token cost estimate and the copy-detector number. The text is never stored. |
| 6. Output | `data/eval/runs/<name>.json` and `data/eval/reports/run-<name>.md`. |
| 7. `--langsmith` (optional) | Also runs the experiment on LangSmith and checks its scores equal the local ones. A difference exits with an error. |

**The four named setups** (pick with `--config`):

| Name | Embedder | Database | Query prefix | Rewrite |
|---|---|---|---|---|
| `baseline` | bge-small | `papers` | none | on |
| `no_rewrite` | bge-small | `papers` | none | off |
| `prefix` | bge-small | `papers` | "Represent this sentence for searching relevant passages: " | on |
| `bge_base` | bge-base | `eval_bge-base-en-v1.5` | none | on |

**Result:** tested with fakes only.

## Subcommand 2: `compare A B`

**What it is:** says whether run B is really better than run A, using T6's fair comparison.

**Example:** `compare baseline prefix`

It writes `reports/compare-<B>-vs-<A>.md` and prints the verdict (adopt or keep A). It refuses runs made against a different version of the question file.

**Second use:** `compare baseline baseline --threshold-b 0.5` compares the same run at two similarity cut-offs.

**Result:** tested with fakes only.

## Subcommand 3: `curve RUN`

**What it is:** shows how the scores change as the similarity cut-off moves from 0.40 to 0.80.

**Example:** `curve baseline` writes `reports/curve-baseline.md`.

**Why:** the app drops results below 0.6. This tells us if 0.6 is a good value.

**Result:** tested with fakes only.

## Subcommand 4: `sections`

**What it is:** lists every section of every paper with its chunk ids. This is what the question writer (T8) needs.

**Example:** `sections --paper maira_2`

**Option:** `--dump-text DIR` also writes each paper's text to `DIR`. The runner refuses a folder inside the repo, so paper text cannot end up in git.

**Result:** run live by the orchestrating session. It found 595 pieces, all made with bge-small.

## Subcommand 5: `pool RUN...`

**What it is:** after the runs, lists sections that showed up in the top 10 but are not in the answer key. We then check them by hand, because they may be correct answers we forgot to label. This is called **pooling**.

**Example:** `pool baseline no_rewrite prefix bge_base --out pool.md`

By default it looks at the top 10 results with similarity 0.40 or higher, and skips the off-corpus questions.

**Result:** tested with fakes only.

## The LangSmith part

**What it is:** `run --langsmith` sends the same retrieval and the same scoring formulas to LangSmith as an experiment, at 5 results and cut-off 0.6.

**Example of the safety net:** if LangSmith's average nDCG differs from the local one, the run exits with an error. So the two can never silently disagree.

**Result:** tested with a fake `aevaluate` (LangSmith's experiment function).

## Not checked yet

| Needs | What |
|---|---|
| Real Qdrant | the whole `run`, the pre-flight on real data, reading chunk text |
| Real Ollama | a rewrite through the real client (does it accept `temperature` and `seed`?) |
| Real embedder | loading bge-base and its real vectors |
| Real LangSmith | the real `aevaluate` call and whether synced example ids work as its data |
| Real chunks | whether the text is always present in the database (a missing text fails the run loudly) |

T9 is the first real run and will cover all of these.

## Decisions for you

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | The runner imports a private name `_estimated_tokens` from the app (the app's own token estimate). Accept? | A: accept, same pattern as T5. B: rename it public, which touches app code. | A |
| 2 | `pool` ignores results under 0.40. Keep? | Keep 0.40 (the low end of the curve), or use 0.60 for a shorter list. | Keep 0.40; pass `--threshold` if the list is too long |

## Choices the agent made that the plan did not fix

| Choice | Why |
|---|---|
| Token cost counts only the chunk text | The report compares k values. A constant label cost would not change the comparison. |
| `run --langsmith` runs the local path first, then the experiment | Gives the "numbers match" check and a local run file. Costs one extra search pass. |
| Errors inside LangSmith's target or evaluators are recorded and re-raised | LangSmith logs and swallows them by default. |
| `compare`, `curve`, `pool` refuse runs made with a different question file | Old labels against new runs would be meaningless. Re-run instead. |
| A run file is overwritten on re-run | Runs are committed in git, so history is kept. |
| The git hash gets `-dirty` if tracked files changed | Reproducibility, without failing a run over metadata. |
| The rewrite model must equal `OLLAMA_MODEL` (or `GEMINI_MODEL`) | The notebook is keyed by model. A mismatch would poison it. |
| `--embedder`, `--skip-validate`, `--preflight-only`, `--no-llm` added | T8 needs partial sets and a pre-flight check. A new embedder needs its own database name. |
| `.env` is loaded in `main()` through an injectable function | Same effect as `ingest.py`, and tests never read the real `.env`. |

## What T8 and T9 need from this

- **T8:** `sections` for the section list and text, and `run --preflight-only` to check the answer key. The `maira_2` appendix rule is still a guess.
- **T9:** the first `run --config baseline` needs Ollama up for the rewrites. Commit `data/eval/rewrites.jsonl`. `bge_base` needs its database built first with `scripts.ingest --collection eval_bge-base-en-v1.5` and `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`. After the question file changes (v0 to v1), all runs must be redone. `run --langsmith` needs `LANGSMITH_API_KEY` and a synced dataset.

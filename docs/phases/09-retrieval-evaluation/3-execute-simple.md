# Phase 9: what each task does, in plain words

Status: T1 to T8 are built and tested. The 50 questions exist. Nothing has been scored yet (that is T9).

## The big picture

Our app answers questions about medical AI papers. It first **searches** a database for the most relevant pieces of the papers. Then an AI writes an answer from those pieces.

Phase 9 asks: **how good is that search, and can we make it better?**

To answer, we need:
1. A **test**: about 50 questions where we know which parts of which papers should come up (T2, T8).
2. A **ruler**: formulas that turn "the search found these parts" into a score (T1).
3. A **runner** that asks the search each question and scores the result (T5, T7).
4. A **fair comparison**: is setup B really better than setup A, or just lucky (T6)?

## Status at a glance

| Task | In one line | Status | Tests |
|---|---|---|---|
| T1 | The ruler: scoring formulas | done | 35 |
| T2 | The test format and its checker | done | 46 |
| T3 | Several search databases side by side | done | 46 |
| T4 | Copy the test questions to LangSmith | done | 19 |
| T5 | Ask the search one question, check the answer key | done | 48 |
| T6 | Compare two setups fairly, write reports | done | 29 |
| T7 | The command-line runner that ties it together | done | 58 |
| T8 | Write the 50 questions | done | 4 |
| T9 to T11 | Real runs, decisions, docs | not started | - |

All checks pass on the whole project: 428 tests, `ruff`, `mypy`, `bandit` clean.

---

## T1: The ruler (scoring formulas)

**What it is:** a small set of formulas that score one search result.

**Example:** you ask "which paper describes X?". The search returns a list. The right answer is in the list, but at position 3. Is that good? The formulas turn it into numbers:

| Score | Question it answers | Example |
|---|---|---|
| Recall | How many of the right answers did it find? | found 1 of 2 = 0.5 |
| MRR | How high up was the first right answer? | position 1 = 1.0, position 3 = 0.33 |
| nDCG | Like MRR, but it also counts how good each right answer is. Rewards putting the best first. | **the main score of this phase** |

Two more rules: results below a similarity cut-off are dropped, and the same section counts only once.

**Result:** done. 35 tests with answers worked out by hand.

**Not checked:** nothing open.

---

## T2: The test format and its checker

**What it is:** the rules for the 50-question test file, plus a program that checks the file follows them.

**Each question has:** the text, a type, and which paper sections are the right answers (2 = ideal, 1 = acceptable).

**The six types and how many of each:**

| Type | Count | Meaning |
|---|---|---|
| factual | 10 | one simple fact |
| paraphrased | 10 | same, but reworded so it does not copy the paper |
| cross_paper | 8 | needs two or more papers |
| table | 8 | the answer is in a table |
| multi_section | 9 | spread over several sections of one paper |
| off_corpus | 5 | no paper answers it (the search should find nothing) |

**What the checker enforces, for example:**
- the exact counts above;
- every one of the 10 papers appears in at least 3 questions;
- a paraphrased question must not contain a paper title or model name;
- the file may not contain paper text, only our own questions and labels.

**Result:** done. 46 tests.

**Not checked:** one rule about odd section names in the `maira_2` paper is a guess. T8 checks it against the real names.

---

## T3: Several search databases side by side

**What it is:** the app has one database called `papers`. To compare embedding models (the part that turns text into numbers), each model needs its own database. Otherwise one model's data overwrites the other's.

**What changed:**
- the loader and the search can be told which database to use, and still use `papers` if you say nothing;
- the loader script got a `--collection` option;
- a new search function also returns the **similarity score** and whether a piece is text or a table. The ruler needs both.

**Result:** done. Tests show a second database leaves `papers` alone, and the old app tests still pass.

**Differs from the plan:** the plan wanted the old search rebuilt on the new one. The old one returns more details (title, authors, page), so that was not possible without changing the app's output. The two share the database call and read the results separately. Some code is duplicated. You accepted this.

**Not checked:** the `--collection` option was not run against a real database in this task.

---

## T4: Copy the test questions to LangSmith

**What it is:** LangSmith is an online tool for tracking experiments. This task copies our question file there. The file in git stays the master copy.

**What it does:**
- file unchanged since the last copy: does nothing;
- file changed: adds new questions, updates changed ones, removes deleted ones;
- labels that version with the git commit, so you know which file version a result used;
- refuses to run if the file has uncommitted changes or the API key is missing.

**Result:** done. 19 tests on a fake LangSmith with no internet.

**Differs from the plan (accepted):**
- the "last uploaded version" marker is stored on each question, not on the whole dataset (this LangSmith version cannot change dataset info afterwards);
- the script also checks the file against T2's rules before uploading.

**Not checked:** it has never talked to the real LangSmith. The first real sync in T9 will show.

---

## T5: Ask the search one question, check the answer key

**What it is:** the code that takes one test question and runs it through the same steps as the app, then hands back the ranked results.

**Think of a library test.** We ask the librarian a question and write down the shelf order they give us. Later we compare it to the answer key.

| Step | What happens | Example |
|---|---|---|
| 1. Rewrite (optional) | A small AI turns the question into a search-friendly form. | "How good is MAIRA-2 at writing X-ray reports?" becomes "MAIRA-2 radiology report generation performance" |
| 2. Embed | The text becomes a list of numbers describing its meaning. | about 384 numbers |
| 3. Search | The database returns the 20 closest pieces, each with a score from 0 to 1. | piece 1: 0.78, piece 2: 0.71, ... |
| 4. Hand back | The ranked list goes to the ruler (T1). | Is the right section in the list, and where? |

We keep all 20 results and their scores on purpose. Later we can ask "what if only the top 5?" or "what if we drop everything under 0.6?" without searching again.

### Part A: rewrite memory

**Problem:** the AI rewriter does not always give the same wording. A score change could then be luck, not a real improvement.

**Fix:** a notebook file (`rewrites.jsonl`). The first rewrite of a question is written down. Every later run reads it from the notebook. All experiments use the same rewrites, even with the AI switched off. If the question, the model or the rewrite instructions change, the entry is redone.

**If the AI fails:**

| | In the app | In the test |
|---|---|---|
| What happens | quietly uses the original question | stops with an error |
| Why | the user should still get an answer | otherwise "with rewrite" would secretly be "without rewrite" and the comparison would be wrong |

### Part B: safety check before scoring

It looks through the whole database and stops with one list of problems if:
- a section named in our answer key does not exist;
- a table question has no table;
- the database was built with a different embedding model than the run uses (this catches an old Phase 8 bug).

### Part C: copy detector

**Problem:** a "paraphrased" question is meant to test meaning, not word matching. If it copies a sentence from the paper, the search finds it by matching words and the score proves nothing.

**The detector** finds the longest run of words in a row that the question shares with the paper text.

| | Text |
|---|---|
| Paper says | "the model was trained on 1.2 million chest radiographs" |
| Question A | "How was it trained on 1.2 million chest radiographs?" |
| Question B | "What amount of X-ray data went into learning?" |

Question A shares 7 words in a row: a copy, too easy. Question B shares 0: a good test.

**Rule:** a paraphrased question sharing 4 or more words in a row gets rewritten (done in T8). The detector only measures. It saves the number, never the text.

**Result:** done. 48 tests on fakes.

**Not checked:** never run against the real database or the real AI. T7 and T9 do that.

---

## T6: Compare two setups fairly, write reports

**What it is:** setup A is today's search. Setup B is a change, for example "add a prefix to the query". B scores 0.02 higher on average. Is that real, or luck from 50 questions?

**How it decides:**
- It re-draws the 50 questions at random 1,000 times and checks whether B still wins. This is the **bootstrap**. The random seed is fixed, so you always get the same answer.
- B replaces A only if it wins across the 95 % range **and** no question type got clearly worse. (Example: table questions drop while the average rises.)
- It counts wins, losses and ties per question type.
- It builds a score-versus-cut-off curve for the similarity threshold.

**Reports:** Markdown files. Each states the setup and says "golden set LLM-labelled (Claude), not human-verified". Types with few questions get a warning.

**Result:** done. 29 tests on made-up data. `numpy` is now declared in `pyproject.toml`.

**Not checked:** the bootstrap numbers were not compared with an independent implementation.

---

## T7: The runner (command line)

**What it is:** one command, `python -m scripts.eval_retrieval`, that ties T4, T5 and T6 together. Five subcommands:

| Subcommand | What it does | Example |
|---|---|---|
| `run` | Checks the answer key, asks every question, writes a run file and a report. | `run --config baseline` |
| `compare A B` | Is run B really better than run A? | `compare baseline prefix` |
| `curve RUN` | Score at each similarity cut-off from 0.40 to 0.80. | `curve baseline` |
| `sections` | Lists every section of every paper with its chunk ids. Used to write the questions. | `sections --paper maira_2` |
| `pool RUN...` | Lists sections that showed up in the top 10 but are not in the answer key. Used to find missing labels. | `pool baseline prefix` |

**The four setups we will compare:**

| Name | Embedder | Query prefix | Rewrite |
|---|---|---|---|
| `baseline` | bge-small | none | on |
| `no_rewrite` | bge-small | none | off |
| `prefix` | bge-small | "Represent this sentence for searching relevant passages: " | on |
| `bge_base` | bge-base (bigger) | none | on |

**Also in `run`:**
- records, per result, an estimate of how many tokens the text costs (so the report can show the cost of k = 3, 5, 10). No paper text is stored;
- `--langsmith` also runs the experiment on LangSmith and checks the numbers match the local ones. If they differ, it exits with an error.

**Result:** done. 58 tests on fakes. The logic lives in `app/evaluation/` so `mypy` and `bandit` check it. The script itself is a thin 284-line wrapper.

**Checked live (by the orchestrating session, after the agent finished):** `sections` against the real Qdrant works: 595 pieces, all made with bge-small.

**Not checked:** `run`, rewrites through the real Ollama, and the LangSmith experiment. T9 is the first real run.

**Decisions for you (both recommended: accept):**

| # | Question | Recommendation |
|---|---|---|
| 1 | The runner imports a private name `_estimated_tokens` from the app (same pattern as T5). Accept? | Yes. The alternative changes app code for an eval need. |
| 2 | The `pool` list uses a similarity cut-off of 0.40. Keep? | Yes. Pass `--threshold` if the list is too long. |

---

## T8: Write the 50 questions

**What it is:** Claude reads the 10 papers and writes the 50 test questions, each with the paper sections that answer it. This is the answer key for the whole phase.

**How:** two separate passes. First a draft. Then a skeptical second read of every question: does the section really answer it, is another section also an answer, is the question unclear?

**Example:** q044 asks about LLaVA-Med's two training stages. The check added the Abstract as an extra answer, because it also states both stages.

| Fact | Value |
|---|---|
| Questions | 50 (10 factual, 10 paraphrased, 8 cross-paper, 8 table, 9 multi-section, 5 off-corpus) |
| Labels | 142 (83 "ideal", 59 "acceptable") |
| Changed in the second pass | 10 questions got extra labels, none rewritten or dropped |
| Closest copy of paper wording in a paraphrased question | 3 words in a row (limit is 4) |
| Safety filter | no question is flagged as unsafe |

**Result:** done. 4 new tests. The real check against the live database passed: every labelled section exists.

**Not checked:** one author made all labels. No human has read them. The next step (T9) looks at what the search finds that we did not label, which can reveal missing labels.

**Weak spot, and a decision for you:** some papers have wrongly named sections (for example a figure caption used as a heading). The checker's rule for odd `maira_2` names is too narrow and bans nothing.

| Option | What it means | Recommendation |
|---|---|---|
| A. Leave it for now | T9 shows if these sections really show up in results | Yes |
| B. Extend the banned list now | Cleaner labels, but a code change outside T8 | Only if T9 shows a problem |

Full question list and a spot-check list of the 10 least certain questions: `3-execute-T8.md`.

---

## What comes next

| Task | What |
|---|---|
| T9 | Run the four setups, add missing labels (pooling), freeze the question set, sync to LangSmith |
| T10 | Compare, decide the defaults, write ADR-0025. Stops and asks you if rewrite-off or bge-base wins |
| T11 | Docs |

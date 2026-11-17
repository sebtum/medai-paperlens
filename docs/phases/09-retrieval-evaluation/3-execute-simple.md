# Phase 9, first wave, in plain words

This is the simple version of `3-execute.md`. Nothing is committed yet.

## The big picture

Our app answers questions about medical AI papers. It first **searches** a database for the
most relevant pieces of the papers, then asks an AI to write an answer from them.

Phase 9 asks: **how good is that search, and can we make it better?**

To answer, we need three things:
1. A **test**: about 50 questions where we know which parts of which papers should come up.
2. A **ruler**: formulas that turn "the search found these parts" into a score.
3. A **test runner** that asks the app's search each question and compares the results to the answers we know.

The first wave built the ruler, the test format and two helpers. Nobody has run the test yet.

## Task by task

### T1: the ruler (scoring formulas)
**What it is:** a small set of formulas that score a search result.

**Example:** you ask "which paper describes X?". The search returns a list. The right answer
is in the list, but at position 3. Is that good? The formulas turn this into numbers:
- **Recall:** how many of the right answers did it find? (found 1 of 2 = 0.5)
- **MRR:** how high up was the first right answer? (position 1 = 1.0, position 3 = 0.33)
- **nDCG:** like MRR, but it also counts how *good* each right answer is and rewards putting the best ones first. This is the main score for this phase.

It also drops results whose similarity score is below a cut-off and keeps only the top few.
If the same section shows up twice, it counts once.

**Result:** done. 35 small tests, with expected numbers worked out by hand.

### T2: the test paper format and its checker
**What it is:** the rules for the 50-question test file, plus a program that checks the file
follows the rules.

**Each question has:**
- the question text,
- a type (simple fact, reworded question, spans several papers, asks about a table, spans several sections, or has no answer in our papers),
- which paper sections are the right answers and how good each is (2 = ideal, 1 = acceptable).

**What the checker enforces, for example:**
- exactly 10 factual, 10 reworded, 8 cross-paper, 8 table, 9 multi-section and 5 no-answer questions;
- every one of the 10 papers appears in at least 3 questions;
- a "reworded" question must not contain the paper's title or model name, otherwise it's too easy;
- the file may not contain paper text, only our own questions and labels.

**Result:** done. 46 tests. No real question file exists yet.

### T3: let the app keep more than one search database
**What it is:** so far the app has one database called `papers`. To compare different
"embedding models" (the part that turns text into numbers so it can be searched), we need
one database per model. Otherwise one model's data overwrites the other's.

**What changed:**
- the loader (`ingest`) and the search can be told which database to use, and still use `papers` if you say nothing;
- the loader script got a `--collection` option;
- a new search function also returns the **similarity score** and the kind of piece it found (text or table), which the ruler needs.

**Result:** done. New tests show a second database leaves `papers` alone, and the old app tests still pass.

**One thing that differs from the plan:** the plan said the old search function should be
rebuilt on top of the new one. The old one returns more details (title, authors, page) than
the new one, so that wasn't possible without changing what the app returns. The two share the
database call and each reads the results itself. It works, but some code is duplicated.

**Not checked:** the `--collection` option was not run against a real database.

### T4: copy the test questions to LangSmith
**What it is:** LangSmith is an online tool for tracking experiments. This task copies our
question file there, so results can be tracked and compared over time. The file in git stays
the master copy.

**What it does:**
- if the file hasn't changed since the last copy, it does nothing;
- if it changed, it adds new questions, updates changed ones and removes deleted ones;
- it labels that version with the git commit, so you can tell which file version a result used;
- it refuses to run if the file has uncommitted changes, or if the API key is missing, so you can't upload something that isn't saved in git.

**Result:** done. 19 tests, all on a fake LangSmith with no internet. It has never talked to the real one.

**Two things that differ from the plan:**
- The plan wanted to remember "what version did I last upload" on the whole dataset. This version of LangSmith can't change that after the dataset is created. Instead each question stores it. The effect is the same.
- The script also checks the question file against T2's rules before uploading, so you can't upload a half-finished file.

## Did it all work together?

Yes. I ran the whole project's checks after all four tasks were in:

| Check | Result |
|---|---|
| Tests | 289 passed |
| Style (`ruff`) | clean |
| Types (`mypy`) | clean |
| Security scan (`bandit`) | clean |

`pip-audit` wasn't run, because no new libraries were added.

## What's still open

- **Not tested against real services:** nothing has run against the real database or the real LangSmith yet.
- **Decisions for you:**
  1. OK with the T4 change (version marker stored per question)?
  2. OK with the T3 duplicated reading code, or should it be cleaned up?
  3. Keep the T4 refusal to upload an invalid file?
  4. Tick T1–T4 off in `2-plan.md`?

## What comes next

- **Wave 2:** T5 builds the part that actually runs a question through the search (including the AI query rewriter) and checks that our labels match the real database. T6 builds the comparison maths and the report files. These two can run at the same time.
- **Later:** T8 writes the 50 questions, T9 runs everything, T10 decides whether to change any app defaults.

---

# Wave 2, in plain words

Nothing is committed yet. Two helpers were built in parallel. All checks pass on the combined code (366 tests, up from 289).

## What these two tasks are for

Wave 1 built the ruler and the test format. Wave 2 builds two more parts:
- **T5:** the part that asks the search one question.
- **T6:** the part that compares two search setups fairly and writes the report.

Still no real run: the 50 questions do not exist yet.

## T5: ask the search one question, and check the answer key

**What it is:** the code that takes one test question and sends it through the same steps as the app. It does the optional AI rewrite, turns the text into numbers, searches, and keeps the top 20 results with their scores.

**Three parts:**
- **Rewrite memory.** The AI rewrites each question once. The result is saved in a file, so later runs give the same rewrite and need no AI running. If the AI fails, the run stops. (In the app, a failure silently uses the original question. For a test that would hide the problem.)
- **Safety check before scoring.** It looks through the whole database and stops with one list of problems if:
  - a section named in our answer key does not exist,
  - a table question has no table,
  - the database was built with a different embedding model than the run uses (this catches an old bug from Phase 8).
- **Copy detector.** It measures the longest run of words a question shares with the text it should find. A "reworded" question that copies too much is too easy.

**Result:** done, 48 tests, all on fakes. Never run against the real database or the real AI.

## T6: compare two setups fairly, write reports

**What it is:** say setup A is today's search and B is a change, for example "add a prefix to the query". B scores 0.02 higher on average. Is that real, or luck from 50 questions?

**How it decides:**
- It re-draws the 50 questions at random 1,000 times and checks whether B still wins. This is called the bootstrap. The random seed is fixed, so you always get the same answer.
- B only replaces A if it wins across the 95 % range, **and** no question type got clearly worse. (Example: table questions drop while the average rises.)
- It counts wins, losses and ties per question type.
- It builds the score-versus-cut-off curve for the similarity threshold.
- Reports are Markdown files. Each states the setup and says "golden set LLM-labelled (Claude), not human-verified". Types with few questions get a warning.

**Result:** done, 29 tests, on made-up data.

## Did it all work together?

| Check | Result |
|---|---|
| Tests | 366 passed |
| Style (ruff) | clean |
| Types (mypy) | clean |
| Security scan (bandit) | clean |
| pip-audit | not run (see decision 1) |

## Decisions for you

| # | Question | Recommendation |
|---|---|---|
| 1 | The bootstrap uses `numpy`, which is installed only as a side effect of other libraries. Declare it in `pyproject.toml`? | Yes, then run `pip-audit`. Otherwise a future dependency change could remove it silently. |
| 2 | The report cannot show "how many tokens the evidence costs" for k = 3/5/10, because we store no paper text. Should the runner (T7) record a token count per result? | Yes, add it to T7. |
| 3 | The compare code does not force the main metric (nDCG@5, threshold 0.6). OK? | Yes. The report states the cut-offs anyway. |
| 4 | T5 imports a private name (`_PROMPT`) from the app's rewriter to fingerprint the prompt. OK? | Yes. |
| 5 | Tick T5 and T6 in `2-plan.md`? | Yes, once you agree. |

## What comes next

Wave 3: T7 builds the command-line runner that ties T4, T5 and T6 together. T8 is where Claude writes the 50 questions, and it needs T7 first (to list each paper's sections).

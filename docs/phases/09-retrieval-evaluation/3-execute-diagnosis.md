# Why are the retrieval scores low? A first diagnosis

**Status:** analysis only. Checks 1 and 2 of part 6 are done (results in part 8). Golden set is LLM-labelled (Claude), not human-verified.

**Short answer (updated after the checks):** crowding is real but small (about +0.03 nDCG@5 at best). The labels are not too strict. The main loss is ranking: the right section is usually in the top 20 (Recall 0.82) but often not in the top 5 (0.53). That points at a reranker (Phase 10) first, then chunking, then a biomedical embedder.

Words used below:
- **Collapse:** several chunks of the same section count as one section.
- **Crowding:** several chunks of one section use up the 5 slots, so fewer distinct sections reach the top 5.
- **Right paper, wrong section:** a top-5 chunk from a paper that has a label for the question, but from a section that has none.

## 1. How these numbers were made

- Source: the stored run files `data/eval/runs/{baseline,prefix,bge_base}.json` (top 20 chunks per question). No new search was run.
- 45 on-corpus questions. Off-corpus questions are skipped.
- The top 20 chunks were collapsed into distinct sections first. Then Recall@k was computed over those sections, with any grade counting.
- Because of that, these numbers read higher than the report's Recall@5. Example: baseline 0.58 here against 0.51 in the report, which cuts to 5 chunks first.
- The script was ad hoc and is not saved in the repo. The numbers are a first look, not a result to decide on.

## 2. What the data says

| Measure | Baseline | Prefix | What it suggests |
|---|---|---|---|
| At least one labelled section in the top 5 | 87 % | 89 % | The search is not blind. It usually finds an answer. |
| No labelled section anywhere in the top 20 | 4 % | 2 % | Total failures are rare. |
| Share of labelled sections found at k=5 / 10 / 20 | 0.58 / 0.73 / 0.78 | 0.60 / 0.78 / 0.82 | A wider list helps up to 10, then flattens. |
| Top-5 chunks from the wrong paper | 22 % | 20 % | The embedder mostly picks the right paper. |
| Top-5 chunks from the right paper, wrong section | 36 % | 36 % | Most of the loss is here. |

**Example:** q001 (factual). The top hits come from the right paper, and several are tables from the same section. That fills the slots with one section.

## 3. Per type (prefix run)

| Type (n) | R@5 | R@10 | R@20 | Hit in top 5 | Hit only at 6 to 20 | Nothing in top 20 | Right paper, wrong section | Wrong paper |
|---|---|---|---|---|---|---|---|---|
| all (45) | 0.60 | 0.78 | 0.82 | 0.89 | 0.09 | 0.02 | 0.36 | 0.20 |
| factual (10) | 0.85 | 0.85 | 0.85 | 0.90 | 0.00 | 0.10 | 0.58 | 0.18 |
| paraphrased (10) | 0.59 | 0.90 | 0.96 | 1.00 | 0.00 | 0.00 | 0.26 | 0.20 |
| cross_paper (8) | 0.22 | 0.40 | 0.50 | 0.75 | 0.25 | 0.00 | 0.47 | 0.20 |
| table (8) | 0.58 | 0.90 | 0.94 | 0.75 | 0.25 | 0.00 | 0.25 | 0.38 |
| multi_section (9) | 0.66 | 0.79 | 0.82 | 1.00 | 0.00 | 0.00 | 0.24 | 0.09 |

Notes:
- `cross_paper` stays low even at k=20 (0.50). Its questions have 6 to 7 labels, so a top 5 cannot hold them all.
- `table` has the most wrong-paper hits (0.38). Some table chunks are scrambled (T8 weak-spot list).
- For `factual`, 58 % of top-5 chunks are right paper, wrong section. I first read this as crowding or missing labels. Part 8 shows neither explains it: those chunks are real filler, not missed answers.

## 4. The three suspects

| Suspect | Evidence | Read |
|---|---|---|
| Embedding model | bge-base (bigger) gave +0.026 nDCG@5, not significant. Wrong-paper rate is only about 20 %. | Probably not the main cause. A medical-domain embedder is untested, so it is not ruled out. |
| Niche domain | All 10 papers are medical AI, so many chunks look alike. Paraphrased vs factual gap is 0.22. The prefix closed about a third of it. | A likely factor, mainly as confusion between similar sections and papers. |
| Chunking | The top 5 chunks collapse to fewer than 5 sections. This is why the report's Recall@5 (0.51) is below 0.58 here. 36 % of top-5 chunks sit in unlabelled sections of the right paper. Some tables are scrambled. | **Revised:** crowding is a minor part (part 8). Chunk size is still untested. Not "the most likely culprit" any more. |

## 5. Two causes not on the original list

| Cause | Evidence |
|---|---|
| Test strictness | `cross_paper` has 6 to 7 labels, so Recall@5 has a low ceiling. Labels are whole sections, so a good chunk in a neighbouring section counts as a miss. **Measured later (prefix run):** dividing Recall@5 by its ceiling moves the overall value from 0.531 to 0.545 and `cross_paper` from 0.221 to 0.281. So the ceiling is a minor part of the low score. |
| Missing labels | Pooling only saw the top 10 of the four setups, and one LLM labelled everything and judged every candidate. Top-5 hits were inside the pool and were already read (375 of 383 rejected), so the open risk is ranks 11 and lower, and the same reader's blind spots. A second reader is what the check in part 6 adds. **Result (part 8):** a second reader graded 24 of these hits and found 0 answers. Missing labels are not the explanation. |

## 6. Cheap checks, cheapest first

| Check | Cost | What it tells you |
|---|---|---|
| Collapse chunks into sections before taking the top 5, using the stored runs and the repo's metric functions | Minutes, no search | Whether crowding explains most of the gap. **Done: no, it is minor (part 8).** |
| Read about 20 "right paper, wrong section" hits | About 30 min | Whether they are missing labels. **Done: no, 0 of 24 were answers (part 8).** |
| Different chunk size into a new collection (`--collection` is supported), then re-run | Hours | Whether chunking matters. |
| A biomedical embedder into a new collection, then re-run | Hours | Whether the domain matters. |
| Reranker and hybrid search | Phase 10 | Targets the right-paper-wrong-section problem directly. |

## 7. Recommendation

The first two checks are done. They point at ranking, so the next step is a reranker in Phase 10. The open follow-ups, with triggers, are in [`docs/roadmap.md`](../../roadmap.md#phase-9-diagnosis-follow-ups-decide-later). None blocks Verify.

**Not checked:** sections 2 and 3 come from one ad hoc pass over the stored runs. Part 8 numbers come from a second ad hoc script (not saved in the repo) over the same runs, and a second-reader pass whose text stayed outside the repo.

## 8. Results of checks 1 and 2

### Check 1: crowding (stored runs, k=5, threshold 0.6, 45 questions)

How crowded is the top 5 today:

| Run | Different sections in the top 5 (of 5) | Questions with a repeated section | Recall@20 |
|---|---|---|---|
| baseline | 4.11 | 58 % | 0.78 |
| no_rewrite | 3.89 | 64 % | 0.79 |
| prefix | 4.11 | 60 % | 0.82 |
| bge_base | 3.73 | 76 % | 0.80 |

Most questions repeat one section once, not more. Now the same 5 slots with a cap on chunks per section (prefix run, all 45 questions):

| Cap per section | Recall@5 | nDCG@5 | Precision@5 | Questions better / worse |
|---|---|---|---|---|
| none (today) | 0.531 | 0.566 | 0.431 | |
| 3 | 0.531 | 0.566 | 0.427 | 0 / 0 |
| 2 | 0.531 | 0.566 | 0.404 | 0 / 0 |
| 1 (collapse before cut) | 0.596 | 0.593 | 0.334 | 5 / 0 |

Reading:
- A cap of 2 or 3 changes nothing on the prefix run.
- Only cap 1 helps: nDCG@5 +0.027, Recall@5 +0.065, 5 questions better and none worse. For `bge_base` the cap helps a bit more (nDCG@5 0.550 to 0.567 at cap 2 or 3).
- Cap 1 gains most for `factual` (Recall@5 0.65 to 0.85). It does nothing for `cross_paper` and `table`.
- Our metrics cannot see the cost of a cap (a needed second chunk of a section is invisible). So even this small gain is an upper bound. Not adopted.
- Recall@20 is 0.82 and Recall@5 is 0.53. The answer is mostly in the stored top 20 but not in the top 5. That gap is a ranking problem, not a crowding problem.

### Check 2: second reader (24 right-paper-unlabelled hits)

A different Claude model (Opus), blind to labels, graded 36 top-5 chunks from the prefix run with full text: 24 right paper but unlabelled section, 8 labelled, 4 wrong paper.

| Group | n | Reader grade 0 | Grade 1 | Grade 2 |
|---|---|---|---|---|
| Right paper, unlabelled section | 24 | 24 | 0 | 0 |
| Labelled hits (control) | 8 | 2 | 2 | 4 |
| Wrong paper | 4 | 4 | 0 | 0 |

Reading:
- None of 24 unlabelled hits was an answer. Missing labels do not explain the 36 %.
- The reader was, if anything, stricter than the labels (2 of 8 labelled hits got 0).
- Limits: the reader is also Claude, so not independent. 24 hits is a small sample (0 of 24 still allows up to about 12 % in the whole group). The paper `pathalign` was left out (see below). Hits at ranks 6 and lower were not sampled.

### Side findings

- The live Qdrant collection `papers` holds 570 points and no `pathalign` chunks, while the stored runs report 595 chunks. Something removed that paper from the live collection after the runs. The eval preflight will probably refuse to run until `scripts.ingest --overwrite` restores it. Not changed here.
- That collection shows `indexed_vectors_count` 0 and `full_scan_threshold` 10000. At 570 points Qdrant scans every vector, so today's search is exact, not approximate.

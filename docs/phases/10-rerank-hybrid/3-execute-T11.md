# Phase 10, T11: Chunk size

Smaller chunks were tried and not adopted. They gave about the same ranking quality with half the text per answer and half the reranker time, but the gain is not safe.

## What was checked

After the reranker, 28.4 % of the top-5 chunks came from the right paper but a wrong section. The limit set in the discuss stage was 25 %, so the full experiment ran (owner approved).

Details: the papers were cut again with a cap of 256 tokens (the whole embedded text, title line included; today 512), in two ways, into eval-only collections. `eval_chunk256_merge` glues neighbouring pieces under one heading together up to the cap (1,258 chunks). `eval_chunk256_nomerge` keeps pieces apart (1,961 chunks). Today `papers` has 595 chunks and was not touched. Pipeline: dense search, MiniLM reranker, depth 20, rewritten question.

## Result

| Run | Chunks | nDCG@5 | Wrong-section share | Evidence tokens at 5 | Reranker time, median |
|---|---|---|---|---|---|
| `rerank_minilm` (adopted) | 595 | 0.683 | 0.284 | 2556 | 1592 ms |
| `chunk256_merge` | 1,258 | 0.703 | 0.258 | 1282 | 795 ms |
| `chunk256_nomerge` | 1,961 | 0.697 | 0.258 | 1105 | 780 ms |

| Against `rerank_minilm` | Mean delta | 95 % range | Wins / losses / ties | Verdict |
|---|---|---|---|---|
| `chunk256_merge` | +0.020 | -0.0323 to +0.0840 | 12 / 13 / 20 | A stays |
| `chunk256_nomerge` | +0.014 | -0.0402 to +0.0772 | 12 / 15 / 18 | A stays; multi_section clearly worse |

## What it means

- **Not adopted.** Neither variant has a safe gain. `papers` and the app stay as they are.
- **The wrong-section share only moved from 28.4 % to 25.8 %**, still above the 25 % limit. Cutting smaller does not remove the problem.
- **Same score with about half the text.** Evidence tokens fell from 2556 to 1282 (merge) and the reranker time from 1592 ms to 795 ms. Whether shorter context helps the answer step can only be judged with the generation evaluation (Phase 11). If it does, re-test the merge variant then.

## Idea for later: smaller chunks to save time or tokens

Smaller chunks are a good option for later, to cut the cost of the answer step. Two ways to use them, and they pull against each other:

| Use | Evidence text per answer | Reranker time | Open question |
|---|---|---|---|
| Keep top 5 | about half (1282 vs 2556 tokens) | about half (795 vs 1592 ms) | Do answers stay as good with shorter pieces? (Phase 11) |
| Raise to top 10 | about the same as today (2 x 1282 = 2564 tokens) | still about half, if depth stays 20 | Does seeing 10 focused pieces beat 5 big ones? The token saving is gone, but the spread over more sections may help answers. |

Details: the reranker time depends on how many candidates it scores (depth 20) and how long each is, not on the top-k cut, so top 10 alone does not bring it back. It would come back if the depth also had to double to 40 to keep the same text coverage (depth 40 took about twice as long on the current chunks). Both variants are untested: no `chunk256` run exists at top 10 or depth 40, and the effect on answers needs the generation evaluation. If tried, re-test the merge variant, compare nDCG@10 and the generation score, and record the answer-step time too (the local LLM's time grows with prompt tokens, which is the saving that top 10 would give back).

## Labels and golden set

The two runs put 20 sections into the top 5 that no earlier pass had seen. They were read by text only: 3 added (grade 1), 17 rejected with a reason. Golden hash is now `0c62886fd489` (65 questions, 164 labels), and every kept run was re-run on it. The six bge and medcpt runs stay on the old hash `cc08395a87ca` (owner decision). Numbers in ADR-0025 and ADR-0026 are from earlier hashes and are not changed.

## Code kept

`ingest --chunk-tokens N --no-merge-peers` and `run --rechunked`. Both are eval only and refuse the app collection `papers`. `--rechunked` skips the source-chunk-number checks (re-chunking changes the numbers) and keeps the labelled-section check.

## Not checked

- Effect on generated answers.
- `source_chunks` entries on the re-cut collections.
- 45 on-corpus questions, 8 to 10 per type, labels from the same author as the questions (LLM-labelled, not human-verified).

Full record: ADR-0028 addendum and `data/eval/CHANGELOG.md`.

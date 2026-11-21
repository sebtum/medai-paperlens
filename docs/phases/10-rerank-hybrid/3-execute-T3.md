# T3: Reranker provider

**Status:** done. All five acceptance criteria pass. All three model IDs load and score a pair.

## What

A reranker is a second model that reads the question and a passage together and gives a relevance score. T3 adds a small interface for it, one sentence-transformers version, and a test double. Nothing uses it yet. The workflow step (T8) and the eval (T5) come next.

- Details: no new dependency. `torch` and `sentence-transformers` were already installed (`app/retrieval/rerank.py`).

## Example

```python
from app.retrieval.rerank import get_reranker, rerank_passage

reranker = get_reranker()          # None when RERANKER_MODEL is empty, unset or "none"
if reranker is not None:
    passages = [rerank_passage(c.title, c.section, c.excerpt) for c in citations]
    scores = await reranker.score(question, passages)   # list[float], each in 0..1
```

In tests, use `FakeReranker(scores=[0.9, 0.1])` from `tests/conftest.py`. It never loads a model.

- Details: `Reranker` Protocol is `async score(query, passages) -> list[float]` plus a `model_name` property. `CrossEncoderReranker.score` returns `[]` for no passages without calling the model. It runs the model in `asyncio.to_thread` and passes `activation_fn=torch.nn.Sigmoid()`. `CrossEncoder` is built with `max_length=512`. `get_reranker()` is an `lru_cache(maxsize=1)` singleton, also the FastAPI dependency. Unset `RERANKER_MODEL` means off for now, because the default model is chosen after the Phase 10 comparison (T7).

## Result

| Acceptance criterion | Result | Where |
|---|---|---|
| `Reranker` Protocol and `CrossEncoderReranker` (thread, sigmoid) | Pass | `test_score_returns_sigmoid_scores_in_unit_range`, `test_score_runs_in_a_worker_thread`, `test_score_sends_query_passage_pairs_in_order`, `test_score_empty_passages_skips_the_model`, `test_protocol_is_satisfied_by_both_implementations` |
| `get_reranker_model_name()` (empty or `none` is off) and `get_reranker()` singleton | Pass | `test_model_name_off_values` (6 cases), `test_model_name_unset_is_off`, `test_model_name_reads_env`, `test_get_reranker_off_returns_none_without_loading_a_model`, `test_get_reranker_builds_once_with_max_length` |
| `rerank_passage()` | Pass | `test_rerank_passage_joins_title_section_text` |
| `FakeReranker` plus autouse fixture, no test loads a real model | Pass | `test_autouse_fixture_blocks_real_model_load`, `test_fake_reranker_*` |
| Manual check of three model IDs, not in CI | Pass, see table | below |

Gates:

| Gate | Result |
|---|---|
| `tests/test_rerank.py` | 20 passed |
| Full `tests/` | 506 passed (includes the other agent's T1 work as it stood at the time) |
| `ruff check` and `ruff format --check` on the 3 touched files | clean |
| `mypy app/` | no issues, 43 files |
| `bandit` on `rerank.py` | clean |

- Details: the autouse fixture `_no_real_reranker` clears `RERANKER_MODEL`, replaces `app.retrieval.rerank.CrossEncoder` with a function that raises, and calls `reset_reranker_cache()` before and after each test (it replaced `get_reranker.cache_clear()`). The refusal is an `AssertionError`, which is not a load error, so a test that reaches it fails loudly. A test that needs a built reranker patches `CrossEncoder` itself. Mid-run, one T1 test (`test_prompt_hash_is_the_hash_of_the_real_prompt`) failed once because T1 was still changing files. It passed in the later full runs.
- Details: mypy needed `pairs: list[Any]` in `_score_sync`. The library types its pairs as an invariant list of a union, which a plain `list[tuple[str, str]]` does not match.

### Manual model check

Run once, outside CI, with a script in the scratchpad (`check_rerankers.py`). Pair used: a MAIRA-2 question against a matching passage and against a pasta recipe. Machine: CPU only, torch 2.14.0+cpu, sentence-transformers 5.6.0.

| Model ID | Revision (commit) | Weights | Download + load | Warm score, 2 pairs | Good / bad pair (sigmoid) |
|---|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (redirects to `...-L6-v2`; both resolve) | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 91 MB | 4.1 s (files already cached) | 0.017 s | 0.9987 / 0.00001 |
| `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | 1.11 GB (safetensors) | 192.8 s including download | 0.172 s | 0.9993 / 0.00004 |
| `ncbi/MedCPT-Cross-Encoder` | `71caf65d4927987813984f54c284405a13fcca49` | 438 MB (`pytorch_model.bin` only) | 46.5 s including download | 0.104 s | 0.99999988 / 0.0000002 |

All three loaded without a workaround, including the MedCPT `.bin` file. All three rank the good pair far above the bad one. The sigmoid output stays within 0..1.

- Details: the old MiniLM ID `cross-encoder/ms-marco-MiniLM-L-6-v2` still works. The Hub reports the renamed ID `ms-marco-MiniLM-L6-v2`. Use either. The plan's note on the rename is correct.
- Details: raw logits. MiniLM returns real logits without our sigmoid (6.68 and -11.46). For bge-reranker-base and MedCPT the library output without an `activation_fn` already equals the sigmoid score. So the explicit sigmoid is needed for MiniLM and does no harm for the other two. It is not applied twice: with the explicit sigmoid the scores equal the library defaults, not a squashed version of them.
- Details: disk. The Hugging Face cache now holds about 3.8 GB for these models, more than the real need. My first attempt used `snapshot_download`, which also fetched ONNX and OpenVINO variants. That left two duplicate MiniLM folders of 849 MB each. Real weights are 91 MB. A normal `CrossEncoder` load only fetches what it needs. MedCPT shows two snapshot folders (`71caf65...` and `75e855e...`), 837 MB in total, so the tokenizer or a second file set came from another revision. I did not delete anything, because the delete command was denied. The cache folders `models--cross-encoder--ms-marco-MiniLM-L6-v2` and `models--cross-encoder--ms-marco-MiniLM-L-6-v2` can be removed by hand to reclaim about 1.6 GB.
- Details: Windows warned that symlinks are not available (no Developer Mode). The Hub cache copies files instead, which is why the folders are larger. Not a problem for loading.

## Not checked

- Speed on realistic input. The manual check scored 2 short pairs. It says nothing about 20 or 40 passages of up to 512 tokens. T5 and T7 measure real times.
- Container behaviour. The first-start download inside Docker and the shared model cache are T8.
- Score quality on the real eval. The check only shows each model separates one obvious good pair from one obvious bad pair.
- Thread safety under many parallel requests. One worker thread per call was tested, not concurrency.
- `get_reranker` as a FastAPI dependency in an API test. No route uses it yet (T8).
- The default `RERANKER_MODEL` value. Unset means off until T7 decides.

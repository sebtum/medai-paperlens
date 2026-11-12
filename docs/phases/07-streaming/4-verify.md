# Phase 7: Streaming — Verify

## Acceptance criteria
| Task | Criterion | ✔/✘ | Evidence |
|---|---|---|---|
| T1 | `stream()` on Protocol + Ollama (`/api/generate`, `/api/chat`), no `thinking`, HTTP errors raise | ✔ | `tests/test_llm_stream.py`; mypy clean |
| T2 | Gemini `stream()` skips empty chunks | ✔ | `tests/test_llm_stream.py` (fake `_client`; google-genai not installed locally) |
| T3 | Shared `FakeLlm` satisfies `LlmProvider`; old assertions unchanged | ✔ | `tests/conftest.py`; suite green |
| T4 | `generate_node` streams via writer; no-op under `ainvoke` | ✔ | `tests/test_workflow.py` |
| T5 | `/query/stream` order, headers, unsafe (R3), error (R5), `final == /query` (R2), 422 | ✔ | `tests/test_query_stream.py` |
| T6 | Disconnect cancels graph + LLM (R6) | ✔ | raw-ASGI test in `tests/test_query_stream.py` |
| T7 | Opt-in LangSmith, no client when unset (R8, R9), `.env.example` | ✔ | `tests/test_llm_stream.py`, `tests/conftest.py`; not run against a real LangSmith account |
| T8 | UI: `parse_sse_lines`, `stream_query_api`, step list, error mapping | ✔ (unit) | `tests/test_ui_client.py`; **not checked in a live Streamlit session** |
| T9 | Docs, `curl -N` manual check | ✔ / partial | `curl -N` showed incremental events (Execute stage); Ollama-side stop on Ctrl-C **not observed** (see Follow-ups) |

R7 (no thinking tokens) is covered by T1. R10 (SSE headers) is covered by the T5 header assertions. The Compose path itself was not run (see Demo path).

## Quality gates
- [x] pytest — 77 passed (1 unrelated qdrant version warning)
- [x] ruff — all checks passed
- [x] mypy — no issues in 29 source files
- [x] bandit — no issues identified
- [x] pip-audit — **clean against the lock** (`uv export --frozen` → `pip-audit -r … --no-deps`: "No known vulnerabilities found"). The plain local `pip-audit` still fails because the venv is stale, not the lock. Local run reports 58 vulns in 6 packages (pillow 12.3.0, filelock 3.29.0, certifi 2026.4.22, msgpack 1.2.2, setuptools 84.0.0, GitPython 3.1.62). `uv.lock` already pins fixed versions (pillow 13.0.0, filelock 4.2.4, certifi 2026.10.23, msgpack 1.2.4, setuptools 84.2.1) and has no GitPython. CI runs `uv sync` then `pip-audit`, so it audits the lock's versions. CI's own run is still to be seen on the PR.

## Demo path
- [ ] `docker compose up --build` works end to end — **not run**: Docker was not running in this session. `docker-compose.yml` has no changes for streaming; SSE headers (`Cache-Control: no-cache`, `X-Accel-Buffering: no`) come from the app. Needs a manual run before merge.

## Docs
- [x] ADRs — 0017, 0018, 0019 exist and are in `docs/adr/README.md`
- [x] README — event table, `curl -N`, LangSmith opt-in (per T9 note)
- [x] CLAUDE.md — architecture, API contract, phase status (per T9 note; the "Phase 7 complete / next is Verify" wording will need updating on sign-off)
- [x] roadmap — row 7 marked ✓

## Follow-ups
1. Run `docker compose up --build`, submit a query from the UI, and confirm tokens appear incrementally.
2. Click through the Streamlit UI against a live backend (U2 spike was avoided by design, never observed). Check that a 30 s+ stream is not interrupted.
3. Ctrl-C a `curl -N` mid-generation and watch Ollama (`ollama ps` or its log) to confirm generation stops.
4. Refresh the local venv from `uv.lock` (without `uv run`; use `pip install --ignore-installed --no-deps` per package) so local `pip-audit` matches CI.
5. Untracked `main.py` and `plan_*.md` in the repo root are not part of this phase. Do not include them in the PR.
6. `app/api/routes/query.py` imports the private `_ERROR_ANSWER` from `generate.py`. Consider making it public (minor).
7. Commit `4-verify.md` and update CLAUDE.md status to "Phase 7 verified" on sign-off.

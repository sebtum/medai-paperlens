# ADR-0019: LangSmith for observability and evaluation (opt-in)

## Status

Accepted (Phase 7).

## Context

As the workflow grows (streaming, full-text ingestion, rerank), we need per-node
latency and the exact prompts and outputs sent to the LLM. Phases 9 and 11 also
need datasets, experiments and LLM-judge evaluators. The user chose LangSmith for
all of these rather than home-grown metric reports. `langsmith` 0.12.5 is already
installed as a transitive dependency of langchain-core (via LangGraph).

## Decision

Use LangSmith, opt-in only:

- It is enabled only when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is set.
  `LANGSMITH_PROJECT` is optional. All three are documented in `.env.example`.
- `langsmith` becomes a direct dependency. This adds no new package.
- LangGraph node runs are traced automatically through langchain-core when tracing is on.
- Provider calls get `@traceable(run_type="llm")` on `generate` and `stream`. `stream`
  uses a join reducer so a run records the full answer.
- Scope per phase: Phase 7 covers tracing only. Datasets and evaluation come in
  Phases 9 and 11, each with its own decision record if it changes this one.

Tests and CI never set `LANGSMITH_TRACING`. An autouse fixture removes the
`LANGSMITH_*` variables. With tracing off, the app runs fully offline and
`@traceable` makes no network calls.

## Alternatives Considered

### Langfuse (self-hostable)

Pros:
- Can run locally, which fits the local-first approach.

Cons:
- More infrastructure (another Compose service with its own database). Parked for now.

### Logging only

Pros:
- No external service.

Cons:
- No trace tree, prompt view or eval tooling. Phases 9 and 11 would need custom reports.

## Consequences

Positive:
- Per-node latency and prompt/output inspection with no code in the nodes.
- The same tool carries the evaluation work in Phases 9 and 11.

Negative:
- When enabled, prompts and outputs go to an external service. This is acceptable
  because it is opt-in, the corpus is public papers, and ADR-0003 forbids patient
  data and PHI. If the corpus scope ever changes, this must be reconsidered.
- Ties evaluation tooling to a hosted vendor.

## Follow-up

- Revisit Langfuse if a fully local trace store becomes a requirement.

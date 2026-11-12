# Working process

Every roadmap phase ([`roadmap.md`](roadmap.md)) runs in four stages. Each stage
writes a short handoff file, and the next stage starts from a **fresh session**
that reads only that file. This keeps context small and avoids decisions getting
lost or distorted in long sessions.

Stage files live in `docs/phases/NN-name/` (e.g. `docs/phases/07-streaming/`) and
are committed in the phase PR together with its ADRs.

## Stages

| Stage | Reads | Writes | Gate |
|---|---|---|---|
| 1 Discuss | `CLAUDE.md`, the roadmap row, relevant code | `1-discuss.md` | User sign-off |
| 2 Plan | `1-discuss.md` | `2-plan.md` | User sign-off |
| 3 Execute | `2-plan.md`, one task block at a time | Code + tests; tick the task in `2-plan.md` with a one-line note | Stop only on deviation from the plan |
| 4 Verify | `2-plan.md` + the diff | `4-verify.md` | User sign-off → PR |

- **Discuss** settles *what*: goal, scope, requirements, assumptions, unknowns, risks.
- **Plan** settles *how*: ordered tasks with dependencies, parallel groups, and
  testable acceptance criteria.
- **Execute** implements task by task. Quick check per task = that task's tests +
  `ruff check` on touched files. If reality diverges from the plan, stop, update
  `2-plan.md`, and get sign-off before continuing.
- **Verify** checks the whole phase: every acceptance criterion, the five quality
  gates, the Docker Compose demo path, and docs.

## Context hygiene

- Start each stage with `/clear`. First message: *"Phase N, stage X — read
  `docs/phases/NN-name/<previous file>`"*.
- Tasks in the same parallel group that touch no shared files may run as
  subagents, each given only its own task block.
- Handoff files stay short (one or two screens). Record decisions and reasons, not
  transcripts.

## When to skip the process

Do the change directly (quick check + five quality gates) when **all** hold:

- no new ADR,
- no new dependency,
- no change to a shared interface (Protocols, API contract, `WorkflowState`),
- roughly three files or fewer.

Examples: bug fix, dependency bump, doc edit, small refactor. Anything on the
roadmap, or anything that needs an ADR, uses the full process.

## Templates

### `1-discuss.md`

```markdown
# Phase N: <name> — Discuss

## Goal
## Scope
### In
### Out
## Requirements
## Assumptions
## Unknowns
<!-- each: how and when it gets resolved (spike, measurement, user decision) -->
## Risks
## ADRs to write
```

### `2-plan.md`

```markdown
# Phase N: <name> — Plan

## Task graph
<!-- parallel groups in order, e.g. G1: T1 ‖ T2 → G2: T3 -->

## Tasks

### [ ] T1: <title>
- Files:
- Depends on:
- Group:
- Acceptance criteria:
  - [ ] ...
- Note: <!-- filled during Execute -->

## ADRs
```

### `4-verify.md`

```markdown
# Phase N: <name> — Verify

## Acceptance criteria
| Task | Criterion | ✔/✘ | Evidence |
|---|---|---|---|

## Quality gates
- [ ] pytest  - [ ] ruff  - [ ] mypy  - [ ] bandit  - [ ] pip-audit

## Demo path
- [ ] `docker compose up --build` works end to end

## Docs
- [ ] ADRs  - [ ] README  - [ ] CLAUDE.md  - [ ] roadmap

## Follow-ups
```

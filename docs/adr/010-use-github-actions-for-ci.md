# ADR-0008: Use GitHub Actions for CI

## Status

Accepted

## Context

The project needs continuous integration to catch broken code and demonstrate professional development workflow.

Every pull request should run automated checks before merge.

The CI pipeline should support:

- dependency installation
- linting
- type checking
- unit tests
- Docker image build

## Decision

Use GitHub Actions for CI.

The initial CI workflow will run on:

- push to `main`
- pull requests

The workflow will execute:

- `ruff check app tests`
- `mypy app`
- `pytest`
- `docker build`

## Alternatives Considered

### No CI

Pros:
- Fastest setup
- No workflow maintenance

Cons:
- Weak engineering signal
- Broken code can be merged unnoticed
- Bad fit for a portfolio project claiming production awareness

### GitLab CI

Pros:
- Strong CI/CD platform
- Good DevOps features

Cons:
- Project is hosted on GitHub
- Adds unnecessary platform split

### Local-only checks

Pros:
- Simple
- No external automation required

Cons:
- Not enforceable
- Easy to skip
- Less visible to reviewers

## Consequences

Positive:
- Pull requests are automatically validated
- Stronger professional workflow
- Better control over agent-assisted changes
- Visible CI status in GitHub

Negative:
- CI failures require maintenance
- Docker builds may slow the pipeline
- Some integration tests may need mocking to avoid heavy dependencies

## Follow-up

Start with a small CI pipeline. Add security scanning, integration tests, and evaluation smoke tests only after the MVP works.

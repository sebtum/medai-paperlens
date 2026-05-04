# ADR-0007: Use pytest, ruff, and mypy for quality gates

## Status

Accepted

## Context

The project should demonstrate professional engineering practices, not only AI experimentation.

The codebase needs automated checks for behavior, style, and type consistency. This is especially important because coding agents will assist development, and generated code must be reviewed and controlled.

## Decision

Use the following tools as local and CI quality gates:

- pytest for automated tests
- ruff for linting
- mypy for static type checking

These tools will run locally before completing work and in GitHub Actions on push and pull requests.

## Alternatives Considered

### pytest only

Pros:
- Simple
- Focuses on behavior
- Fast to adopt

Cons:
- Does not catch style issues
- Does not catch many type/interface mistakes
- Weaker protection against poor agent-generated code

### unittest instead of pytest

Pros:
- Built into Python
- No extra dependency

Cons:
- More verbose
- Less ergonomic
- Weaker developer experience for this project

### Black + Flake8 + isort instead of ruff

Pros:
- Common mature setup
- Fine-grained tools

Cons:
- More dependencies
- More configuration
- Ruff covers most required linting and formatting needs with less overhead

### No type checker

Pros:
- Faster initial coding
- Less friction

Cons:
- Worse interface discipline
- More runtime errors
- Weaker signal for production-quality Python

## Consequences

Positive:
- Better code quality
- Better control over coding-agent output
- Safer refactoring
- Stronger portfolio signal
- CI can catch problems before merge

Negative:
- Adds setup and configuration overhead
- mypy may require additional type annotations
- Some generated code may need cleanup before passing checks

## Follow-up

Start with moderate mypy strictness. Increase strictness later only if it improves maintainability without slowing MVP delivery.

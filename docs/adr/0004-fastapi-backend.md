# ADR-0004: Use FastAPI for the backend

## Status

Accepted

## Context

The system needs a backend API for health checks, query handling, and later integration with UI, retrieval, and agentic workflow components.

## Decision

Use FastAPI for the backend.

## Alternatives Considered

### Flask

Pros:
- Simple
- Mature
- Lightweight

Cons:
- Less built-in typing and request validation
- Less aligned with modern Python API development

### Streamlit-only app

Pros:
- Fastest UI development
- Fewer components

Cons:
- Weak API separation
- Less production-like architecture
- Harder to test backend logic independently

## Consequences

Positive:
- Clear API boundary
- Good testability
- Stronger production signal
- Easy local documentation through OpenAPI

Negative:
- Adds one more component compared with Streamlit-only

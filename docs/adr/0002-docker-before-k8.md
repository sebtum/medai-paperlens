# ADR-0002: Use Docker Compose before Kubernetes

## Status

Accepted

## Context

The project should be reproducible and deployable, but the MVP must remain achievable in two weeks.

Kubernetes would demonstrate deployment awareness, but it adds significant operational complexity before the core product exists.

## Decision

Use Docker Compose as the primary local deployment mechanism.

Kubernetes manifests may be added later as a stretch goal after the API, UI, retrieval, and evaluation pipeline work locally.

## Alternatives Considered

### No containers

Pros:
- Fastest initial setup
- Fewer files

Cons:
- Weaker reproducibility
- Less realistic production signal
- Harder to run services like Qdrant consistently

### Kubernetes from the beginning

Pros:
- Strong DevOps signal
- Closer to production infrastructure

Cons:
- Too much complexity for MVP
- Distracts from core AI engineering
- Harder local debugging

## Consequences

Positive:
- Reproducible local setup
- Easier onboarding
- Supports multi-service architecture later

Negative:
- Not a full production deployment
- Kubernetes experience remains limited unless added later

# ADR-0001: Use Qdrant for vector search

## Status

Accepted

## Context

The system needs local vector search for retrieval-augmented generation.

## Decision

Use Qdrant as the vector database.

## Alternatives Considered

### FAISS

Pros:
- Lightweight
- Fast local similarity search

Cons:
- Less product-like metadata filtering and service deployment story

### Chroma

Pros:
- Simple developer experience
- Common in RAG prototypes

Cons:
- Less convincing for production-style architecture

### PostgreSQL + pgvector

Pros:
- Strong production story
- Combines relational metadata and vector search

Cons:
- More database setup complexity for a two-week project

## Consequences

Positive:
- Clear local service via Docker Compose
- Good metadata filtering
- More realistic service architecture

Negative:
- Extra infrastructure component
- Requires Docker service orchestration

# ADR-0001: Use constrained agentic RAG instead of full deep research

## Status

Accepted

## Context

The project needs to demonstrate agentic RAG concepts for medical AI literature analysis.

A naive RAG system retrieves once and generates once. This is simple, but weak for vague research questions because it does not classify intent, rewrite poor queries, or validate whether the answer is grounded.

A full autonomous deep research agent is more powerful but too broad for a two-week MVP. It is harder to test, slower to run, less reproducible, and easier to overbuild.

## Decision

Use a constrained agentic RAG workflow with explicit nodes:

- query classification
- query rewriting
- retrieval
- structured evidence extraction
- answer generation
- grounding validation
- optional one-step retry

The system will not implement autonomous multi-step web research in the MVP.

## Alternatives Considered

### Naive RAG

Pros:
- Simple
- Fast to implement
- Easy to test

Cons:
- No validation step
- Weak handling of vague queries
- Less impressive as an AI engineering portfolio project

### Full Deep Research Agent

Pros:
- More powerful
- Stronger demo if implemented well
- Closer to frontier agentic systems

Cons:
- Too broad for MVP
- Harder to evaluate
- Less reproducible
- More failure modes
- Higher risk of unfinished implementation

## Consequences

Positive:
- More robust than naive RAG
- Easier to explain in interviews
- Testable workflow
- Clear safety boundaries

Negative:
- More complex than naive RAG
- Adds latency
- Less flexible than a fully autonomous research agent

## Follow-up

Revisit full paper-search automation after the local corpus, evaluation, and grounding validator work reliably.

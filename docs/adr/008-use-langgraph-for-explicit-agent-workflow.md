# ADR-0005: Use LangGraph for explicit agent workflow

## Status

Accepted

## Context

The project needs to demonstrate agentic RAG concepts without becoming a fully autonomous deep research system.

The workflow includes multiple explicit steps:

- query classification
- query rewriting
- retrieval (fills citations + evidence)
- answer generation
- grounding validation
- optional one-step retry

A simple function chain could work at first, but the project should make workflow decisions visible and testable.

## Decision

Use LangGraph for the agentic workflow once the MVP reaches the multi-step workflow stage.

LangGraph will be used to represent the constrained workflow as explicit nodes and edges.

The project will not use unconstrained autonomous agents in the MVP.

## Alternatives Considered

### Plain Python orchestration

Pros:
- Simple
- Easy to debug
- No additional framework dependency

Cons:
- Workflow decisions become less visible as complexity grows
- Retry and branching logic can become messy
- Weaker signal for agentic workflow design

### LangChain chains

Pros:
- Familiar ecosystem
- Useful abstractions for LLM applications
- Faster initial implementation for simple chains

Cons:
- Less explicit control over graph-like workflows
- Can become opaque if overused
- Easier to create framework-heavy code without clear architecture

### Fully autonomous agent framework

Pros:
- More flexible
- Stronger “agentic” demo if done well

Cons:
- Too broad for the MVP
- Harder to test
- Less reproducible
- Higher risk of unpredictable behavior

## Consequences

Positive:
- Explicit workflow structure
- Clear agentic architecture
- Easier to explain in interviews
- Supports branching and one-step corrective retry

Negative:
- Adds framework complexity
- Not needed for the first `/health` or naive retrieval slice
- Requires disciplined use to avoid overengineering

## Follow-up

Do not introduce LangGraph until the basic API, retrieval, and citation response work. Start with plain Python interfaces, then move orchestration into LangGraph when branching is needed.

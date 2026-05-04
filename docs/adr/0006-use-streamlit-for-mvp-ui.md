# ADR-0003: Use Streamlit for MVP UI

## Status

Accepted

## Context

MedAI PaperLens needs a simple user interface for entering medical AI research questions and displaying structured, citation-grounded answers.

The project has a two-week MVP scope. The UI should demonstrate the product workflow without consuming too much development time.

## Decision

Use Streamlit for the MVP user interface.

The UI will be responsible for:

- collecting the user question
- calling the FastAPI backend
- displaying the answer
- displaying citations
- displaying confidence and grounding status
- optionally showing debug information such as route, rewritten query, and retrieved chunks

## Alternatives Considered

### React / Next.js

Pros:
- Strong production frontend signal
- More flexible UI
- Better long-term web application architecture

Cons:
- Too much frontend overhead for a two-week AI engineering MVP
- More setup and state-management work
- Distracts from the AI workflow, evaluation, and system design

### Gradio

Pros:
- Very fast for ML demos
- Easy to deploy on Hugging Face Spaces
- Good for model interaction demos

Cons:
- Less flexible for product-like layout
- Weaker signal for API-based system architecture

### Backend-only API

Pros:
- Fastest option
- Keeps focus on backend and RAG pipeline

Cons:
- Less demo-friendly
- Harder for recruiters or reviewers to understand the product quickly

## Consequences

Positive:
- Fast MVP development
- Easy demo experience
- Good enough for portfolio presentation
- Allows focus on backend, RAG, evaluation, and safety

Negative:
- Not a production-grade frontend
- Less customizable than a full React frontend
- May need replacement if the product grows

## Follow-up

If the project evolves beyond the MVP, evaluate replacing Streamlit with a React or Next.js frontend.

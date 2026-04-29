# Product Brief: MedAI PaperLens

## Problem

Medical AI literature is fragmented across many papers, datasets, metrics, and model families. Researchers and AI engineers need a faster way to compare methods and identify limitations without relying on unsupported LLM summaries.

## Target User

AI engineers, students, and researchers exploring medical AI literature.

## MVP Goal

Given a research question, the system retrieves relevant medical AI paper information and produces a structured, citation-grounded summary.

## In Scope

- Paper search over a constrained corpus
- Query rewriting
- Structured extraction of task, dataset, model, metrics, and limitations
- Citation-grounded synthesis
- Grounding validation
- Local-first development

## Out of Scope

- Patient data
- Clinical diagnosis
- Treatment advice
- Real-time medical decision support
- Autonomous web browsing in the MVP
- Full deep research agent behavior

## Success Criteria

- The system can answer at least 10 predefined research questions.
- Each generated answer includes citations.
- Unsupported answers are flagged or refused.
- The project runs locally with Docker Compose.
- CI passes on every pull request.

# ADR-0003: Avoid patient data and clinical decision support

## Status

Accepted

## Context

The project is in the medical AI domain, but it is intended as a portfolio and learning project.

Using patient data, symptoms, diagnoses, or treatment recommendations would increase legal, ethical, and evaluation complexity.

## Decision

The MVP will focus on medical AI literature intelligence only.

The system will not process patient data, provide diagnosis, recommend treatments, perform symptom triage, or claim clinical decision support functionality.

## Alternatives Considered

### Build a symptom checker

Pros:
- Easy to understand as a product
- More obviously medical to non-technical users

Cons:
- High safety risk
- Hard to validate
- Bad regulatory positioning
- Likely to look naive to medical AI reviewers

### Build a clinical decision support prototype

Pros:
- Closer to real medical AI applications

Cons:
- Too risky for a two-week portfolio project
- Requires stronger validation
- Could create misleading claims

## Consequences

Positive:
- Safer scope
- Easier compliance story
- Better alignment with literature-focused medical AI engineering

Negative:
- Less direct clinical product feel
- Requires clear explanation in README

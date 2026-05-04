# ADR-0004: Use Ollama as default local LLM provider

## Status

Accepted

## Context

The project needs an LLM for query rewriting, answer generation, and grounding-related reasoning.

Because the project is in the medical AI domain, the default setup should avoid sending user input or medical content to external providers.

The project also needs to be runnable with free or open-source tooling.

## Decision

Use Ollama as the default local LLM provider.

The system will access the LLM through an internal provider interface so that other providers can be added later without changing the core workflow.

External providers such as OpenRouter may be added later as optional integrations, but they must be disabled by default.

## Alternatives Considered

### OpenRouter

Pros:
- Easy access to many models
- Useful for comparing model behavior
- No local GPU required for some models

Cons:
- External provider dependency
- Privacy and data-routing concerns
- Free-tier limits
- Worse default choice for a medical AI portfolio project

### Direct Hugging Face Transformers

Pros:
- Full control over model loading
- Strong open-source alignment
- No external API dependency

Cons:
- More engineering complexity
- Harder local setup
- More hardware-sensitive

### Hosted commercial APIs

Pros:
- Strong model quality
- Simple API integration
- Reliable inference

Cons:
- Requires API keys
- May cost money
- Less aligned with local-first and privacy-first project goals

## Consequences

Positive:
- Local-first design
- No required paid LLM API
- Better privacy positioning
- Good portfolio signal for responsible medical AI engineering

Negative:
- Model quality may be weaker than hosted frontier models
- Local inference may be slower
- User setup depends on local machine capabilities

## Follow-up

Add an `LLMProvider` abstraction so that Ollama, mock LLMs, and optional external providers can share the same interface.

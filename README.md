# MedAI PaperLens

MedAI PaperLens is an AI engineering portfolio project for medical AI literature intelligence.

It helps users explore medical AI papers by retrieving relevant literature, extracting structured evidence, comparing methods, and generating citation-grounded summaries.

## Scope

The MVP focuses on medical AI papers related to:

- chest X-ray visual question answering
- medical vision-language models
- radiology report generation
- evaluation methods and limitations

## Non-goals

This project does not provide:

- diagnosis
- treatment recommendations
- symptom triage
- patient-specific medical advice
- clinical decision support

## Planned Architecture

User question  
→ query classifier  
→ query rewriter  
→ paper retriever  
→ evidence extractor  
→ answer generator  
→ grounding validator  
→ cited summary

## Tech Stack

- Python
- FastAPI
- Streamlit
- Qdrant
- LangGraph
- Ollama
- Docker Compose
- GitHub Actions
- pytest
- ruff

## Status

Early development.

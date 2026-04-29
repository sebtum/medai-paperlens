## 1. Scope & Constraints
MedAI PaperLens is an MVP tool for medical AI literature intelligence.
- **Strictly Forbidden:** RAG, agents, vector databases, authentication, frontend, or cloud deployment unless explicitly requested.
- **No Clinical Claims:** Not a clinical decision support tool. No HIPAA, GDPR, or medical-device compliance.
- **Security:** Never commit patient data, API keys, or private papers. Use local-first defaults.

## 2. Tech Stack & Commands
- **Stack:** Python 3.11+, FastAPI, pytest, ruff, Docker.
- **Linting/Formatting:** `ruff check .` / `ruff format .`
- **Testing:** `pytest` (Focus on citation grounding and refusal logic).
- **Execution:** `uvicorn app.main:app --reload`
- **Containerization:** `docker build -t medai-paperlens-api .`

## 3. Project Structure
- `app/`: FastAPI backend and core logic.
- `tests/`: pytest suite.
- `docs/`: Product, architecture, and ADR documentation.
- `data/`: Sample data and prompt fixtures.

## 4. Coding Standards
- **Style:** PEP 8, 4-space indentation.
- **Naming:** `snake_case` (functions/modules), `PascalCase` (classes). Use domain-driven names (e.g., `evidence_extractor`).
- **Typing:** Type hints are mandatory for shared interfaces.
- **Dependencies:** Prefer standard library. Do not add LangChain or heavy frameworks without approval.

## 5. Workflow
- **Git:** One task per branch. Keep PRs small and scoped.
- **Commits:** Use imperative mood (e.g., `Add paper retriever`).
- **Quality:** Every PR must include/update tests and provide logs/screenshots for API changes.
- **Documentation:** Update `docs/` immediately if scope or architecture assumptions change.

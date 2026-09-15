# Architecture Decision Records

| # | Title | Status | Supersedes / Superseded by |
|---|---|---|---|
| [0001](0001-constrained-rag.md) | Use constrained agentic RAG instead of full deep research | Accepted | |
| [0002](0002-docker-before-k8.md) | Use Docker Compose before Kubernetes | Accepted | |
| [0003](0003-avoid-patient-data-and-clinical-decision-support.md) | Avoid patient data and clinical decision support | Accepted | |
| [0004](0004-fastapi-backend.md) | Use FastAPI for the backend | Accepted | |
| [0005](0005-use-qdrant-for-vector-search.md) | Use Qdrant for vector search | Accepted | |
| [0006](0006-use-streamlit-for-mvp-ui.md) | Use Streamlit for MVP UI | Accepted | |
| [0007](0007-use-ollama-as-default-local-llm-provider.md) | Use Ollama as default local LLM provider | Accepted | Extended by 0016 |
| [0008](0008-use-langgraph-for-explicit-agent-workflow.md) | Use LangGraph for explicit agent workflow | Accepted | |
| [0009](0009-use-pytest-ruff-mypy-for-quality-gates.md) | Use pytest, ruff, and mypy for quality gates | Accepted | |
| [0010](0010-use-github-actions-for-ci.md) | Use GitHub Actions for CI | Accepted | |
| [0011](0011-default-ollama-model-qwen25-3b.md) | Use qwen2.5:3b as default Ollama model | Superseded | Superseded by 0013 (then 0015) |
| [0012](0012-embedding-provider-protocol.md) | EmbeddingProvider Protocol for decoupled embedding inference | Accepted | |
| [0013](0013-upgrade-default-model-qwen35-4b.md) | Upgrade default Ollama model to qwen3.5:4b | Superseded | Supersedes 0011; superseded by 0015 |
| [0014](0014-llm-provider-protocol.md) | LlmProvider Protocol for decoupled LLM inference | Accepted | |
| [0015](0015-default-model-qwen3-06b-no-thinking.md) | Default model qwen3:0.6b with thinking disabled | Accepted | Supersedes 0011, 0013 |
| [0016](0016-optional-hosted-llm-provider.md) | Optional hosted LLM provider (Gemini) behind LLM_PROVIDER | Accepted | Extends 0007 |

See [0000-template.md](0000-template.md) for the ADR format.

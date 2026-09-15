FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS api
COPY --from=builder /app/.venv /app/.venv
WORKDIR /app
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.14-slim AS ui
COPY --from=builder /app/.venv /app/.venv
WORKDIR /app
COPY ui/ ui/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

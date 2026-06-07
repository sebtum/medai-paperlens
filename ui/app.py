import asyncio
import os
import time
from typing import Any, TypedDict

import httpx
import streamlit as st

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT: str = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT: str = f"{API_BASE_URL}/health"
MODEL_STATUS_ENDPOINT: str = f"{API_BASE_URL}/model/status"
REQUEST_TIMEOUT: float = 300.0

_ERR_CONNECT = "connect_error"
_ERR_TIMEOUT = "timeout"
_ERR_HTTP = "http_error"


class CitationData(TypedDict):
    title: str
    authors: list[str]
    year: int
    doi: str | None
    excerpt: str


class QueryResult(TypedDict):
    answer: str
    citations: list[CitationData]
    confidence: float
    grounded: bool
    debug: dict[str, Any]


class ModelStatus(TypedDict):
    warm: bool
    model: str
    estimated_warmup_seconds: int | None


async def check_backend_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(HEALTH_ENDPOINT)
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def check_model_status() -> ModelStatus | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(MODEL_STATUS_ENDPOINT)
            response.raise_for_status()
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return None


async def call_query_api(
    question: str,
    score_threshold: float = 0.4,
) -> tuple[QueryResult | None, str | None]:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        ) as client:
            response = await client.post(
                QUERY_ENDPOINT,
                json={"question": question, "score_threshold": score_threshold},
            )
            response.raise_for_status()
            return response.json(), None
    except httpx.ConnectError:
        return None, _ERR_CONNECT
    except httpx.TimeoutException:
        return None, _ERR_TIMEOUT
    except httpx.HTTPStatusError:
        return None, _ERR_HTTP


def _build_spinner_text(model_status: ModelStatus | None) -> str:
    if model_status is None or model_status["warm"]:
        return "Querying…"
    est = model_status["estimated_warmup_seconds"]
    est_text = f" (~{est // 60} min)" if est is not None else ""
    return (
        f"Model '{model_status['model']}' is loading{est_text}"
        f" — please wait…"
    )


@st.cache_data(ttl=10)
def _cached_backend_health() -> bool:
    return asyncio.run(check_backend_health())


def render_confidence_indicator(confidence: float) -> None:
    st.progress(confidence, text=f"Confidence: {confidence:.0%}")


def render_citation_card(citation: CitationData, index: int) -> None:
    authors_str = ", ".join(citation["authors"]) if citation["authors"] else "Unknown"
    label = f"{index}. {citation['title']} ({citation['year']})"
    with st.expander(label):
        st.write(f"**Authors:** {authors_str}")
        if citation["doi"]:
            st.write(f"**DOI:** {citation['doi']}")
        st.write(f"**Excerpt:** {citation['excerpt']}")


def render_citations(citations: list[CitationData]) -> None:
    st.subheader("Citations")
    if not citations:
        st.info("No citations found for this query.")
        return
    for i, citation in enumerate(citations, start=1):
        render_citation_card(citation, i)


def render_answer_section(result: QueryResult) -> None:
    if result["grounded"]:
        st.success("Grounded")
    else:
        st.warning("Not grounded")
    render_confidence_indicator(result["confidence"])
    st.subheader("Answer")
    st.write(result["answer"])


def render_refusal(result: QueryResult) -> None:
    st.error(
        "This query was identified as a request for personal medical advice. "
        "MedAI PaperLens provides literature summaries only — not clinical guidance."
    )
    st.write(result["answer"])


def render_error(error_type: str) -> None:
    if error_type == _ERR_CONNECT:
        st.error(
            f"Backend not reachable. "
            f"Check that the API server is running at {API_BASE_URL}."
        )
    elif error_type == _ERR_TIMEOUT:
        st.warning(
            "Request timed out. Ollama may still be loading the model — "
            "please try again in a moment."
        )
    else:
        st.error("The backend returned an unexpected error. Check the server logs.")


@st.fragment(run_every=5)
def _chat_zone() -> None:
    status = asyncio.run(check_model_status())

    if status is None:
        st.warning("Cannot determine model status. Retrying…")
        return

    if not status["warm"]:
        est = status["estimated_warmup_seconds"]
        est_text = f" (~{est // 60} min)" if est is not None else ""
        st.info(
            f"Model '{status['model']}' is loading{est_text}."
            " Chat will be available when ready."
        )
        return

    # Model is warm — render chat form
    st.caption(f"Model: `{status['model']}`")
    with st.form("query_form"):
        question = st.text_area(
            "Research question",
            placeholder=(
                "e.g. What methods are used for chest X-ray classification?"
            ),
            height=100,
        )
        score_threshold = st.slider(
            "Relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help=(
                "Minimum similarity score for retrieved papers"
                " (lower = more results, higher = stricter)"
            ),
        )
        submitted = st.form_submit_button("Search")

    if submitted and question.strip():
        st.session_state["result"] = None
        st.session_state["error"] = None
        start = time.monotonic()
        with st.spinner("Querying…"):
            result, error = asyncio.run(
                call_query_api(question.strip(), score_threshold=score_threshold)
            )
        st.session_state["result"] = result
        st.session_state["error"] = error
        st.session_state["elapsed"] = time.monotonic() - start
    elif submitted:
        st.warning("Please enter a research question.")

    result: QueryResult | None = st.session_state.get("result")
    error: str | None = st.session_state.get("error")
    elapsed: float | None = st.session_state.get("elapsed")

    if error is not None:
        render_error(error)
    elif result is not None:
        if result["debug"].get("route") == "unsafe_medical_advice":
            render_refusal(result)
        else:
            render_answer_section(result)
            render_citations(result["citations"])

    if elapsed is not None and error != _ERR_CONNECT:
        st.caption(f"Response time: {elapsed:.1f} s")

    debug_data: dict[str, Any] = (
        result.get("debug", {}) if isinstance(result, dict) else {}
    )
    if debug_data:
        route = debug_data.get("route", "unknown")
        rewritten = debug_data.get("rewritten", "")
        with st.expander("Debug", expanded=False):
            st.write(f"**Route:** `{route}`")
            if rewritten:
                st.write(f"**Rewritten question:** {rewritten}")
            if error:
                st.error(f"Error type: `{error}`")


def main() -> None:
    st.set_page_config(page_title="MedAI PaperLens", layout="wide")
    st.title("MedAI PaperLens")
    st.caption(
        "Citation-grounded summaries over a curated corpus of medical AI papers. "
        "Not a clinical tool — for research exploration only."
    )

    if not _cached_backend_health():
        st.error(
            f"Backend not reachable. "
            f"Start the API server at {API_BASE_URL} before using the app."
        )
        st.stop()

    _chat_zone()


if __name__ == "__main__":
    main()

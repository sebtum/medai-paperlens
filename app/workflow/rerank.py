import logging

from app.models.query import Citation
from app.retrieval.rerank import Reranker, UnavailableReranker, rerank_passage
from app.workflow.state import WorkflowState

logger = logging.getLogger(__name__)

RERANK_KEEP = 5
"""How many passages the rerank step hands to generate."""


def _cut(citations: list[Citation], keep: int, *, fallback: bool) -> dict:
    """Vector order cut to `keep`, with no reranker score."""
    return {
        "citations": citations[:keep],
        "rerank_top_score": None,
        "rerank_fallback": fallback,
    }


def make_rerank_node(reranker: Reranker | None, *, keep: int = RERANK_KEEP):
    """Workflow step between retrieve and generate (Phase 10).

    Retrieve hands over the vector search's top hits; this step re-scores them with
    the cross-encoder and keeps the best `keep`. With no reranker it only cuts to
    `keep`, so the graph and the stream events are the same in every setup. When the
    reranker fails, the vector order is kept (cut to `keep`) and `rerank_fallback`
    is set; the request still succeeds.
    """

    async def rerank_node(state: WorkflowState) -> dict:
        citations = state["citations"]
        if reranker is None or not citations:
            return _cut(citations, keep, fallback=False)

        query = state["rewritten"] or state["question"]
        passages = [
            rerank_passage(c.title, c.section or "", c.excerpt) for c in citations
        ]
        try:
            scores = await reranker.score(query, passages)
            if len(scores) != len(citations):
                raise ValueError(
                    f"reranker returned {len(scores)} scores "
                    f"for {len(citations)} passages"
                )
        except Exception:
            if isinstance(reranker, UnavailableReranker):
                # The load failure was logged with its traceback when it happened.
                logger.warning(
                    "rerank_node: %r unavailable, keeping vector order",
                    reranker.model_name,
                )
            else:
                logger.exception("rerank_node: reranking failed, keeping vector order")
            return _cut(citations, keep, fallback=True)

        # Highest score first; equal scores keep the vector order (sort is stable).
        order = sorted(range(len(citations)), key=lambda i: -scores[i])[:keep]
        logger.info(
            "rerank_node: kept %d of %d (top score %.3f)",
            len(order),
            len(citations),
            scores[order[0]],
        )
        return {
            "citations": [citations[i] for i in order],
            "rerank_top_score": scores[order[0]],
            "rerank_fallback": False,
        }

    return rerank_node

from langgraph.graph import END, START, StateGraph
from qdrant_client import AsyncQdrantClient

from app.llm.base import LlmProvider
from app.retrieval.embedding import EmbeddingProvider
from app.workflow.classify import classify_node
from app.workflow.generate import make_generate_node
from app.workflow.retrieve import make_retrieve_node
from app.workflow.rewrite import make_rewrite_node
from app.workflow.state import WorkflowState
from app.workflow.validate import validate_node


def route_safety(state: WorkflowState) -> str:
    if state["is_unsafe"]:
        return "unsafe"
    return "safe"


def build_workflow(
    client: AsyncQdrantClient,
    provider: EmbeddingProvider,
    llm: LlmProvider,
):
    graph: StateGraph = StateGraph(WorkflowState)

    graph.add_node("classify", classify_node)
    graph.add_node("rewrite", make_rewrite_node(llm))
    graph.add_node("retrieve", make_retrieve_node(client, provider))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_safety,
        {"unsafe": END, "safe": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)

    return graph.compile()

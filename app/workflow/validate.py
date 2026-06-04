from app.workflow.state import WorkflowState


def validate_node(state: WorkflowState) -> dict:
    return {"grounded": len(state["citations"]) > 0 and bool(state["answer"])}

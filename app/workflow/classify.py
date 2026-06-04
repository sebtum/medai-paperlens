from app.workflow.state import WorkflowState

_UNSAFE_PATTERNS: frozenset[str] = frozenset({
    "diagnose me",
    "my symptoms",
    "should i take",
    "can i take",
    "prescribe me",
    "my prescription",
    "medical advice",
    "treat me",
    "my condition",
    "my illness",
    "my disease",
    "am i sick",
    "i have been diagnosed",
})


def classify_node(state: WorkflowState) -> dict:
    lower = state["question"].lower()
    if any(pattern in lower for pattern in _UNSAFE_PATTERNS):
        return {
            "is_unsafe": True,
            "route": "unsafe_medical_advice",
            "answer": "Literature summaries only — no medical advice.",
            "grounded": False,
        }
    return {"is_unsafe": False, "route": "retrieval"}

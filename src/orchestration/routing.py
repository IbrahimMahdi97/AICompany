"""The orchestrator's control logic: shared state + deterministic routing.

Deliberately free of any LLM call and of any langgraph import, so it stays trivially
testable and cannot be entangled with the QA agent's judgment.
"""

from typing import TypedDict

from .contracts import CodeFile, QaVerdict, SolutionSpec


class GraphState(TypedDict):
    goal: str
    spec: SolutionSpec | None
    test_files: list[CodeFile]     # frozen after test_author runs once
    impl_files: list[CodeFile]     # rewritten by the developer each round
    verdict: QaVerdict | None
    defects: list[str]             # failing-test feedback handed back to the developer
    answers: str | None
    qa_round: int
    max_qa_rounds: int
    qa_log: list[str]


def route_after_ba(state: GraphState) -> str:
    """Need human clarification first, or is the spec final?"""
    spec = state["spec"]
    assert spec is not None
    if spec.open_questions and state.get("answers") is None:
        return "clarify"
    return "test_author"


def route_after_qa(state: GraphState) -> str:
    """Deliver, escalate, or loop back to the developer with real failures."""
    verdict = state["verdict"]
    assert verdict is not None
    if verdict.passed:
        return "deliver"
    if state["qa_round"] >= state["max_qa_rounds"]:
        return "deliver"  # escalate; the caller inspects verdict.passed
    return "developer"

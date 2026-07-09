"""Wires the agents into a LangGraph state machine.

The graph runtime + the two router functions + the clarify gate together ARE the
orchestrator. None of them call an LLM.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import business_analyst, developer, qa, test_author
from .contracts import AcceptanceCriterion, CodeFile, CriterionResult, QaVerdict, SolutionSpec
from .routing import GraphState, route_after_ba, route_after_qa

# The graph state carries these Pydantic models, so they get written into the checkpoint.
# Registering them tells LangGraph 1.2+'s msgpack deserializer they are known, expected
# types instead of emitting an "unregistered type" warning (and being blocked in a future
# release). An explicit allowlist also tightens security: only these custom types — plus
# LangGraph's own safe built-ins — may be revived from a checkpoint.
_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        SolutionSpec,
        AcceptanceCriterion,
        CodeFile,
        QaVerdict,
        CriterionResult,
    ]
)


def clarify(state: GraphState) -> dict:
    """Human-in-the-loop gate. interrupt() pauses the graph, persists state, and
    surfaces the questions; the run resumes when Command(resume=...) supplies answers."""
    spec = state["spec"]
    assert spec is not None
    answers = interrupt({"type": "clarifying_questions", "questions": spec.open_questions})
    return {"answers": answers}


def build_graph():
    b = StateGraph(GraphState)
    b.add_node("business_analyst", business_analyst)
    b.add_node("clarify", clarify)
    b.add_node("test_author", test_author)
    b.add_node("developer", developer)
    b.add_node("qa", qa)

    b.add_edge(START, "business_analyst")
    b.add_conditional_edges(
        "business_analyst", route_after_ba, {"clarify": "clarify", "test_author": "test_author"}
    )
    b.add_edge("clarify", "business_analyst")
    b.add_edge("test_author", "developer")
    b.add_edge("developer", "qa")
    b.add_conditional_edges("qa", route_after_qa, {"developer": "developer", "deliver": END})

    # A checkpointer is REQUIRED for interrupt()/resume.
    return b.compile(checkpointer=InMemorySaver(serde=_CHECKPOINT_SERDE))

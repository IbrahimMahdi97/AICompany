"""The orchestrator's routing is pure and deterministic, so it is fully unit-testable."""

from orchestration.contracts import AcceptanceCriterion, CriterionResult, QaVerdict, SolutionSpec
from orchestration.routing import route_after_ba, route_after_qa


def _spec(open_questions: list[str]) -> SolutionSpec:
    return SolutionSpec(
        requirements=[],
        acceptance_criteria=[AcceptanceCriterion(id="AC-1", text="x")],
        data_model="",
        api_contracts=["def f() -> None: ..."],
        open_questions=open_questions,
    )


def _state(**kw) -> dict:
    base = dict(
        goal="g", spec=None, test_files=[], impl_files=[], verdict=None,
        defects=[], answers=None, qa_round=0, max_qa_rounds=3, qa_log=[],
    )
    base.update(kw)
    return base


def _verdict(passed: bool) -> QaVerdict:
    return QaVerdict(
        passed=passed, results=[CriterionResult(id="AC-1", passed=passed, note="")], defects=[]
    )


def test_ba_routes_to_clarify_when_questions_and_no_answers():
    assert route_after_ba(_state(spec=_spec(["q1"]), answers=None)) == "clarify"


def test_ba_routes_to_test_author_when_no_questions():
    assert route_after_ba(_state(spec=_spec([]))) == "test_author"


def test_ba_proceeds_after_answers_supplied():
    assert route_after_ba(_state(spec=_spec(["q1"]), answers="answered")) == "test_author"


def test_qa_delivers_on_pass():
    assert route_after_qa(_state(verdict=_verdict(True), qa_round=1)) == "deliver"


def test_qa_loops_back_on_failure_within_budget():
    assert route_after_qa(_state(verdict=_verdict(False), qa_round=1, max_qa_rounds=3)) == "developer"


def test_qa_escalates_at_round_budget():
    assert route_after_qa(_state(verdict=_verdict(False), qa_round=3, max_qa_rounds=3)) == "deliver"

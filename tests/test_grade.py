"""Grading is the objective heart of QA, so it gets real unit tests.

These run without an API key or any heavy dependency -- only the standard library
and pydantic are imported.
"""

from orchestration.contracts import AcceptanceCriterion
from orchestration.sandbox import ExecResult, grade


def _res(xml: str, rc: int = 1, out: str = "", err: str = "") -> ExecResult:
    return ExecResult(returncode=rc, stdout=out, stderr=err,
                      artifacts={"report.xml": xml} if xml else {})


def test_passing_and_failing_criteria_are_scored_independently():
    xml = (
        '<testsuites><testsuite name="pytest" tests="2" failures="1">'
        '<testcase classname="test_spec" name="test_ac_1__ok"/>'
        '<testcase classname="test_spec" name="test_ac_2__bad">'
        '<failure message="assert 1 == 2">boom</failure></testcase>'
        "</testsuite></testsuites>"
    )
    criteria = [AcceptanceCriterion(id="AC-1", text="a"), AcceptanceCriterion(id="AC-2", text="b")]
    results, defects = grade(_res(xml, rc=1), criteria)
    by_id = {r.id: r for r in results}
    assert by_id["AC-1"].passed is True
    assert by_id["AC-2"].passed is False
    assert any("AC-2" in d for d in defects)


def test_uncovered_criterion_does_not_pass():
    xml = '<testsuites><testsuite><testcase name="test_ac_1__ok"/></testsuite></testsuites>'
    criteria = [AcceptanceCriterion(id="AC-1", text="a"), AcceptanceCriterion(id="AC-9", text="uncovered")]
    results, _ = grade(_res(xml, rc=0), criteria)
    by_id = {r.id: r for r in results}
    assert by_id["AC-1"].passed is True
    assert by_id["AC-9"].passed is False
    assert "no executable test" in by_id["AC-9"].note


def test_missing_report_is_a_failure_with_defects():
    criteria = [AcceptanceCriterion(id="AC-1", text="a")]
    results, defects = grade(_res("", rc=2, err="collection error"), criteria)
    assert all(not r.passed for r in results)
    assert defects


def test_timeout_is_reported_as_a_defect():
    xml = '<testsuites><testsuite><testcase name="test_ac_1__ok"/></testsuite></testsuites>'
    criteria = [AcceptanceCriterion(id="AC-1", text="a")]
    _, defects = grade(_res(xml, rc=124), criteria)
    assert any("timed out" in d.lower() for d in defects)

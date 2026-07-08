"""The agents.

    Agent 1  Business Analyst    -> business_analyst  (LLM)
    Agent 3a Product Owner / QA  -> test_author       (LLM, runs once from the spec)
    Agent 2  Full-Stack Dev      -> developer         (LLM)
    Agent 3b QA gate             -> qa                (NO LLM -- runs the tests, reads pass/fail)

LLM clients and the sandbox are built lazily so importing this module needs no API key
and triggers no docker probe (keeps tests and CI light).
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from .config import settings
from .contracts import Implementation, QaVerdict, SolutionSpec, TestSuite
from .routing import GraphState
from .sandbox import Sandbox, get_sandbox, grade, trim

SYSTEM_BA = (
    "You are the Business Analyst. You elicit and formalize requirements. You NEVER write "
    "code. In api_contracts, pin the EXACT public API the implementation must expose from "
    "solution.py. Put anything ambiguous or missing in open_questions."
)
SYSTEM_TEST_AUTHOR = (
    "You are the Product Owner / QA acceptance-test author. Turn each acceptance criterion "
    "into runnable pytest tests in ONE file, test_spec.py, importing the public API from "
    "`solution`. Cover EVERY criterion with at least one test. Name each test function "
    "test_<id>__<desc>, where <id> is the criterion id lowercased with dashes as underscores "
    "(AC-1 -> ac_1), e.g. test_ac_1__requires_email_confirmation. Standard library + pytest "
    "only -- no network, no external services; build any fakes/stubs inline. Tests must be "
    "deterministic."
)
SYSTEM_DEV = (
    "You are the Full-Stack Developer. Implement the FROZEN spec as runnable Python. The "
    "entry-point module MUST be solution.py and expose EXACTLY the API in the spec's "
    "api_contracts. Standard library only (the sandbox has no network). If failing-test "
    "feedback is provided, fix those failures WITHOUT changing the public API. Return files "
    "as {path, content}. You do not see the test source -- only the failures they produce."
)


@lru_cache(maxsize=1)
def _ba():
    return ChatAnthropic(model=settings.model_ba, max_tokens=4096).with_structured_output(SolutionSpec)


@lru_cache(maxsize=1)
def _test_author():
    return ChatAnthropic(model=settings.model_test, max_tokens=8192).with_structured_output(TestSuite)


@lru_cache(maxsize=1)
def _developer():
    return ChatAnthropic(model=settings.model_dev, max_tokens=8192).with_structured_output(Implementation)


@lru_cache(maxsize=1)
def _sandbox() -> Sandbox:
    return get_sandbox()


def business_analyst(state: GraphState) -> dict:  # Agent 1
    prompt = f"Business goal:\n{state['goal']}"
    if state.get("answers"):
        prompt += (
            "\n\nAnswers to your earlier open questions (fold in, clear the resolved "
            f"ones):\n{state['answers']}"
        )
    return {"spec": _ba().invoke([("system", SYSTEM_BA), ("human", prompt)])}


def test_author(state: GraphState) -> dict:  # Agent 3a
    spec = state["spec"]
    human = f"Write the pytest acceptance suite for this spec.\n\n{spec.model_dump_json(indent=2)}"
    suite = _test_author().invoke([("system", SYSTEM_TEST_AUTHOR), ("human", human)])
    return {"test_files": suite.files}


def developer(state: GraphState) -> dict:  # Agent 2
    spec = state["spec"]
    human = f"Implement this spec. Entry point must be solution.py.\n\n{spec.model_dump_json(indent=2)}"
    if state.get("defects"):
        human += "\n\nThe previous version FAILED these tests -- fix them:\n" + "\n".join(
            f"- {d}" for d in state["defects"]
        )
    impl = _developer().invoke([("system", SYSTEM_DEV), ("human", human)])
    return {"impl_files": impl.files}


def qa(state: GraphState) -> dict:  # Agent 3b -- deterministic execution gate
    files = {f.path: f.content for f in state["impl_files"]}
    for f in state["test_files"]:
        files[f.path] = f.content

    res = _sandbox().run_pytest(files, timeout=settings.sandbox_timeout)
    results, defects = grade(res, state["spec"].acceptance_criteria)
    passed = bool(results) and all(r.passed for r in results)

    verdict = QaVerdict(
        passed=passed,
        results=results,
        defects=defects,
        raw_output=trim(res.stdout + "\n" + res.stderr, 1500),
    )
    round_no = state["qa_round"] + 1
    log = state.get("qa_log", []) + [
        f"Round {round_no}: passed={passed}, "
        f"criteria={sum(r.passed for r in results)}/{len(results)}, defects={len(defects)}"
    ]
    return {"verdict": verdict, "qa_round": round_no, "defects": defects, "qa_log": log}

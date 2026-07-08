"""Command-line entry point: `orchestrate` or `python -m orchestration.main`."""

import argparse

DEFAULT_GOAL = (
    "Build a pure-Python member profile service: update_profile(member_id, changes) records "
    "edits but holds email changes as 'pending' until confirm_email(token) is called; "
    "get_profile(member_id) returns the current confirmed profile."
)


def run(goal: str, thread_id: str = "run-1") -> dict:
    from langgraph.types import Command

    from .config import settings
    from .graph import build_graph

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: dict = {
        "goal": goal,
        "spec": None,
        "test_files": [],
        "impl_files": [],
        "verdict": None,
        "defects": [],
        "answers": None,
        "qa_round": 0,
        "max_qa_rounds": settings.max_qa_rounds,
        "qa_log": [],
    }

    result = graph.invoke(initial, config)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n[Orchestrator] The Business Analyst needs clarification:")
        for q in payload["questions"]:
            print("  ?", q)
        result = graph.invoke(Command(resume=input("Your answers: ")), config)
    return result


def _print_summary(result: dict) -> None:
    verdict = result["verdict"]
    print("\n===== RESULT =====")
    if verdict.passed:
        print(f"DELIVERED after {result['qa_round']} QA round(s).")
    else:
        print(f"ESCALATED -- still failing after {result['qa_round']} rounds.")
    print("\nAcceptance criteria:")
    for r in verdict.results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.id}: {r.note}")
    if verdict.defects:
        print("\nOutstanding defects:")
        for d in verdict.defects:
            print("  -", d)
    print("\nQA log:\n  " + "\n  ".join(result["qa_log"]))
    print(f"\nFinal code: {[f.path for f in result['impl_files']]}")


def cli() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the BA/Developer/QA agent orchestration.")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="The business goal to build.")
    parser.add_argument("--thread-id", default="run-1", help="Checkpoint thread id for this run.")
    args = parser.parse_args()
    _print_summary(run(args.goal, args.thread_id))


if __name__ == "__main__":
    cli()

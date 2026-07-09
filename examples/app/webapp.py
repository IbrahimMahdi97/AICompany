"""Agent Studio: a web UI that takes a goal, runs the BA/Developer/QA build, and lets you
run the generated code -- all in the browser.

    python examples/app/webapp.py            # then open http://127.0.0.1:8000/

Flow: type a goal -> POST /run drives the LangGraph orchestration until it either needs
clarification (the analyst's questions come back to the UI) or finishes. Answering posts to
/resume. On completion the generated solution.py is saved (output/<thread-id>/), its public
functions are introspected, and the UI renders a form per function that calls /call.

Standard library HTTP server, binds to localhost. Requires ANTHROPIC_API_KEY in .env (the
build calls the Claude API). SECURITY: this runs model-generated code locally on input you
type -- run it on goals/output you're comfortable executing.
"""

from __future__ import annotations

# ruff: noqa: I001 -- imports are intentionally ordered after sys.path / .env bootstrap below
import argparse
import json
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root: examples/app/webapp.py -> ../..
sys.path.insert(0, str(HERE))              # for the sibling `serve` module
sys.path.insert(0, str(ROOT / "src"))      # for the `orchestration` package

import serve  # noqa: E402 -- path set up above

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.types import Command  # noqa: E402

from orchestration.config import settings  # noqa: E402
from orchestration.graph import build_graph  # noqa: E402
from orchestration.main import save_deliverables  # noqa: E402

# One graph for the whole process: its in-memory checkpointer keys runs by thread_id, which
# is what makes the interrupt/resume (clarifying-questions) flow work across HTTP requests.
GRAPH = build_graph()
_MODULES: dict[str, object] = {}  # thread_id -> loaded solution module, for /call


def _initial_state(goal: str) -> dict:
    return {
        "goal": goal, "spec": None, "test_files": [], "impl_files": [], "verdict": None,
        "defects": [], "answers": None, "qa_round": 0,
        "max_qa_rounds": settings.max_qa_rounds, "qa_log": [],
    }


def _step(result: dict, thread_id: str) -> dict:
    """Turn a graph result into a JSON response: questions, a finished summary, or done+funcs."""
    if "__interrupt__" in result:
        return {
            "status": "needs_input",
            "thread_id": thread_id,
            "questions": result["__interrupt__"][0].value["questions"],
        }

    out_dir = save_deliverables(result, thread_id)
    verdict = result["verdict"]
    functions: list = []
    sol = Path(out_dir) / "solution.py"
    if sol.is_file():
        try:
            mod = serve._load_module(str(out_dir), "solution")
            _MODULES[thread_id] = mod
            functions = serve.discover(mod)
        except Exception:  # noqa: BLE001 -- a broken solution shouldn't kill the response
            functions = []

    return {
        "status": "done",
        "delivered": verdict.passed,
        "rounds": result["qa_round"],
        "criteria": [{"id": r.id, "passed": r.passed, "note": r.note} for r in verdict.results],
        "defects": verdict.defects,
        "files": [f.path for f in result["impl_files"]],
        "output_dir": str(out_dir),
        "functions": functions,
    }


def _run(goal: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return _step(GRAPH.invoke(_initial_state(goal), config), thread_id)


def _resume(thread_id: str, answers: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return _step(GRAPH.invoke(Command(resume=answers), config), thread_id)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "studio.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        try:
            payload = self._json_body()
            if self.path == "/run":
                result = _run(payload["goal"], payload["thread_id"])
            elif self.path == "/resume":
                result = _resume(payload["thread_id"], payload.get("answers", "default for all"))
            elif self.path == "/call":
                mod = _MODULES.get(payload.get("thread_id", ""))
                if mod is None:
                    result = {"ok": False, "error": "This build's code is no longer loaded. Rebuild first."}
                else:
                    value = serve.call_function(mod, payload.get("func", ""), payload.get("args", {}))
                    result = {"ok": True, "result": serve._to_jsonable(value)}
            else:
                self._send(404, b'{"error":"not found"}')
                return
        except Exception as e:  # noqa: BLE001 -- surface any failure to the UI
            tail = "".join(traceback.format_exc().splitlines(keepends=True)[-4:])
            # this shape works for both /run,/resume (checks .status) and /call (checks .ok)
            err = {"status": "error", "ok": False, "error": f"{type(e).__name__}: {e}\n\n{tail}"}
            self._send(200, json.dumps(err).encode("utf-8"))
            return
        self._send(200, json.dumps(result).encode("utf-8"))

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Agent Studio web UI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Agent Studio at {url}  (Ctrl-C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

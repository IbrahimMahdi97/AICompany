# Agent Orchestration (BA / Developer / QA)

A small, production-shaped multi-agent system built on [LangGraph](https://langchain-ai.github.io/langgraph/).
A **Business Analyst** turns a goal into a spec, a **Product Owner / QA** turns that spec
into executable acceptance tests, a **Developer** writes real code, and a **QA gate** grades
the code by actually running those tests in a sandbox. The **orchestrator is not an agent** —
it is the graph's control plane (routing + a human-in-the-loop gate) and never calls an LLM,
so it can't be entangled with QA's judgment.

## Architecture

```
START → business_analyst ─┬─(open questions)→ clarify → business_analyst
                          └─(final spec)─────→ test_author → developer → qa ─┬─(all pass)──→ END
                                              (writes suite once)            ├─(rounds≥max)→ END (escalate)
                                                                             └─(failures)──→ developer
```

| Role | Node | LLM? | Responsibility |
|------|------|------|----------------|
| Agent 1 — Business Analyst | `business_analyst` | yes | Formalize requirements; pin the exact API in `api_contracts`; raise `open_questions`. |
| Agent 3a — Product Owner / QA | `test_author` | yes | Turn each acceptance criterion into a pytest (runs once from the frozen spec). |
| Agent 2 — Full-Stack Developer | `developer` | yes | Implement the spec as real files; fix failing tests each round. |
| Agent 3b — QA gate | `qa` | **no** | Run the tests in a sandbox and read pass/fail. Deterministic. |
| Orchestrator | routers + `clarify` + runtime | **no** | Route work, run the human gate, bound the retry loop. |

QA's verdict is grounded in execution, not opinion: a criterion passes only if it has at
least one test and all of its tests pass.

## ⚠️ Security

The `developer` node emits code written by a model. **Running it is inherently risky.**
- `DockerSandbox` is the real boundary: a throwaway container with `--network none`, a
  non-root user, and memory/PID/CPU caps.
- `LocalSubprocessSandbox` is a convenience for local development **only** and is **not**
  isolation — model code can read your filesystem and reach the network from it.

Set `SANDBOX_BACKEND=docker` (or leave it `auto`, which prefers Docker when present) for
anything that isn't a throwaway experiment.

## Requirements

- Python 3.11+
- An Anthropic API key
- Docker (recommended, for the real sandbox)

## Setup (Windows / PowerShell)

```powershell
cd E:\AI\orchestraiton

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"

Copy-Item .env.example .env
# then edit .env and set ANTHROPIC_API_KEY

# Build the sandbox image (needs Docker Desktop):
docker build -f Dockerfile.sandbox -t agent-sandbox:py312 .
```

(macOS/Linux: `source .venv/bin/activate`, `cp .env.example .env`.)

## Run

```powershell
# uses the default demo goal
python -m orchestration.main

# or provide your own
python -m orchestration.main --goal "Build a URL shortener with expiry and click counts."
```

If the analyst raises clarifying questions, the run pauses and prompts you in the console;
your answer resumes it from exactly where it stopped (state is checkpointed).

## How QA grades

`test_author` names each test `test_<criterion-id>__<description>`, e.g. `test_ac_1__requires_email_confirmation`.
The `qa` node runs the suite, writes JUnit XML, and `grade()` buckets each result back to its
criterion by that prefix. This is what makes the pass/fail objective and per-criterion.

## Project layout

```
src/orchestration/
  contracts.py   Pydantic artifacts passed between stages
  config.py      Environment-driven settings
  sandbox.py     Docker + local sandboxes and the grade() scorer
  routing.py     Graph state + pure routing (the orchestrator's decisions)
  agents.py      The four agent nodes (LLMs built lazily)
  graph.py       Graph wiring + the clarify (human-in-the-loop) node
  main.py        CLI entry point
tests/           Deterministic unit tests for grade() and routing (no API key needed)
```

## Testing & CI

```powershell
ruff check .
mypy src/orchestration/contracts.py src/orchestration/sandbox.py src/orchestration/routing.py src/orchestration/config.py
pytest -q
```

The unit tests cover the deterministic core (grading and routing) and require no API key,
so CI stays green without secrets. See `.github/workflows/ci.yml`.

## Extending

- Give the developer a real project layout (multiple modules, a package) instead of flat files.
- Add a lint/type gate before the tests so QA isn't the first thing to catch a syntax error.
- Swap `InMemorySaver` for `SqliteSaver`/`PostgresSaver` so a failed run can resume.
- Add a contract-check node that fails fast if `solution.py` and the tests disagree on the API.

## Push to GitHub

Create an **empty** repo on GitHub first (no README/license), then:

```powershell
cd E:\AI\orchestraiton
git init
git add .
git commit -m "Initial commit: BA/Developer/QA agent orchestration"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## License

MIT — see [LICENSE](LICENSE).

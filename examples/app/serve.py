"""Auto-UI web runner: turn ANY generated solution.py into a runnable web app.

The orchestrator produces a `solution.py` exposing some public functions (the API from the
spec's api_contracts). This runner imports that module, discovers its public functions, and
serves a browser UI with one form per function -- you fill in the arguments, click Run, and
see the return value. It works for any goal, not just the calculator, because it reflects
whatever functions the generated module happens to expose.

Standard library only (no install), binds to localhost.

    python examples/app/serve.py output/run-1          # point at a generated run
    python examples/app/serve.py path/to/solution.py   # or a file directly
    python examples/app/serve.py output/run-1 --port 8080 --no-open

SECURITY: this imports and executes model-generated code and calls it with input you type.
Run it locally on code you've looked at -- it binds to 127.0.0.1 only and is not an
internet-facing service.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _load_module(target: str, module_name: str):
    """Import ``<module_name>.py`` from a directory or an explicit .py file path."""
    p = Path(target).resolve()
    file = p / f"{module_name}.py" if p.is_dir() else p
    if not file.is_file():
        raise SystemExit(f"Could not find {file}. Point me at a generated run dir or a .py file.")
    sys.path.insert(0, str(file.parent))  # let the module import sibling files
    spec = importlib.util.spec_from_file_location(file.stem, file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {file}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the module's top-level code
    return mod


def _ann_str(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return ""
    return getattr(ann, "__name__", None) or str(ann).replace("typing.", "")


def discover(mod) -> list[dict]:
    """Public top-level functions defined in the module, with their parameters."""
    funcs = []
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_") or getattr(fn, "__module__", None) != mod.__name__:
            continue
        params = []
        for pname, p in inspect.signature(fn).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            params.append({
                "name": pname,
                "annotation": _ann_str(p.annotation),
                "required": p.default is inspect.Parameter.empty,
                "default": None if p.default is inspect.Parameter.empty else repr(p.default),
            })
        funcs.append({"name": name, "doc": inspect.getdoc(fn) or "", "params": params})
    return sorted(funcs, key=lambda f: f["name"])


def _coerce(raw: str, annotation: Any):
    """Turn a form string into a typed argument, guided by the annotation."""
    raw = raw.strip()
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if annotation is str:
        return raw
    # dict / list / unannotated: accept JSON (5, [1,2], {"a":1}, true), else a plain string
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def call_function(mod, func_name: str, args: dict[str, str]) -> Any:
    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"No such function: {func_name}")
    sig = inspect.signature(fn)
    kwargs = {}
    for pname, p in sig.parameters.items():
        if pname not in args or args[pname] == "":
            if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise ValueError(f"Missing required argument: {pname}")
            continue
        kwargs[pname] = _coerce(args[pname], p.annotation)
    return fn(**kwargs)


def _to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #0b1220; color: #e2e8f0; }
  @media (prefers-color-scheme: light) { body { background: #eef2f7; color: #0f172a; } }
  header { padding: 22px 24px; border-bottom: 1px solid rgba(148,163,184,.25); }
  header h1 { margin: 0; font-size: 20px; }
  header .sub { color: #94a3b8; font-size: 13px; margin-top: 4px; }
  main { max-width: 760px; margin: 0 auto; padding: 20px 16px 60px; display: grid; gap: 18px; }
  .card { background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.22);
    border-radius: 16px; padding: 18px; }
  .fn { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 15px; font-weight: 600; }
  .doc { color: #94a3b8; font-size: 13px; white-space: pre-wrap; margin: 6px 0 14px; }
  .row { display: grid; grid-template-columns: 160px 1fr; gap: 10px;
    align-items: center; margin-bottom: 8px; }
  .row label { font-family: ui-monospace, monospace; font-size: 13px; }
  .row .ann { color: #64748b; font-weight: 400; }
  input { width: 100%; padding: 9px 11px; border-radius: 9px; border: 1px solid rgba(148,163,184,.35);
    background: rgba(255,255,255,.04); color: inherit; font-size: 14px;
    font-family: ui-monospace, monospace; }
  input:focus { outline: 2px solid #14b8a6; border-color: transparent; }
  button { margin-top: 6px; padding: 9px 18px; border: none; border-radius: 10px; cursor: pointer;
    background: #14b8a6; color: #05201d; font-weight: 700; font-size: 14px; }
  button:hover { background: #2dd4bf; }
  .out { margin-top: 12px; padding: 12px 14px; border-radius: 10px; font-family: ui-monospace, monospace;
    font-size: 13px; white-space: pre-wrap; word-break: break-word; display: none; }
  .out.ok { display: block; background: rgba(20,184,166,.12); border: 1px solid rgba(20,184,166,.4); }
  .out.err { display: block; background: rgba(248,113,113,.12);
    border: 1px solid rgba(248,113,113,.5); color: #fca5a5; }
  .empty { color: #94a3b8; }
</style></head>
<body>
  <header><h1>__TITLE__</h1><div class="sub">__SUB__</div></header>
  <main id="app"></main>
<script>
const FUNCS = /*FUNCS*/[];
const app = document.getElementById("app");

if (FUNCS.length === 0) {
  app.innerHTML = '<div class="card empty">No public functions found in the module.</div>';
}
for (const f of FUNCS) {
  const card = document.createElement("div");
  card.className = "card";
  const sig = f.params.map(p => p.name + (p.annotation ? ": " + p.annotation : "")).join(", ");
  card.innerHTML = '<div class="fn">' + f.name + "(" + sig + ")</div>"
    + (f.doc ? '<div class="doc">' + escapeHtml(f.doc) + "</div>" : "");
  const rows = [];
  for (const p of f.params) {
    const ann = p.annotation ? ' <span class="ann">' + p.annotation + "</span>" : "";
    const ph = p.required ? "required" : ("optional, default " + (p.default ?? "None"));
    rows.push('<div class="row"><label>' + p.name + ann + "</label>"
      + '<input data-fn="' + f.name + '" data-p="' + p.name + '" placeholder="' + ph + '"></div>');
  }
  const body = document.createElement("div");
  body.innerHTML = rows.join("")
    + '<button data-run="' + f.name + '">Run</button>'
    + '<div class="out" id="out-' + f.name + '"></div>';
  card.appendChild(body);
  app.appendChild(card);
}

app.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-run]");
  if (!btn) return;
  const name = btn.dataset.run;
  const args = {};
  document.querySelectorAll('input[data-fn="' + CSS.escape(name) + '"]')
    .forEach(i => { args[i.dataset.p] = i.value; });
  const out = document.getElementById("out-" + name);
  out.className = "out"; out.textContent = "...";
  try {
    const res = await fetch("/call", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ func: name, args }),
    });
    const data = await res.json();
    if (data.ok) { out.className = "out ok"; out.textContent = format(data.result); }
    else { out.className = "out err"; out.textContent = data.error; }
  } catch (err) { out.className = "out err"; out.textContent = String(err); }
});

function format(v) { return typeof v === "string" ? v : JSON.stringify(v, null, 2); }
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
</script>
</body></html>
"""


def make_handler(mod, funcs, title, sub):
    page = (
        PAGE.replace("__TITLE__", title)
        .replace("__SUB__", sub)
        .replace("/*FUNCS*/[]", json.dumps(funcs))
    )
    page_bytes = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page_bytes, "text/html; charset=utf-8")
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/call":
                self._send(404, b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = call_function(mod, payload.get("func", ""), payload.get("args", {}))
                body = json.dumps({"ok": True, "result": _to_jsonable(result)}).encode("utf-8")
            except Exception as e:  # noqa: BLE001 -- surface any call error to the UI
                tail = "".join(traceback.format_exc().splitlines(keepends=True)[-4:])
                msg = f"{type(e).__name__}: {e}\n\n{tail}"
                body = json.dumps({"ok": False, "error": msg}).encode("utf-8")
            self._send(200, body)

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a generated solution.py as a web app.")
    parser.add_argument("target", help="Path to a generated run dir (output/<thread-id>) or a .py file.")
    parser.add_argument("--module", default="solution", help="Module name to load (default: solution).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Don't open a browser automatically.")
    args = parser.parse_args()

    mod = _load_module(args.target, args.module)
    funcs = discover(mod)
    title = f"{args.module}.py — {len(funcs)} function(s)"
    sub = f"Loaded from {args.target}. Fill in arguments and click Run."
    handler = make_handler(mod, funcs, title, sub)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {args.module}.py ({len(funcs)} function(s)) at {url}")
    print("Functions:", ", ".join(f["name"] for f in funcs) or "(none found)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

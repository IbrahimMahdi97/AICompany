"""Isolated execution of model-written code, plus deterministic grading of the result.

SECURITY: the developer node emits code written by a model. Running it is inherently
risky. DockerSandbox (ephemeral container, no network, non-root, resource caps) is the
only real boundary here. LocalSubprocessSandbox is a convenience for local development
ONLY -- it is NOT isolation: model code can read your filesystem and reach the network.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# report.xml is produced by model-written tests running in the sandbox, so it is
# untrusted input. defusedxml rejects entity-expansion ("billion laughs") and external
# entity attacks that the stdlib parser would happily process.
from defusedxml.ElementTree import fromstring as xml_fromstring

from .config import settings
from .contracts import AcceptanceCriterion, CriterionResult

# pytest is invoked identically in every backend.
_PYTEST_ARGS = ["-q", "--tb=short", "-p", "no:cacheprovider", "--junitxml=report.xml", "test_spec.py"]


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    artifacts: dict[str, str] = field(default_factory=dict)


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, rejecting absolute paths and ``..`` traversal.

    ``rel`` comes from model-generated CodeFile.path and is untrusted. This runs on the
    HOST (before any container mount), so an absolute path or ``../`` would otherwise let
    model output write anywhere on the host filesystem.
    """
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"unsafe file path escapes sandbox root: {rel!r}")
    return target


def _write_files(root: str, files: dict[str, str]) -> None:
    root_path = Path(root)
    for rel, content in files.items():
        p = _safe_join(root_path, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def trim(s: str, n: int = 800) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def _as_text(v: bytes | str | None) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else v.decode(errors="replace")


class Sandbox(Protocol):
    def run_pytest(self, files: dict[str, str], timeout: int = 60) -> ExecResult: ...


class LocalSubprocessSandbox:
    """Convenience only -- NOT a security boundary."""

    def run_pytest(self, files: dict[str, str], timeout: int = 60) -> ExecResult:
        with tempfile.TemporaryDirectory() as d:
            _write_files(d, files)
            cmd = [sys.executable, "-m", "pytest", *_PYTEST_ARGS]
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
            try:
                p = subprocess.run(
                    cmd, cwd=d, capture_output=True, text=True, timeout=timeout, env=env
                )
                rc, out, err = p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                rc = 124
                out = _as_text(e.stdout)
                err = _as_text(e.stderr) + f"\nTIMEOUT after {timeout}s"
            report = Path(d) / "report.xml"
            xml = report.read_text(encoding="utf-8") if report.exists() else ""
        return ExecResult(rc, out, err, {"report.xml": xml})


class DockerSandbox:
    """The real answer: a throwaway container with no network, non-root, capped resources.

    Requires an image with pytest baked in (network is off at run time):
        docker build -f Dockerfile.sandbox -t agent-sandbox:py312 .
    """

    def __init__(self, image: str | None = None):
        self.image = image or settings.sandbox_image

    def run_pytest(self, files: dict[str, str], timeout: int = 60) -> ExecResult:
        with tempfile.TemporaryDirectory() as d:
            _write_files(d, files)
            os.chmod(d, 0o777)  # let the non-root container user write report.xml
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "512m", "--pids-limit", "256", "--cpus", "1",
                "--user", "1000:1000",
                "-v", f"{d}:/work", "-w", "/work",
                self.image,
                "python", "-m", "pytest", *_PYTEST_ARGS,
            ]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 20)
                rc, out, err = p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired:
                rc, out, err = 124, "", f"TIMEOUT after {timeout}s (container killed)"
            report = Path(d) / "report.xml"
            xml = report.read_text(encoding="utf-8") if report.exists() else ""
        return ExecResult(rc, out, err, {"report.xml": xml})


def get_sandbox() -> Sandbox:
    backend = settings.sandbox_backend
    if backend == "local":
        return LocalSubprocessSandbox()
    if backend == "docker" or (backend == "auto" and shutil.which("docker")):
        return DockerSandbox()
    print(
        "!! docker not found -- falling back to LocalSubprocessSandbox (NOT isolated). "
        "Do not run untrusted code this way in production."
    )
    return LocalSubprocessSandbox()


def grade(
    res: ExecResult, criteria: list[AcceptanceCriterion]
) -> tuple[list[CriterionResult], list[str]]:
    """Map real test results back to acceptance criteria via the naming convention
    (a test named test_ac_1__* covers criterion AC-1). Pure and deterministic."""
    defects: list[str] = []
    xml = res.artifacts.get("report.xml", "")
    if not xml:
        defects.append("Test run produced no results:\n" + trim(res.stdout + "\n" + res.stderr))
        return [CriterionResult(id=c.id, passed=False, note="no test results") for c in criteria], defects

    cases: list[tuple[str, bool, str]] = []  # (name, passed, message)
    for tc in xml_fromstring(xml).iter("testcase"):
        name = tc.get("name", "")
        bad = tc.find("failure")
        if bad is None:
            bad = tc.find("error")
        if bad is not None:
            msg = (bad.get("message", "") or "") + "\n" + (bad.text or "")
            cases.append((name, False, trim(msg, 400)))
        else:
            cases.append((name, True, ""))

    results: list[CriterionResult] = []
    for c in criteria:
        prefix = "test_" + c.id.lower().replace("-", "_") + "__"
        mine = [x for x in cases if x[0].startswith(prefix)]
        if not mine:
            results.append(
                CriterionResult(id=c.id, passed=False, note="no executable test covers this criterion")
            )
            continue
        failed = [x for x in mine if not x[1]]
        if failed:
            results.append(
                CriterionResult(id=c.id, passed=False, note=f"{len(failed)}/{len(mine)} test(s) failing")
            )
            defects += [f"[{c.id}] {name}: {msg}" for name, _, msg in failed]
        else:
            results.append(CriterionResult(id=c.id, passed=True, note=f"{len(mine)} test(s) passing"))

    if res.returncode == 124:
        defects.append("Test run timed out -- likely an infinite loop in the implementation.")
    elif res.returncode not in (0, 1):
        defects.append("pytest collection/setup error:\n" + trim(res.stdout + "\n" + res.stderr))

    return results, defects

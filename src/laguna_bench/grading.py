from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .tasks import Task


def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return max(matches, key=len).strip() if matches else text.strip()


def grade_generation(task: Task, output: str) -> dict[str, Any]:
    if task.grader == "exact_json":
        try:
            parsed = json.loads(output.strip())
        except json.JSONDecodeError as exc:
            return {"score": 0.0, "passed": 0, "total": 1, "detail": f"invalid JSON: {exc}"}
        expected = {"items": ["birch-tree", "cedar", "maple"], "count": 3}
        passed = parsed == expected
        return {"score": float(passed), "passed": int(passed), "total": 1, "detail": "exact match" if passed else f"got {parsed!r}"}
    if task.grader == "python":
        return _grade_python(task, extract_code(output))
    raise ValueError(f"unsupported generation grader: {task.grader}")


def grade_workspace(task: Task, workspace: Path) -> dict[str, Any]:
    if task.grader == "agent_invoice":
        answer = workspace / "answer.txt"
        value = answer.read_text().strip() if answer.exists() else ""
        passed = value == "INV-1042"
        return {"score": float(passed), "passed": int(passed), "total": 1, "detail": f"answer={value!r}"}
    if task.grader == "pytest":
        return run_pytest(workspace, task.test_count)
    raise ValueError(f"unsupported workspace grader: {task.grader}")


def _grade_python(task: Task, code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="laguna-generation-") as tmp:
        root = Path(tmp)
        (root / "candidate.py").write_text(code + "\n")
        for path, content in task.fixture.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return run_pytest(root, task.test_count)


def run_pytest(workspace: Path, total: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONHASHSEED": "0"},
        )
        output = proc.stdout[-12000:]
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "passed": 0, "total": total, "detail": "pytest timed out"}
    passed_matches = re.findall(r"(\d+) passed", output)
    passed = int(passed_matches[-1]) if passed_matches else 0
    score = min(passed / total, 1.0) if total else float(proc.returncode == 0)
    return {"score": score, "passed": passed, "total": total, "detail": output, "exit_code": proc.returncode}

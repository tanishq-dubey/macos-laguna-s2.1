from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import run_agent
from .grading import grade_generation, grade_workspace
from .tasks import Task


def run_suite(
    backend,
    tasks: list[Task],
    output_root: Path,
    *,
    agent_prompt_cache: bool = False,
) -> tuple[Path, dict[str, Any]]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = backend.metadata()["model_id"].replace("/", "--")
    run_dir = output_root / f"{stamp}-{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "outputs").mkdir()
    (run_dir / "workspaces").mkdir()
    record: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_metadata(),
        "backend": backend.metadata(),
        "settings": {
            "temperature": getattr(backend, "temperature", 0.0),
            "top_p": getattr(backend, "top_p", 1.0),
            "seed": getattr(backend, "seed", None),
            "agent_prompt_cache": agent_prompt_cache,
        },
        "tasks": [],
    }
    for task in tasks:
        started = time.perf_counter()
        if task.kind == "generation":
            generation = backend.generate(
                [{"role": "system", "content": "Follow the user's output format exactly."}, {"role": "user", "content": task.prompt}],
                max_tokens=task.max_tokens,
            )
            (run_dir / "outputs" / f"{task.id}.txt").write_text(generation.text)
            grade = grade_generation(task, generation.text)
            result = {"id": task.id, "kind": task.kind, "tier": task.tier, "grade": grade, "metrics": generation.to_dict()}
        else:
            with tempfile.TemporaryDirectory(prefix=f"laguna-{task.id}-") as tmp:
                workspace = Path(tmp)
                materialize_fixture(task, workspace)
                agent_result = run_agent(backend, task, workspace, use_prompt_cache=agent_prompt_cache)
                grade = grade_workspace(task, workspace)
                shutil.copytree(workspace, run_dir / "workspaces" / task.id, dirs_exist_ok=True)
            (run_dir / "outputs" / f"{task.id}.json").write_text(json.dumps(agent_result["transcript"], indent=2, ensure_ascii=False))
            result = {"id": task.id, "kind": task.kind, "tier": task.tier, "grade": grade, "agent": {k: v for k, v in agent_result.items() if k != "transcript"}}
        result["wall_seconds"] = time.perf_counter() - started
        record["tasks"].append(result)
        _write_record(run_dir, record)
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    scores = [item["grade"]["score"] for item in record["tasks"]]
    record["aggregate_score"] = sum(scores) / len(scores) if scores else 0.0
    _write_record(run_dir, record)
    return run_dir, record


def materialize_fixture(task: Task, workspace: Path) -> None:
    for relative, content in task.fixture.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def machine_metadata() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(command("sysctl", "-n", "hw.memsize") or 0),
        "python": platform.python_version(),
    }


def _write_record(run_dir: Path, record: dict[str, Any]) -> None:
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    lines = [
        f"# {record['backend']['model_id']}",
        "",
        f"Load time: {record['backend'].get('load_seconds', 0):.2f}s",
        "",
        "| Task | Score | Passed | Time | Gen tok/s | Peak GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in record["tasks"]:
        metrics = item.get("metrics") or item.get("agent", {}).get("metrics", {})
        tps = metrics.get("generation_tps", metrics.get("effective_generation_tps", 0.0))
        lines.append(
            f"| {item['id']} | {item['grade']['score']:.3f} | {item['grade']['passed']}/{item['grade']['total']} | "
            f"{item['wall_seconds']:.2f}s | {tps:.2f} | {metrics.get('peak_memory_gb', 0.0):.2f} |"
        )
    if "aggregate_score" in record:
        lines.extend(["", f"Aggregate score: **{record['aggregate_score']:.3f}**"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

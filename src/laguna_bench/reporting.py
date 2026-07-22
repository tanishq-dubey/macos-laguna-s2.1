from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tasks import TASKS


def build_comparison(output_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    expected = {task.id for task in TASKS}
    sweep_memory: dict[str, tuple[float, str]] = {}
    for path in sorted((output_root / "sweeps").glob("*.json")):
        try:
            sweep = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model_id = sweep.get("backend", {}).get("model_id")
        cases = sweep.get("cases", [])
        peak = max((float(case.get("peak_memory_gb") or 0.0) for case in cases), default=0.0)
        if model_id and peak:
            metric = next((case.get("memory_metric") for case in cases if case.get("memory_metric")), "mlx_peak")
            sweep_memory[model_id] = peak, metric
    candidates: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(output_root.glob("*/run.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if {item.get("id") for item in record.get("tasks", [])} != expected:
            continue
        settings = record.get("settings", {})
        if settings.get("temperature", 0.0) != 0.0 or settings.get("top_p", 1.0) != 1.0:
            continue
        model_id = record.get("backend", {}).get("model_id")
        if model_id:
            backend = record.get("backend", {})
            # Runs produced before engine metadata was added were necessarily
            # the original in-process mlx-vlm backend.
            engine = backend.get("engine") or ("mlx-vlm" if backend.get("mlx_vlm_version") else "unknown")
            candidates[(model_id, engine)] = (path, record)

    rows: list[dict[str, Any]] = []
    for (model_id, engine), (path, record) in sorted(candidates.items()):
        generation = [item for item in record["tasks"] if item["kind"] == "generation"]
        agents = [item for item in record["tasks"] if item["kind"] == "agentic"]
        gen_tokens = sum(item["metrics"].get("generation_tokens", 0) for item in generation)
        gen_seconds = sum(item["metrics"].get("elapsed_seconds", 0) for item in generation)
        all_metrics = [item.get("metrics") or item.get("agent", {}).get("metrics", {}) for item in record["tasks"]]
        peak_memory = max((metrics.get("peak_memory_gb", 0.0) for metrics in all_metrics), default=0.0)
        memory_metric = "mlx_peak" if peak_memory else ""
        if not peak_memory and model_id in sweep_memory:
            peak_memory, memory_metric = sweep_memory[model_id]
        rows.append(
            {
                "model": model_id,
                "engine": engine,
                "revision": record["backend"].get("resolved_revision") or record["backend"].get("revision"),
                "score": record.get("aggregate_score", 0.0),
                "generation_score": sum(item["grade"]["score"] for item in generation) / len(generation),
                "agentic_score": sum(item["grade"]["score"] for item in agents) / len(agents),
                "generation_tps": gen_tokens / gen_seconds if gen_seconds else 0.0,
                "peak_memory_gb": peak_memory,
                "memory_metric": memory_metric,
                "load_seconds": record["backend"].get("load_seconds", 0.0),
                "run": str(path.parent),
            }
        )
    destination = output_root / "comparison.md"
    lines = [
        "# Laguna S 2.1 local comparison",
        "",
        "Latest complete six-task run per model.",
        "",
        "| Model | Engine | Score | Gen | Agent | Gen tok/s | Peak GB | Memory metric | Load s |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['engine']} | {row['score']:.3f} | {row['generation_score']:.3f} | "
            f"{row['agentic_score']:.3f} | {row['generation_tps']:.2f} | {row['peak_memory_gb']:.2f} | "
            f"{row['memory_metric']} | {row['load_seconds']:.2f} |"
        )
    destination.write_text("\n".join(lines) + "\n")
    return destination, rows

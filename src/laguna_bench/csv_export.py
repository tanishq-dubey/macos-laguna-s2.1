from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .catalog import QUANTS


FIELDS = [
    "record_type",
    "timestamp",
    "model_id",
    "source_repo",
    "model_file",
    "engine",
    "revision",
    "model_bytes",
    "quantization",
    "mlx_version",
    "mlx_lm_version",
    "mlx_vlm_version",
    "jang_version",
    "jang_revision",
    "transformers_version",
    "loader_file",
    "loader_sha256",
    "fix_mistral_regex",
    "wired_limit_bytes",
    "platform",
    "macos_version",
    "python_version",
    "chip",
    "memory_bytes",
    "artifact",
    "profile",
    "case_id",
    "category",
    "task_kind",
    "tier",
    "status",
    "error",
    "recommended",
    "score",
    "passed",
    "total",
    "agent_finished",
    "agent_steps",
    "requested_context_tokens",
    "actual_context_tokens",
    "max_new_tokens",
    "temperature",
    "top_p",
    "kv_bits",
    "kv_quant_scheme",
    "max_kv_size",
    "prefill_step_size",
    "needle_pass",
    "prompt_tokens",
    "generation_tokens",
    "cached_tokens",
    "prompt_tps",
    "generation_tps",
    "peak_memory_gb",
    "memory_metric",
    "elapsed_seconds",
    "wall_seconds",
    "finish_reason",
    "output_sha256",
]


def _backend_fields(record: dict[str, Any], path: Path) -> dict[str, Any]:
    backend = record.get("backend", {})
    machine = record.get("machine", {})
    engine = backend.get("engine") or ("mlx-vlm" if backend.get("mlx_vlm_version") else "unknown")
    quantization = backend.get("quantization")
    return {
        "timestamp": record.get("started_at"),
        "model_id": backend.get("model_id"),
        "model_file": backend.get("model_file"),
        "engine": engine,
        "revision": backend.get("resolved_revision") or backend.get("requested_revision") or backend.get("revision"),
        "model_bytes": backend.get("model_bytes"),
        "quantization": json.dumps(quantization, sort_keys=True) if isinstance(quantization, (dict, list)) else quantization,
        "mlx_version": backend.get("mlx_version"),
        "mlx_lm_version": backend.get("mlx_lm_version"),
        "mlx_vlm_version": backend.get("mlx_vlm_version"),
        "jang_version": backend.get("jang_version"),
        "jang_revision": backend.get("jang_revision"),
        "transformers_version": backend.get("transformers_version"),
        "loader_file": backend.get("loader_file"),
        "loader_sha256": backend.get("loader_sha256"),
        "fix_mistral_regex": backend.get("fix_mistral_regex"),
        "wired_limit_bytes": backend.get("wired_limit_bytes"),
        "platform": machine.get("platform"),
        "macos_version": machine.get("macos"),
        "python_version": machine.get("python"),
        "chip": machine.get("chip"),
        "memory_bytes": machine.get("memory_bytes"),
        "artifact": str(path),
    }


def _task_rows(output_root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(output_root.glob("*/run.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        common = _backend_fields(record, path)
        for task in record.get("tasks", []):
            agent = task.get("agent", {})
            metrics = task.get("metrics") or agent.get("metrics", {})
            grade = task.get("grade", {})
            yield {
                **common,
                "record_type": "task",
                "case_id": task.get("id"),
                "task_kind": task.get("kind"),
                "tier": task.get("tier"),
                "status": "ok",
                "score": grade.get("score"),
                "passed": grade.get("passed"),
                "total": grade.get("total"),
                "agent_finished": agent.get("finished"),
                "agent_steps": agent.get("steps"),
                "temperature": record.get("settings", {}).get("temperature"),
                "top_p": record.get("settings", {}).get("top_p", 1.0),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "generation_tokens": metrics.get("generation_tokens"),
                "cached_tokens": metrics.get("cached_tokens"),
                "prompt_tps": metrics.get("prompt_tps"),
                "generation_tps": metrics.get("generation_tps") or metrics.get("effective_generation_tps"),
                "peak_memory_gb": metrics.get("peak_memory_gb"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "wall_seconds": task.get("wall_seconds"),
                "finish_reason": metrics.get("finish_reason"),
            }


def _sweep_rows(output_root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((output_root / "sweeps").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        common = _backend_fields(record, path)
        for case in record.get("cases", []):
            yield {
                **common,
                "record_type": "performance",
                "profile": record.get("profile"),
                "case_id": case.get("id"),
                "category": case.get("category"),
                "status": case.get("status"),
                "error": case.get("error"),
                "requested_context_tokens": case.get("context_tokens"),
                "actual_context_tokens": case.get("actual_context_tokens"),
                "max_new_tokens": case.get("max_new_tokens"),
                "temperature": case.get("temperature"),
                "top_p": case.get("top_p"),
                "kv_bits": case.get("kv_bits"),
                "kv_quant_scheme": case.get("kv_quant_scheme"),
                "max_kv_size": case.get("max_kv_size"),
                "prefill_step_size": case.get("prefill_step_size"),
                "needle_pass": case.get("needle_pass"),
                "prompt_tokens": case.get("prompt_tokens"),
                "generation_tokens": case.get("generation_tokens"),
                "cached_tokens": case.get("cached_tokens"),
                "prompt_tps": case.get("prompt_tps"),
                "generation_tps": case.get("generation_tps"),
                "peak_memory_gb": case.get("peak_memory_gb"),
                "memory_metric": case.get("memory_metric"),
                "elapsed_seconds": case.get("elapsed_seconds"),
                "finish_reason": case.get("finish_reason"),
                "output_sha256": case.get("output_sha256"),
            }


def _catalog_rows() -> Iterable[dict[str, Any]]:
    for quant in QUANTS:
        yield {
            "record_type": "variant",
            "model_id": quant["model"],
            "source_repo": quant.get("repo", quant["model"]),
            "model_file": quant.get("file"),
            "engine": quant.get("engine"),
            "model_bytes": int(quant["size_gib"] * 1024**3) if quant["size_gib"] else None,
            "quantization": quant["bits"],
            "status": quant["status"],
            "recommended": quant["recommended"],
        }


def _failure_rows(output_root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((output_root / "failures").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        error = " ".join(str(record.get("error") or "").split())
        yield {
            **_backend_fields(record, path),
            "record_type": "failure",
            "profile": record.get("command"),
            "status": "error",
            "error": f"{record.get('error_type')}: {error}",
        }


def _existing_result_rows(destination: Path) -> Iterable[dict[str, Any]]:
    if not destination.exists():
        return
    try:
        with destination.open(newline="") as handle:
            for row in csv.DictReader(handle):
                # Catalog entries come from source so stale metadata is replaced.
                if row.get("record_type") != "variant":
                    yield row
    except (OSError, csv.Error):
        return


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    def value(field: str) -> str:
        return "" if row.get(field) is None else str(row.get(field))

    record_type = value("record_type")
    if record_type == "variant":
        return record_type, value("model_id"), value("model_file"), value("quantization")
    if value("artifact"):
        return record_type, value("artifact"), value("case_id")
    return tuple(value(field) for field in FIELDS)


def export_csv(output_root: Path) -> tuple[Path, int]:
    destination = output_root / "laguna_s21_full_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = (
        list(_catalog_rows())
        + list(_existing_result_rows(destination))
        + list(_task_rows(output_root))
        + list(_sweep_rows(output_root))
        + list(_failure_rows(output_root))
    )
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in candidates:
        # Locally available artifacts come last and refresh older CSV rows.
        merged[_row_key(row)] = row
    rows = list(merged.values())
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination, len(rows)

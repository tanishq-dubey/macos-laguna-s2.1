from __future__ import annotations

import json
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .runner import machine_metadata


def _peak_rss_gib() -> float:
    maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    bytes_used = maximum if sys.platform == "darwin" else maximum * 1024
    return bytes_used / 1024**3


def _select_result(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return next(row for row in rows if int(row.get(field) or 0) > 0)


def run_llama_profile(
    *,
    executable: Path,
    model_file: Path,
    model_id: str,
    revision: str | None,
    output_root: Path,
    prompt_tokens: int = 16384,
    generation_tokens: int = 256,
) -> tuple[Path, dict[str, Any]]:
    started = datetime.now(UTC)
    command = [
        str(executable),
        "-m",
        str(model_file),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-ngl",
        "999",
        "-fa",
        "on",
        "-r",
        "1",
        "-o",
        "json",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = json.loads(completed.stdout)
    prompt = _select_result(rows, "n_prompt")
    generation = _select_result(rows, "n_gen")
    peak_memory_gb = _peak_rss_gib()
    backend = {
        "model_id": model_id,
        "model_file": model_file.name,
        "engine": f"llama.cpp-{prompt.get('build_commit', 'unknown')}",
        "resolved_revision": revision,
        "model_bytes": model_file.stat().st_size,
        "quantization": prompt.get("model_type"),
    }
    cases = [
        {
            "id": f"context-{prompt_tokens}",
            "category": "context",
            "context_tokens": prompt_tokens,
            "actual_context_tokens": prompt_tokens,
            "max_new_tokens": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "status": "ok",
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt["avg_ts"],
            "generation_tokens": 0,
            "generation_tps": 0.0,
            "peak_memory_gb": peak_memory_gb,
            "memory_metric": "process_max_rss",
            "elapsed_seconds": prompt["avg_ns"] / 1e9,
        },
        {
            "id": f"decode-{generation_tokens}",
            "category": "decode",
            "context_tokens": 0,
            "actual_context_tokens": 0,
            "max_new_tokens": generation_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "status": "ok",
            "prompt_tokens": 0,
            "prompt_tps": 0.0,
            "generation_tokens": generation_tokens,
            "generation_tps": generation["avg_ts"],
            "peak_memory_gb": peak_memory_gb,
            "memory_metric": "process_max_rss",
            "elapsed_seconds": generation["avg_ns"] / 1e9,
        },
    ]
    record = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "profile": "quant",
        "machine": machine_metadata(),
        "backend": backend,
        "command": command,
        "cases": cases,
    }
    destination = output_root / "sweeps"
    destination.mkdir(parents=True, exist_ok=True)
    slug = model_id.replace("/", "--").replace(":", "-")
    path = destination / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{slug}-llama-quant.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path, record

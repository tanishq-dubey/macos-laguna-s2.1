from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .runner import machine_metadata


def record_failure(output_root: Path, command: str, arguments: dict[str, Any], exc: Exception) -> Path:
    timestamp = datetime.now(UTC)
    model_id = str(arguments.get("model") or "unknown")
    record = {
        "schema_version": 1,
        "started_at": timestamp.isoformat(),
        "machine": machine_metadata(),
        "backend": {
            "model_id": model_id,
            "engine": arguments.get("engine") or "mlx-vlm",
            "requested_revision": arguments.get("revision"),
        },
        "command": command,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()},
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    destination = output_root / "failures"
    destination.mkdir(parents=True, exist_ok=True)
    slug = model_id.replace("/", "--")
    path = destination / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{slug}-{command}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path

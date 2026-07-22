from __future__ import annotations

import re
from pathlib import Path


_SPLIT_GGUF = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$"
)


def model_payload_bytes(model_file: Path) -> int:
    """Return the complete payload size for a single or split GGUF."""
    match = _SPLIT_GGUF.match(model_file.name)
    if match is None:
        return model_file.stat().st_size

    count = int(match.group("count"))
    prefix = match.group("prefix")
    shards = [
        model_file.with_name(f"{prefix}-{index:05d}-of-{count:05d}.gguf")
        for index in range(1, count + 1)
    ]
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(f"missing split GGUF shard(s): {', '.join(missing)}")
    return sum(shard.stat().st_size for shard in shards)

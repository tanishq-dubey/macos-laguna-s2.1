from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backend import MLXBackend
from .runner import machine_metadata


@dataclass(frozen=True)
class SweepCase:
    id: str
    category: str
    context_tokens: int
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    kv_bits: float | None = None
    kv_quant_scheme: str = "uniform"
    max_kv_size: int | None = None
    prefill_step_size: int = 2048
    prompt_kind: str = "needle"


def sweep_cases(profile: str) -> list[SweepCase]:
    context = [
        SweepCase(f"context-{size}", "context", size, 32)
        for size in (256, 1024, 4096, 16384, 65536, 131072, 262144)
    ]
    decode = [
        SweepCase(f"decode-{size}", "decode", 1024, size, prompt_kind="decode")
        for size in (64, 256, 1024)
    ]
    sampling = [
        SweepCase("sampling-t02-p095", "sampling", 1024, 256, temperature=0.2, top_p=0.95, prompt_kind="decode"),
        SweepCase("sampling-t07-p095", "sampling", 1024, 256, temperature=0.7, top_p=0.95, prompt_kind="decode"),
    ]
    cache = [
        SweepCase("kv-uniform-8", "kv-cache", 16384, 32, kv_bits=8),
        SweepCase("kv-uniform-4", "kv-cache", 16384, 32, kv_bits=4),
        SweepCase("kv-turboquant-3.5", "kv-cache", 16384, 32, kv_bits=3.5, kv_quant_scheme="turboquant"),
        SweepCase("kv-window-4096", "kv-cache", 16384, 32, max_kv_size=4096),
        SweepCase("prefill-step-512", "prefill", 16384, 32, prefill_step_size=512),
        SweepCase("prefill-step-8192", "prefill", 16384, 32, prefill_step_size=8192),
    ]
    profiles = {
        "context": context,
        "decode": decode,
        "sampling": sampling,
        "cache": cache,
        "quant": [context[3], decode[1]],
        "full": context + decode + sampling + cache,
    }
    return profiles[profile]


def _render_prompt(backend: MLXBackend, messages: list[dict[str, str]]) -> str:
    return backend.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _prompt_token_count(backend: MLXBackend, messages: list[dict[str, str]]) -> int:
    prompt = _render_prompt(backend, messages)
    encoded = backend.tokenizer(prompt, add_special_tokens=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def build_messages(backend: MLXBackend, case: SweepCase) -> tuple[list[dict[str, str]], str | None, int]:
    needle = f"LAGUNA-{case.context_tokens:06d}-S21"
    if case.prompt_kind == "needle":
        prefix = f"Memorize this retrieval key: {needle}. The following material is irrelevant.\n"
        suffix = f"\nEnd of irrelevant material. Return only the retrieval key that began this message:"
        expected: str | None = needle
    else:
        prefix = "This is fixed benchmark context. Read it silently.\n"
        suffix = (
            "\nContinue with a detailed numbered technical checklist. Keep generating until the token limit; "
            "do not conclude, summarize, or emit an end marker."
        )
        expected = None

    filler = " Laguna benchmark filler 0123456789."

    def messages_for(repeats: int) -> list[dict[str, str]]:
        return [{"role": "user", "content": prefix + filler * repeats + suffix}]

    low, high = 0, max(1, case.context_tokens)
    while _prompt_token_count(backend, messages_for(high)) < case.context_tokens:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if _prompt_token_count(backend, messages_for(middle)) <= case.context_tokens:
            low = middle
        else:
            high = middle
    messages = messages_for(low)
    return messages, expected, _prompt_token_count(backend, messages)


def run_sweep(
    backend: MLXBackend,
    profile: str,
    output_root: Path,
    *,
    cases: list[SweepCase] | None = None,
) -> tuple[Path, dict[str, Any]]:
    import mlx.core as mx
    started = datetime.now(UTC)
    selected = cases or sweep_cases(profile)
    rows: list[dict[str, Any]] = []
    for case in selected:
        messages, expected, actual_context = build_messages(backend, case)
        mx.clear_cache()
        mx.reset_peak_memory()
        options: dict[str, Any] = {
            "temperature": case.temperature,
            "top_p": case.top_p,
            "prefill_step_size": case.prefill_step_size,
        }
        if case.kv_bits is not None:
            options.update(kv_bits=case.kv_bits, kv_quant_scheme=case.kv_quant_scheme)
        if case.max_kv_size is not None:
            options["max_kv_size"] = case.max_kv_size
        row: dict[str, Any] = {
            **asdict(case),
            "actual_context_tokens": actual_context,
            "expected": expected,
            "status": "ok",
            "error": None,
            "available_memory_before_gb": _available_memory_gb(),
        }
        try:
            result = backend.generate(
                messages,
                max_tokens=case.max_new_tokens,
                generation_options=options,
            )
            output = result.text
            row.update(
                result.to_dict(),
                output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                needle_pass=(expected in output if expected is not None else None),
                output=output,
            )
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", needle_pass=False if expected else None)
        row["available_memory_after_gb"] = _available_memory_gb()
        rows.append(row)

    record = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "machine": machine_metadata(),
        "backend": backend.metadata(),
        "cases": rows,
    }
    destination = output_root / "sweeps"
    destination.mkdir(parents=True, exist_ok=True)
    slug = backend.model_id.replace("/", "--")
    path = destination / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{slug}-{profile}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path, record


def _available_memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except (OSError, RuntimeError):
        return None

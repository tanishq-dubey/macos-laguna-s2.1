from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


TASK_IDS = {
    "generation-small",
    "generation-medium",
    "generation-large",
    "agentic-small",
    "agentic-medium",
    "agentic-large",
}

LOCAL_MODELS = {
    "unsloth/Laguna-S-2.1-GGUF:UD-IQ1_S": "IQ1_S",
    "unsloth/Laguna-S-2.1-GGUF:UD-IQ1_M": "IQ1_M",
    "mlx-community/Laguna-S-2.1-oQ2e": "oQ2e",
    "pipenetwork/Laguna-S-2.1-MLX-2bit": "PIPE 2B",
    "unsloth/Laguna-S-2.1-GGUF:UD-Q2_K_XL": "Q2_K_XL",
    "JANGQ-AI/Laguna-S-2.1-JANG_2L": "JANG 2L",
    "mlx-community/Laguna-S-2.1-oQ3e": "oQ3e",
    "pipenetwork/Laguna-S-2.1-MLX-3bit": "PIPE 3B",
    "mlx-community/Laguna-S-2.1-oQ4e": "oQ4e",
    "poolside/Laguna-S-2.1-NVFP4-mlx": "NVFP4",
}

REFERENCE_CHIP = "Apple M5 Max"


@dataclass(frozen=True)
class PublishedScore:
    model: str
    size: str
    score: float
    third_party: bool


@dataclass(frozen=True)
class LocalPoint:
    model: str
    label: str
    decode_tps: float
    peak_memory_gb: float
    suite_score: float


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_published_scores(path: Path) -> list[PublishedScore]:
    scores = []
    for row in _read_rows(path):
        scores.append(
            PublishedScore(
                model=row["model"],
                size=row["size"],
                score=float(row["terminal_bench_2_1"]),
                third_party=row["third_party"].lower() == "true",
            )
        )
    return sorted(scores, key=lambda item: item.score, reverse=True)


def _latest_complete_suite_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["record_type"] != "task" or row["model_id"] not in LOCAL_MODELS:
            continue
        if row["chip"] != REFERENCE_CHIP:
            continue
        if row["temperature"] not in {"", "0", "0.0"}:
            continue
        grouped[(row["model_id"], row["artifact"])].append(row)

    complete: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (model, artifact), task_rows in grouped.items():
        if {row["case_id"] for row in task_rows} != TASK_IDS:
            continue
        score = sum(float(row["score"]) for row in task_rows) / len(TASK_IDS)
        timestamp = max(row["timestamp"] for row in task_rows)
        complete[model].append((timestamp or artifact, score))

    return {model: max(values)[1] for model, values in complete.items()}


def load_local_points(path: Path) -> list[LocalPoint]:
    rows = _read_rows(path)
    suite_scores = _latest_complete_suite_scores(rows)
    performance: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row["model_id"]
        if (
            row["record_type"] == "performance"
            and model in LOCAL_MODELS
            and row["chip"] == REFERENCE_CHIP
            and row["case_id"] == "decode-256"
            and row["status"] == "ok"
        ):
            current = performance.get(model)
            if current is None or row["timestamp"] > current["timestamp"]:
                performance[model] = row

    missing = set(LOCAL_MODELS) - (suite_scores.keys() & performance.keys())
    if missing:
        raise ValueError(f"missing complete chart data for: {', '.join(sorted(missing))}")

    return [
        LocalPoint(
            model=model,
            label=label,
            decode_tps=float(performance[model]["generation_tps"]),
            peak_memory_gb=float(performance[model]["peak_memory_gb"]),
            suite_score=suite_scores[model],
        )
        for model, label in LOCAL_MODELS.items()
    ]


def _text(
    parts: list[str],
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    weight: int = 400,
    fill: str = "#dbe6ff",
    anchor: str = "start",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def _line(parts: list[str], x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1) -> None:
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" />'
    )


def render_results_chart(results_csv: Path, poolside_csv: Path, destination: Path) -> Path:
    published = load_published_scores(poolside_csv)
    local = load_local_points(results_csv)
    fastest = max(local, key=lambda point: point.decode_tps)
    destination.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img" aria-labelledby="title desc">',
        '<title id="title">Laguna S 2.1 published capability and local Apple Silicon performance</title>',
        '<desc id="desc">Poolside published Terminal-Bench 2.1 scores beside locally measured MLX and llama.cpp decode speed and peak memory for Laguna S 2.1 quantizations on an M5 Max.</desc>',
        '<rect width="1600" height="1000" fill="#191919" />',
        '<style>text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", ui-monospace, monospace; }</style>',
    ]
    purple = "#b17ce8"
    foreground = "#f0efed"
    muted = "#b9b8b4"
    bar_gray = "#74736e"
    rule = "#666561"

    _text(parts, 32, 57, "LAGUNA S 2.1", size=34, weight=700, fill=purple)
    _text(parts, 322, 57, "ON APPLE SILICON", size=34, weight=700, fill=foreground)
    _text(parts, 32, 98, f"FASTEST FIXED DECODE: {fastest.decode_tps:.2f} TOK/S, {fastest.peak_memory_gb:.2f} GB ({fastest.label})", size=23, weight=500, fill=muted)
    _text(parts, 32, 132, "PUBLISHED CAPABILITY AND LOCAL INFERENCE USE DIFFERENT SOURCES AND SCALES", size=15, fill="#8f8e8a")

    # Left panel: Poolside's published Terminal-Bench comparison.
    _line(parts, 24, 177, 950, 177, stroke=rule, width=1.5)
    _text(parts, 24, 220, "TERMINAL-BENCH 2.1", size=24, weight=500, fill=foreground)
    _text(parts, 24, 249, "POOLSIDE PUBLISHED, RESOLVED TASKS (%)", size=14, fill=muted)

    plot_left = 86.0
    plot_right = 918.0
    baseline = 760.0
    plot_height = 430.0
    slot = (plot_right - plot_left) / len(published)
    bar_width = 60.0
    label_lines = {
        "Claude Fable 5": ("CLAUDE", "FABLE 5"),
        "Muse Spark 1.1": ("MUSE", "SPARK 1.1"),
        "Qwen 3.7 Max": ("QWEN", "3.7 MAX"),
        "Tencent HY3": ("TENCENT", "HY3"),
        "Laguna S 2.1": ("LAGUNA", "S 2.1"),
        "DeepSeek-V4-Pro Max": ("DEEPSEEK", "V4 PRO MAX"),
        "Nemotron 3 Ultra": ("NEMOTRON", "3 ULTRA"),
    }
    _line(parts, plot_left - 18, baseline, plot_right + 10, baseline, stroke=rule)
    for index, item in enumerate(published):
        center = plot_left + slot * (index + 0.5)
        height = plot_height * item.score / 100
        y = baseline - height
        is_laguna = item.model == "Laguna S 2.1"
        color = purple if is_laguna else bar_gray
        label_color = purple if is_laguna else foreground
        suffix = "*" if item.third_party else ""
        parts.append(f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}" />')
        _text(parts, center, y - 13, f"{item.score:g}{suffix}", size=18, weight=500, fill=label_color, anchor="middle")
        lines = label_lines.get(item.model, (item.model.upper(), ""))
        _text(parts, center, 793, lines[0], size=12, weight=700 if is_laguna else 500, fill=label_color, anchor="middle")
        if lines[1]:
            _text(parts, center, 811, lines[1], size=11, fill=label_color, anchor="middle")
        if item.size:
            _text(parts, center, 834, item.size, size=10, fill="#85847f", anchor="middle")

    _text(parts, 24, 872, "* POOLSIDE MARKS THIS TERMINAL-BENCH SCORE AS THIRD-PARTY REPORTED", size=12, fill="#85847f")

    # Right panels: local decode throughput and memory in the reference style.
    local_left, local_right = 1025.0, 1548.0
    local_slot = (local_right - local_left) / len(local)
    local_bar_width = min(84.0, local_slot * 0.72)

    _line(parts, 985, 177, 1576, 177, stroke=rule, width=1.5)
    _text(parts, 985, 220, "FIXED 256-TOKEN DECODE TOK/S", size=22, weight=500, fill=foreground)
    _text(parts, 985, 249, "MLX-VLM, MLX-LM, JANG, OR LLAMA.CPP; GREEDY", size=14, fill=muted)
    decode_base, decode_height = 474.0, 175.0
    _line(parts, local_left - 25, decode_base, local_right + 20, decode_base, stroke=rule)
    for index, point in enumerate(local):
        center = local_left + local_slot * (index + 0.5)
        height = decode_height * point.decode_tps / 72
        y = decode_base - height
        primary = point.model == fastest.model
        color = purple if primary else bar_gray
        text_color = purple if primary else foreground
        parts.append(f'<rect x="{center - local_bar_width / 2:.1f}" y="{y:.1f}" width="{local_bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}" />')
        _text(parts, center, y - 13, f"{point.decode_tps:.2f}", size=18, weight=500, fill=text_color, anchor="middle")
        _text(parts, center, 505, point.label.upper(), size=10, weight=700 if primary else 500, fill=text_color, anchor="middle")
        _text(parts, center, 525, f"{point.suite_score * 100:.1f}%", size=9, fill="#85847f", anchor="middle")

    _line(parts, 985, 575, 1576, 575, stroke=rule, width=1.5)
    _text(parts, 985, 618, "PEAK MEMORY (GB)", size=24, weight=500, fill=foreground)
    _text(parts, 985, 647, "MLX ALLOCATOR PEAK; GGUF PROCESS MAX RSS", size=14, fill=muted)
    memory_base, memory_height = 870.0, 175.0
    _line(parts, local_left - 25, memory_base, local_right + 20, memory_base, stroke=rule)
    for index, point in enumerate(local):
        center = local_left + local_slot * (index + 0.5)
        height = memory_height * point.peak_memory_gb / 80
        y = memory_base - height
        primary = point.model == fastest.model
        color = purple if primary else bar_gray
        text_color = purple if primary else foreground
        parts.append(f'<rect x="{center - local_bar_width / 2:.1f}" y="{y:.1f}" width="{local_bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}" />')
        _text(parts, center, y - 13, f"{point.peak_memory_gb:.2f}", size=18, weight=500, fill=text_color, anchor="middle")
        _text(parts, center, 901, point.label.upper(), size=10, weight=700 if primary else 500, fill=text_color, anchor="middle")

    _text(parts, 24, 934, "POOLSIDE: MODEL CARD, 21 JULY 2026    LOCAL: COMMITTED CSV, FIXED 256-TOKEN DECODE PROFILE", size=13, fill="#969590")
    _text(parts, 24, 965, "oQ2e CONTEXT: 64K 32.03 TOK/S, 41.19 GB    |    256K 12.16 TOK/S, 52.87 GB    |    RETRIEVAL PASSED", size=13, weight=500, fill=purple)
    _text(parts, 1576, 965, "MACOS-LAGUNA-S2.1", size=13, weight=700, fill=foreground, anchor="end")
    parts.append("</svg>\n")
    destination.write_text("\n".join(parts))
    return destination

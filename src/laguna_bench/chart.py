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
    "mlx-community/Laguna-S-2.1-oQ2e": "oQ2e (2.7-bit)",
    "mlx-community/Laguna-S-2.1-oQ3e": "oQ3e",
    "poolside/Laguna-S-2.1-NVFP4-mlx": "NVFP4 MLX",
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
            and row["case_id"] == "context-16384"
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
    destination.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">',
        '<title id="title">Laguna S 2.1 published capability and local Apple Silicon performance</title>',
        '<desc id="desc">Poolside published Terminal-Bench 2.1 scores beside locally measured MLX decode speed and peak memory for three Laguna S 2.1 quantizations.</desc>',
        '<rect width="1600" height="900" fill="#090e1a" />',
        '<style>text { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }</style>',
    ]
    _text(parts, 64, 70, "Laguna S 2.1: published capability, measured Mac performance", size=34, weight=700, fill="#f6f8ff")
    _text(parts, 64, 105, "Separate sources and scales. The local suite is not Terminal-Bench.", size=18, fill="#8fa2c9")

    # Left panel: Poolside's published Terminal-Bench comparison.
    parts.append('<rect x="50" y="135" width="760" height="650" rx="20" fill="#111a2d" stroke="#263554" />')
    _text(parts, 82, 182, "Poolside published", size=15, weight=700, fill="#62d6b3")
    _text(parts, 82, 215, "Terminal-Bench 2.1", size=25, weight=700, fill="#f6f8ff")
    _text(parts, 82, 242, "Resolved tasks (%)", size=15, fill="#8fa2c9")

    bar_x = 360.0
    bar_width = 390.0
    plot_top = 280.0
    row_height = 49.0
    for tick in (0, 25, 50, 75, 100):
        x = bar_x + bar_width * tick / 100
        _line(parts, x, plot_top - 18, x, plot_top + row_height * len(published) - 8, stroke="#263554")
        _text(parts, x, plot_top - 28, str(tick), size=13, fill="#7183a7", anchor="middle")

    for index, item in enumerate(published):
        y = plot_top + index * row_height
        is_laguna = item.model == "Laguna S 2.1"
        color = "#62d6b3" if is_laguna else "#5774b8"
        label_color = "#f6f8ff" if is_laguna else "#dbe6ff"
        suffix = "*" if item.third_party else ""
        _text(parts, 82, y + 20, item.model, size=16, weight=700 if is_laguna else 500, fill=label_color)
        if item.size:
            _text(parts, 82, y + 39, item.size, size=12, fill="#7183a7")
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_width * item.score / 100:.1f}" '
            f'height="28" rx="6" fill="{color}" />'
        )
        _text(parts, bar_x + bar_width * item.score / 100 - 9, y + 20, f"{item.score:g}{suffix}", size=14, weight=700, fill="#09111d", anchor="end")

    _text(parts, 82, 756, "* Poolside marks this Terminal-Bench score as third-party reported.", size=13, fill="#7183a7")

    # Right panel: local 16K quant scatter plot.
    parts.append('<rect x="835" y="135" width="715" height="650" rx="20" fill="#111a2d" stroke="#263554" />')
    _text(parts, 867, 182, "Measured locally", size=15, weight=700, fill="#7bb8ff")
    _text(parts, 867, 215, "MLX quant profile at 16K", size=25, weight=700, fill="#f6f8ff")
    _text(parts, 867, 242, "128 GB M5 Max, mlx-vlm 0.6.6, greedy decoding", size=15, fill="#8fa2c9")

    plot_left, plot_right = 930.0, 1490.0
    plot_bottom, plot_top_y = 600.0, 300.0
    memory_min, memory_max = 30.0, 80.0
    speed_min, speed_max = 25.0, 65.0
    for tick in (30, 40, 50, 60, 70, 80):
        x = plot_left + (tick - memory_min) / (memory_max - memory_min) * (plot_right - plot_left)
        _line(parts, x, plot_top_y, x, plot_bottom, stroke="#263554")
        _text(parts, x, plot_bottom + 27, str(tick), size=13, fill="#7183a7", anchor="middle")
    for tick in (30, 40, 50, 60):
        y = plot_bottom - (tick - speed_min) / (speed_max - speed_min) * (plot_bottom - plot_top_y)
        _line(parts, plot_left, y, plot_right, y, stroke="#263554")
        _text(parts, plot_left - 14, y + 5, str(tick), size=13, fill="#7183a7", anchor="end")
    _text(parts, (plot_left + plot_right) / 2, 647, "Peak MLX memory (GB)", size=15, fill="#8fa2c9", anchor="middle")
    _text(parts, 870, 455, "Decode tok/s", size=15, fill="#8fa2c9")

    offsets = {"oQ2e (2.7-bit)": (14, -18), "oQ3e": (14, 31), "NVFP4 MLX": (-14, -18)}
    for point in local:
        x = plot_left + (point.peak_memory_gb - memory_min) / (memory_max - memory_min) * (plot_right - plot_left)
        y = plot_bottom - (point.decode_tps - speed_min) / (speed_max - speed_min) * (plot_bottom - plot_top_y)
        primary = point.label.startswith("oQ2e")
        color = "#62d6b3" if primary else "#7bb8ff"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{color}" stroke="#f6f8ff" stroke-width="2" />')
        dx, dy = offsets[point.label]
        anchor = "end" if dx < 0 else "start"
        _text(parts, x + dx, y + dy, point.label, size=16, weight=700, fill="#f6f8ff", anchor=anchor)
        _text(
            parts,
            x + dx,
            y + dy + 20,
            f"{point.decode_tps:.2f} tok/s, {point.peak_memory_gb:.2f} GB, {point.suite_score * 100:.1f}% suite",
            size=12,
            fill="#8fa2c9",
            anchor=anchor,
        )

    parts.append('<rect x="867" y="680" width="651" height="74" rx="12" fill="#17243c" />')
    _text(parts, 891, 709, "oQ2e context scaling", size=15, weight=700, fill="#62d6b3")
    _text(parts, 891, 737, "64K: 32.03 tok/s, 41.19 GB    |    256K: 12.16 tok/s, 52.87 GB    |    retrieval passed", size=14, fill="#dbe6ff")

    _text(parts, 64, 835, "Published scores: Poolside Laguna S 2.1 model card, benchmark table dated 21 July 2026.", size=15, fill="#8fa2c9")
    _text(parts, 64, 866, "Local results: committed CSV. 16K points use fixed-length generation; suite scores cover six coding tasks.", size=15, fill="#8fa2c9")
    _text(parts, 1536, 866, "macos-laguna-s2.1", size=14, weight=700, fill="#62d6b3", anchor="end")
    parts.append("</svg>\n")
    destination.write_text("\n".join(parts))
    return destination

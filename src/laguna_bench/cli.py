from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backend import MLXBackend, OpenAIBackend
from .catalog import QUANTS
from .chart import render_results_chart
from .csv_export import export_csv
from .failures import record_failure
from .runner import run_suite
from .reporting import build_comparison
from .sweep import SweepCase, run_sweep
from .tasks import TASKS, select_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laguna-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list tasks and known MLX quantizations")
    compare = sub.add_parser("compare", help="compare the latest complete run for each model")
    compare.add_argument("--output", type=Path, default=Path("results"))
    export = sub.add_parser("export", help="export all task and performance records to one tidy CSV")
    export.add_argument("--output", type=Path, default=Path("results"))
    chart = sub.add_parser("chart", help="render the published and local results chart")
    chart.add_argument("--results", type=Path, default=Path("results/laguna_s21_full_results.csv"))
    chart.add_argument("--poolside", type=Path, default=Path("charts/poolside_terminal_bench_2_1.csv"))
    chart.add_argument("--output", type=Path, default=Path("charts/laguna-s21-results.svg"))
    sweep = sub.add_parser("sweep", help="run context, decode, sampling, and KV-cache performance cases")
    sweep.add_argument("--model", required=True, help="Hugging Face model ID or local snapshot path")
    sweep.add_argument("--revision")
    sweep.add_argument("--profile", choices=["full", "context", "decode", "sampling", "cache", "quant"], default="full")
    sweep.add_argument("--context", action="append", type=int, help="custom target context; repeat for multiple sizes")
    sweep.add_argument("--max-new-tokens", type=int, default=32, help="decode cap for custom --context cases")
    sweep.add_argument("--output", type=Path, default=Path("results"))
    sweep.add_argument("--seed", type=int, default=20260721)
    run = sub.add_parser("run", help="run benchmark tasks")
    run.add_argument("--model", required=True, help="Hugging Face model ID or local snapshot path")
    run.add_argument("--engine", choices=["mlx-vlm", "openai"], default="mlx-vlm")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI endpoint when --engine=openai")
    run.add_argument("--api-key", help="API key for a local OpenAI-compatible endpoint")
    run.add_argument("--revision")
    run.add_argument("--task", action="append", dest="tasks", help="task ID; repeat to select multiple")
    run.add_argument("--kind", action="append", choices=["generation", "agentic"])
    run.add_argument("--tier", action="append", choices=["small", "medium", "large"])
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--seed", type=int, default=20260721)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument(
        "--agent-prompt-cache",
        action="store_true",
        help="experimental KV reuse; not exact on Laguna hybrid caches in mlx-vlm 0.6.6",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print("Tasks:")
        for task in TASKS:
            print(f"  {task.id:20} kind={task.kind:10} max_tokens={task.max_tokens:4} max_steps={task.max_steps}")
        print("\nQuant ladder:")
        for quant in QUANTS:
            marker = "*" if quant["recommended"] else " "
            print(f" {marker} {quant['size_gib']:6.2f} GiB  {quant['model']}  ({quant['status']})")
        return 0
    if args.command == "compare":
        path, rows = build_comparison(args.output)
        print(path.read_text(), end="")
        print(f"Comparison: {path} ({len(rows)} model(s))")
        return 0
    if args.command == "export":
        path, rows = export_csv(args.output)
        print(f"CSV: {path} ({rows} row(s))")
        return 0
    if args.command == "chart":
        path = render_results_chart(args.results, args.poolside, args.output)
        print(f"Chart: {path}")
        return 0
    try:
        if args.command == "sweep":
            print(f"Loading {args.model} with mlx-vlm ...", flush=True)
            backend = MLXBackend(args.model, revision=args.revision, seed=args.seed)
            cases = None
            if args.context:
                cases = [
                    SweepCase(f"context-{size}", "context", size, args.max_new_tokens)
                    for size in args.context
                ]
            print(f"Loaded in {backend.load_seconds:.2f}s; running {len(cases) if cases else args.profile} sweep ...", flush=True)
            path, record = run_sweep(backend, args.profile, args.output, cases=cases)
            for case in record["cases"]:
                print(
                    f"{case['id']}: {case['status']} prompt={case.get('prompt_tokens', 0)} "
                    f"pp={case.get('prompt_tps', 0):.2f} tg={case.get('generation_tps', 0):.2f} "
                    f"peak={case.get('peak_memory_gb', 0):.2f}GB needle={case.get('needle_pass')}"
                )
            csv_path, _ = export_csv(args.output)
            print(f"Sweep: {path}\nCSV: {csv_path}")
            return 0
        tasks = select_tasks(args.tasks, args.kind, args.tier)
        if not tasks:
            raise ValueError("task selection is empty")
        print(f"Loading {args.model} with {args.engine} ...", flush=True)
        if args.engine == "mlx-vlm":
            backend = MLXBackend(
                args.model,
                revision=args.revision,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"Loaded in {backend.load_seconds:.2f}s; running {len(tasks)} task(s) ...", flush=True)
        else:
            backend = OpenAIBackend(args.model, args.base_url, seed=args.seed, api_key=args.api_key)
            print(f"Connected to {args.base_url}; running {len(tasks)} task(s) ...", flush=True)
        run_dir, record = run_suite(
            backend,
            tasks,
            args.output,
            agent_prompt_cache=args.agent_prompt_cache,
        )
        for item in record["tasks"]:
            print(f"{item['id']}: {item['grade']['passed']}/{item['grade']['total']} ({item['grade']['score']:.3f})")
        print(f"Results: {run_dir}")
        return 0
    except Exception as exc:
        failure = record_failure(getattr(args, "output", Path("results")), args.command, vars(args), exc)
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc), "artifact": str(failure)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

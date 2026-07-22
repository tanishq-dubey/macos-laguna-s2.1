import csv
import json

from laguna_bench.csv_export import FIELDS, export_csv
from laguna_bench.reporting import build_comparison
from laguna_bench.tasks import TASKS


def test_comparison_uses_complete_runs(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    tasks = []
    for task in TASKS:
        metrics = {"generation_tokens": 10, "elapsed_seconds": 2, "peak_memory_gb": 3}
        item = {"id": task.id, "kind": task.kind, "grade": {"score": 1.0}}
        if task.kind == "generation":
            item["metrics"] = metrics
        else:
            item["agent"] = {"metrics": metrics}
        tasks.append(item)
    (run / "run.json").write_text(json.dumps({"backend": {"model_id": "model/a", "load_seconds": 1}, "tasks": tasks, "aggregate_score": 1.0}))
    destination, rows = build_comparison(tmp_path)
    assert destination.exists()
    assert rows[0]["generation_tps"] == 5.0
    assert rows[0]["score"] == 1.0


def test_comparison_uses_sweep_memory_for_server_runs(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    tasks = []
    for task in TASKS:
        metrics = {"generation_tokens": 10, "elapsed_seconds": 2, "peak_memory_gb": 0}
        item = {"id": task.id, "kind": task.kind, "grade": {"score": 1.0}}
        item["metrics" if task.kind == "generation" else "agent"] = (
            metrics if task.kind == "generation" else {"metrics": metrics}
        )
        tasks.append(item)
    (run / "run.json").write_text(
        json.dumps({"backend": {"model_id": "server/model", "engine": "openai-compatible"}, "tasks": tasks})
    )
    sweeps = tmp_path / "sweeps"
    sweeps.mkdir()
    (sweeps / "profile.json").write_text(
        json.dumps(
            {
                "backend": {"model_id": "server/model"},
                "cases": [{"peak_memory_gb": 34.22, "memory_metric": "process_max_rss"}],
            }
        )
    )

    _, rows = build_comparison(tmp_path)
    assert rows[0]["peak_memory_gb"] == 34.22
    assert rows[0]["memory_metric"] == "process_max_rss"


def test_csv_export_includes_catalog_and_task_rows(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "started_at": "2026-01-01T00:00:00Z",
                "backend": {"model_id": "model/a", "engine": "mlx-vlm"},
                "machine": {"chip": "test"},
                "tasks": [
                    {
                        "id": "generation-small",
                        "kind": "generation",
                        "tier": "small",
                        "grade": {"score": 1, "passed": 1, "total": 1},
                        "metrics": {"generation_tokens": 2, "generation_tps": 3},
                    }
                ],
            }
        )
    )
    destination, count = export_csv(tmp_path)
    rows = list(csv.DictReader(destination.open()))
    assert count == len(rows)
    assert {row["record_type"] for row in rows} == {"variant", "task"}
    assert next(row for row in rows if row["record_type"] == "task")["generation_tps"] == "3"


def test_csv_export_merges_existing_community_results_without_duplicates(tmp_path):
    destination = tmp_path / "laguna_s21_full_results.csv"
    contributed_row = {
        "record_type": "performance",
        "timestamp": "2026-02-01T00:00:00Z",
        "model_id": "contributor/model",
        "engine": "mlx-vlm",
        "chip": "Apple M4 Max",
        "case_id": "context-4096",
        "generation_tps": "42.5",
    }
    stale_variant = {
        "record_type": "variant",
        "model_id": "stale/catalog-entry",
    }
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([stale_variant, contributed_row])

    _, first_count = export_csv(tmp_path)
    _, second_count = export_csv(tmp_path)
    rows = list(csv.DictReader(destination.open()))

    assert first_count == second_count == len(rows)
    assert sum(row["model_id"] == "contributor/model" for row in rows) == 1
    assert all(row["model_id"] != "stale/catalog-entry" for row in rows)


def test_csv_export_keeps_multiple_files_from_one_quant_repository(tmp_path):
    destination, _ = export_csv(tmp_path)
    rows = list(csv.DictReader(destination.open()))
    unsloth = [row for row in rows if row["model_id"] == "unsloth/Laguna-S-2.1-GGUF:UD-IQ1_M"]

    assert len(unsloth) == 1
    assert unsloth[0]["model_file"] == "Laguna-S-2.1-UD-IQ1_M.gguf"
    assert unsloth[0]["source_repo"] == "unsloth/Laguna-S-2.1-GGUF"
    assert unsloth[0]["engine"] == "llama.cpp"


def test_csv_export_refreshes_existing_row_from_local_artifact(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    run_path = run / "run.json"
    existing_row = {
        "record_type": "task",
        "model_id": "model/a",
        "artifact": str(run_path),
        "case_id": "generation-small",
        "generation_tps": "1",
    }
    destination = tmp_path / "laguna_s21_full_results.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(existing_row)
    run_path.write_text(
        json.dumps(
            {
                "backend": {"model_id": "model/a", "engine": "mlx-vlm"},
                "machine": {"macos": "27.0", "python": "3.13.12"},
                "tasks": [
                    {
                        "id": "generation-small",
                        "kind": "generation",
                        "grade": {"score": 1},
                        "metrics": {"generation_tps": 9},
                    }
                ],
            }
        )
    )

    export_csv(tmp_path)
    rows = list(csv.DictReader(destination.open()))
    matching = [row for row in rows if row["model_id"] == "model/a"]
    assert len(matching) == 1
    assert matching[0]["generation_tps"] == "9"
    assert matching[0]["macos_version"] == "27.0"


def test_comparison_prefers_canonical_greedy_runs(tmp_path):
    for name, temperature, score in (("a", 0.0, 1.0), ("b", 0.7, 0.5)):
        run = tmp_path / name
        run.mkdir()
        tasks = []
        for task in TASKS:
            metrics = {"generation_tokens": 1, "elapsed_seconds": 1}
            item = {"id": task.id, "kind": task.kind, "grade": {"score": score}}
            item["metrics" if task.kind == "generation" else "agent"] = metrics if task.kind == "generation" else {"metrics": metrics}
            tasks.append(item)
        (run / "run.json").write_text(
            json.dumps(
                {
                    "backend": {"model_id": "model/a", "engine": "mlx-vlm"},
                    "settings": {"temperature": temperature, "top_p": 0.95 if temperature else 1.0},
                    "tasks": tasks,
                    "aggregate_score": score,
                }
            )
        )
    _, rows = build_comparison(tmp_path)
    assert rows[0]["score"] == 1.0

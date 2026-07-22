import json
import csv

from laguna_bench.csv_export import export_csv
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

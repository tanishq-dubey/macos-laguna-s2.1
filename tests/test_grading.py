from pathlib import Path

from laguna_bench.grading import extract_code, grade_generation, grade_workspace
from laguna_bench.runner import materialize_fixture
from laguna_bench.tasks import TASKS


def task(task_id):
    return next(item for item in TASKS if item.id == task_id)


def test_exact_json_grader():
    result = grade_generation(task("generation-small"), '{"items":["birch-tree","cedar","maple"],"count":3}')
    assert result["score"] == 1.0


def test_extract_code():
    assert extract_code("Here\n```python\ndef f():\n    pass\n```") == "def f():\n    pass"


def test_generation_medium_reference_passes():
    code = """def merge_intervals(intervals):
    result = []
    for start, end in sorted((list(x) for x in intervals), key=lambda x: x[0]):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
"""
    assert grade_generation(task("generation-medium"), code)["score"] == 1.0


def test_agent_small_grader(tmp_path: Path):
    benchmark = task("agentic-small")
    materialize_fixture(benchmark, tmp_path)
    (tmp_path / "answer.txt").write_text("INV-1042\n")
    assert grade_workspace(benchmark, tmp_path)["score"] == 1.0

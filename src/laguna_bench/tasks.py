from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Task:
    id: str
    kind: Literal["generation", "agentic"]
    tier: Literal["small", "medium", "large"]
    prompt: str
    max_tokens: int
    max_steps: int = 1
    fixture: dict[str, str] = field(default_factory=dict)
    grader: str = ""
    test_count: int = 1


GEN_MEDIUM_TESTS = r'''from candidate import merge_intervals

def test_empty(): assert merge_intervals([]) == []
def test_single(): assert merge_intervals([[2, 4]]) == [[2, 4]]
def test_overlap(): assert merge_intervals([[1, 4], [2, 6]]) == [[1, 6]]
def test_touching(): assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]
def test_unsorted(): assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]
def test_nested(): assert merge_intervals([[1, 9], [3, 4], [5, 8]]) == [[1, 9]]
def test_negative(): assert merge_intervals([[-4, -1], [-2, 2]]) == [[-4, 2]]
def test_no_mutation():
    source = [[5, 6], [1, 2]]
    merge_intervals(source)
    assert source == [[5, 6], [1, 2]]
'''


GEN_LARGE_TESTS = r'''import pytest
from candidate import build_stages

def test_empty(): assert build_stages([]) == []
def test_one(): assert build_stages([{"name":"a","depends_on":[]}]) == [["a"]]
def test_parallel_sorted():
    jobs=[{"name":"z","depends_on":[]},{"name":"a","depends_on":[]}]
    assert build_stages(jobs) == [["a","z"]]
def test_chain():
    jobs=[{"name":"c","depends_on":["b"]},{"name":"a","depends_on":[]},{"name":"b","depends_on":["a"]}]
    assert build_stages(jobs) == [["a"],["b"],["c"]]
def test_diamond():
    jobs=[{"name":"d","depends_on":["b","c"]},{"name":"c","depends_on":["a"]},{"name":"b","depends_on":["a"]},{"name":"a","depends_on":[]}]
    assert build_stages(jobs) == [["a"],["b","c"],["d"]]
def test_independent_can_start_early():
    jobs=[{"name":"c","depends_on":["b"]},{"name":"b","depends_on":["a"]},{"name":"a","depends_on":[]},{"name":"x","depends_on":[]}]
    assert build_stages(jobs) == [["a","x"],["b"],["c"]]
def test_duplicate():
    with pytest.raises(ValueError): build_stages([{"name":"a","depends_on":[]},{"name":"a","depends_on":[]}])
def test_missing_dependency():
    with pytest.raises(ValueError): build_stages([{"name":"a","depends_on":["missing"]}])
def test_self_cycle():
    with pytest.raises(ValueError): build_stages([{"name":"a","depends_on":["a"]}])
def test_long_cycle():
    with pytest.raises(ValueError): build_stages([{"name":"a","depends_on":["c"]},{"name":"b","depends_on":["a"]},{"name":"c","depends_on":["b"]}])
'''


AGENT_MEDIUM_FIXTURE = {
    "README.md": """# Duration parser bug\n\n`parse_duration` should accept a sequence of integer components using `d`, `h`, `m`, and `s` in descending order, tolerate surrounding whitespace, and reject malformed or repeated units with `ValueError`. Fix the implementation without weakening the tests.\n""",
    "duration.py": r'''import re

UNIT_SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1}

def parse_duration(value: str) -> int:
    """Convert strings such as ``1h 30m`` to seconds."""
    match = re.match(r"\s*(\d+)\s*([dhms])", value)
    if not match:
        raise ValueError("invalid duration")
    amount, unit = match.groups()
    return int(amount) * UNIT_SECONDS[unit]
''',
    "test_duration.py": r'''import pytest
from duration import parse_duration

def test_seconds(): assert parse_duration("45s") == 45
def test_compound(): assert parse_duration("1h 30m") == 5400
def test_all_units(): assert parse_duration("2d 3h 4m 5s") == 183845
def test_spaces(): assert parse_duration("  1h   2m  ") == 3720
def test_bad_suffix():
    with pytest.raises(ValueError): parse_duration("10m nope")
def test_wrong_order():
    with pytest.raises(ValueError): parse_duration("1m 2h")
def test_duplicate():
    with pytest.raises(ValueError): parse_duration("1h 2h")
def test_empty():
    with pytest.raises(ValueError): parse_duration("")
''',
}


AGENT_LARGE_FIXTURE = {
    "README.md": """# Event store feature\n\nImplement `EventStore.append_many(stream, events, expected_version)` and `EventStore.read(stream, after_version=0, limit=None)`. Read `SPEC.md`, preserve the public API, and make the full test suite pass. Do not edit tests.\n""",
    "SPEC.md": """Each stream has monotonically increasing versions beginning at 1. `append_many` atomically appends dictionaries and returns stored event dictionaries augmented with `stream` and `version`. If expected_version differs from the current version, raise `ConcurrencyError` and append nothing. Inputs and returned/read values must be isolated copies. `read` returns events with version strictly greater than `after_version`, in order; `limit=None` means unlimited, while a negative limit raises ValueError. Unknown streams read as an empty list.\n""",
    "event_store/__init__.py": "from .store import ConcurrencyError, EventStore\n\n__all__ = ['ConcurrencyError', 'EventStore']\n",
    "event_store/store.py": r'''class ConcurrencyError(RuntimeError):
    pass


class EventStore:
    def __init__(self):
        self._streams = {}

    def append_many(self, stream, events, expected_version):
        raise NotImplementedError

    def read(self, stream, after_version=0, limit=None):
        raise NotImplementedError
''',
    "test_event_store.py": r'''import pytest
from event_store import ConcurrencyError, EventStore

def test_unknown_stream(): assert EventStore().read("none") == []
def test_append_versions_and_metadata():
    s=EventStore(); out=s.append_many("orders", [{"type":"opened"},{"type":"paid"}], 0)
    assert [e["version"] for e in out] == [1,2]
    assert all(e["stream"] == "orders" for e in out)
def test_append_to_existing():
    s=EventStore(); s.append_many("x", [{"n":1}], 0)
    assert s.append_many("x", [{"n":2}], 1)[0]["version"] == 2
def test_conflict_is_atomic():
    s=EventStore(); s.append_many("x", [{"n":1}], 0)
    with pytest.raises(ConcurrencyError): s.append_many("x", [{"n":2},{"n":3}], 0)
    assert [e["n"] for e in s.read("x")] == [1]
def test_read_after():
    s=EventStore(); s.append_many("x", [{"n":1},{"n":2},{"n":3}], 0)
    assert [e["n"] for e in s.read("x", after_version=1)] == [2,3]
def test_limit():
    s=EventStore(); s.append_many("x", [{"n":1},{"n":2},{"n":3}], 0)
    assert [e["n"] for e in s.read("x", limit=2)] == [1,2]
def test_zero_limit():
    s=EventStore(); s.append_many("x", [{"n":1}], 0)
    assert s.read("x", limit=0) == []
def test_negative_limit():
    with pytest.raises(ValueError): EventStore().read("x", limit=-1)
def test_input_is_copied():
    s=EventStore(); event={"type":"a","data":{"v":1}}; s.append_many("x", [event], 0)
    event["data"]["v"]=9
    assert s.read("x")[0]["data"]["v"] == 1
def test_output_is_copied():
    s=EventStore(); out=s.append_many("x", [{"data":{"v":1}}], 0); out[0]["data"]["v"]=9
    read=s.read("x"); read[0]["data"]["v"]=8
    assert s.read("x")[0]["data"]["v"] == 1
''',
}


TASKS = [
    Task(
        id="generation-small",
        kind="generation",
        tier="small",
        max_tokens=128,
        grader="exact_json",
        prompt="""Return only a JSON object, with no Markdown or explanation. Given the tokens [\" Cedar \", \"maple\", \"CEDAR\", \"birch-tree\", \"Maple\"], trim whitespace, lowercase them, remove duplicates, sort lexicographically, and include both the resulting `items` array and its `count`.""",
    ),
    Task(
        id="generation-medium",
        kind="generation",
        tier="medium",
        max_tokens=768,
        grader="python",
        test_count=8,
        fixture={"test_candidate.py": GEN_MEDIUM_TESTS},
        prompt="""Write a Python function `merge_intervals(intervals)` that returns overlapping or touching closed intervals merged. Inputs may be unsorted and must not be mutated. Return only one Python code block containing a complete implementation; use no third-party packages.""",
    ),
    Task(
        id="generation-large",
        kind="generation",
        tier="large",
        max_tokens=1400,
        grader="python",
        test_count=10,
        fixture={"test_candidate.py": GEN_LARGE_TESTS},
        prompt="""Write a complete Python module defining `build_stages(jobs)`. Each job is a dict with a unique nonempty string `name` and a `depends_on` list of names. Return the earliest parallel execution stages as `list[list[str]]`: every stage contains all currently runnable jobs sorted lexicographically. Return [] for no jobs. Raise ValueError for duplicate names, missing dependencies, or any cycle. Do not mutate inputs. Return only one Python code block and use no third-party packages.""",
    ),
    Task(
        id="agentic-small",
        kind="agentic",
        tier="small",
        max_tokens=384,
        max_steps=5,
        grader="agent_invoice",
        fixture={
            "README.md": "Inspect ledger.csv. Find the only invoice ID appearing twice with different amounts, then write only that ID and a newline to answer.txt.\n",
            "ledger.csv": "invoice_id,amount\nINV-1003,19.50\nINV-1042,81.00\nINV-1011,7.25\nINV-1042,18.00\nINV-1099,42.10\n",
        },
        prompt="Work on the repository task described in README.md. Inspect the files, complete the task, and use submit_answer when done.",
    ),
    Task(
        id="agentic-medium",
        kind="agentic",
        tier="medium",
        max_tokens=640,
        max_steps=9,
        grader="pytest",
        test_count=8,
        fixture=AGENT_MEDIUM_FIXTURE,
        prompt="Fix the repository bug described in README.md. Inspect relevant files, edit only what is needed, run tests, and use submit_answer when done.",
    ),
    Task(
        id="agentic-large",
        kind="agentic",
        tier="large",
        max_tokens=900,
        max_steps=14,
        grader="pytest",
        test_count=10,
        fixture=AGENT_LARGE_FIXTURE,
        prompt="Implement the repository feature described in README.md. Read the specification and existing code, make focused edits, run tests, and use submit_answer when done.",
    ),
]


def select_tasks(ids: list[str] | None, kinds: list[str] | None, tiers: list[str] | None) -> list[Task]:
    selected = TASKS
    if ids:
        unknown = sorted(set(ids) - {task.id for task in TASKS})
        if unknown:
            raise ValueError(f"unknown task(s): {', '.join(unknown)}")
        selected = [task for task in selected if task.id in ids]
    if kinds:
        selected = [task for task in selected if task.kind in kinds]
    if tiers:
        selected = [task for task in selected if task.tier in tiers]
    return selected

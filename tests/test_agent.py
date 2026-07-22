from pathlib import Path

import pytest

from laguna_bench.agent import ToolCall, execute_tool, parse_tool_call, safe_path
from laguna_bench.tasks import TASKS


def test_parse_native_tool_call():
    call = parse_tool_call("<tool_call>write_file<arg_key>path</arg_key><arg_value>x.py</arg_value><arg_key>content</arg_key><arg_value>print(1)</arg_value></tool_call>")
    assert call == ToolCall("write_file", {"path": "x.py", "content": "print(1)"})


def test_parse_json_fallback():
    call = parse_tool_call('{"tool":"read_file","arguments":{"path":"README.md"}}')
    assert call == ToolCall("read_file", {"path": "README.md"})


def test_safe_path_rejects_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../secret")


def test_write_is_confined(tmp_path: Path):
    task = TASKS[3]
    result = execute_tool(ToolCall("write_file", {"path": "nested/x.txt", "content": "ok"}), tmp_path, task)
    assert "Wrote" in result
    assert (tmp_path / "nested/x.txt").read_text() == "ok"


def test_list_files_handles_macos_private_tmp_alias(tmp_path: Path):
    task = TASKS[3]
    (tmp_path / "README.md").write_text("hello")
    result = execute_tool(ToolCall("list_files", {"path": "."}), tmp_path, task)
    assert result == "README.md"

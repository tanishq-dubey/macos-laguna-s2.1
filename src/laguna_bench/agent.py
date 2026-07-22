from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import Generation
from .grading import run_pytest
from .tasks import Task


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "list_files", "description": "List files below a workspace-relative directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a UTF-8 workspace file, creating parent directories.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the repository's fixed pytest suite.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "submit_answer", "description": "Finish the task and briefly summarize the result.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


AGENT_SYSTEM = """You are a coding agent operating in a disposable benchmark workspace. Use the provided tools to inspect and modify only workspace-relative paths. Do not merely describe edits: perform them. Call run_tests when relevant. When the task is complete, call submit_answer. Make exactly one tool call per response."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_call(text: str) -> ToolCall | None:
    native = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if native:
        body = native.group(1)
        name_match = re.match(r"\s*([^<\s]+)", body)
        if not name_match:
            return None
        args: dict[str, Any] = {}
        pairs = re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, flags=re.DOTALL)
        for raw_key, raw_value in pairs:
            key, value = html.unescape(raw_key.strip()), html.unescape(raw_value)
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        return ToolCall(name_match.group(1), args)
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("tool"), str):
        return ToolCall(data["tool"], data.get("arguments") or {})
    return None


def run_agent(backend, task: Task, workspace: Path, *, use_prompt_cache: bool = False) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": task.prompt},
    ]
    transcript: list[dict[str, Any]] = []
    generations: list[Generation] = []
    prompt_cache_state = (
        backend.new_prompt_cache()
        if use_prompt_cache and hasattr(backend, "new_prompt_cache")
        else None
    )
    finished = False
    for step in range(1, task.max_steps + 1):
        generation = backend.generate(
            messages,
            max_tokens=task.max_tokens,
            tools=TOOL_SCHEMAS,
            prompt_cache_state=prompt_cache_state,
        )
        generations.append(generation)
        call = parse_tool_call(generation.text)
        entry: dict[str, Any] = {"step": step, "generation": generation.to_dict()}
        if call is None:
            entry["error"] = "no parseable tool call"
            transcript.append(entry)
            messages.append({"role": "assistant", "content": generation.text})
            messages.append({"role": "tool", "content": "Invalid response: call exactly one available tool."})
            continue
        result = execute_tool(call, workspace, task)
        entry["tool"] = {"name": call.name, "arguments": call.arguments, "result": result}
        transcript.append(entry)
        call_id = f"call_{step}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        if call.name == "submit_answer":
            finished = True
            break
    return {"finished": finished, "steps": len(transcript), "transcript": transcript, "metrics": combine_metrics(generations)}


def execute_tool(call: ToolCall, workspace: Path, task: Task) -> str:
    workspace = workspace.resolve()
    try:
        if call.name == "list_files":
            base = safe_path(workspace, str(call.arguments.get("path", ".")))
            if not base.exists() or not base.is_dir():
                return "ERROR: directory not found"
            return "\n".join(str(p.resolve().relative_to(workspace)) for p in sorted(base.rglob("*")) if p.is_file())[:50000]
        if call.name == "read_file":
            path = safe_path(workspace, str(call.arguments["path"]))
            if not path.is_file():
                return "ERROR: file not found"
            return path.read_text()[:50000]
        if call.name == "write_file":
            path = safe_path(workspace, str(call.arguments["path"]))
            content = call.arguments.get("content")
            if not isinstance(content, str):
                return "ERROR: content must be a string"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"Wrote {len(content.encode())} bytes to {path.relative_to(workspace)}"
        if call.name == "run_tests":
            grade = run_pytest(workspace, task.test_count)
            return f"score={grade['passed']}/{grade['total']} exit_code={grade.get('exit_code')}\n{grade['detail']}"
        if call.name == "submit_answer":
            return "Submitted: " + str(call.arguments.get("summary", ""))
        return f"ERROR: unknown tool {call.name!r}"
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def safe_path(workspace: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("path must be workspace-relative")
    root = workspace.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes workspace")
    return target


def combine_metrics(generations: list[Generation]) -> dict[str, Any]:
    prompt_tokens = sum(g.prompt_tokens for g in generations)
    generation_tokens = sum(g.generation_tokens for g in generations)
    elapsed = sum(g.elapsed_seconds for g in generations)
    return {
        "turns": len(generations),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "cached_tokens": sum(g.cached_tokens for g in generations),
        "elapsed_seconds": elapsed,
        "effective_generation_tps": generation_tokens / elapsed if elapsed else 0.0,
        "peak_memory_gb": max((g.peak_memory_gb for g in generations), default=0.0),
    }

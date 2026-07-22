from pathlib import Path

from laguna_bench.backend import _native_tool_call, _openai_messages, _snapshot_metadata


def test_openai_messages_stringify_tool_arguments():
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "x"}}}]}]
    normalized = _openai_messages(messages)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == '{"path":"x"}'


def test_native_tool_call_round_trip_shape():
    assert _native_tool_call("read_file", {"path": "x"}) == "<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value></tool_call>"


def test_snapshot_metadata_falls_back_to_cache_ref(monkeypatch, tmp_path: Path):
    repo = tmp_path / "models--org--model"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)

    import huggingface_hub
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    assert _snapshot_metadata("org/model", None) == {
        "resolved_revision": "a" * 40,
        "snapshot_path": str(snapshot),
        "model_bytes": 7,
    }

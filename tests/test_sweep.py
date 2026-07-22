import psutil

from laguna_bench.sweep import _available_memory_gb, sweep_cases


def test_full_sweep_covers_performance_dimensions():
    cases = sweep_cases("full")
    categories = {case.category for case in cases}
    assert categories == {"context", "decode", "sampling", "kv-cache", "prefill"}
    assert max(case.context_tokens for case in cases) == 262144
    assert max(case.max_new_tokens for case in cases) == 1024


def test_quant_profile_is_short_and_comparable():
    cases = sweep_cases("quant")
    assert [(case.context_tokens, case.max_new_tokens) for case in cases] == [(16384, 32), (1024, 256)]


def test_memory_telemetry_is_optional(monkeypatch):
    monkeypatch.setattr(psutil, "virtual_memory", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    assert _available_memory_gb() is None

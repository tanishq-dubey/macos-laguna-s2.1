import psutil

from laguna_bench.sweep import _available_memory_gb, _prompt_token_count, sweep_cases


class _EncodeOnlyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "rendered prompt"

    def encode(self, prompt, add_special_tokens=True):
        return [1, 2, 3]


def test_prompt_token_count_supports_mlx_lm_wrapper():
    backend = type("Backend", (), {"tokenizer": _EncodeOnlyTokenizer()})()
    assert _prompt_token_count(backend, [{"role": "user", "content": "x"}]) == 3


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

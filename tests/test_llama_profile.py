import json
from pathlib import Path

from laguna_bench.llama_profile import run_llama_profile


def test_llama_profile_records_standard_cases(tmp_path):
    executable = tmp_path / "fake-llama-bench"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s' '[{\"build_commit\":\"abc123\",\"model_type\":\"test IQ2\",\"n_prompt\":4,\"n_gen\":0,\"avg_ts\":100.0,\"avg_ns\":40000000},"
        "{\"build_commit\":\"abc123\",\"model_type\":\"test IQ2\",\"n_prompt\":0,\"n_gen\":8,\"avg_ts\":50.0,\"avg_ns\":160000000}]'\n"
    )
    executable.chmod(0o755)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"gguf")

    destination, record = run_llama_profile(
        executable=executable,
        model_file=model_file,
        model_id="repo/model:IQ2",
        revision="revision-a",
        output_root=tmp_path,
        prompt_tokens=4,
        generation_tokens=8,
    )

    assert destination.exists()
    assert json.loads(destination.read_text()) == record
    assert record["backend"]["resolved_revision"] == "revision-a"
    assert [case["id"] for case in record["cases"]] == ["context-4", "decode-8"]
    assert record["cases"][1]["generation_tps"] == 50.0

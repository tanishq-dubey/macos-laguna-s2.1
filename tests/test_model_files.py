from pathlib import Path

import pytest

from laguna_bench.model_files import model_payload_bytes


def test_model_payload_bytes_for_single_file(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    assert model_payload_bytes(model) == 4


def test_model_payload_bytes_sums_split_gguf(tmp_path: Path):
    shards = [
        tmp_path / f"model-{index:05d}-of-00003.gguf" for index in range(1, 4)
    ]
    for index, shard in enumerate(shards, start=1):
        shard.write_bytes(b"x" * index)
    assert model_payload_bytes(shards[0]) == 6
    assert model_payload_bytes(shards[1]) == 6


def test_model_payload_bytes_rejects_missing_shard(tmp_path: Path):
    model = tmp_path / "model-00001-of-00002.gguf"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="00002-of-00002"):
        model_payload_bytes(model)

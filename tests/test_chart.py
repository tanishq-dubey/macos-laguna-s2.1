from pathlib import Path

import pytest

from laguna_bench.chart import load_local_points, load_published_scores, render_results_chart


ROOT = Path(__file__).parents[1]


def test_chart_sources_have_expected_laguna_results():
    published = load_published_scores(ROOT / "charts/poolside_terminal_bench_2_1.csv")
    local = load_local_points(ROOT / "results/laguna_s21_full_results.csv")

    laguna = next(item for item in published if item.model == "Laguna S 2.1")
    oq2e = next(item for item in local if item.label.startswith("oQ2e"))
    assert laguna.score == 70.2
    iq1 = next(item for item in local if item.label == "IQ1_M GGUF")
    pipe2 = next(item for item in local if item.label == "PIPE 2-BIT")
    q2xl = next(item for item in local if item.label == "Q2_K_XL")
    assert oq2e.decode_tps == pytest.approx(55.06, abs=0.005)
    assert oq2e.peak_memory_gb == pytest.approx(37.22, abs=0.005)
    assert oq2e.suite_score == 1.0
    assert iq1.decode_tps == pytest.approx(62.68, abs=0.005)
    assert iq1.peak_memory_gb == pytest.approx(34.22, abs=0.005)
    assert iq1.suite_score == pytest.approx(0.792, abs=0.001)
    assert pipe2.decode_tps == pytest.approx(68.49, abs=0.005)
    assert pipe2.peak_memory_gb == pytest.approx(38.65, abs=0.005)
    assert pipe2.suite_score == 1.0
    assert q2xl.decode_tps == pytest.approx(55.93, abs=0.005)
    assert q2xl.peak_memory_gb == pytest.approx(37.99, abs=0.005)
    assert q2xl.suite_score == 0.875


def test_chart_renders_standalone_svg(tmp_path):
    destination = tmp_path / "chart.svg"
    render_results_chart(
        ROOT / "results/laguna_s21_full_results.csv",
        ROOT / "charts/poolside_terminal_bench_2_1.csv",
        destination,
    )
    svg = destination.read_text()
    assert svg.startswith("<svg")
    assert "Laguna S 2.1" in svg
    assert "68.49 TOK/S" in svg

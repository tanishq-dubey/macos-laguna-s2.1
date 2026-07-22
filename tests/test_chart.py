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
    iq1s = next(item for item in local if item.label == "IQ1_S")
    iq2xxs = next(item for item in local if item.label == "I2XX")
    iq2m = next(item for item in local if item.label == "I2M")
    iq1 = next(item for item in local if item.label == "IQ1_M")
    pipe2 = next(item for item in local if item.label == "P2B")
    q2xl = next(item for item in local if item.label == "Q2XL")
    iq3xxs = next(item for item in local if item.label == "I3XX")
    iq3s = next(item for item in local if item.label == "I3S")
    jang = next(item for item in local if item.label == "J2L")
    pipe3 = next(item for item in local if item.label == "P3B")
    oq4e = next(item for item in local if item.label == "oQ4e")
    assert oq2e.decode_tps == pytest.approx(55.06, abs=0.005)
    assert oq2e.peak_memory_gb == pytest.approx(37.22, abs=0.005)
    assert oq2e.suite_score == 1.0
    assert iq1.decode_tps == pytest.approx(62.68, abs=0.005)
    assert iq1.peak_memory_gb == pytest.approx(34.22, abs=0.005)
    assert iq1.suite_score == pytest.approx(0.792, abs=0.001)
    assert iq1s.decode_tps == pytest.approx(55.85, abs=0.005)
    assert iq1s.peak_memory_gb == pytest.approx(32.48, abs=0.005)
    assert iq1s.suite_score == pytest.approx(0.658, abs=0.001)
    assert iq2xxs.decode_tps == pytest.approx(57.35, abs=0.005)
    assert iq2xxs.peak_memory_gb == pytest.approx(35.66, abs=0.005)
    assert iq2xxs.suite_score == pytest.approx(0.646, abs=0.001)
    assert iq2m.decode_tps == pytest.approx(60.29, abs=0.005)
    assert iq2m.peak_memory_gb == pytest.approx(35.74, abs=0.005)
    assert iq2m.suite_score == pytest.approx(0.625, abs=0.001)
    assert pipe2.decode_tps == pytest.approx(68.49, abs=0.005)
    assert pipe2.peak_memory_gb == pytest.approx(38.65, abs=0.005)
    assert pipe2.suite_score == 1.0
    assert q2xl.decode_tps == pytest.approx(55.93, abs=0.005)
    assert q2xl.peak_memory_gb == pytest.approx(37.99, abs=0.005)
    assert q2xl.suite_score == 0.875
    assert iq3xxs.decode_tps == pytest.approx(55.13, abs=0.005)
    assert iq3xxs.peak_memory_gb == pytest.approx(42.27, abs=0.005)
    assert iq3xxs.suite_score == pytest.approx(0.708, abs=0.001)
    assert iq3s.decode_tps == pytest.approx(53.77, abs=0.005)
    assert iq3s.peak_memory_gb == pytest.approx(46.13, abs=0.005)
    assert iq3s.suite_score == pytest.approx(0.708, abs=0.001)
    assert jang.decode_tps == pytest.approx(49.29, abs=0.005)
    assert jang.peak_memory_gb == pytest.approx(45.29, abs=0.005)
    assert jang.suite_score == pytest.approx(0.417, abs=0.001)
    assert pipe3.decode_tps == pytest.approx(65.31, abs=0.005)
    assert pipe3.peak_memory_gb == pytest.approx(52.34, abs=0.005)
    assert pipe3.suite_score == 0.875
    assert oq4e.decode_tps == pytest.approx(52.00, abs=0.005)
    assert oq4e.peak_memory_gb == pytest.approx(65.12, abs=0.005)
    assert oq4e.suite_score == 0.875


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

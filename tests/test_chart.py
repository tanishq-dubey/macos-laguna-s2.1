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
    assert oq2e.decode_tps == pytest.approx(52.48, abs=0.005)
    assert oq2e.peak_memory_gb == pytest.approx(38.46, abs=0.005)
    assert oq2e.suite_score == 1.0


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
    assert "52.48 TOK/S" in svg

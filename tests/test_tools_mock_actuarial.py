import json
from pathlib import Path

from tools.mock_actuarial import MockParams, compute_mock_reserving, generate_mock_triangle
from tools.validation import validate_triangle_records


def test_mock_triangle_and_validation():
    triangle = generate_mock_triangle(MockParams(scenario="base", triangle_size=6, seed=1))
    report = validate_triangle_records(triangle)
    assert report.valid is True
    assert len(triangle) > 0


def test_mock_reserving_is_deterministic():
    p = MockParams(scenario="stress", triangle_size=7, seed=9)
    out1 = compute_mock_reserving(p)
    out2 = compute_mock_reserving(p)
    assert out1.total_reserve == out2.total_reserve
    assert out1.reserve_by_ay == out2.reserve_by_ay
    assert len(out1.factors) > 0
    assert out1.total_reserve > 0
    assert all(f >= 1.0 for f in out1.factors)


def test_chain_ladder_against_fixture():
    fixture = json.loads(
        Path("tests/fixtures/triangle_sample.json").read_text(encoding="utf-8")
    )
    params = MockParams(**fixture["params"])
    output = compute_mock_reserving(params)

    assert output.total_reserve == fixture["expected_total_reserve"]
    assert output.factors == fixture["expected_factors"]
    assert output.cdf == fixture["expected_cdf"]
    assert output.ibnr_by_ay == fixture["expected_ibnr_by_ay"]

import json
from pathlib import Path

from tools.reserving import (
    ReservingParams,
    compute_analytical_mack_deficit_probability,
    compute_chain_ladder_from_triangle_records,
    compute_chain_ladder_reserving,
    compute_mack_lognormal_p_def,
    compute_mack_standard_error,
    compute_risk_metrics,
    load_example_triangle,
    risk_metrics_skipped_zero_ibnr,
)
from tools.validation import validate_triangle_records


def test_example_triangle_and_validation():
    triangle = load_example_triangle(ReservingParams(scenario="base", triangle_size=6))
    report = validate_triangle_records(triangle)
    assert report.valid is True
    assert len(triangle) > 0


def test_chain_ladder_reserving_is_deterministic():
    p = ReservingParams(scenario="stress", triangle_size=7)
    out1 = compute_chain_ladder_reserving(p)
    out2 = compute_chain_ladder_reserving(p)
    assert out1.total_reserve == out2.total_reserve
    assert out1.reserve_by_ay == out2.reserve_by_ay
    assert len(out1.factors) > 0
    assert out1.total_reserve > 0
    assert all(f >= 1.0 for f in out1.factors)


def test_chain_ladder_against_fixture():
    fixture = json.loads(
        Path("tests/fixtures/triangle_sample.json").read_text(encoding="utf-8")
    )
    params = ReservingParams(**fixture["params"])
    output = compute_chain_ladder_reserving(params)

    assert output.total_reserve == fixture["expected_total_reserve"]
    assert output.factors == fixture["expected_factors"]
    assert output.cdf == fixture["expected_cdf"]
    assert output.ibnr_by_ay == fixture["expected_ibnr_by_ay"]


def test_bootstrap_risk_metrics_shape_and_targets():
    params = ReservingParams(scenario="base", triangle_size=8, simulations=4000, random_seed=123)
    output = compute_chain_ladder_reserving(params)
    metrics = compute_risk_metrics(
        triangle_records=output.triangle_records,
        base_reserve=output.total_reserve,
        simulations=4000,
        seed=123,
    )

    assert 0 <= metrics["p_def"] <= 1
    assert metrics["var_995"] >= metrics["var_95"]
    assert metrics["tvar_995"] >= metrics["var_995"]
    assert metrics["rm_required_005"] >= 0
    assert metrics["p_def_after_rm"] <= 0.01
    assert len(metrics["simulated_ibnr"]) == 4000
    assert metrics["risk_analysis_skipped"] is False
    assert metrics["rm_required_005"] == round(
        max(0.0, metrics["var_995"] - metrics["base_reserve"]), 2
    )
    assert "capital_surplus_regime" in metrics
    assert "low_p_def_extreme_tail_warning" in metrics
    assert metrics["analysis_basis"] == "gross_before_reinsurance"


def test_chain_ladder_allows_negative_total_ibnr_when_ldf_below_one():
    """Incomplete last AY with only first dev: ultimate = latest * tail product can fall below case."""
    triangle = [
        {"accident_year": 2019, "development_period": 0, "value": 5000.0},
        {"accident_year": 2020, "development_period": 0, "value": 10000.0},
        {"accident_year": 2020, "development_period": 1, "value": 9000.0},
        {"accident_year": 2020, "development_period": 2, "value": 8500.0},
    ]
    out = compute_chain_ladder_from_triangle_records(triangle)
    assert out.total_reserve < 0
    assert any(v < 0 for v in out.ibnr_by_ay.values())


def test_risk_metrics_skipped_when_zero_ibnr():
    m = risk_metrics_skipped_zero_ibnr(0.0)
    assert m["risk_analysis_skipped"] is True
    assert m["skip_reason"] == "zero_best_estimate"
    assert m["simulated_ibnr"] == []
    assert m["rm_required_005"] == 0.0


def test_risk_metrics_skipped_negative_flags():
    m = risk_metrics_skipped_zero_ibnr(-100.0)
    assert m["skip_reason"] == "negative_best_estimate"
    assert m["reserve_surplus_regime"] is True


def test_mack_standard_error_positive_on_example_triangle():
    params = ReservingParams(scenario="base", triangle_size=8)
    output = compute_chain_ladder_reserving(params)
    mack = compute_mack_standard_error(output.triangle_records)
    assert mack["mack_se_total"] > 0
    assert mack["mack_se_by_ay"]
    assert len(mack["mack_process_variances"]) == len(output.factors)


def test_mack_lognormal_p_def_between_zero_and_one():
    out = compute_mack_lognormal_p_def(base_reserve=1000.0, mack_se=120.0)
    p = float(out["p_def_mack_analytical"])
    assert 0.0 < p < 1.0
    assert out["mack_method"] == "lognormal_cdf"


def test_risk_metrics_includes_bootstrap_and_mack_p_def():
    params = ReservingParams(scenario="base", triangle_size=8, simulations=2000, random_seed=7)
    output = compute_chain_ladder_reserving(params)
    metrics = compute_risk_metrics(
        triangle_records=output.triangle_records,
        base_reserve=output.total_reserve,
        simulations=2000,
        seed=7,
    )
    assert metrics["p_def"] == metrics["p_def_bootstrap"]
    assert metrics["p_default_mack"] == metrics["p_def_mack_analytical"]
    assert metrics["deficit_probability_definition"]
    assert metrics["default_probability_definition"]
    assert 0.0 <= metrics["p_def_mack_analytical"] <= 1.0
    assert metrics["mack_se_total"] > 0
    assert metrics["mack_method"] == "lognormal_cdf"


def test_analytical_mack_matches_risk_metrics_fields():
    params = ReservingParams(scenario="stress", triangle_size=7)
    output = compute_chain_ladder_reserving(params)
    analytical = compute_analytical_mack_deficit_probability(
        triangle_records=output.triangle_records,
        base_reserve=output.total_reserve,
    )
    metrics = compute_risk_metrics(
        triangle_records=output.triangle_records,
        base_reserve=output.total_reserve,
        simulations=500,
        seed=1,
    )
    assert analytical["p_def_mack_analytical"] == metrics["p_def_mack_analytical"]
    assert analytical["mack_se_total"] == metrics["mack_se_total"]

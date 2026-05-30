import pandas as pd
import pytest

from tools.reserving import compute_chain_ladder_from_triangle_records
from tools.triangle_from_transactions import parse_transaction_dataframe


def test_transactions_pivot_and_cumulative():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020, 2020, 2021, 2021],
            "Development_Lag": [1, 2, 1, 2],
            "Paid_Amount": [100.0, 50.0, 80.0, 40.0],
        }
    )
    records, report = parse_transaction_dataframe(df, values_mode="incremental")
    assert report.valid
    by_cell = {(r["accident_year"], r["development_period"]): r["value"] for r in records}
    assert by_cell[(2020, 1)] == 100.0
    assert by_cell[(2020, 2)] == 150.0
    assert by_cell[(2021, 1)] == 80.0
    assert by_cell[(2021, 2)] == 120.0


def test_negative_rows_ok_when_cell_total_non_negative():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020, 2020],
            "Development_Lag": [1, 1],
            "Paid_Amount": [-10.0, 100.0],
        }
    )
    _, report = parse_transaction_dataframe(df)
    assert report.valid
    assert any("negative row" in w.lower() for w in report.warnings)


def test_transactions_negative_paid_clipped_with_warning():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020],
            "Development_Lag": [1],
            "Paid_Amount": [-1.0],
        }
    )
    _, report = parse_transaction_dataframe(df)
    assert report.valid
    assert any("clipped" in w.lower() for w in report.warnings)


def test_transactions_allow_lag_zero():
    df = pd.DataFrame(
        {
            "Accident_Year": [2019, 2019, 2020],
            "Development_Lag": [0, 1, 0],
            "Paid_Amount": [100.0, 50.0, 80.0],
        }
    )
    records, report = parse_transaction_dataframe(df, values_mode="incremental")
    assert report.valid
    assert (2019, 0) in {(r["accident_year"], r["development_period"]) for r in records}
    assert (2019, 1) in {(r["accident_year"], r["development_period"]) for r in records}


def test_transactions_reject_gap_in_lags():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020, 2020],
            "Development_Lag": [1, 3],
            "Paid_Amount": [10.0, 5.0],
        }
    )
    _, report = parse_transaction_dataframe(df)
    assert not report.valid


def test_synonym_column_names():
    df = pd.DataFrame(
        {
            "AY": [2020, 2020, 2021],
            "Lag": [1, 2, 1],
            "Claims_Paid": [100.0, 40.0, 90.0],
            "ignore_me": [1, 2, 3],
        }
    )
    records, report = parse_transaction_dataframe(df, values_mode="incremental")
    assert report.valid
    by_cell = {(r["accident_year"], r["development_period"]): r["value"] for r in records}
    assert by_cell[(2020, 1)] == 100.0
    assert by_cell[(2020, 2)] == 140.0
    assert by_cell[(2021, 1)] == 90.0


def test_explicit_cumulative_no_double_sum():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020, 2020],
            "Development_Lag": [1, 2],
            "Paid_Amount": [100.0, 150.0],
        }
    )
    records, report = parse_transaction_dataframe(df, values_mode="cumulative")
    assert report.valid
    by_cell = {(r["accident_year"], r["development_period"]): r["value"] for r in records}
    assert by_cell[(2020, 1)] == 100.0
    assert by_cell[(2020, 2)] == 150.0


def test_auto_incremental_register_gets_cumsum_positive_reserve():
    """staible-style rows (5000→500→50) must be cumulated, not read as cumulative levels."""
    from pathlib import Path

    import pandas as pd

    csv_path = Path("staible-reserves.csv")
    if not csv_path.is_file():
        pytest.skip("staible-reserves.csv not in repo root")
    df = pd.read_csv(csv_path)
    records, report = parse_transaction_dataframe(df, values_mode="auto")
    assert report.valid
    assert any("decreasing payment" in w.lower() or "incremental" in w.lower() for w in report.warnings)
    out = compute_chain_ladder_from_triangle_records(records)
    assert out.total_reserve > 0


def test_auto_cumulative_subrogation_no_double_sum_ldf_below_one():
    """Cumulative triangle with recoveries (dip) must not be chain-summed as incremental."""
    rows: list[dict] = []
    tri = {
        2020: [(0, 5000), (1, 7000), (2, 5500), (3, 5600), (4, 5600)],
        2021: [(0, 5500), (1, 7700), (2, 6100), (3, 6250)],
        2022: [(0, 6000), (1, 8500), (2, 6700)],
        2023: [(0, 5800), (1, 8200)],
        2024: [(0, 6200)],
    }
    for ay, cells in tri.items():
        for lag, val in cells:
            rows.append({"Accident_Year": ay, "Development_Lag": lag, "Paid_Amount": float(val)})
    df = pd.DataFrame(rows)
    records, report = parse_transaction_dataframe(df, values_mode="auto")
    assert report.valid
    assert any("recover" in w.lower() for w in report.warnings)
    out = compute_chain_ladder_from_triangle_records(records)
    assert out.factors[1] < 1.0
    assert abs(out.factors[1] - 18300 / 23200) < 1e-3
    assert out.total_reserve < 0


def test_wide_triangle_auto():
    df = pd.DataFrame(
        {
            "Origin": [2019, 2020],
            "0": [50.0, 60.0],
            "1": [80.0, 95.0],
        }
    )
    records, report = parse_transaction_dataframe(df, layout="wide")
    assert report.valid
    by_cell = {(r["accident_year"], r["development_period"]): r["value"] for r in records}
    assert by_cell[(2019, 0)] == 50.0
    assert by_cell[(2019, 1)] == 80.0
    assert by_cell[(2020, 0)] == 60.0
    assert by_cell[(2020, 1)] == 95.0

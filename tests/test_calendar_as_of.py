import pandas as pd

from tools.triangle_from_transactions import parse_transaction_dataframe


def test_as_of_filter_removes_future_calendar_observations():
    """When newer accident years have later calendar years, min(max cy per AY) caps valuation."""
    rows: list[dict] = []
    for lag in range(1, 11):
        rows.append(
            {
                "AccidentYear": 1998,
                "DevelopmentLag": lag,
                "DevelopmentYear": 1998 + lag,
                "CumPaidLoss": float(100 * lag),
            }
        )
    for lag in range(1, 11):
        rows.append(
            {
                "AccidentYear": 2007,
                "DevelopmentLag": lag,
                "DevelopmentYear": 2007 + lag,
                "CumPaidLoss": float(50 * lag),
            }
        )
    df = pd.DataFrame(rows)
    records, report = parse_transaction_dataframe(df)
    assert report.valid
    assert any("as-of filter" in w.lower() for w in report.warnings)
    ay2007 = [r for r in records if r["accident_year"] == 2007]
    assert ay2007
    assert max(r["development_period"] for r in ay2007) == 1


def test_no_calendar_column_skips_filter():
    df = pd.DataFrame(
        {
            "Accident_Year": [2020, 2020, 2021],
            "Development_Lag": [1, 2, 1],
            "Paid_Amount": [100.0, 40.0, 90.0],
        }
    )
    records, report = parse_transaction_dataframe(df, values_mode="incremental")
    assert report.valid
    assert not any("as-of filter" in w.lower() for w in report.warnings)

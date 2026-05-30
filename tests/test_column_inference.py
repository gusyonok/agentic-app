import pandas as pd

from tools.column_inference import infer_column_mapping
from tools.triangle_from_transactions import parse_transaction_dataframe


def test_wkcomp_like_headers_pick_cumulative_paid_not_premium():
    """CAS-style long file: premiums can be negative; cumulative paid should win."""
    df = pd.DataFrame(
        {
            "GRCODE": [1, 1],
            "GRNAME": ["x", "x"],
            "AccidentYear": [1998, 1998],
            "DevelopmentYear": [1999, 2000],
            "DevelopmentLag": [1, 2],
            "IncurredLosses": [5000.0, 4800.0],
            "CumPaidLoss": [1201.0, 2652.0],
            "BulkLoss": [0.0, 0.0],
            "EarnedPremDIR": [8000.0, 8000.0],
            "EarnedPremCeded": [-250.0, -250.0],
            "EarnedPremNet": [7750.0, 7750.0],
            "Single": [0.0, 0.0],
            "PostedReserves2007": [100.0, 100.0],
        }
    )
    m = infer_column_mapping(df)
    assert m["accident_year"] == "AccidentYear"
    assert m["development_lag"] == "DevelopmentLag"
    assert m["paid_amount"] == "CumPaidLoss"

    records, report = parse_transaction_dataframe(df)
    assert report.valid
    assert len(records) >= 2
    by_cell = {(r["accident_year"], r["development_period"]): r["value"] for r in records}
    assert by_cell[(1998, 1)] == 1201.0
    assert by_cell[(1998, 2)] == 2652.0


def test_auto_fallback_when_first_amount_column_has_negatives():
    df = pd.DataFrame(
        {
            "AccidentYear": [2020, 2020],
            "DevelopmentLag": [1, 2],
            "BadAmount": [-10.0, 5.0],
            "CumPaidLoss": [100.0, 150.0],
        }
    )
    # Simulate a bad first pick
    r, rep = parse_transaction_dataframe(
        df,
        column_map={"accident_year": "AccidentYear", "development_lag": "DevelopmentLag", "paid_amount": "BadAmount"},
    )
    assert rep.valid
    assert any("switched" in w.lower() for w in rep.warnings)
    assert {(x["accident_year"], x["development_period"]): x["value"] for x in r} == {
        (2020, 1): 100.0,
        (2020, 2): 150.0,
    }

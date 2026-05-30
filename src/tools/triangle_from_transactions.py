"""Build cumulative loss triangles from flexible transaction / triangle uploads."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Literal

import pandas as pd

from core.models import ValidationReport
from tools.column_inference import (
    infer_calendar_year_column,
    infer_column_mapping,
    infer_valuation_year_hint_from_column_names,
    ranked_amount_fallbacks,
    try_identify_wide_triangle,
)
from tools.validation import validate_triangle_records

CANONICAL = ("accident_year", "development_lag", "paid_amount")

_NEG_CLIP_TOL = 1e-6


def _series_all_non_negative(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return not v.empty and bool((v >= -_NEG_CLIP_TOL).all())


def _agg_non_negative_by_ay_lag(df: pd.DataFrame, ay: str, lag: str, amt: str) -> bool:
    """True if Σ amount over rows sharing the same (ay, lag) is never meaningfully negative."""
    t = df[[ay, lag, amt]].copy()
    t["_v"] = pd.to_numeric(t[amt], errors="coerce")
    t = t.dropna(subset=[ay, lag, "_v"])
    if t.empty:
        return False
    g = t.groupby([ay, lag], sort=False)["_v"].sum()
    return bool((g >= -_NEG_CLIP_TOL).all())


def _pick_long_amount_working(
    df: pd.DataFrame,
    ay_n: str,
    lag_n: str,
    amt_candidates: list[str],
    warnings: list[str],
) -> tuple[pd.DataFrame | None, str | None]:
    """Pick loss amount column: strict non-negative rows → OK cell totals → clip negatives."""
    for cand in amt_candidates:
        paid_try = pd.to_numeric(df[cand], errors="coerce")
        if paid_try.isna().all():
            continue
        if _series_all_non_negative(paid_try):
            out = df[[ay_n, lag_n, cand]].rename(
                columns={ay_n: "accident_year", lag_n: "development_lag", cand: "paid_amount"}
            )
            return out, cand

    for cand in amt_candidates:
        paid_try = pd.to_numeric(df[cand], errors="coerce")
        if paid_try.isna().all():
            continue
        if not _agg_non_negative_by_ay_lag(df, ay_n, lag_n, cand):
            continue
        n_neg = int((paid_try.notna() & (paid_try < -_NEG_CLIP_TOL)).sum())
        if n_neg:
            warnings.append(
                f"«{cand}» has {n_neg} negative row(s); (accident year × lag) totals are non-negative — using this column."
            )
        out = df[[ay_n, lag_n, cand]].rename(
            columns={ay_n: "accident_year", lag_n: "development_lag", cand: "paid_amount"}
        )
        return out, cand

    for cand in amt_candidates:
        paid_try = pd.to_numeric(df[cand], errors="coerce")
        if paid_try.isna().all():
            continue
        n_clip = int((paid_try.notna() & (paid_try < -_NEG_CLIP_TOL)).sum())
        sub = df[[ay_n, lag_n, cand]].copy()
        sub[cand] = pd.to_numeric(sub[cand], errors="coerce").clip(lower=0.0)
        if n_clip:
            warnings.append(
                f"«{cand}»: clipped {n_clip} negative row value(s) to 0 to build the triangle."
            )
        out = sub.rename(
            columns={ay_n: "accident_year", lag_n: "development_lag", cand: "paid_amount"}
        )
        return out, cand

    return None, None


def _wide_to_long(df: pd.DataFrame, accident_col: str) -> pd.DataFrame:
    id_vars = [accident_col]
    value_vars = [c for c in df.columns if c != accident_col]
    long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="_lag_raw", value_name="_val")
    long = long.dropna(subset=["_val"])
    lags: list[int] = []
    for x in long["_lag_raw"]:
        s = str(x).strip()
        slow = s.lower()
        if slow.startswith("dev"):
            s2 = slow.replace("dev_", "", 1).replace("dev", "", 1)
        else:
            s2 = s
        d = "".join(ch for ch in s2 if ch.isdigit() or ch == ".")
        try:
            lags.append(int(float(d)))
        except ValueError:
            lags.append(-999)
    long["_lag"] = lags
    long = long[long["_lag"] != -999]
    out = pd.DataFrame(
        {
            "accident_year": pd.to_numeric(long[accident_col], errors="coerce").astype("Int64"),
            "development_lag": long["_lag"].astype(int),
            "paid_amount": pd.to_numeric(long["_val"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["accident_year", "development_lag", "paid_amount"])
    out["accident_year"] = out["accident_year"].astype(int)
    return out


def _trim_rows_to_inferred_as_of_date(
    df: pd.DataFrame,
    inferred_prelim: dict[str, Any],
    warnings: list[str],
) -> pd.DataFrame:
    """Remove rows observed after inferred valuation date so the triangle is not a full square."""
    if inferred_prelim.get("layout") != "long":
        return df
    ay = inferred_prelim.get("accident_year")
    if not ay or ay not in df.columns:
        return df
    exclude = {
        c
        for c in (
            inferred_prelim.get("accident_year"),
            inferred_prelim.get("development_lag"),
            inferred_prelim.get("paid_amount"),
        )
        if c
    }
    cal = infer_calendar_year_column(df, exclude)
    if not cal or cal not in df.columns:
        return df

    cy = pd.to_numeric(df[cal], errors="coerce")
    ayv = pd.to_numeric(df[ay], errors="coerce")
    m = cy.notna() & ayv.notna()
    if int(m.sum()) < 8:
        return df

    gg = df.loc[m, [ay, cal]].copy()
    gg[cal] = pd.to_numeric(gg[cal], errors="coerce")
    per_ay_max = gg.groupby(ay, sort=False)[cal].max()
    stat_cut = float(per_ay_max.min())
    if pd.isna(stat_cut):
        return df

    name_hint = infer_valuation_year_hint_from_column_names([str(c) for c in df.columns])
    cutoff = stat_cut
    if name_hint is not None and 1950 <= name_hint <= 2100:
        cutoff = min(cutoff, float(name_hint))

    before = len(df)
    keep = cy.isna() | (cy <= cutoff)
    df2 = df.loc[keep].copy()
    dropped = before - len(df2)
    if dropped > 0:
        warnings.append(
            f"As-of filter: kept «{cal}» ≤ {int(cutoff)} (inferred valuation). "
            f"Removed {dropped} row(s) so recent accident years leave future development for the model."
        )
    return df2


def _pivot_has_any_negative_cell(pivot: pd.DataFrame, tol: float = 1e-6) -> bool:
    """True if any aggregated (AY × lag) cell is meaningfully negative."""
    for ay in pivot.index:
        row = pivot.loc[ay].dropna()
        for v in row.tolist():
            if float(v) < -tol:
                return True
    return False


def _pivot_has_any_decrease_along_dev(pivot: pd.DataFrame, tol: float = 1e-6) -> bool:
    """True if cumulative-style levels drop between consecutive development lags (e.g. subrogation)."""
    for ay in pivot.index:
        vals = pivot.loc[ay].dropna().sort_index()
        if len(vals) < 2:
            continue
        for i in range(len(vals) - 1):
            if float(vals.iloc[i + 1]) + tol < float(vals.iloc[i]):
                return True
    return False


def _row_has_increase_then_decrease(vals: list[float], tol: float = 1e-6) -> bool:
    """Cumulative with recoveries: run-off rises, then falls (e.g. 7000 → 5500)."""
    if len(vals) < 3:
        return False
    saw_increase = False
    for i in range(1, len(vals)):
        prev_v, cur_v = float(vals[i - 1]), float(vals[i])
        if cur_v > prev_v + tol:
            saw_increase = True
        elif saw_increase and cur_v + tol < prev_v:
            return True
    return False


def _row_is_non_increasing_with_drop(vals: list[float], tol: float = 1e-6) -> bool:
    """Typical incremental cashflow row: each lag ≤ previous, at least one strict drop (5000 → 500 → 50)."""
    if len(vals) < 2:
        return False
    has_drop = False
    for i in range(1, len(vals)):
        prev_v, cur_v = float(vals[i - 1]), float(vals[i])
        if cur_v > prev_v + tol:
            return False
        if cur_v + tol < prev_v:
            has_drop = True
    return has_drop


def _auto_incremental_needs_cumsum(pivot: pd.DataFrame, tol: float = 1e-6) -> bool:
    """Auto-detect incremental cashflows vs cumulative triangle.

    - Negative aggregated cells → incremental (cumsum).
    - Row with increase then decrease → cumulative with recoveries (no cumsum).
    - Row that only falls (or stays flat) along lags → incremental payments (cumsum).
    - Otherwise non-decreasing → cumulative (no cumsum).
    """
    if _pivot_has_any_negative_cell(pivot, tol=tol):
        return True

    incremental_rows = 0
    recovery_rows = 0
    for ay in pivot.index:
        vals = [float(v) for v in pivot.loc[ay].dropna().sort_index().tolist()]
        if len(vals) < 2:
            continue
        if _row_has_increase_then_decrease(vals, tol=tol):
            recovery_rows += 1
        elif _row_is_non_increasing_with_drop(vals, tol=tol):
            incremental_rows += 1

    if incremental_rows > 0 and recovery_rows == 0:
        return True
    if incremental_rows > recovery_rows:
        return True
    return False


def parse_transaction_dataframe(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str | None] | None = None,
    layout: Literal["auto", "long", "wide"] = "auto",
    values_mode: Literal["auto", "incremental", "cumulative"] = "auto",
) -> tuple[list[dict[str, Any]], ValidationReport]:
    """Build cumulative triangle records from long or wide data.

    column_map keys: accident_year, development_lag, paid_amount — values are *original* column names.
    None means infer (auto) for that slot when using long layout.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return [], ValidationReport(valid=False, errors=["Uploaded file is empty."])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    inferred_prelim = infer_column_mapping(df)
    df = _trim_rows_to_inferred_as_of_date(df, inferred_prelim, warnings)

    inferred = infer_column_mapping(df)
    use_wide = False
    ay_col: str | None = None

    if layout == "wide":
        ay_col = (column_map or {}).get("accident_year")
        if not ay_col:
            ay_col = try_identify_wide_triangle(df)
        if not ay_col or ay_col not in df.columns:
            return [], ValidationReport(
                valid=False,
                errors=[
                    "Wide layout: pick a valid accident-year / row-label column that exists in the file.",
                ],
            )
        use_wide = True
        warnings.append(f"Layout: wide triangle; accident-year column «{ay_col}», other columns as development lags.")
    elif layout == "long":
        use_wide = False
    else:  # auto
        wide_first = try_identify_wide_triangle(df)
        has_long = (
            column_map
            and column_map.get("accident_year")
            and column_map.get("development_lag")
            and column_map.get("paid_amount")
        )
        if wide_first and inferred.get("layout") != "long" and not has_long:
            # Prefer wide if we can't get 3 long columns
            if inferred.get("layout") != "long" or len([k for k in CANONICAL if inferred.get(k)]) < 3:
                use_wide = True
                ay_col = wide_first
                warnings.append(
                    f"Auto-detected wide triangle format (year column «{ay_col}»). "
                    "Switch to Long layout in the sidebar if this is wrong."
                )
        elif inferred.get("layout") == "long":
            use_wide = False
        elif wide_first:
            use_wide = True
            ay_col = wide_first
            warnings.append(
                f"Auto-detected wide triangle (year column «{ay_col}»). "
                "If incorrect, set Layout to Long and map columns manually."
            )
        else:
            use_wide = False

    if use_wide:
        working = _wide_to_long(df, ay_col)
        pa_w = pd.to_numeric(working["paid_amount"], errors="coerce")
        neg_w = pa_w.notna() & (pa_w < -_NEG_CLIP_TOL)
        if neg_w.any():
            warnings.append(
                f"Wide triangle: clipped {int(neg_w.sum())} negative value(s) to 0 before aggregation.",
            )
            working = working.copy()
            working["paid_amount"] = pa_w.clip(lower=0.0)
    else:
        cmap = column_map or {}
        ay_n = cmap.get("accident_year") or inferred.get("accident_year")
        lag_n = cmap.get("development_lag") or inferred.get("development_lag")
        amt_n = cmap.get("paid_amount") or inferred.get("paid_amount")
        missing_roles = [r for r, v in [("accident_year", ay_n), ("development_lag", lag_n), ("paid_amount", amt_n)] if not v]
        if missing_roles:
            return [], ValidationReport(
                valid=False,
                errors=[
                    f"Could not map columns for long format (missing: {missing_roles}). "
                    "Use sidebar column mapping or save file with clearer headers "
                    "(e.g. Accident_Year, Development_Lag, Paid_Amount).",
                ],
            )
        assert ay_n and lag_n and amt_n
        for orig, role in [(ay_n, "accident_year"), (lag_n, "development_lag")]:
            if orig not in df.columns:
                return [], ValidationReport(
                    valid=False,
                    errors=[f"Column «{orig}» not found in file. Available: {list(df.columns)}."],
                )
        if amt_n not in df.columns:
            return [], ValidationReport(
                valid=False,
                errors=[f"Column «{amt_n}» not found in file. Available: {list(df.columns)}."],
            )
        amt_requested = amt_n
        fallback_order = ranked_amount_fallbacks(df, str(ay_n), str(lag_n))
        amt_candidates: list[str] = []
        seen_amt: set[str] = set()
        for c in [amt_n] + fallback_order:
            if not c or c not in df.columns or c in seen_amt:
                continue
            seen_amt.add(str(c))
            amt_candidates.append(str(c))

        working, amt_n_effective = _pick_long_amount_working(
            df, str(ay_n), str(lag_n), amt_candidates, warnings
        )

        if working is None or not amt_n_effective:
            return [], ValidationReport(
                valid=False,
                errors=[
                    "Could not find a usable numeric amount column after trying loss candidates. "
                    "Use Advanced to map «Amount» to paid or incurred losses, or clean the data.",
                ],
            )
        if amt_n_effective != amt_requested:
            warnings.append(
                f"Amount column switched from «{amt_requested}» to «{amt_n_effective}» "
                "(first column that passes row / cell / clip rules).",
            )
        amt_n = amt_n_effective
        extras = [c for c in df.columns if c not in (ay_n, lag_n, amt_n)]
        if extras:
            warnings.append(f"Ignored {len(extras)} extra column(s): {extras[:8]}{'…' if len(extras) > 8 else ''}.")

    paid = pd.to_numeric(working["paid_amount"], errors="coerce")
    if paid.isna().any():
        errors.append("Amount column contains non-numeric values.")

    try:
        ay = working["accident_year"].astype(int)
        lag = working["development_lag"].astype(int)
    except (ValueError, TypeError):
        errors.append("Accident year and development lag must be integers (after mapping).")
        return [], ValidationReport(valid=False, errors=errors)

    if errors:
        return [], ValidationReport(valid=False, errors=errors)

    agg = (
        pd.DataFrame({"accident_year": ay, "development_lag": lag, "paid_amount": paid})
        .groupby(["accident_year", "development_lag"], as_index=False)["paid_amount"]
        .sum()
    )

    pivot_inc = agg.pivot(index="accident_year", columns="development_lag", values="paid_amount")
    pivot_inc = pivot_inc.sort_index(axis=0).sort_index(axis=1)

    for ay_val in pivot_inc.index:
        row = pivot_inc.loc[ay_val]
        present = row.dropna()
        if present.empty:
            continue
        lags = sorted(int(x) for x in present.index.tolist())
        mn, mx = min(lags), max(lags)
        expected = list(range(mn, mx + 1))
        if lags != expected:
            errors.append(
                f"Accident year {ay_val}: development lags must be consecutive "
                f"from {mn} to {mx} with no gaps; found {lags!r}."
            )

    if errors:
        return [], ValidationReport(valid=False, errors=errors)

    need_cumsum: bool
    if values_mode == "incremental":
        need_cumsum = True
        warnings.append("Values treated as incremental (explicit): cumulative sum applied per accident year.")
    elif values_mode == "cumulative":
        need_cumsum = False
        warnings.append("Values treated as already cumulative (explicit).")
    else:
        need_cumsum = _auto_incremental_needs_cumsum(pivot_inc)
        if need_cumsum:
            if _pivot_has_any_negative_cell(pivot_inc):
                warnings.append(
                    "Auto: negative aggregated cell(s) — incremental cashflows; cumulative sum per accident year."
                )
            else:
                warnings.append(
                    "Auto: decreasing payment pattern along lags (typical incremental register) — "
                    "cumulative sum per accident year."
                )
        elif _pivot_has_any_decrease_along_dev(pivot_inc):
            warnings.append(
                "Auto: cumulative triangle with recoveries (increase then decrease along lags) — "
                "cell values used as cumulative incurred/paid (no extra sum). "
                "If amounts are incremental payments per lag, choose «Incremental» in Advanced."
            )
        else:
            warnings.append(
                "Auto: non-decreasing by lag — treating as cumulative incurred/paid (no extra sum)."
            )

    cum = pivot_inc.cumsum(axis=1, skipna=False) if need_cumsum else pivot_inc
    records: list[dict[str, Any]] = []
    for ay_val in cum.index:
        for lag_c in cum.columns:
            val = cum.loc[ay_val, lag_c]
            if pd.notna(val):
                records.append(
                    {
                        "accident_year": int(ay_val),
                        "development_period": int(lag_c),
                        "value": round(float(val), 2),
                    }
                )

    tri_report = validate_triangle_records(records)
    if not tri_report.valid:
        return records, tri_report

    if len(set(r["accident_year"] for r in records)) < 2:
        warnings.append("Very few accident years; chain-ladder and bootstrap may be unstable.")

    return records, ValidationReport(
        valid=True,
        errors=[],
        warnings=warnings + tri_report.warnings,
        data_quality_flags=tri_report.data_quality_flags,
    )


def load_dataframe_from_upload(name: str, data: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Load CSV/XLSX into DataFrame. Returns (df, error_message)."""
    lower = name.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(BytesIO(data)), None
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(data)), None
        return None, "Unsupported file type. Upload .csv or .xlsx."
    except Exception as exc:
        return None, f"Failed to read file: {exc}"


def parse_uploaded_file(
    name: str,
    data: bytes,
    *,
    column_map: dict[str, str | None] | None = None,
    layout: Literal["auto", "long", "wide"] = "auto",
    values_mode: Literal["auto", "incremental", "cumulative"] = "auto",
) -> tuple[list[dict[str, Any]], ValidationReport]:
    df, err = load_dataframe_from_upload(name, data)
    if err or df is None:
        return [], ValidationReport(valid=False, errors=[err or "Failed to load file."])
    return parse_transaction_dataframe(
        df,
        column_map=column_map,
        layout=layout,
        values_mode=values_mode,
    )
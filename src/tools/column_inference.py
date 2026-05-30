"""Heuristics for mapping messy upload columns to accident year / lag / amount (used by UI and parser)."""

from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd


def _normalize_token(s: str) -> str:
    return "".join(c for c in str(s).strip().lower() if c.isalnum() or c == "_")


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


_TOKEN_TO_ROLE: dict[str, str] = {}
for role, synonyms in {
    "accident_year": (
        "accident_year",
        "accidentyear",
        "acc_yr",
        "accyear",
        "ay",
        "origin_year",
        "origyear",
        "underwriting_year",
        "uw_year",
        "policy_year",
        "year",
        "yr",
        "випуску",
    ),
    "development_lag": (
        "development_lag",
        "developmentlag",
        "dev_lag",
        "devlag",
        "lag",
        "period",
        "development_period",
        "dev_period",
        "dp",
        "cohort_period",
        "calendar_period",
        "age",
        "maturity",
    ),
    "paid_amount": (
        "paid_amount",
        "paidamount",
        "paid",
        "amount",
        "paid_loss",
        "paidloss",
        "payments",
        "claims_paid",
        "incurred",
        "loss",
        "value",
        "total",
        "sum",
    ),
}.items():
    for s in synonyms:
        _TOKEN_TO_ROLE[_normalize_token(s)] = role


def _role_score_for_column(col: str, role: str) -> float:
    """Higher = better match. col is original header string."""
    norm = _normalize_token(col)
    if not norm:
        return 0.0
    if norm in _TOKEN_TO_ROLE and _TOKEN_TO_ROLE[norm] == role:
        best = 100.0
    else:
        parts = _normalize_column_name(col).split("_")
        best = 0.0
        for p in parts:
            if not p:
                continue
            if p in _TOKEN_TO_ROLE and _TOKEN_TO_ROLE[p] == role:
                best = max(best, 80.0)
            for token, r in _TOKEN_TO_ROLE.items():
                if r != role:
                    continue
                if len(token) >= 3 and len(p) >= 3 and (token in p or p in token):
                    best = max(best, 40.0)
        if role == "accident_year" and any(x in norm for x in ("year", "yr")) and "dev" not in norm:
            best = max(best, 25.0)
        if role == "development_lag" and any(x in norm for x in ("lag", "dev", "period", "age")):
            best = max(best, 25.0)
        if role == "paid_amount" and any(x in norm for x in ("paid", "amount", "loss", "payment", "claim")):
            best = max(best, 25.0)
    # Calendar development year is not accident year; lag should be DevelopmentLag not DevelopmentYear when both exist.
    if role == "accident_year" and "development" in norm and "accident" not in norm:
        best = min(best, 15.0)
    if role == "development_lag" and norm == "developmentyear":
        best = min(best, 35.0)
    # Prefer cumulative paid / claims over premiums and static reserves for triangle amounts.
    if role == "paid_amount":
        if "cum" in norm and "paid" in norm:
            best += 45.0
        if "cumpaid" in norm:
            best += 15.0
        if "claim" in norm and "paid" in norm:
            best += 20.0
        if "prem" in norm and "paid" not in norm:
            best -= 70.0
        if "earned" in norm:
            best -= 70.0
        if "ceded" in norm:
            best -= 55.0
        if "posted" in norm or "reserve" in norm:
            best -= 45.0
        if "bulk" in norm:
            best -= 20.0
    return max(0.0, best)


def _amount_data_fit_score(df: pd.DataFrame, ay_col: str, lag_col: str, amt_col: str) -> float:
    """0..1 — prefers non-negative, mostly non-decreasing run-off (cumulative paid style)."""
    try:
        t = df[[ay_col, lag_col, amt_col]].copy()
    except KeyError:
        return 0.0
    ay = pd.to_numeric(t[ay_col], errors="coerce")
    lag = pd.to_numeric(t[lag_col], errors="coerce")
    amt = pd.to_numeric(t[amt_col], errors="coerce")
    m = ay.notna() & lag.notna() & amt.notna()
    if int(m.sum()) < 3:
        return 0.0
    neg_frac = float((amt[m] < 0).mean())
    tmp = pd.DataFrame({"_ay": ay[m], "_lag": lag[m], "_amt": amt[m]})
    mono_fracs: list[float] = []
    for _, g in tmp.groupby("_ay"):
        g2 = g.sort_values("_lag")
        vals = g2["_amt"].tolist()
        if len(vals) < 2:
            continue
        ok = sum(1 for i in range(len(vals) - 1) if float(vals[i]) <= float(vals[i + 1]) + 1e-6) / (len(vals) - 1)
        mono_fracs.append(ok)
    mono = float(sum(mono_fracs) / len(mono_fracs)) if mono_fracs else 0.5
    return max(0.0, (1.0 - neg_frac) ** 2) * (0.35 + 0.65 * mono)


def _refine_paid_amount_choice(df: pd.DataFrame, picked: dict[str, str]) -> None:
    """Adjust paid_amount using sample data so premiums / incurred revisions lose to cumulative paid."""
    ay = picked.get("accident_year")
    lag = picked.get("development_lag")
    if not ay or not lag or ay not in df.columns or lag not in df.columns:
        return
    candidates: list[tuple[str, float, float]] = []
    for c in df.columns:
        c = str(c)
        if c in (ay, lag):
            continue
        nsc = _role_score_for_column(c, "paid_amount")
        if nsc < 8.0 and not any(
            k in _normalize_token(c) for k in ("paid", "loss", "incurred", "claim", "bulk")
        ):
            continue
        dsc = _amount_data_fit_score(df, ay, lag, c)
        combined = (nsc / 100.0) * 0.4 + dsc * 0.6
        candidates.append((c, nsc, combined))
    if not candidates:
        return
    candidates.sort(key=lambda x: -x[2])
    picked["paid_amount"] = candidates[0][0]


def _calendar_year_score_for_column(col: str) -> float:
    """Calendar / transaction year of the observation (not accident year, not lag index)."""
    norm = _normalize_token(col)
    if not norm:
        return 0.0
    exact = {
        "developmentyear": 100.0,
        "calyear": 95.0,
        "calendaryear": 95.0,
        "transactionyear": 90.0,
        "reportingyear": 88.0,
        "valuationyear": 95.0,
        "evaluationyear": 90.0,
        "closingyear": 85.0,
        "financialyear": 75.0,
        "fiscalyear": 75.0,
        "asofyear": 92.0,
        "cutoffyear": 90.0,
    }
    if norm in exact:
        return exact[norm]
    best = 0.0
    if "development" in norm and "year" in norm and "lag" not in norm and "accident" not in norm:
        best = max(best, 70.0)
    if any(k in norm for k in ("calendar", "transaction", "report", "valuation", "eval", "closing")) and "year" in norm:
        best = max(best, 65.0)
    return best


def infer_calendar_year_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Pick a column that holds calendar years of development (e.g. DevelopmentYear), if any."""
    best_sc = 0.0
    best_col: str | None = None
    min_rows = max(8, len(df) // 50)
    for c in df.columns:
        cs = str(c)
        if cs in exclude:
            continue
        sc = _calendar_year_score_for_column(cs)
        if sc < 35.0:
            continue
        v = pd.to_numeric(df[cs], errors="coerce")
        vn = v.dropna()
        if len(vn) < min_rows:
            continue
        frac_yearish = float(((vn >= 1900) & (vn <= 2200)).mean())
        if frac_yearish < 0.85:
            continue
        if sc > best_sc:
            best_sc = sc
            best_col = cs
    return best_col


def infer_valuation_year_hint_from_column_names(columns: list[str]) -> int | None:
    """Extract a likely valuation year from headers (e.g. PostedReserves2007)."""
    hints: list[int] = []
    for c in columns:
        cs = str(c).lower()
        if not any(
            k in cs
            for k in (
                "posted",
                "reserve",
                "reserves",
                "valuation",
                "eval",
                "asof",
                "cutoff",
                "triangle",
                "snapshot",
            )
        ):
            continue
        for m in re.finditer(r"\b(19|20)\d{2}\b", str(c)):
            y = int(m.group(0))
            if 1950 <= y <= 2100:
                hints.append(y)
    return min(hints) if hints else None


def lag_label_parses_as_int(header: str) -> bool:
    s = str(header).strip()
    if s.isdigit():
        return True
    low = s.lower()
    if low.startswith("dev"):
        tail = low.replace("dev_", "", 1).replace("dev", "", 1).strip(" _")
        return tail.isdigit()
    return False


def try_identify_wide_triangle(df: pd.DataFrame) -> str | None:
    """Return name of first column if dataframe looks like wide triangle; else None."""
    if df.shape[1] < 3:
        return None
    first = df.columns[0]
    rest = df.iloc[:, 1:]
    numeric_ok = 0
    lag_named = 0
    for c in rest.columns:
        s = pd.to_numeric(rest[c], errors="coerce")
        if s.notna().sum() >= max(1, len(df) // 2):
            numeric_ok += 1
        if lag_label_parses_as_int(str(c)):
            lag_named += 1

    ycol = df[first]
    year_like = pd.to_numeric(ycol, errors="coerce")
    if year_like.notna().sum() < max(1, len(df) // 2):
        return None
    yv = year_like.dropna()
    if yv.empty or not ((yv >= 1800) & (yv <= 2200)).all():
        return None
    if numeric_ok < 2:
        return None
    if lag_named == 0 and numeric_ok < (df.shape[1] - 1):
        return None
    return str(first)


def infer_column_mapping(df: pd.DataFrame) -> dict[str, Any]:
    """Suggest long-format column mapping + layout hint."""
    cols = [str(c) for c in df.columns]
    scores: dict[str, tuple[float, str]] = {
        "accident_year": (0.0, ""),
        "development_lag": (0.0, ""),
        "paid_amount": (0.0, ""),
    }
    for c in cols:
        for role in scores:
            sc = _role_score_for_column(c, role)
            if sc > scores[role][0]:
                scores[role] = (sc, c)

    used: set[str] = set()
    picked: dict[str, str] = {}
    ranked = sorted(
        [(scores[r][0], r, scores[r][1]) for r in scores],
        key=lambda x: -x[0],
    )
    for sc, role, col in ranked:
        if sc < 15 or not col or col in used:
            continue
        picked[role] = col
        used.add(col)

    layout: Literal["long", "wide", "unknown"] = "unknown"
    if len(picked) >= 3:
        layout = "long"
    elif try_identify_wide_triangle(df) is not None:
        layout = "wide"

    if picked.get("accident_year") and picked.get("development_lag"):
        _refine_paid_amount_choice(df, picked)

    return {
        "layout": layout,
        "accident_year": picked.get("accident_year"),
        "development_lag": picked.get("development_lag"),
        "paid_amount": picked.get("paid_amount"),
        "scores": {k: scores[k][0] for k in scores},
    }


def ranked_amount_fallbacks(df: pd.DataFrame, ay_col: str, lag_col: str) -> list[str]:
    """Ordered candidate amount columns when the primary choice fails validation (e.g. negatives)."""
    candidates: list[tuple[str, float]] = []
    for c in df.columns:
        c = str(c)
        if c in (ay_col, lag_col):
            continue
        nsc = _role_score_for_column(c, "paid_amount")
        if nsc < 8.0 and not any(
            k in _normalize_token(c) for k in ("paid", "loss", "incurred", "claim", "bulk")
        ):
            continue
        dsc = _amount_data_fit_score(df, ay_col, lag_col, c)
        combined = (nsc / 100.0) * 0.4 + dsc * 0.6
        candidates.append((c, combined))
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates]


__all__ = [
    "infer_column_mapping",
    "try_identify_wide_triangle",
    "lag_label_parses_as_int",
    "ranked_amount_fallbacks",
    "infer_calendar_year_column",
    "infer_valuation_year_hint_from_column_names",
]

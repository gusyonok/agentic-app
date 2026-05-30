"""Chain-ladder reserving, Mack analytical uncertainty, and ODP Pearson bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm

from core.models import MockActuarialOutput


SCENARIO_MULTIPLIER = {
    "base": 1.0,
    "optimistic": 0.92,
    "stress": 1.15,
}

DEFICIT_PROBABILITY_DEFINITION = (
    "Monte Carlo (bootstrap): P(simulated IBNR exceeds the booked IBNR reserve)."
)
DEFAULT_PROBABILITY_DEFINITION = (
    "Mack analytical: P(future claim payments exceed reserves) via lognormal CDF "
    "(mean = IBNR reserve, σ = Mack standard error)."
)


@dataclass
class ReservingParams:
    """Parameters for building the built-in example triangle (tests / demos only)."""

    scenario: str = "base"
    triangle_size: int = 8
    simulations: int = 10000
    random_seed: int = 42


def load_example_triangle(params: ReservingParams) -> list[dict]:
    multiplier = SCENARIO_MULTIPLIER.get(params.scenario, 1.0)
    example_triangle = {
        2014: [5120, 8170, 9610, 10210, 10520, 10650, 10720, 10760],
        2015: [5340, 8560, 10040, 10710, 11020, 11210, 11320],
        2016: [5580, 8840, 10480, 11250, 11640, 11810],
        2017: [5890, 9220, 11010, 11960, 12420],
        2018: [6210, 9680, 11680, 12790],
        2019: [6480, 10110, 12310],
        2020: [6830, 10620],
        2021: [7210],
    }
    triangle: list[dict] = []
    years = sorted(example_triangle.keys())[-params.triangle_size :]
    for ay in years:
        for dev, value in enumerate(example_triangle[ay], start=1):
            triangle.append(
                {
                    "accident_year": ay,
                    "development_period": dev,
                    "value": round(value * multiplier, 2),
                }
            )
    return triangle


def _ordered_dev_labels(by_ay: dict[int, dict[int, float]]) -> list[int]:
    return sorted({d for m in by_ay.values() for d in m.keys()})


def _records_to_cumulative_matrix(
    triangle_records: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    by_ay: dict[int, dict[int, float]] = {}
    for row in triangle_records:
        ay = int(row["accident_year"])
        dev = int(row["development_period"])
        by_ay.setdefault(ay, {})[dev] = float(row["value"])
    ays = sorted(by_ay.keys())
    dev_labels = _ordered_dev_labels(by_ay)
    n, mcol = len(ays), len(dev_labels)
    dev_to_col = {d: j for j, d in enumerate(dev_labels)}
    c = np.full((n, mcol), np.nan, dtype=float)
    mask = np.zeros((n, mcol), dtype=bool)
    for i, ay in enumerate(ays):
        for dev, val in by_ay[ay].items():
            j = dev_to_col[dev]
            c[i, j] = val
            mask[i, j] = True
    return c, mask, ays, dev_labels


def _volume_weighted_ldfs(c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, m = c.shape
    ldfs: list[float] = []
    for j in range(m - 1):
        num = den = 0.0
        for i in range(c.shape[0]):
            if mask[i, j] and mask[i, j + 1]:
                num += c[i, j + 1]
                den += c[i, j]
        ldfs.append(num / den if den > 0 else 1.0)
    return np.array(ldfs, dtype=float)


def _fitted_cumulative(c: np.ndarray, mask: np.ndarray, ldfs: np.ndarray) -> np.ndarray:
    n, m = c.shape
    c_hat = np.full_like(c, np.nan)
    for i in range(n):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        j0 = int(idx[0])
        c_hat[i, j0] = c[i, j0]
        for j in range(j0 + 1, m):
            c_hat[i, j] = ldfs[j - 1] * c_hat[i, j - 1]
    return c_hat


def _observed_incrementals(c: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = c.shape
    d = np.zeros_like(c)
    d_mask = np.zeros_like(mask)
    for i in range(n):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        j0 = int(idx[0])
        d[i, j0] = c[i, j0]
        d_mask[i, j0] = True
        for j in range(j0 + 1, m):
            if mask[i, j] and mask[i, j - 1]:
                d[i, j] = c[i, j] - c[i, j - 1]
                d_mask[i, j] = True
    return d, d_mask


def _fitted_incrementals(c_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, m = c_hat.shape
    d_hat = np.zeros_like(c_hat)
    for i in range(n):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        j0 = int(idx[0])
        d_hat[i, j0] = c_hat[i, j0]
        for j in range(j0 + 1, m):
            d_hat[i, j] = c_hat[i, j] - c_hat[i, j - 1]
    return d_hat


def compute_chain_ladder_from_triangle_records(
    triangle_records: list[dict],
) -> MockActuarialOutput:
    """Volume-weighted chain ladder on cumulative triangle in long format."""
    by_ay: dict[int, dict[int, float]] = {}
    for row in triangle_records:
        ay = int(row["accident_year"])
        dev = int(row["development_period"])
        by_ay.setdefault(ay, {})[dev] = float(row["value"])

    dev_list = _ordered_dev_labels(by_ay)
    factors: list[float] = []
    for d_curr, d_next in zip(dev_list, dev_list[1:]):
        numerator = 0.0
        denominator = 0.0
        for dev_map in by_ay.values():
            if d_curr in dev_map and d_next in dev_map:
                numerator += dev_map[d_next]
                denominator += dev_map[d_curr]
        factors.append(round(numerator / denominator, 6) if denominator else 1.0)

    cdf: list[float] = []
    for idx in range(len(factors)):
        tail = 1.0
        for later in factors[idx:]:
            tail *= later
        cdf.append(round(tail, 6))

    latest_by_ay: dict[str, float] = {}
    ultimate_by_ay: dict[str, float] = {}
    ibnr_by_ay: dict[str, float] = {}
    reserve_by_ay: dict[str, float] = {}

    for ay in sorted(by_ay):
        dev_map = by_ay[ay]
        latest_dev = max(dev_map)
        latest_val = dev_map[latest_dev]
        latest_by_ay[str(ay)] = round(latest_val, 2)

        try:
            idx_latest = dev_list.index(latest_dev)
        except ValueError:
            idx_latest = len(factors)
        if idx_latest >= len(factors):
            ultimate = latest_val
        else:
            ultimate = latest_val * cdf[idx_latest]
        # Signed IBNR: ultimate − case (latest cumulative). Negative ⇒ over-reserved vs CL ultimate.
        ibnr = ultimate - latest_val
        ultimate_by_ay[str(ay)] = round(ultimate, 2)
        ibnr_by_ay[str(ay)] = round(ibnr, 2)
        reserve_by_ay[str(ay)] = round(ibnr, 2)

    total = round(sum(reserve_by_ay.values()), 2)

    return MockActuarialOutput(
        triangle_records=triangle_records,
        factors=factors,
        cdf=cdf,
        latest_by_ay=latest_by_ay,
        ultimate_by_ay=ultimate_by_ay,
        ibnr_by_ay=ibnr_by_ay,
        reserve_by_ay=reserve_by_ay,
        total_reserve=total,
        diagnostics={
            "triangle_size": len(by_ay),
            "method": "deterministic_chain_ladder",
            "ldf_selection": "volume_weighted",
            "ldf_from_dev": dev_list[:-1],
            "ldf_to_dev": dev_list[1:],
        },
    )


def compute_chain_ladder_reserving(params: ReservingParams) -> MockActuarialOutput:
    triangle = load_example_triangle(params)
    out = compute_chain_ladder_from_triangle_records(triangle_records=triangle)
    out.diagnostics = {
        **out.diagnostics,
        "scenario": params.scenario,
        "dataset": "real_example_triangle",
    }
    return out


def _quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level, method="linear"))


def _pearson_residuals_odp(
    d_obs: np.ndarray,
    d_hat: np.ndarray,
    inc_mask: np.ndarray,
    eps: float = 1e-9,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Pearson residuals r_ij = (D_ij - mu_ij) / sqrt(mu_ij) for ODP (phi=1).

    mu_ij are fitted incrementals from chain-ladder; only cells with inc_mask and mu > eps.
    """
    positions: list[tuple[int, int]] = []
    residuals: list[float] = []
    n, m = d_obs.shape
    for i in range(n):
        for j in range(m):
            if not inc_mask[i, j]:
                continue
            mu = float(d_hat[i, j])
            if mu <= eps:
                continue
            r = (float(d_obs[i, j]) - mu) / np.sqrt(mu)
            positions.append((i, j))
            residuals.append(r)
    return positions, np.array(residuals, dtype=float)


def _cumulative_from_incrementals(
    d: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    n, m = d.shape
    c_new = np.full((n, m), np.nan, dtype=float)
    for i in range(n):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        prev = 0.0
        for j in idx:
            c_new[i, j] = prev + d[i, j]
            prev = c_new[i, j]
    return c_new


def _triangle_records_from_matrix(
    c: np.ndarray,
    mask: np.ndarray,
    ays: list[int],
    dev_labels: list[int],
) -> list[dict]:
    records: list[dict] = []
    for i, ay in enumerate(ays):
        for j in range(c.shape[1]):
            if mask[i, j] and not np.isnan(c[i, j]):
                records.append(
                    {
                        "accident_year": int(ay),
                        "development_period": int(dev_labels[j]),
                        "value": round(float(c[i, j]), 2),
                    }
                )
    return records


def _bootstrap_pearson_residual_ibnr(
    triangle_records: list[dict],
    simulations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Bootstrap IBNR by resampling centered Pearson residuals (ODP / GLM-CL)."""
    rng = np.random.default_rng(seed)
    c, mask, ays, dev_labels = _records_to_cumulative_matrix(triangle_records)
    ldfs = _volume_weighted_ldfs(c, mask)
    c_hat = _fitted_cumulative(c, mask, ldfs)
    d_obs, d_inc_mask = _observed_incrementals(c, mask)
    d_hat = _fitted_incrementals(c_hat, mask)

    positions, r_vec = _pearson_residuals_odp(d_obs, d_hat, d_inc_mask)
    meta: dict[str, float | int | str] = {
        "bootstrap_method": "pearson_odp_chain_ladder",
        "residual_count": int(len(r_vec)),
    }

    if len(r_vec) < 2:
        base = compute_chain_ladder_from_triangle_records(triangle_records).total_reserve
        noise = rng.normal(0.0, max(base * 0.01, 1.0), size=simulations)
        meta["bootstrap_note"] = "insufficient_residuals_fell_back_to_gaussian_noise"
        return np.clip(base + noise, 0.0, None), meta

    r_centered = r_vec - float(np.mean(r_vec))
    eps = 1e-9
    sim_ibnr: list[float] = []
    n, m = d_obs.shape

    for _ in range(simulations):
        d_star = np.zeros_like(d_obs)
        for i in range(n):
            for j in range(m):
                if not d_inc_mask[i, j]:
                    continue
                mu = float(d_hat[i, j])
                if mu <= eps:
                    d_star[i, j] = float(d_obs[i, j])
                else:
                    r_s = float(rng.choice(r_centered))
                    inc = mu + np.sqrt(mu) * r_s
                    d_star[i, j] = max(0.0, inc)
        c_star = _cumulative_from_incrementals(d_star, d_inc_mask)
        boot_records = _triangle_records_from_matrix(c_star, mask, ays, dev_labels)
        sim_ibnr.append(
            compute_chain_ladder_from_triangle_records(boot_records).total_reserve
        )

    return np.array(sim_ibnr, dtype=float), meta


def _mack_process_variances(
    c: np.ndarray,
    mask: np.ndarray,
    ldfs: np.ndarray,
) -> np.ndarray:
    """Mack (1993) process variance sigma_j^2 for each development period."""
    n, m = c.shape
    sigma_sq = np.zeros(m - 1, dtype=float)
    for j in range(m - 1):
        cells: list[float] = []
        for i in range(n):
            if mask[i, j] and mask[i, j + 1] and c[i, j] > 0:
                ratio = c[i, j + 1] / c[i, j]
                cells.append(c[i, j] * (ratio - ldfs[j]) ** 2)
        if len(cells) > 1:
            sigma_sq[j] = float(np.sum(cells) / (len(cells) - 1))
    return sigma_sq


def compute_mack_standard_error(triangle_records: list[dict]) -> dict[str, float | dict[str, float] | list[float]]:
    """Mack standard errors of IBNR by accident year and in aggregate (process variance only)."""
    c, mask, ays, _dev_labels = _records_to_cumulative_matrix(triangle_records)
    ldfs = _volume_weighted_ldfs(c, mask)
    c_hat = _fitted_cumulative(c, mask, ldfs)
    sigma_sq = _mack_process_variances(c, mask, ldfs)
    n, m = c.shape

    se_by_ay: dict[str, float] = {}
    var_total = 0.0
    for i, ay in enumerate(ays):
        idx = np.where(mask[i])[0]
        if idx.size == 0:
            continue
        k = int(idx[-1])
        latest = float(c[i, k])
        ultimate = float(c_hat[i, m - 1])
        ibnr = ultimate - latest
        if latest <= 0 or ibnr <= 1e-9:
            se_by_ay[str(ay)] = 0.0
            continue

        inner = 0.0
        for j in range(k, m - 1):
            c_ij = float(c_hat[i, j])
            fj = float(ldfs[j])
            if c_ij <= 0 or fj <= 0:
                continue
            inner += float(sigma_sq[j] / (fj**2)) * (1.0 / latest + 1.0 / c_ij)
        var_i = (ibnr**2) * inner
        var_total += var_i
        se_by_ay[str(ay)] = round(float(np.sqrt(max(var_i, 0.0))), 4)

    return {
        "mack_se_total": round(float(np.sqrt(max(var_total, 0.0))), 4),
        "mack_se_by_ay": se_by_ay,
        "mack_process_variances": [round(float(v), 6) for v in sigma_sq.tolist()],
    }


def _lognormal_params_from_mean_se(mean: float, se: float) -> tuple[float, float]:
    """Map mean and std on original scale to scipy lognorm (s, scale=exp(mu))."""
    if mean <= 0 or se <= 0:
        return 0.0, mean
    cv_sq = (se / mean) ** 2
    sigma_log = float(np.sqrt(np.log1p(cv_sq)))
    mu_log = float(np.log(mean) - 0.5 * sigma_log**2)
    return sigma_log, mu_log


def compute_mack_lognormal_p_def(base_reserve: float, mack_se: float) -> dict[str, float | str]:
    """Mack default probability: P(IBNR > reserve) via lognormal CDF."""
    base = float(base_reserve)
    se = float(mack_se)
    if base <= 1e-9:
        return {
            "p_def_mack_analytical": 0.0,
            "lognormal_sigma": 0.0,
            "lognormal_mu": 0.0,
            "mack_method": "lognormal_cdf",
            "mack_note": "zero_base_reserve",
        }
    if se <= 1e-9:
        return {
            "p_def_mack_analytical": 0.0,
            "lognormal_sigma": 0.0,
            "lognormal_mu": float(np.log(base)),
            "mack_method": "lognormal_cdf",
            "mack_note": "zero_mack_se",
        }

    sigma_log, mu_log = _lognormal_params_from_mean_se(base, se)
    scale = float(np.exp(mu_log))
    p_def = float(1.0 - lognorm.cdf(base, s=sigma_log, scale=scale))
    return {
        "p_def_mack_analytical": round(p_def, 6),
        "lognormal_sigma": round(sigma_log, 6),
        "lognormal_mu": round(mu_log, 6),
        "mack_method": "lognormal_cdf",
    }


def compute_analytical_mack_default_probability(
    triangle_records: list[dict],
    base_reserve: float,
) -> dict[str, float | dict[str, float] | list[float] | str]:
    """Mack SE + lognormal default probability for total IBNR vs base reserve."""
    mack = compute_mack_standard_error(triangle_records)
    analytical = compute_mack_lognormal_p_def(
        base_reserve=base_reserve,
        mack_se=float(mack["mack_se_total"]),
    )
    return {**mack, **analytical}


def risk_metrics_skipped_zero_ibnr(base_reserve: float) -> dict:
    """Placeholder metrics when best estimate IBNR is zero or negative — no bootstrap."""
    br = round(float(base_reserve), 2)
    negative_be = br < -1e-9
    return {
        "base_reserve": br,
        "risk_analysis_skipped": True,
        "skip_reason": "negative_best_estimate" if negative_be else "zero_best_estimate",
        "reserve_surplus_regime": negative_be,
        "simulations": 0,
        "p_def": 0.0,
        "p_def_bootstrap": 0.0,
        "p_default_mack": 0.0,
        "p_def_mack_analytical": 0.0,
        "deficit_probability_definition": DEFICIT_PROBABILITY_DEFINITION,
        "default_probability_definition": DEFAULT_PROBABILITY_DEFINITION,
        "mack_se_total": 0.0,
        "mack_method": "lognormal_cdf",
        "mack_note": "skipped_zero_or_negative_ibnr",
        "var_95": 0.0,
        "var_995": 0.0,
        "var_999": 0.0,
        "tvar_995": 0.0,
        "rm_required_005": 0.0,
        "rm_millions_005": 0.0,
        "p_def_after_rm": 0.0,
        "heavy_tail": False,
        "tail_ratio_var999_var995": None,
        "tail_gap_tvar995_var995": 0.0,
        "stability_cv_var995": 0.0,
        "stable_10k": True,
        "stability_batch_vars": [],
        "adequacy_note": (
            "Best estimate IBNR is negative: reserves appear redundant vs chain-ladder ultimate; "
            "stochastic tail analysis is not applied."
            if negative_be
            else "IBNR best estimate is zero; tail simulation not applicable."
        ),
        "simulated_ibnr": [],
        "bootstrap_method": "skipped",
        "residual_count": 0,
        "capital_surplus_regime": True,
        "low_p_def_extreme_tail_warning": False,
        "analysis_basis": "gross_before_reinsurance",
    }


def compute_risk_metrics(
    triangle_records: list[dict],
    base_reserve: float,
    simulations: int = 10000,
    seed: int = 42,
) -> dict:
    sim_ibnr, boot_meta = _bootstrap_pearson_residual_ibnr(
        triangle_records=triangle_records,
        simulations=simulations,
        seed=seed,
    )
    mack_analytical = compute_analytical_mack_default_probability(
        triangle_records=triangle_records,
        base_reserve=base_reserve,
    )

    var_95 = _quantile(sim_ibnr, 0.95)
    var_995 = _quantile(sim_ibnr, 0.995)
    var_999 = _quantile(sim_ibnr, 0.999)
    tvar_995 = float(sim_ibnr[sim_ibnr >= var_995].mean())
    p_def_bootstrap = float(np.mean(sim_ibnr > base_reserve))
    p_default_mack = float(mack_analytical.get("p_def_mack_analytical") or 0.0)
    # Risk margin cannot be negative (VaR 99.5% ≤ base ⇒ surplus vs regulatory quantile).
    rm_required = max(0.0, var_995 - base_reserve)
    p_def_after_rm = float(np.mean(sim_ibnr > (base_reserve + rm_required)))

    tail_gap = max(0.0, tvar_995 - var_995)
    tail_ratio = var_999 / var_995 if var_995 > 0 else float("inf")
    heavy_tail = bool(tail_ratio >= 1.1 or (tail_gap / max(var_995, 1e-9)) >= 0.1)

    rel_excess_pct = (
        ((var_995 - base_reserve) / base_reserve * 100.0) if base_reserve > 1e-9 else 0.0
    )
    capital_surplus_regime = bool(var_995 <= base_reserve + 1e-6)
    low_p_def_extreme_tail_warning = bool(
        p_def_bootstrap < 0.10
        and base_reserve > 1e-9
        and rel_excess_pct >= 30.0
    )

    n_batches = 5
    batch_size = max(1000, min(simulations, 10000) // n_batches)
    batch_vars: list[float] = []
    for batch_idx in range(n_batches):
        batch_sim, _ = _bootstrap_pearson_residual_ibnr(
            triangle_records=triangle_records,
            simulations=batch_size,
            seed=seed + 1000 + batch_idx,
        )
        batch_vars.append(_quantile(batch_sim, 0.995))
    stability_cv = float(np.std(batch_vars) / max(np.mean(batch_vars), 1e-9))
    stable_for_10k = stability_cv <= 0.02

    if p_def_bootstrap > 0.5:
        adequacy_note = (
            "P_def is above 50%; base reserves appear insufficient even for moderate outcomes."
        )
    else:
        adequacy_note = "Base reserves are above median stress outcomes, but tail risk remains material."

    rm_millions = rm_required / 1_000_000.0

    return {
        "base_reserve": round(base_reserve, 2),
        "risk_analysis_skipped": False,
        "simulations": int(simulations),
        "p_def": round(p_def_bootstrap, 6),
        "p_def_bootstrap": round(p_def_bootstrap, 6),
        "p_default_mack": round(p_default_mack, 6),
        "p_def_mack_analytical": mack_analytical.get("p_def_mack_analytical", 0.0),
        "deficit_probability_definition": DEFICIT_PROBABILITY_DEFINITION,
        "default_probability_definition": DEFAULT_PROBABILITY_DEFINITION,
        "mack_se_total": mack_analytical.get("mack_se_total", 0.0),
        "mack_method": mack_analytical.get("mack_method", "lognormal_cdf"),
        "lognormal_sigma": mack_analytical.get("lognormal_sigma"),
        "lognormal_mu": mack_analytical.get("lognormal_mu"),
        "mack_note": mack_analytical.get("mack_note"),
        "var_95": round(var_95, 2),
        "var_995": round(var_995, 2),
        "var_999": round(var_999, 2),
        "tvar_995": round(tvar_995, 2),
        "rm_required_005": round(rm_required, 2),
        "rm_millions_005": round(rm_millions, 6),
        "p_def_after_rm": round(p_def_after_rm, 6),
        "heavy_tail": heavy_tail,
        "tail_ratio_var999_var995": round(tail_ratio, 4) if np.isfinite(tail_ratio) else None,
        "tail_gap_tvar995_var995": round(tail_gap, 2),
        "stability_cv_var995": round(stability_cv, 6),
        "stable_10k": stable_for_10k,
        "stability_batch_vars": [round(v, 2) for v in batch_vars],
        "adequacy_note": adequacy_note,
        "simulated_ibnr": [round(float(v), 4) for v in sim_ibnr.tolist()],
        "rel_excess_var995_vs_base_pct": round(rel_excess_pct, 4),
        "capital_surplus_regime": capital_surplus_regime,
        "low_p_def_extreme_tail_warning": low_p_def_extreme_tail_warning,
        "analysis_basis": "gross_before_reinsurance",
        **{k: v for k, v in boot_meta.items()},
        **{
            k: v
            for k, v in mack_analytical.items()
            if k
            not in {
                "p_def_mack_analytical",
                "mack_se_total",
                "mack_method",
                "lognormal_sigma",
                "lognormal_mu",
                "mack_note",
            }
        },
    }


# Backward-compatible alias (tests / older imports).
compute_analytical_mack_deficit_probability = compute_analytical_mack_default_probability

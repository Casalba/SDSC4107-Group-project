"""HK equity health checks, jump tests, and walk-forward GARCH(1,1)+Jump calibration (B1)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, special, stats

from src.common.config import REPO_ROOT
from src.common.metrics import git_provenance
from src.hk.data import HKTable, load_hk_audit, load_hk_daily, write_csv_table

TRADING_DAYS_PER_YEAR = 252.0


def ljung_box_test(series: np.ndarray, lags: int = 10, fit_df: int = 0) -> tuple[float, float]:
    """Compute Ljung-Box Q(lags) portmanteau statistic and chi-square p-value."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= lags + fit_df + 1:
        raise ValueError(f"Series length {n} too short for lags={lags}, fit_df={fit_df}")
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0, 1.0
    q_stat = 0.0
    for k in range(1, lags + 1):
        rho_k = float(np.dot(centered[k:], centered[:-k]) / denom)
        q_stat += (rho_k * rho_k) / (n - k)
    q_stat *= n * (n + 2)
    df = max(1, lags - fit_df)
    return float(q_stat), float(stats.chi2.sf(q_stat, df))


def compute_stylized_facts(log_returns: np.ndarray) -> dict[str, Any]:
    """Compute distributional and dependence stylized facts for daily decimal log-returns."""
    r = np.asarray(log_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        raise ValueError(f"Need >= 30 return observations, got {n}")
    mean_d = float(np.mean(r))
    std_d = float(np.std(r, ddof=1))
    jb_stat, jb_p = stats.jarque_bera(r)
    lb_r10, lb_r10_p = ljung_box_test(r, lags=10)
    lb_r20, lb_r20_p = ljung_box_test(r, lags=20)
    lb_abs10, lb_abs10_p = ljung_box_test(np.abs(r), lags=10)
    lb_abs20, lb_abs20_p = ljung_box_test(np.abs(r), lags=20)
    lb_sq10, lb_sq10_p = ljung_box_test(r * r, lags=10)
    lb_sq20, lb_sq20_p = ljung_box_test(r * r, lags=20)
    return {
        "n_obs": n,
        "daily_mean_pct": mean_d * 100.0,
        "daily_std_pct": std_d * 100.0,
        "ann_mean_pct": mean_d * TRADING_DAYS_PER_YEAR * 100.0,
        "ann_vol_pct": std_d * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0,
        "min_pct": float(np.min(r) * 100.0),
        "max_pct": float(np.max(r) * 100.0),
        "skewness": float(stats.skew(r, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "lb_ret_q10": lb_r10,
        "lb_ret_p10": lb_r10_p,
        "lb_ret_q20": lb_r20,
        "lb_ret_p20": lb_r20_p,
        "lb_abs_q10": lb_abs10,
        "lb_abs_p10": lb_abs10_p,
        "lb_abs_q20": lb_abs20,
        "lb_abs_p20": lb_abs20_p,
        "lb_sq_q10": lb_sq10,
        "lb_sq_p10": lb_sq10_p,
        "lb_sq_q20": lb_sq20,
        "lb_sq_p20": lb_sq20_p,
    }


def lee_mykland_jump_test(
    log_returns: np.ndarray,
    dates: list[str] | np.ndarray | None = None,
    window: int = 60,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Detect daily return jumps via Lee-Mykland (2008) bipower variation L-statistics."""
    r = np.asarray(log_returns, dtype=np.float64)
    n = len(r)
    if n <= window + 2:
        raise ValueError(f"Need > window+2 ({window + 2}) returns, got {n}")
    c_bv = math.sqrt(2.0 / math.pi)
    abs_r = np.abs(r)
    bpv_local = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        seg = abs_r[t - window + 1 : t] * abs_r[t - window : t - 1]
        sigma_sq = float(np.sum(seg)) / ((window - 2) * (c_bv * c_bv))
        bpv_local[t] = max(math.sqrt(sigma_sq), 1e-8)

    valid = np.isfinite(bpv_local)
    l_stat = np.zeros(n, dtype=np.float64)
    l_stat[valid] = abs_r[valid] / bpv_local[valid]
    n_eff = int(np.sum(valid))
    c_n = math.sqrt(2.0 * math.log(n_eff)) / c_bv - (
        math.log(math.pi) + math.log(math.log(n_eff))
    ) / (2.0 * c_bv * math.sqrt(2.0 * math.log(n_eff)))
    s_n = 1.0 / (c_bv * math.sqrt(2.0 * math.log(n_eff)))
    threshold_l = c_n + s_n * (-math.log(-math.log(1.0 - alpha)))

    jump_idx = np.flatnonzero(valid & (l_stat > threshold_l))
    pt_idx = np.flatnonzero(valid & (l_stat > 3.5))
    top_pool = jump_idx if len(jump_idx) >= 5 else pt_idx
    top_idx = top_pool[np.argsort(-l_stat[top_pool])][:10]
    rv = float(np.sum(r[1:] * r[1:]))
    bv = float((math.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1]))
    jump_var_share = max(0.0, (rv - bv) / max(rv, 1e-12))
    top_events = [
        {
            "date": str(dates[idx]) if dates is not None else str(idx),
            "log_return_pct": float(r[idx] * 100.0),
            "l_stat": float(l_stat[idx]),
        }
        for idx in top_idx
    ]
    return {
        "window": int(window),
        "alpha": float(alpha),
        "n_evaluated": n_eff,
        "critical_l_stat": float(threshold_l),
        "n_jumps": int(len(jump_idx)),
        "jump_freq_pct": float(len(jump_idx) / max(1, n_eff) * 100.0),
        "n_jumps_pt35": int(len(pt_idx)),
        "jump_freq_pt35_pct": float(len(pt_idx) / max(1, n_eff) * 100.0),
        "bns_jump_var_share_pct": float(jump_var_share * 100.0),
        "realized_ann_vol_pct": float(
            math.sqrt(rv / max(1, n - 1) * TRADING_DAYS_PER_YEAR) * 100.0
        ),
        "bipower_ann_vol_pct": float(math.sqrt(bv / max(1, n - 1) * TRADING_DAYS_PER_YEAR) * 100.0),
        "top_jumps": top_events,
    }


def compute_regime_stats(table: HKTable, index_col: str = "hsi") -> dict[str, dict[str, float]]:
    """Summarize annualized return, volatility, max drawdown, and excess kurtosis by regime."""
    dates = np.array([str(d) for d in table["date"]])
    prices = np.asarray(table[index_col], dtype=np.float64)
    rets = np.diff(np.log(prices))
    ret_dates = dates[1:]
    ret_prices = prices[1:]

    masks: dict[str, np.ndarray] = {
        "full_2010_2024": np.ones(len(rets), dtype=bool),
        "train_2010_2018": ret_dates <= "2018-12-31",
        "test_2019_2024": ret_dates >= "2019-01-01",
        "regime_2015H2": (ret_dates >= "2015-07-01") & (ret_dates <= "2015-12-31"),
        "regime_2020H1": (ret_dates >= "2020-01-01") & (ret_dates <= "2020-06-30"),
        "regime_2022": (ret_dates >= "2022-01-01") & (ret_dates <= "2022-12-31"),
    }
    masks["non_crisis_other"] = ~(
        masks["regime_2015H2"] | masks["regime_2020H1"] | masks["regime_2022"]
    )

    out: dict[str, dict[str, float]] = {}
    for name, m in masks.items():
        r_seg = rets[m]
        p_seg = ret_prices[m]
        running_max = np.maximum.accumulate(p_seg)
        max_dd = float(np.min(p_seg / running_max - 1.0) * 100.0)
        ex_k = float(stats.kurtosis(r_seg, fisher=True, bias=False)) if len(r_seg) > 10 else 0.0
        out[name] = {
            "n_days": int(len(r_seg)),
            "ann_return_pct": float(np.mean(r_seg) * TRADING_DAYS_PER_YEAR * 100.0),
            "ann_vol_pct": float(np.std(r_seg, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0),
            "excess_kurtosis": ex_k,
            "max_drawdown_pct": max_dd,
            "worst_day_pct": float(np.min(r_seg) * 100.0),
        }
    return out


# ------------------------------------------------------------------ GARCH(1,1) + Jump
def _unpack_garch(theta: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(theta[0])
    omega = float(np.exp(np.clip(theta[1], -14.0, 8.0)))
    logits = np.array([theta[2], theta[3], 0.0], dtype=np.float64)
    logits -= np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return mu, omega, float(probs[0] * 0.9985), float(probs[1] * 0.9985)


def _pack_garch(mu: float, omega: float, alpha: float, beta: float) -> np.ndarray:
    cap = 0.9985
    p_a, p_b = max(alpha / cap, 1e-6), max(beta / cap, 1e-6)
    p_rem = max(1.0 - p_a - p_b, 1e-6)
    return np.array(
        [
            mu,
            math.log(max(omega, 1e-6)),
            math.log(p_a / p_rem),
            math.log(p_b / p_rem),
        ],
        dtype=np.float64,
    )


def _unpack_jump(theta: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    mu, omega, alpha, beta = _unpack_garch(theta[:4])
    lam = float(np.exp(np.clip(theta[4], -12.0, 1.0)))
    mu_j = float(theta[5])
    sigma_j = float(np.exp(np.clip(theta[6], -8.0, 4.0)))
    return mu, omega, alpha, beta, lam, mu_j, sigma_j


def _pack_jump(
    mu: float, omega: float, alpha: float, beta: float, lam: float, mu_j: float, sigma_j: float
) -> np.ndarray:
    base = _pack_garch(mu, omega, alpha, beta)
    return np.concatenate(
        [
            base,
            np.array(
                [math.log(max(lam, 1e-5)), mu_j, math.log(max(sigma_j, 1e-4))], dtype=np.float64
            ),
        ]
    )


def _filter_var_pct(
    y: np.ndarray, mu: float, omega: float, alpha: float, beta: float, init_var: float
) -> np.ndarray:
    sigma2 = np.empty(len(y), dtype=np.float64)
    res2 = (y - mu) ** 2
    s2 = max(float(init_var), 1e-6)
    for t in range(len(y)):
        sigma2[t] = s2
        s2 = omega + alpha * res2[t] + beta * s2
    return sigma2


def _garch_loglik_vec(params: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu, omega, alpha, beta = (float(v) for v in params)
    sigma2 = np.maximum(_filter_var_pct(y, mu, omega, alpha, beta, float(np.var(y, ddof=1))), 1e-8)
    resid = y - mu
    return -0.5 * (math.log(2.0 * math.pi) + np.log(sigma2) + (resid * resid) / sigma2)


def _jump_loglik_vec(params: np.ndarray, y: np.ndarray, k_max: int = 10) -> np.ndarray:
    mu, omega, alpha, beta, lam, mu_j, sigma_j = (float(v) for v in params)
    j_sec = lam * (mu_j * mu_j + sigma_j * sigma_j)
    uncond_diff = max((omega + alpha * j_sec) / max(1e-5, 1.0 - alpha - beta), 1e-4)
    sigma2 = np.maximum(
        _filter_var_pct(y, mu, omega, alpha, beta, min(uncond_diff, float(np.var(y)))), 1e-8
    )
    k = np.arange(k_max + 1, dtype=np.float64)[:, None]
    log_pois = -lam + k * math.log(lam) - special.gammaln(k + 1.0)
    var_k = sigma2[None, :] + k * (sigma_j * sigma_j)
    diff = y[None, :] - ((mu - lam * mu_j) + k * mu_j)
    log_norm = -0.5 * (math.log(2.0 * math.pi) + np.log(var_k) + (diff * diff) / var_k)
    return special.logsumexp(log_pois + log_norm, axis=0)


def _robust_qmle_se(
    theta_hat: np.ndarray,
    y: np.ndarray,
    unpack_fn: Any,
    loglik_fn: Any,
    skip_init: int = 10,
) -> np.ndarray:
    """Bollerslev-Wooldridge sandwich SE mapped from unconstrained theta via Delta Method."""
    p = len(theta_hat)
    h_th = np.full(p, 1e-4, dtype=np.float64)
    psi0 = np.array(unpack_fn(theta_hat), dtype=np.float64)
    ll0 = loglik_fn(psi0, y)[skip_init:]
    scores = np.empty((len(ll0), p), dtype=np.float64)
    jac = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        step = np.zeros(p, dtype=np.float64)
        step[j] = h_th[j]
        psi_p = np.array(unpack_fn(theta_hat + step), dtype=np.float64)
        psi_m = np.array(unpack_fn(theta_hat - step), dtype=np.float64)
        jac[:, j] = (psi_p - psi_m) / (2.0 * h_th[j])
        scores[:, j] = (loglik_fn(psi_p, y)[skip_init:] - loglik_fn(psi_m, y)[skip_init:]) / (
            2.0 * h_th[j]
        )

    meat = scores.T @ scores
    hess = np.zeros((p, p), dtype=np.float64)
    f0 = float(np.sum(ll0))
    for i in range(p):
        for j in range(i, p):
            ei, ej = np.zeros(p, dtype=np.float64), np.zeros(p, dtype=np.float64)
            ei[i], ej[j] = h_th[i], h_th[j]
            if i == j:
                fp = float(np.sum(loglik_fn(np.array(unpack_fn(theta_hat + ei)), y)[skip_init:]))
                fm = float(np.sum(loglik_fn(np.array(unpack_fn(theta_hat - ei)), y)[skip_init:]))
                hess[i, i] = -(fp - 2.0 * f0 + fm) / (h_th[i] * h_th[i])
            else:
                fpp = float(
                    np.sum(loglik_fn(np.array(unpack_fn(theta_hat + ei + ej)), y)[skip_init:])
                )
                fpm = float(
                    np.sum(loglik_fn(np.array(unpack_fn(theta_hat + ei - ej)), y)[skip_init:])
                )
                fmp = float(
                    np.sum(loglik_fn(np.array(unpack_fn(theta_hat - ei + ej)), y)[skip_init:])
                )
                fmm = float(
                    np.sum(loglik_fn(np.array(unpack_fn(theta_hat - ei - ej)), y)[skip_init:])
                )
                val = -(fpp - fpm - fmp + fmm) / (4.0 * h_th[i] * h_th[j])
                hess[i, j] = hess[j, i] = val

    eigvals, eigvecs = np.linalg.eigh(0.5 * (hess + hess.T))
    hess_inv = (eigvecs / np.maximum(eigvals, 1e-4)) @ eigvecs.T
    cov_theta = hess_inv @ meat @ hess_inv
    cov_psi = jac @ cov_theta @ jac.T
    return np.sqrt(np.maximum(np.diag(cov_psi), 1e-12))


def _resid_diags(z: np.ndarray) -> dict[str, float]:
    q10, p10 = ljung_box_test(z, lags=10)
    q20, p20 = ljung_box_test(z, lags=20)
    sq_q10, sq_p10 = ljung_box_test(z * z, lags=10)
    sq_q20, sq_p20 = ljung_box_test(z * z, lags=20)
    return {
        "std_q10": q10,
        "std_p10": p10,
        "std_q20": q20,
        "std_p20": p20,
        "sq_q10": sq_q10,
        "sq_p10": sq_p10,
        "sq_q20": sq_q20,
        "sq_p20": sq_p20,
    }


def fit_garch_jump_mle(log_returns: np.ndarray) -> dict[str, Any]:
    """Fit both GARCH(1,1) and Poisson-Gaussian GARCH(1,1)+Jump via MLE on decimal log-returns."""
    y = np.asarray(log_returns, dtype=np.float64)
    y = y[np.isfinite(y)] * 100.0
    var_y, mean_y = float(np.var(y, ddof=1)), float(np.mean(y))
    n_obs = len(y)

    best_g = None
    for a0, b0 in [(0.06, 0.90), (0.09, 0.86), (0.04, 0.93)]:
        x0 = _pack_garch(mean_y, var_y * (1.0 - a0 - b0), a0, b0)
        res = optimize.minimize(
            lambda th: float(-np.sum(_garch_loglik_vec(np.array(_unpack_garch(th)), y))),
            x0,
            method="BFGS",
            options={"gtol": 1e-6, "maxiter": 400},
        )
        if best_g is None or res.fun < best_g.fun:
            best_g = res
    assert best_g is not None
    g_params = np.array(_unpack_garch(best_g.x), dtype=np.float64)
    g_se = _robust_qmle_se(best_g.x, y, _unpack_garch, _garch_loglik_vec)
    g_ll = float(-best_g.fun)

    mu_g, om_g, al_g, be_g = (float(v) for v in g_params)
    best_j = None
    starts = [
        (0.03, -0.5, 2.2, 0.70 * om_g, max(0.03, 0.65 * al_g), min(0.94, be_g + 0.02)),
        (0.06, -0.3, 1.8, 0.65 * om_g, max(0.03, 0.75 * al_g), be_g),
        (0.015, -1.0, 2.8, 0.80 * om_g, max(0.02, 0.55 * al_g), min(0.95, be_g + 0.03)),
    ]
    for lam0, muj0, sj0, om0, al0, be0 in starts:
        x0 = _pack_jump(mu_g, om0, al0, be0, lam0, muj0, sj0)
        res = optimize.minimize(
            lambda th: float(-np.sum(_jump_loglik_vec(np.array(_unpack_jump(th)), y))),
            x0,
            method="L-BFGS-B",
            options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 600},
        )
        if best_j is None or res.fun < best_j.fun:
            best_j = res
    assert best_j is not None
    j_params = np.array(_unpack_jump(best_j.x), dtype=np.float64)
    j_se = _robust_qmle_se(best_j.x, y, _unpack_jump, _jump_loglik_vec)
    j_ll = float(-best_j.fun)

    mu_j_m, om_j, al_j, be_j, lam_j, jump_mu, jump_sig = (float(v) for v in j_params)
    sig2_g = _filter_var_pct(y, mu_g, om_g, al_g, be_g, var_y)
    z_g = (y - mu_g) / np.sqrt(np.maximum(sig2_g, 1e-8))
    j_sec = lam_j * (jump_mu**2 + jump_sig**2)
    uncond_diff_j = (om_j + al_j * j_sec) / max(1e-6, 1.0 - al_j - be_j)
    sig2_j = _filter_var_pct(y, mu_j_m, om_j, al_j, be_j, min(uncond_diff_j, var_y))
    z_j = (y - mu_j_m) / np.sqrt(np.maximum(sig2_j + j_sec, 1e-8))

    lr_stat = max(0.0, 2.0 * (j_ll - g_ll))
    uncond_ann_vol = math.sqrt(
        (om_j + (1.0 - be_j) * j_sec) / max(1e-6, 1.0 - al_j - be_j) * TRADING_DAYS_PER_YEAR
    )
    return {
        "n_train_obs": n_obs,
        "units": "daily_pct_returns (y_t = 100 * log(S_t/S_{t-1}))",
        "garch11": {
            "mu": mu_g,
            "omega": om_g,
            "alpha": al_g,
            "beta": be_g,
            "persistence": al_g + be_g,
            "se": {
                k: float(v) for k, v in zip(["mu", "omega", "alpha", "beta"], g_se, strict=True)
            },
            "loglik": g_ll,
            "aic": float(8.0 - 2.0 * g_ll),
            "bic": float(4.0 * math.log(n_obs) - 2.0 * g_ll),
            "residuals": _resid_diags(z_g),
        },
        "garch11_jump": {
            "mu": mu_j_m,
            "omega": om_j,
            "alpha": al_j,
            "beta": be_j,
            "persistence": al_j + be_j,
            "lambda_jump": lam_j,
            "mu_jump": jump_mu,
            "sigma_jump": jump_sig,
            "uncond_ann_vol_pct": float(uncond_ann_vol),
            "se": {
                k: float(v)
                for k, v in zip(
                    ["mu", "omega", "alpha", "beta", "lambda_jump", "mu_jump", "sigma_jump"],
                    j_se,
                    strict=True,
                )
            },
            "loglik": j_ll,
            "aic": float(14.0 - 2.0 * j_ll),
            "bic": float(7.0 * math.log(n_obs) - 2.0 * j_ll),
            "residuals": _resid_diags(z_j),
        },
        "lrt_jump_vs_plain": {
            "lr_stat": float(lr_stat),
            "df": 3,
            "p_value": float(stats.chi2.sf(lr_stat, df=3)),
        },
    }


def forecast_causal_volatility(
    prices: np.ndarray,
    calib: dict[str, Any],
    horizon_days: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter causal one-step and horizon-averaged annualized volatility (decimal) along prices."""
    p = np.asarray(prices, dtype=np.float64)
    y = 100.0 * np.diff(np.log(p))
    n = len(p)

    g = calib["garch11"]
    pers_g = min(g["alpha"] + g["beta"], 0.999)
    uncond_g = g["omega"] / max(1e-6, 1.0 - pers_g)
    var_g = np.empty(n, dtype=np.float64)
    s2 = uncond_g
    var_g[0] = s2
    for t, yt in enumerate(y):
        s2 = g["omega"] + g["alpha"] * (yt - g["mu"]) ** 2 + g["beta"] * s2
        var_g[t + 1] = s2
    w_g = (1.0 - pers_g**horizon_days) / (horizon_days * max(1e-6, 1.0 - pers_g))
    ann_g = np.sqrt((uncond_g + (var_g - uncond_g) * w_g) * TRADING_DAYS_PER_YEAR) / 100.0

    j = calib["garch11_jump"]
    j_var = j["lambda_jump"] * (j["mu_jump"] ** 2 + j["sigma_jump"] ** 2)
    pers_j = min(j["alpha"] + j["beta"], 0.999)
    uncond_j = (j["omega"] + j["alpha"] * j_var) / max(1e-6, 1.0 - pers_j)
    var_j = np.empty(n, dtype=np.float64)
    s2_j = uncond_j
    var_j[0] = s2_j
    for t, yt in enumerate(y):
        s2_j = j["omega"] + j["alpha"] * (yt - j["mu"]) ** 2 + j["beta"] * s2_j
        var_j[t + 1] = s2_j
    w_j = (1.0 - pers_j**horizon_days) / (horizon_days * max(1e-6, 1.0 - pers_j))
    ann_j = np.sqrt((uncond_j + (var_j - uncond_j) * w_j + j_var) * TRADING_DAYS_PER_YEAR) / 100.0
    return ann_g, ann_j


def run_calibration(
    train_end: str = "2018-12-31",
    out_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run B1 health checks and strict walk-forward GARCH+Jump calibration on HSI & HSCE."""
    tbl = load_hk_daily()
    audit_path = REPO_ROOT / "data" / "interim" / "hk_daily_full_with_audit.csv"
    tbl_full = load_hk_audit(audit_path) if audit_path.exists() else tbl
    calib_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "results" / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = REPO_ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    prov = git_provenance()
    outputs: dict[str, dict[str, Any]] = {}
    vol_cols: list[str] = ["date"]
    vol_data: dict[str, Any] = {"date": [str(d) for d in tbl_full["date"]]}
    dates = np.array([str(d) for d in tbl["date"]])

    for index_col in ("hsi", "hsce"):
        prices = np.asarray(tbl[index_col], dtype=np.float64)
        log_ret = np.diff(np.log(prices))
        ret_dates = dates[1:]
        train_mask = ret_dates <= train_end
        calib_res = fit_garch_jump_mle(log_ret[train_mask])
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "index": index_col,
            "train_period": f"{dates[0]}..{train_end}",
            "full_period": f"{dates[0]}..{dates[-1]}",
            **prov,
            "stylized_facts_full": compute_stylized_facts(log_ret),
            "stylized_facts_train": compute_stylized_facts(log_ret[train_mask]),
            "jump_test_full": lee_mykland_jump_test(log_ret, dates=ret_dates),
            "regimes": compute_regime_stats(tbl, index_col=index_col),
            "calibration": calib_res,
        }
        (calib_dir / f"{index_col}_garch.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        outputs[index_col] = payload

        ann_g, ann_j = forecast_causal_volatility(
            np.asarray(tbl_full[index_col], dtype=np.float64), calib_res
        )
        vol_cols.extend([f"{index_col}_garch_vol", f"{index_col}_jump_vol"])
        vol_data[f"{index_col}_garch_vol"] = ann_g
        vol_data[f"{index_col}_jump_vol"] = ann_j

    vol_data["vhsi"] = np.asarray(tbl_full["vhsi"], dtype=np.float64)
    vol_data["hsi_vrp0_vol"] = vol_data["hsi_jump_vol"] * 1.00
    vol_data["hsi_vrp3_vol"] = vol_data["hsi_jump_vol"] * 1.03
    vol_data["hsi_vrp5_vol"] = vol_data["hsi_jump_vol"] * 1.05
    vol_cols.extend(["vhsi", "hsi_vrp0_vol", "hsi_vrp3_vol", "hsi_vrp5_vol"])
    write_csv_table(interim_dir / "hk_vol_forecasts.csv", vol_cols, vol_data)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate GARCH(1,1)+Jump models for HSI & HSCE.")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    args = parser.parse_args(argv)
    res = run_calibration(train_end=args.train_end)
    for idx, p in res.items():
        j = p["calibration"]["garch11_jump"]
        r = j["residuals"]
        print(
            f"[{idx.upper()}] alpha={j['alpha']:.4f}, beta={j['beta']:.4f}, "
            f"lam={j['lambda_jump']:.4f}, mu_j={j['mu_jump']:.3f}, sig_j={j['sigma_jump']:.3f}, "
            f"uncond_vol={j['uncond_ann_vol_pct']:.2f}%, "
            f"LB_z_p10={r['std_p10']:.3f}, LB_z2_p10={r['sq_p10']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

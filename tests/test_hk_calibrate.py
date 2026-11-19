"""Unit tests for HK stylized facts, jump tests, GARCH+Jump calibration, and HK YAML configs."""

from __future__ import annotations

import numpy as np
import pytest
from src.common.config import REPO_ROOT, load_config
from src.common.seed import seed_everything
from src.hk.calibrate import (
    compute_stylized_facts,
    fit_garch_jump_mle,
    lee_mykland_jump_test,
    ljung_box_test,
)


def test_ljung_box_and_stylized_facts() -> None:
    seed_everything(42)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.01, size=600)
    sf = compute_stylized_facts(rets)
    assert sf["n_obs"] == 600
    assert 12.0 < sf["ann_vol_pct"] < 20.0
    assert sf["lb_ret_p10"] > 0.01
    q_wn, p_wn = ljung_box_test(rets, lags=10)
    assert q_wn > 0.0 and p_wn > 0.01


def test_lee_mykland_detects_synthetic_jump() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.01, size=300)
    rets[200] = -0.09  # -9 sigma shock
    res = lee_mykland_jump_test(rets, window=60, alpha=0.01)
    assert res["n_jumps"] >= 1
    assert any(abs(ev["log_return_pct"] + 9.0) < 1e-6 for ev in res["top_jumps"])


def test_garch_jump_mle_convergence_on_synthetic_series() -> None:
    rng = np.random.default_rng(123)
    n = 1500
    omega, alpha, beta = 0.06, 0.08, 0.87
    lam, mu_j, sig_j = 0.03, -0.8, 2.2
    sigma2 = (omega + alpha * lam * (mu_j**2 + sig_j**2)) / (1.0 - alpha - beta)
    y = np.empty(n, dtype=np.float64)
    for t in range(n):
        nj = rng.poisson(lam)
        j_sum = rng.normal(mu_j, sig_j, size=nj).sum() if nj > 0 else 0.0
        shock = np.sqrt(sigma2) * rng.normal() + j_sum - lam * mu_j
        y[t] = 0.02 + shock
        sigma2 = omega + alpha * (shock**2) + beta * sigma2

    fit = fit_garch_jump_mle(y * 0.01)
    j = fit["garch11_jump"]
    assert 0.85 < j["persistence"] < 0.995
    assert j["lambda_jump"] > 0.0
    assert all(np.isfinite(v) and v > 0.0 for v in j["se"].values())
    # Residuals must exhibit no significant autocorrelation
    assert j["residuals"]["std_p10"] > 0.05
    assert j["residuals"]["sq_p10"] > 0.05
    assert fit["lrt_jump_vs_plain"]["p_value"] < 0.05


@pytest.mark.parametrize("cfg_name", ["fut1bp.yaml", "fut10bp.yaml", "etf30bp.yaml"])
def test_hk_yaml_configs_load_cleanly(cfg_name: str) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "hk" / cfg_name)
    assert cfg["exp"] == "hk"
    assert cfg["n_steps"] == 30
    assert cfg["market"]["d"] == 1
    assert float(cfg["eps"]) > 0.0
    assert cfg["hk"]["train_end"] == "2018-12-31"

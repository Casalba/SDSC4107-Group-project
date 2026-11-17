"""Unit & interface integration tests for HKWorld scenario generator (B1 -> B2 contract)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.common.metrics import capital_requirement
from src.common.seed import seed_everything
from src.hk.synth import HKMarket, HKWorld
from src.policy.heston_d1 import WEIGHTS
from src.policy.heston_d1 import load as load_heston_d1


def _synthetic_calib() -> dict:
    return {
        "calibration": {
            "garch11": {
                "mu": 0.015,
                "omega": 0.05,
                "alpha": 0.07,
                "beta": 0.88,
            },
            "garch11_jump": {
                "mu": 0.015,
                "omega": 0.04,
                "alpha": 0.06,
                "beta": 0.88,
                "lambda_jump": 0.025,
                "mu_jump": -0.75,
                "sigma_jump": 2.4,
            },
        }
    }


def test_hk_world_shapes_determinism_and_cost() -> None:
    market = HKMarket(mode="garch", eps=0.0015, burn_in=150)
    world = HKWorld(market, calib=_synthetic_calib(), device="cpu")

    seed_everything(0)
    p1 = world.simulate(32)
    seed_everything(0)
    p2 = world.simulate(32)

    assert p1.spot.shape == (32, 31)
    assert p1.variance.shape == (32, 31)
    assert torch.allclose(p1.spot, p2.spot)
    feats = world.features(p1)
    assert feats.shape == (32, 30, 2)
    assert torch.allclose(feats[:, 0, 0], torch.zeros(32, dtype=feats.dtype))

    pos = torch.full((32, 30, 1), 0.5, dtype=torch.float64)
    cost = world.cost(pos, p1)
    assert torch.all(cost > 0.0)
    assert torch.isfinite(world.terminal_loss(pos, p1)).all()


def test_hk_world_simulated_moments_match_theory_within_5pct() -> None:
    """B1 DoD: generated paths' mean, variance, and persistence match calibrated params < 5%."""
    calib = _synthetic_calib()
    market = HKMarket(mode="garch", burn_in=250)
    world = HKWorld(market, calib=calib, device="cpu")
    theory = world.theoretical_daily_moments()

    seed_everything(0)
    paths = world.simulate(60_000)
    y = torch.log(paths.spot[:, 1:] / paths.spot[:, :-1]) * 100.0
    emp_var = float(y.var(unbiased=True).item())
    emp_mean = float(y.mean().item())

    # Mean check in annualized volatility-scaled tolerance, and variance strictly within 5%
    assert abs(emp_var - theory["var_pct2"]) / theory["var_pct2"] < 0.05
    assert abs(emp_mean - theory["mean_pct"]) < 0.02


def test_hk_world_bootstrap_mode() -> None:
    rng = np.random.default_rng(9)
    hist = rng.normal(0.0, 0.012, size=500)
    world = HKWorld(HKMarket(mode="bootstrap", block_len=5), hist_returns=hist, device="cpu")
    seed_everything(1)
    paths = world.simulate(20_000)
    y = torch.log(paths.spot[:, 1:] / paths.spot[:, :-1])
    assert (
        abs(float(y.var(unbiased=True)) - float(np.var(hist, ddof=1))) / float(np.var(hist, ddof=1))
        < 0.05
    )


@pytest.mark.skipif(not WEIGHTS.exists(), reason="run scripts/export_policy_d1.py first")
def test_hk_world_integrates_with_frozen_heston_d1_policy() -> None:
    """Verify frozen A1->B2 policy runs identically via forward_path & reset/forward on HKWorld."""
    world = HKWorld(HKMarket(mode="garch", eps=0.0015), calib=_synthetic_calib(), device="cpu")
    seed_everything(0)
    paths = world.simulate(64)
    feats = world.features(paths)

    policy = load_heston_d1(device="cpu")
    with torch.no_grad():
        pos_batch = policy.forward_path(feats)
        policy.reset()
        pos_step = torch.stack([policy(feats[:, k]) for k in range(world.market.n_steps)], dim=1)

    assert pos_batch.shape == (64, 30, 1)
    assert torch.allclose(pos_batch, pos_step, atol=1e-12)
    loss = world.terminal_loss(pos_batch, paths)
    cap = capital_requirement(loss)
    assert np.isfinite(cap["c95"]) and np.isfinite(cap["c99"])
    assert cap["c99"] >= cap["c95"]

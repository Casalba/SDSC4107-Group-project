"""Unit tests for HK backtest engine, BS delta benchmark, and 500-share lot rounding (B2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import norm
from src.hk.backtest import DeltaHedgePolicy, apply_etf_lot_rounding, run_hk_backtest
from src.hk.data import HKWindow


class _FixedSchedulePolicy:
    """Deterministic test policy returning a pre-specified position schedule."""

    def __init__(self, schedule: list[float]) -> None:
        self._schedule = schedule
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def __call__(self, x_k: torch.Tensor) -> torch.Tensor:
        val = self._schedule[self._step]
        self._step += 1
        return torch.full((x_k.shape[0], 1), val, dtype=x_k.dtype, device=x_k.device)


def _make_window(
    spot: np.ndarray,
    futures: np.ndarray | None = None,
    etf_adj: np.ndarray | None = None,
    etf_raw: np.ndarray | None = None,
    strike: float | None = None,
    r: float = 0.02,
    q: float = 0.03,
    vhsi: float = 0.20,
) -> HKWindow:
    n_pts = len(spot)
    return HKWindow(
        month="2020-01",
        split="test",
        dates=tuple(f"2020-01-{i + 1:02d}" for i in range(n_pts)),
        spot=spot.astype(np.float64),
        futures=(spot if futures is None else futures).astype(np.float64),
        etf_adj=(spot / 1000.0 if etf_adj is None else etf_adj).astype(np.float64),
        etf_raw=(spot / 1000.0 if etf_raw is None else etf_raw).astype(np.float64),
        r=np.full(n_pts, r, dtype=np.float64),
        q=np.full(n_pts, q, dtype=np.float64),
        vhsi=np.full(n_pts, vhsi, dtype=np.float64),
        strike=float(spot[0] if strike is None else strike),
    )


def test_single_window_hand_calculation() -> None:
    """Hand-verify payoff, hedge PnL, proportional cost (eq. 5.7), and terminal loss."""
    # 3-step path: normalized spot = [100, 102, 101, 105], strike = 100
    spot = np.array([20000.0, 20400.0, 20200.0, 21000.0], dtype=np.float64)
    win = _make_window(spot=spot, strike=20000.0)
    schedule = [0.5, 0.7, 0.4]
    eps = 0.001

    cfg = {
        "n_steps": 3,
        "eps": eps,
        "market": {"instrument": "hsi_futures_synthetic"},
        "stats": {"block_size": 1, "n_boot": 50, "boot_seed": 42},
    }
    res = run_hk_backtest(_FixedSchedulePolicy(schedule), [win], cfg, device="cpu")

    # Hand calculation on S0 = 100 normalized scale:
    # Payoff: max(105 - 100, 0) = 5.0
    # Hedge PnL: 0.5*(102-100) + 0.7*(101-102) + 0.4*(105-101) = 1.0 - 0.7 + 1.6 = 1.9
    # Cost (initial build charged, no terminal liquidation):
    #   eps * (|0.5-0|*100 + |0.7-0.5|*102 + |0.4-0.7|*101) = 0.001 * (50 + 20.4 + 30.3) = 0.1007
    # Terminal loss: 5.0 - 1.9 + 0.1007 = 3.2007
    # Turnover: 0.5 + 0.2 + 0.3 = 1.0
    assert res["arrays"]["payoff"][0] == pytest.approx(5.0, abs=1e-12)
    assert res["arrays"]["hedge_pnl"][0] == pytest.approx(1.9, abs=1e-12)
    assert res["arrays"]["cost"][0] == pytest.approx(0.1007, abs=1e-12)
    assert res["arrays"]["terminal_loss"][0] == pytest.approx(3.2007, abs=1e-12)
    assert res["arrays"]["turnover"][0] == pytest.approx(1.0, abs=1e-12)
    assert res["rounding_error"] is None


def test_etf_board_lot_rounding() -> None:
    """Verify 500-share board-lot discretization and ETF backtest rounding summary."""
    # S0 = 20,000 * 50 HKD = 1,000,000 HKD notional; ETF price = 20.0 HKD -> 1.0 delta = 50,000 shares
    spot = torch.tensor([[20000.0, 20200.0, 20400.0]], dtype=torch.float64)
    etf_raw = torch.tensor([[20.0, 20.2, 20.4]], dtype=torch.float64)
    # Continuous positions: 0.503 -> 25,150 shares -> rounds to 25,000; 0.51 -> 25,247.5 -> rounds to 25,000
    pos = torch.tensor([[0.503, 0.51]], dtype=torch.float64)

    eff_pos, rounded_shares = apply_etf_lot_rounding(
        positions=pos, spot_paths=spot, etf_raw_paths=etf_raw, lot_size=500, index_multiplier=50.0
    )
    assert torch.all(rounded_shares % 500.0 == 0.0)
    assert rounded_shares[0, 0].item() == pytest.approx(25000.0)
    assert eff_pos[0, 0].item() == pytest.approx(0.50, abs=1e-12)

    win = _make_window(
        spot=np.array([20000.0, 20200.0, 20400.0]),
        etf_adj=np.array([20.0, 20.2, 20.4]),
        etf_raw=np.array([20.0, 20.2, 20.4]),
    )
    cfg = {
        "n_steps": 2,
        "eps": 0.0015,
        "hk": {"instrument": "etf2800_total_return", "lot_size_shares": 500},
        "stats": {"block_size": 1, "n_boot": 50, "boot_seed": 42},
    }
    res = run_hk_backtest(_FixedSchedulePolicy([0.503, 0.51]), [win], cfg, device="cpu")
    assert res["rounding_error"] is not None
    assert res["rounding_error"]["abs_mean_loss_diff"] >= 0.0


def test_delta_hedge_matches_analytical_bs() -> None:
    """DeltaHedgePolicy must match analytical BS Call Delta within < 1e-6 (spec §6 B2 #3)."""
    n_steps = 30
    dt = 1.0 / 365.0
    rng = np.random.default_rng(7)
    spot = 25000.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 0.01, n_steps)])))
    r, q, sigma = 0.025, 0.032, 0.22
    strike = 25000.0
    win = _make_window(spot=spot, strike=strike, r=r, q=q, vhsi=sigma)

    policy = DeltaHedgePolicy(n_steps=n_steps, dt=dt)
    cfg = {
        "n_steps": n_steps,
        "eps": 0.0,
        "market": {"instrument": "hsi_futures_synthetic"},
        "stats": {"block_size": 1, "n_boot": 50, "boot_seed": 42},
    }
    res = run_hk_backtest(policy, [win], cfg, device="cpu")
    got_deltas = res["arrays"]["positions"][0]

    expected_deltas = []
    for k in range(n_steps):
        tau = (n_steps - k) * dt
        d1 = (math.log(spot[k] / strike) + (r - q + 0.5 * sigma**2) * tau) / (
            sigma * math.sqrt(tau)
        )
        expected_deltas.append(math.exp(-q * tau) * norm.cdf(d1))

    max_err = float(np.max(np.abs(got_deltas - np.asarray(expected_deltas))))
    assert max_err < 1e-6

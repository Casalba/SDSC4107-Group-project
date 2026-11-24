"""Unit tests for HK loss accounting and moving/circular block bootstrap inference (B2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.common.metrics import capital_requirement, proportional_cost, turnover
from src.hk.stats import (
    block_bootstrap_ci,
    compute_window_losses,
    paired_block_bootstrap,
    summarize_hk_losses,
)


def test_compute_window_losses_matches_common_metrics() -> None:
    """HK loss summary must reuse and exactly match src/common/metrics.py."""
    rng = np.random.default_rng(123)
    m_win, n_steps = 40, 10
    prices = 100.0 * np.exp(
        np.cumsum(
            np.hstack([np.zeros((m_win, 1)), rng.normal(0.0, 0.015, (m_win, n_steps))]), axis=1
        )
    )
    positions = rng.uniform(0.2, 0.8, size=(m_win, n_steps))
    eps = 0.0015

    res = compute_window_losses(prices=prices, positions=positions, strikes=100.0, eps=eps)
    summary = summarize_hk_losses(
        losses=res["terminal_loss"],
        positions=positions,
        hedge_prices=prices,
        eps=eps,
    )

    cap = capital_requirement(res["terminal_loss"])
    pos_t = torch.as_tensor(positions, dtype=torch.float64).unsqueeze(-1)
    p_t = torch.as_tensor(prices, dtype=torch.float64).unsqueeze(-1)
    expected_turn = float(turnover(pos_t).sum(dim=-1).mean().item())
    expected_cost_notional = float(
        (proportional_cost(pos_t, p_t, eps=eps) / p_t[:, 0, 0]).mean().item()
    )

    assert summary["c95"] == pytest.approx(cap["c95"], abs=1e-12)
    assert summary["c99"] == pytest.approx(cap["c99"], abs=1e-12)
    assert summary["turnover"] == pytest.approx(expected_turn, abs=1e-12)
    assert summary["cost_notional"] == pytest.approx(expected_cost_notional, abs=1e-12)


def test_block_bootstrap_ci_reproducible_and_well_ordered() -> None:
    """Circular block bootstrap (block=12, B=2000) should be deterministic and bracket point stats."""
    rng = np.random.default_rng(2026)
    losses = rng.normal(loc=2.5, scale=1.2, size=180)

    out1 = block_bootstrap_ci(losses, block_size=12, n_boot=2000, ci_level=0.95, seed=42)
    out2 = block_bootstrap_ci(losses, block_size=12, n_boot=2000, ci_level=0.95, seed=42)

    assert out1 == out2
    assert out1["n_obs"] == 180
    assert out1["block_size"] == 12
    assert out1["n_boot"] == 2000

    for key in ("mean_loss", "c95", "c99"):
        blk = out1[key]
        assert blk["ci_low"] <= blk["point"] <= blk["ci_high"]
        assert blk["std"] > 0.0

    # For N(2.5, 1.2^2) with N=180, true mean 2.5 should lie inside the 95% CI
    assert out1["mean_loss"]["ci_low"] < 2.5 < out1["mean_loss"]["ci_high"]


def test_paired_block_bootstrap_identical_and_constant_shift() -> None:
    """Paired bootstrap uses shared block indices: identical -> 0 diff; constant shift -> exact."""
    rng = np.random.default_rng(99)
    base = rng.normal(loc=3.0, scale=1.0, size=180)

    same = paired_block_bootstrap(base, base, block_size=12, n_boot=500, seed=42)
    assert same["diff_c95"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert same["diff_c95"]["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert same["diff_c95"]["ci_high"] == pytest.approx(0.0, abs=1e-12)
    assert same["diff_c95"]["ci_excludes_zero"] is False
    assert same["diff_c95"]["p1_within_5pct"] is True
    assert same["diff_c95"]["p2_strictly_better"] is False

    # Constant -1.0 improvement in every window -> exact -1.0 diff across all bootstrap resamples
    better = base - 1.0
    shifted = paired_block_bootstrap(better, base, block_size=12, n_boot=500, seed=42)
    assert shifted["diff_c95"]["point"] == pytest.approx(-1.0, abs=1e-12)
    assert shifted["diff_c95"]["ci_low"] == pytest.approx(-1.0, abs=1e-12)
    assert shifted["diff_c95"]["ci_high"] == pytest.approx(-1.0, abs=1e-12)
    assert shifted["diff_c95"]["ci_excludes_zero"] is True
    assert shifted["diff_c95"]["p2_strictly_better"] is True
    assert shifted["diff_c99"]["point"] == pytest.approx(-1.0, abs=1e-12)

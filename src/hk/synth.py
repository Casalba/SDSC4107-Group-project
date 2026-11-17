"""Hong Kong market scenario generator aligned with A-group World interface (B1)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from src.common.config import REPO_ROOT
from src.common.metrics import proportional_cost
from src.common.seed import resolve_device
from src.hk.data import load_hk_daily


@dataclass(frozen=True)
class HKMarket:
    """HK single-asset (d=1) market configuration aligned with BSMarket / HestonMarket.

    `mode`: 'garch' (calibrated GARCH(1,1)+Jump, M2 primary), 'garch_plain', or 'bootstrap'
    (circular block bootstrap over 2010-2018 training returns, robustness benchmark).
    """

    index: str = "hsi"
    mode: str = "garch"
    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 30.0 / 365.0
    n_steps: int = 30
    eps: float = 0.0
    d: int = 1
    zero_mean: bool = False
    burn_in: int = 200
    block_len: int = 5
    calib_path: str | Path | None = None

    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def n_features(self) -> int:
        return 2

    def time_grid(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.arange(self.n_steps + 1, dtype=dtype, device=device) * self.dt


class HKPaths(NamedTuple):
    """Simulated HK daily paths: spot (B, n+1), annualized variance (B, n+1), time (n+1,)."""

    spot: Tensor
    variance: Tensor
    time: Tensor


class HKWorld:
    """HK scenario world implementing A1's World protocol (simulate/features/prices/payoff/cost)."""

    def __init__(
        self,
        market: HKMarket,
        calib: Mapping[str, Any] | None = None,
        hist_returns: np.ndarray | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if market.d != 1:
            raise ValueError(f"HKWorld only supports d=1 (no HK variance swap), got d={market.d}")
        self.market = market
        self.device = resolve_device(device)
        self.dtype = dtype

        if calib is not None:
            self.calib: dict[str, Any] = dict(calib)
        elif market.mode in {"garch", "garch_plain"}:
            cpath = (
                Path(market.calib_path)
                if market.calib_path is not None
                else REPO_ROOT / "results" / "calib" / f"{market.index}_garch.json"
            )
            if not cpath.exists():
                raise FileNotFoundError(
                    f"Calibration JSON not found: {cpath}; run `python -m src.hk.calibrate`"
                )
            self.calib = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            self.calib = {}

        if hist_returns is not None:
            self._hist_ret = torch.as_tensor(
                np.asarray(hist_returns, dtype=np.float64), dtype=self.dtype, device=self.device
            )
        elif market.mode == "bootstrap":
            df = load_hk_daily()
            train_df = df.loc[df["date"] <= "2018-12-31"]
            log_r = np.diff(np.log(train_df[market.index].to_numpy(dtype=np.float64)))
            self._hist_ret = torch.as_tensor(log_r, dtype=self.dtype, device=self.device)
        else:
            self._hist_ret = None

    def _model_params(self) -> dict[str, float]:
        """Extract percentage-return dynamics parameters for GARCH or GARCH+Jump simulation."""
        cal_block = self.calib.get("calibration", self.calib)
        if self.market.mode == "garch":
            p = cal_block["garch11_jump"]
            return {
                "mu": 0.0 if self.market.zero_mean else float(p["mu"]),
                "omega": float(p["omega"]),
                "alpha": float(p["alpha"]),
                "beta": float(p["beta"]),
                "lambda_jump": float(p["lambda_jump"]),
                "mu_jump": float(p["mu_jump"]),
                "sigma_jump": float(p["sigma_jump"]),
            }
        p = cal_block["garch11"]
        return {
            "mu": 0.0 if self.market.zero_mean else float(p["mu"]),
            "omega": float(p["omega"]),
            "alpha": float(p["alpha"]),
            "beta": float(p["beta"]),
            "lambda_jump": 0.0,
            "mu_jump": 0.0,
            "sigma_jump": 0.0,
        }

    def theoretical_daily_moments(self) -> dict[str, float]:
        """Return analytical stationary daily pct-return mean, variance, and persistence."""
        if self.market.mode == "bootstrap":
            assert self._hist_ret is not None
            y = self._hist_ret * 100.0
            return {
                "mean_pct": float(y.mean().item()),
                "var_pct2": float(y.var(unbiased=True).item()),
                "ann_vol_pct": float((y.std(unbiased=True) * (252.0**0.5)).item()),
            }
        p = self._model_params()
        j_sec = p["lambda_jump"] * (p["mu_jump"] ** 2 + p["sigma_jump"] ** 2)
        denom = max(1e-6, 1.0 - p["alpha"] - p["beta"])
        exp_diff_var = (p["omega"] + p["alpha"] * j_sec) / denom
        total_var = exp_diff_var + j_sec
        return {
            "mean_pct": p["mu"],
            "diff_var_pct2": exp_diff_var,
            "var_pct2": total_var,
            "ann_vol_pct": float((total_var * 252.0) ** 0.5),
            "persistence": p["alpha"] + p["beta"],
        }

    def simulate(self, n_paths: int, init_var_pct2: float | None = None) -> HKPaths:
        """Sample `n_paths` daily paths of length `n_steps + 1` using global torch RNG."""
        m = self.market
        if m.mode == "bootstrap":
            return self._simulate_bootstrap(n_paths)
        return self._simulate_garch(n_paths, init_var_pct2=init_var_pct2)

    def _simulate_garch(self, n_paths: int, init_var_pct2: float | None = None) -> HKPaths:
        m = self.market
        p = self._model_params()
        mu = p["mu"]
        omega = p["omega"]
        alpha = p["alpha"]
        beta = p["beta"]
        lam = p["lambda_jump"]
        mu_j = p["mu_jump"]
        sig_j = p["sigma_jump"]

        j_sec = lam * (mu_j * mu_j + sig_j * sig_j)
        denom = max(1e-6, 1.0 - alpha - beta)
        uncond_diff = (omega + alpha * j_sec) / denom
        start_var = uncond_diff if init_var_pct2 is None else float(init_var_pct2)

        n_total = (0 if init_var_pct2 is not None else m.burn_in) + m.n_steps
        sigma2 = torch.full((n_paths,), start_var, dtype=self.dtype, device=self.device)
        z_all = torch.randn((n_paths, n_total), dtype=self.dtype, device=self.device)
        if lam > 0.0:
            rate = torch.full((n_paths, n_total), lam, dtype=self.dtype, device=self.device)
            n_jumps = torch.poisson(rate)
            z_jump = torch.randn((n_paths, n_total), dtype=self.dtype, device=self.device)
            jump_all = n_jumps * mu_j + torch.sqrt(n_jumps) * sig_j * z_jump
        else:
            jump_all = torch.zeros((n_paths, n_total), dtype=self.dtype, device=self.device)

        keep_offset = n_total - m.n_steps
        log_rets = torch.empty((n_paths, m.n_steps), dtype=self.dtype, device=self.device)
        ann_vars = torch.empty((n_paths, m.n_steps + 1), dtype=self.dtype, device=self.device)
        # Convert percentage^2 daily variance to decimal annualized variance: (* 252 / 1e4).
        scale_to_ann = 252.0 / 10000.0

        for step in range(n_total):
            if step >= keep_offset:
                ann_vars[:, step - keep_offset] = (sigma2 + j_sec) * scale_to_ann
            shock = torch.sqrt(sigma2) * z_all[:, step] + jump_all[:, step] - lam * mu_j
            y_t = mu + shock
            if step >= keep_offset:
                log_rets[:, step - keep_offset] = y_t * 0.01
            sigma2 = torch.clamp(omega + alpha * (shock * shock) + beta * sigma2, min=1e-8)

        ann_vars[:, -1] = (sigma2 + j_sec) * scale_to_ann
        cum_log = torch.cat(
            [torch.zeros((n_paths, 1), dtype=self.dtype, device=self.device), log_rets.cumsum(1)],
            dim=1,
        )
        return HKPaths(
            spot=m.s0 * cum_log.exp(),
            variance=ann_vars,
            time=m.time_grid(self.dtype, self.device),
        )

    def _simulate_bootstrap(self, n_paths: int) -> HKPaths:
        if self._hist_ret is None or len(self._hist_ret) < self.market.n_steps:
            raise ValueError("Need historical training returns for bootstrap simulation")
        m = self.market
        pool = self._hist_ret
        t_len = int(pool.numel())
        blk = max(1, m.block_len)
        n_blocks = (m.n_steps + blk - 1) // blk
        starts = torch.randint(0, t_len, (n_paths, n_blocks, 1), device=self.device)
        offsets = torch.arange(blk, device=self.device).view(1, 1, blk)
        idx = (starts + offsets) % t_len
        sampled = pool[idx].reshape(n_paths, n_blocks * blk)[:, : m.n_steps]
        if m.zero_mean:
            sampled = sampled - pool.mean()
        cum_log = torch.cat(
            [torch.zeros((n_paths, 1), dtype=self.dtype, device=self.device), sampled.cumsum(1)],
            dim=1,
        )
        ann_var = torch.full(
            (n_paths, m.n_steps + 1),
            float((pool.var(unbiased=True) * 252.0).item()),
            dtype=self.dtype,
            device=self.device,
        )
        return HKPaths(
            spot=m.s0 * cum_log.exp(),
            variance=ann_var,
            time=m.time_grid(self.dtype, self.device),
        )

    def features(self, paths: HKPaths) -> Tensor:
        """Build policy input tensor (B, n_steps, 2) = [log(S_k/S_0), t_k/T] per spec §5.1."""
        m = self.market
        n = m.n_steps
        log_ret = torch.log(paths.spot[:, :n] / m.s0)
        t_norm = (paths.time[:n] / m.maturity).unsqueeze(0).expand(int(paths.spot.shape[0]), n)
        return torch.stack([log_ret, t_norm], dim=-1)

    def prices(self, paths: HKPaths) -> Tensor:
        """Return tradable asset price tensor (B, n_steps + 1, 1)."""
        return paths.spot.unsqueeze(-1)

    def payoff(self, paths: HKPaths) -> Tensor:
        """European call payoff (S_T - K)+ with terminal shape (B,)."""
        return torch.clamp(paths.spot[:, -1] - self.market.strike, min=0.0)

    def cost(self, positions: Tensor, paths: HKPaths) -> Tensor:
        """Proportional transaction cost per paper Eq. (5.7) using shared common.metrics."""
        return proportional_cost(positions, self.prices(paths), self.market.eps)

    def terminal_loss(self, positions: Tensor, paths: HKPaths) -> Tensor:
        """Zero-initial-capital terminal hedging loss L = Z - (delta·S)_T + cost."""
        prices = self.prices(paths)
        hedge = (positions * (prices[:, 1:] - prices[:, :-1])).sum(dim=-1).sum(dim=-1)
        return self.payoff(paths) - hedge + self.cost(positions, paths)

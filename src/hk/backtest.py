
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import norm

from src.hk.data import HKWindow
from src.hk.stats import block_bootstrap_ci, compute_window_losses, summarize_hk_losses
from src.policy.policy import Policy


class DeltaHedgePolicy:

    def __init__(self, n_steps: int = 30, dt: float = 1.0 / 365.0) -> None:
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self._step: int = 0
        self._s0: np.ndarray | None = None
        self._k: np.ndarray | None = None
        self._r: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def set_market_context(
        self,
        s0: np.ndarray | float,
        strike: np.ndarray | float,
        r_path: np.ndarray,
        q_path: np.ndarray,
        sigma_path: np.ndarray,
    ) -> None:
        r_arr = np.atleast_2d(np.asarray(r_path, dtype=np.float64))
        q_arr = np.atleast_2d(np.asarray(q_path, dtype=np.float64))
        sig_arr = np.atleast_2d(np.asarray(sigma_path, dtype=np.float64))
        batch = r_arr.shape[0]

        self._s0 = np.broadcast_to(np.asarray(s0, dtype=np.float64), (batch,)).copy()
        self._k = np.broadcast_to(np.asarray(strike, dtype=np.float64), (batch,)).copy()
        self._r = r_arr
        self._q = q_arr
        self._sigma = sig_arr

    def reset(self) -> None:
        self._step = 0

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        if (
            self._s0 is None
            or self._k is None
            or self._r is None
            or self._q is None
            or self._sigma is None
        ):
            raise RuntimeError("Call set_market_context() before running DeltaHedgePolicy.")

        k_idx = min(self._step, self.n_steps - 1)
        log_m = x_k[:, 0].detach().cpu().numpy()
        spot_k = self._s0 * np.exp(log_m)
        tau = max((self.n_steps - k_idx) * self.dt, 0.0)

        r_k = self._r[:, k_idx]
        q_k = self._q[:, k_idx]
        sig_k = np.maximum(self._sigma[:, k_idx], 1e-6)

        if tau <= 1e-12:
            delta = (spot_k > self._k).astype(np.float64)
        else:
            sqrt_tau = math.sqrt(tau)
            d1 = (np.log(spot_k / self._k) + (r_k - q_k + 0.5 * sig_k**2) * tau) / (
                sig_k * sqrt_tau
            )
            delta = np.exp(-q_k * tau) * norm.cdf(d1)

        self._step += 1
        return torch.as_tensor(delta[:, None], dtype=x_k.dtype, device=x_k.device)

    def __call__(self, x_k: torch.Tensor) -> torch.Tensor:
        return self.forward(x_k)


def apply_etf_lot_rounding(
    positions: torch.Tensor,
    spot_paths: torch.Tensor,
    etf_raw_paths: torch.Tensor,
    lot_size: int = 500,
    index_multiplier: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    option_notional_hkd = spot_paths[:, :1] * float(index_multiplier)
    cont_shares = positions * (option_notional_hkd / torch.clamp(etf_raw_paths[:, :-1], min=1e-6))
    rounded_shares = torch.round(cont_shares / float(lot_size)) * float(lot_size)
    eff_positions = rounded_shares * etf_raw_paths[:, :-1] / option_notional_hkd
    return eff_positions, rounded_shares


@torch.no_grad()
def run_hk_backtest(
    policy: Policy,
    windows: list[HKWindow],
    cfg: dict[str, Any],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    if not windows:
        raise ValueError("windows list cannot be empty")

    n_steps = int(cfg.get("n_steps", 30))
    eps = float(cfg.get("eps", 0.0))
    market_cfg = cfg.get("market", {})
    hk_cfg = cfg.get("hk_backtest", cfg.get("hk", {}))
    stats_cfg = cfg.get("stats", hk_cfg.get("bootstrap", {}))

    instrument = str(market_cfg.get("instrument", hk_cfg.get("instrument", "synthetic_futures")))
    lot_size = hk_cfg.get("lot_size", hk_cfg.get("lot_size_shares"))
    multiplier = float(hk_cfg.get("contract_multiplier_hkd", 50.0))

    m_win = len(windows)
    raw_s = torch.tensor(np.stack([w.spot for w in windows]), dtype=torch.float64, device=device)
    raw_f = torch.tensor(np.stack([w.futures for w in windows]), dtype=torch.float64, device=device)
    raw_etf_adj = torch.tensor(
        np.stack([w.etf_adj for w in windows]), dtype=torch.float64, device=device
    )
    raw_etf_unadj = torch.tensor(
        np.stack([w.etf_raw for w in windows]), dtype=torch.float64, device=device
    )
    raw_k = torch.tensor([w.strike for w in windows], dtype=torch.float64, device=device)

    norm_s = 100.0 * (raw_s / raw_s[:, :1])
    norm_f = 100.0 * (raw_f / raw_f[:, :1])
    norm_k = 100.0 * (raw_k / raw_s[:, 0])

    if isinstance(policy, DeltaHedgePolicy):
        policy.set_market_context(
            s0=np.array([w.spot[0] for w in windows]),
            strike=np.array([w.strike for w in windows]),
            r_path=np.stack([w.r for w in windows]),
            q_path=np.stack([w.q for w in windows]),
            sigma_path=np.stack([w.vhsi for w in windows]),
        )

    policy.reset()
    pos_list: list[torch.Tensor] = []
    log_m = torch.log(norm_s[:, :-1] / norm_s[:, :1])
    for k in range(n_steps):
        t_feat = torch.full(
            (m_win, 1), float(k) / float(n_steps), dtype=torch.float64, device=device
        )
        x_k = torch.cat([log_m[:, k : k + 1], t_feat], dim=1)
        pos_list.append(policy(x_k))

    cont_positions = torch.cat(pos_list, dim=1)

    rounding_error_summary: dict[str, float] | None = None
    if "etf" in instrument:
        norm_etf = 100.0 * (raw_etf_adj / raw_etf_adj[:, :1])
        cont_res = compute_window_losses(
            prices=norm_s, positions=cont_positions, strikes=norm_k, eps=eps, hedge_prices=norm_etf
        )
        if lot_size is not None and int(lot_size) > 0:
            eff_positions, _ = apply_etf_lot_rounding(
                positions=cont_positions,
                spot_paths=raw_s,
                etf_raw_paths=raw_etf_unadj,
                lot_size=int(lot_size),
                index_multiplier=multiplier,
            )
            final_res = compute_window_losses(
                prices=norm_s,
                positions=eff_positions,
                strikes=norm_k,
                eps=eps,
                hedge_prices=norm_etf,
            )
            diff_loss = (final_res["terminal_loss"] - cont_res["terminal_loss"]).cpu().numpy()
            rounding_error_summary = {
                "mean_loss_diff": float(np.mean(diff_loss)),
                "abs_mean_loss_diff": float(np.mean(np.abs(diff_loss))),
                "max_abs_loss_diff": float(np.max(np.abs(diff_loss))),
            }
        else:
            eff_positions = cont_positions
            final_res = cont_res
        hedge_path_used = norm_etf
    else:
        eff_positions = cont_positions
        hedge_path_used = norm_f
        final_res = compute_window_losses(
            prices=norm_s,
            positions=eff_positions,
            strikes=norm_k,
            eps=eps,
            hedge_prices=hedge_path_used,
        )

    losses_np = final_res["terminal_loss"].cpu().numpy()
    summary = summarize_hk_losses(
        losses=losses_np,
        positions=eff_positions,
        hedge_prices=hedge_path_used,
        eps=eps,
    )
    boot = block_bootstrap_ci(
        losses=losses_np,
        block_size=int(stats_cfg.get("block_size", 12)),
        n_boot=int(stats_cfg.get("n_boot", 2000)),
        ci_level=float(stats_cfg.get("ci_level", 0.95)),
        seed=int(stats_cfg.get("boot_seed", 42)),
    )

    return {
        "window_ids": [w.month for w in windows],
        "splits": [w.split for w in windows],
        "summary": summary,
        "bootstrap": boot,
        "rounding_error": rounding_error_summary,
        "arrays": {
            "terminal_loss": losses_np,
            "payoff": final_res["payoff"].cpu().numpy(),
            "hedge_pnl": final_res["hedge_pnl"].cpu().numpy(),
            "cost": final_res["cost"].cpu().numpy(),
            "turnover": final_res["turnover"].cpu().numpy(),
            "positions": eff_positions.cpu().numpy(),
            "cont_positions": cont_positions.cpu().numpy(),
        },
    }

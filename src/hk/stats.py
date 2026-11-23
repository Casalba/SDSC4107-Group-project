"""港股交叉驗證統計與檢定模組（B2 核心元件，對齊 plan.md §4.7 與 src/common/metrics.py）。

提供：
1. compute_window_losses: 單/多窗口零期初資本終端損失 L = Z - Hedge_PnL + Cost 核算。
2. summarize_hk_losses: 計算主指標 C*95、C*99、mean_loss、turnover、cost_notional。
3. block_bootstrap_ci: 單策略 Monthly Block Bootstrap（block=12, B=2000）點估計與 95% CI。
4. paired_block_bootstrap: 雙策略成對 Block Bootstrap 差值檢定（供 P1 / P2 / P4 假設檢定）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.common.metrics import capital_requirement, proportional_cost, turnover


def compute_window_losses(
    prices: torch.Tensor | np.ndarray,
    positions: torch.Tensor | np.ndarray,
    strikes: torch.Tensor | np.ndarray | float,
    eps: float,
    hedge_prices: torch.Tensor | np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """計算零期初資本下的終端損失 L = Z - (delta · H)_T + Cost。

    參數：
        prices: 標的指數/現貨路徑，shape (M, n_steps + 1) 或 (M, n_steps + 1, 1)。
        positions: 每日目標/有效倉位，shape (M, n_steps) 或 (M, n_steps, 1)，預設 delta_{-1} = 0。
        strikes: 各窗口 ATM Call 行權價 K，標量或 shape (M,)。
        eps: 單邊比例成本率（例如期貨 1bp 往返 -> eps=0.00005；ETF 30bp 往返 -> eps=0.0015）。
        hedge_prices: 對沖工具價格序列（若為期貨腿則傳入合成期貨 F_t；若為 None 則預設等於 prices）。

    回傳：
        包含 terminal_loss, payoff, hedge_pnl, cost, turnover, cost_notional 的張量字典，shape 均為 (M,)。
    """
    s_path = torch.as_tensor(prices, dtype=torch.float64)
    if s_path.ndim == 3:
        s_path = s_path.squeeze(-1)

    pos = torch.as_tensor(positions, dtype=torch.float64)
    if pos.ndim == 3:
        pos = pos.squeeze(-1)

    if s_path.ndim != 2 or pos.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors (M, T+1) and (M, T), got {s_path.shape} and {pos.shape}"
        )
    if s_path.shape[0] != pos.shape[0] or s_path.shape[1] != pos.shape[1] + 1:
        raise ValueError(f"Shape mismatch: prices {s_path.shape} vs positions {pos.shape}")

    if hedge_prices is None:
        h_path = s_path
    else:
        h_path = torch.as_tensor(hedge_prices, dtype=torch.float64)
        if h_path.ndim == 3:
            h_path = h_path.squeeze(-1)
        if h_path.shape != s_path.shape:
            raise ValueError(f"hedge_prices shape {h_path.shape} must match prices {s_path.shape}")

    k_tensor = torch.as_tensor(strikes, dtype=torch.float64, device=s_path.device)
    if k_tensor.ndim == 0:
        k_tensor = k_tensor.expand(s_path.shape[0])

    # 1. 期權到期賠付 Z = max(S_T - K, 0)
    payoff = torch.clamp(s_path[:, -1] - k_tensor, min=0.0)

    # 2. 對沖交易損益 sum_{k=0}^{n-1} delta_k * (H_{k+1} - H_k)
    d_hedge = h_path[:, 1:] - h_path[:, :-1]
    hedge_pnl = (pos * d_hedge).sum(dim=-1)

    # 3. 比例交易成本（首次建倉 delta_{-1}=0 計費，到期不強制平倉，對齊論文式 5.7）
    cost = proportional_cost(pos.unsqueeze(-1), h_path.unsqueeze(-1), eps=float(eps))

    # 4. 零期初資本下的終端損失 L = Z - Hedge_PnL + Cost
    terminal_loss = payoff - hedge_pnl + cost

    # 5. 換手率與成本佔名義本金比例
    window_turnover = turnover(pos.unsqueeze(-1)).sum(dim=-1)
    cost_notional = cost / torch.clamp(h_path[:, 0], min=1e-12)

    return {
        "terminal_loss": terminal_loss,
        "payoff": payoff,
        "hedge_pnl": hedge_pnl,
        "cost": cost,
        "turnover": window_turnover,
        "cost_notional": cost_notional,
    }


def summarize_hk_losses(
    losses: torch.Tensor | np.ndarray,
    positions: torch.Tensor | np.ndarray | None = None,
    hedge_prices: torch.Tensor | np.ndarray | None = None,
    eps: float = 0.0,
) -> dict[str, float]:
    """匯總港股窗口主指標：C*95, C*99, mean_loss, turnover, cost_notional。"""
    loss_t = torch.as_tensor(losses, dtype=torch.float64).reshape(-1)
    cap = capital_requirement(loss_t)
    summary: dict[str, float] = {
        "c95": float(cap["c95"]),
        "c99": float(cap["c99"]),
        "mean_loss": float(loss_t.mean().item()),
    }

    if positions is not None:
        pos = torch.as_tensor(positions, dtype=torch.float64)
        if pos.ndim == 2:
            pos = pos.unsqueeze(-1)
        summary["turnover"] = float(turnover(pos).sum(dim=-1).mean().item())

        if hedge_prices is not None:
            h_path = torch.as_tensor(hedge_prices, dtype=torch.float64)
            if h_path.ndim == 2:
                h_path = h_path.unsqueeze(-1)
            cost_vec = proportional_cost(pos, h_path, eps=float(eps))
            notional = torch.clamp(h_path[:, 0, 0], min=1e-12)
            summary["cost_notional"] = float((cost_vec / notional).mean().item())

    return summary


def _sample_circular_block_indices(
    n_obs: int,
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成 Circular Block Bootstrap 索引矩陣，shape (n_boot, n_obs)。"""
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    b_len = min(int(block_size), n_obs)
    n_blocks = math.ceil(n_obs / b_len)

    # 隨機抽取起始月份索引 (n_boot, n_blocks)
    starts = rng.integers(0, n_obs, size=(n_boot, n_blocks))
    offsets = np.arange(b_len)[None, None, :]
    # 環形取模確保首尾月份被抽到的機率均等
    idx = (starts[:, :, None] + offsets) % n_obs
    return idx.reshape(n_boot, n_blocks * b_len)[:, :n_obs]


def block_bootstrap_ci(
    losses: torch.Tensor | np.ndarray,
    block_size: int = 12,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """單策略 Monthly Block Bootstrap（預設 block=12, B=2000），回傳點估計與 95% CI。"""
    arr = np.asarray(losses, dtype=np.float64).reshape(-1)
    n_obs = arr.shape[0]
    if n_obs == 0:
        raise ValueError("losses cannot be empty")

    point_cap = capital_requirement(torch.from_numpy(arr))
    point_mean = float(np.mean(arr))

    rng = np.random.default_rng(seed)
    boot_idx = _sample_circular_block_indices(n_obs, block_size=block_size, n_boot=n_boot, rng=rng)
    boot_samples = torch.from_numpy(arr[boot_idx])  # (n_boot, n_obs)

    k95 = max(1, math.ceil((1.0 - 0.95) * n_obs - 1e-12))
    k99 = max(1, math.ceil((1.0 - 0.99) * n_obs - 1e-12))

    c95_dist = torch.topk(boot_samples, k95, largest=True, dim=1).values.mean(dim=1).numpy()
    c99_dist = torch.topk(boot_samples, k99, largest=True, dim=1).values.mean(dim=1).numpy()
    mean_dist = boot_samples.mean(dim=1).numpy()

    alpha_tail = (1.0 - ci_level) / 2.0 * 100.0
    q_low, q_high = alpha_tail, 100.0 - alpha_tail

    return {
        "n_obs": int(n_obs),
        "block_size": int(block_size),
        "n_boot": int(n_boot),
        "c95": {
            "point": float(point_cap["c95"]),
            "ci_low": float(np.percentile(c95_dist, q_low)),
            "ci_high": float(np.percentile(c95_dist, q_high)),
            "std": float(np.std(c95_dist, ddof=1)),
        },
        "c99": {
            "point": float(point_cap["c99"]),
            "ci_low": float(np.percentile(c99_dist, q_low)),
            "ci_high": float(np.percentile(c99_dist, q_high)),
            "std": float(np.std(c99_dist, ddof=1)),
        },
        "mean_loss": {
            "point": point_mean,
            "ci_low": float(np.percentile(mean_dist, q_low)),
            "ci_high": float(np.percentile(mean_dist, q_high)),
            "std": float(np.std(mean_dist, ddof=1)),
        },
    }


def paired_block_bootstrap(
    losses_a: torch.Tensor | np.ndarray,
    losses_b: torch.Tensor | np.ndarray,
    block_size: int = 12,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """雙策略成對 Block Bootstrap 檢定（計算 A - B 差值與相對差異的 95% CI）。

    用於檢定：
    - P1（1bp 期貨腿）：|rel_diff_c95| < 5%
    - P2（30bp ETF 腿）：diff_c95 < 0 且 ci_excludes_zero == True
    - P4（M2 vs M1）：本地重訓相較 Zero-shot 的 C*95 改善幅度
    """
    arr_a = np.asarray(losses_a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(losses_b, dtype=np.float64).reshape(-1)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"losses_a {arr_a.shape} and losses_b {arr_b.shape} must have identical shape"
        )

    n_obs = arr_a.shape[0]
    cap_a = capital_requirement(torch.from_numpy(arr_a))
    cap_b = capital_requirement(torch.from_numpy(arr_b))

    point_diff_c95 = float(cap_a["c95"] - cap_b["c95"])
    point_rel_c95 = float(point_diff_c95 / max(abs(cap_b["c95"]), 1e-12))
    point_diff_c99 = float(cap_a["c99"] - cap_b["c99"])

    rng = np.random.default_rng(seed)
    # 關鍵：A 與 B 必須共用完全相同的隨機區塊索引
    boot_idx = _sample_circular_block_indices(n_obs, block_size=block_size, n_boot=n_boot, rng=rng)
    boot_a = torch.from_numpy(arr_a[boot_idx])
    boot_b = torch.from_numpy(arr_b[boot_idx])

    k95 = max(1, math.ceil((1.0 - 0.95) * n_obs - 1e-12))
    k99 = max(1, math.ceil((1.0 - 0.99) * n_obs - 1e-12))

    c95_a = torch.topk(boot_a, k95, largest=True, dim=1).values.mean(dim=1).numpy()
    c95_b = torch.topk(boot_b, k95, largest=True, dim=1).values.mean(dim=1).numpy()
    c99_a = torch.topk(boot_a, k99, largest=True, dim=1).values.mean(dim=1).numpy()
    c99_b = torch.topk(boot_b, k99, largest=True, dim=1).values.mean(dim=1).numpy()

    diff_c95_dist = c95_a - c95_b
    rel_c95_dist = diff_c95_dist / np.maximum(np.abs(c95_b), 1e-12)
    diff_c99_dist = c99_a - c99_b

    alpha_tail = (1.0 - ci_level) / 2.0 * 100.0
    q_low, q_high = alpha_tail, 100.0 - alpha_tail

    c95_ci_low = float(np.percentile(diff_c95_dist, q_low))
    c95_ci_high = float(np.percentile(diff_c95_dist, q_high))
    excludes_zero = bool((c95_ci_low > 0.0) or (c95_ci_high < 0.0))

    return {
        "n_obs": int(n_obs),
        "block_size": int(block_size),
        "n_boot": int(n_boot),
        "diff_c95": {
            "point": point_diff_c95,
            "ci_low": c95_ci_low,
            "ci_high": c95_ci_high,
            "ci_excludes_zero": excludes_zero,
            "p1_within_5pct": bool(abs(point_rel_c95) < 0.05),
            "p2_strictly_better": bool((point_diff_c95 < 0.0) and excludes_zero),
        },
        "rel_diff_c95": {
            "point": point_rel_c95,
            "ci_low": float(np.percentile(rel_c95_dist, q_low)),
            "ci_high": float(np.percentile(rel_c95_dist, q_high)),
        },
        "diff_c99": {
            "point": point_diff_c99,
            "ci_low": float(np.percentile(diff_c99_dist, q_low)),
            "ci_high": float(np.percentile(diff_c99_dist, q_high)),
        },
    }

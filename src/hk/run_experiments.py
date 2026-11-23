"""
用法（專案根目錄）：
    python -m src.hk.run_experiments --split all --tag hsi_main
    python -m src.hk.run_experiments --split test --tag hsi_main
    python -m src.hk.run_experiments --index hsce --split all --tag hsce_robustness
"""


from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import norm

from src.common.config import load_config
from src.common.seed import resolve_device
from src.hk.backtest import DeltaHedgePolicy, run_hk_backtest
from src.hk.data import HKWindow, extract_monthly_windows
from src.hk.stats import paired_block_bootstrap
from src.hk.train_hk import load_m2_policy
from src.policy.heston_d1 import WEIGHTS as D1_WEIGHTS
from src.policy.heston_d1 import load as load_d1_policy


def discover_m2_checkpoints(m2_root: Path, run_group: str, seeds: list[int]) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for s in seeds:
        ckpt = m2_root / run_group / str(s) / "policy.pt"
        if ckpt.exists():
            found.append((s, ckpt))
    return found


def aggregate_multi_seed_backtests(
    seed_results: list[tuple[int, dict[str, Any]]],
    bench_losses: np.ndarray,
    m1_losses: np.ndarray | None,
    block_size: int = 12,
    n_boot: int = 2000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    metrics_keys = ["c95", "c99", "mean_loss", "turnover", "cost_notional"]
    agg_metrics: dict[str, dict[str, float]] = {}
    for k in metrics_keys:
        vals = [res["summary"][k] for _, res in seed_results if k in res["summary"]]
        if vals:
            agg_metrics[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    seed0_res = seed_results[0][1]
    seed0_losses = seed0_res["arrays"]["terminal_loss"]
    mean_loss_series = np.mean([res["arrays"]["terminal_loss"] for _, res in seed_results], axis=0)

    vs_bench_seed0 = paired_block_bootstrap(
        losses_a=seed0_losses,
        losses_b=bench_losses,
        block_size=block_size,
        n_boot=n_boot,
        seed=boot_seed,
    )
    vs_bench_ens = paired_block_bootstrap(
        losses_a=mean_loss_series,
        losses_b=bench_losses,
        block_size=block_size,
        n_boot=n_boot,
        seed=boot_seed,
    )

    vs_m1 = None
    if m1_losses is not None:
        vs_m1 = paired_block_bootstrap(
            losses_a=seed0_losses,
            losses_b=m1_losses,
            block_size=block_size,
            n_boot=n_boot,
            seed=boot_seed,
        )

    return {
        "n_seeds": len(seed_results),
        "seeds": [s for s, _ in seed_results],
        "metrics_mean_std": agg_metrics,
        "paired_vs_bench_seed0": vs_bench_seed0,
        "paired_vs_bench_mean_series": vs_bench_ens,
        "paired_vs_m1_seed0": vs_m1,
        "seed0_bootstrap": seed0_res["bootstrap"],
        "seed0_rounding_error": seed0_res["rounding_error"],
        "seed0_losses": seed0_losses,
    }


def compute_premium_comparison_table(
    windows: list[HKWindow],
    strategy_losses: dict[str, np.ndarray],
    vrp_grid: list[float] = [0.0, 0.03, 0.05],
) -> dict[str, Any]:
    t_mat = 30.0 / 365.0
    sqrt_t = math.sqrt(t_mat)
    s0_norm = 100.0

    def bs_norm_call_price(k_norm: np.ndarray, r0: np.ndarray, q0: np.ndarray, sig: np.ndarray) -> np.ndarray:
        sig_safe = np.maximum(sig, 1e-6)
        d1 = (np.log(s0_norm / k_norm) + (r0 - q0 + 0.5 * sig_safe**2) * t_mat) / (sig_safe * sqrt_t)
        d2 = d1 - sig_safe * sqrt_t
        return s0_norm * np.exp(-q0 * t_mat) * norm.cdf(d1) - k_norm * np.exp(-r0 * t_mat) * norm.cdf(d2)

    k_arr = np.array([100.0 * (w.strike / w.spot[0]) for w in windows], dtype=np.float64)
    r_arr = np.array([w.r[0] for w in windows], dtype=np.float64)
    q_arr = np.array([w.q[0] for w in windows], dtype=np.float64)
    vhsi0_arr = np.array([w.vhsi[0] for w in windows], dtype=np.float64)

    table: dict[str, Any] = {}
    prem_vhsi = bs_norm_call_price(k_arr, r_arr, q_arr, vhsi0_arr)
    table["vhsi_actual_iv"] = {
        name: float(np.mean(prem_vhsi - losses)) for name, losses in strategy_losses.items()
    }

    rankings: list[list[str]] = []
    for vrp in vrp_grid:
        prem = bs_norm_call_price(k_arr, r_arr, q_arr, vhsi0_arr * (1.0 + vrp))
        net_pnl_means = {
            name: float(np.mean(prem - losses)) for name, losses in strategy_losses.items()
        }
        ranked = sorted(net_pnl_means.keys(), key=lambda k: net_pnl_means[k], reverse=True)
        rankings.append(ranked)
        table[f"vhsi_plus_vrp_{int(round(vrp * 100))}pct"] = {
            "mean_net_pnl": net_pnl_means,
            "ranking_best_to_worst": ranked,
        }

    table["ranking_flipped_across_vrp"] = len({tuple(r) for r in rankings}) > 1
    return table


def format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# HK Cross-Validation Results Summary ({report['index'].upper()} | {report['split']})",
        "",
        f"- **Index**: `{report['index']}` | **Split**: `{report['split']}` ({report['n_windows']} monthly windows)",
        "",
        "## 1. Main Table: Capital Requirement C* & Trading Metrics",
        "",
        "| Cost Leg | Strategy | C*95 | C*99 | Mean Loss | Turnover | Cost / Notional |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for leg_name, leg_data in report["legs"].items():
        b_sum = leg_data["bench"]["summary"]
        lines.append(
            f"| {leg_name} | Bench (Delta) | {b_sum['c95']:.4f} | {b_sum['c99']:.4f} | "
            f"{b_sum['mean_loss']:.4f} | {b_sum['turnover']:.4f} | {b_sum['cost_notional']:.5f} |"
        )
        if leg_data["m1_zeroshot"] is not None:
            m1_sum = leg_data["m1_zeroshot"]["summary"]
            lines.append(
                f"| {leg_name} | M1 (Zero-shot) | {m1_sum['c95']:.4f} | {m1_sum['c99']:.4f} | "
                f"{m1_sum['mean_loss']:.4f} | {m1_sum['turnover']:.4f} | {m1_sum['cost_notional']:.5f} |"
            )
        if leg_data["m2_refit_a50"] is not None:
            m2_m = leg_data["m2_refit_a50"]["metrics_mean_std"]
            lines.append(
                f"| {leg_name} | M2 (Re-fit, α=0.5) | "
                f"{m2_m['c95']['mean']:.4f} ± {m2_m['c95']['std']:.4f} | "
                f"{m2_m['c99']['mean']:.4f} ± {m2_m['c99']['std']:.4f} | "
                f"{m2_m['mean_loss']['mean']:.4f} ± {m2_m['mean_loss']['std']:.4f} | "
                f"{m2_m['turnover']['mean']:.4f} | {m2_m['cost_notional']['mean']:.5f} |"
            )
        if leg_data["m2_refit_a99"] is not None:
            m99_m = leg_data["m2_refit_a99"]["metrics_mean_std"]
            lines.append(
                f"| {leg_name} | M2 (Re-fit, α=0.99) | "
                f"{m99_m['c95']['mean']:.4f} ± {m99_m['c95']['std']:.4f} | "
                f"{m99_m['c99']['mean']:.4f} ± {m99_m['c99']['std']:.4f} | "
                f"{m99_m['mean_loss']['mean']:.4f} ± {m99_m['mean_loss']['std']:.4f} | "
                f"{m99_m['turnover']['mean']:.4f} | {m99_m['cost_notional']['mean']:.5f} |"
            )

    lines.extend(["", "## 2. Hypothesis Tests (P1–P4)", ""])
    for h_key, h_val in report["hypotheses"].items():
        lines.append(f"- **{h_key}**: `{json.dumps(h_val, ensure_ascii=False)}`")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HK Cross-Validation Master Experiment Runner (B2)")
    parser.add_argument("--index", choices=["hsi", "hsce"], default="hsi", help="標的指數 (hsi 或 hsce)")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all", help="回測樣本區間")
    parser.add_argument("--m2-mode", choices=["garch", "garch_plain", "bootstrap"], default="garch")
    parser.add_argument("--m2-root", default="results", help="M2 訓練權重所在根目錄")
    parser.add_argument("--out-dir", default="results/hk_summary", help="彙總報表輸出目錄")
    parser.add_argument("--tag", default="hsi_main", help="輸出檔名前綴標籤")
    parser.add_argument("--device", default="auto", help="運算裝置")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)

    split_arg = None if args.split == "all" else args.split
    windows = extract_monthly_windows(index_col=args.index, split=split_arg)
    if not windows:
        raise RuntimeError(f"無可用窗口（index={args.index}, split={args.split}）")

    leg_configs = {
        "fut1bp": "configs/hk/fut1bp.yaml",
        "fut10bp": "configs/hk/fut10bp.yaml",
        "etf30bp": "configs/hk/etf30bp.yaml",
    }

    m1_policy = load_d1_policy(D1_WEIGHTS, device=device) if D1_WEIGHTS.exists() else None
    m2_root = Path(args.m2_root)

    legs_report: dict[str, Any] = {}
    for leg_name, cfg_path in leg_configs.items():
        cfg = load_config(cfg_path)
        seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
        hk_cfg = cfg.get("hk", {})
        b_size = int(hk_cfg.get("block_bootstrap_months", 12))
        n_boot = int(hk_cfg.get("block_bootstrap_B", 2000))
        vrp_grid = list(hk_cfg.get("vrp_levels", [0.0, 0.03, 0.05]))

        bench_policy = DeltaHedgePolicy(n_steps=int(cfg.get("n_steps", 30)))
        res_bench = run_hk_backtest(bench_policy, windows, cfg, device=device)
        bench_losses = res_bench["arrays"]["terminal_loss"]

        res_m1 = None
        m1_losses = None
        m1_vs_bench = None
        if m1_policy is not None:
            res_m1 = run_hk_backtest(m1_policy, windows, cfg, device=device)
            m1_losses = res_m1["arrays"]["terminal_loss"]
            m1_vs_bench = paired_block_bootstrap(
                losses_a=m1_losses, losses_b=bench_losses, block_size=b_size, n_boot=n_boot, seed=42
            )

        kind_tag = str(cfg.get("kind", "hk_backtest"))
        leg_m2_root = m2_root / leg_name
        m2_a50_agg = None
        ckpts_a50 = discover_m2_checkpoints(leg_m2_root, f"{kind_tag}_{args.index}_{args.m2_mode}_m2_a50", seeds)
        if ckpts_a50:
            seed_res_50 = [
                (s, run_hk_backtest(load_m2_policy(p, device=device), windows, cfg, device=device))
                for s, p in ckpts_a50
            ]
            m2_a50_agg = aggregate_multi_seed_backtests(
                seed_res_50, bench_losses, m1_losses, block_size=b_size, n_boot=n_boot, boot_seed=42
            )

        m2_a99_agg = None
        ckpts_a99 = discover_m2_checkpoints(leg_m2_root, f"{kind_tag}_{args.index}_{args.m2_mode}_m2_a99", seeds)
        if ckpts_a99:
            seed_res_99 = [
                (s, run_hk_backtest(load_m2_policy(p, device=device), windows, cfg, device=device))
                for s, p in ckpts_a99
            ]
            m2_a99_agg = aggregate_multi_seed_backtests(
                seed_res_99, bench_losses, m1_losses, block_size=b_size, n_boot=n_boot, boot_seed=42
            )

        strat_loss_map = {"bench": bench_losses}
        if m1_losses is not None:
            strat_loss_map["m1"] = m1_losses
        if m2_a50_agg is not None:
            strat_loss_map["m2_a50"] = m2_a50_agg["seed0_losses"]
        prem_table = compute_premium_comparison_table(windows, strat_loss_map, vrp_grid=vrp_grid)

        if m2_a50_agg is not None:
            m2_a50_agg.pop("seed0_losses", None)
        if m2_a99_agg is not None:
            m2_a99_agg.pop("seed0_losses", None)

        legs_report[leg_name] = {
            "bench": {"summary": res_bench["summary"], "bootstrap": res_bench["bootstrap"]},
            "m1_zeroshot": (

                {
                    "summary": res_m1["summary"],
                    "bootstrap": res_m1["bootstrap"],
                    "rounding_error": res_m1["rounding_error"],
                    "paired_vs_bench": m1_vs_bench,
                }
                if res_m1 is not None
                else None
            ),
            "m2_refit_a50": m2_a50_agg,
            "m2_refit_a99": m2_a99_agg,
            "premium_appendix": prem_table,
        }

    fut = legs_report["fut1bp"]
    etf = legs_report["etf30bp"]
    hypotheses: dict[str, Any] = {
        "P1_futures_1bp_parity": (
            fut["m2_refit_a50"]["paired_vs_bench_seed0"]["diff_c95"]
            if fut["m2_refit_a50"] is not None
            else (fut["m1_zeroshot"]["paired_vs_bench"]["diff_c95"] if fut["m1_zeroshot"] else "pending")
        ),
        "P2_etf_30bp_advantage": (
            etf["m2_refit_a50"]["paired_vs_bench_seed0"]["diff_c95"]
            if etf["m2_refit_a50"] is not None
            else (etf["m1_zeroshot"]["paired_vs_bench"]["diff_c95"] if etf["m1_zeroshot"] else "pending")
        ),
        "P3_risk_aversion_crossover_etf": (
            {
                "c95_a99_minus_a50": (
                    etf["m2_refit_a99"]["metrics_mean_std"]["c95"]["mean"]
                    - etf["m2_refit_a50"]["metrics_mean_std"]["c95"]["mean"]
                ),
                "c99_a99_minus_a50": (
                    etf["m2_refit_a99"]["metrics_mean_std"]["c99"]["mean"]
                    - etf["m2_refit_a50"]["metrics_mean_std"]["c99"]["mean"]
                ),
            }
            if (etf["m2_refit_a50"] is not None and etf["m2_refit_a99"] is not None)
            else "pending"
        ),
        "P4_refit_vs_zeroshot_etf": (
            etf["m2_refit_a50"]["paired_vs_m1_seed0"]["diff_c95"]
            if (etf["m2_refit_a50"] is not None and etf["m2_refit_a50"]["paired_vs_m1_seed0"] is not None)
            else "pending"
        ),
    }

    full_report = {
        "tag": args.tag,
        "index": args.index,
        "split": args.split,
        "m2_mode": args.m2_mode,
        "n_windows": len(windows),
        "legs": legs_report,
        "hypotheses": hypotheses,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}_{args.index}_{args.split}.json"
    md_path = out_dir / f"{args.tag}_{args.index}_{args.split}.md"

    json_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_text = format_markdown_report(full_report)
    md_path.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"Saved report -> {json_path} & {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
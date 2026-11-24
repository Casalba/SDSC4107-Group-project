"""
Usage (from repository root):
    python -m scripts.make_hk_b2_figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.common.config import REPO_ROOT, load_config
from src.common.seed import resolve_device
from src.hk.backtest import DeltaHedgePolicy, run_hk_backtest
from src.hk.data import extract_monthly_windows
from src.hk.run_experiments import discover_m2_checkpoints
from src.hk.train_hk import load_m2_policy
from src.policy.heston_d1 import WEIGHTS as D1_WEIGHTS
from src.policy.heston_d1 import load as load_d1_policy

CRISIS_REGIMES: dict[str, tuple[str, str, str]] = {
    "2015H2\n(China A-Share Crash)": ("2015-06", "2015-12", "2015H2 China A-Share & FX Shock"),
    "2020H1\n(COVID-19 Shock)": ("2020-01", "2020-06", "2020H1 COVID-19 Liquidity Shock"),
    "2022\n(HK Tech & Rate Hike)": ("2022-01", "2022-12", "2022 HK Tech Drawdown & Rate Hikes"),
}


def _set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Summary JSON not found: {path}. Run `python -m src.hk.run_experiments` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def plot_capital_comparison(
    all_report: dict[str, Any],
    test_report: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5), sharey=True)

    legs = ["fut1bp", "fut10bp", "etf30bp"]
    leg_labels = ["Futures (1 bp)", "Futures Stress (10 bp)", "2800.HK ETF (30 bp)"]
    strategies = [
        ("bench", "Bench (VHSI Delta)", "#4A6FE3"),
        ("m1_zeroshot", "M1 (Zero-Shot Heston)", "#D95F02"),
        ("m2_refit_a50", "M2 (Re-fit, α=0.50)", "#1B9E77"),
        ("m2_refit_a99", "M2 (Re-fit, α=0.99)", "#7570B3"),
    ]

    datasets = [
        (axes[0], all_report, "Full Sample (2010–2024, N = 180 Months)"),
        (axes[1], test_report, "Out-of-Sample Test (2019–2024, N = 72 Months)"),
    ]

    x = np.arange(len(legs))
    width = 0.19

    for ax, rep, title in datasets:
        for idx, (s_key, s_label, color) in enumerate(strategies):
            vals_95: list[float] = []
            errs_95: list[float] = []
            vals_99: list[float] = []
            for leg in legs:
                block = rep["legs"].get(leg, {}).get(s_key)
                if block is None:
                    vals_95.append(np.nan)
                    errs_95.append(0.0)
                    vals_99.append(np.nan)
                elif "metrics_mean_std" in block:
                    vals_95.append(float(block["metrics_mean_std"]["c95"]["mean"]))
                    errs_95.append(float(block["metrics_mean_std"]["c95"]["std"]))
                    vals_99.append(float(block["metrics_mean_std"]["c99"]["mean"]))
                else:
                    vals_95.append(float(block["summary"]["c95"]))
                    errs_95.append(0.0)
                    vals_99.append(float(block["summary"]["c99"]))

            offset = (idx - 1.5) * width
            bars = ax.bar(
                x + offset,
                vals_95,
                width=width * 0.92,
                color=color,
                alpha=0.88,
                yerr=errs_95,
                capsize=3,
                error_kw={"elinewidth": 1.1, "ecolor": "#222222"},
                label=s_label if ax is axes[0] else None,
            )
            ax.scatter(
                x + offset,
                vals_99,
                color="#111111",
                marker="D",
                s=24,
                zorder=5,
                label="C*99 Tail Risk" if (ax is axes[0] and idx == 0) else None,
            )
            for bar, v95, e95 in zip(bars, vals_95, errs_95, strict=False):
                if np.isfinite(v95):
                    # Place label cleanly above error bar if small, or inside a crisp white badge if tall error bar
                    if e95 > 0.35:
                        y_pos = v95 + 0.14
                        bbox_kw = {
                            "facecolor": "white",
                            "alpha": 0.8,
                            "edgecolor": "none",
                            "pad": 0.6,
                        }
                    else:
                        y_pos = v95 + e95 + 0.12
                        bbox_kw = None
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y_pos,
                        f"{v95:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        bbox=bbox_kw,
                        zorder=6,
                    )

        ax.set_title(title, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(leg_labels)
        ax.set_ylim(0, 10.6)

    axes[0].set_ylabel("Capital Requirement C* (per S₀ = 100 Notional)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved Figure 1 -> {out_path}")


def plot_turnover_frontier(all_report: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))

    leg_configs = [
        (axes[0], "fut1bp", "HSI Synthetic Futures (1 bp Round-Trip)"),
        (axes[1], "etf30bp", "2800.HK ETF + 500-Share Lots (30 bp Round-Trip)"),
    ]
    strategies = [
        ("bench", "Bench (VHSI Delta)", "#4A6FE3", "o"),
        ("m1_zeroshot", "M1 (Zero-Shot)", "#D95F02", "s"),
        ("m2_refit_a50", "M2 (Re-fit, α=0.50)", "#1B9E77", "^"),
        ("m2_refit_a99", "M2 (Re-fit, α=0.99)", "#7570B3", "D"),
    ]

    label_offsets: dict[tuple[str, str], tuple[float, float, str, str]] = {
        ("fut1bp", "bench"): (10, 4, "left", "bottom"),
        ("fut1bp", "m1_zeroshot"): (-10, -2, "right", "center"),
        ("fut1bp", "m2_refit_a50"): (10, -4, "left", "top"),
        ("fut1bp", "m2_refit_a99"): (-10, 12, "right", "bottom"),
        ("etf30bp", "bench"): (10, 4, "left", "bottom"),
        ("etf30bp", "m1_zeroshot"): (-10, -2, "right", "center"),
        ("etf30bp", "m2_refit_a50"): (8, 12, "left", "bottom"),
        ("etf30bp", "m2_refit_a99"): (0, -16, "center", "top"),
    }

    for ax, leg, title in leg_configs:
        pts: dict[str, tuple[float, float, float]] = {}
        for s_key, s_label, color, marker in strategies:
            block = all_report["legs"][leg].get(s_key)
            if block is None:
                continue
            if "metrics_mean_std" in block:
                turn = float(block["metrics_mean_std"]["turnover"]["mean"])
                c95 = float(block["metrics_mean_std"]["c95"]["mean"])
                c99 = float(block["metrics_mean_std"]["c99"]["mean"])
            else:
                turn = float(block["summary"]["turnover"])
                c95 = float(block["summary"]["c95"])
                c99 = float(block["summary"]["c99"])
            pts[s_key] = (turn, c95, c99)

            ax.scatter(turn, c95, color=color, marker=marker, s=115, zorder=5, label=s_label)
            dx, dy, ha, va = label_offsets.get((leg, s_key), (8, 0, "left", "center"))
            ax.annotate(
                f"{s_label}\n(C*95={c95:.2f}, C*99={c99:.2f})",
                (turn, c95),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=8.3,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
            )

        if "m2_refit_a50" in pts and "m2_refit_a99" in pts:
            t50, c50, _ = pts["m2_refit_a50"]
            t99, c99, _ = pts["m2_refit_a99"]
            ax.annotate(
                "",
                xy=(t99, c99),
                xytext=(t50, c50),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#444444",
                    "lw": 1.7,
                    "ls": "--",
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
            )

        ax.set_title(title, pad=10)
        ax.set_xlabel("Average Monthly Turnover (Sum |Δδ_k|)")
        ax.set_xlim(1.10, 3.10)
        ax.set_ylim(4.2, 7.6)

    axes[0].set_ylabel("95% Tail Capital Requirement C*95")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved Figure 2 -> {out_path}")


def evaluate_and_plot_crisis_regimes(
    out_fig_path: Path,
    out_json_path: Path,
    m2_root: Path = Path("results"),
) -> dict[str, Any]:
    """Compute window-level losses on 2800.HK (30bp) and break down 2015H2, 2020H1, and 2022 crises."""
    device = resolve_device("auto")
    windows = extract_monthly_windows(index_col="hsi", split=None)
    months = [w.month for w in windows]
    cfg = load_config("configs/hk/etf30bp.yaml")
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]

    res_bench = run_hk_backtest(DeltaHedgePolicy(n_steps=30), windows, cfg, device=device)
    loss_bench = res_bench["arrays"]["terminal_loss"]
    cost_bench = res_bench["arrays"]["cost"]

    m1_pol = load_d1_policy(D1_WEIGHTS, device=device)
    res_m1 = run_hk_backtest(m1_pol, windows, cfg, device=device)
    loss_m1 = res_m1["arrays"]["terminal_loss"]
    cost_m1 = res_m1["arrays"]["cost"]

    leg_dir = m2_root / "etf30bp"
    ckpts_50 = discover_m2_checkpoints(leg_dir, "hk_backtest_hsi_garch_m2_a50", seeds)
    ckpts_99 = discover_m2_checkpoints(leg_dir, "hk_backtest_hsi_garch_m2_a99", seeds)

    res_50_list = [
        run_hk_backtest(load_m2_policy(p, device=device), windows, cfg, device=device)
        for _, p in ckpts_50
    ]
    res_99_list = [
        run_hk_backtest(load_m2_policy(p, device=device), windows, cfg, device=device)
        for _, p in ckpts_99
    ]

    loss_m2_50 = np.mean([r["arrays"]["terminal_loss"] for r in res_50_list], axis=0)
    cost_m2_50 = np.mean([r["arrays"]["cost"] for r in res_50_list], axis=0)

    loss_m2_99 = np.mean([r["arrays"]["terminal_loss"] for r in res_99_list], axis=0)
    cost_m2_99 = np.mean([r["arrays"]["cost"] for r in res_99_list], axis=0)

    crisis_summary: dict[str, Any] = {}
    regime_labels: list[str] = []
    mean_losses_by_strat: dict[str, list[float]] = {
        "Bench (Delta)": [],
        "M1 (Zero-Shot)": [],
        "M2 (Re-fit, α=0.50)": [],
        "M2 (Re-fit, α=0.99)": [],
    }
    max_losses_by_strat: dict[str, list[float]] = {k: [] for k in mean_losses_by_strat}

    for short_label, (m_start, m_end, full_name) in CRISIS_REGIMES.items():
        idx = [i for i, m in enumerate(months) if m_start <= m <= m_end]
        regime_labels.append(short_label)

        series_map = {
            "Bench (Delta)": (loss_bench[idx], cost_bench[idx]),
            "M1 (Zero-Shot)": (loss_m1[idx], cost_m1[idx]),
            "M2 (Re-fit, α=0.50)": (loss_m2_50[idx], cost_m2_50[idx]),
            "M2 (Re-fit, α=0.99)": (loss_m2_99[idx], cost_m2_99[idx]),
        }

        reg_stats: dict[str, Any] = {
            "n_months": len(idx),
            "range": f"{m_start}..{m_end}",
            "strategies": {},
        }
        for s_name, (l_arr, c_arr) in series_map.items():
            m_val = float(np.mean(l_arr))
            mx_val = float(np.max(l_arr))
            c_val = float(np.mean(c_arr))
            mean_losses_by_strat[s_name].append(m_val)
            max_losses_by_strat[s_name].append(mx_val)
            reg_stats["strategies"][s_name] = {
                "mean_loss": m_val,
                "max_single_month_loss": mx_val,
                "mean_tx_cost": c_val,
            }
        crisis_summary[full_name] = reg_stats

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(crisis_summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Crisis Regime Breakdown -> {out_json_path}")

    print("\n=== Crisis Regime Stress Breakdown (2800.HK ETF, 30bp + 500-Share Lots) ===")
    for r_name, r_info in crisis_summary.items():
        print(f"\n[{r_name}] (N = {r_info['n_months']} months)")
        for s_name, st in r_info["strategies"].items():
            print(
                f"  - {s_name:<22}: Mean Loss = {st['mean_loss']:.4f} | "
                f"Worst Month = {st['max_single_month_loss']:.4f} | Mean Tx Cost = {st['mean_tx_cost']:.4f}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
    x = np.arange(len(regime_labels))
    width = 0.19
    colors = ["#4A6FE3", "#D95F02", "#1B9E77", "#7570B3"]

    for idx, ((s_name, m_vals), color) in enumerate(
        zip(mean_losses_by_strat.items(), colors, strict=False)
    ):
        offset = (idx - 1.5) * width
        bars = axes[0].bar(
            x + offset, m_vals, width=width * 0.92, color=color, alpha=0.9, label=s_name
        )
        for b, v in zip(bars, m_vals, strict=False):
            axes[0].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.08,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    for idx, ((_s_name, mx_vals), color) in enumerate(
        zip(max_losses_by_strat.items(), colors, strict=False)
    ):
        offset = (idx - 1.5) * width
        bars = axes[1].bar(x + offset, mx_vals, width=width * 0.92, color=color, alpha=0.9)
        for b, v in zip(bars, mx_vals, strict=False):
            axes[1].text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.15,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    axes[0].set_title("Mean Monthly Hedging Loss During HK Crises", pad=10)
    axes[0].set_ylabel("Hedging Loss (per S₀ = 100 Notional)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regime_labels)
    axes[0].set_ylim(0, max(max(v) for v in mean_losses_by_strat.values()) * 1.22)

    axes[1].set_title("Worst Single-Month Tail Loss During HK Crises", pad=10)
    axes[1].set_ylabel("Max Monthly Loss (per S₀ = 100 Notional)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regime_labels)
    axes[1].set_ylim(0, max(max(v) for v in max_losses_by_strat.values()) * 1.18)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved Figure 3 -> {out_fig_path}")
    return crisis_summary


def main() -> None:
    _set_pub_style()
    summary_dir = REPO_ROOT / "results" / "hk_summary"
    fig_dir = REPO_ROOT / "figures"

    all_rep = load_summary_json(summary_dir / "hsi_main_hsi_all.json")
    test_rep = load_summary_json(summary_dir / "hsi_main_hsi_test.json")

    plot_capital_comparison(
        all_report=all_rep,
        test_report=test_rep,
        out_path=fig_dir / "fig_hk_b2_capital_comparison.png",
    )
    plot_turnover_frontier(
        all_report=all_rep,
        out_path=fig_dir / "fig_hk_b2_turnover_frontier.png",
    )
    evaluate_and_plot_crisis_regimes(
        out_fig_path=fig_dir / "fig_hk_b2_crisis_regimes.png",
        out_json_path=summary_dir / "hsi_crisis_regimes.json",
    )


if __name__ == "__main__":
    main()

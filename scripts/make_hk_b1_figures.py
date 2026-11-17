"""Publication figures for B1 HK data health check and GARCH+Jump calibration (300 DPI).

Strictly follows docs/spec.md §10:
- 300 DPI PNG, English axis labels & legends, NO set_title / suptitle (title goes in caption)
- Palette: model/GARCH = blue (#1f77b4), empirical/HSCE/benchmark = grey (#7f7f7f),
  reference/95% CI/split = red dashed (#d62728).

Usage:
    python -m scripts.make_hk_b1_figures
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hk.data import HKTable, load_hk_daily, read_csv_table  # noqa: E402

FIG_DIR = REPO_ROOT / "figures"
CALIB_DIR = REPO_ROOT / "results" / "calib"
VOL_CSV = REPO_ROOT / "data" / "interim" / "hk_vol_forecasts.csv"

BLUE = "#1f77b4"
GREY = "#7f7f7f"
RED = "#d62728"


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    c = x - np.mean(x)
    denom = float(np.dot(c, c))
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=np.float64)
    return np.array([float(np.dot(c[k:], c[:-k]) / denom) for k in range(1, max_lag + 1)])


def make_stylized_facts_fig(tbl: HKTable, hsi_cal: dict, hsce_cal: dict) -> Path:
    """Plot standardized tail density (a) and absolute/raw return ACF (b)."""
    r_hsi = np.diff(np.log(np.asarray(tbl["hsi"], dtype=np.float64)))
    r_hsce = np.diff(np.log(np.asarray(tbl["hsce"], dtype=np.float64)))
    z_hsi = (r_hsi - np.mean(r_hsi)) / np.std(r_hsi, ddof=1)
    z_hsce = (r_hsce - np.mean(r_hsce)) / np.std(r_hsce, ddof=1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    bins = np.linspace(-6.5, 6.5, 66)
    ax0.hist(
        z_hsi,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.38,
        edgecolor=BLUE,
        color=BLUE,
        lw=1.2,
        label="^HSI daily log-returns",
    )
    ax0.hist(
        z_hsce,
        bins=bins,
        density=True,
        histtype="step",
        color=GREY,
        lw=1.4,
        label="^HSCE daily log-returns",
    )
    x_grid = np.linspace(-6.5, 6.5, 400)
    phi = np.exp(-0.5 * x_grid**2) / np.sqrt(2.0 * np.pi)
    ax0.plot(x_grid, phi, color=RED, ls="--", lw=1.6, label="Standard Normal N(0, 1)")
    ax0.set_yscale("log")
    ax0.set_ylim(2e-4, 1.65)
    ax0.set_xlim(-6.5, 6.5)
    ax0.set_xlabel("(a) Standardized Daily Log-Return (z-score)")
    ax0.set_ylabel("Empirical Density (log scale)")
    ax0.legend(frameon=False, fontsize=8.3, loc="upper left")
    ax0.grid(True, which="major", ls=":", alpha=0.45)

    sf_hsi, jt_hsi = hsi_cal["stylized_facts_full"], hsi_cal["jump_test_full"]
    sf_hsce, jt_hsce = hsce_cal["stylized_facts_full"], hsce_cal["jump_test_full"]
    ax0.text(
        0.97,
        0.96,
        f"Excess Kurt: HSI {sf_hsi['excess_kurtosis']:.2f} | HSCE {sf_hsce['excess_kurtosis']:.2f}\n"
        f"BNS Jump Share: HSI {jt_hsi['bns_jump_var_share_pct']:.1f}% | HSCE {jt_hsce['bns_jump_var_share_pct']:.1f}%\n"
        f"LM (>3.5σ): HSI {jt_hsi['n_jumps_pt35']} | HSCE {jt_hsce['n_jumps_pt35']}",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.92,
        },
    )

    lags = np.arange(1, 31)
    acf_abs = _acf(np.abs(r_hsi), 30)
    acf_raw = _acf(r_hsi, 30)
    ci_95 = 1.96 / np.sqrt(len(r_hsi))

    ax1.bar(lags - 0.16, acf_abs, width=0.32, color=BLUE, alpha=0.85, label="^HSI ACF(|r_t|)")
    ax1.bar(lags + 0.16, acf_raw, width=0.32, color=GREY, alpha=0.80, label="^HSI ACF(r_t)")
    ax1.axhline(ci_95, color=RED, ls="--", lw=1.3, label=f"95% White-Noise Band (±{ci_95:.3f})")
    ax1.axhline(-ci_95, color=RED, ls="--", lw=1.3)
    ax1.axhline(0.0, color="#334155", lw=0.8)
    ax1.set_xlim(0.2, 30.8)
    ax1.set_ylim(-0.10, max(0.32, float(np.max(acf_abs)) + 0.06))
    ax1.set_xlabel("(b) Lag k (HKEX Trading Days)")
    ax1.set_ylabel("Sample Autocorrelation")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax1.grid(True, axis="y", ls=":", alpha=0.45)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_hk_stylized_facts.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _rolling_21d_vol_pct(prices: np.ndarray) -> np.ndarray:
    lr = np.empty(len(prices), dtype=np.float64)
    lr[0] = np.nan
    lr[1:] = np.diff(np.log(prices))
    out = np.full(len(prices), np.nan, dtype=np.float64)
    for t in range(21, len(prices)):
        out[t] = float(np.std(lr[t - 20 : t + 1], ddof=1) * np.sqrt(252.0) * 100.0)
    return out


def make_garch_vol_fig(tbl: HKTable, vol_tbl: HKTable) -> Path:
    """Plot walk-forward GARCH(1,1)+Jump volatility vs 21d realized volatility & ^VHSI (2010-2024)."""
    sub_vol = vol_tbl.filter_date_range("2010-01-01", "2024-12-31")
    dates_dt = np.array([dt.datetime.fromisoformat(str(d)) for d in tbl["date"]])
    rv_21 = _rolling_21d_vol_pct(np.asarray(tbl["hsi"], dtype=np.float64))
    garch_j_pct = np.asarray(sub_vol["hsi_jump_vol"], dtype=np.float64) * 100.0
    vhsi_pct = np.asarray(sub_vol["vhsi"], dtype=np.float64) * 100.0

    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    regimes = [
        ("2015-07-01", "2015-12-31", "2015H2"),
        ("2020-01-01", "2020-06-30", "2020H1"),
        ("2022-01-01", "2022-12-31", "2022"),
    ]
    for idx, (d0, d1, lbl) in enumerate(regimes):
        dt0, dt1 = dt.datetime.fromisoformat(d0), dt.datetime.fromisoformat(d1)
        ax.axvspan(
            dt0,
            dt1,
            color="#e2e8f0",
            alpha=0.75,
            label="Stress Regimes (2015H2, 2020H1, 2022)" if idx == 0 else None,
        )
        mid = dt0 + (dt1 - dt0) / 2
        ax.text(
            mid,
            10.6,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#475569",
        )

    ax.plot(dates_dt, rv_21, color=GREY, lw=1.0, alpha=0.80, label="21-Day Realized Vol (HSI)")
    ax.plot(
        dates_dt,
        garch_j_pct,
        color=BLUE,
        lw=1.35,
        label="GARCH(1,1)+Jump Filter (Train 2010–2018)",
    )
    ax.plot(
        dates_dt,
        vhsi_pct,
        color="#0f766e",
        lw=0.95,
        alpha=0.75,
        label="^VHSI (^HSIL, 30-Day ATM IV)",
    )
    ax.axvline(
        dt.datetime(2019, 1, 2),
        color=RED,
        ls="--",
        lw=1.6,
        label="Walk-Forward Split (2019-01-02 OOS Start)",
    )

    ax.set_xlim(dt.datetime(2010, 1, 1), dt.datetime(2025, 1, 5))
    ax.set_ylim(8.0, 76.0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date (HKEX Trading Calendar, Asia/Hong_Kong)")
    ax.set_ylabel("Annualized Volatility (% p.a.)")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.92,
        fontsize=8.1,
        ncol=2,
        loc="upper left",
    )
    ax.grid(True, ls=":", alpha=0.45)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_hk_garch_vol.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    tbl = load_hk_daily()
    vol_tbl = read_csv_table(VOL_CSV)
    hsi_cal = json.loads((CALIB_DIR / "hsi_garch.json").read_text(encoding="utf-8"))
    hsce_cal = json.loads((CALIB_DIR / "hsce_garch.json").read_text(encoding="utf-8"))
    p1 = make_stylized_facts_fig(tbl, hsi_cal, hsce_cal)
    p2 = make_garch_vol_fig(tbl, vol_tbl)
    print(f"Wrote {p1} and {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

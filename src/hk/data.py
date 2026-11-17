"""Hong Kong market data helpers, cleaning, alignment, and monthly window extraction (B1)."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch

from src.common.config import REPO_ROOT

PROCESSED_COLUMNS: tuple[str, ...] = (
    "date",
    "hsi",
    "hsce",
    "etf2800",
    "div_yield",
    "hibor1m",
    "vhsi",
)

AUDIT_COLUMNS: tuple[str, ...] = (
    "date",
    "hsi",
    "hsce",
    "etf2800",
    "etf2800_raw",
    "div_cash_hkd",
    "div_yield",
    "hibor1m",
    "vhsi",
    "is_halted_2800",
    "is_filled_hsi",
    "is_filled_hsce",
    "is_filled_etf2800",
    "is_filled_hibor1m",
    "is_filled_vhsi",
)

STRING_COLUMNS: frozenset[str] = frozenset({"date", "source"})


@dataclass
class HKTable:
    """Lightweight stdlib/NumPy tabular container avoiding runtime pandas dependency in CI."""

    columns: tuple[str, ...]
    data: dict[str, np.ndarray | list[str]]

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(self.data[self.columns[0]])

    def __getitem__(self, col: str) -> np.ndarray | list[str]:
        return self.data[col]

    def has_nans(self) -> bool:
        """Check whether any numeric column contains NaN or Inf."""
        for col in self.columns:
            vals = self.data[col]
            if isinstance(vals, np.ndarray) and not np.all(np.isfinite(vals)):
                return True
        return False

    def slice_rows(self, start: int, stop: int, col: str) -> np.ndarray | list[str]:
        """Slice rows `[start:stop]` from column `col`."""
        return self.data[col][start:stop]

    def filter_date_range(self, start_date: str, end_date: str) -> HKTable:
        """Return a new HKTable filtered inclusively by ISO date range."""
        dates = self.data["date"]
        idx = [i for i, d in enumerate(dates) if start_date <= str(d) <= end_date]
        new_data: dict[str, np.ndarray | list[str]] = {}
        for col in self.columns:
            arr = self.data[col]
            if isinstance(arr, list):
                new_data[col] = [arr[i] for i in idx]
            else:
                new_data[col] = arr[idx]
        return HKTable(columns=self.columns, data=new_data)

    def to_dataframe(self) -> Any:
        """Convert to pandas.DataFrame lazily (only when pandas is installed)."""
        import pandas as pd

        return pd.DataFrame({col: self.data[col] for col in self.columns})


def _parse_float(raw: str | None) -> float:
    if raw is None or raw.strip() in {"", "nan", "NaN", "None", "null", "NA"}:
        return float("nan")
    return float(raw)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows using stdlib csv module with UTF-8 encoding."""
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_table(
    path: str | Path,
    columns: tuple[str, ...] | list[str],
    data: dict[str, np.ndarray | list[Any]],
) -> Path:
    """Write a columnar dict deterministically to CSV with LF line endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(data[columns[0]]) if columns else 0
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for i in range(n_rows):
            row_vals: list[Any] = []
            for col in columns:
                val = data[col][i]
                if isinstance(val, (np.floating, float)):
                    row_vals.append("" if np.isnan(val) else repr(float(val)))
                elif isinstance(val, (np.integer, int)):
                    row_vals.append(int(val))
                else:
                    row_vals.append(val)
            writer.writerow(row_vals)
    return target


def read_csv_table(path: str | Path) -> HKTable:
    """Load a CSV into HKTable (date/string columns as list[str], numeric columns as float64 ndarray)."""
    target = Path(path)
    with target.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = tuple(reader.fieldnames or ())
        bufs: dict[str, list[Any]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                if c in STRING_COLUMNS:
                    bufs[c].append(str(row[c]))
                else:
                    bufs[c].append(_parse_float(row[c]))
    data: dict[str, np.ndarray | list[str]] = {}
    for c in cols:
        data[c] = bufs[c] if c in STRING_COLUMNS else np.asarray(bufs[c], dtype=np.float64)
    return HKTable(columns=cols, data=data)


@dataclass(frozen=True)
class HKWindow:
    """One monthly 30-trading-day HK derivative hedging window.

    Units: `spot` and `futures` in index points (or HKD if ETF); `etf_adj` and `etf_raw` in HKD;
    `r`, `q`, and `vhsi` in annualized decimals; `strike` in index points rounded to nearest 100.
    """

    month: str
    split: str
    dates: tuple[str, ...]
    spot: np.ndarray
    futures: np.ndarray
    etf_adj: np.ndarray
    etf_raw: np.ndarray
    r: np.ndarray
    q: np.ndarray
    vhsi: np.ndarray
    strike: float


def compute_backward_adjusted_prices(
    close: np.ndarray,
    ex_div_cash: np.ndarray,
) -> np.ndarray:
    """Compute backward total-return adjusted prices (HKD) from unadjusted closes and ex-dividends.

    Uses factor A_t = A_{t+1} * (1 - D_{t+1} / C_t) when cash dividend D_{t+1} goes ex on t+1.
    """
    c = np.asarray(close, dtype=np.float64)
    d = np.asarray(ex_div_cash, dtype=np.float64)
    if c.ndim != 1 or d.shape != c.shape:
        raise ValueError(
            f"close and ex_div_cash must have equal 1D shape, got {c.shape} vs {d.shape}"
        )
    if len(c) == 0:
        return c.copy()
    adj = c.copy()
    factor = 1.0
    for idx in range(len(c) - 2, -1, -1):
        div_next = d[idx + 1]
        if div_next > 0.0 and c[idx] > 0.0:
            factor *= max(1e-8, 1.0 - div_next / c[idx])
        adj[idx] = c[idx] * factor
    return adj


def compute_trailing_dividend_yield(
    dates: list[str] | tuple[str, ...] | np.ndarray,
    raw_close: np.ndarray,
    ex_div_cash: np.ndarray,
    lookback_days: int = 365,
) -> np.ndarray:
    """Compute trailing 365-calendar-day cash dividend yield q_t = sum_{365d} D / raw_close_t."""
    ord_days = np.array(
        [dt.date.fromisoformat(str(s)[:10]).toordinal() for s in dates], dtype=np.int64
    )
    c = np.asarray(raw_close, dtype=np.float64)
    d = np.asarray(ex_div_cash, dtype=np.float64)
    if len(ord_days) != len(c) or c.shape != d.shape:
        raise ValueError("dates, raw_close, and ex_div_cash must have identical length")
    div_idx = np.flatnonzero(d > 0.0)
    div_days = ord_days[div_idx]
    div_vals = d[div_idx]
    out = np.zeros(len(c), dtype=np.float64)
    for i, cur_day in enumerate(ord_days):
        if c[i] <= 0.0:
            continue
        mask = (div_days > cur_day - lookback_days) & (div_days <= cur_day)
        out[i] = float(div_vals[mask].sum() / c[i])
    return out


def build_processed_datasets(
    raw_dir: str | Path,
    asof: str,
    interim_dir: str | Path,
    processed_dir: str | Path,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> tuple[HKTable, HKTable]:
    """Align raw Yahoo + HKMA CSVs to official HKEX trading days and write interim/processed CSVs."""
    raw_root = Path(raw_dir)
    hsi_rows = read_csv_rows(raw_root / "yahoo" / f"hsi_{asof}.csv")
    hsce_rows = read_csv_rows(raw_root / "yahoo" / f"hsce_{asof}.csv")
    etf_rows = read_csv_rows(raw_root / "yahoo" / f"etf2800_{asof}.csv")
    div_rows = read_csv_rows(raw_root / "yahoo" / f"etf2800_div_{asof}.csv")
    vhsi_rows = read_csv_rows(raw_root / "yahoo" / f"vhsi_{asof}.csv")
    hibor_rows = read_csv_rows(raw_root / "hkma" / f"hibor1m_{asof}.csv")

    # Official HKEX cash-index trading calendar defined by non-null HSI closes.
    hsi_map = {
        r["date"]: _parse_float(r["close"])
        for r in hsi_rows
        if np.isfinite(_parse_float(r["close"]))
    }
    hsce_map = {
        r["date"]: _parse_float(r["close"])
        for r in hsce_rows
        if np.isfinite(_parse_float(r["close"]))
    }
    etf_map = {
        r["date"]: (_parse_float(r["close"]), _parse_float(r.get("volume")))
        for r in etf_rows
        if np.isfinite(_parse_float(r["close"]))
    }
    vhsi_map = {
        r["date"]: _parse_float(r["close"])
        for r in vhsi_rows
        if np.isfinite(_parse_float(r["close"]))
    }
    hibor_map = {
        r["date"]: _parse_float(r["ir_1m"])
        for r in hibor_rows
        if np.isfinite(_parse_float(r.get("ir_1m")))
    }
    div_map: dict[str, float] = {}
    for r in div_rows:
        amt = _parse_float(r["amount"])
        if np.isfinite(amt):
            div_map[r["date"]] = div_map.get(r["date"], 0.0) + amt

    dates_all = sorted(hsi_map.keys())
    n_raw = len(dates_all)
    hsce_arr = np.full(n_raw, np.nan, dtype=np.float64)
    etf_raw_arr = np.full(n_raw, np.nan, dtype=np.float64)
    hibor_arr = np.full(n_raw, np.nan, dtype=np.float64)
    vhsi_arr = np.full(n_raw, np.nan, dtype=np.float64)
    div_arr = np.zeros(n_raw, dtype=np.float64)
    halted_arr = np.zeros(n_raw, dtype=np.int64)
    fill_hsce = np.zeros(n_raw, dtype=np.int64)
    fill_etf = np.zeros(n_raw, dtype=np.int64)
    fill_hibor = np.zeros(n_raw, dtype=np.int64)
    fill_vhsi = np.zeros(n_raw, dtype=np.int64)

    for i, d_str in enumerate(dates_all):
        div_arr[i] = div_map.get(d_str, 0.0)
        if d_str in hsce_map:
            hsce_arr[i] = hsce_map[d_str]
        elif i > 0:
            hsce_arr[i] = hsce_arr[i - 1]
            fill_hsce[i] = 1
        if d_str in etf_map:
            c_val, v_val = etf_map[d_str]
            etf_raw_arr[i] = c_val
            if not np.isfinite(v_val) or v_val <= 0.0:
                halted_arr[i] = 1
        elif i > 0:
            etf_raw_arr[i] = etf_raw_arr[i - 1]
            fill_etf[i] = 1
            halted_arr[i] = 1
        if d_str in hibor_map:
            hibor_arr[i] = hibor_map[d_str]
        elif i > 0:
            hibor_arr[i] = hibor_arr[i - 1]
            fill_hibor[i] = 1
        if d_str in vhsi_map:
            vhsi_arr[i] = vhsi_map[d_str]
        elif i > 0:
            vhsi_arr[i] = vhsi_arr[i - 1]
            fill_vhsi[i] = 1

    hsi_arr = np.array([hsi_map[d] for d in dates_all], dtype=np.float64)
    valid = (
        np.isfinite(hsi_arr)
        & np.isfinite(hsce_arr)
        & np.isfinite(etf_raw_arr)
        & np.isfinite(hibor_arr)
        & np.isfinite(vhsi_arr)
    )
    keep = np.flatnonzero(valid)
    dates_clean = [dates_all[i] for i in keep]
    hsi_c = hsi_arr[keep]
    hsce_c = hsce_arr[keep]
    etf_raw_c = etf_raw_arr[keep]
    div_c = div_arr[keep]
    hibor_dec = hibor_arr[keep] / 100.0
    vhsi_dec = vhsi_arr[keep] / 100.0
    etf_adj_c = compute_backward_adjusted_prices(etf_raw_c, div_c)
    div_yield_c = compute_trailing_dividend_yield(dates_clean, etf_raw_c, div_c, lookback_days=365)

    audit_data: dict[str, np.ndarray | list[Any]] = {
        "date": dates_clean,
        "hsi": hsi_c,
        "hsce": hsce_c,
        "etf2800": etf_adj_c,
        "etf2800_raw": etf_raw_c,
        "div_cash_hkd": div_c,
        "div_yield": div_yield_c,
        "hibor1m": hibor_dec,
        "vhsi": vhsi_dec,
        "is_halted_2800": halted_arr[keep],
        "is_filled_hsi": np.zeros(len(dates_clean), dtype=np.int64),
        "is_filled_hsce": fill_hsce[keep],
        "is_filled_etf2800": fill_etf[keep],
        "is_filled_hibor1m": fill_hibor[keep],
        "is_filled_vhsi": fill_vhsi[keep],
    }
    audit_tbl = HKTable(columns=AUDIT_COLUMNS, data=audit_data)

    main_idx = [i for i, d_str in enumerate(dates_clean) if start_date <= d_str <= end_date]
    proc_data: dict[str, np.ndarray | list[Any]] = {
        "date": [dates_clean[i] for i in main_idx],
        "hsi": hsi_c[main_idx],
        "hsce": hsce_c[main_idx],
        "etf2800": etf_adj_c[main_idx],
        "div_yield": div_yield_c[main_idx],
        "hibor1m": hibor_dec[main_idx],
        "vhsi": vhsi_dec[main_idx],
    }
    proc_tbl = HKTable(columns=PROCESSED_COLUMNS, data=proc_data)

    write_csv_table(Path(interim_dir) / "hk_daily_full_with_audit.csv", AUDIT_COLUMNS, audit_data)
    write_csv_table(Path(processed_dir) / "hk_daily.csv", PROCESSED_COLUMNS, proc_data)
    return proc_tbl, audit_tbl


def load_hk_daily(path: str | Path | None = None) -> HKTable:
    """Load the clean aligned HK daily dataset (7 columns; date sorted ascending)."""
    target = Path(path) if path is not None else REPO_ROOT / "data" / "processed" / "hk_daily.csv"
    if not target.exists():
        raise FileNotFoundError(
            f"HK daily CSV not found: {target}; run `python scripts/fetch_hk.py`"
        )
    tbl = read_csv_table(target)
    missing = [col for col in PROCESSED_COLUMNS if col not in tbl.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {target}")
    return tbl


def load_hk_audit(path: str | Path | None = None) -> HKTable:
    """Load the full extended HK audit dataset (2009-2025 settlement runway + quality flags)."""
    target = (
        Path(path)
        if path is not None
        else REPO_ROOT / "data" / "interim" / "hk_daily_full_with_audit.csv"
    )
    if not target.exists():
        raise FileNotFoundError(
            f"HK audit CSV not found: {target}; run `python scripts/fetch_hk.py`"
        )
    return read_csv_table(target)


@overload
def round_hsi_strike(spot: float, step: float = 100.0) -> float: ...


@overload
def round_hsi_strike(spot: np.ndarray, step: float = 100.0) -> np.ndarray: ...


def round_hsi_strike(spot: float | np.ndarray, step: float = 100.0) -> float | np.ndarray:
    """Round spot index level to nearest listed ATM strike increment (default 100 index pts)."""
    if np.ndim(spot) == 0:
        return float(round(float(spot) / step) * step)
    arr = np.asarray(spot, dtype=np.float64)
    return np.round(arr / step) * step


def synthetic_futures_path(
    spot: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    maturity: float = 30.0 / 365.0,
) -> np.ndarray:
    """Compute synthetic front-month index futures prices F_k = S_k * exp((r_k - q_k) * tau_k)."""
    s = np.asarray(spot, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if s.ndim != 1 or r_arr.shape != s.shape or q_arr.shape != s.shape:
        raise ValueError("spot, r, and q must be 1D arrays of equal length n_steps + 1")
    n_steps = len(s) - 1
    tau = maturity * (1.0 - np.arange(n_steps + 1, dtype=np.float64) / max(1, n_steps))
    return s * np.exp((r_arr - q_arr) * tau)


def extract_monthly_windows(
    table: HKTable | None = None,
    *,
    index_col: str = "hsi",
    n_steps: int = 30,
    maturity: float = 30.0 / 365.0,
    split: str | None = None,
    train_end: str = "2018-12-31",
    start_month: str = "2010-01",
    end_month: str = "2024-12",
) -> list[HKWindow]:
    """Extract non-overlapping monthly 30-day option windows anchored on the 1st trading day."""
    if index_col not in {"hsi", "hsce"}:
        raise ValueError(f"index_col must be 'hsi' or 'hsce', got {index_col!r}")

    if table is None:
        audit_file = REPO_ROOT / "data" / "interim" / "hk_daily_full_with_audit.csv"
        tbl = load_hk_audit(audit_file) if audit_file.exists() else load_hk_daily()
    else:
        tbl = table

    has_raw = "etf2800_raw" in tbl.columns
    dates = [str(d) for d in tbl["date"]]
    month_first_idx: dict[str, int] = {}
    for i, d_str in enumerate(dates):
        ym = d_str[:7]
        if start_month <= ym <= end_month and ym not in month_first_idx:
            month_first_idx[ym] = i

    windows: list[HKWindow] = []
    for ym in sorted(month_first_idx):
        idx0 = month_first_idx[ym]
        if idx0 + n_steps >= len(dates):
            raise ValueError(
                f"Insufficient settlement runway after {dates[idx0]} for month {ym} (need {n_steps + 1} days)"
            )
        w_split = "train" if dates[idx0] <= train_end else "test"
        if split is not None and w_split != split:
            continue
        end = idx0 + n_steps + 1
        spot = np.asarray(tbl.slice_rows(idx0, end, index_col), dtype=np.float64)
        etf_adj = np.asarray(tbl.slice_rows(idx0, end, "etf2800"), dtype=np.float64)
        etf_raw = (
            np.asarray(tbl.slice_rows(idx0, end, "etf2800_raw"), dtype=np.float64)
            if has_raw
            else etf_adj.copy()
        )
        r = np.asarray(tbl.slice_rows(idx0, end, "hibor1m"), dtype=np.float64)
        q = np.asarray(tbl.slice_rows(idx0, end, "div_yield"), dtype=np.float64)
        vhsi = np.asarray(tbl.slice_rows(idx0, end, "vhsi"), dtype=np.float64)
        fut = synthetic_futures_path(spot, r, q, maturity=maturity)
        windows.append(
            HKWindow(
                month=ym,
                split=w_split,
                dates=tuple(dates[idx0:end]),
                spot=spot,
                futures=fut,
                etf_adj=etf_adj,
                etf_raw=etf_raw,
                r=r,
                q=q,
                vhsi=vhsi,
                strike=round_hsi_strike(float(spot[0]), step=100.0),
            )
        )
    return windows


def window_features(
    spot_paths: np.ndarray | torch.Tensor,
    maturity: float = 30.0 / 365.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert spot paths (n+1,) or (B, n+1) into d=1 policy features (B, n, 2) = [log(S_k/S_0), t_k/T]."""
    spot = torch.as_tensor(spot_paths, dtype=dtype, device=device)
    if spot.ndim == 1:
        spot = spot.unsqueeze(0)
    if spot.ndim != 2 or spot.shape[1] < 2:
        raise ValueError(
            f"Expected spot_paths of shape (n+1,) or (B, n+1), got {tuple(spot.shape)}"
        )
    batch, n_pts = spot.shape
    n_steps = n_pts - 1
    log_ret = torch.log(spot[:, :n_steps] / spot[:, :1])
    dt_step = maturity / n_steps
    time_frac = (torch.arange(n_steps, dtype=dtype, device=device) * dt_step) / maturity
    t_feat = time_frac.unsqueeze(0).expand(batch, n_steps)
    return torch.stack([log_ret, t_feat], dim=-1)

"""Unit tests for HK market data alignment, dividend adjustment, and windowing (B1)."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import torch
from src.common.config import REPO_ROOT
from src.hk.data import (
    PROCESSED_COLUMNS,
    build_processed_datasets,
    compute_backward_adjusted_prices,
    compute_trailing_dividend_yield,
    extract_monthly_windows,
    load_hk_daily,
    round_hsi_strike,
    synthetic_futures_path,
    window_features,
)

PROCESSED_CSV = REPO_ROOT / "data" / "processed" / "hk_daily.csv"


def _write_test_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_backward_dividend_adjustment_preserves_total_return() -> None:
    close = np.array([10.0, 10.0, 9.5, 10.0], dtype=np.float64)
    div = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float64)
    adj = compute_backward_adjusted_prices(close, div)
    assert adj[-1] == pytest.approx(close[-1])
    assert adj[1] == pytest.approx(adj[2], rel=1e-10)
    assert adj[0] == pytest.approx(adj[1], rel=1e-10)


def test_trailing_dividend_yield_365d_window() -> None:
    dates = ["2010-01-04", "2010-06-01", "2011-01-03", "2011-06-02"]
    close = np.array([20.0, 20.0, 25.0, 20.0])
    div = np.array([0.0, 0.50, 0.0, 0.50])
    dy = compute_trailing_dividend_yield(dates, close, div, lookback_days=365)
    assert dy[0] == pytest.approx(0.0)
    assert dy[1] == pytest.approx(0.50 / 20.0)
    assert dy[2] == pytest.approx(0.50 / 25.0)
    assert dy[3] == pytest.approx(0.50 / 20.0)


def test_round_strike_and_synthetic_futures() -> None:
    assert round_hsi_strike(21449.9) == 21400.0
    assert round_hsi_strike(21451.0) == 21500.0
    arr = round_hsi_strike(np.array([19940.0, 19960.0]))
    assert np.allclose(arr, np.array([19900.0, 20000.0]))

    spot = np.full(31, 20000.0)
    r = np.full(31, 0.04)
    q = np.full(31, 0.03)
    fut = synthetic_futures_path(spot, r, q, maturity=30.0 / 365.0)
    assert fut.shape == (31,)
    assert fut[0] == pytest.approx(20000.0 * np.exp(0.01 * 30.0 / 365.0))
    assert fut[-1] == pytest.approx(20000.0)


def test_build_processed_and_extract_windows_in_memory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    bdates: list[str] = []
    cur = dt.date(2018, 11, 1)
    while cur <= dt.date(2019, 3, 31):
        if cur.weekday() < 5:
            bdates.append(cur.isoformat())
        cur += dt.timedelta(days=1)

    n = len(bdates)
    hsi = 26000.0 + np.linspace(0.0, 500.0, n)
    _write_test_csv(
        raw / "yahoo" / "hsi_20260101.csv",
        ["date", "close"],
        [{"date": d, "close": float(v)} for d, v in zip(bdates, hsi, strict=True)],
    )
    _write_test_csv(
        raw / "yahoo" / "hsce_20260101.csv",
        ["date", "close"],
        [{"date": d, "close": float(v * 0.4)} for d, v in zip(bdates, hsi, strict=True)],
    )
    vhsi_rows = [{"date": d, "close": ("" if i == 10 else 22.0)} for i, d in enumerate(bdates)]
    _write_test_csv(raw / "yahoo" / "vhsi_20260101.csv", ["date", "close"], vhsi_rows)
    _write_test_csv(
        raw / "yahoo" / "etf2800_20260101.csv",
        ["date", "close", "volume"],
        [{"date": d, "close": 26.0, "volume": 1000000.0} for d in bdates],
    )
    _write_test_csv(
        raw / "yahoo" / "etf2800_div_20260101.csv",
        ["date", "amount", "source"],
        [{"date": bdates[5], "amount": 0.52, "source": "test"}],
    )
    _write_test_csv(
        raw / "hkma" / "hibor1m_20260101.csv",
        ["date", "ir_1m"],
        [{"date": d, "ir_1m": 2.5} for d in bdates],
    )

    proc, audit = build_processed_datasets(
        raw,
        asof="20260101",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        start_date="2019-01-01",
        end_date="2019-02-28",
    )
    assert proc.columns == PROCESSED_COLUMNS
    assert not proc.has_nans()
    assert int(np.sum(audit["is_filled_vhsi"])) == 1
    assert float(proc["hibor1m"][0]) == pytest.approx(0.025)
    assert float(proc["vhsi"][0]) == pytest.approx(0.22)

    wins = extract_monthly_windows(
        audit, index_col="hsi", n_steps=30, start_month="2018-11", end_month="2019-01"
    )
    assert [w.month for w in wins] == ["2018-11", "2018-12", "2019-01"]
    assert [w.split for w in wins] == ["train", "train", "test"]
    feats = window_features(np.stack([w.futures for w in wins], axis=0))
    assert feats.shape == (3, 30, 2)
    assert torch.allclose(feats[:, 0, :], torch.zeros((3, 2), dtype=torch.float64))


@pytest.mark.skipif(not PROCESSED_CSV.exists(), reason="run scripts/fetch_hk.py first")
def test_on_disk_hk_daily_and_180_windows() -> None:
    tbl = load_hk_daily(PROCESSED_CSV)
    assert tbl.columns == PROCESSED_COLUMNS
    assert not tbl.has_nans()
    assert str(tbl["date"][0]).startswith("2010-01")
    assert str(tbl["date"][-1]).startswith("2024-12")
    wins = extract_monthly_windows(index_col="hsi")
    assert len(wins) == 180
    assert sum(1 for w in wins if w.split == "train") == 108
    assert sum(1 for w in wins if w.split == "test") == 72
    assert all(len(w.spot) == 31 for w in wins)

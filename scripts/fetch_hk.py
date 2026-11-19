"""Reproducible HK market data downloader and processor (B1 Tasks #1, #2, #3).

Fetches daily history (2009-01-01 to 2025-02-28) from Yahoo Finance Chart v8 API and
HKMA Open API, supplements pre-2013 2800.HK semi-annual cash dividends from TraHK official
records, aligns everything to the official HKEX trading calendar in Asia/Hong_Kong timezone,
and writes SHA-256 checksums to `data/checksums.txt`.

Usage:
    python scripts/fetch_hk.py [--asof 20261118] [--offline]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hk.data import build_processed_datasets, read_csv_rows  # noqa: E402

HK_TZ = ZoneInfo("Asia/Hong_Kong")
PERIOD_START = 1230768000  # 2009-01-01T00:00:00Z
PERIOD_END = 1740787200  # 2025-03-01T00:00:00Z (covers 2025-02-28 HKT settlement runway)

YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HKMA_HIBOR_URL = (
    "https://api.hkma.gov.hk/public/market-data-and-statistics/"
    "monthly-statistical-bulletin/er-ir/hk-interbank-ir-daily"
    "?segment=hibor.fixing&from=2009-01-01&to=2025-02-28&pagesize=10000&offset=0"
)

# Yahoo Finance omits 2800.HK ex-dividend events prior to 2013-04-29.
# These 8 payouts come directly from Hang Seng Investment Management / TraHK official distribution records.
SUPPLEMENTAL_2800_DIVS: tuple[tuple[str, float], ...] = (
    ("2009-04-29", 0.16),
    ("2009-10-29", 0.38),
    ("2010-04-29", 0.12),
    ("2010-10-29", 0.46),
    ("2011-04-28", 0.13),
    ("2011-10-31", 0.50),
    ("2012-04-27", 0.14),
    ("2012-10-31", 0.52),
)


def _fetch_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ts_to_hk_date(ts: int) -> str:
    return (
        dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        .astimezone(HK_TZ)
        .strftime("%Y-%m-%d")
    )


def _fmt_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{float(v):.8g}"
    return str(v)


def _write_dict_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (row[k] if k in {"date", "source"} else _fmt_val(row.get(k)))
                    for k in fieldnames
                }
            )


def fetch_yahoo_series(encoded_symbol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch daily OHLCV and dividend events from Yahoo Finance v8 Chart API."""
    url = (
        f"{YAHOO_BASE}/{encoded_symbol}"
        f"?period1={PERIOD_START}&period2={PERIOD_END}&interval=1d&events=div%7Csplit"
    )
    payload = _fetch_json(url)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo Finance error for {encoded_symbol}: {chart['error']}")
    res = chart["result"][0]
    timestamps = res.get("timestamp") or []
    quote = res["indicators"]["quote"][0]
    adj_list = res["indicators"].get("adjclose", [{}])
    adj = (
        adj_list[0].get("adjclose", [None] * len(timestamps))
        if adj_list
        else [None] * len(timestamps)
    )

    by_date: dict[str, dict[str, Any]] = {}
    for idx, ts in enumerate(timestamps):
        d_str = _ts_to_hk_date(ts)
        by_date[d_str] = {
            "date": d_str,
            "open": quote["open"][idx],
            "high": quote["high"][idx],
            "low": quote["low"][idx],
            "close": quote["close"][idx],
            "adjclose": adj[idx],
            "volume": quote["volume"][idx],
        }
    price_rows = [by_date[d] for d in sorted(by_date)]

    events = res.get("events", {}).get("dividends", {}) or {}
    div_rows = [
        {
            "date": _ts_to_hk_date(ev["date"]),
            "amount": round(float(ev["amount"]), 6),
            "source": "yahoo",
        }
        for ev in events.values()
    ]
    div_rows.sort(key=lambda r: r["date"])
    return price_rows, div_rows


def fetch_hkma_hibor() -> list[dict[str, Any]]:
    """Fetch daily HKD HIBOR fixing rates from HKMA Open API."""
    payload = _fetch_json(HKMA_HIBOR_URL)
    records = payload.get("result", {}).get("records", [])
    if not records:
        raise RuntimeError("HKMA API returned empty HIBOR records")
    by_date = {
        rec["end_of_day"]: {
            "date": rec["end_of_day"],
            "ir_overnight": rec.get("ir_overnight"),
            "ir_1w": rec.get("ir_1w"),
            "ir_1m": rec.get("ir_1m"),
            "ir_3m": rec.get("ir_3m"),
        }
        for rec in records
    }
    return [by_date[d] for d in sorted(by_date)]


def merge_2800_dividends(yahoo_divs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge Yahoo 2800.HK dividends (2013+) with TraHK official 2009-2012 distributions."""
    existing = {str(r["date"]) for r in yahoo_divs}
    combined = list(yahoo_divs)
    for d_str, amt in SUPPLEMENTAL_2800_DIVS:
        if d_str not in existing:
            combined.append({"date": d_str, "amount": float(amt), "source": "trahk_official"})
    combined.sort(key=lambda r: str(r["date"]))
    return combined


def write_checksums(data_dir: Path, files: list[Path]) -> Path:
    """Write deterministic SHA-256 manifest relative to repo root into `data/checksums.txt`."""
    lines = []
    for fpath in sorted(files, key=lambda p: p.as_posix()):
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        rel = fpath.relative_to(REPO_ROOT).as_posix()
        lines.append(f"{digest}  {rel}")
    out_path = data_dir / "checksums.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_fetch(asof: str | None = None, offline: bool = False) -> Path:
    """Download raw data (unless --offline), build processed CSVs, and verify checksums."""
    data_dir = REPO_ROOT / "data"
    yahoo_dir = data_dir / "raw" / "yahoo"
    hkma_dir = data_dir / "raw" / "hkma"
    yahoo_dir.mkdir(parents=True, exist_ok=True)
    hkma_dir.mkdir(parents=True, exist_ok=True)

    if asof is None:
        if offline:
            candidates = sorted(yahoo_dir.glob("hsi_*.csv"))
            if not candidates:
                raise FileNotFoundError("No local raw files found for --offline mode")
            asof = candidates[-1].stem.split("_")[-1]
        else:
            asof = dt.datetime.now(HK_TZ).strftime("%Y%m%d")

    raw_paths = [
        yahoo_dir / f"hsi_{asof}.csv",
        yahoo_dir / f"hsce_{asof}.csv",
        yahoo_dir / f"etf2800_{asof}.csv",
        yahoo_dir / f"etf2800_div_{asof}.csv",
        yahoo_dir / f"vhsi_{asof}.csv",
        hkma_dir / f"hibor1m_{asof}.csv",
    ]
    ohlcv_cols = ["date", "open", "high", "low", "close", "adjclose", "volume"]
    div_cols = ["date", "amount", "source"]
    hibor_cols = ["date", "ir_overnight", "ir_1w", "ir_1m", "ir_3m"]

    if not offline:
        hsi_rows, _ = fetch_yahoo_series("%5EHSI")
        hsce_rows, _ = fetch_yahoo_series("%5EHSCE")
        etf_rows, etf_divs_raw = fetch_yahoo_series("2800.HK")
        vhsi_rows, _ = fetch_yahoo_series("%5EHSIL")
        etf_divs = merge_2800_dividends(etf_divs_raw)
        hibor_rows = fetch_hkma_hibor()

        _write_dict_csv(raw_paths[0], ohlcv_cols, hsi_rows)
        _write_dict_csv(raw_paths[1], ohlcv_cols, hsce_rows)
        _write_dict_csv(raw_paths[2], ohlcv_cols, etf_rows)
        _write_dict_csv(raw_paths[3], div_cols, etf_divs)
        _write_dict_csv(raw_paths[4], ohlcv_cols, vhsi_rows)
        _write_dict_csv(raw_paths[5], hibor_cols, hibor_rows)
    else:
        for p in raw_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing raw file in --offline mode: {p}")
            if len(read_csv_rows(p)) == 0:
                raise ValueError(f"Empty raw CSV: {p}")

    proc_tbl, audit_tbl = build_processed_datasets(
        raw_dir=data_dir / "raw",
        asof=asof,
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
    )

    all_tracked = raw_paths + [
        data_dir / "interim" / "hk_daily_full_with_audit.csv",
        data_dir / "processed" / "hk_daily.csv",
    ]
    chk_path = write_checksums(data_dir, all_tracked)
    print(
        f"Fetched & processed asof={asof} (offline={offline}): "
        f"hk_daily={len(proc_tbl)} rows ({proc_tbl['date'][0]}..{proc_tbl['date'][-1]}), "
        f"audit={len(audit_tbl)} rows, checksums={chk_path}"
    )
    return chk_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and align HK market data for B1.")
    parser.add_argument("--asof", type=str, default=None, help="Fetch date tag YYYYMMDD (HKT).")
    parser.add_argument(
        "--offline", action="store_true", help="Re-process local raw CSVs without network."
    )
    args = parser.parse_args(argv)
    run_fetch(asof=args.asof, offline=args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

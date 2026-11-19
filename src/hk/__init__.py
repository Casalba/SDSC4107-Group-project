"""Hong Kong data pipeline deliverables (B1): ^HSI, ^HSCE, 2800.HK, ^VHSI/^HSIL, HIBOR."""

from src.hk.data import (
    AUDIT_COLUMNS,
    PROCESSED_COLUMNS,
    HKTable,
    HKWindow,
    build_processed_datasets,
    compute_backward_adjusted_prices,
    compute_trailing_dividend_yield,
    extract_monthly_windows,
    load_hk_audit,
    load_hk_daily,
    read_csv_table,
    round_hsi_strike,
    synthetic_futures_path,
    window_features,
)

__all__ = [
    "AUDIT_COLUMNS",
    "PROCESSED_COLUMNS",
    "HKTable",
    "HKWindow",
    "build_processed_datasets",
    "compute_backward_adjusted_prices",
    "compute_trailing_dividend_yield",
    "extract_monthly_windows",
    "load_hk_audit",
    "load_hk_daily",
    "read_csv_table",
    "round_hsi_strike",
    "synthetic_futures_path",
    "window_features",
]

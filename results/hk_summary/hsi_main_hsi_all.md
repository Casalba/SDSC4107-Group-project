# HK Cross-Validation Results Summary (HSI | all)

- **Index**: `hsi` | **Split**: `all` (180 monthly windows)

## 1. Main Table: Capital Requirement C* & Trading Metrics

| Cost Leg | Strategy | C*95 | C*99 | Mean Loss | Turnover | Cost / Notional |
|---|---|---:|---:|---:|---:|---:|
| fut1bp | Bench (Delta) | 5.1215 | 6.4768 | 2.5157 | 2.1423 | 0.00011 |
| fut1bp | M1 (Zero-shot) | 6.7222 | 7.7330 | 2.4567 | 2.7685 | 0.00014 |
| fut1bp | M2 (Re-fit, α=0.5) | 4.6550 ± 0.1738 | 5.5378 ± 0.3904 | 2.3643 ± 0.0243 | 2.1367 | 0.00011 |
| fut1bp | M2 (Re-fit, α=0.99) | 5.1240 ± 0.0489 | 6.6597 ± 0.1076 | 2.5729 ± 0.0043 | 1.8578 | 0.00009 |
| fut10bp | Bench (Delta) | 5.2166 | 6.5934 | 2.6120 | 2.1423 | 0.00107 |
| fut10bp | M1 (Zero-shot) | 6.9162 | 7.8645 | 2.5832 | 2.7685 | 0.00141 |
| etf30bp | Bench (Delta) | 5.2562 | 6.7040 | 2.7139 | 2.1487 | 0.00323 |
| etf30bp | M1 (Zero-shot) | 7.0889 | 7.8258 | 2.7712 | 2.7643 | 0.00421 |
| etf30bp | M2 (Re-fit, α=0.5) | 5.7349 ± 0.6984 | 7.7727 ± 1.2383 | 2.4832 ± 0.0153 | 1.4076 | 0.00211 |
| etf30bp | M2 (Re-fit, α=0.99) | 5.4231 ± 0.0436 | 6.8969 ± 0.0893 | 2.6889 ± 0.0048 | 1.6491 | 0.00247 |

## 2. Hypothesis Tests (P1–P4)

- **P1_futures_1bp_parity**: `{"point": -0.178682909078554, "ci_low": -0.760884924082115, "ci_high": 0.4624751179095025, "ci_excludes_zero": false, "p1_within_5pct": true, "p2_strictly_better": false}`
- **P2_etf_30bp_advantage**: `{"point": 1.2650239324949126, "ci_low": 0.15820809982285067, "ci_high": 2.495261380119819, "ci_excludes_zero": true, "p1_within_5pct": false, "p2_strictly_better": false}`
- **P3_risk_aversion_crossover_etf**: `{"c95_a99_minus_a50": -0.3117318568046459, "c99_a99_minus_a50": -0.8758142566502771}`
- **P4_refit_vs_zeroshot_etf**: `{"point": -0.5677128086265375, "ci_low": -1.8379799232966925, "ci_high": 0.6012863576723522, "ci_excludes_zero": false, "p1_within_5pct": false, "p2_strictly_better": false}`

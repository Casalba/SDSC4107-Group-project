# HK Cross-Validation Results Summary (HSCE | all)

- **Index**: `hsce` | **Split**: `all` (180 monthly windows)

## 1. Main Table: Capital Requirement C* & Trading Metrics

| Cost Leg | Strategy | C*95 | C*99 | Mean Loss | Turnover | Cost / Notional |
|---|---|---:|---:|---:|---:|---:|
| fut1bp | Bench (Delta) | 5.7465 | 6.7687 | 2.9383 | 2.2771 | 0.00011 |
| fut1bp | M1 (Zero-shot) | 7.1144 | 8.2234 | 2.8910 | 2.8884 | 0.00015 |
| fut10bp | Bench (Delta) | 5.8594 | 6.9093 | 3.0407 | 2.2771 | 0.00114 |
| fut10bp | M1 (Zero-shot) | 7.3125 | 8.3722 | 3.0232 | 2.8884 | 0.00147 |
| etf30bp | Bench (Delta) | 9.2113 | 13.4802 | 3.0935 | 2.2947 | 0.00345 |
| etf30bp | M1 (Zero-shot) | 10.0002 | 12.7521 | 3.1881 | 2.8816 | 0.00439 |

## 2. Hypothesis Tests (P1–P4)

- **P1_futures_1bp_parity**: `{"point": 1.367869407367845, "ci_low": 0.47502880399249303, "ci_high": 2.263914629805146, "ci_excludes_zero": true, "p1_within_5pct": false, "p2_strictly_better": false}`
- **P2_etf_30bp_advantage**: `{"point": 0.7889054785224499, "ci_low": -0.15635853431624802, "ci_high": 1.6148784407868932, "ci_excludes_zero": false, "p1_within_5pct": false, "p2_strictly_better": false}`
- **P3_risk_aversion_crossover_etf**: `"pending"`
- **P4_refit_vs_zeroshot_etf**: `"pending"`
